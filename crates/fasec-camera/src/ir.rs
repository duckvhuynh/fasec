//! IR capture thread: WinRT infrared source -> Gray8 `IrFrame` -> `LatestIrPair`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use fasec_core::{CameraHealth, IrFrame};

use crate::capture::{STOP_TIMEOUT, ms_since, with_error};
use crate::{CameraError, HealthConfig, HealthMonitor};

/// How long `start` waits for the first frame.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(5);
/// Poll interval while the source has no new frame (IR runs at ~15 fps).
const POLL: Duration = Duration::from_millis(5);
/// Health luma samples every 61st pixel, as `capture::LUMA_STRIDE` does for RGB.
const LUMA_STRIDE: usize = 61;

/// The newest illuminated frame, paired with the unlit (ambient) frame closest before it.
#[derive(Debug, Default)]
pub struct LatestIrPair {
    slot: Mutex<Pair>,
    cv: Condvar,
}

#[derive(Debug, Default)]
struct Pair {
    /// Newest unconsumed lit frame and the ambient frame that was latest when it arrived.
    lit: Option<(IrFrame, Option<IrFrame>)>,
    /// Latest ambient frame; kept after a take so the next lit frame can pair with it.
    ambient: Option<IrFrame>,
}

impl LatestIrPair {
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `f`. A lit frame replaces an unconsumed older lit frame; an unlit one only
    /// becomes the ambient partner of the next lit frame.
    pub fn put(&self, f: IrFrame) {
        let mut p = self.lock();
        if f.illuminated {
            p.lit = Some((f, p.ambient.clone()));
            drop(p);
            self.cv.notify_all();
        } else {
            p.ambient = Some(f);
        }
    }

    /// Waits up to `timeout` for a lit frame with `seq > after_seq` and takes it out, with its
    /// ambient partner (`None` if no unlit frame came before it).
    pub fn take_newer(
        &self,
        after_seq: u64,
        timeout: Duration,
    ) -> Option<(IrFrame, Option<IrFrame>)> {
        let is_newer = |p: &Pair| p.lit.as_ref().is_some_and(|(f, _)| f.seq > after_seq);
        let (mut p, _) = self
            .cv
            .wait_timeout_while(self.lock(), timeout, |p| !is_newer(p))
            .unwrap_or_else(PoisonError::into_inner);
        if is_newer(&p) { p.lit.take() } else { None }
    }

    // Writers only swap fields, so a poisoned lock holds consistent data.
    fn lock(&self) -> MutexGuard<'_, Pair> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One Gray8 frame as read from the source, before sequencing.
struct RawIr {
    data: Vec<u8>,
    width: u32,
    height: u32,
    illuminated: bool,
}

/// Where the IR thread gets frames from. `Ok(None)`: no new frame yet, poll again.
trait IrFrameSource {
    fn next_ir(&mut self) -> Result<Option<RawIr>, CameraError>;
}

/// A running IR capture thread. Dropping it (or `stop`) stops the stream and waits at most
/// `STOP_TIMEOUT` for the thread; a thread stuck in the driver is detached, not joined.
pub struct IrCapture {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<Status>>,
    t0: Instant,
    sharing_mode: &'static str,
    thread: Option<JoinHandle<()>>,
}

struct Status {
    error: Option<CameraError>,
    health: HealthMonitor,
}

impl IrCapture {
    /// Opens IR source `source_id` (an `IrSourceInfo::id`) on its own thread, `SharedReadOnly`
    /// first and `ExclusiveControl` only if that fails, and streams into `slot`. Returns once
    /// the first frame arrived. Blocks up to ~5 s: never call it on a UI thread.
    pub fn start(source_id: &str, slot: Arc<LatestIrPair>) -> Result<IrCapture, CameraError> {
        #[cfg(windows)]
        {
            let id = source_id.to_owned();
            Self::spawn(move || win::open(&id), slot)
        }
        #[cfg(not(windows))]
        {
            let _ = (source_id, slot);
            Err(CameraError::Backend("IR capture needs Windows".into()))
        }
    }

    /// `open` runs on the capture thread and names the sharing mode it got.
    fn spawn<S: IrFrameSource>(
        open: impl FnOnce() -> Result<(S, &'static str), CameraError> + Send + 'static,
        slot: Arc<LatestIrPair>,
    ) -> Result<IrCapture, CameraError> {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(Status {
            error: None,
            health: HealthMonitor::new(HealthConfig::default()),
        }));
        let t0 = Instant::now();
        let (tx, rx) = mpsc::sync_channel(1);
        let (t_stop, t_status) = (Arc::clone(&stop), Arc::clone(&status));
        let thread = std::thread::Builder::new()
            .name("fasec-ir".into())
            .spawn(move || match open() {
                Ok((mut src, mode)) => run(&mut src, mode, tx, &slot, &t_stop, &t_status, t0),
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            })
            .map_err(|e| CameraError::Backend(format!("spawn IR thread: {e}")))?;
        let started = rx.recv_timeout(FIRST_FRAME_TIMEOUT);
        // Built before matching so every failure path stops the thread through Drop.
        let mut capture = IrCapture {
            stop,
            status,
            t0,
            sharing_mode: "",
            thread: Some(thread),
        };
        match started {
            Ok(Ok(mode)) => {
                capture.sharing_mode = mode;
                Ok(capture)
            }
            Ok(Err(e)) => Err(e),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(CameraError::Backend(format!(
                "IR camera delivered no frame within {} s",
                FIRST_FRAME_TIMEOUT.as_secs()
            ))),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                Err(CameraError::Backend("IR thread exited during start".into()))
            }
        }
    }

    pub fn stop(self) {}

    /// `SharedReadOnly` or `ExclusiveControl`.
    pub fn sharing_mode(&self) -> &'static str {
        self.sharing_mode
    }

    /// `Lost` after 1.5 s without an illuminated frame, `Covered` after > 2 s of black lit
    /// frames, or the current error mapped like `Capture::health`.
    pub fn health(&self) -> CameraHealth {
        let now_ms = ms_since(self.t0);
        let s = self.status.lock().unwrap_or_else(PoisonError::into_inner);
        with_error(s.health.evaluate(now_ms), s.error.clone())
    }
}

impl Drop for IrCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let Some(t) = self.thread.take() else {
            return;
        };
        let deadline = Instant::now() + STOP_TIMEOUT;
        while !t.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if !t.is_finished() {
            tracing::error!("IR capture thread did not stop within {STOP_TIMEOUT:?}; detaching it");
        } else if t.join().is_err() {
            tracing::error!("IR capture thread panicked");
        }
    }
}

/// Frame loop. Reports the first frame (or the error before it) through `started`.
fn run(
    src: &mut impl IrFrameSource,
    mode: &'static str,
    started: mpsc::SyncSender<Result<&'static str, CameraError>>,
    slot: &LatestIrPair,
    stop: &AtomicBool,
    status: &Mutex<Status>,
    t0: Instant,
) {
    let mut started = Some(started);
    let (mut seq, mut errors) = (0u64, 0u32);
    while !stop.load(Ordering::Relaxed) {
        let frame = src.next_ir().and_then(|raw| {
            raw.map(|r| {
                IrFrame::new(
                    seq + 1,
                    ms_since(t0),
                    r.width,
                    r.height,
                    r.illuminated,
                    r.data,
                )
                .map_err(|e| CameraError::Backend(e.to_string()))
            })
            .transpose()
        });
        match frame {
            Ok(Some(f)) => {
                seq = f.seq;
                errors = 0;
                let mut s = status.lock().unwrap_or_else(PoisonError::into_inner);
                s.error = None;
                // Only lit frames count: an IR stream without them is useless for liveness.
                if f.illuminated {
                    s.health.on_frame(f.ts_ms, sampled_mean(&f.data));
                }
                drop(s);
                slot.put(f);
                if let Some(tx) = started.take() {
                    let _ = tx.send(Ok(mode));
                }
            }
            Ok(None) => std::thread::sleep(POLL),
            Err(e) => {
                if let Some(tx) = started.take() {
                    let _ = tx.send(Err(e));
                    return;
                }
                if errors == 0 {
                    tracing::warn!(error = %e, "IR frame failed");
                }
                status.lock().unwrap_or_else(PoisonError::into_inner).error = Some(e);
                errors = errors.saturating_add(1);
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn sampled_mean(gray: &[u8]) -> f32 {
    let (mut sum, mut n) = (0u64, 0u64);
    for &p in gray.iter().step_by(LUMA_STRIDE) {
        sum += u64::from(p);
        n += 1;
    }
    if n == 0 { 0.0 } else { sum as f32 / n as f32 }
}

/// Maps a WinRT error to the camera error callers branch on (same HRESULTs as `error.rs`).
#[cfg(windows)]
pub(crate) fn win_err(what: &str, e: &windows::core::Error) -> CameraError {
    match e.code().0 as u32 {
        0x8007_0005 => CameraError::PermissionDenied,
        0xC00D_3704 | 0xC00D_3EA3 => CameraError::Busy,
        code => CameraError::Backend(format!("{what}: {} (0x{code:08X})", e.message())),
    }
}

#[cfg(windows)]
mod win {
    use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Capture::Frames::{
        MediaFrameReader, MediaFrameReaderStartStatus, MediaFrameSourceGroup,
    };
    use windows::Media::Capture::{
        MediaCapture, MediaCaptureInitializationSettings, MediaCaptureMemoryPreference,
        MediaCaptureSharingMode, StreamingCaptureMode,
    };
    use windows::Storage::Streams::{Buffer, DataReader};

    use super::{IrFrameSource, RawIr, win_err};
    use crate::{CameraError, normalize_id};

    /// Open WinRT reader; stopped and closed on drop.
    pub(super) struct WinIr {
        capture: MediaCapture,
        reader: MediaFrameReader,
    }

    /// Finds `source_id` in any source group and opens it, `SharedReadOnly` first.
    pub(super) fn open(source_id: &str) -> Result<(WinIr, &'static str), CameraError> {
        let want = normalize_id(source_id);
        let err = |what| move |e: windows::core::Error| win_err(what, &e);
        let groups = MediaFrameSourceGroup::FindAllAsync()
            .and_then(|op| op.join())
            .map_err(err("enumerate frame sources"))?;
        let mut found = None;
        for g in groups {
            for s in g.SourceInfos().map_err(err("source infos"))? {
                let id = s.Id().map_err(err("source id"))?;
                if normalize_id(&id.to_string()) == want {
                    found = Some((g.clone(), id));
                }
            }
            if found.is_some() {
                break;
            }
        }
        let (group, id) = found.ok_or_else(|| CameraError::NotFound {
            id: source_id.to_owned(),
        })?;
        match open_with(&group, &id, MediaCaptureSharingMode::SharedReadOnly) {
            Ok(src) => Ok((src, "SharedReadOnly")),
            Err(e) => {
                tracing::warn!(error = %e, "IR SharedReadOnly failed; trying ExclusiveControl");
                open_with(&group, &id, MediaCaptureSharingMode::ExclusiveControl)
                    .map(|src| (src, "ExclusiveControl"))
            }
        }
    }

    fn open_with(
        group: &MediaFrameSourceGroup,
        id: &windows::core::HSTRING,
        mode: MediaCaptureSharingMode,
    ) -> Result<WinIr, CameraError> {
        let err = |what| move |e: windows::core::Error| win_err(what, &e);
        let settings = MediaCaptureInitializationSettings::new().map_err(err("settings"))?;
        (|| {
            settings.SetSourceGroup(group)?;
            settings.SetSharingMode(mode)?;
            settings.SetMemoryPreference(MediaCaptureMemoryPreference::Cpu)?;
            settings.SetStreamingCaptureMode(StreamingCaptureMode::Video)
        })()
        .map_err(err("settings"))?;
        let capture = MediaCapture::new().map_err(err("MediaCapture"))?;
        let reader = (|| {
            capture.InitializeWithSettingsAsync(&settings)?.join()?;
            let source = capture.FrameSources()?.Lookup(id)?;
            capture.CreateFrameReaderAsync(&source)?.join()
        })()
        .map_err(err("open IR source"));
        let reader = match reader {
            Ok(r) => r,
            Err(e) => {
                let _ = capture.Close(); // releases the camera; nothing more to do on failure
                return Err(e);
            }
        };
        // From here Drop stops the reader and closes the capture.
        let src = WinIr { capture, reader };
        let status = src
            .reader
            .StartAsync()
            .and_then(|op| op.join())
            .map_err(err("start IR reader"))?;
        if status != MediaFrameReaderStartStatus::Success {
            return Err(CameraError::Backend(format!(
                "IR reader start status {}",
                status.0
            )));
        }
        Ok(src)
    }

    impl IrFrameSource for WinIr {
        fn next_ir(&mut self) -> Result<Option<RawIr>, CameraError> {
            // Fails (null) when no new frame has arrived since the last call.
            let Ok(frame) = self.reader.TryAcquireLatestFrame() else {
                return Ok(None);
            };
            let err = |what| move |e: windows::core::Error| win_err(what, &e);
            let video = frame.VideoMediaFrame().map_err(err("video frame"))?;
            // Unknown lighting counts as unlit: it can then never pass as a lit frame.
            let illuminated = video
                .InfraredMediaFrame()
                .and_then(|i| i.IsIlluminated())
                .unwrap_or(false);
            let bmp = video.SoftwareBitmap().map_err(err("IR bitmap"))?;
            let raw = gray8(&bmp, illuminated).map_err(err("IR pixels"));
            let _ = bmp.Close();
            let _ = frame.Close();
            raw.map(Some)
        }
    }

    fn gray8(bmp: &SoftwareBitmap, illuminated: bool) -> windows::core::Result<RawIr> {
        let bmp = if bmp.BitmapPixelFormat()? == BitmapPixelFormat::Gray8 {
            bmp.clone()
        } else {
            SoftwareBitmap::Convert(bmp, BitmapPixelFormat::Gray8)?
        };
        let (w, h) = (bmp.PixelWidth()?, bmp.PixelHeight()?);
        let (w, h) = (w.unsigned_abs(), h.unsigned_abs());
        let buf = Buffer::Create(w * h)?;
        bmp.CopyToBuffer(&buf)?;
        let mut data = vec![0u8; buf.Length()? as usize];
        DataReader::FromBuffer(&buf)?.ReadBytes(&mut data)?;
        // A length mismatch (row padding) is rejected by `IrFrame::new`.
        Ok(RawIr {
            data,
            width: w,
            height: h,
            illuminated,
        })
    }

    impl Drop for WinIr {
        fn drop(&mut self) {
            if let Err(e) = self.reader.StopAsync().and_then(|op| op.join()) {
                tracing::warn!(error = %e, "IR reader stop failed");
            }
            let _ = self.capture.Close(); // releases the camera for Windows Hello
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ir(seq: u64, lit: bool) -> IrFrame {
        IrFrame::new(seq, seq, 1, 1, lit, vec![if lit { 90 } else { 0 }]).expect("1x1 frame")
    }

    fn seqs(p: Option<(IrFrame, Option<IrFrame>)>) -> Option<(u64, Option<u64>)> {
        p.map(|(l, a)| (l.seq, a.map(|a| a.seq)))
    }

    #[test]
    fn newest_lit_comes_with_the_ambient_frame_just_before_it() {
        let slot = LatestIrPair::new();
        for (seq, lit) in [(1, false), (2, true), (3, false), (4, true), (5, false)] {
            slot.put(ir(seq, lit));
        }
        // u5 is newer than l4 but is not the ambient frame before it.
        assert_eq!(seqs(slot.take_newer(0, Duration::ZERO)), Some((4, Some(3))));
    }

    #[test]
    fn a_lit_frame_is_never_handed_out_twice() {
        let slot = LatestIrPair::new();
        slot.put(ir(1, false));
        slot.put(ir(2, true));
        assert_eq!(seqs(slot.take_newer(0, Duration::ZERO)), Some((2, Some(1))));
        assert_eq!(seqs(slot.take_newer(0, Duration::from_millis(20))), None);
        slot.put(ir(3, false)); // ambient alone is not a pair
        assert_eq!(seqs(slot.take_newer(0, Duration::from_millis(20))), None);
        slot.put(ir(4, true));
        assert_eq!(seqs(slot.take_newer(2, Duration::ZERO)), Some((4, Some(3))));
    }

    #[test]
    fn ambient_survives_a_take_and_lit_without_ambient_pairs_with_none() {
        let slot = LatestIrPair::new();
        slot.put(ir(1, true));
        assert_eq!(seqs(slot.take_newer(0, Duration::ZERO)), Some((1, None)));
        slot.put(ir(2, false));
        slot.put(ir(3, true));
        assert_eq!(seqs(slot.take_newer(1, Duration::ZERO)), Some((3, Some(2))));
        // Two lit frames in a row: the second still pairs with the last unlit one.
        slot.put(ir(4, true));
        assert_eq!(seqs(slot.take_newer(3, Duration::ZERO)), Some((4, Some(2))));
    }

    #[test]
    fn take_newer_keeps_a_lit_frame_that_is_not_newer() {
        let slot = LatestIrPair::new();
        slot.put(ir(3, true));
        assert_eq!(seqs(slot.take_newer(3, Duration::ZERO)), None);
        assert_eq!(seqs(slot.take_newer(2, Duration::ZERO)), Some((3, None)));
    }

    #[test]
    fn a_waiting_consumer_wakes_on_a_lit_frame() {
        let slot = Arc::new(LatestIrPair::new());
        let producer = {
            let slot = Arc::clone(&slot);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                slot.put(ir(1, false));
                std::thread::sleep(Duration::from_millis(20));
                slot.put(ir(2, true));
            })
        };
        assert_eq!(
            seqs(slot.take_newer(0, Duration::from_secs(2))),
            Some((2, Some(1)))
        );
        producer.join().expect("producer thread");
    }

    /// Alternates lit and unlit 2x1 frames every 5 ms, like the ASUS IR camera.
    struct Alternating(bool);
    impl IrFrameSource for Alternating {
        fn next_ir(&mut self) -> Result<Option<RawIr>, CameraError> {
            std::thread::sleep(Duration::from_millis(5));
            self.0 = !self.0;
            Ok(Some(RawIr {
                data: vec![if self.0 { 90 } else { 0 }; 2],
                width: 2,
                height: 1,
                illuminated: self.0,
            }))
        }
    }

    /// Delivers one lit frame, then blocks forever, like a stalled driver.
    struct StallsAfterFirst(bool);
    impl IrFrameSource for StallsAfterFirst {
        fn next_ir(&mut self) -> Result<Option<RawIr>, CameraError> {
            if std::mem::replace(&mut self.0, true) {
                loop {
                    std::thread::park();
                }
            }
            Ok(Some(RawIr {
                data: vec![90],
                width: 1,
                height: 1,
                illuminated: true,
            }))
        }
    }

    /// Only unlit frames: the emitter never fires (or Hello took it over).
    struct NeverLit;
    impl IrFrameSource for NeverLit {
        fn next_ir(&mut self) -> Result<Option<RawIr>, CameraError> {
            std::thread::sleep(Duration::from_millis(5));
            Ok(Some(RawIr {
                data: vec![0],
                width: 1,
                height: 1,
                illuminated: false,
            }))
        }
    }

    #[test]
    fn capture_pairs_lit_frames_with_the_previous_ambient_and_stops_fast() {
        let slot = Arc::new(LatestIrPair::new());
        let cap = IrCapture::spawn(|| Ok((Alternating(false), "Fake")), Arc::clone(&slot))
            .expect("start");
        assert_eq!(cap.sharing_mode(), "Fake");
        let mut last = 0;
        for _ in 0..3 {
            let (lit, ambient) = slot
                .take_newer(last, Duration::from_secs(1))
                .expect("lit frame");
            assert!(lit.illuminated && lit.seq > last);
            if let Some(a) = ambient {
                assert!(!a.illuminated && a.seq < lit.seq, "{a:?} vs {lit:?}");
            }
            last = lit.seq;
        }
        assert_eq!(cap.health(), CameraHealth::Ok);
        let t0 = Instant::now();
        cap.stop();
        assert!(
            t0.elapsed() < Duration::from_millis(500),
            "{:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn stop_is_bounded_when_the_source_stalls() {
        let cap = IrCapture::spawn(
            || Ok((StallsAfterFirst(false), "Fake")),
            Arc::new(LatestIrPair::new()),
        )
        .expect("start");
        std::thread::sleep(Duration::from_millis(50)); // thread is now parked in next_ir
        let t0 = Instant::now();
        cap.stop();
        let took = t0.elapsed();
        assert!(
            took >= STOP_TIMEOUT && took < Duration::from_millis(2500),
            "{took:?}"
        );
    }

    #[test]
    fn health_is_lost_when_no_lit_frame_arrives() {
        let cap = IrCapture::spawn(|| Ok((NeverLit, "Fake")), Arc::new(LatestIrPair::new()))
            .expect("start");
        assert_eq!(cap.health(), CameraHealth::Ok, "grace period after start");
        std::thread::sleep(Duration::from_millis(1600));
        assert_eq!(cap.health(), CameraHealth::Lost);
    }

    #[test]
    fn start_reports_the_open_or_first_read_error() {
        let open_err = IrCapture::spawn(
            || Err::<(NeverLit, _), _>(CameraError::NotFound { id: "ir".into() }),
            Arc::new(LatestIrPair::new()),
        )
        .err();
        assert_eq!(open_err, Some(CameraError::NotFound { id: "ir".into() }));
        struct Fails;
        impl IrFrameSource for Fails {
            fn next_ir(&mut self) -> Result<Option<RawIr>, CameraError> {
                Err(CameraError::Busy)
            }
        }
        let read_err =
            IrCapture::spawn(|| Ok((Fails, "Fake")), Arc::new(LatestIrPair::new())).err();
        assert_eq!(read_err, Some(CameraError::Busy));
    }

    /// Live: the IR source paired with the first RGB camera alternates lit and unlit frames.
    #[test]
    #[ignore = "needs IR camera"]
    fn ir_capture_delivers_alternating_frames() {
        for s in crate::ir_sources().expect("enumerate IR") {
            println!("ir source {:?} device {}", s.name, s.device_id);
        }
        let rgb = crate::list().expect("list cameras");
        let rgb = rgb.first().expect("no RGB camera");
        let ir = crate::paired_ir_for(&rgb.id)
            .expect("enumerate IR")
            .expect("no IR source paired with the RGB camera");
        println!(
            "rgb {:?} -> ir {:?} (group {})",
            rgb.name, ir.name, ir.group_id
        );
        let slot = Arc::new(LatestIrPair::new());
        let cap = IrCapture::start(&ir.id, Arc::clone(&slot)).expect("start IR");
        println!("sharing mode: {}", cap.sharing_mode());
        let t0 = Instant::now();
        let (mut last, mut lit, mut paired, mut blank_ambient) = (0, 0u64, 0u64, 0u64);
        while t0.elapsed() < Duration::from_secs(3) {
            let Some((l, a)) = slot.take_newer(last, Duration::from_millis(200)) else {
                continue;
            };
            assert!(l.illuminated && l.seq > last);
            assert_eq!(l.data.len(), (l.width * l.height) as usize);
            if let Some(a) = a {
                assert!(!a.illuminated && a.seq < l.seq);
                paired += 1;
                blank_ambient += u64::from(a.is_blank());
                // Aggregates only: IR pixels never reach logs.
                let max = a.data.iter().copied().max().unwrap_or(0);
                if paired <= 3 {
                    println!(
                        "ambient {}: mean {:.2} max {max}; lit {} mean {:.2}",
                        a.seq,
                        sampled_mean(&a.data),
                        l.seq,
                        sampled_mean(&l.data)
                    );
                }
            }
            if lit == 0 {
                println!("first lit frame: {:?}", l);
            }
            lit += 1;
            last = l.seq;
        }
        let health = cap.health();
        cap.stop();
        println!(
            "{last} frames in 3 s, {lit} lit, {paired} with ambient ({blank_ambient} ambient all-zero), health {health:?}"
        );
        assert!(last >= 20, "only {last} frames in 3 s");
        let share = lit as f64 / last as f64;
        assert!((0.3..=0.7).contains(&share), "lit share {share:.2}");
        // Covered is legitimate if the lens is covered.
        assert!(
            matches!(health, CameraHealth::Ok | CameraHealth::Covered),
            "{health:?}"
        );
    }
}
