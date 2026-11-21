//! Capture thread: camera -> RGB8 -> transform -> `LatestFrame`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use fasec_config::CameraConfig;
use fasec_core::{CameraHealth, Frame};
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, FrameFormat, RequestedFormat, RequestedFormatType};

use crate::device::{self, ComGuard};
use crate::permission::{self, Access};
use crate::{CameraError, HealthConfig, HealthMonitor, LatestFrame, transform};

/// How long `stop`/`Drop` waits for the capture thread before detaching it.
const STOP_TIMEOUT: Duration = Duration::from_secs(2);
/// How long `start` waits for the first frame. Busy surfaces only on the first read.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(5);
/// Covered detection averages every 61st pixel: a full-frame mean on every 720p frame costs
/// about one frame time in debug builds (measured: 18 -> 9 frames in 2 s). Prime, so it does
/// not divide common widths and the sampled columns shift from row to row.
const LUMA_STRIDE: usize = 61;

/// A running capture thread. Dropping it (or `stop`) stops the stream and waits at most
/// `STOP_TIMEOUT` for the thread; a thread stuck in a blocking read is detached, not joined.
pub struct Capture {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<Status>>,
    /// Origin of the capture clock (`Frame::ts_ms` and the health monitor's `now_ms`).
    t0: Instant,
    thread: Option<JoinHandle<()>>,
}

/// What the capture thread last saw, read by `last_frame_at` / `last_error` / `health`.
struct Status {
    frame_at: Option<Instant>,
    error: Option<CameraError>,
    health: HealthMonitor,
}

/// Where the capture thread gets raw RGB8 frames (`data, width, height`) from.
trait FrameSource {
    fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError>;
}

impl Capture {
    /// Opens camera `info_id` (a `CameraInfo::id`) on its own thread and starts streaming into
    /// `slot`. Returns once the first frame has arrived, or with the open/first-read error
    /// (e.g. `Busy` when another process holds the camera; a vague error while Windows privacy
    /// settings deny camera access becomes `PermissionDenied`).
    ///
    /// Blocks until then, up to ~7 s when the camera never delivers: never call it on a UI thread.
    pub fn start(
        info_id: &str,
        cfg: &CameraConfig,
        slot: Arc<LatestFrame>,
    ) -> Result<Capture, CameraError> {
        let (id, open_cfg) = (info_id.to_owned(), cfg.clone());
        Self::spawn(move || open_stream(&id, &open_cfg), cfg, slot)
            .map_err(|e| explain(e, permission::desktop_camera_access()))
    }

    /// `open` runs on the capture thread (nokhwa's `Camera` is not `Send`).
    fn spawn<S: FrameSource>(
        open: impl FnOnce() -> Result<S, CameraError> + Send + 'static,
        cfg: &CameraConfig,
        slot: Arc<LatestFrame>,
    ) -> Result<Capture, CameraError> {
        let stop = Arc::new(AtomicBool::new(false));
        let status = Arc::new(Mutex::new(Status {
            frame_at: None,
            error: None,
            health: HealthMonitor::new(HealthConfig::default()),
        }));
        let t0 = Instant::now();
        let (tx, rx) = mpsc::sync_channel(1);
        let cfg = cfg.clone();
        let (t_stop, t_status) = (Arc::clone(&stop), Arc::clone(&status));
        let thread = std::thread::Builder::new()
            .name("fasec-camera".into())
            .spawn(move || {
                let _com = ComGuard::init();
                let first = open().and_then(|mut src| {
                    let frame = next_frame(&mut src, &cfg, 1, t0)?;
                    Ok((src, frame))
                });
                let mut src = match first {
                    Ok((src, frame)) => {
                        let _ = tx.send(Ok(()));
                        deliver(frame, &slot, &t_status);
                        src
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                run(&mut src, &cfg, &slot, &t_stop, &t_status, t0);
            })
            .map_err(|e| CameraError::Backend(format!("spawn capture thread: {e}")))?;
        let started = rx.recv_timeout(FIRST_FRAME_TIMEOUT);
        // Built before matching so every failure path stops the thread through Drop.
        let capture = Capture {
            stop,
            status,
            t0,
            thread: Some(thread),
        };
        match started {
            Ok(Ok(())) => Ok(capture),
            Ok(Err(e)) => Err(e),
            Err(mpsc::RecvTimeoutError::Timeout) => Err(CameraError::Backend(format!(
                "camera delivered no frame within {} s",
                FIRST_FRAME_TIMEOUT.as_secs()
            ))),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(CameraError::Backend(
                "capture thread exited during start".into(),
            )),
        }
    }

    pub fn stop(self) {}

    /// When the last frame was put into the slot; `None` before the first frame.
    pub fn last_frame_at(&self) -> Option<Instant> {
        self.status().frame_at
    }

    /// Latest error of the current error streak; cleared by the next good frame. Lets health
    /// tell a stalled camera (no frames, `None`) from an erroring one (`Some`).
    pub fn last_error(&self) -> Option<CameraError> {
        self.status().error.clone()
    }

    /// Health right now (FR-CAM-005/009): `Lost` after 1.5 s without a frame, `Covered` after
    /// > 2 s of black frames, or the mapped `last_error` (see `with_error`).
    pub fn health(&self) -> CameraHealth {
        let now_ms = ms_since(self.t0);
        let s = self.status();
        with_error(s.health.evaluate(now_ms), s.error.clone())
    }

    fn status(&self) -> MutexGuard<'_, Status> {
        self.status.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Drop for Capture {
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
            // Dropping the handle detaches the thread; it exits on its own if the read returns.
            tracing::error!(
                "camera capture thread did not stop within {STOP_TIMEOUT:?}; detaching it"
            );
        } else if t.join().is_err() {
            tracing::error!("camera capture thread panicked");
        }
    }
}

/// nokhwa camera, stopped and dropped under the MF lock (see `device::mf_lock`).
struct NokhwaSource(Option<Camera>);

impl FrameSource for NokhwaSource {
    fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError> {
        let Some(cam) = self.0.as_mut() else {
            return Err(CameraError::Backend("camera already closed".into()));
        };
        let buf = cam.frame()?;
        let (w, h) = (buf.resolution().width(), buf.resolution().height());
        Ok((buf.decode_image::<RgbFormat>()?.into_raw(), w, h))
    }
}

impl Drop for NokhwaSource {
    fn drop(&mut self) {
        let _mf = device::mf_lock();
        if let Some(mut cam) = self.0.take() {
            if let Err(e) = cam.stop_stream() {
                tracing::warn!(error = %e, "stop_stream failed");
            }
            drop(cam);
        }
    }
}

fn open_stream(id: &str, cfg: &CameraConfig) -> Result<NokhwaSource, CameraError> {
    let _mf = device::mf_lock();
    let mut cam = device::open(id)?;
    let available = device::native_formats(&mut cam)?;
    let fmt = pick_format(&available, cfg.resolution[0], cfg.resolution[1], cfg.fps)
        .ok_or_else(|| CameraError::Backend("camera offers no RGB-decodable format".into()))?;
    tracing::info!(camera = %cam.info().human_name(), format = %fmt, "opening camera");
    cam.set_camera_requset(RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::Exact(fmt),
    ))?;
    cam.open_stream()?;
    Ok(NokhwaSource(Some(cam)))
}

/// Closest resolution (squared w/h distance), then closest fps; on ties prefer uncompressed
/// formats (cheaper to convert than MJPEG).
fn pick_format(formats: &[CameraFormat], w: u32, h: u32, fps: u32) -> Option<CameraFormat> {
    let dist = |a: u32, b: u32| u64::from(a.abs_diff(b));
    formats.iter().copied().min_by_key(|f| {
        let res = dist(f.width(), w).pow(2) + dist(f.height(), h).pow(2);
        (
            res,
            f.frame_rate().abs_diff(fps),
            f.format() == FrameFormat::MJPEG,
        )
    })
}

/// A vague start error while Windows privacy settings deny camera access is a permission error.
fn explain(e: CameraError, access: Access) -> CameraError {
    match (e, access) {
        (CameraError::Backend(_), Access::DeniedByUser | Access::DeniedBySystem) => {
            CameraError::PermissionDenied
        }
        (e, _) => e,
    }
}

/// Combines the monitor's verdict with the current capture error.
/// Busy and PermissionDenied say why frames stopped, so they beat the monitor's Lost/Covered;
/// a generic error only shows while the monitor still says Ok.
fn with_error(health: CameraHealth, error: Option<CameraError>) -> CameraHealth {
    match (error, health) {
        (Some(CameraError::Busy), _) => CameraHealth::Busy,
        (Some(CameraError::PermissionDenied), _) => CameraHealth::PermissionDenied,
        (Some(CameraError::NotFound { .. }), _) => CameraHealth::Lost,
        (Some(CameraError::Backend(s)), CameraHealth::Ok) => CameraHealth::Error(s),
        (_, health) => health,
    }
}

fn ms_since(t0: Instant) -> u64 {
    u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Mean Rec.601 luma (same weights as `Frame::mean_luma`) over every `LUMA_STRIDE`-th pixel.
fn sampled_luma(rgb: &[u8]) -> f32 {
    let (mut sum, mut n) = (0.0f64, 0u32);
    for px in rgb.chunks_exact(3).step_by(LUMA_STRIDE) {
        sum += 0.299 * f64::from(px[0]) + 0.587 * f64::from(px[1]) + 0.114 * f64::from(px[2]);
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        (sum / f64::from(n)) as f32
    }
}

fn deliver(frame: Frame, slot: &LatestFrame, status: &Mutex<Status>) {
    let (ts_ms, luma) = (frame.ts_ms, sampled_luma(&frame.data));
    slot.put(frame);
    let mut s = status.lock().unwrap_or_else(PoisonError::into_inner);
    s.frame_at = Some(Instant::now());
    s.error = None;
    s.health.on_frame(ts_ms, luma);
}

/// Frame loop after the first frame (seq 1) was delivered.
fn run(
    src: &mut impl FrameSource,
    cfg: &CameraConfig,
    slot: &LatestFrame,
    stop: &AtomicBool,
    status: &Mutex<Status>,
    t0: Instant,
) {
    let mut seq = 1u64;
    let mut errors = 0u32;
    while !stop.load(Ordering::Relaxed) {
        match next_frame(src, cfg, seq + 1, t0) {
            Ok(frame) => {
                seq = frame.seq;
                errors = 0;
                deliver(frame, slot, status);
            }
            Err(e) => {
                // Log only the first error of a streak; `last_error` always has the latest.
                if errors == 0 {
                    tracing::warn!(error = %e, "camera frame failed");
                }
                status.lock().unwrap_or_else(PoisonError::into_inner).error = Some(e);
                errors = errors.saturating_add(1);
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn next_frame(
    src: &mut impl FrameSource,
    cfg: &CameraConfig,
    seq: u64,
    t0: Instant,
) -> Result<Frame, CameraError> {
    let (rgb, w, h) = src.next_rgb()?;
    if rgb.len() != w as usize * h as usize * 3 {
        return Err(CameraError::Backend(format!(
            "decoded {} bytes for {w}x{h}",
            rgb.len()
        )));
    }
    let (data, w, h) = transform::apply(&rgb, w, h, cfg.rotation, cfg.mirror);
    let ts_ms = ms_since(t0);
    Frame::new(seq, ts_ms, w, h, data).map_err(|e| CameraError::Backend(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(w: u32, h: u32, fmt: FrameFormat, fps: u32) -> CameraFormat {
        CameraFormat::new_from(w, h, fmt, fps)
    }

    /// Delivers a fixed 2x1 frame every 5 ms.
    struct Fake;
    impl FrameSource for Fake {
        fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError> {
            std::thread::sleep(Duration::from_millis(5));
            Ok((vec![1, 2, 3, 4, 5, 6], 2, 1))
        }
    }

    /// Delivers one frame, then blocks forever, like a stalled camera.
    struct StallsAfterFirst(bool);
    impl FrameSource for StallsAfterFirst {
        fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError> {
            if std::mem::replace(&mut self.0, true) {
                loop {
                    std::thread::park();
                }
            }
            Ok((vec![0; 3], 1, 1))
        }
    }

    /// Delivers bright frames every 5 ms, but fails with `Busy` while the flag is set.
    struct FailsWhile(Arc<AtomicBool>);
    impl FrameSource for FailsWhile {
        fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError> {
            std::thread::sleep(Duration::from_millis(5));
            if self.0.load(Ordering::Relaxed) {
                Err(CameraError::Busy)
            } else {
                Ok((vec![200; 3], 1, 1))
            }
        }
    }

    /// Delivers a 1x1 grey frame of the given level every 5 ms.
    struct Solid(u8);
    impl FrameSource for Solid {
        fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError> {
            std::thread::sleep(Duration::from_millis(5));
            Ok((vec![self.0; 3], 1, 1))
        }
    }

    fn spawn_with<S: FrameSource + Send + 'static>(src: S) -> Capture {
        Capture::spawn(
            move || Ok(src),
            &CameraConfig::default(),
            Arc::new(LatestFrame::new()),
        )
        .expect("start")
    }

    #[test]
    fn last_error_tracks_the_current_error_streak() {
        let fail = Arc::new(AtomicBool::new(false));
        let cap = spawn_with(FailsWhile(Arc::clone(&fail)));
        assert_eq!(cap.last_error(), None);
        assert_eq!(cap.health(), CameraHealth::Ok);
        fail.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(cap.last_error(), Some(CameraError::Busy));
        assert_eq!(cap.health(), CameraHealth::Busy);
        fail.store(false, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(cap.last_error(), None, "cleared by the next good frame");
        assert_eq!(cap.health(), CameraHealth::Ok);
    }

    #[test]
    fn health_lost_when_the_camera_stalls() {
        let cap = spawn_with(StallsAfterFirst(false));
        assert_eq!(cap.health(), CameraHealth::Ok);
        std::thread::sleep(Duration::from_millis(1000));
        assert_eq!(cap.health(), CameraHealth::Ok, "1 s without a frame");
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(cap.health(), CameraHealth::Lost, "1.6 s without a frame");
    }

    #[test]
    fn health_covered_when_frames_stay_black() {
        let black = spawn_with(Solid(0));
        let bright = spawn_with(Solid(200));
        std::thread::sleep(Duration::from_millis(1500));
        assert_eq!(black.health(), CameraHealth::Ok, "black for 1.5 s");
        std::thread::sleep(Duration::from_millis(700));
        assert_eq!(black.health(), CameraHealth::Covered, "black for 2.2 s");
        assert_eq!(bright.health(), CameraHealth::Ok);
    }

    #[test]
    fn sampled_luma_matches_full_mean() {
        let (w, h) = (1280u32, 720u32);
        for level in [0u8, 9, 10, 128, 255] {
            let f = Frame::new(0, 0, w, h, vec![level; (w * h * 3) as usize]).expect("frame");
            assert!(
                (sampled_luma(&f.data) - f.mean_luma()).abs() < 1e-3,
                "{level}"
            );
        }
        // Left-to-right ramp with coloured channels: a stride dividing the width would sample
        // the same columns on every row and miss the mean.
        let data: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let x = u8::try_from((i % w) * 255 / (w - 1)).expect("0..=255");
                [x, x / 2, 255 - x]
            })
            .collect();
        let f = Frame::new(0, 0, w, h, data).expect("frame");
        assert!((sampled_luma(&f.data) - f.mean_luma()).abs() < 1.0);
        assert_eq!(sampled_luma(&[]), 0.0);
    }

    #[test]
    fn capture_errors_map_to_health() {
        use CameraHealth as H;
        let backend = || Some(CameraError::Backend("driver fault".into()));
        let busy = Some(CameraError::Busy);
        let denied = Some(CameraError::PermissionDenied);
        assert_eq!(with_error(H::Ok, None), H::Ok);
        assert_eq!(with_error(H::Lost, None), H::Lost);
        assert_eq!(
            with_error(H::Ok, backend()),
            H::Error("driver fault".into())
        );
        // Lost/Covered beat a generic error, but not Busy or PermissionDenied.
        assert_eq!(with_error(H::Lost, backend()), H::Lost);
        assert_eq!(with_error(H::Covered, backend()), H::Covered);
        assert_eq!(with_error(H::Ok, busy.clone()), H::Busy);
        assert_eq!(with_error(H::Lost, busy), H::Busy);
        assert_eq!(with_error(H::Covered, denied), H::PermissionDenied);
        // The device vanished mid-stream.
        let gone = Some(CameraError::NotFound { id: "x".into() });
        assert_eq!(with_error(H::Ok, gone), H::Lost);
    }

    #[test]
    fn start_error_uses_privacy_setting_when_backend_is_vague() {
        let vague = || CameraError::Backend("boom".into());
        assert_eq!(
            explain(vague(), Access::DeniedByUser),
            CameraError::PermissionDenied
        );
        assert_eq!(
            explain(vague(), Access::DeniedBySystem),
            CameraError::PermissionDenied
        );
        assert_eq!(explain(vague(), Access::Allowed), vague());
        assert_eq!(explain(vague(), Access::Unknown), vague());
        // Specific errors are kept: they already say what happened.
        assert_eq!(
            explain(CameraError::Busy, Access::DeniedByUser),
            CameraError::Busy
        );
        let nf = CameraError::NotFound { id: "x".into() };
        assert_eq!(explain(nf.clone(), Access::DeniedByUser), nf);
    }

    struct Fails;
    impl FrameSource for Fails {
        fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError> {
            Err(CameraError::Busy)
        }
    }

    #[test]
    fn pick_format_closest_resolution_then_fps_then_uncompressed() {
        let formats = [
            f(640, 480, FrameFormat::YUYV, 30),
            f(1280, 720, FrameFormat::MJPEG, 30),
            f(1280, 720, FrameFormat::YUYV, 10),
            f(1920, 1080, FrameFormat::MJPEG, 30),
        ];
        assert_eq!(pick_format(&formats, 1280, 720, 30), Some(formats[1]));
        assert_eq!(pick_format(&formats, 1280, 720, 10), Some(formats[2]));
        assert_eq!(pick_format(&formats, 1300, 700, 30), Some(formats[1]));
        let tie = [
            f(640, 480, FrameFormat::MJPEG, 30),
            f(640, 480, FrameFormat::NV12, 30),
        ];
        assert_eq!(pick_format(&tie, 640, 480, 30), Some(tie[1]));
        assert_eq!(pick_format(&[], 640, 480, 30), None);
    }

    #[test]
    fn capture_loop_sequences_and_transforms_frames() {
        let cfg = CameraConfig {
            rotation: 90,
            ..CameraConfig::default()
        };
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::spawn(|| Ok(Fake), &cfg, Arc::clone(&slot)).expect("start");
        let a = slot.take_newer(0, Duration::from_secs(1)).expect("frame");
        let b = slot
            .take_newer(a.seq, Duration::from_secs(1))
            .expect("frame");
        assert!(b.seq > a.seq && b.ts_ms >= a.ts_ms);
        assert_eq!((b.width, b.height), (1, 2));
        assert_eq!(&b.data[..], &[1, 2, 3, 4, 5, 6]);
        assert!(cap.last_frame_at().is_some());
        let t0 = Instant::now();
        cap.stop();
        assert!(
            t0.elapsed() < Duration::from_millis(500),
            "{:?}",
            t0.elapsed()
        );
    }

    #[test]
    fn stop_is_bounded_when_the_camera_stalls() {
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::spawn(
            || Ok(StallsAfterFirst(false)),
            &CameraConfig::default(),
            slot,
        )
        .expect("start");
        std::thread::sleep(Duration::from_millis(50)); // thread is now parked in next_rgb
        let t0 = Instant::now();
        cap.stop();
        let took = t0.elapsed();
        assert!(
            took >= STOP_TIMEOUT && took < Duration::from_millis(2500),
            "{took:?}"
        );
    }

    #[test]
    fn start_reports_first_read_error() {
        let err = Capture::spawn(
            || Ok(Fails),
            &CameraConfig::default(),
            Arc::new(LatestFrame::new()),
        )
        .err();
        assert_eq!(err, Some(CameraError::Busy));
    }

    #[test]
    #[ignore = "needs a camera"]
    fn capture_delivers_frames() {
        let cams = crate::list().expect("list cameras");
        let cam = cams.first().expect("no camera on this machine");
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::start(&cam.id, &CameraConfig::default(), Arc::clone(&slot))
            .expect("start capture");
        let deadline = Instant::now() + Duration::from_secs(2);
        let (mut last_seq, mut n) = (0, 0);
        while Instant::now() < deadline {
            if let Some(frame) = slot.take_newer(last_seq, Duration::from_millis(200)) {
                assert!(frame.seq > last_seq);
                assert_eq!(frame.data.len(), (frame.width * frame.height * 3) as usize);
                if n == 0 {
                    println!(
                        "first frame: seq {} ts {} ms {}x{}",
                        frame.seq, frame.ts_ms, frame.width, frame.height
                    );
                }
                last_seq = frame.seq;
                n += 1;
            }
        }
        println!("{n} frames in 2 s, last seq {last_seq}");
        assert!(n >= 10, "only {n} frames in 2 s");
        assert!(cap.last_frame_at().is_some());
        let health = cap.health();
        println!("health: {health:?}");
        // Covered is legitimate if the lens is covered or the room is dark.
        assert!(
            matches!(health, CameraHealth::Ok | CameraHealth::Covered),
            "{health:?}"
        );
        cap.stop();
    }

    /// WinRT/DeviceWatcher spell the same id in upper case.
    #[test]
    #[ignore = "needs a camera"]
    fn capture_opens_id_in_any_case() {
        let id = crate::list().expect("list cameras")[0]
            .id
            .to_ascii_uppercase();
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::start(&id, &CameraConfig::default(), Arc::clone(&slot))
            .expect("start with upper-case id");
        assert!(slot.take_newer(0, Duration::from_secs(1)).is_some());
        cap.stop();
    }

    /// Child half of `capture_held_by_another_process_is_busy`; a no-op unless spawned by it.
    #[test]
    #[ignore = "needs a camera"]
    fn hold_camera_for_busy_test() {
        if std::env::var_os("FASEC_HOLD_CAMERA").is_none() {
            return;
        }
        let id = crate::list().expect("list cameras")[0].id.clone();
        let cap = Capture::start(&id, &CameraConfig::default(), Arc::new(LatestFrame::new()))
            .expect("holder start");
        println!("HOLDING");
        std::thread::sleep(Duration::from_secs(8));
        cap.stop();
    }

    // Busy only reproduces across processes: a second open in the same process shares the stream.
    #[test]
    #[ignore = "needs a camera"]
    fn capture_held_by_another_process_is_busy() {
        use std::io::BufRead;
        let id = crate::list().expect("list cameras")[0].id.clone();
        let mut child = std::process::Command::new(std::env::current_exe().expect("test exe"))
            .args(["capture::tests::hold_camera_for_busy_test", "--exact"])
            .args(["--ignored", "--nocapture"])
            .env("FASEC_HOLD_CAMERA", "1")
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("spawn holder");
        let stdout = child.stdout.take().expect("piped stdout");
        let held = std::io::BufReader::new(stdout)
            .lines()
            .map_while(Result::ok)
            .any(|l| l.contains("HOLDING"));
        let second = held.then(|| {
            Capture::start(&id, &CameraConfig::default(), Arc::new(LatestFrame::new())).err()
        });
        let _ = child.kill();
        let _ = child.wait();
        println!("second process start: {second:?}");
        assert!(held, "holder process never got the camera");
        assert_eq!(second, Some(Some(CameraError::Busy)));
    }

    #[test]
    #[ignore = "needs a camera"]
    fn capture_unknown_id_is_not_found() {
        let err = Capture::start(
            r"\\?\no-such-camera",
            &CameraConfig::default(),
            Arc::new(LatestFrame::new()),
        )
        .err();
        assert!(matches!(err, Some(CameraError::NotFound { .. })), "{err:?}");
    }

    type Step = Result<(Vec<u8>, u32, u32), CameraError>;

    /// Plays back what the test sends, one result per `next_rgb`; errors out once the sender is gone.
    struct Scripted(mpsc::Receiver<Step>);
    impl FrameSource for Scripted {
        fn next_rgb(&mut self) -> Step {
            self.0
                .recv()
                .unwrap_or_else(|_| Err(CameraError::Backend("script over".into())))
        }
    }

    fn good() -> Step {
        Ok((vec![200; 3], 1, 1))
    }

    /// Polls (instead of fixed sleeps) until `cond` holds.
    fn wait_for(what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn errors_never_reach_the_slot_and_seq_stays_contiguous_after_recovery() {
        let (tx, rx) = mpsc::channel();
        tx.send(good()).expect("first frame");
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::spawn(
            move || Ok(Scripted(rx)),
            &CameraConfig::default(),
            Arc::clone(&slot),
        )
        .expect("start");
        assert_eq!(
            slot.take_newer(0, Duration::from_secs(1)).map(|f| f.seq),
            Some(1)
        );

        // A wrong-sized decode is an error, not a (corrupt) frame.
        tx.send(Ok((vec![0; 5], 2, 1))).expect("send");
        wait_for(
            "size mismatch error",
            || matches!(cap.last_error(), Some(CameraError::Backend(m)) if m.contains("decoded 5 bytes")),
        );
        assert!(matches!(cap.health(), CameraHealth::Error(_)));
        assert!(slot.take_newer(1, Duration::from_millis(50)).is_none());

        // Latest error of the streak wins; each maps to its health (FR-CAM-005/008).
        tx.send(Err(CameraError::Busy)).expect("send");
        wait_for("Busy", || cap.last_error() == Some(CameraError::Busy));
        assert_eq!(cap.health(), CameraHealth::Busy);
        tx.send(Err(CameraError::PermissionDenied)).expect("send");
        wait_for("PermissionDenied", || {
            cap.last_error() == Some(CameraError::PermissionDenied)
        });
        assert_eq!(cap.health(), CameraHealth::PermissionDenied);
        tx.send(Err(CameraError::NotFound { id: "x".into() }))
            .expect("send");
        wait_for("NotFound", || {
            matches!(cap.last_error(), Some(CameraError::NotFound { .. }))
        });
        assert_eq!(
            cap.health(),
            CameraHealth::Lost,
            "unplugged is Lost at once"
        );
        assert!(slot.take_newer(1, Duration::from_millis(50)).is_none());

        // Recovery: failed reads consumed no sequence numbers, the next frame is 2.
        tx.send(good()).expect("send");
        assert_eq!(
            slot.take_newer(1, Duration::from_secs(2)).map(|f| f.seq),
            Some(2)
        );
        wait_for("error cleared", || cap.last_error().is_none());
        assert_eq!(cap.health(), CameraHealth::Ok);
        drop(tx); // unblocks the source so Drop does not wait for it
        drop(cap);
    }

    #[test]
    fn start_rejects_a_first_frame_of_the_wrong_size() {
        let (tx, rx) = mpsc::channel();
        tx.send(Ok((vec![0; 5], 2, 1))).expect("send");
        let err = Capture::spawn(
            move || Ok(Scripted(rx)),
            &CameraConfig::default(),
            Arc::new(LatestFrame::new()),
        )
        .err();
        assert!(matches!(err, Some(CameraError::Backend(m)) if m.contains("decoded 5 bytes")));
        drop(tx);
    }

    #[test]
    fn start_reports_an_open_failure() {
        let err = Capture::spawn(
            || Err::<Fake, _>(CameraError::NotFound { id: "gone".into() }),
            &CameraConfig::default(),
            Arc::new(LatestFrame::new()),
        )
        .err();
        assert_eq!(err, Some(CameraError::NotFound { id: "gone".into() }));
    }

    #[test]
    fn capture_applies_mirror_from_config() {
        let cfg = CameraConfig {
            mirror: true,
            ..CameraConfig::default()
        };
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::spawn(|| Ok(Fake), &cfg, Arc::clone(&slot)).expect("start");
        let f = slot.take_newer(0, Duration::from_secs(1)).expect("frame");
        assert_eq!((f.width, f.height), (2, 1));
        assert_eq!(&f.data[..], &[4, 5, 6, 1, 2, 3]);
        drop(cap);
    }

    #[test]
    fn slow_consumer_gets_the_latest_frame_and_never_the_same_one_twice() {
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::spawn(|| Ok(Fake), &CameraConfig::default(), Arc::clone(&slot))
            .expect("start");
        // Consumer far slower than the camera (5 ms/frame): frames are dropped, not queued.
        std::thread::sleep(Duration::from_millis(300));
        let first = slot.take_newer(0, Duration::from_secs(1)).expect("frame");
        assert!(first.seq > 5, "got a stale frame {}", first.seq);
        let (mut last, mut skipped) = (first.seq, false);
        for _ in 0..5 {
            std::thread::sleep(Duration::from_millis(60));
            let f = slot
                .take_newer(last, Duration::from_secs(1))
                .expect("frame");
            assert!(f.seq > last);
            skipped |= f.seq > last + 1;
            last = f.seq;
        }
        assert!(skipped, "a slow consumer must see gaps (dropped frames)");
        drop(cap);
    }

    #[test]
    fn stalled_camera_hands_out_nothing_stale() {
        let slot = Arc::new(LatestFrame::new());
        let cap = Capture::spawn(
            || Ok(StallsAfterFirst(false)),
            &CameraConfig::default(),
            Arc::clone(&slot),
        )
        .expect("start");
        assert_eq!(
            slot.take_newer(0, Duration::from_secs(1)).map(|f| f.seq),
            Some(1)
        );
        let at = cap.last_frame_at();
        assert!(slot.take_newer(1, Duration::from_millis(200)).is_none());
        assert!(
            slot.take_newer(0, Duration::ZERO).is_none(),
            "frame 1 was taken"
        );
        assert_eq!(cap.last_frame_at(), at, "no frame arrived");
    }
}
