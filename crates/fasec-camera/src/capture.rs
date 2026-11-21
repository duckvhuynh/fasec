//! Capture thread: camera -> RGB8 -> transform -> `LatestFrame`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use fasec_config::CameraConfig;
use fasec_core::Frame;
use nokhwa::Camera;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, FrameFormat, RequestedFormat, RequestedFormatType};

use crate::device::{self, ComGuard};
use crate::{CameraError, LatestFrame, transform};

/// How long `stop`/`Drop` waits for the capture thread before detaching it.
const STOP_TIMEOUT: Duration = Duration::from_secs(2);
/// How long `start` waits for the first frame. Busy surfaces only on the first read.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(5);

/// A running capture thread. Dropping it (or `stop`) stops the stream and waits at most
/// `STOP_TIMEOUT` for the thread; a thread stuck in a blocking read is detached, not joined.
pub struct Capture {
    stop: Arc<AtomicBool>,
    last: Arc<Mutex<Option<Instant>>>,
    thread: Option<JoinHandle<()>>,
}

/// Where the capture thread gets raw RGB8 frames (`data, width, height`) from.
trait FrameSource {
    fn next_rgb(&mut self) -> Result<(Vec<u8>, u32, u32), CameraError>;
}

impl Capture {
    /// Opens camera `info_id` (a `CameraInfo::id`) on its own thread and starts streaming into
    /// `slot`. Returns once the first frame has arrived, or with the open/first-read error
    /// (e.g. `Busy` when another process holds the camera).
    pub fn start(
        info_id: &str,
        cfg: &CameraConfig,
        slot: Arc<LatestFrame>,
    ) -> Result<Capture, CameraError> {
        let (id, open_cfg) = (info_id.to_owned(), cfg.clone());
        Self::spawn(move || open_stream(&id, &open_cfg), cfg, slot)
    }

    /// `open` runs on the capture thread (nokhwa's `Camera` is not `Send`).
    fn spawn<S: FrameSource>(
        open: impl FnOnce() -> Result<S, CameraError> + Send + 'static,
        cfg: &CameraConfig,
        slot: Arc<LatestFrame>,
    ) -> Result<Capture, CameraError> {
        let stop = Arc::new(AtomicBool::new(false));
        let last = Arc::new(Mutex::new(None));
        let (tx, rx) = mpsc::sync_channel(1);
        let cfg = cfg.clone();
        let (t_stop, t_last) = (Arc::clone(&stop), Arc::clone(&last));
        let thread = std::thread::Builder::new()
            .name("fasec-camera".into())
            .spawn(move || {
                let _com = ComGuard::init();
                let t0 = Instant::now();
                let first = open().and_then(|mut src| {
                    let frame = next_frame(&mut src, &cfg, 1, t0)?;
                    Ok((src, frame))
                });
                let mut src = match first {
                    Ok((src, frame)) => {
                        let _ = tx.send(Ok(()));
                        deliver(frame, &slot, &t_last);
                        src
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                run(&mut src, &cfg, &slot, &t_stop, &t_last, t0);
            })
            .map_err(|e| CameraError::Backend(format!("spawn capture thread: {e}")))?;
        let started = rx.recv_timeout(FIRST_FRAME_TIMEOUT);
        // Built before matching so every failure path stops the thread through Drop.
        let capture = Capture {
            stop,
            last,
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
        *self.last.lock().unwrap_or_else(PoisonError::into_inner)
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

fn deliver(frame: Frame, slot: &LatestFrame, last: &Mutex<Option<Instant>>) {
    slot.put(frame);
    *last.lock().unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
}

/// Frame loop after the first frame (seq 1) was delivered.
fn run(
    src: &mut impl FrameSource,
    cfg: &CameraConfig,
    slot: &LatestFrame,
    stop: &AtomicBool,
    last: &Mutex<Option<Instant>>,
    t0: Instant,
) {
    let mut seq = 1u64;
    let mut errors = 0u32;
    while !stop.load(Ordering::Relaxed) {
        match next_frame(src, cfg, seq + 1, t0) {
            Ok(frame) => {
                seq = frame.seq;
                errors = 0;
                deliver(frame, slot, last);
            }
            Err(e) => {
                // Stall/loss health is derived from `last_frame_at`; log only the first error.
                if errors == 0 {
                    tracing::warn!(error = %e, "camera frame failed");
                }
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
    let ts_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX);
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
}
