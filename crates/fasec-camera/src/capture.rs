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

/// A running capture thread. Dropping it (or `stop`) stops the stream and joins the thread.
pub struct Capture {
    stop: Arc<AtomicBool>,
    last: Arc<Mutex<Option<Instant>>>,
    thread: Option<JoinHandle<()>>,
}

impl Capture {
    /// Opens camera `info_id` (a `CameraInfo::id`) on its own thread and starts streaming into
    /// `slot`. Returns once the stream is open, or with the open error.
    pub fn start(
        info_id: &str,
        cfg: &CameraConfig,
        slot: Arc<LatestFrame>,
    ) -> Result<Capture, CameraError> {
        let stop = Arc::new(AtomicBool::new(false));
        let last = Arc::new(Mutex::new(None));
        let (tx, rx) = mpsc::sync_channel(1);
        let (id, cfg) = (info_id.to_owned(), cfg.clone());
        let (t_stop, t_last) = (Arc::clone(&stop), Arc::clone(&last));
        let thread = std::thread::Builder::new()
            .name("fasec-camera".into())
            .spawn(move || {
                let _com = ComGuard::init();
                let mut cam = match open_stream(&id, &cfg) {
                    Ok(cam) => cam,
                    Err(e) => {
                        let _ = tx.send(Err(e));
                        return;
                    }
                };
                let _ = tx.send(Ok(()));
                run(&mut cam, &cfg, &slot, &t_stop, &t_last);
                if let Err(e) = cam.stop_stream() {
                    tracing::warn!(error = %e, "stop_stream failed");
                }
            })
            .map_err(|e| CameraError::Backend(format!("spawn capture thread: {e}")))?;
        let started = rx.recv();
        // Built before matching so an early thread exit is joined by Drop.
        let capture = Capture {
            stop,
            last,
            thread: Some(thread),
        };
        match started {
            Ok(Ok(())) => Ok(capture),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(CameraError::Backend(
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
        // shortcut: a camera that stalls without erroring blocks this join until Media
        // Foundation's synchronous read returns; move to an async SourceReader if that shows up.
        if let Some(t) = self.thread.take()
            && t.join().is_err()
        {
            tracing::error!("camera capture thread panicked");
        }
    }
}

fn open_stream(id: &str, cfg: &CameraConfig) -> Result<Camera, CameraError> {
    let mut cam = device::open(id)?;
    let available = device::native_formats(&mut cam)?;
    let fmt = pick_format(&available, cfg.resolution[0], cfg.resolution[1], cfg.fps)
        .ok_or_else(|| CameraError::Backend("camera offers no RGB-decodable format".into()))?;
    tracing::info!(camera = %cam.info().human_name(), format = %fmt, "opening camera");
    cam.set_camera_requset(RequestedFormat::new::<RgbFormat>(
        RequestedFormatType::Exact(fmt),
    ))?;
    cam.open_stream()?;
    Ok(cam)
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

fn run(
    cam: &mut Camera,
    cfg: &CameraConfig,
    slot: &LatestFrame,
    stop: &AtomicBool,
    last: &Mutex<Option<Instant>>,
) {
    let t0 = Instant::now();
    let mut seq = 0u64;
    let mut errors = 0u32;
    while !stop.load(Ordering::Relaxed) {
        match next_frame(cam, cfg, seq + 1, t0) {
            Ok(frame) => {
                seq = frame.seq;
                errors = 0;
                slot.put(frame);
                *last.lock().unwrap_or_else(PoisonError::into_inner) = Some(Instant::now());
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
    cam: &mut Camera,
    cfg: &CameraConfig,
    seq: u64,
    t0: Instant,
) -> Result<Frame, CameraError> {
    let buf = cam.frame()?;
    let (w, h) = (buf.resolution().width(), buf.resolution().height());
    let rgb = buf.decode_image::<RgbFormat>()?.into_raw();
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
