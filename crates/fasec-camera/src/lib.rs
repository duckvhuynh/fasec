//! Camera enumeration, capture, hot-plug and IR source pairing (Windows).

pub mod capture;
pub mod device;
pub mod error;
pub mod health;
pub mod permission;
pub mod slot;
pub mod transform;
pub mod watch;

pub use capture::Capture;
pub use device::{CameraFormat, CameraInfo, list, normalize_id};
pub use error::CameraError;
pub use health::{HealthConfig, HealthMonitor};
pub use permission::{Access, desktop_camera_access};
pub use slot::LatestFrame;
pub use watch::{CameraEvent, WatchGuard, watch};

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "needs a camera"]
    fn list_cameras_smoke() {
        let cams = crate::list().expect("list cameras");
        for c in &cams {
            println!(
                "name={:?}\n  id={:?}\n  formats={:?}",
                c.name, c.id, c.formats
            );
        }
        assert!(!cams.is_empty(), "no camera found");
        assert!(cams.iter().all(|c| !c.id.is_empty()));
    }
}
