//! Camera enumeration, capture, hot-plug and IR source pairing (Windows).

pub mod capture;
pub mod device;
pub mod error;
pub mod slot;
pub mod transform;

pub use capture::Capture;
pub use device::{CameraFormat, CameraInfo, list};
pub use error::CameraError;
pub use slot::LatestFrame;

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
