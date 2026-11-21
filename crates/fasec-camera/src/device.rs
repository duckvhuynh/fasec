//! Camera enumeration (Media Foundation via nokhwa).

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{ApiBackend, CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::{Camera, FormatDecoder, NokhwaError};

use crate::CameraError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraInfo {
    /// Media Foundation symbolic link (`MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK`):
    /// independent of enumeration order. Persist this, never an index.
    pub id: String,
    pub name: String,
    pub formats: Vec<CameraFormat>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CameraFormat {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// Lists video capture devices with the formats we can decode to RGB8. A device whose formats
/// cannot be read (e.g. held exclusively by another app) is still listed, with no formats.
pub fn list() -> Result<Vec<CameraInfo>, CameraError> {
    // Own thread so nokhwa's COM init never touches the caller's apartment.
    std::thread::spawn(|| {
        let _com = ComGuard::init();
        let devices = nokhwa::query(ApiBackend::MediaFoundation)?;
        Ok(devices
            .into_iter()
            .map(|d| {
                let id = d.misc();
                let formats = open(&id)
                    .and_then(|mut cam| formats(&mut cam))
                    .unwrap_or_else(|e| {
                        tracing::warn!(camera = %d.human_name(), error = %e, "cannot read camera formats");
                        Vec::new()
                    });
                CameraInfo {
                    id,
                    name: d.human_name(),
                    formats,
                }
            })
            .collect())
    })
    .join()
    .map_err(|_| CameraError::Backend("camera enumeration thread panicked".into()))?
}

/// Opens (without streaming) the device whose symbolic link is `id`.
pub(crate) fn open(id: &str) -> Result<Camera, CameraError> {
    Camera::with_backend(
        CameraIndex::String(id.to_owned()),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        ApiBackend::MediaFoundation,
    )
    .map_err(|e| match e {
        // nokhwa's MF backend resolves the symlink to the current index and reports a miss so.
        NokhwaError::OpenDeviceError(_, ref why) if why == "Not Found" => {
            CameraError::NotFound { id: id.to_owned() }
        }
        e => e.into(),
    })
}

/// RGB-decodable formats, as raw nokhwa formats (fourcc kept for format selection).
pub(crate) fn native_formats(
    cam: &mut Camera,
) -> Result<Vec<nokhwa::utils::CameraFormat>, CameraError> {
    Ok(cam
        .compatible_camera_formats()?
        .into_iter()
        .filter(|f| RgbFormat::FORMATS.contains(&f.format()))
        .collect())
}

fn formats(cam: &mut Camera) -> Result<Vec<CameraFormat>, CameraError> {
    let mut v: Vec<CameraFormat> = native_formats(cam)?
        .into_iter()
        .map(|f| CameraFormat {
            width: f.width(),
            height: f.height(),
            fps: f.frame_rate(),
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

/// Per-thread COM apartment, initialised with the same flags nokhwa uses so its own
/// `CoInitializeEx` on this thread succeeds (S_FALSE) instead of failing with RPC_E_CHANGED_MODE.
pub(crate) struct ComGuard(#[cfg_attr(not(windows), allow(dead_code))] bool);

impl ComGuard {
    pub(crate) fn init() -> Self {
        #[cfg(windows)]
        {
            use windows::Win32::System::Com::{
                COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE, CoInitializeEx,
            };
            // SAFETY: plain per-thread COM init; balanced by CoUninitialize in Drop on success.
            let hr =
                unsafe { CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE) };
            ComGuard(hr.is_ok())
        }
        #[cfg(not(windows))]
        ComGuard(false)
    }
}

impl Drop for ComGuard {
    fn drop(&mut self) {
        #[cfg(windows)]
        if self.0 {
            // SAFETY: matches the successful CoInitializeEx on this same thread.
            unsafe { windows::Win32::System::Com::CoUninitialize() };
        }
    }
}
