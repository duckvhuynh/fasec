//! Camera enumeration (Media Foundation via nokhwa).

use std::sync::{Mutex, MutexGuard, PoisonError};

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
        // Held for the whole enumeration: every device opened here is also dropped here.
        let _mf = mf_lock();
        let devices = nokhwa::query(ApiBackend::MediaFoundation)?;
        Ok(devices
            .into_iter()
            .map(|d| {
                let formats = open_symlink(&d.misc())
                    .and_then(|mut cam| formats(&mut cam))
                    .unwrap_or_else(|e| {
                        tracing::warn!(camera = %d.human_name(), error = %e, "cannot read camera formats");
                        Vec::new()
                    });
                CameraInfo {
                    id: normalize_id(&d.misc()),
                    name: d.human_name(),
                    formats,
                }
            })
            .collect())
    })
    .join()
    .map_err(|_| CameraError::Backend("camera enumeration thread panicked".into()))?
}

/// Canonical form of a device id (ASCII lowercase). MF symlinks and WinRT device ids name the
/// same device path in different letter case; every id we hand out or look up goes through this.
pub fn normalize_id(id: &str) -> String {
    id.to_ascii_lowercase()
}

/// Serialises nokhwa's MF init, device open and device drop across threads: nokhwa keeps a
/// non-atomic camera refcount and calls `MFShutdown` when it reaches zero. Never hold it while
/// streaming.
pub(crate) fn mf_lock() -> MutexGuard<'static, ()> {
    static MF: Mutex<()> = Mutex::new(());
    MF.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Opens (without streaming) the device `id` (a `CameraInfo::id`, matched in any letter case).
/// Caller holds `mf_lock`.
pub(crate) fn open(id: &str) -> Result<Camera, CameraError> {
    let want = normalize_id(id);
    // nokhwa compares symlinks exactly, so look up the exact spelling MF uses right now.
    let symlink = nokhwa::query(ApiBackend::MediaFoundation)?
        .into_iter()
        .map(|d| d.misc())
        .find(|s| normalize_id(s) == want)
        .ok_or_else(|| CameraError::NotFound { id: id.to_owned() })?;
    open_symlink(&symlink)
}

/// Opens (without streaming) the device whose MF symbolic link is exactly `id`.
fn open_symlink(id: &str) -> Result<Camera, CameraError> {
    Camera::with_backend(
        CameraIndex::String(id.to_owned()),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        ApiBackend::MediaFoundation,
    )
    .map_err(|e| match e {
        // nokhwa's MF backend resolves the symlink to the current index and reports a miss so
        // (verified live in Task 10); here only if the device vanished since `query`.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_id_matches_winrt_and_mf_ids() {
        // Same ASUS webcam as seen by WinRT (DeviceInformation) and by MF (nokhwa misc()).
        let winrt = r"\\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\GLOBAL";
        let mf = r"\\?\usb#vid_3277&pid_0096&mi_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global";
        assert_eq!(normalize_id(winrt), mf);
        assert_eq!(normalize_id(mf), mf);
    }
}
