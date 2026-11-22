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

/// A Windows Hello style infrared frame source (WinRT `MediaFrameSourceKind::Infrared`).
/// Ids are normalised (`normalize_id`); `IrCapture::start` takes `id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrSourceInfo {
    /// WinRT `MediaFrameSourceInfo::Id`, unique within its group.
    pub id: String,
    pub name: String,
    pub group_id: String,
    pub device_id: String,
}

/// One WinRT frame source of any kind, as enumerated; the input of `pick_paired`.
#[derive(Debug, Clone)]
pub(crate) struct FrameSourceEntry {
    pub(crate) infrared: bool,
    pub(crate) info: IrSourceInfo,
}

/// Every infrared source on the machine. The same IR device can appear in several groups.
pub fn ir_sources() -> Result<Vec<IrSourceInfo>, CameraError> {
    Ok(frame_sources()?
        .into_iter()
        .filter(|s| s.infrared)
        .map(|s| s.info)
        .collect())
}

/// The IR source that belongs to RGB camera `rgb_id` (a `CameraInfo::id`), if any.
pub fn paired_ir_for(rgb_id: &str) -> Result<Option<IrSourceInfo>, CameraError> {
    Ok(pick_paired(rgb_id, &frame_sources()?))
}

/// An IR source in a group that also holds the RGB camera, else the one IR device on the same
/// USB `VID_xxxx&PID_xxxx` (`None` if two devices share it), else `None`.
pub(crate) fn pick_paired(rgb_id: &str, sources: &[FrameSourceEntry]) -> Option<IrSourceInfo> {
    let rgb = normalize_id(rgb_id);
    let groups: Vec<&str> = sources
        .iter()
        .filter(|s| !s.infrared && normalize_id(&s.info.device_id) == rgb)
        .map(|s| s.info.group_id.as_str())
        .collect();
    let ir = sources.iter().filter(|s| s.infrared).map(|s| &s.info);
    let same_group = ir.clone().find(|s| groups.contains(&s.group_id.as_str()));
    same_group
        .or_else(|| {
            let usb = usb_vid_pid(&rgb)?;
            let mut same_usb = ir.filter(|s| usb_vid_pid(&normalize_id(&s.device_id)) == Some(usb));
            let first = same_usb.next()?;
            // Two identical cameras: guessing could pair the RGB with the other one's IR.
            same_usb
                .all(|s| normalize_id(&s.device_id) == normalize_id(&first.device_id))
                .then_some(first)
        })
        .cloned()
}

/// `vid_xxxx&pid_xxxx` of a normalised USB device path.
fn usb_vid_pid(id: &str) -> Option<&str> {
    let at = id.find("vid_")?;
    let key = id.get(at..at + "vid_xxxx&pid_xxxx".len())?;
    key.as_bytes()[8..].starts_with(b"&pid_").then_some(key)
}

#[cfg(windows)]
fn frame_sources() -> Result<Vec<FrameSourceEntry>, CameraError> {
    use windows::Media::Capture::Frames::{MediaFrameSourceGroup, MediaFrameSourceKind};
    // Own thread: blocking on WinRT async ops must not happen on a caller's UI thread.
    let enumerate = || -> windows::core::Result<Vec<FrameSourceEntry>> {
        let groups = MediaFrameSourceGroup::FindAllAsync()?.join()?;
        let (mut out, mut group_names) = (Vec::new(), Vec::new());
        for g in groups {
            let (group_id, group_name) = (normalize_id(&g.Id()?.to_string()), g.DisplayName()?);
            for s in g.SourceInfos()? {
                let device = s.DeviceInformation()?;
                out.push(FrameSourceEntry {
                    infrared: s.SourceKind()? == MediaFrameSourceKind::Infrared,
                    info: IrSourceInfo {
                        id: normalize_id(&s.Id()?.to_string()),
                        name: device.Name()?.to_string(),
                        group_id: group_id.clone(),
                        device_id: normalize_id(&device.Id()?.to_string()),
                    },
                });
            }
            group_names.push((group_id, group_name.to_string()));
        }
        // The device's own group carries its product name ("ASUS IR camera"); the interface
        // name is "<hostname> - Front" on the dev laptop.
        for s in &mut out {
            if let Some((_, n)) = group_names.iter().find(|(g, _)| *g == s.info.device_id) {
                s.info.name.clone_from(n);
            }
        }
        Ok(out)
    };
    std::thread::spawn(enumerate)
        .join()
        .map_err(|_| CameraError::Backend("IR enumeration thread panicked".into()))?
        .map_err(|e| crate::ir::win_err("enumerate frame sources", &e))
}

#[cfg(not(windows))]
fn frame_sources() -> Result<Vec<FrameSourceEntry>, CameraError> {
    Ok(Vec::new())
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

    /// The ASUS layout from SP-7: group 0 RGB only, group 1 RGB + IR, group 2 IR only.
    fn asus(ir_only_first: bool) -> Vec<FrameSourceEntry> {
        let rgb = r"\\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777}\GLOBAL";
        let ir = r"\\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7}\GLOBAL";
        let src = |infrared, id: &str, group: &str, device: &str| FrameSourceEntry {
            infrared,
            info: IrSourceInfo {
                id: normalize_id(id),
                name: "n".into(),
                group_id: normalize_id(group),
                device_id: normalize_id(device),
            },
        };
        let mut v = vec![
            src(false, "Source#0@rgb", rgb, rgb),
            src(false, "Source#0@rgb", "dmft", rgb),
            src(true, "Source#1@ir", "dmft", ir),
        ];
        let ir_only = src(true, "Source#0@ir", ir, ir);
        if ir_only_first {
            v.insert(0, ir_only);
        } else {
            v.push(ir_only);
        }
        v
    }

    const ASUS_RGB_MF: &str = r"\\?\usb#vid_3277&pid_0096&mi_00#6&ead33bb&0&0000#{e5323777}\global";

    #[test]
    fn pairs_the_ir_source_in_the_rgb_cameras_group_first() {
        for ir_only_first in [false, true] {
            let got = pick_paired(ASUS_RGB_MF, &asus(ir_only_first)).map(|s| s.group_id);
            assert_eq!(
                got.as_deref(),
                Some("dmft"),
                "ir_only_first={ir_only_first}"
            );
        }
    }

    #[test]
    fn falls_back_to_the_same_usb_vid_pid_in_any_case() {
        // No shared group: only the IR-only group is left.
        let sources: Vec<_> = asus(false)
            .into_iter()
            .filter(|s| s.info.group_id != "dmft")
            .collect();
        let got = pick_paired(&ASUS_RGB_MF.to_ascii_uppercase(), &sources);
        assert_eq!(got.map(|s| s.id).as_deref(), Some("source#0@ir"));
    }

    #[test]
    fn no_pair_for_another_device_or_a_non_usb_id() {
        let sources: Vec<_> = asus(false)
            .into_iter()
            .filter(|s| s.info.group_id != "dmft")
            .collect();
        let other = r"\\?\usb#vid_046d&pid_0825&mi_00#7&1&0&0000#{e5323777}\global";
        assert_eq!(pick_paired(other, &sources), None);
        assert_eq!(pick_paired(r"\\?\root#camera#0000", &sources), None);
        assert_eq!(
            pick_paired(r"\\?\usb#vid_3277", &sources),
            None,
            "truncated id"
        );
        assert_eq!(pick_paired(ASUS_RGB_MF, &[]), None);
    }

    #[test]
    fn two_ir_devices_on_the_same_vid_pid_are_ambiguous() {
        let mut sources: Vec<_> = asus(false)
            .into_iter()
            .filter(|s| s.info.group_id != "dmft")
            .collect();
        // The same IR device listed in a second group is still one device.
        let mut again = sources.last().expect("ir-only source").clone();
        again.info.group_id = "other group".into();
        sources.push(again);
        assert!(pick_paired(ASUS_RGB_MF, &sources).is_some());
        // A second, identical USB camera: no group match, so we cannot tell which IR is ours.
        let twin = r"\\?\usb#vid_3277&pid_0096&mi_02#7&1234abc&0&0002#{24e552d7}\global";
        let mut other = sources.last().expect("ir-only source").clone();
        other.info.group_id = twin.into();
        other.info.device_id = twin.into();
        sources.push(other);
        assert_eq!(pick_paired(ASUS_RGB_MF, &sources), None);
    }

    #[test]
    fn a_non_ascii_id_does_not_panic() {
        // 'é' straddles byte 8 of the would-be `vid_xxxx&pid_xxxx` key.
        assert_eq!(usb_vid_pid(r"\\?\usb#vid_123é&pid_0096&mi_00"), None);
        assert_eq!(
            pick_paired(r"\\?\usb#vid_123é&pid_0096&mi_00", &asus(false)),
            None
        );
        assert_eq!(
            usb_vid_pid(r"\\?\usb#vid_3277&pid_0096&mi_00"),
            Some("vid_3277&pid_0096")
        );
    }
}
