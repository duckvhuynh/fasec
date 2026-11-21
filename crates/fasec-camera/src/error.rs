use nokhwa::NokhwaError;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CameraError {
    #[error("camera {id} not found")]
    NotFound { id: String },
    #[error("camera is in use by another application")]
    Busy,
    #[error("camera access denied (Windows privacy settings)")]
    PermissionDenied,
    #[error("camera backend error: {0}")]
    Backend(String),
}

// nokhwa flattens Media Foundation errors to strings; the HRESULT survives as `0x........`.
// Not reproduced live (needs the privacy toggle); falls back to `Backend` if the text differs.
const ACCESS_DENIED: [&str; 1] = ["0x80070005"]; // E_ACCESSDENIED
const BUSY: [&str; 2] = [
    // Verified live: a second process opening the ASUS webcam gets this on its first frame read.
    "0xC00D3704", // MF_E_HW_MFT_FAILED_START_STREAMING
    "0xC00D3EA3", // MF_E_VIDEO_RECORDING_DEVICE_PREEMPTED (not reproduced)
];

impl From<NokhwaError> for CameraError {
    fn from(e: NokhwaError) -> Self {
        let msg = e.to_string();
        if ACCESS_DENIED.iter().any(|c| msg.contains(c)) {
            CameraError::PermissionDenied
        } else if BUSY.iter().any(|c| msg.contains(c)) {
            CameraError::Busy
        } else {
            CameraError::Backend(msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_hresults_from_nokhwa_messages() {
        let open = |why: &str| NokhwaError::OpenDeviceError("cam".into(), why.into());
        assert_eq!(
            CameraError::from(open("Access is denied. (0x80070005)")),
            CameraError::PermissionDenied
        );
        // Exact text from the live repro (second process, first `frame()` call).
        assert_eq!(
            CameraError::from(NokhwaError::ReadFrameError(
                "Hardware MFT failed to start streaming due to lack of hardware resources. (0xC00D3704)"
                    .into()
            )),
            CameraError::Busy
        );
        assert!(
            matches!(CameraError::from(open("boom")), CameraError::Backend(m) if m.contains("boom"))
        );
    }
}
