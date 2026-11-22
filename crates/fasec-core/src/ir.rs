//! Infrared frames from a Windows Hello style NIR camera.

use std::sync::Arc;

use crate::CoreError;

/// Gray8, row-major IR frame; `data.len() == width * height` is enforced on construction.
/// `illuminated` is true when the camera's IR emitter lit this frame (the camera alternates
/// lit and unlit frames). Not serialisable: IR pixels never leave the process.
#[derive(Clone, PartialEq)]
pub struct IrFrame {
    pub seq: u64,
    pub ts_ms: u64,
    pub width: u32,
    pub height: u32,
    pub illuminated: bool,
    pub data: Arc<[u8]>,
}

impl IrFrame {
    pub fn new(
        seq: u64,
        ts_ms: u64,
        width: u32,
        height: u32,
        illuminated: bool,
        data: Vec<u8>,
    ) -> Result<IrFrame, CoreError> {
        let expected = (width as usize).checked_mul(height as usize);
        if expected != Some(data.len()) {
            return Err(CoreError::FrameSize {
                width,
                height,
                expected: expected.unwrap_or(usize::MAX),
                actual: data.len(),
            });
        }
        Ok(IrFrame {
            seq,
            ts_ms,
            width,
            height,
            illuminated,
            data: data.into(),
        })
    }

    /// True when every pixel is 0: some drivers blank the unlit frames.
    pub fn is_blank(&self) -> bool {
        self.data.iter().all(|&p| p == 0)
    }
}

impl std::fmt::Debug for IrFrame {
    // pixels are left out on purpose
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrFrame")
            .field("seq", &self.seq)
            .field("ts_ms", &self.ts_ms)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("illuminated", &self.illuminated)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_wrong_length() {
        assert!(IrFrame::new(0, 0, 4, 2, true, vec![0; 7]).is_err());
        assert!(IrFrame::new(0, 0, 4, 2, true, vec![0; 8]).is_ok());
    }

    #[test]
    fn blank_detection() {
        assert!(
            IrFrame::new(0, 0, 2, 2, false, vec![0; 4])
                .unwrap()
                .is_blank()
        );
        assert!(
            !IrFrame::new(0, 0, 2, 2, true, vec![0, 0, 9, 0])
                .unwrap()
                .is_blank()
        );
    }

    #[test]
    fn debug_has_no_pixels() {
        let f = IrFrame::new(1, 2, 2, 1, true, vec![123, 45]).unwrap();
        let s = format!("{f:?}");
        assert!(!s.contains("123") && !s.contains("45"), "{s}");
    }
}
