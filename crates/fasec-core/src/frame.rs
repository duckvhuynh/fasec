//! Camera frames.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::CoreError;

/// RGB8, row-major frame; `data.len() == width * height * 3` is enforced on construction
/// and on deserialisation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "FrameDe")]
pub struct Frame {
    pub seq: u64,
    pub ts_ms: u64,
    pub width: u32,
    pub height: u32,
    pub data: Arc<[u8]>,
}

impl Frame {
    pub fn new(
        seq: u64,
        ts_ms: u64,
        width: u32,
        height: u32,
        data: Vec<u8>,
    ) -> Result<Frame, CoreError> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3));
        if expected != Some(data.len()) {
            return Err(CoreError::FrameSize {
                width,
                height,
                expected: expected.unwrap_or(usize::MAX),
                actual: data.len(),
            });
        }
        Ok(Frame {
            seq,
            ts_ms,
            width,
            height,
            data: data.into(),
        })
    }

    /// Mean Rec.601 luma in 0..255; 0 for an empty frame.
    pub fn mean_luma(&self) -> f32 {
        let (mut r, mut g, mut b) = (0u64, 0u64, 0u64);
        for px in self.data.chunks_exact(3) {
            r += u64::from(px[0]);
            g += u64::from(px[1]);
            b += u64::from(px[2]);
        }
        let n = (self.data.len() / 3) as f64;
        if n == 0.0 {
            return 0.0;
        }
        ((0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64) / n) as f32
    }
}

/// Wire shape for `Frame`; decoding goes through `Frame::new` so the length invariant holds.
#[derive(Deserialize)]
struct FrameDe {
    seq: u64,
    ts_ms: u64,
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl TryFrom<FrameDe> for Frame {
    type Error = CoreError;
    fn try_from(f: FrameDe) -> Result<Self, CoreError> {
        Frame::new(f.seq, f.ts_ms, f.width, f.height, f.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_new_rejects_bad_len() {
        assert!(Frame::new(0, 0, 4, 2, vec![0; 4 * 2 * 3 - 1]).is_err());
        assert!(Frame::new(0, 0, 4, 2, vec![0; 4 * 2 * 3 + 3]).is_err());
        assert!(Frame::new(0, 0, 4, 2, vec![0; 4 * 2 * 3]).is_ok());
    }

    #[test]
    fn frame_deserialize_validates_len() {
        let f = Frame::new(7, 8, 1, 1, vec![1, 2, 3]).unwrap();
        let json = serde_json::to_string(&f).unwrap();
        assert_eq!(serde_json::from_str::<Frame>(&json).unwrap(), f);
        let bad = r#"{"seq":0,"ts_ms":0,"width":2,"height":1,"data":[1,2,3]}"#;
        assert!(serde_json::from_str::<Frame>(bad).is_err());
    }

    #[test]
    fn mean_luma_of_white_is_255() {
        let f = Frame::new(1, 2, 3, 3, vec![255; 27]).unwrap();
        assert!((f.mean_luma() - 255.0).abs() < 1e-3);
    }

    #[test]
    fn mean_luma_of_black_is_0() {
        let f = Frame::new(1, 2, 3, 3, vec![0; 27]).unwrap();
        assert_eq!(f.mean_luma(), 0.0);
    }

    #[test]
    fn mean_luma_uses_rec601_weights() {
        // one pure-red pixel: 0.299 * 255
        let f = Frame::new(0, 0, 1, 1, vec![255, 0, 0]).unwrap();
        assert!((f.mean_luma() - 0.299 * 255.0).abs() < 1e-3);
    }

    #[test]
    fn mean_luma_of_empty_frame_is_0() {
        let f = Frame::new(0, 0, 0, 0, vec![]).unwrap();
        assert_eq!(f.mean_luma(), 0.0);
    }
}
