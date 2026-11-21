//! Stage traits implemented by the detector, embedder, liveness and pose models.

use fasec_core::{Frame, HeadPose, LivenessTier, PointF, RectF};

use crate::VisionError;

/// One detected face; coordinates normalised 0..1 to the frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub bbox: RectF,
    /// Right eye, left eye, nose tip, right mouth corner, left mouth corner (YuNet order).
    pub landmarks5: [PointF; 5],
    pub score: f32,
}

pub trait FaceDetector: Send {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>, VisionError>;
}

pub trait FaceEmbedder: Send {
    fn embed(&mut self, frame: &Frame, det: &Detection) -> Result<Vec<f32>, VisionError>;
    fn model_id(&self) -> &str;
}

pub trait LivenessModel: Send {
    /// 0..1, 1 = live.
    fn score(&mut self, frame: &Frame, det: &Detection) -> Result<f32, VisionError>;
    fn tier(&self) -> LivenessTier;
}

pub trait PoseEstimator: Send {
    fn estimate(&mut self, frame: &Frame, det: &Detection) -> Result<HeadPose, VisionError>;
}
