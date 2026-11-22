//! Shared types: frames, tracks, threat levels, events, errors.
//!
//! All coordinates in public types are normalised to 0..1 relative to frame width/height.

pub mod embedding;
pub mod frame;
pub mod geom;
pub mod ir;
pub mod threat;
pub mod vision;

pub use embedding::{Embedding, TrackEmbedding};
pub use frame::Frame;
pub use geom::{PointF, RectF};
pub use ir::IrFrame;
pub use threat::ThreatLevel;
pub use vision::{
    CameraHealth, HeadPose, IdentityState, LivenessState, LivenessTier, StageTimings, TrackId,
    TrackObs, VisionReport, liveness_fits,
};

/// Errors produced by `fasec-core` constructors.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CoreError {
    #[error("frame data length {actual} does not match {width}x{height}x3 = {expected}")]
    FrameSize {
        width: u32,
        height: u32,
        expected: usize,
        actual: usize,
    },
}
