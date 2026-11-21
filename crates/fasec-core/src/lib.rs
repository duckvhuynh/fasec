//! Shared types: frames, tracks, threat levels, events, errors.
//!
//! All coordinates in public types are normalised to 0..1 relative to frame width/height.

pub mod frame;
pub mod geom;
pub mod threat;
pub mod vision;

pub use frame::Frame;
pub use geom::{PointF, RectF};
pub use threat::ThreatLevel;
pub use vision::{
    CameraHealth, HeadPose, IdentityState, LivenessState, LivenessTier, StageTimings, TrackId,
    TrackObs, VisionReport,
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
