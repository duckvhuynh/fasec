//! Model runtime and vision pipeline: detect, track, recognize, liveness, attention.

pub mod error;
pub mod image_ops;
pub mod runtime;
pub mod traits;

pub use error::VisionError;
pub use runtime::Runtime;
pub use traits::{Detection, FaceDetector, FaceEmbedder, LivenessModel, PoseEstimator};
