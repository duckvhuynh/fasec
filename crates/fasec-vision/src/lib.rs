//! Model runtime and vision pipeline: detect, track, recognize, liveness, attention.

pub mod attention;
pub mod detect;
pub mod error;
pub mod image_ops;
pub mod liveness;
pub mod pipeline;
pub mod recognize;
pub mod runtime;
pub mod track;
pub mod traits;

pub use detect::{YuNet, YuNetConfig};
pub use error::VisionError;
pub use pipeline::{Pipeline, PipelineConfig};
pub use runtime::Runtime;
pub use traits::{Detection, FaceDetector, FaceEmbedder, LivenessModel, PoseEstimator};
