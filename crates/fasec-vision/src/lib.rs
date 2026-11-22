//! Model runtime and vision pipeline: detect, track, recognize, liveness, attention.

pub mod attention;
pub mod detect;
pub mod error;
pub mod image_ops;
pub mod ir;
pub mod liveness;
pub mod pipeline;
pub mod recognize;
pub mod runtime;
pub mod track;
pub mod traits;

pub use detect::{YuNet, YuNetConfig};
pub use error::VisionError;
/// The liveness crop cut-off, shared with enrolment (which does not link the models).
pub use fasec_core::liveness_fits;
pub use pipeline::{IrEvidence, Pipeline, PipelineConfig, ProcessOutput, TrackIr};
pub use recognize::RECOGNISER_MODEL_ID;
pub use runtime::Runtime;
pub use traits::{Detection, FaceDetector, FaceEmbedder, LivenessModel, PoseEstimator};
