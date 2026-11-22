//! IR (Windows Hello NIR camera) face detection, association with RGB tracks and liveness cues.

pub mod assoc;
pub mod cues;
pub mod detect;

pub use assoc::IrAssociator;
pub use cues::{IrCues, IrLivenessConfig, IrSpoofReason, IrVerdict, ir_cues, judge};
pub use detect::ir_faces;
