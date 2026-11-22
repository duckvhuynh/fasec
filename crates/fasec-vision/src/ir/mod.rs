//! IR (Windows Hello NIR camera) face detection, association with RGB tracks and liveness cues.

pub mod assoc;
pub mod cues;
pub mod detect;

pub use assoc::IrAssociator;
pub use cues::{
    IrCues, IrLivenessConfig, IrSpoofReason, IrVerdict, ambient_is_blank, ir_cues, judge,
};
pub use detect::ir_faces;
