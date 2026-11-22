//! IR (Windows Hello NIR camera) face detection and association with RGB tracks.

pub mod assoc;
pub mod detect;

pub use assoc::IrAssociator;
pub use detect::ir_faces;
