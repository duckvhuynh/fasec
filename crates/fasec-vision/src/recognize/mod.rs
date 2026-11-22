//! Face recognition: 5-point alignment and SFace embeddings.

pub mod align;
pub mod sface;

pub use sface::{RECOGNISER_MODEL_ID, SFACE_COSINE_THRESHOLD, SFace, cosine};
