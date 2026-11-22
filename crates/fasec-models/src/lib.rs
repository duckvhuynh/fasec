//! Model manifest, downloader, SHA-256 verification and licence info.

mod fetch;
mod manifest;
mod verified;
mod verify;

pub use fetch::{ensure, model_path};
pub use manifest::{Manifest, ModelEntry};
pub use verified::VerifiedModel;
pub use verify::{sha256_file, verify_file};

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest parse error: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("model `{id}` is not fetchable (empty/invalid url, sha256 or file name)")]
    NotFetchable { id: String },
    #[error("sha256 mismatch for {id_or_path}: expected {expected}, got {actual}")]
    HashMismatch {
        id_or_path: String,
        expected: String,
        actual: String,
    },
    #[error("download failed: {0}")]
    Download(String),
}
