use crate::protect::ProtectError;

#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("gallery database error: {0}")]
    Db(String),
    #[error(transparent)]
    Protect(#[from] ProtectError),
    #[error("an owner is already enrolled")]
    OwnerExists,
    #[error("person not found")]
    NotFound,
    #[error("template {template_id} is corrupt or cannot be decrypted")]
    CorruptTemplate { template_id: i64 },
    #[error("unsupported gallery schema version {0}")]
    UnsupportedSchema(u32),
    #[error("PIN error: {0}")]
    Pin(String),
    #[error(
        "templates were enrolled with {stored:?} but the current model is {current}; re-enrolment required"
    )]
    ModelMismatch {
        stored: Vec<String>,
        current: String,
    },
    /// Embedding is not `EMBEDDING_DIM` finite values.
    #[error("embedding must be {} finite values", crate::store::EMBEDDING_DIM)]
    InvalidEmbedding,
    /// Template metadata (e.g. quality) is out of range.
    #[error("invalid template: {0}")]
    InvalidTemplate(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Gallery rows contradict each other (e.g. one person with two roles).
    #[error("inconsistent gallery: {0}")]
    InconsistentGallery(String),
}

impl From<rusqlite::Error> for IdentityError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Db(e.to_string())
    }
}
