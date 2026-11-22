use crate::{ModelEntry, ModelError, model_path};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

/// Model bytes that matched their manifest SHA-256. Sessions are built from these exact bytes,
/// so swapping the file on disk afterwards has no effect (finding I2, NFR-SEC-004).
#[derive(Clone)]
pub struct VerifiedModel {
    id: String,
    bytes: Arc<[u8]>,
}

impl VerifiedModel {
    /// Reads `dir/<file>` once, hashes those bytes and returns them only if they match `entry`.
    pub fn load(dir: &Path, entry: &ModelEntry) -> Result<VerifiedModel, ModelError> {
        if !entry.has_valid_sha256() {
            return Err(ModelError::NotFetchable {
                id: entry.id.clone(),
            });
        }
        let path = model_path(dir, entry)?;
        let bytes = std::fs::read(&path)?;
        let actual = hex::encode(Sha256::digest(&bytes));
        if !actual.eq_ignore_ascii_case(&entry.sha256) {
            return Err(ModelError::HashMismatch {
                id_or_path: format!("{} ({})", entry.id, path.display()),
                expected: entry.sha256.clone(),
                actual,
            });
        }
        tracing::debug!(id = %entry.id, len = bytes.len(), "model verified");
        Ok(VerifiedModel {
            id: entry.id.clone(),
            bytes: bytes.into(),
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl std::fmt::Debug for VerifiedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerifiedModel")
            .field("id", &self.id)
            .field("len", &self.bytes.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::tests::{HELLO_SHA, temp_dir};
    use crate::{ModelEntry, ModelError};

    fn entry(sha256: &str) -> ModelEntry {
        ModelEntry {
            id: "test-model".into(),
            task: "t".into(),
            url: String::new(),
            sha256: sha256.into(),
            file: Some("m.onnx".into()),
            size: None,
            license: "MIT".into(),
            source: "s".into(),
            default: false,
        }
    }

    #[test]
    fn verified_model_rejects_mismatch() {
        let d = temp_dir("verified_mismatch");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        let wrong = "1".repeat(64);
        match VerifiedModel::load(&d, &entry(&wrong)) {
            Err(ModelError::HashMismatch {
                id_or_path,
                expected,
                actual,
            }) => {
                assert!(id_or_path.contains("test-model"), "{id_or_path}");
                assert_eq!(expected, wrong);
                assert_eq!(actual, HELLO_SHA);
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn verified_model_refuses_empty_sha() {
        let d = temp_dir("verified_empty_sha");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        let r = VerifiedModel::load(&d, &entry(""));
        assert!(matches!(r, Err(ModelError::NotFetchable { .. })), "{r:?}");
    }

    #[test]
    fn verified_model_bytes_are_the_hashed_bytes() {
        let d = temp_dir("verified_bytes");
        let p = d.join("m.onnx");
        std::fs::write(&p, b"hello fasec\n").unwrap();
        let m = VerifiedModel::load(&d, &entry(HELLO_SHA)).unwrap();
        std::fs::write(&p, b"swapped on disk").unwrap();
        assert_eq!(m.bytes(), b"hello fasec\n");
        assert_eq!(m.id(), "test-model");
        // ...and the swapped file is rejected at the next load.
        assert!(matches!(
            VerifiedModel::load(&d, &entry(HELLO_SHA)),
            Err(ModelError::HashMismatch { .. })
        ));
    }

    #[test]
    fn verified_model_debug_shows_id_and_len_only() {
        let d = temp_dir("verified_debug");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        let s = format!("{:?}", VerifiedModel::load(&d, &entry(HELLO_SHA)).unwrap());
        assert!(s.contains("test-model") && s.contains("12"), "{s}");
        assert!(!s.contains("104"), "{s}"); // b'h' — no byte dump
    }
}
