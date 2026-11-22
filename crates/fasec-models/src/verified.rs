use crate::{ModelEntry, ModelError, model_path};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

/// Read ceiling for a manifest entry without a `size`.
const MAX_UNSIZED_BYTES: u64 = 512 * 1024 * 1024;

/// Model bytes that matched their manifest SHA-256. Sessions are built from these exact bytes,
/// so swapping the file on disk afterwards has no effect (finding I2, NFR-SEC-004).
#[derive(Clone)]
pub struct VerifiedModel {
    id: String,
    bytes: Arc<[u8]>,
}

impl VerifiedModel {
    /// Reads `dir/<file>` once, hashes those bytes and returns them only if they match `entry`.
    /// A file whose length differs from `entry.size` (or, without a size, exceeds 512 MiB) is
    /// refused before it is read.
    pub fn load(dir: &Path, entry: &ModelEntry) -> Result<VerifiedModel, ModelError> {
        VerifiedModel::load_capped(dir, entry, MAX_UNSIZED_BYTES)
    }

    /// `load` with `unsized_cap` as the ceiling for entries without a size.
    fn load_capped(
        dir: &Path,
        entry: &ModelEntry,
        unsized_cap: u64,
    ) -> Result<VerifiedModel, ModelError> {
        if !entry.has_valid_sha256() {
            return Err(ModelError::NotFetchable {
                id: entry.id.clone(),
            });
        }
        let path = model_path(dir, entry)?;
        let size_err = |reason: String| ModelError::Size {
            id: entry.id.clone(),
            reason,
        };
        let file = File::open(&path)?;
        let len = file.metadata()?.len();
        let cap = entry.size.unwrap_or(unsized_cap);
        match entry.size {
            Some(n) if len != n => {
                return Err(size_err(format!("file is {len} bytes, manifest says {n}")));
            }
            None if len > cap => {
                return Err(size_err(format!("file is {len} bytes, limit is {cap}")));
            }
            _ => {}
        }
        // One byte past the cap catches a file that grew between the stat and the read.
        let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
        file.take(cap.saturating_add(1)).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > cap {
            return Err(size_err(format!(
                "file grew past {cap} bytes while reading"
            )));
        }
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
    fn verified_model_refuses_file_larger_than_manifest_size() {
        let d = temp_dir("verified_oversize");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        let mut e = entry(HELLO_SHA);
        e.size = Some(11);
        // A size error, not a hash mismatch: the file is refused before it is read.
        match VerifiedModel::load(&d, &e) {
            Err(ModelError::Size { id, .. }) => assert_eq!(id, "test-model"),
            other => panic!("expected Size error, got {other:?}"),
        }
        e.size = Some(12);
        assert_eq!(VerifiedModel::load(&d, &e).unwrap().bytes().len(), 12);
    }

    #[test]
    fn verified_model_without_size_is_capped() {
        let d = temp_dir("verified_unsized_cap");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        match VerifiedModel::load_capped(&d, &entry(HELLO_SHA), 11) {
            Err(ModelError::Size { id, .. }) => assert_eq!(id, "test-model"),
            other => panic!("expected Size error, got {other:?}"),
        }
        VerifiedModel::load_capped(&d, &entry(HELLO_SHA), 12).unwrap();
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
