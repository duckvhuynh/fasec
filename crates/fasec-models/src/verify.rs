use crate::ModelError;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::Path;

/// Lower-case hex SHA-256 of a file, streamed.
pub fn sha256_file(path: &Path) -> Result<String, ModelError> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex::encode(hasher.finalize()))
}

/// Ok if the file's SHA-256 equals `sha256_hex` (case-insensitive), else `HashMismatch`.
pub fn verify_file(path: &Path, sha256_hex: &str) -> Result<(), ModelError> {
    let actual = sha256_file(path)?;
    if actual.eq_ignore_ascii_case(sha256_hex) {
        Ok(())
    } else {
        Err(ModelError::HashMismatch {
            id_or_path: path.display().to_string(),
            expected: sha256_hex.to_string(),
            actual,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ModelError;
    use std::path::PathBuf;

    /// sha256("hello fasec\n")
    pub(crate) const HELLO_SHA: &str =
        "67609a8d15c99c8fb9253f99ed4d678cbfd6fe88018366d0e68b46d77c4469a2";

    pub(crate) fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("fasec-models-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn verify_ok() {
        let p = temp_dir("verify_ok").join("f.bin");
        std::fs::write(&p, b"hello fasec\n").unwrap();
        verify_file(&p, HELLO_SHA).unwrap();
        verify_file(&p, &HELLO_SHA.to_uppercase()).unwrap();
    }

    #[test]
    fn verify_mismatch_reports_expected_and_actual() {
        let p = temp_dir("verify_mismatch").join("f.bin");
        std::fs::write(&p, b"hello fasec\n").unwrap();
        let wrong = "1".repeat(64);
        match verify_file(&p, &wrong) {
            Err(ModelError::HashMismatch {
                id_or_path,
                expected,
                actual,
            }) => {
                assert!(id_or_path.contains("f.bin"));
                assert_eq!(expected, wrong);
                assert_eq!(actual, HELLO_SHA);
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
    }
}
