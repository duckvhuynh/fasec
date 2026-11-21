use crate::{ModelEntry, ModelError, verify_file};
use std::path::{Path, PathBuf};

/// `dir/<entry.file>` if set, else `dir/<last url path segment>`.
pub fn model_path(dir: &Path, entry: &ModelEntry) -> Result<PathBuf, ModelError> {
    let name = match entry.file.as_deref() {
        Some(f) => f,
        None => {
            let path = entry.url.split(['?', '#']).next().unwrap_or_default();
            path.rsplit('/').next().unwrap_or_default()
        }
    };
    // The name comes from a manifest: refuse anything that could escape `dir`.
    let bad = name.is_empty() || name == "." || name == ".." || name.contains(['/', '\\', ':']);
    if bad {
        return Err(ModelError::NotFetchable {
            id: entry.id.clone(),
        });
    }
    Ok(dir.join(name))
}

/// Returns the path of a verified model file, downloading it if missing.
///
/// Never leaves an unverified file at the returned path and never overwrites an
/// existing file whose hash does not match.
pub fn ensure(dir: &Path, entry: &ModelEntry) -> Result<PathBuf, ModelError> {
    let not_fetchable = || ModelError::NotFetchable {
        id: entry.id.clone(),
    };
    if !entry.has_valid_sha256() {
        return Err(not_fetchable());
    }
    let path = model_path(dir, entry)?;
    if path.exists() {
        verify_file(&path, &entry.sha256).map_err(|e| with_id(e, entry))?;
        tracing::debug!(id = %entry.id, path = %path.display(), "model verified");
        return Ok(path);
    }
    if entry.url.is_empty() {
        return Err(not_fetchable());
    }

    std::fs::create_dir_all(dir)?;
    let mut part = path.clone().into_os_string();
    part.push(".part");
    let part = PathBuf::from(part);
    tracing::info!(id = %entry.id, url = %entry.url, "downloading model");
    let result = download(&entry.url, &part)
        .and_then(|()| verify_file(&part, &entry.sha256).map_err(|e| with_id(e, entry)))
        .and_then(|()| Ok(std::fs::rename(&part, &path)?));
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result.map(|()| path)
}

fn download(url: &str, to: &Path) -> Result<(), ModelError> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| ModelError::Download(format!("{url}: {e}")))?;
    let mut reader = resp.into_body().into_reader();
    let mut file = std::fs::File::create(to)?;
    std::io::copy(&mut reader, &mut file)
        .map_err(|e| ModelError::Download(format!("{url}: {e}")))?;
    file.sync_all()?;
    Ok(())
}

/// Name the model id (not just the path) in hash mismatch errors.
fn with_id(e: ModelError, entry: &ModelEntry) -> ModelError {
    match e {
        ModelError::HashMismatch {
            id_or_path,
            expected,
            actual,
        } => ModelError::HashMismatch {
            id_or_path: format!("{} ({id_or_path})", entry.id),
            expected,
            actual,
        },
        e => e,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::tests::{HELLO_SHA, temp_dir};
    use crate::{ModelEntry, ModelError};

    fn entry(url: &str, sha256: &str, file: Option<&str>) -> ModelEntry {
        ModelEntry {
            id: "test-model".into(),
            task: "t".into(),
            url: url.into(),
            sha256: sha256.into(),
            file: file.map(Into::into),
            size: None,
            license: "MIT".into(),
            source: "s".into(),
            default: false,
        }
    }

    #[test]
    fn model_path_prefers_file_then_url_segment() {
        let d = Path::new("m");
        let e = entry("https://h/a/b.onnx?x=1", HELLO_SHA, None);
        assert_eq!(model_path(d, &e).unwrap(), d.join("b.onnx"));
        let e = entry("https://h/a/b.onnx", HELLO_SHA, Some("c.onnx"));
        assert_eq!(model_path(d, &e).unwrap(), d.join("c.onnx"));
        assert!(model_path(d, &entry("", HELLO_SHA, None)).is_err());
        assert!(model_path(d, &entry("https://h/", HELLO_SHA, None)).is_err());
        assert!(model_path(d, &entry("", HELLO_SHA, Some("../evil"))).is_err());
    }

    #[test]
    fn ensure_refuses_missing_file_with_empty_url() {
        let d = temp_dir("ensure_missing");
        let r = ensure(&d, &entry("", HELLO_SHA, Some("m.onnx")));
        assert!(matches!(r, Err(ModelError::NotFetchable { ref id }) if id == "test-model"));
    }

    #[test]
    fn ensure_refuses_empty_sha() {
        let d = temp_dir("ensure_empty_sha");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        let r = ensure(&d, &entry("http://127.0.0.1:9/m.onnx", "", None));
        assert!(matches!(r, Err(ModelError::NotFetchable { .. })));
    }

    #[test]
    fn ensure_accepts_manually_imported_file() {
        let d = temp_dir("ensure_manual");
        std::fs::write(d.join("m.onnx"), b"hello fasec\n").unwrap();
        let p = ensure(&d, &entry("", HELLO_SHA, Some("m.onnx"))).unwrap();
        assert_eq!(p, d.join("m.onnx"));
    }

    #[test]
    fn ensure_reports_mismatch_on_existing_bad_file() {
        let d = temp_dir("ensure_bad");
        std::fs::write(d.join("m.onnx"), b"corrupted").unwrap();
        let r = ensure(&d, &entry("http://127.0.0.1:9/m.onnx", HELLO_SHA, None));
        match r {
            Err(ModelError::HashMismatch {
                id_or_path,
                expected,
                ..
            }) => {
                assert!(id_or_path.contains("test-model"));
                assert_eq!(expected, HELLO_SHA);
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        // not overwritten
        assert_eq!(std::fs::read(d.join("m.onnx")).unwrap(), b"corrupted");
    }

    #[test]
    fn ensure_keeps_valid_existing_file_without_network() {
        let d = temp_dir("ensure_valid");
        std::fs::write(d.join("x"), b"hello fasec\n").unwrap();
        let p = ensure(&d, &entry("http://127.0.0.1:9/x", HELLO_SHA, None)).unwrap();
        assert_eq!(p, d.join("x"));
    }

    #[test]
    fn ensure_download_failure_leaves_no_file() {
        let d = temp_dir("ensure_dl_fail");
        let r = ensure(&d, &entry("http://127.0.0.1:9/x", HELLO_SHA, None));
        assert!(matches!(r, Err(ModelError::Download(_))), "{r:?}");
        assert!(!d.join("x").exists());
        assert!(!d.join("x.part").exists());
    }
}
