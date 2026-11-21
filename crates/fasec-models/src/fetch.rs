use crate::{ModelEntry, ModelError, verify_file};
use std::io::Read;
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
    let result = download(&entry.url, &part, entry.size)
        .and_then(|()| verify_file(&part, &entry.sha256).map_err(|e| with_id(e, entry)))
        .and_then(|()| Ok(std::fs::rename(&part, &path)?));
    if result.is_err() {
        let _ = std::fs::remove_file(&part);
    }
    result.map(|()| path)
}

/// Streams `url` to `to`; with `max_size`, fails once more than that many bytes arrive.
fn download(url: &str, to: &Path, max_size: Option<u64>) -> Result<(), ModelError> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| ModelError::Download(format!("{url}: {e}")))?;
    // Read at most one byte past the manifest size so an oversized body cannot fill the disk.
    let mut reader = resp
        .into_body()
        .into_reader()
        .take(max_size.map_or(u64::MAX, |s| s.saturating_add(1)));
    let mut file = std::fs::File::create(to)?;
    let written = std::io::copy(&mut reader, &mut file)
        .map_err(|e| ModelError::Download(format!("{url}: {e}")))?;
    if let Some(max) = max_size.filter(|&max| written > max) {
        return Err(ModelError::Download(format!(
            "{url}: body exceeds manifest size of {max} bytes"
        )));
    }
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

    /// Serves one fixed HTTP response on 127.0.0.1 and returns its URL.
    fn serve_once(body: &'static [u8]) -> String {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut req = Vec::new();
            let mut buf = [0u8; 1024];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = s.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                req.extend_from_slice(&buf[..n]);
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(body);
        });
        format!("http://127.0.0.1:{port}/x")
    }

    #[test]
    fn ensure_downloads_and_verifies() {
        let d = temp_dir("ensure_dl_ok");
        let mut e = entry(&serve_once(b"hello fasec\n"), HELLO_SHA, None);
        e.size = Some(12);
        assert_eq!(ensure(&d, &e).unwrap(), d.join("x"));
        assert_eq!(std::fs::read(d.join("x")).unwrap(), b"hello fasec\n");
        assert!(!d.join("x.part").exists());
    }

    #[test]
    fn ensure_rejects_body_larger_than_manifest_size() {
        let d = temp_dir("ensure_dl_too_big");
        let mut e = entry(&serve_once(b"hello fasec\n"), HELLO_SHA, None);
        e.size = Some(5);
        match ensure(&d, &e) {
            Err(ModelError::Download(msg)) => assert!(msg.contains("size"), "{msg}"),
            other => panic!("expected Download size error, got {other:?}"),
        }
        assert!(!d.join("x").exists());
        assert!(!d.join("x.part").exists());
    }

    #[test]
    fn ensure_removes_part_on_downloaded_hash_mismatch() {
        let d = temp_dir("ensure_dl_bad_hash");
        let e = entry(&serve_once(b"not the model"), HELLO_SHA, None);
        match ensure(&d, &e) {
            Err(ModelError::HashMismatch { id_or_path, .. }) => {
                assert!(id_or_path.contains("test-model"))
            }
            other => panic!("expected HashMismatch, got {other:?}"),
        }
        assert!(!d.join("x").exists());
        assert!(!d.join("x.part").exists());
    }
}
