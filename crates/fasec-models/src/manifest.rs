use crate::ModelError;
use serde::Deserialize;
use std::path::Path;

/// One `[[model]]` entry of `models/manifest.toml` (same schema for `assets/test-assets.toml`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub task: String,
    pub url: String,
    pub sha256: String,
    /// Local file name; needed when `url` is empty (manually imported exports).
    pub file: Option<String>,
    pub size: Option<u64>,
    pub license: String,
    pub source: String,
    #[serde(default)]
    pub default: bool,
}

impl ModelEntry {
    /// True when the entry may be downloaded: non-empty `url` and a well-formed sha256.
    pub fn is_fetchable(&self) -> bool {
        !self.url.is_empty() && self.has_valid_sha256()
    }

    pub(crate) fn has_valid_sha256(&self) -> bool {
        self.sha256.len() == 64 && self.sha256.bytes().all(|b| b.is_ascii_hexdigit())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Manifest {
    pub model: Vec<ModelEntry>,
}

impl Manifest {
    pub fn parse(s: &str) -> Result<Manifest, ModelError> {
        Ok(toml::from_str(s)?)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Manifest, ModelError> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    pub fn get(&self, id: &str) -> Option<&ModelEntry> {
        self.model.iter().find(|e| e.id == id)
    }

    pub fn defaults(&self) -> impl Iterator<Item = &ModelEntry> {
        self.model.iter().filter(|e| e.default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPO_MANIFEST: &str = include_str!("../../../models/manifest.toml");

    fn entry(url: &str, sha256: &str) -> ModelEntry {
        ModelEntry {
            id: "x".into(),
            task: "t".into(),
            url: url.into(),
            sha256: sha256.into(),
            file: None,
            size: None,
            license: "MIT".into(),
            source: "s".into(),
            default: false,
        }
    }

    #[test]
    fn parse_repo_manifest() {
        let m = Manifest::parse(REPO_MANIFEST).unwrap();
        assert!(m.get("yunet-2023mar").is_some());
        let defaults: Vec<&str> = m.defaults().map(|e| e.id.as_str()).collect();
        assert_eq!(defaults, ["yunet-2023mar", "sface-2021dec"]);
        assert!(m.defaults().all(ModelEntry::is_fetchable));
        assert!(!m.get("6drepnet").unwrap().is_fetchable());
    }

    #[test]
    fn empty_url_not_fetchable() {
        assert!(!entry("", &"a".repeat(64)).is_fetchable());
        assert!(!entry("https://e/x.onnx", "").is_fetchable());
        assert!(entry("https://e/x.onnx", &"a".repeat(64)).is_fetchable());
    }

    #[test]
    fn bad_sha_len_not_fetchable() {
        assert!(!entry("https://e/x.onnx", &"a".repeat(63)).is_fetchable());
        assert!(!entry("https://e/x.onnx", &"g".repeat(64)).is_fetchable());
    }
}
