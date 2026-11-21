//! ONNX Runtime wrapper: execution-provider selection and session creation.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use fasec_config::ExecutionProvider;
use ort::environment::Environment;
use ort::session::Session;

use crate::VisionError;

/// Creates ONNX Runtime sessions on the configured execution provider.
pub struct Runtime {
    env: Environment,
    requested: ExecutionProvider,
    /// DirectML is in use; `Auto` clears it (for good) on the first DirectML failure.
    directml: AtomicBool,
}

impl Runtime {
    /// `DirectMl` fails if DirectML is unavailable; `Auto` falls back to CPU, logging why.
    pub fn new(ep: ExecutionProvider) -> Result<Runtime, VisionError> {
        let directml = match ep {
            ExecutionProvider::Cpu => false,
            ExecutionProvider::DirectMl => match directml::available() {
                Ok(()) => true,
                Err(why) => {
                    return Err(VisionError::Ort(format!("DirectML unavailable: {why}")));
                }
            },
            ExecutionProvider::Auto => match directml::available() {
                Ok(()) => true,
                Err(why) => {
                    tracing::warn!(%why, "DirectML unavailable, using CPU");
                    false
                }
            },
        };
        let rt = Runtime {
            env: ort::init().with_name("fasec").build()?,
            requested: ep,
            directml: AtomicBool::new(directml),
        };
        tracing::info!(requested = ?ep, active = rt.active_ep(), "onnx runtime ready");
        Ok(rt)
    }

    /// Loads `model_path`; the caller must have SHA-256 verified it (`fasec_models::verify_file`).
    pub fn session(&self, model_path: &Path) -> Result<Session, VisionError> {
        if !model_path.is_file() {
            return Err(VisionError::Model(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }
        if self.directml.load(Ordering::Relaxed) {
            match directml::session(&self.env, model_path) {
                Ok(s) => return Ok(s),
                Err(e) if self.requested == ExecutionProvider::Auto => {
                    tracing::warn!(error = %e, model = %model_path.display(),
                        "DirectML session failed, falling back to CPU");
                    self.directml.store(false, Ordering::Relaxed);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(Session::builder(&self.env)?.commit_from_file(model_path)?)
    }

    /// `"directml"` or `"cpu"`.
    pub fn active_ep(&self) -> &'static str {
        if self.directml.load(Ordering::Relaxed) {
            "directml"
        } else {
            "cpu"
        }
    }
}

#[cfg(windows)]
mod directml {
    use std::path::Path;

    use ort::environment::Environment;
    use ort::ep::{DirectML, ExecutionProvider};
    use ort::session::Session;

    pub fn available() -> Result<(), String> {
        match DirectML::is_available() {
            Ok(true) => Ok(()),
            Ok(false) => Err("onnxruntime build has no DirectML provider".into()),
            Err(e) => Err(e.to_string()),
        }
    }

    pub fn session(env: &Environment, model_path: &Path) -> ort::Result<Session> {
        Session::builder(env)?
            // DirectML does not support memory patterns (onnxruntime DirectML EP docs).
            .with_memory_pattern(false)?
            .with_execution_providers([DirectML::default().build().error_on_failure()])?
            .commit_from_file(model_path)
    }
}

#[cfg(not(windows))]
mod directml {
    use std::path::Path;

    use ort::environment::Environment;
    use ort::session::Session;

    pub fn available() -> Result<(), String> {
        Err("DirectML is Windows-only".into())
    }

    pub fn session(_env: &Environment, _model_path: &Path) -> ort::Result<Session> {
        Err(ort::Error::new("DirectML is Windows-only"))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn verified_yunet() -> PathBuf {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let manifest = fasec_models::Manifest::load(models.join("manifest.toml")).unwrap();
        let entry = manifest.get("yunet-2023mar").unwrap();
        let path = fasec_models::model_path(&models, entry).unwrap();
        fasec_models::verify_file(&path, &entry.sha256).unwrap();
        path
    }

    #[test]
    fn runtime_cpu_is_cpu() {
        assert_eq!(
            Runtime::new(ExecutionProvider::Cpu).unwrap().active_ep(),
            "cpu"
        );
    }

    #[test]
    fn runtime_missing_model_is_error() {
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        assert!(rt.session(Path::new("does/not/exist.onnx")).is_err());
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn runtime_cpu_loads_yunet() {
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let s = rt.session(&verified_yunet()).unwrap();
        assert_eq!(s.inputs().len(), 1);
        assert!(!s.outputs().is_empty());
        assert_eq!(rt.active_ep(), "cpu");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn runtime_auto_loads_yunet() {
        let rt = Runtime::new(ExecutionProvider::Auto).unwrap();
        let s = rt.session(&verified_yunet()).unwrap();
        assert_eq!(s.inputs().len(), 1);
        println!("active_ep = {}", rt.active_ep());
        assert!(["directml", "cpu"].contains(&rt.active_ep()));
    }
}
