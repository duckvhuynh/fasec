//! ONNX Runtime wrapper: execution-provider selection and session creation.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

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
            env: shared_env()?,
            requested: ep,
            directml: AtomicBool::new(directml),
        };
        tracing::info!(requested = ?ep, active = rt.active_ep(), "onnx runtime ready");
        Ok(rt)
    }

    /// Loads `model_path`; the caller must have SHA-256 verified it (`fasec_models::verify_file`).
    pub fn session(&self, model_path: &Path) -> Result<Session, VisionError> {
        self.session_with_ep(model_path).map(|(s, _)| s)
    }

    /// Like [`Runtime::session`], plus the EP that session runs on.
    pub(crate) fn session_with_ep(
        &self,
        model_path: &Path,
    ) -> Result<(Session, &'static str), VisionError> {
        if !model_path.is_file() {
            return Err(VisionError::Model(format!(
                "model file not found: {}",
                model_path.display()
            )));
        }
        if self.directml.load(Ordering::Relaxed) {
            match directml::session(&self.env, model_path) {
                Ok(s) => return Ok((s, "directml")),
                Err(e) if self.requested == ExecutionProvider::Auto => {
                    tracing::warn!(error = %e, model = %model_path.display(),
                        "DirectML session failed, falling back to CPU");
                    self.directml.store(false, Ordering::Relaxed);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok((cpu_session(&self.env, model_path)?, "cpu"))
    }

    /// A session for one model stage. Under `Auto`, a DirectML session whose first run fails is
    /// rebuilt on CPU (see [`StageSession::run`]).
    pub(crate) fn stage_session(&self, model_path: &Path) -> Result<StageSession, VisionError> {
        let (session, ep) = self.session_with_ep(model_path)?;
        let rebuild: Option<Rebuild<Session>> =
            if ep == "directml" && self.requested == ExecutionProvider::Auto {
                let (env, path) = (self.env.clone(), model_path.to_path_buf());
                Some(Box::new(move || cpu_session(&env, &path)))
            } else {
                None
            };
        Ok(StageSession {
            session,
            ep,
            rebuild,
        })
    }

    /// EP used for *new* sessions: `"directml"` or `"cpu"`. Existing sessions keep theirs, and a
    /// stage may have fallen back to CPU on its own; ask the stage (e.g. `YuNet::ep`).
    pub fn active_ep(&self) -> &'static str {
        if self.directml.load(Ordering::Relaxed) {
            "directml"
        } else {
            "cpu"
        }
    }
}

/// ort allows one environment per process, ever (even after it is dropped), so every `Runtime`
/// shares this one.
static ENV: Mutex<Option<Environment>> = Mutex::new(None);

fn shared_env() -> Result<Environment, VisionError> {
    let mut env = ENV.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(e) = &*env {
        return Ok(e.clone());
    }
    let e = ort::init().with_name("fasec").build()?;
    *env = Some(e.clone());
    Ok(e)
}

fn cpu_session(env: &Environment, model_path: &Path) -> Result<Session, VisionError> {
    Ok(Session::builder(env)?.commit_from_file(model_path)?)
}

type Rebuild<S> = Box<dyn FnOnce() -> Result<S, VisionError> + Send>;

/// A model session and the EP it runs on, with an optional one-shot CPU rebuild.
pub(crate) struct StageSession<S = Session> {
    session: S,
    ep: &'static str,
    /// Armed until the first run completes.
    rebuild: Option<Rebuild<S>>,
}

impl<S> StageSession<S> {
    pub(crate) fn session(&self) -> &S {
        &self.session
    }

    pub(crate) fn ep(&self) -> &'static str {
        self.ep
    }

    /// Runs `f` on the session. If the very first run fails and a CPU rebuild is armed (DirectML
    /// under `Auto`), the session is rebuilt on CPU and `f` retried once; later failures pass
    /// through.
    pub(crate) fn run<T>(
        &mut self,
        mut f: impl FnMut(&mut S) -> Result<T, VisionError>,
    ) -> Result<T, VisionError> {
        let res = f(&mut self.session);
        let Some(rebuild) = self.rebuild.take() else {
            return res;
        };
        match res {
            Ok(v) => Ok(v),
            Err(e) => {
                tracing::warn!(error = %e, "first DirectML run failed, rebuilding session on CPU");
                self.session = rebuild()?;
                self.ep = "cpu";
                f(&mut self.session)
            }
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
pub(crate) mod tests {
    use std::path::PathBuf;

    use super::*;

    pub(crate) fn verified_yunet() -> PathBuf {
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

    /// Fake session: `true` = "DirectML" that fails to run.
    fn stage(fails: bool, armed: bool) -> StageSession<bool> {
        StageSession {
            session: fails,
            ep: if armed { "directml" } else { "cpu" },
            rebuild: armed.then(|| Box::new(|| Ok(false)) as Rebuild<bool>),
        }
    }

    fn fake_run(s: &mut bool) -> Result<u32, VisionError> {
        if *s {
            Err(VisionError::Ort("dml run failed".into()))
        } else {
            Ok(7)
        }
    }

    #[test]
    fn stage_first_run_failure_rebuilds_on_cpu() {
        let mut s = stage(true, true);
        assert_eq!(s.run(fake_run).unwrap(), 7);
        assert_eq!(s.ep(), "cpu");
        assert!(!*s.session());
    }

    #[test]
    fn stage_failure_after_first_run_passes_through() {
        let mut s = stage(false, true);
        assert_eq!(s.run(fake_run).unwrap(), 7);
        s.session = true; // later DirectML failure: no rebuild any more
        assert!(s.run(fake_run).is_err());
        assert_eq!(s.ep(), "directml");
    }

    #[test]
    fn stage_without_rebuild_passes_error_through() {
        let mut s = stage(true, false);
        assert!(s.run(fake_run).is_err());
        assert_eq!(s.ep(), "cpu");
    }

    #[test]
    fn runtime_can_be_created_repeatedly() {
        let a = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let b = Runtime::new(ExecutionProvider::Cpu).unwrap();
        drop((a, b));
        Runtime::new(ExecutionProvider::Cpu).unwrap();
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
