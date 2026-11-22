//! ONNX Runtime wrapper: execution-provider selection and session creation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use fasec_config::ExecutionProvider;
use fasec_models::VerifiedModel;
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

    /// A session built from `model`'s verified bytes (never from the file on disk).
    pub fn session_from_memory(&self, model: &VerifiedModel) -> Result<Session, VisionError> {
        self.session_with_ep(model).map(|(s, _)| s)
    }

    /// Like [`Runtime::session_from_memory`], plus the EP that session runs on.
    pub(crate) fn session_with_ep(
        &self,
        model: &VerifiedModel,
    ) -> Result<(Session, &'static str), VisionError> {
        if self.directml.load(Ordering::Relaxed) {
            match directml::session(&self.env, model.bytes()) {
                Ok(s) => return Ok((s, "directml")),
                Err(e) if self.requested == ExecutionProvider::Auto => {
                    tracing::warn!(error = %e, model = model.id(),
                        "DirectML session failed, falling back to CPU");
                    self.directml.store(false, Ordering::Relaxed);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok((cpu_session(&self.env, model.bytes())?, "cpu"))
    }

    /// A session for one model stage. Under `Auto`, a DirectML session whose first run fails is
    /// rebuilt on CPU from the same verified bytes (see [`StageSession::run`]).
    pub(crate) fn stage_session(&self, model: &VerifiedModel) -> Result<StageSession, VisionError> {
        let (session, ep) = self.session_with_ep(model)?;
        let rebuild = (ep == "directml" && self.requested == ExecutionProvider::Auto).then(|| {
            let env = self.env.clone();
            cpu_rebuild(model, move |bytes| cpu_session(&env, bytes))
        });
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

fn cpu_session(env: &Environment, model: &[u8]) -> Result<Session, VisionError> {
    Ok(Session::builder(env)?.commit_from_memory(model)?)
}

type Rebuild<S> = Box<dyn FnOnce() -> Result<S, VisionError> + Send>;

/// A one-shot rebuild that holds its own handle on `model`'s verified bytes, so the fallback
/// never re-reads the file.
fn cpu_rebuild<S>(
    model: &VerifiedModel,
    build: impl FnOnce(&[u8]) -> Result<S, VisionError> + Send + 'static,
) -> Rebuild<S> {
    let model = model.clone();
    Box::new(move || build(model.bytes()))
}

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

    pub fn session(env: &Environment, model: &[u8]) -> ort::Result<Session> {
        Session::builder(env)?
            // DirectML does not support memory patterns (onnxruntime DirectML EP docs).
            .with_memory_pattern(false)?
            .with_execution_providers([DirectML::default().build().error_on_failure()])?
            .commit_from_memory(model)
    }
}

#[cfg(not(windows))]
mod directml {
    use ort::environment::Environment;
    use ort::session::Session;

    pub fn available() -> Result<(), String> {
        Err("DirectML is Windows-only".into())
    }

    pub fn session(_env: &Environment, _model: &[u8]) -> ort::Result<Session> {
        Err(ort::Error::new("DirectML is Windows-only"))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::{Path, PathBuf};

    use super::*;

    fn models_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }

    /// `models/<id>` verified against the embedded manifest.
    pub(crate) fn verified(id: &str) -> VerifiedModel {
        let entry = fasec_models::Manifest::embedded().get(id).unwrap();
        VerifiedModel::load(&models_dir(), entry).unwrap()
    }

    pub(crate) fn verified_yunet() -> VerifiedModel {
        verified("yunet-2023mar")
    }

    /// A verified "model" whose bytes are `hello fasec\n` (not ONNX).
    fn fake_model(name: &str) -> VerifiedModel {
        let dir = std::env::temp_dir().join(format!("fasec-rt-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.onnx"), b"hello fasec\n").unwrap();
        let entry = fasec_models::ModelEntry {
            id: "fake".into(),
            task: "t".into(),
            url: String::new(),
            sha256: "67609a8d15c99c8fb9253f99ed4d678cbfd6fe88018366d0e68b46d77c4469a2".into(),
            file: Some("m.onnx".into()),
            size: None,
            license: "MIT".into(),
            source: "s".into(),
            default: false,
        };
        VerifiedModel::load(&dir, &entry).unwrap()
    }

    #[test]
    fn fallback_rebuild_uses_verified_bytes() {
        let model = fake_model("rebuild");
        let addr = model.bytes().as_ptr() as usize;
        let rebuild = cpu_rebuild(&model, move |b| Ok(b.as_ptr() as usize == addr));
        drop(model); // the rebuild keeps the verified bytes alive itself
        assert!(
            rebuild().unwrap(),
            "rebuild got other bytes than the verified ones"
        );
    }

    #[test]
    fn session_from_memory_rejects_non_onnx_bytes() {
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        assert!(rt.session_from_memory(&fake_model("garbage")).is_err());
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
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn runtime_cpu_loads_yunet() {
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let s = rt.session_from_memory(&verified_yunet()).unwrap();
        assert_eq!(s.inputs().len(), 1);
        assert!(!s.outputs().is_empty());
        assert_eq!(rt.active_ep(), "cpu");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn runtime_auto_loads_yunet() {
        let rt = Runtime::new(ExecutionProvider::Auto).unwrap();
        let s = rt.session_from_memory(&verified_yunet()).unwrap();
        assert_eq!(s.inputs().len(), 1);
        println!("active_ep = {}", rt.active_ep());
        assert!(["directml", "cpu"].contains(&rt.active_ep()));
    }
}
