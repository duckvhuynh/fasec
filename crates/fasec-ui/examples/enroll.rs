//! Guided enrolment (FR-ID-002): live camera, pose prompts, reject reasons and the owner PIN.
//!
//! `cargo run -p fasec-ui --example enroll --release -- --gallery <path> --name <name>
//!  [--role owner|authorised] [--camera <id>] [--seconds N]`
//!
//! Order: gallery checks (an owner already enrolled, no PIN for an authorised person) fail
//! before any window opens; then the PIN step (new owner: choose it twice; otherwise the owner
//! PIN, 3 attempts, FR-ID-006); only then the camera starts. Threads follow `live_debug`: the UI
//! thread draws and owns the gallery, one worker owns the `Pipeline`, `Capture` and `Enrolment`
//! and stops the camera as soon as the last sample is accepted or the window closes. Nothing is
//! written until all samples are in; then person, templates and (new owner) PIN go in one
//! transaction (`Gallery::enrol`). The preview is mirrored for display (selfie view) and the
//! prompts are worded for it. `--seconds N` closes the window after N s (smoke runs).

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use eframe::egui::{
    self, Color32, ColorImage, Id, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2, ViewportCommand, text_edit::TextEditState,
};
use fasec_camera::{Capture, LatestFrame};
use fasec_config::{CameraConfig, ExecutionProvider};
use fasec_core::{CameraHealth, Frame, HeadPose, TrackObs};
use fasec_identity::{
    EnrolConfig, EnrolledSample, Enrolment, Gallery, IdentityError, NewTemplate, PersonId, Pose,
    RejectReason, Role, SampleVerdict,
};
use fasec_platform::Dpapi;
use fasec_vision::{Pipeline, PipelineConfig, RECOGNISER_MODEL_ID, Runtime};
use zeroize::{Zeroize, Zeroizing};

const USAGE: &str = "usage: enroll --gallery <path> --name <name> [--role owner|authorised] \
                     [--camera <id>] [--seconds N]";
const MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
const FETCH_HINT: &str =
    "Models missing or invalid: run `cargo run -p fasec-models --example fetch`";
/// Same bound as `live_debug`: `Capture::stop` takes up to ~2 s.
const QUIT_TIMEOUT: Duration = Duration::from_secs(3);
const MAX_PIN_ATTEMPTS: u32 = 3;
/// How long the final message stays up before the window closes itself.
const LINGER: Duration = Duration::from_secs(4);
const PIN_IDS: [&str; 2] = ["pin_first", "pin_second"];

#[derive(Debug, PartialEq)]
struct Args {
    gallery: PathBuf,
    name: String,
    role: Role,
    camera: Option<String>,
    seconds: Option<u64>,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let (mut gallery, mut name, mut role, mut camera, mut seconds) =
        (None, None, Role::Owner, None, None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .with_context(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--gallery" => gallery = Some(PathBuf::from(value)),
            "--name" => name = Some(value.trim().to_owned()),
            "--role" => {
                role = match value.as_str() {
                    "owner" => Role::Owner,
                    "authorised" => Role::Authorised,
                    _ => bail!("--role must be owner or authorised\n{USAGE}"),
                }
            }
            "--camera" => camera = Some(value.clone()),
            "--seconds" => seconds = Some(value.parse().context(USAGE)?),
            _ => bail!("unknown argument {flag}\n{USAGE}"),
        }
    }
    let name = name.filter(|n| !n.is_empty());
    Ok(Args {
        gallery: gallery.with_context(|| format!("--gallery is required\n{USAGE}"))?,
        name: name.with_context(|| format!("--name is required\n{USAGE}"))?,
        role,
        camera,
        seconds,
    })
}

/// What the PIN screen asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PinStep {
    /// First owner, no PIN yet: choose one (typed twice).
    Create,
    /// A PIN exists: the owner must type it before anyone is enrolled (FR-ID-006).
    Verify,
}

/// Decides the PIN step, or refuses before any camera is opened.
fn pin_step(g: &Gallery, role: Role) -> Result<PinStep> {
    let has_owner = g.persons()?.iter().any(|p| p.role == Role::Owner);
    match (role, has_owner, g.has_pin()?) {
        (Role::Owner, true, _) => bail!(
            "an owner is already enrolled in this gallery; use --role authorised to add someone else"
        ),
        (_, _, true) => Ok(PinStep::Verify),
        (Role::Owner, false, false) => Ok(PinStep::Create),
        (Role::Authorised, _, false) => {
            bail!("no owner PIN is set: enrol the owner first (--role owner)")
        }
    }
}

/// Same rule as `Gallery::set_pin` (4–12 ASCII digits), checked before the camera starts.
fn valid_pin(pin: &str) -> bool {
    (4..=12).contains(&pin.len()) && pin.bytes().all(|b| b.is_ascii_digit())
}

/// Prompt text and screen arrow (unit vector, y down) for `p` in the mirrored preview.
/// `enrol.rs` poses are image-relative (+yaw = nose towards image right), so `Pose::Left` is the
/// nose moving to image-left: screen-right once mirrored, i.e. the user's own right.
fn prompt(p: Pose) -> (&'static str, Option<Vec2>) {
    match p {
        Pose::Front => ("Look straight at the camera", None),
        Pose::Left => ("Turn slightly right", Some(Vec2::new(1.0, 0.0))),
        Pose::Right => ("Turn slightly left", Some(Vec2::new(-1.0, 0.0))),
        Pose::Up => ("Tilt your head slightly up", Some(Vec2::new(0.0, -1.0))),
        Pose::Down => ("Tilt your head slightly down", Some(Vec2::new(0.0, 1.0))),
    }
}

fn reject_message(r: RejectReason) -> &'static str {
    match r {
        RejectReason::NoFace => "No face found: look at the camera",
        RejectReason::MultipleFaces => "Only one face please",
        RejectReason::TooSmall => "Come a little closer to the camera",
        RejectReason::Blurry => "Hold still: image is blurry",
        RejectReason::NotLive => "Checking it is a live face: hold still in good light",
        RejectReason::SpoofDetected => "Looks like a photo or a screen: a real face is needed",
        RejectReason::WrongPose { want } => prompt(want).0,
        RejectReason::NoEmbeddingYet => "Hold that pose…",
        RejectReason::CameraUnhealthy => {
            "Camera problem: check the camera is connected and uncovered"
        }
        RejectReason::FrameMismatch => "Processing: hold still",
        RejectReason::DifferentPerson => "Only the person who started enrolling, please",
    }
}

/// Writes the person, every template (SFace model id) and the new owner PIN in one
/// transaction (`Gallery::enrol`): all or nothing.
fn commit(
    g: &mut Gallery,
    name: &str,
    role: Role,
    new_pin: Option<&str>,
    samples: &[EnrolledSample],
) -> Result<PersonId, IdentityError> {
    let templates: Vec<NewTemplate<'_>> = samples
        .iter()
        .map(|s| {
            (
                RECOGNISER_MODEL_ID,
                s.pose,
                s.quality,
                s.embedding.as_slice(),
            )
        })
        .collect();
    g.enrol(name, role, new_pin, &templates)
}

fn main() -> Result<()> {
    let args = parse_args(&std::env::args().skip(1).collect::<Vec<_>>())?;
    if let Some(dir) = args.gallery.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let gallery = Gallery::open(&args.gallery, Arc::new(Dpapi::new()))
        .with_context(|| format!("opening gallery {}", args.gallery.display()))?;
    let step = pin_step(&gallery, args.role)?;
    let camera = match args.camera.clone() {
        Some(id) => id,
        None => {
            fasec_camera::list()
                .context("listing cameras")?
                .into_iter()
                .next()
                .context("no camera found")?
                .id
        }
    };
    let outcome = Arc::new(Mutex::new(None::<Result<String, String>>));
    let app_outcome = Arc::clone(&outcome);
    eframe::run_native(
        "fasec enrol",
        eframe::NativeOptions::default(),
        Box::new(move |cc| {
            let app = App::new(
                cc.egui_ctx.clone(),
                args,
                gallery,
                step,
                camera,
                app_outcome,
            );
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow!("eframe: {e}"))?;
    let outcome = outcome
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take();
    match outcome {
        Some(Ok(msg)) => {
            println!("{msg}");
            Ok(())
        }
        Some(Err(msg)) => bail!(msg),
        None => bail!("window closed before enrolment finished: nothing was written"),
    }
}

enum Cmd {
    Start(String),
    Quit,
}

/// What the worker publishes; the UI clones it once per paint. Holds no embedding.
#[derive(Clone, Default)]
struct Shared {
    /// Bumped on every publish so the UI re-uploads the texture only for a new frame.
    version: u64,
    status: String,
    frame: Option<Frame>,
    tracks: Vec<TrackObs>,
    target: Option<Pose>,
    progress: (usize, usize),
    last_reject: Option<RejectReason>,
    /// Models loaded; the worker waits for `Cmd::Start`.
    ready: bool,
    /// Set once every sample is collected and the camera is stopped.
    complete: bool,
}

enum Stage {
    Pin {
        step: PinStep,
        fields: [Zeroizing<String>; 2],
        wrong: u32,
        error: String,
        /// Focus the first field on the next paint.
        autofocus: bool,
    },
    Capture,
    Finished {
        msg: String,
        ok: bool,
    },
}

enum PinOutcome {
    /// Go on to the camera, keeping the new owner PIN if one was chosen.
    Accept(Option<Zeroizing<String>>),
    Retry(String),
    Fail(String),
}

struct App {
    name: String,
    role: Role,
    camera: String,
    gallery: Gallery,
    stage: Stage,
    /// The new owner PIN, kept (zeroised on drop) until it is written with the person.
    new_pin: Option<Zeroizing<String>>,
    shared: Arc<Mutex<Shared>>,
    tx: Sender<Cmd>,
    worker: Option<JoinHandle<Option<Vec<EnrolledSample>>>>,
    texture: Option<TextureHandle>,
    texture_version: u64,
    close_at: Option<Instant>,
    outcome: Arc<Mutex<Option<Result<String, String>>>>,
}

impl App {
    fn new(
        ctx: egui::Context,
        args: Args,
        gallery: Gallery,
        step: PinStep,
        camera: String,
        outcome: Arc<Mutex<Option<Result<String, String>>>>,
    ) -> App {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (tx, rx) = mpsc::channel();
        let w_shared = Arc::clone(&shared);
        // Models load while the PIN is typed; the camera waits for `Cmd::Start`.
        let worker = std::thread::Builder::new()
            .name("enroll-worker".into())
            .spawn(move || worker(rx, w_shared, ctx));
        let mut app = App {
            name: args.name,
            role: args.role,
            camera,
            gallery,
            stage: Stage::Pin {
                step,
                // Room for any valid PIN, so typing never reallocates (and strands a copy).
                fields: std::array::from_fn(|_| Zeroizing::new(String::with_capacity(32))),
                wrong: 0,
                error: String::new(),
                autofocus: true,
            },
            new_pin: None,
            shared,
            tx,
            worker: None,
            texture: None,
            texture_version: 0,
            close_at: args
                .seconds
                .map(|s| Instant::now() + Duration::from_secs(s)),
            outcome,
        };
        match worker {
            Ok(w) => app.worker = Some(w),
            Err(e) => app.finish(false, format!("spawn worker: {e}")),
        }
        app
    }

    fn finish(&mut self, ok: bool, msg: String) {
        *self.outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(if ok {
            Ok(msg.clone())
        } else {
            Err(msg.clone())
        });
        self.new_pin = None;
        self.stage = Stage::Finished { msg, ok };
        let at = Instant::now() + LINGER;
        self.close_at = Some(self.close_at.map_or(at, |t| t.min(at)));
    }

    /// `ready`: models are loaded, so a correct PIN leads straight to the camera.
    fn pin_screen(&mut self, ui: &mut egui::Ui, ready: bool) {
        let Stage::Pin {
            step,
            fields,
            wrong,
            error,
            autofocus,
        } = &mut self.stage
        else {
            return;
        };
        use PinOutcome::*;
        let step = *step;
        ui.heading(format!("Enrol {} ({:?})", self.name, self.role));
        ui.add_space(8.0);
        let labels: &[&str] = match step {
            PinStep::Create => &["Choose an owner PIN (4–12 digits)", "Type it again"],
            PinStep::Verify => &["Owner PIN"],
        };
        let mut enter = false;
        for (i, label) in labels.iter().enumerate() {
            ui.label(*label);
            let edit = egui::TextEdit::singleline(&mut *fields[i])
                .id(Id::new(PIN_IDS[i]))
                .password(true)
                .char_limit(12);
            let r = ui.add(edit);
            // Once per (re)shown form, so focus is never grabbed back from field 2.
            if i == 0 && std::mem::take(autofocus) {
                r.request_focus();
            }
            if r.lost_focus() && ui.input(|inp| inp.key_pressed(egui::Key::Enter)) {
                // Enter moves to the next field; on the last one it submits.
                if i + 1 < labels.len() {
                    ui.memory_mut(|m| m.request_focus(Id::new(PIN_IDS[i + 1])));
                } else {
                    enter = true;
                }
            }
        }
        if !error.is_empty() {
            ui.colored_label(Color32::RED, error.as_str());
        }
        if !ready {
            ui.label("Loading models…");
        }
        let clicked = ui
            .add_enabled(ready, egui::Button::new("Continue"))
            .clicked();
        if !(ready && (clicked || enter)) {
            return;
        }
        let [first, second] = fields;
        let outcome = match step {
            PinStep::Create if !valid_pin(first) => Retry("The PIN must be 4 to 12 digits".into()),
            PinStep::Create if **first != **second => Retry("The two PINs do not match".into()),
            PinStep::Create => Accept(Some(std::mem::take(first))),
            PinStep::Verify => match self.gallery.verify_pin(first) {
                Ok(true) => Accept(None),
                Ok(false) => {
                    *wrong += 1;
                    match MAX_PIN_ATTEMPTS.saturating_sub(*wrong) {
                        0 => Fail(format!(
                            "Wrong PIN {MAX_PIN_ATTEMPTS} times: nothing was written"
                        )),
                        left => Retry(format!("Wrong PIN ({left} attempt(s) left)")),
                    }
                }
                Err(e) => Fail(format!("PIN check failed: {e}")),
            },
        };
        first.zeroize();
        second.zeroize();
        // The text edits keep undo copies of what was typed; drop them.
        for id in PIN_IDS {
            ui.ctx()
                .data_mut(|d| d.remove::<TextEditState>(Id::new(id)));
        }
        match outcome {
            Accept(new_pin) => {
                self.new_pin = new_pin;
                let _ = self.tx.send(Cmd::Start(self.camera.clone()));
                self.stage = Stage::Capture;
            }
            Retry(msg) => (*error, *autofocus) = (msg, true),
            Fail(msg) => self.finish(false, msg),
        }
    }

    fn info_panel(&self, ui: &mut egui::Ui, s: &Shared) {
        let (have, need) = s.progress;
        match s.target {
            Some(p) => ui.heading(RichText::new(prompt(p).0).size(26.0).strong()),
            None if s.complete => ui.heading("All samples collected"),
            None => ui.heading("…"),
        };
        if s.target.and_then(|p| prompt(p).1).is_some() {
            ui.label("Follow the arrow on the picture.");
        }
        let frac = if need == 0 {
            0.0
        } else {
            have as f32 / need as f32
        };
        ui.add(egui::ProgressBar::new(frac).text(format!("{have}/{need}")));
        match s.last_reject {
            Some(r) => ui.colored_label(Color32::from_rgb(230, 140, 0), reject_message(r)),
            None => ui.label(" "),
        };
        if let [t] = s.tracks.as_slice()
            && let Some(HeadPose { yaw, pitch, .. }) = t.head_pose
        {
            ui.small(format!("head: yaw {yaw:.0}°  pitch {pitch:.0}°"));
        }
        ui.small(&s.status);
    }

    fn video(&mut self, ui: &mut egui::Ui, s: &Shared) {
        let Some(frame) = &s.frame else {
            self.texture = None;
            ui.centered_and_justified(|ui| {
                ui.heading(if s.status.is_empty() { "-" } else { &s.status })
            });
            return;
        };
        if self.texture.is_none() || self.texture_version != s.version {
            let img =
                ColorImage::from_rgb([frame.width as usize, frame.height as usize], &frame.data);
            match &mut self.texture {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => {
                    self.texture =
                        Some(ui.ctx().load_texture("camera", img, TextureOptions::LINEAR));
                }
            }
            self.texture_version = s.version;
        }
        let Some(tex) = &self.texture else { return };
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        let avail = ui.available_size();
        let scale = (avail.x / fw).min(avail.y / fh).max(0.01);
        let (resp, painter) = ui.allocate_painter(Vec2::new(fw, fh) * scale, Sense::hover());
        let r = resp.rect;
        // Mirrored for display only (a selfie view); frame data and boxes stay image-relative.
        let uv = Rect::from_min_max(Pos2::new(1.0, 0.0), Pos2::new(0.0, 1.0));
        painter.image(tex.id(), r, uv, Color32::WHITE);
        let at = |x: f32, y: f32| r.min + Vec2::new((1.0 - x) * r.width(), y * r.height());
        let color = if s.tracks.len() == 1 {
            Color32::GREEN
        } else {
            Color32::RED
        };
        for t in &s.tracks {
            let b = t.bbox;
            let rect = Rect::from_two_pos(at(b.x, b.y), at(b.x + b.w, b.y + b.h));
            painter.rect_stroke(rect, 0.0, Stroke::new(2.0, color), StrokeKind::Outside);
        }
        // From the face (or the picture's centre) towards where the nose should go.
        if let Some(dir) = s.target.and_then(|p| prompt(p).1) {
            let centre = match s.tracks.as_slice() {
                [t] => at(t.bbox.x + t.bbox.w / 2.0, t.bbox.y + t.bbox.h / 2.0),
                _ => r.center(),
            };
            let len = r.width().min(r.height()) * 0.25;
            painter.arrow(centre, dir * len, Stroke::new(6.0, Color32::YELLOW));
        }
    }

    /// Once the worker has returned, writes its samples or reports why there are none.
    fn poll_worker(&mut self) {
        let Some(w) = self.worker.take_if(|w| w.is_finished()) else {
            return;
        };
        let samples = match w.join() {
            Ok(Some(samples)) => samples,
            // The worker gave up (models, camera): its last status says why.
            Ok(None) => {
                let status = lock(&self.shared).status.clone();
                return self.finish(false, format!("{status}: nothing was written"));
            }
            Err(_) => return self.finish(false, "Worker crashed: nothing was written".into()),
        };
        let pin = self.new_pin.as_ref().map(|p| p.as_str());
        let (n, name) = (samples.len(), self.name.clone());
        match commit(&mut self.gallery, &name, self.role, pin, &samples) {
            Ok(_) => self.finish(true, format!("Enrolled {name}: {n} templates saved")),
            Err(e) => self.finish(false, format!("Saving failed: {e:#}")),
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.close_at.is_some_and(|t| Instant::now() >= t) {
            ui.ctx().send_viewport_cmd(ViewportCommand::Close);
        }
        // Keeps status and the close timer ticking while no frames arrive.
        ui.ctx().request_repaint_after(Duration::from_millis(200));
        let s = lock(&self.shared).clone();
        // In the PIN stage a finished worker can only mean a model-load failure: report it now.
        if !matches!(self.stage, Stage::Finished { .. }) {
            self.poll_worker();
        }
        match &self.stage {
            Stage::Pin { .. } => {
                egui::CentralPanel::default_margins().show(ui, |ui| {
                    self.pin_screen(ui, s.ready);
                    ui.add_space(16.0);
                    ui.small(&s.status);
                });
            }
            Stage::Capture => {
                egui::Panel::top("info").show(ui, |ui| self.info_panel(ui, &s));
                egui::CentralPanel::default_margins().show(ui, |ui| self.video(ui, &s));
            }
            Stage::Finished { msg, ok } => {
                let color = if *ok { Color32::GREEN } else { Color32::RED };
                let msg = RichText::new(msg.as_str()).size(22.0).color(color);
                egui::CentralPanel::default_margins().show(ui, |ui| {
                    ui.centered_and_justified(|ui| ui.label(msg));
                });
            }
        }
    }

    fn on_exit(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        let Some(w) = self.worker.take() else { return };
        let deadline = Instant::now() + QUIT_TIMEOUT;
        while !w.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if w.is_finished() {
            let _ = w.join();
        } else {
            eprintln!(
                "worker still busy (camera starting?): detached, process exit releases the camera"
            );
        }
    }
}

// A panicking holder cannot leave `Shared` half-written in a harmful way (plain field stores).
fn lock(m: &Mutex<Shared>) -> MutexGuard<'_, Shared> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn publish(shared: &Mutex<Shared>, ctx: &egui::Context, f: impl FnOnce(&mut Shared)) {
    let mut s = lock(shared);
    f(&mut s);
    s.version += 1;
    drop(s);
    ctx.request_repaint();
}

fn load_pipeline() -> Result<(Runtime, Pipeline)> {
    let rt = Runtime::new(ExecutionProvider::Auto).context("onnx runtime")?;
    let p = Pipeline::from_models(&rt, Path::new(MODELS_DIR), PipelineConfig::default())
        .with_context(|| format!("loading models from {MODELS_DIR}"))?;
    Ok((rt, p))
}

/// Owns pipeline, camera and enrolment. Returns the samples once all are accepted (camera
/// already stopped); `None` on `Quit`, a vanished UI, or a model/camera error (status says why).
fn worker(
    rx: Receiver<Cmd>,
    shared: Arc<Mutex<Shared>>,
    ctx: egui::Context,
) -> Option<Vec<EnrolledSample>> {
    publish(&shared, &ctx, |s| s.status = "loading models…".into());
    // `_rt` is kept alive for as long as the pipeline's sessions.
    let (_rt, mut pipeline) = match load_pipeline() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e:#}");
            publish(&shared, &ctx, |s| s.status = format!("{e:#}\n{FETCH_HINT}"));
            return None;
        }
    };
    let mut enrol = Enrolment::new(EnrolConfig::default());
    publish(&shared, &ctx, |s| {
        (s.status, s.ready) = ("models loaded".into(), true);
        (s.target, s.progress) = (enrol.current_target(), enrol.progress());
    });
    let id = match rx.recv() {
        Ok(Cmd::Start(id)) => id,
        Ok(Cmd::Quit) | Err(_) => return None,
    };
    let slot = Arc::new(LatestFrame::new());
    let cap = match Capture::start(&id, &CameraConfig::default(), Arc::clone(&slot)) {
        Ok(c) => c,
        Err(e) => {
            publish(&shared, &ctx, |s| s.status = format!("camera error: {e}"));
            return None;
        }
    };
    publish(&shared, &ctx, |s| s.status = "camera running".into());
    let mut seq = 0;
    loop {
        match rx.try_recv() {
            Ok(Cmd::Quit) | Err(TryRecvError::Disconnected) => {
                cap.stop();
                return None;
            }
            Ok(Cmd::Start(_)) | Err(TryRecvError::Empty) => {}
        }
        let Some(frame) = slot.take_newer(seq, Duration::from_millis(100)) else {
            let h = cap.health();
            if h != CameraHealth::Ok {
                publish(&shared, &ctx, |s| {
                    s.tracks.clear();
                    s.last_reject = Some(RejectReason::CameraUnhealthy);
                });
            }
            continue;
        };
        seq = frame.seq;
        let out = match pipeline.process(&frame, cap.health()) {
            Ok(o) => o,
            Err(e) => {
                publish(&shared, &ctx, |s| s.status = format!("vision error: {e}"));
                continue;
            }
        };
        // Only the single face's own embedding; `offer` re-checks track and timestamp.
        let emb = out
            .report
            .tracks
            .first()
            .and_then(|t| out.embeddings.iter().find(|e| e.track == t.id));
        let verdict = enrol.offer(&frame, &out.report, emb);
        let done = enrol.current_target().is_none();
        let (target, progress) = (enrol.current_target(), enrol.progress());
        let update = move |s: &mut Shared| {
            match verdict {
                SampleVerdict::Rejected(r) => s.last_reject = Some(r),
                SampleVerdict::Accepted { .. } | SampleVerdict::Complete => s.last_reject = None,
            }
            (s.target, s.progress) = (target, progress);
            s.tracks = out.report.tracks;
            s.frame = Some(frame);
            s.status = if done {
                "camera stopped"
            } else {
                "camera running"
            }
            .into();
            s.complete = done;
        };
        if done {
            cap.stop();
            publish(&shared, &ctx, update);
            // Embeddings computed for this frame (`out.embeddings`) were zeroised on drop.
            return enrol.finish().ok();
        }
        publish(&shared, &ctx, update);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fasec_core::Embedding;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn parses_full_and_default_args() {
        let a = parse_args(&args(
            "--gallery g.db --name Ann --role authorised --camera cam1 --seconds 5",
        ))
        .unwrap();
        assert_eq!(
            a,
            Args {
                gallery: "g.db".into(),
                name: "Ann".into(),
                role: Role::Authorised,
                camera: Some("cam1".into()),
                seconds: Some(5),
            }
        );
        let a = parse_args(&args("--name Bob --gallery x/g.db")).unwrap();
        assert_eq!((a.role, a.camera, a.seconds), (Role::Owner, None, None));
    }

    #[test]
    fn rejects_bad_args() {
        for bad in [
            "",
            "--gallery g.db",
            "--name Ann",
            "--gallery g.db --name",
            "--gallery g.db --name Ann --role admin",
            "--gallery g.db --name Ann --seconds x",
            "--gallery g.db --name Ann --pin 1234",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
        let blank = ["--gallery", "g.db", "--name", "  "].map(String::from);
        assert!(parse_args(&blank).is_err());
    }

    #[test]
    fn pin_rule_matches_store() {
        for ok in ["1234", "000000000000"] {
            assert!(valid_pin(ok), "{ok}");
        }
        for bad in ["", "123", "1234567890123", "12a4", "１２３４", " 1234"] {
            assert!(!valid_pin(bad), "{bad:?}");
        }
    }

    #[test]
    fn reject_reasons_read_as_plain_words() {
        use RejectReason::*;
        assert_eq!(reject_message(Blurry), "Hold still: image is blurry");
        assert_eq!(reject_message(MultipleFaces), "Only one face please");
        assert_eq!(
            reject_message(WrongPose { want: Pose::Left }),
            "Turn slightly right"
        );
        assert_eq!(
            reject_message(WrongPose { want: Pose::Right }),
            "Turn slightly left"
        );
        let all = [
            NoFace,
            MultipleFaces,
            TooSmall,
            Blurry,
            NotLive,
            SpoofDetected,
            WrongPose { want: Pose::Up },
            NoEmbeddingYet,
            CameraUnhealthy,
            FrameMismatch,
            DifferentPerson,
        ];
        for r in all {
            let m = reject_message(r);
            assert!(!m.is_empty() && !m.contains('{'), "{r:?}");
        }
    }

    #[test]
    fn prompts_are_from_the_users_view_of_the_mirrored_preview() {
        // Pose::Left = negative yaw = nose to image-left = screen-right when mirrored.
        let (text, arrow) = prompt(Pose::Left);
        assert_eq!(text, "Turn slightly right");
        assert!(arrow.unwrap().x > 0.0);
        let (text, arrow) = prompt(Pose::Right);
        assert_eq!(text, "Turn slightly left");
        assert!(arrow.unwrap().x < 0.0);
        // Vertical is not mirrored: +pitch = looking down.
        assert!(prompt(Pose::Up).1.unwrap().y < 0.0);
        assert!(prompt(Pose::Down).1.unwrap().y > 0.0);
        assert_eq!(prompt(Pose::Front).1, None);
    }

    fn samples(dim: usize) -> Vec<EnrolledSample> {
        let poses = [Pose::Front, Pose::Left, Pose::Right, Pose::Up, Pose::Down];
        poses
            .iter()
            .flat_map(|&pose| [pose, pose])
            .enumerate()
            .map(|(i, pose)| EnrolledSample {
                pose,
                quality: 0.9,
                embedding: Embedding::from(vec![0.01 * (i + 1) as f32; dim]),
            })
            .collect()
    }

    // Real DPAPI on an in-memory gallery: nothing touches disk.
    #[cfg(windows)]
    fn gallery() -> Gallery {
        Gallery::open_in_memory(Arc::new(Dpapi::new())).unwrap()
    }

    #[cfg(windows)]
    #[test]
    fn pin_step_follows_roles() {
        let mut g = gallery();
        assert_eq!(pin_step(&g, Role::Owner).unwrap(), PinStep::Create);
        assert!(pin_step(&g, Role::Authorised).is_err(), "no PIN yet");
        commit(&mut g, "Owner", Role::Owner, Some("2468"), &samples(128)).unwrap();
        assert!(pin_step(&g, Role::Owner).is_err(), "owner exists");
        assert_eq!(pin_step(&g, Role::Authorised).unwrap(), PinStep::Verify);
    }

    #[cfg(windows)]
    #[test]
    fn commit_writes_everything_or_nothing() {
        let mut g = gallery();
        let id = commit(&mut g, "Ann", Role::Owner, Some("2468"), &samples(128)).unwrap();
        let t = g.templates(&id).unwrap();
        assert_eq!(t.len(), 10);
        assert!(t.iter().all(|t| t.model_id == RECOGNISER_MODEL_ID));
        assert_eq!(
            g.templates_for_model(RECOGNISER_MODEL_ID).unwrap().len(),
            10
        );
        assert!(g.verify_pin("2468").unwrap());

        // A bad template part-way: no person, no templates, no PIN left behind.
        let mut g = gallery();
        let mut bad = samples(128);
        bad[7] = samples(3).remove(0);
        assert!(commit(&mut g, "Ann", Role::Owner, Some("2468"), &bad).is_err());
        assert!(g.persons().unwrap().is_empty());
        assert!(
            g.templates_for_model(RECOGNISER_MODEL_ID)
                .unwrap()
                .is_empty()
        );
        assert!(!g.has_pin().unwrap());
    }
}
