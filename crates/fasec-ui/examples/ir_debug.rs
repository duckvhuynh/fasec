//! IR liveness viewer and local recorder: the RGB camera, its paired IR camera's lit frame and
//! lit − ambient side by side, with every IR cue number and verdict per track, live.
//!
//! `cargo run -p fasec-ui --example ir_debug [-- --seconds N] [--record <label>]`
//!
//! One worker thread owns both cameras, the pipeline and an `IrProbe` (its own YuNet and IR↔RGB
//! associator, fed the same lit pairs and report tracks as the pipeline, so its per-frame
//! verdicts are the pipeline's). It publishes the latest frames, report and cues; the UI only
//! draws. Once a second it prints each track's cue numbers (numbers only, never pixels).
//!
//! Recording is off until the Record toggle is switched on (or `--record <label>`, which
//! switches it on at start). Each lit pair and the RGB frame processed with it then go to a
//! writer thread, which writes them only under
//! `%LOCALAPPDATA%\fasec\recordings\<label>-<yyyymmdd-hhmmss>\` (UTC; layout in
//! `fasec_replay::recording`), at most 600 steps a session. Steps the writer cannot keep up with
//! are dropped and counted. Replay one with `fasec-replay run --ir-input <session dir>`.

#[path = "support/ir_link.rs"]
mod ir_link;

use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result, bail, ensure};
use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Painter, Pos2, Rect, RichText, Sense, Stroke,
    StrokeKind, TextureHandle, TextureOptions, Vec2, ViewportCommand,
};
use fasec_camera::{Capture, LatestFrame};
use fasec_config::{CameraConfig, ExecutionProvider, IrMode};
use fasec_core::{CameraHealth, Frame, IrFrame, LivenessState, RectF, VisionReport};
use fasec_replay::probe::{Ambient, IrProbe, ProbeFrame};
use fasec_replay::recording::{MAX_STEPS, PRESETS, Recorder, recordings_root, sanitize_label};
use fasec_vision::ir::{IrCues, IrLivenessConfig, IrVerdict};
use fasec_vision::{Pipeline, PipelineConfig, Runtime, TrackIr};
use ir_link::{IrLink, IrView, tier_line, track_ir_label};

const USAGE: &str = "usage: ir_debug [--seconds N] [--record <label>]";
const MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
const FETCH_HINT: &str =
    "Models missing or invalid: run `cargo run -p fasec-models --example fetch`";
/// As in `live_debug`: `Capture::stop` is bounded at ~2 s.
const QUIT_TIMEOUT: Duration = Duration::from_secs(3);
/// Steps waiting for the writer; a step beyond this is dropped and counted.
const REC_QUEUE: usize = 8;

#[derive(Debug, Default, PartialEq)]
struct Args {
    seconds: Option<u64>,
    record: Option<String>,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut a = Args::default();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .with_context(|| format!("{flag} needs a value\n{USAGE}"))?;
        match flag.as_str() {
            "--seconds" => a.seconds = Some(value.parse().context(USAGE)?),
            "--record" => {
                ensure!(
                    sanitize_label(value).is_some(),
                    "--record needs a label with a letter or digit"
                );
                a.record = Some(value.clone());
            }
            _ => bail!("unknown argument {flag}\n{USAGE}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args(&std::env::args().skip(1).collect::<Vec<_>>())?;
    let ir_mode = ir_link::ir_mode_from_config().map_err(anyhow::Error::msg)?;
    let cams = fasec_camera::list().map_err(|e| anyhow::anyhow!("camera list: {e}"))?;
    let cam = cams.into_iter().next().context("no camera found")?;
    println!("RGB camera: {}", cam.name);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 860.0]),
        ..Default::default()
    };
    eframe::run_native(
        "fasec IR debug",
        options,
        Box::new(move |cc| {
            Ok(Box::new(App::new(
                cc.egui_ctx.clone(),
                cam.id,
                args,
                ir_mode,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

/// One step for the recorder: the RGB frame and the lit pair it was processed with.
type RecStep = (Frame, IrFrame, Option<IrFrame>);

/// A recording session as the UI shows it.
#[derive(Default)]
struct RecStatus {
    dir: PathBuf,
    written: u64,
    dropped: u64,
    /// Why the writer stopped by itself (step limit or a write error).
    ended: Option<String>,
}

/// The worker's end of a session; dropping it ends the session once queued steps are written.
struct RecLink {
    tx: SyncSender<RecStep>,
    status: Arc<Mutex<RecStatus>>,
}

enum Cmd {
    /// Start (`Some`) or stop (`None`) handing steps to a recorder.
    Record(Option<RecLink>),
    Quit,
}

/// What the worker publishes; the UI clones it once per paint.
#[derive(Clone, Default)]
struct Shared {
    status: String,
    /// Bumped with each new RGB frame / new lit pair, so textures re-upload only then.
    rgb_version: u64,
    ir_version: u64,
    rgb: Option<Frame>,
    health: Option<CameraHealth>,
    report: Option<VisionReport>,
    /// The pipeline's fused IR evidence per track.
    track_ir: Vec<TrackIr>,
    /// The latest lit pair and the probe's measurements on it.
    lit: Option<IrFrame>,
    ambient: Option<IrFrame>,
    probe: Option<Result<ProbeFrame, String>>,
    ir: IrView,
}

/// A texture re-uploaded only when its source version changes.
#[derive(Default)]
struct Tex {
    handle: Option<TextureHandle>,
    version: u64,
}

impl Tex {
    fn get(
        &mut self,
        ctx: &egui::Context,
        name: &str,
        version: u64,
        make: impl FnOnce() -> ColorImage,
    ) -> &TextureHandle {
        if self.handle.is_none() || self.version != version {
            let img = make();
            match &mut self.handle {
                Some(t) => t.set(img, TextureOptions::LINEAR),
                None => self.handle = Some(ctx.load_texture(name, img, TextureOptions::LINEAR)),
            }
            self.version = version;
        }
        self.handle.as_ref().expect("uploaded above")
    }
}

struct App {
    shared: Arc<Mutex<Shared>>,
    tx: Sender<Cmd>,
    worker: Option<JoinHandle<()>>,
    close_at: Option<Instant>,
    textures: [Tex; 3],
    /// Index into `PRESETS`; `PRESETS.len()` means the free-text `custom` label.
    preset: usize,
    custom: String,
    recording: bool,
    /// The current or last session, and its writer thread.
    session: Option<Arc<Mutex<RecStatus>>>,
    writer: Option<JoinHandle<()>>,
    rec_error: Option<String>,
}

impl App {
    fn new(ctx: egui::Context, cam_id: String, args: Args, ir_mode: IrMode) -> App {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (tx, rx) = mpsc::channel();
        let w_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("ir-debug-worker".into())
            .spawn(move || worker(rx, w_shared, ctx, cam_id, ir_mode));
        let worker = match worker {
            Ok(w) => Some(w),
            Err(e) => {
                lock(&shared).status = format!("spawn worker: {e}");
                None
            }
        };
        let mut app = App {
            shared,
            tx,
            worker,
            close_at: args
                .seconds
                .map(|s| Instant::now() + Duration::from_secs(s)),
            textures: Default::default(),
            preset: 0,
            custom: String::new(),
            recording: false,
            session: None,
            writer: None,
            rec_error: None,
        };
        if let Some(label) = args.record {
            (app.preset, app.custom) = (PRESETS.len(), label);
            app.start_recording();
        }
        app
    }

    fn label(&self) -> &str {
        PRESETS.get(self.preset).copied().unwrap_or(&self.custom)
    }

    fn start_recording(&mut self) {
        self.rec_error = None;
        let Some(root) = recordings_root() else {
            self.rec_error = Some("LOCALAPPDATA is not set: recording unavailable".into());
            return;
        };
        let rec = match Recorder::create(&root, self.label(), SystemTime::now()) {
            Ok(r) => r,
            Err(e) => {
                self.rec_error = Some(format!("{e:#}"));
                return;
            }
        };
        let status = Arc::new(Mutex::new(RecStatus {
            dir: rec.dir().to_owned(),
            ..RecStatus::default()
        }));
        let (tx, rx) = mpsc::sync_channel(REC_QUEUE);
        let w_status = Arc::clone(&status);
        let spawned = std::thread::Builder::new()
            .name("ir-recorder".into())
            .spawn(move || write_steps(rec, rx, &w_status));
        match spawned {
            Ok(w) => self.writer = Some(w),
            Err(e) => {
                self.rec_error = Some(format!("spawn recorder: {e}"));
                return;
            }
        }
        println!("recording to {}", lock(&status).dir.display());
        let link = RecLink {
            tx,
            status: Arc::clone(&status),
        };
        let _ = self.tx.send(Cmd::Record(Some(link)));
        (self.session, self.recording) = (Some(status), true);
    }

    fn stop_recording(&mut self) {
        let _ = self.tx.send(Cmd::Record(None));
        self.recording = false;
        if let Some(s) = &self.session {
            let s = lock(s);
            println!(
                "recording stopped: {} steps so far in {} ({} dropped), queued steps finishing",
                s.written,
                s.dir.display(),
                s.dropped
            );
        }
    }

    fn recorder_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Recorder");
        let root = recordings_root();
        let where_ = root
            .as_deref()
            .map_or("-".into(), |p| p.display().to_string());
        ui.label(
            RichText::new(format!(
                "Recordings stay on this PC in {where_}. Delete the folder to remove them."
            ))
            .strong(),
        );
        ui.add_enabled_ui(!self.recording, |ui| {
            let name = |i: usize| PRESETS.get(i).copied().unwrap_or("custom…");
            egui::ComboBox::from_id_salt("label")
                .selected_text(name(self.preset))
                .show_ui(ui, |ui| {
                    for i in 0..=PRESETS.len() {
                        ui.selectable_value(&mut self.preset, i, name(i));
                    }
                });
            if self.preset == PRESETS.len() {
                ui.add(egui::TextEdit::singleline(&mut self.custom).hint_text("label"));
            }
        });
        let folder = sanitize_label(self.label());
        ui.small(folder.as_deref().map_or_else(
            || "the label needs a letter or digit".to_owned(),
            |l| format!("folder: {l}-<yyyymmdd-hhmmss> (UTC)"),
        ));
        let mut on = self.recording;
        let toggle = ui.add_enabled_ui(on || folder.is_some(), |ui| {
            ui.toggle_value(&mut on, RichText::new("● Record").color(Color32::RED))
        });
        if toggle.inner.changed() {
            if on {
                self.start_recording();
            } else {
                self.stop_recording();
            }
        }
        if let Some(e) = &self.rec_error {
            ui.colored_label(Color32::RED, e);
        }
        if let Some(s) = &self.session {
            let s = lock(s);
            let state = if self.recording {
                "recording"
            } else {
                "last session"
            };
            ui.label(format!(
                "{state}: {} / {MAX_STEPS} steps, {} dropped\n{}",
                s.written,
                s.dropped,
                s.dir.display()
            ));
            if let Some(why) = &s.ended {
                ui.colored_label(Color32::YELLOW, why);
            }
        }
        if let Some(root) = root.filter(|r| r.is_dir())
            && ui.button("Open folder").clicked()
        {
            let _ = std::process::Command::new("explorer").arg(root).spawn();
        }
    }

    fn ir_panel(&self, ui: &mut egui::Ui, s: &Shared) {
        ui.heading("IR");
        let color = if s.ir.on {
            Color32::GREEN
        } else {
            Color32::YELLOW
        };
        ui.colored_label(color, &s.ir.status);
        ui.label(format!(
            "source: {} | sharing: {} | lit fps {:.1}",
            s.ir.source.as_deref().unwrap_or("-"),
            if s.ir.sharing.is_empty() {
                "-"
            } else {
                s.ir.sharing
            },
            s.ir.fps
        ));
        if let Some(r) = &s.report {
            ui.label(RichText::new(tier_line(r.liveness_tier, &s.ir)).strong());
        }
        if let Some(lit) = &s.lit {
            ui.label(format!(
                "lit seq {} {}x{} | ambient: {}",
                lit.seq,
                lit.width,
                lit.height,
                match s.ambient.as_ref() {
                    None => "none".to_owned(),
                    Some(a) => format!("seq {}", a.seq),
                }
            ));
        }
        match &s.probe {
            None => {
                ui.label("no lit pair measured yet");
            }
            Some(Err(e)) => {
                ui.colored_label(Color32::RED, format!("IR cues failed: {e}"));
            }
            Some(Ok(p)) => {
                let (amb, amb_color) = match p.ambient {
                    Ambient::Usable => ("ambient usable: cues on lit − ambient", Color32::GREEN),
                    Ambient::Blank => (
                        "AMBIENT BLANK (near-black unlit frame): cues on the lit frame alone, \
                         no lit − ambient",
                        Color32::YELLOW,
                    ),
                    Ambient::Missing => (
                        "no ambient frame: cues on the lit frame alone",
                        Color32::YELLOW,
                    ),
                };
                ui.colored_label(amb_color, amb);
                ui.label(format!(
                    "IR faces: {} | mapping: {}",
                    p.faces.len(),
                    if p.trusted {
                        "trusted"
                    } else {
                        "learning (needs 5 frames with one clear face on each camera)"
                    }
                ));
            }
        }
    }

    fn tracks_panel(&self, ui: &mut egui::Ui, s: &Shared) {
        ui.heading("Tracks (latest lit frame)");
        let Some(Ok(p)) = &s.probe else { return };
        if p.tracks.is_empty() {
            ui.label("no RGB track");
        }
        let cfg = PipelineConfig::default().ir;
        for t in &p.tracks {
            let obs = s
                .report
                .as_ref()
                .and_then(|r| r.tracks.iter().find(|o| o.id == t.track));
            let fused = obs.map_or("gone".to_owned(), |o| liveness_label(&o.liveness));
            let evidence = s
                .track_ir
                .iter()
                .find(|i| i.track == t.track)
                .map_or("IR -".to_owned(), track_ir_label);
            ui.label(
                RichText::new(format!("#{}  fused: {fused}  ({evidence})", t.track.0)).strong(),
            );
            ui.label(format!(
                "this frame: {} | IR face: {}",
                verdict_label(t.verdict),
                t.matched.map_or("none matched".to_owned(), |d| format!(
                    "matched (score {:.2})",
                    d.score
                ))
            ));
            let Some(c) = &t.cues else {
                ui.small("no cues: mapping untrusted, or the box is off the IR frame");
                continue;
            };
            egui::Grid::new(("cues", t.track.0))
                .striped(true)
                .show(ui, |ui| {
                    for (name, value, rule) in cue_rows(c, &cfg) {
                        ui.monospace(name);
                        ui.monospace(value);
                        ui.small(rule);
                        ui.end_row();
                    }
                });
            ui.add_space(6.0);
        }
    }

    fn images(&mut self, ui: &mut egui::Ui, s: &Shared) {
        let ctx = ui.ctx().clone();
        let [t_rgb, t_lit, t_diff] = &mut self.textures;
        ui.columns(3, |cols| {
            cols[0].label("RGB (boxes: fused liveness)");
            if let Some(f) = &s.rgb {
                let tex = t_rgb.get(&ctx, "rgb", s.rgb_version, || {
                    ColorImage::from_rgb([f.width as usize, f.height as usize], &f.data)
                });
                let (painter, r) = show_image(&mut cols[0], tex, (f.width, f.height));
                for t in s.report.iter().flat_map(|r| &r.tracks) {
                    let color = liveness_color(&t.liveness);
                    let text = format!(
                        "#{} w {:.0}px\n{}",
                        t.id.0,
                        t.face_width_px,
                        liveness_label(&t.liveness)
                    );
                    boxed(&painter, r, t.bbox, color, &text);
                }
            } else {
                cols[0].label(&s.status);
            }

            cols[1].label("IR lit (white: IR faces, cyan: RGB box mapped)");
            cols[2].label("IR lit − ambient");
            let Some(lit) = &s.lit else {
                cols[1].label("no lit IR frame");
                return;
            };
            let size = [lit.width as usize, lit.height as usize];
            let tex = t_lit.get(&ctx, "ir-lit", s.ir_version, || {
                ColorImage::from_gray(size, &lit.data)
            });
            let (painter, r) = show_image(&mut cols[1], tex, (lit.width, lit.height));
            if let Some(Ok(p)) = &s.probe {
                for d in &p.faces {
                    let rect = to_screen(r, d.bbox);
                    painter.rect_stroke(
                        rect,
                        0.0,
                        Stroke::new(1.0, Color32::WHITE),
                        StrokeKind::Outside,
                    );
                }
                for t in &p.tracks {
                    let color = verdict_color(t.verdict);
                    let text = format!("#{} {}", t.track.0, verdict_label(t.verdict));
                    match t.matched {
                        Some(d) => boxed(&painter, r, d.bbox, color, &text),
                        None => boxed(&painter, r, t.mapped, Color32::LIGHT_BLUE, &text),
                    }
                }
            }

            match &s.probe {
                Some(Ok(p)) if p.ambient == Ambient::Usable => {
                    let Some(amb) = s
                        .ambient
                        .as_ref()
                        .filter(|a| a.data.len() == lit.data.len())
                    else {
                        return;
                    };
                    let diff: Vec<u8> = lit
                        .data
                        .iter()
                        .zip(amb.data.iter())
                        .map(|(l, a)| l.saturating_sub(*a))
                        .collect();
                    let tex = t_diff.get(&ctx, "ir-diff", s.ir_version, || {
                        ColorImage::from_gray(size, &diff)
                    });
                    show_image(&mut cols[2], tex, (lit.width, lit.height));
                }
                Some(Ok(p)) if p.ambient == Ambient::Blank => {
                    cols[2].colored_label(
                        Color32::YELLOW,
                        "AMBIENT BLANK: the unlit frame is near-black (driver blanking), \
                         so there is no lit − ambient image or cue",
                    );
                }
                _ => {
                    cols[2].colored_label(Color32::YELLOW, "no ambient frame with this lit frame");
                }
            }
        });
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.close_at.is_some_and(|t| Instant::now() >= t) {
            ui.ctx().send_viewport_cmd(ViewportCommand::Close);
        }
        ui.ctx().request_repaint_after(Duration::from_millis(200));
        // the writer stopped by itself (step limit or write error): the toggle goes off
        let ended = self
            .session
            .as_ref()
            .is_some_and(|s| lock(s).ended.is_some());
        if self.recording && ended {
            self.stop_recording();
        }
        let s = lock(&self.shared).clone();
        egui::Panel::right("info")
            .default_size(430.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.recorder_panel(ui);
                    ui.separator();
                    self.ir_panel(ui, &s);
                    ui.separator();
                    self.tracks_panel(ui, &s);
                });
            });
        egui::CentralPanel::default_margins().show(ui, |ui| self.images(ui, &s));
    }

    fn on_exit(&mut self) {
        let _ = self.tx.send(Cmd::Quit);
        let Some(w) = self.worker.take() else { return };
        if !join_within(w, QUIT_TIMEOUT) {
            eprintln!(
                "worker still busy (camera starting?): detached, process exit releases the camera"
            );
        }
        // the worker dropped its recorder link on quitting: the writer finishes the queued steps
        let Some(w) = self.writer.take() else { return };
        let done = join_within(w, QUIT_TIMEOUT);
        if let Some(s) = &self.session {
            let s = lock(s);
            let cut = if done {
                ""
            } else {
                " (writer still busy at exit)"
            };
            println!(
                "recorded {} steps in {} ({} dropped){cut}",
                s.written,
                s.dir.display(),
                s.dropped
            );
        }
    }
}

/// Joins `w` if it finishes within `timeout`; otherwise leaves it detached.
fn join_within(w: JoinHandle<()>, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while !w.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    let done = w.is_finished();
    if done {
        let _ = w.join();
    }
    done
}

/// Draws `tex` as large as fits in what is left of `ui`; returns its painter and screen rect.
fn show_image(ui: &mut egui::Ui, tex: &TextureHandle, (w, h): (u32, u32)) -> (Painter, Rect) {
    let (fw, fh) = (w as f32, h as f32);
    let avail = ui.available_size();
    let scale = (avail.x / fw).min(avail.y / fh).max(0.01);
    let (resp, painter) = ui.allocate_painter(Vec2::new(fw, fh) * scale, Sense::hover());
    let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
    painter.image(tex.id(), resp.rect, uv, Color32::WHITE);
    (painter, resp.rect)
}

fn to_screen(r: Rect, b: RectF) -> Rect {
    let at = |x: f32, y: f32| r.min + Vec2::new(x * r.width(), y * r.height());
    Rect::from_min_max(at(b.x, b.y), at(b.x + b.w, b.y + b.h))
}

/// A box with `text` above it.
fn boxed(painter: &Painter, r: Rect, b: RectF, color: Color32, text: &str) {
    let rect = to_screen(r, b);
    painter.rect_stroke(rect, 0.0, Stroke::new(2.0, color), StrokeKind::Outside);
    let galley = painter.layout_no_wrap(text.to_owned(), FontId::monospace(12.0), color);
    let at = Align2::LEFT_BOTTOM.anchor_size(rect.left_top(), galley.size());
    painter.rect_filled(at.expand(2.0), 2.0, Color32::from_black_alpha(170));
    painter.galley(at.min, galley, color);
}

fn liveness_label(l: &LivenessState) -> String {
    match l {
        LivenessState::Live { score, tier } => format!("live {score:.2} {tier:?}"),
        LivenessState::Spoof { score, tier } => format!("SPOOF {score:.2} {tier:?}"),
        LivenessState::Unknown => "liveness ?".to_owned(),
    }
}

fn liveness_color(l: &LivenessState) -> Color32 {
    match l {
        LivenessState::Live { .. } => Color32::GREEN,
        LivenessState::Spoof { .. } => Color32::RED,
        LivenessState::Unknown => Color32::YELLOW,
    }
}

fn verdict_label(v: IrVerdict) -> String {
    match v {
        IrVerdict::Live { score } => format!("IR live {score:.2}"),
        IrVerdict::Spoof { score, reason } => format!("IR SPOOF {reason:?} {score:.2}"),
        IrVerdict::Unknown => "IR unknown".to_owned(),
    }
}

fn verdict_color(v: IrVerdict) -> Color32 {
    match v {
        IrVerdict::Live { .. } => Color32::GREEN,
        IrVerdict::Spoof { .. } => Color32::RED,
        IrVerdict::Unknown => Color32::YELLOW,
    }
}

/// `n/a` for a cue that could not be measured (NaN).
fn num(v: f32) -> String {
    if v.is_finite() {
        format!("{v:.3}")
    } else {
        "n/a".to_owned()
    }
}

/// Every `IrCues` field with its value and the rule `judge` applies to it.
fn cue_rows(c: &IrCues, cfg: &IrLivenessConfig) -> [(&'static str, String, String); 8] {
    [
        (
            "face_present",
            c.face_present.to_string(),
            "false: NoIrFace".into(),
        ),
        (
            "face_vs_background",
            num(c.face_vs_background),
            format!("below {}: unknown", cfg.min_face_vs_background),
        ),
        (
            "relief",
            num(c.relief),
            format!("below {}: FlatReflectance", cfg.min_relief),
        ),
        (
            "lit_face",
            num(c.lit_face),
            format!("below {}: unknown", cfg.min_lit_face),
        ),
        (
            "lit_background",
            num(c.lit_background),
            format!("above {}: flooded, unknown", cfg.max_background),
        ),
        (
            "lit_minus_ambient",
            num(c.lit_minus_ambient),
            format!(
                "below {} × lit_face: self-lit check",
                cfg.min_emitter_fraction
            ),
        ),
        (
            "ambient_background",
            num(c.ambient_background),
            format!(
                "ambient face ≥ {} × this: SelfEmitting",
                cfg.self_emit_contrast
            ),
        ),
        (
            "glints",
            c.glints.to_string(),
            "bonus to a live score only".into(),
        ),
    ]
}

/// One stdout line per track: numbers only.
fn print_cues(p: &ProbeFrame) {
    let cfg = PipelineConfig::default().ir;
    println!(
        "  IR faces {} | mapping {} | ambient {:?}",
        p.faces.len(),
        if p.trusted { "trusted" } else { "learning" },
        p.ambient
    );
    for t in &p.tracks {
        let cues = t.cues.as_ref().map_or("no cues".to_owned(), |c| {
            cue_rows(c, &cfg)
                .map(|(n, v, _)| format!("{n} {v}"))
                .join(" ")
        });
        println!("  #{} {} | {cues}", t.track.0, verdict_label(t.verdict));
    }
}

/// Writes steps until the worker drops its sender, the step limit or a write error.
fn write_steps(mut rec: Recorder, rx: Receiver<RecStep>, status: &Mutex<RecStatus>) {
    for (rgb, lit, ambient) in rx {
        let res = rec.write(&rgb, &lit, ambient.as_ref());
        let mut s = lock(status);
        match res {
            Ok(true) => s.written = rec.steps(),
            Ok(false) => {
                s.ended = Some(format!("stopped at the {MAX_STEPS}-step limit"));
                return;
            }
            Err(e) => {
                eprintln!("recording: {e:#}");
                s.ended = Some(format!("write failed: {e:#}"));
                return;
            }
        }
    }
}

// Plain field stores only, so a panicking holder leaves nothing half-written.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn publish(shared: &Mutex<Shared>, ctx: &egui::Context, f: impl FnOnce(&mut Shared)) {
    f(&mut lock(shared));
    ctx.request_repaint();
}

/// Owns both cameras, the pipeline and the probe; exits on `Quit` or when the UI is gone.
fn worker(
    rx: Receiver<Cmd>,
    shared: Arc<Mutex<Shared>>,
    ctx: egui::Context,
    cam_id: String,
    ir_mode: IrMode,
) {
    publish(&shared, &ctx, |s| s.status = "loading models…".into());
    let load = || -> Result<(Runtime, Pipeline, IrProbe)> {
        let rt = Runtime::new(ExecutionProvider::Auto).context("onnx runtime")?;
        let cfg = PipelineConfig::default();
        let models = Path::new(MODELS_DIR);
        let p = Pipeline::from_models(&rt, models, cfg)
            .with_context(|| format!("loading models from {MODELS_DIR}"))?;
        let probe = IrProbe::from_models(&rt, models, cfg.ir)?;
        Ok((rt, p, probe))
    };
    // `_rt` is kept alive for as long as the sessions.
    let (_rt, mut pipeline, mut probe) = match load() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e:#}");
            publish(&shared, &ctx, |s| {
                s.status = format!("{e:#}\n\n{FETCH_HINT}")
            });
            return;
        }
    };
    let slot = Arc::new(LatestFrame::new());
    let cap = match Capture::start(&cam_id, &CameraConfig::default(), Arc::clone(&slot)) {
        Ok(c) => c,
        Err(e) => {
            publish(&shared, &ctx, |s| s.status = format!("camera error: {e}"));
            return;
        }
    };
    publish(&shared, &ctx, |s| s.status = "starting IR…".into());
    let mut ir = IrLink::new(ir_mode);
    ir.start(&cam_id);
    println!("{}", ir.view().status);
    publish(&shared, &ctx, |s| {
        (s.status, s.ir) = ("running".into(), ir.view())
    });

    let mut rec: Option<RecLink> = None;
    let mut seq = 0;
    let mut last_print = Instant::now();
    let mut latest: Option<Result<ProbeFrame, String>> = None;
    loop {
        loop {
            match rx.try_recv() {
                Ok(Cmd::Record(r)) => rec = r,
                Ok(Cmd::Quit) | Err(TryRecvError::Disconnected) => {
                    let t = Instant::now();
                    cap.stop();
                    ir.stop();
                    println!("cameras stopped in {} ms", t.elapsed().as_millis());
                    return;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        // IR first, so the RGB frame taken next is the one closest to the lit frame.
        let (pair, ir_health) = ir.next();
        let Some(frame) = slot.take_newer(seq, Duration::from_millis(100)) else {
            let h = cap.health();
            if h != CameraHealth::Ok {
                publish(&shared, &ctx, |s| (s.health, s.report) = (Some(h), None));
            }
            continue;
        };
        ir.delivered();
        seq = frame.seq;
        let health = cap.health();
        let ir_ok = ir_health == CameraHealth::Ok;
        let ir_frames = pair.as_ref().map(|(lit, amb)| (lit, amb.as_ref()));
        let result = pipeline.process_with_ir(&frame, ir_frames, health.clone(), ir_health);
        // the probe sees exactly what the pipeline judged IR on
        if !ir_ok {
            probe.reset();
            latest = None;
        }
        if let (Ok(out), Some((lit, amb)), CameraHealth::Ok) = (&result, &pair, &health) {
            match probe.measure(lit, amb.as_ref(), &out.report.tracks) {
                Ok(Some(p)) => latest = Some(Ok(p)),
                Ok(None) => {}
                Err(e) => latest = Some(Err(e.to_string())),
            }
        }
        let writer_gone = match (&rec, &pair) {
            (Some(link), Some((lit, amb))) => {
                match link.tx.try_send((frame.clone(), lit.clone(), amb.clone())) {
                    Ok(()) => false,
                    Err(TrySendError::Full(_)) => {
                        lock(&link.status).dropped += 1;
                        false
                    }
                    // the writer stopped (limit or error); the UI shows why
                    Err(TrySendError::Disconnected(_)) => true,
                }
            }
            _ => false,
        };
        if writer_gone {
            rec = None;
        }
        let ir_view = ir.view();
        if last_print.elapsed() >= Duration::from_secs(1) {
            last_print = Instant::now();
            let tier = result.as_ref().map_or("-".to_owned(), |o| {
                tier_line(o.report.liveness_tier, &ir_view)
            });
            println!("{} | {tier} | lit fps {:.1}", ir_view.status, ir_view.fps);
            if let Ok(o) = &result {
                for (t, i) in o.report.tracks.iter().zip(&o.ir) {
                    println!(
                        "  #{} w {:.0}px | fused {} | {}",
                        t.id.0,
                        t.face_width_px,
                        liveness_label(&t.liveness),
                        track_ir_label(i)
                    );
                }
            }
            match &latest {
                Some(Ok(p)) => print_cues(p),
                Some(Err(e)) => println!("  IR cues failed: {e}"),
                None => {}
            }
        }
        publish(&shared, &ctx, |s| {
            s.rgb_version += 1;
            (s.rgb, s.health, s.ir) = (Some(frame), Some(health), ir_view);
            (s.status, s.report, s.track_ir) = match result {
                Ok(o) => ("running".into(), Some(o.report), o.ir),
                Err(e) => (format!("vision error: {e}"), None, Vec::new()),
            };
            if let Some((lit, amb)) = pair {
                s.ir_version += 1;
                (s.lit, s.ambient) = (Some(lit), amb);
            }
            if !ir_ok {
                (s.lit, s.ambient) = (None, None);
            }
            s.probe = latest.clone();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn parses_args() {
        assert_eq!(parse_args(&[]).unwrap(), Args::default());
        let a = parse_args(&args("--record Real-30cm --seconds 5")).unwrap();
        assert_eq!(
            (a.seconds, a.record.as_deref()),
            (Some(5), Some("Real-30cm"))
        );
        for bad in [
            "--seconds",
            "--seconds x",
            "--record",
            "--record !!",
            "--gallery g",
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn unmeasured_cues_read_as_not_available() {
        let c = IrCues {
            face_present: true,
            lit_minus_ambient: f32::NAN,
            face_vs_background: 2.5,
            relief: 1.05,
            glints: 1,
            lit_face: 120.0,
            lit_background: 30.0,
            ambient_background: f32::NAN,
        };
        let rows = cue_rows(&c, &IrLivenessConfig::default());
        let value = |name: &str| rows.iter().find(|r| r.0 == name).unwrap().1.clone();
        assert_eq!(value("face_vs_background"), "2.500");
        assert_eq!(value("lit_minus_ambient"), "n/a");
        assert_eq!(value("ambient_background"), "n/a");
        assert_eq!(value("glints"), "1");
        assert!(rows[2].2.contains("1.15"), "{:?}", rows[2]);
    }
}
