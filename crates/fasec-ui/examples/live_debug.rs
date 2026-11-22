//! Live debug viewer (M1 exit check): the camera with every tracked face annotated.
//!
//! `cargo run -p fasec-ui --example live_debug --release [-- --seconds N] [--gallery <path>]`
//!
//! The UI thread only draws. One worker thread owns the `Capture` and the `Pipeline`: it
//! starts/stops cameras on request (stopping the old one first), runs the pipeline on
//! every new frame and publishes the latest `(Frame, VisionReport)` plus stats. Once a second it
//! also prints processed FPS and per-stage ms to stdout. `--seconds N` closes the window after
//! N s (for unattended runs).
//!
//! `--gallery <path>` (made by the `enroll` example) adds identity: the worker opens it with
//! DPAPI, builds an `IdentityTracker` and applies it to every frame, and each box shows
//! `name (role) score`, `stranger best` or `…` (pending). A missing file is refused up front; any
//! other gallery error is shown in the side panel and identity stays pending (fail safe).
//!
//! IR: unless `%APPDATA%\fasec\config.toml` says `[liveness] ir = "off"`, the worker also opens
//! the IR camera paired with the RGB one and passes each lit pair to `process_with_ir`. The side
//! panel shows the IR source, sharing mode, health, fps and the liveness tier with its reason;
//! each box shows its IR verdict. Any IR failure leaves RGB running on the RGB tier.

#[path = "support/ir_link.rs"]
mod ir_link;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, bail};
use eframe::egui::{
    self, Align2, Color32, ColorImage, FontId, Pos2, Rect, RichText, Sense, Stroke, StrokeKind,
    TextureHandle, TextureOptions, Vec2, ViewportCommand,
};
use fasec_camera::{CameraInfo, Capture, LatestFrame};
use fasec_config::{CameraConfig, ExecutionProvider, IrMode};
use fasec_core::{CameraHealth, Frame, IdentityState, LivenessState, TrackObs, VisionReport};
use fasec_identity::{Gallery, IdentityTracker, MatcherConfig, Role, VoterConfig};
use fasec_platform::Dpapi;
use fasec_vision::{Pipeline, PipelineConfig, RECOGNISER_MODEL_ID, Runtime, TrackIr};
use ir_link::{IrLink, IrView, tier_line, track_ir_label};

const USAGE: &str = "usage: live_debug [--seconds N] [--gallery <path>]";
const MODELS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models");
const FETCH_HINT: &str =
    "Models missing or invalid: run `cargo run -p fasec-models --example fetch`";
/// How long closing the window waits for the worker. `Capture::stop` is bounded at ~2 s; a
/// worker still inside `Capture::start` (up to ~7 s) is detached and process exit frees the camera.
const QUIT_TIMEOUT: Duration = Duration::from_secs(3);
const STAGES: [&str; 4] = ["detect", "recognize", "liveness", "pose"];

#[derive(Debug, Default, PartialEq)]
struct Args {
    seconds: Option<u64>,
    gallery: Option<PathBuf>,
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
            "--gallery" => a.gallery = Some(PathBuf::from(value)),
            _ => bail!("unknown argument {flag}\n{USAGE}"),
        }
    }
    Ok(a)
}

fn main() -> Result<()> {
    let args = parse_args(&std::env::args().skip(1).collect::<Vec<_>>())?;
    // `Gallery::open` would create an empty gallery; a typo must not silently mean "nobody".
    if let Some(g) = args.gallery.as_ref().filter(|g| !g.is_file()) {
        bail!(
            "gallery {} not found: enrol first with \
             `cargo run -p fasec-ui --example enroll -- --gallery <path> --name <name>`",
            g.display()
        );
    }
    let ir_mode = ir_link::ir_mode_from_config().map_err(anyhow::Error::msg)?;
    let close_after = args.seconds.map(Duration::from_secs);
    let cams = fasec_camera::list().map_err(|e| e.to_string());
    eframe::run_native(
        "fasec live debug",
        eframe::NativeOptions::default(),
        Box::new(move |cc| {
            let app = App::new(
                cc.egui_ctx.clone(),
                cams,
                close_after,
                args.gallery,
                ir_mode,
            );
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe: {e}"))
}

enum Cmd {
    Switch(String),
    Quit,
}

/// What the worker publishes; the UI clones it once per paint.
#[derive(Clone, Default)]
struct Shared {
    /// Bumped on every publish so the UI re-uploads the texture only for a new frame.
    version: u64,
    status: String,
    frame: Option<Frame>,
    /// `None` while the last `process` failed (status says why) or the camera is unhealthy.
    report: Option<VisionReport>,
    health: Option<CameraHealth>,
    ep: &'static str,
    stats: Option<Stats>,
    /// `None` without `--gallery`; otherwise the gallery state (`Err`: identity stays pending).
    gallery: Option<Result<String, String>>,
    /// Person id -> `name (role)`, loaded once with the tracker.
    names: HashMap<String, String>,
    ir: IrView,
    /// The report's IR evidence per track.
    track_ir: Vec<TrackIr>,
}

/// One second of processed frames; stage ms in `STAGES` order.
#[derive(Clone, Copy, Default)]
struct Stats {
    fps: f32,
    avg: [f32; 4],
    max: [f32; 4],
    /// Frame rate the pipeline's scheduler asks for (idle vs active).
    want_fps: u64,
}

struct App {
    cams: Result<Vec<CameraInfo>, String>,
    selected: Option<usize>,
    shared: Arc<Mutex<Shared>>,
    tx: Sender<Cmd>,
    worker: Option<JoinHandle<()>>,
    texture: Option<TextureHandle>,
    texture_version: u64,
    close_at: Option<Instant>,
}

impl App {
    fn new(
        ctx: egui::Context,
        cams: Result<Vec<CameraInfo>, String>,
        close_after: Option<Duration>,
        gallery: Option<PathBuf>,
        ir_mode: IrMode,
    ) -> App {
        let shared = Arc::new(Mutex::new(Shared::default()));
        let (tx, rx) = mpsc::channel();
        let w_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("live-debug-worker".into())
            .spawn(move || worker(rx, w_shared, ctx, gallery, ir_mode));
        let worker = match worker {
            Ok(w) => Some(w),
            Err(e) => {
                lock(&shared).status = format!("spawn worker: {e}");
                None
            }
        };
        let selected = cams.as_ref().ok().filter(|c| !c.is_empty()).map(|_| 0);
        if let (Some(i), Ok(cams)) = (selected, &cams) {
            let _ = tx.send(Cmd::Switch(cams[i].id.clone()));
        }
        App {
            cams,
            selected,
            shared,
            tx,
            worker,
            texture: None,
            texture_version: 0,
            close_at: close_after.map(|d| Instant::now() + d),
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui, s: &Shared) {
        ui.heading("Camera");
        match &self.cams {
            Err(e) => {
                ui.colored_label(Color32::RED, format!("camera list failed: {e}"));
            }
            Ok(cams) if cams.is_empty() => {
                ui.colored_label(Color32::RED, "no camera found");
            }
            Ok(cams) => {
                let before = self.selected;
                let name = |i: Option<usize>| i.map_or("-", |i| cams[i].name.as_str());
                egui::ComboBox::from_id_salt("camera")
                    .selected_text(name(self.selected))
                    .width(260.0)
                    .show_ui(ui, |ui| {
                        for (i, c) in cams.iter().enumerate() {
                            ui.selectable_value(&mut self.selected, Some(i), &c.name);
                        }
                    });
                if let Some(i) = self.selected.filter(|_| self.selected != before) {
                    let _ = self.tx.send(Cmd::Switch(cams[i].id.clone()));
                }
            }
        }
        ui.label(&s.status);
        let (health, color) = match &s.health {
            None => ("-".to_owned(), Color32::GRAY),
            Some(CameraHealth::Ok) => ("Ok".to_owned(), Color32::GREEN),
            Some(h) => (format!("{h:?}"), Color32::RED),
        };
        ui.horizontal(|ui| {
            ui.label("health:");
            ui.colored_label(color, health);
        });
        if let Some(f) = &s.frame {
            ui.label(format!("frame: {}x{} seq {}", f.width, f.height, f.seq));
        }
        ui.separator();
        ui.heading("Vision");
        ui.label(format!("EP for new sessions: {}", or_dash(s.ep)));
        ui.small("(per-stage EP is not exposed by Pipeline; a stage may fall back to CPU on its first run)");
        if let Some(r) = &s.report {
            ui.label(format!("tracks: {}", r.tracks.len()));
            ui.label(RichText::new(tier_line(r.liveness_tier, &s.ir)).strong());
            let t = &r.timings;
            ui.label(format!(
                "last frame ms: det {:.1}  rec {:.1}  live {:.1}  pose {:.1}",
                t.detect_ms, t.recognize_ms, t.liveness_ms, t.pose_ms
            ));
        }
        if let Some(st) = &s.stats {
            ui.label(RichText::new(format!("processed FPS: {:.1}", st.fps)).strong());
            ui.label(format!("scheduler asks for: {} fps", st.want_fps));
            egui::Grid::new("stages").striped(true).show(ui, |ui| {
                ui.label("stage (1 s)");
                ui.label("avg ms");
                ui.label("max ms");
                ui.end_row();
                for (i, name) in STAGES.iter().enumerate() {
                    ui.label(*name);
                    ui.label(format!("{:.1}", st.avg[i]));
                    ui.label(format!("{:.1}", st.max[i]));
                    ui.end_row();
                }
            });
        }
        ui.separator();
        ui.heading("IR");
        let color = if s.ir.on {
            Color32::GREEN
        } else {
            Color32::YELLOW
        };
        ui.colored_label(color, &s.ir.status);
        ui.label(format!("source: {}", s.ir.source.as_deref().unwrap_or("-")));
        ui.label(format!("sharing: {}", or_dash(s.ir.sharing)));
        ui.label(format!(
            "health: {}",
            s.ir.health
                .as_ref()
                .map_or("-".into(), |h| format!("{h:?}"))
        ));
        ui.label(format!("lit fps: {:.1}", s.ir.fps));
        if let Some(g) = &s.gallery {
            ui.separator();
            ui.heading("Identity");
            match g {
                Ok(msg) => ui.label(msg),
                Err(e) => ui.colored_label(Color32::RED, e),
            };
        }
    }

    fn video(&mut self, ui: &mut egui::Ui, s: &Shared) {
        let Some(frame) = &s.frame else {
            self.texture = None;
            ui.centered_and_justified(|ui| ui.heading(or_dash(&s.status)));
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
        let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(tex.id(), r, uv, Color32::WHITE);
        let Some(report) = &s.report else { return };
        let at = |x: f32, y: f32| r.min + Vec2::new(x * r.width(), y * r.height());
        for t in &report.tracks {
            let color = match t.liveness {
                LivenessState::Live { .. } => Color32::GREEN,
                LivenessState::Spoof { .. } => Color32::RED,
                LivenessState::Unknown => Color32::YELLOW,
            };
            let b = t.bbox;
            let rect = Rect::from_min_max(at(b.x, b.y), at(b.x + b.w, b.y + b.h));
            painter.rect_stroke(rect, 0.0, Stroke::new(2.0, color), StrokeKind::Outside);
            for p in &t.landmarks5 {
                painter.circle_filled(at(p.x, p.y), 2.5, color);
            }
            let mut text = label(t);
            if let Some(ir) = s.track_ir.iter().find(|ir| ir.track == t.id) {
                text = format!("{text}\n{}", track_ir_label(ir));
            }
            if s.gallery.is_some() {
                text = format!("{}\n{text}", identity_label(&t.identity, &s.names));
            }
            let galley = painter.layout_no_wrap(text, FontId::monospace(13.0), color);
            let text = Align2::LEFT_BOTTOM.anchor_size(rect.left_top(), galley.size());
            painter.rect_filled(text.expand(2.0), 2.0, Color32::from_black_alpha(170));
            painter.galley(text.min, galley, color);
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
        egui::Panel::right("info")
            .default_size(320.0)
            .show(ui, |ui| self.side_panel(ui, &s));
        egui::CentralPanel::default_margins().show(ui, |ui| self.video(ui, &s));
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

fn label(t: &TrackObs) -> String {
    let live = match &t.liveness {
        LivenessState::Live { score, tier } => format!("live {score:.2} {tier:?}"),
        LivenessState::Spoof { score, tier } => format!("SPOOF {score:.2} {tier:?}"),
        LivenessState::Unknown => "liveness ?".to_owned(),
    };
    let pose = t.head_pose.map_or("pose ?".to_owned(), |p| {
        format!("yaw {:.0} pitch {:.0} roll {:.0}", p.yaw, p.pitch, p.roll)
    });
    format!(
        "#{}  det {:.2}  w {:.0}px\n{live}\n{pose}",
        t.id.0, t.det_score, t.face_width_px
    )
}

/// `name (role) score`, `stranger best` or `…`; never the embedding.
fn identity_label(id: &IdentityState, names: &HashMap<String, String>) -> String {
    match id {
        IdentityState::Known { person, score } => {
            let who = names
                .get(person)
                .map_or("? (not in gallery)", String::as_str);
            format!("{who} {score:.2}")
        }
        IdentityState::Stranger { best } => format!("stranger {best:.2}"),
        IdentityState::Pending | IdentityState::Unknown => "…".to_owned(),
    }
}

/// Names by person id and the tracker for the current recogniser; any error means no tracker.
fn load_identity(path: &Path) -> Result<(HashMap<String, String>, IdentityTracker)> {
    let g = Gallery::open(path, Arc::new(Dpapi::new()))
        .with_context(|| format!("opening gallery {}", path.display()))?;
    let names = g
        .persons()
        .context("reading persons")?
        .into_iter()
        .map(|p| {
            let role = match p.role {
                Role::Owner => "owner",
                Role::Authorised => "authorised",
            };
            (p.id.0, format!("{} ({role})", p.name))
        })
        .collect();
    let tracker = IdentityTracker::from_gallery(
        &g,
        RECOGNISER_MODEL_ID,
        MatcherConfig::default(),
        VoterConfig::default(),
    )
    .context("loading templates")?;
    Ok((names, tracker))
}

fn or_dash(s: &str) -> &str {
    if s.is_empty() { "-" } else { s }
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

/// Owns the camera and the pipeline; exits on `Quit` or when the UI side of the channel is gone.
fn worker(
    rx: Receiver<Cmd>,
    shared: Arc<Mutex<Shared>>,
    ctx: egui::Context,
    gallery: Option<PathBuf>,
    ir_mode: IrMode,
) {
    publish(&shared, &ctx, |s| s.status = "loading models…".into());
    let load = || -> Result<(Runtime, Pipeline)> {
        let rt = Runtime::new(ExecutionProvider::Auto).context("onnx runtime")?;
        let p = Pipeline::from_models(&rt, Path::new(MODELS_DIR), PipelineConfig::default())
            .with_context(|| format!("loading models from {MODELS_DIR}"))?;
        Ok((rt, p))
    };
    // `_rt` is kept alive for as long as the pipeline's sessions.
    let (_rt, mut pipeline) = match load() {
        Ok((rt, p)) => {
            let ep = rt.active_ep();
            println!("models loaded, EP for new sessions: {ep}");
            publish(&shared, &ctx, |s| s.ep = ep);
            (rt, p)
        }
        Err(e) => {
            eprintln!("{e:#}");
            publish(&shared, &ctx, |s| {
                s.status = format!("{e:#}\n\n{FETCH_HINT}")
            });
            return;
        }
    };
    let mut tracker = gallery.and_then(|path| match load_identity(&path) {
        Ok((names, t)) => {
            let msg = format!("{}: {} person(s)", path.display(), names.len());
            publish(&shared, &ctx, |s| {
                (s.gallery, s.names) = (Some(Ok(msg)), names)
            });
            Some(t)
        }
        Err(e) => {
            eprintln!("{e:#}");
            let msg = format!("{e:#}\nidentity stays pending");
            publish(&shared, &ctx, |s| s.gallery = Some(Err(msg)));
            None
        }
    });
    publish(&shared, &ctx, |s| s.status = "waiting for a camera".into());
    // Each capture gets its own slot so a frame from a previous camera is never handed out.
    let mut cap: Option<(Capture, Arc<LatestFrame>)> = None;
    let mut ir = IrLink::new(ir_mode);
    let mut seq = 0;
    let mut window = StatsWindow::new();
    loop {
        // Drain commands: the newest switch wins; Quit or a vanished UI ends the worker.
        let mut switch = None;
        loop {
            match rx.try_recv() {
                Ok(Cmd::Switch(id)) => switch = Some(id),
                Ok(Cmd::Quit) | Err(TryRecvError::Disconnected) => {
                    let t = Instant::now();
                    if let Some((c, _)) = cap.take() {
                        c.stop();
                    }
                    ir.stop();
                    println!("cameras stopped in {} ms", t.elapsed().as_millis());
                    return;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if let Some(id) = switch {
            if let Some((c, _)) = cap.take() {
                c.stop();
            }
            ir.stop();
            publish(&shared, &ctx, |s| {
                s.status = "starting…".into();
                (s.frame, s.report, s.health, s.stats) = (None, None, None, None);
                s.ir = ir.view();
            });
            let slot = Arc::new(LatestFrame::new());
            match Capture::start(&id, &CameraConfig::default(), Arc::clone(&slot)) {
                Ok(c) => {
                    cap = Some((c, slot));
                    publish(&shared, &ctx, |s| s.status = "starting IR…".into());
                    ir.start(&id);
                    println!("{}", ir.view().status);
                    publish(&shared, &ctx, |s| {
                        (s.status, s.ir) = ("running".into(), ir.view());
                    });
                }
                Err(e) => publish(&shared, &ctx, |s| s.status = format!("camera error: {e}")),
            }
            (seq, window) = (0, StatsWindow::new());
            continue;
        }
        let Some((c, slot)) = &cap else {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        };
        // IR first, so the RGB frame taken next is the one closest to the lit frame.
        let (ir_pair, ir_health) = ir.next();
        let Some(frame) = slot.take_newer(seq, Duration::from_millis(100)) else {
            // No frame: still surface a stall (`Lost`) and stop drawing stale boxes.
            let h = c.health();
            if h != CameraHealth::Ok {
                publish(&shared, &ctx, |s| (s.health, s.report) = (Some(h), None));
            }
            continue;
        };
        ir.delivered();
        seq = frame.seq;
        let health = c.health();
        let ir_frames = ir_pair.as_ref().map(|(lit, amb)| (lit, amb.as_ref()));
        // Embeddings go only to the tracker, then are dropped (and zeroised) here.
        let result = pipeline
            .process_with_ir(&frame, ir_frames, health.clone(), ir_health)
            .map(|mut o| {
                if let Some(t) = &mut tracker {
                    t.apply(&mut o.report, &o.embeddings);
                }
                (o.report, o.ir)
            });
        let ir_view = ir.view();
        if let Ok((r, _)) = &result {
            let t = &r.timings;
            window.add([t.detect_ms, t.recognize_ms, t.liveness_ms, t.pose_ms]);
        }
        let stats = window.roll(1000 / pipeline.desired_interval_ms().max(1));
        if let Some(st) = &stats {
            println!(
                "fps {:5.1} | ms avg/max: det {:.1}/{:.1} rec {:.1}/{:.1} live {:.1}/{:.1} pose {:.2}/{:.2} | tracks {} | health {:?}",
                st.fps,
                st.avg[0],
                st.max[0],
                st.avg[1],
                st.max[1],
                st.avg[2],
                st.max[2],
                st.avg[3],
                st.max[3],
                result.as_ref().map_or(0, |(r, _)| r.tracks.len()),
                health
            );
            if let Ok((r, track_ir)) = &result {
                println!(
                    "  {} | {} | lit fps {:.1}",
                    ir_view.status,
                    tier_line(r.liveness_tier, &ir_view),
                    ir_view.fps
                );
                for (t, ir) in r.tracks.iter().zip(track_ir) {
                    println!(
                        "  #{} w {:.0}px | {:?} | {} | {:?}",
                        t.id.0,
                        t.face_width_px,
                        t.liveness,
                        track_ir_label(ir),
                        t.identity
                    );
                }
            }
        }
        publish(&shared, &ctx, |s| {
            (s.status, s.report, s.track_ir) = match result {
                Ok((r, ir)) => ("running".into(), Some(r), ir),
                Err(e) => (format!("vision error: {e}"), None, Vec::new()),
            };
            (s.frame, s.health, s.ir) = (Some(frame), Some(health), ir_view);
            if stats.is_some() {
                s.stats = stats;
            }
        });
    }
}

/// Accumulates per-frame stage timings and rolls them into `Stats` once a second.
struct StatsWindow {
    start: Instant,
    n: u32,
    sum: [f32; 4],
    max: [f32; 4],
}

impl StatsWindow {
    fn new() -> StatsWindow {
        StatsWindow {
            start: Instant::now(),
            n: 0,
            sum: [0.0; 4],
            max: [0.0; 4],
        }
    }

    fn add(&mut self, ms: [f32; 4]) {
        self.n += 1;
        for i in 0..4 {
            self.sum[i] += ms[i];
            self.max[i] = self.max[i].max(ms[i]);
        }
    }

    /// `Some` (and a fresh window) once at least a second has passed.
    fn roll(&mut self, want_fps: u64) -> Option<Stats> {
        let secs = self.start.elapsed().as_secs_f32();
        if secs < 1.0 {
            return None;
        }
        let n = self.n.max(1) as f32;
        let stats = Stats {
            fps: self.n as f32 / secs,
            avg: self.sum.map(|s| s / n),
            max: self.max,
            want_fps,
        };
        *self = StatsWindow::new();
        Some(stats)
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
        let a = parse_args(&args("--gallery g.db --seconds 5")).unwrap();
        assert_eq!((a.seconds, a.gallery), (Some(5), Some("g.db".into())));
        for bad in ["--seconds", "--seconds x", "--gallery", "--pin 1234"] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn identity_labels() {
        let names = HashMap::from([("id1".to_owned(), "Ann (owner)".to_owned())]);
        let known = |person: &str| IdentityState::Known {
            person: person.into(),
            score: 0.714,
        };
        assert_eq!(identity_label(&known("id1"), &names), "Ann (owner) 0.71");
        assert_eq!(
            identity_label(&known("gone"), &names),
            "? (not in gallery) 0.71"
        );
        let stranger = IdentityState::Stranger { best: 0.2 };
        assert_eq!(identity_label(&stranger, &names), "stranger 0.20");
        assert_eq!(identity_label(&IdentityState::Pending, &names), "…");
    }
}
