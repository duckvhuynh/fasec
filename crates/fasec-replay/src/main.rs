//! Offline replay CLI: run the vision pipeline on recorded frames or video with a simulated
//! clock and write one `VisionReport` per line (JSONL). Embeddings, when asked for, go to a
//! separate `<out>.embeddings.jsonl` file, never into the report lines. With `--gallery`, each
//! track's identity is matched against an encrypted face gallery (DPAPI, Windows only).
//!
//! `--ir-input <dir>...` replays recordings made by the `ir_debug` example instead: each RGB
//! frame at its recorded time with its lit IR pair, and each line gains an `ir` object holding
//! every IR cue per track (for tuning the IR thresholds, Task 7).

mod input;
mod output;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fasec_config::ExecutionProvider;
use fasec_core::{CameraHealth, Frame, IrFrame};
use fasec_identity::{Gallery, IdentityTracker, MatcherConfig, SecretProtector, VoterConfig};
use fasec_replay::probe::IrProbe;
use fasec_vision::{Pipeline, PipelineConfig, ProcessOutput, RECOGNISER_MODEL_ID, Runtime};

use crate::input::{Ffmpeg, RawFrame};
use crate::output::{IrLine, OutputOptions};

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the pipeline on every frame of `--input` and write JSONL reports.
    Run(RunArgs),
}

#[derive(clap::Args)]
struct RunArgs {
    /// Directory of `*.png`/`*.jpg` frames (sorted by file name) or a video file (needs ffmpeg).
    #[arg(long, required_unless_present = "ir_input")]
    input: Option<PathBuf>,
    /// Recording session directories made by the `ir_debug` example (under
    /// `%LOCALAPPDATA%\fasec\recordings`), replayed instead of `--input` at their recorded times.
    /// Several are replayed in order through one pipeline, each continuing 500 ms after the last,
    /// so IR trust earned in one carries into the next.
    #[arg(long, num_args = 1.., conflicts_with_all = ["input", "fps"])]
    ir_input: Vec<PathBuf>,
    /// Simulated frame rate: frame `i` gets `ts_ms = i * 1000 / fps`.
    #[arg(long, default_value_t = 15, value_parser = clap::value_parser!(u32).range(1..))]
    fps: u32,
    /// Directory holding the model files (verified against the manifest built into the binary).
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, value_enum, default_value_t = Ep::Auto)]
    ep: Ep,
    /// Output file (default: stdout).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Also write face embeddings (biometric templates, in plaintext) to
    /// `<out>.embeddings.jsonl`. Requires `--out`.
    #[arg(long)]
    with_embeddings: bool,
    /// Include wall-clock stage timings (makes output non-reproducible).
    #[arg(long)]
    with_timings: bool,
    /// Encrypted face gallery (DPAPI, Windows only): tracks become Known/Stranger instead of
    /// staying Pending.
    #[arg(long)]
    gallery: Option<PathBuf>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Ep {
    Auto,
    Cpu,
    Directml,
}

impl From<Ep> for ExecutionProvider {
    fn from(ep: Ep) -> Self {
        match ep {
            Ep::Auto => ExecutionProvider::Auto,
            Ep::Cpu => ExecutionProvider::Cpu,
            Ep::Directml => ExecutionProvider::DirectMl,
        }
    }
}

/// The pipeline as `run` sees it; tests substitute a mock.
trait Process {
    /// `ir`: the lit IR frame recorded with `frame` and its ambient partner, if any.
    fn process(
        &mut self,
        frame: &Frame,
        ir: Option<(&IrFrame, Option<&IrFrame>)>,
    ) -> Result<ProcessOutput>;
    fn desired_interval_ms(&self) -> u64;
}

impl Process for Pipeline {
    fn process(
        &mut self,
        frame: &Frame,
        ir: Option<(&IrFrame, Option<&IrFrame>)>,
    ) -> Result<ProcessOutput> {
        // every recorded step has a lit pair; without one IR is `Lost`, as in `Pipeline::process`
        let ir_health = if ir.is_some() {
            CameraHealth::Ok
        } else {
            CameraHealth::Lost
        };
        Ok(self.process_with_ir(frame, ir, CameraHealth::Ok, ir_health)?)
    }
    fn desired_interval_ms(&self) -> u64 {
        Pipeline::desired_interval_ms(self)
    }
}

/// `inner` followed by the identity vote, which writes each track's `identity`.
struct WithIdentity<'a> {
    inner: &'a mut dyn Process,
    tracker: IdentityTracker,
}

impl Process for WithIdentity<'_> {
    fn process(
        &mut self,
        frame: &Frame,
        ir: Option<(&IrFrame, Option<&IrFrame>)>,
    ) -> Result<ProcessOutput> {
        let mut out = self.inner.process(frame, ir)?;
        self.tracker.apply(&mut out.report, &out.embeddings);
        Ok(out)
    }
    fn desired_interval_ms(&self) -> u64 {
        self.inner.desired_interval_ms()
    }
}

#[cfg(windows)]
fn gallery_protector() -> Result<Arc<dyn SecretProtector>> {
    Ok(Arc::new(fasec_platform::Dpapi::new()))
}

#[cfg(not(windows))]
fn gallery_protector() -> Result<Arc<dyn SecretProtector>> {
    anyhow::bail!("--gallery: DPAPI unsupported on this platform")
}

/// Any gallery, decrypt or model-id error fails the run: identity is never guessed.
fn open_identity(path: &Path) -> Result<IdentityTracker> {
    // `Gallery::open` would create a missing file, silently making everyone a stranger.
    anyhow::ensure!(path.is_file(), "gallery not found: {}", path.display());
    let g = Gallery::open(path, gallery_protector()?)
        .with_context(|| format!("opening gallery {}", path.display()))?;
    IdentityTracker::from_gallery(
        &g,
        RECOGNISER_MODEL_ID,
        MatcherConfig::default(),
        VoterConfig::default(),
    )
    .with_context(|| format!("loading gallery {}", path.display()))
}

/// Feeds every frame to `p` (no skipping) and writes one line each to `w`, plus each embedding
/// computed on it to `emb` if given; returns the frame count. A recorded frame keeps its RGB
/// time and goes in with its lit IR pair, which `probe` (if given) measures for the line.
fn run(
    frames: impl Iterator<Item = Result<RawFrame>>,
    p: &mut dyn Process,
    mut probe: Option<&mut IrProbe>,
    fps: u32,
    opts: OutputOptions,
    w: &mut dyn Write,
    mut emb: Option<&mut dyn Write>,
) -> Result<u64> {
    anyhow::ensure!(fps > 0, "fps must be at least 1");
    let mut n = 0;
    for (seq, raw) in (0u64..).zip(frames) {
        let raw = raw.with_context(|| format!("frame {seq}"))?;
        let session = raw.recorded.as_ref().map_or(0, |r| r.0);
        let rec = raw.recorded.as_ref().map(|r| &r.1);
        let ts_ms = rec.map_or(seq * 1000 / u64::from(fps), |r| r.step.rgb_ts_ms);
        let frame = Frame::new(seq, ts_ms, raw.width, raw.height, raw.data)?;
        let ir = rec.map(|r| (&r.lit, r.ambient.as_ref()));
        let out = p
            .process(&frame, ir)
            .with_context(|| format!("frame {seq}"))?;
        let ir_line = match (rec, probe.as_deref_mut()) {
            (Some(r), Some(probe)) => probe
                .measure(&r.lit, r.ambient.as_ref(), &out.report.tracks)
                .with_context(|| format!("IR of frame {seq}"))?
                .map(|f| IrLine::new(session, &r.step, &f, &out.ir)),
            _ => None,
        };
        if let Some(e) = emb.as_deref_mut() {
            output::write_embeddings(e, out.report.seq, &out.embeddings)?;
        }
        output::write_line(w, out.report, ir_line, p.desired_interval_ms(), opts)?;
        n += 1;
    }
    w.flush()?;
    if let Some(e) = emb {
        e.flush()?;
    }
    Ok(n)
}

const EMBEDDINGS_NEED_OUT: &str =
    "--with-embeddings requires --out (embeddings are never written to stdout)";

fn replay(args: &RunArgs) -> Result<u64> {
    anyhow::ensure!(
        !args.with_embeddings || args.out.is_some(),
        EMBEDDINGS_NEED_OUT
    );
    // Gallery and input opened first so either fails before the models load.
    let tracker = args.gallery.as_deref().map(open_identity).transpose()?;
    let frames = match (&args.ir_input[..], &args.input) {
        ([], Some(input)) => input::open(input, &Ffmpeg::default())?,
        ([], None) => anyhow::bail!("--input or --ir-input is required"),
        (dirs, _) => input::open_recordings(dirs)?,
    };
    let rt = Runtime::new(args.ep.into())?;
    let cfg = PipelineConfig::default();
    let loading = || format!("loading models from {}", args.models.display());
    let mut p = Pipeline::from_models(&rt, &args.models, cfg).with_context(loading)?;
    let mut probe = match &args.ir_input[..] {
        [] => None,
        _ => Some(IrProbe::from_models(&rt, &args.models, cfg.ir).with_context(loading)?),
    };
    match tracker {
        Some(tracker) => {
            let mut p = WithIdentity {
                inner: &mut p,
                tracker,
            };
            write_outputs(args, frames, &mut p, probe.as_mut())
        }
        None => write_outputs(args, frames, &mut p, probe.as_mut()),
    }
}

/// Creates the output file(s) `args` asks for and runs `p` over `frames` into them.
fn write_outputs(
    args: &RunArgs,
    frames: impl Iterator<Item = Result<RawFrame>>,
    p: &mut dyn Process,
    probe: Option<&mut IrProbe>,
) -> Result<u64> {
    let opts = OutputOptions {
        with_timings: args.with_timings,
    };
    let create = |path: &Path| {
        File::create(path)
            .map(BufWriter::new)
            .with_context(|| format!("creating {}", path.display()))
    };
    let Some(path) = &args.out else {
        anyhow::ensure!(!args.with_embeddings, EMBEDDINGS_NEED_OUT);
        let mut w = std::io::stdout().lock();
        return run(frames, p, probe, args.fps, opts, &mut w, None);
    };
    let side = output::embeddings_path(path);
    let mut emb = if args.with_embeddings {
        Some(create(&side)?)
    } else {
        // A sidecar from an earlier run would leave plaintext templates beside this report.
        match std::fs::remove_file(&side) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(e).with_context(|| format!("removing stale {}", side.display()));
            }
            _ => None,
        }
    };
    let emb = emb.as_mut().map(|w| w as &mut dyn Write);
    run(frames, p, probe, args.fps, opts, &mut create(path)?, emb)
}

fn main() -> Result<()> {
    let Cli {
        cmd: Cmd::Run(args),
    } = Cli::parse();
    replay(&args)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use fasec_core::{
        HeadPose, IdentityState, LivenessState, LivenessTier, PointF, RectF, StageTimings,
        TrackEmbedding, TrackId, TrackObs, VisionReport,
    };
    use fasec_replay::recording::Recorder;
    use fasec_vision::{
        Detection, FaceDetector, FaceEmbedder, LivenessModel, PoseEstimator, VisionError,
    };

    use super::*;

    /// Records every frame it sees and returns one track with an embedding.
    #[derive(Default)]
    struct Mock {
        seen: Vec<(u64, u64, u32, u32, u8)>,
    }

    impl Process for Mock {
        fn process(
            &mut self,
            f: &Frame,
            _: Option<(&IrFrame, Option<&IrFrame>)>,
        ) -> Result<ProcessOutput> {
            self.seen
                .push((f.seq, f.ts_ms, f.width, f.height, f.data[0]));
            let p = PointF { x: 0.5, y: 0.5 };
            let report = VisionReport {
                seq: f.seq,
                ts_ms: f.ts_ms,
                tracks: vec![TrackObs {
                    id: TrackId(1),
                    bbox: RectF {
                        x: 0.1,
                        y: 0.1,
                        w: 0.2,
                        h: 0.2,
                    },
                    landmarks5: [p; 5],
                    det_score: 0.9,
                    identity: IdentityState::Unknown,
                    liveness: LivenessState::Unknown,
                    head_pose: None,
                    face_width_px: 10.0,
                    age_ms: 0,
                }],
                camera_health: CameraHealth::Ok,
                liveness_tier: LivenessTier::Rgb,
                timings: StageTimings {
                    detect_ms: 1.5,
                    recognize_ms: 2.5,
                    liveness_ms: 3.5,
                    pose_ms: 4.5,
                },
            };
            Ok(ProcessOutput {
                report,
                embeddings: vec![TrackEmbedding {
                    track: TrackId(1),
                    ts_ms: f.ts_ms,
                    embedding: vec![0.25, -0.5].into(),
                }],
                ir: Vec::new(),
            })
        }
        fn desired_interval_ms(&self) -> u64 {
            66
        }
    }

    fn raw(width: u32, height: u32, v: u8) -> Result<RawFrame> {
        Ok(RawFrame {
            width,
            height,
            data: vec![v; (width * height * 3) as usize],
            recorded: None,
        })
    }

    #[test]
    fn dir_input_sorted_and_timestamped() {
        let dir = tempfile::tempdir().unwrap();
        // Written out of order; distinct size and colour identify each file.
        for (name, w, v) in [("f2.png", 4, 20u8), ("f0.jpg", 2, 0), ("f1.PNG", 3, 10)] {
            let img = image::RgbImage::from_pixel(w, 2, image::Rgb([v, v, v]));
            let fmt = if name.ends_with(".jpg") {
                image::ImageFormat::Jpeg
            } else {
                image::ImageFormat::Png
            };
            img.save_with_format(dir.path().join(name), fmt).unwrap();
        }
        std::fs::write(dir.path().join("notes.txt"), "ignored").unwrap();

        let frames = input::open(dir.path(), &Ffmpeg::default()).unwrap();
        let mut mock = Mock::default();
        let mut out = Vec::new();
        let opts = OutputOptions::default();
        let n = run(frames, &mut mock, None, 4, opts, &mut out, None).unwrap();
        assert_eq!(n, 3);
        let seen: Vec<_> = mock.seen.iter().map(|s| (s.0, s.1, s.2, s.3)).collect();
        assert_eq!(seen, [(0, 0, 2, 2), (1, 250, 3, 2), (2, 500, 4, 2)]);
        // JPEG is lossy; PNG values are exact.
        assert!(mock.seen[0].4 < 3);
        assert_eq!(mock.seen[1].4, 10);
        assert_eq!(mock.seen[2].4, 20);
    }

    #[test]
    fn jsonl_one_line_per_frame() {
        let frames = || vec![raw(2, 2, 0), raw(2, 2, 1), raw(2, 2, 2)].into_iter();
        let mut out = Vec::new();
        let opts = OutputOptions::default();
        run(
            frames(),
            &mut Mock::default(),
            None,
            10,
            opts,
            &mut out,
            None,
        )
        .unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.ends_with('\n'));
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 3);
        for (i, v) in lines.iter().enumerate() {
            assert_eq!(v["seq"], i as u64);
            assert_eq!(v["ts_ms"], i as u64 * 100);
            assert_eq!(v["desired_interval_ms"], 66);
            let track = v["tracks"][0].as_object().unwrap();
            assert!(!track.contains_key("embedding"), "{v}");
            assert_eq!(v["timings"]["detect_ms"], 0.0);
            // The rest of the line is still a `VisionReport`.
            let r: VisionReport = serde_json::from_value(v.clone()).unwrap();
            assert_eq!(r.tracks.len(), 1);
        }

        let mut out = Vec::new();
        let opts = OutputOptions { with_timings: true };
        run(
            frames(),
            &mut Mock::default(),
            None,
            10,
            opts,
            &mut out,
            None,
        )
        .unwrap();
        let first: serde_json::Value =
            serde_json::from_str(String::from_utf8(out).unwrap().lines().next().unwrap()).unwrap();
        assert_eq!(first["timings"]["detect_ms"], 1.5);
    }

    fn args(out: Option<PathBuf>, with_embeddings: bool) -> RunArgs {
        RunArgs {
            input: Some(PathBuf::from("does-not-exist")),
            ir_input: Vec::new(),
            fps: 10,
            models: PathBuf::from("does-not-exist"),
            ep: Ep::Cpu,
            out,
            with_embeddings,
            with_timings: false,
            gallery: None,
        }
    }

    #[test]
    fn with_embeddings_writes_sidecar_file() {
        let frames = || vec![raw(2, 2, 0), raw(2, 2, 1)].into_iter();
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("r.jsonl");
        let side = dir.path().join("r.jsonl.embeddings.jsonl");

        write_outputs(
            &args(Some(out.clone()), false),
            frames(),
            &mut Mock::default(),
            None,
        )
        .unwrap();
        assert!(!side.exists(), "sidecar written without --with-embeddings");

        write_outputs(
            &args(Some(out.clone()), true),
            frames(),
            &mut Mock::default(),
            None,
        )
        .unwrap();
        let report = std::fs::read_to_string(&out).unwrap();
        assert_eq!(report.lines().count(), 2);
        assert!(!report.contains("embedding"), "{report}");
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(&side)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let want = [0, 1]
            .map(|seq| serde_json::json!({"seq": seq, "track": 1, "embedding": [0.25, -0.5]}));
        assert_eq!(lines, want);
    }

    #[test]
    fn run_without_embeddings_removes_stale_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("r.jsonl");
        let side = output::embeddings_path(&out);
        std::fs::write(&side, "{\"seq\":0,\"track\":1,\"embedding\":[0.1]}\n").unwrap();
        let frames = vec![raw(2, 2, 0)].into_iter();
        write_outputs(
            &args(Some(out.clone()), false),
            frames,
            &mut Mock::default(),
            None,
        )
        .unwrap();
        assert!(
            !side.exists(),
            "stale plaintext embeddings left beside a fresh report"
        );
        assert_eq!(std::fs::read_to_string(&out).unwrap().lines().count(), 1);
    }

    #[test]
    fn with_embeddings_requires_out() {
        // Refused before the input is opened or any model loads.
        let e = replay(&args(None, true)).unwrap_err();
        assert!(format!("{e:#}").contains("--out"), "{e:#}");
        let e = write_outputs(
            &args(None, true),
            std::iter::empty(),
            &mut Mock::default(),
            None,
        );
        assert!(format!("{:#}", e.unwrap_err()).contains("--out"));
    }

    #[test]
    fn frame_error_stops_run() {
        let frames = vec![raw(2, 2, 0), Err(anyhow::anyhow!("decode failed"))].into_iter();
        let mut out = Vec::new();
        let err = run(
            frames,
            &mut Mock::default(),
            None,
            10,
            OutputOptions::default(),
            &mut out,
            None,
        );
        assert!(format!("{:#}", err.unwrap_err()).contains("decode failed"));
        assert_eq!(String::from_utf8(out).unwrap().lines().count(), 1);
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn deterministic_output() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let tmp = tempfile::tempdir().unwrap();
        let go = |name: &str| {
            let out = tmp.path().join(name);
            let args = RunArgs {
                input: Some(root.join(".assets")),
                ir_input: Vec::new(),
                fps: 15,
                models: root.join("models"),
                ep: Ep::Cpu,
                out: Some(out.clone()),
                with_embeddings: true,
                with_timings: false,
                gallery: None,
            };
            assert_eq!(replay(&args).unwrap(), 3);
            let side = output::embeddings_path(&out);
            (std::fs::read(out).unwrap(), std::fs::read(side).unwrap())
        };
        let (report, side) = go("a.jsonl");
        assert_eq!((report.clone(), side.clone()), go("b.jsonl"));
        let (report, side) = (
            String::from_utf8(report).unwrap(),
            String::from_utf8(side).unwrap(),
        );
        assert!(!report.contains("embedding"), "{report}");
        assert!(side.contains("\"embedding\":["), "no face found: {side}");
    }

    fn identities(jsonl: &str) -> Vec<Vec<IdentityState>> {
        jsonl
            .lines()
            .map(|l| {
                let r: VisionReport = serde_json::from_str(l).unwrap();
                r.tracks.into_iter().map(|t| t.identity).collect()
            })
            .collect()
    }

    #[test]
    fn identity_tracker_sets_track_identity() {
        use fasec_identity::protect::testing::XorProtector;
        // Empty gallery: the mock's track is Pending until the vote window makes it a Stranger.
        let g = Gallery::open_in_memory(std::sync::Arc::new(XorProtector::default())).unwrap();
        let tracker = IdentityTracker::from_gallery(
            &g,
            RECOGNISER_MODEL_ID,
            MatcherConfig::default(),
            VoterConfig::default(),
        )
        .unwrap();
        let mut mock = Mock::default();
        let mut p = WithIdentity {
            inner: &mut mock,
            tracker,
        };
        let frames = (0..5).map(|i| raw(2, 2, i));
        let mut out = Vec::new();
        run(
            frames,
            &mut p,
            None,
            10,
            OutputOptions::default(),
            &mut out,
            None,
        )
        .unwrap();
        let ids = identities(&String::from_utf8(out).unwrap());
        assert_eq!(ids[..4], vec![vec![IdentityState::Pending]; 4][..]);
        assert!(
            matches!(ids[4][..], [IdentityState::Stranger { .. }]),
            "{ids:?}"
        );
    }

    #[test]
    fn missing_gallery_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.db");
        let mut a = args(None, false);
        a.gallery = Some(path.clone());
        let e = replay(&a).unwrap_err();
        assert!(format!("{e:#}").contains("gallery not found"), "{e:#}");
        assert!(!path.exists(), "a missing gallery must not be created");
    }

    #[test]
    #[cfg(windows)]
    #[ignore = "needs models and .assets portraits (gitignored)"]
    fn gallery_flag_sets_identity() {
        use fasec_identity::{Pose, Role};

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let assets = root.join(".assets");
        let tmp = tempfile::tempdir().unwrap();

        // Enrol face-a-1's embedding, computed by the real pipeline, into a DPAPI gallery.
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let mut p =
            Pipeline::from_models(&rt, &root.join("models"), PipelineConfig::default()).unwrap();
        let img = image::open(assets.join("face-a-1.jpg"))
            .unwrap()
            .into_rgb8();
        let frame = Frame::new(0, 0, img.width(), img.height(), img.into_raw()).unwrap();
        let out = p.process(&frame, CameraHealth::Ok).unwrap();
        assert_eq!(out.embeddings.len(), 1, "face-a-1 must show one face");
        let gallery = tmp.path().join("gallery.db");
        let owner = {
            let dpapi = std::sync::Arc::new(fasec_platform::Dpapi::new());
            let mut g = Gallery::open(&gallery, dpapi).unwrap();
            let id = g.add_person("a", Role::Owner).unwrap();
            let e = out.embeddings[0].embedding.as_slice();
            g.add_template(&id, RECOGNISER_MODEL_ID, Pose::Front, 0.9, e)
                .unwrap();
            id.0
        };

        // 12 copies at 2 fps: recognition runs every 1000 ms, so 6 votes per face.
        let go = |face: &str| {
            let dir = tmp.path().join(face);
            std::fs::create_dir(&dir).unwrap();
            for i in 0..12 {
                let src = assets.join(format!("{face}.jpg"));
                std::fs::copy(src, dir.join(format!("{i:02}.jpg"))).unwrap();
            }
            let out = tmp.path().join(format!("{face}.jsonl"));
            let args = RunArgs {
                input: Some(dir),
                ir_input: Vec::new(),
                fps: 2,
                models: root.join("models"),
                ep: Ep::Cpu,
                out: Some(out.clone()),
                with_embeddings: false,
                with_timings: false,
                gallery: Some(gallery.clone()),
            };
            assert_eq!(replay(&args).unwrap(), 12);
            let ids = identities(&std::fs::read_to_string(out).unwrap());
            eprintln!("{face}: {ids:?}");
            ids
        };

        let a = go("face-a-2");
        assert!(
            matches!(&a[11][..], [IdentityState::Known { person, .. }] if *person == owner),
            "{a:?}"
        );
        let b = go("face-b-1");
        assert!(
            b.iter()
                .flatten()
                .all(|s| !matches!(s, IdentityState::Known { .. })),
            "{b:?}"
        );
        assert!(
            matches!(b[11][..], [IdentityState::Stranger { .. }]),
            "{b:?}"
        );
    }

    #[test]
    fn ir_input_replaces_input_and_fps() {
        let parse = |s: &str| Cli::try_parse_from(s.split_whitespace());
        let Cmd::Run(a) = parse("fasec-replay run --ir-input rec").unwrap().cmd;
        assert_eq!((a.ir_input, a.input), (vec![PathBuf::from("rec")], None));
        let Cmd::Run(a) = parse("fasec-replay run --ir-input a b --ep cpu")
            .unwrap()
            .cmd;
        assert_eq!(a.ir_input, [PathBuf::from("a"), PathBuf::from("b")]);
        for bad in [
            "fasec-replay run",
            "fasec-replay run --ir-input rec --input d",
            "fasec-replay run --ir-input rec --fps 5",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    /// The IR fixtures: both cameras report one face at `FACE`, so the IR↔RGB mapping is the
    /// identity, and a lit IR frame shows that face near the emitter.
    const W: u32 = 100;
    const FACE: RectF = RectF {
        x: 0.3,
        y: 0.3,
        w: 0.4,
        h: 0.4,
    };

    fn face() -> Detection {
        let p = |x, y| PointF { x, y };
        Detection {
            bbox: FACE,
            landmarks5: [
                p(0.42, 0.45),
                p(0.58, 0.45),
                p(0.5, 0.55),
                p(0.44, 0.65),
                p(0.56, 0.65),
            ],
            score: 0.9,
        }
    }

    struct OneFace;
    impl FaceDetector for OneFace {
        fn detect(&mut self, _: &Frame) -> Result<Vec<Detection>, VisionError> {
            Ok(vec![face()])
        }
    }
    struct StubEmb;
    impl FaceEmbedder for StubEmb {
        fn embed(&mut self, _: &Frame, _: &Detection) -> Result<Vec<f32>, VisionError> {
            Ok(vec![1.0; 4])
        }
        fn model_id(&self) -> &str {
            "stub"
        }
    }
    struct StubLive;
    impl LivenessModel for StubLive {
        fn score(&mut self, _: &Frame, _: &Detection) -> Result<f32, VisionError> {
            Ok(0.9)
        }
        fn tier(&self) -> LivenessTier {
            LivenessTier::Rgb
        }
    }
    struct StubPose;
    impl PoseEstimator for StubPose {
        fn estimate(&mut self, _: &Frame, _: &Detection) -> Result<HeadPose, VisionError> {
            Ok(HeadPose {
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                confidence: 1.0,
            })
        }
    }

    /// Background 20, face 150, nose `nose`: face mean 150 + 0.04 · (nose − 150), ring mean 20,
    /// nose ÷ cheeks `nose` ÷ 150.
    fn lit_face(seq: u64, nose: u8) -> IrFrame {
        let data = (0..W * W)
            .map(|i| match (i % W, i / W) {
                (46..54, 51..59) => nose,
                (30..70, 30..70) => 150,
                _ => 20,
            })
            .collect();
        IrFrame::new(seq, 40_000 + seq * 64, W, W, true, data).unwrap()
    }

    fn unlit(seq: u64, level: u8) -> IrFrame {
        let data = vec![level; (W * W) as usize];
        IrFrame::new(seq, 40_000 + seq * 64, W, W, false, data).unwrap()
    }

    #[test]
    fn ir_input_writes_every_cue_per_track() {
        let root = tempfile::tempdir().unwrap();
        let mut rec = Recorder::create(root.path(), "real-30cm", std::time::UNIX_EPOCH).unwrap();
        // steps 0–4 build trust, 5 has a usable ambient frame, 6 a blanked one, 7–9 a flat face
        for i in 0..10u64 {
            let seq = 2 * i + 1;
            let lit = lit_face(seq, if i >= 7 { 150 } else { 130 });
            let ambient = match i {
                5 => Some(unlit(seq - 1, 10)),
                6 => Some(unlit(seq - 1, 0)),
                _ => None,
            };
            let rgb = Frame::new(100 + i, 5000 + i * 130, W, W, vec![0; 30_000]).unwrap();
            assert!(rec.write(&rgb, &lit, ambient.as_ref()).unwrap());
        }
        let frames = input::open_recordings(&[rec.dir().to_owned()]).unwrap();
        let (det, emb, live, pose) = (OneFace, StubEmb, StubLive, StubPose);
        let cfg = PipelineConfig::default();
        let mut p = Pipeline::new(
            Box::new(det),
            Box::new(emb),
            Box::new(live),
            Box::new(pose),
            cfg,
        );
        let mut probe = IrProbe::new(Box::new(OneFace), cfg.ir);
        let mut out = Vec::new();
        let opts = OutputOptions::default();
        let n = run(frames, &mut p, Some(&mut probe), 15, opts, &mut out, None).unwrap();
        assert_eq!(n, 10);
        let lines: Vec<serde_json::Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        for (i, l) in (0u64..).zip(&lines) {
            // the recorded RGB time, not one simulated from --fps
            assert_eq!(l["ts_ms"], 5000 + i * 130);
            let ir = &l["ir"];
            assert_eq!((&ir["n"], &ir["ir_seq"]), (&i.into(), &(2 * i + 1).into()));
            assert_eq!((&ir["ir_w"], &ir["ir_faces"]), (&W.into(), &1.into()));
            assert_eq!(ir["trusted"], i >= 4, "{i}");
        }
        let track = |i: usize| &lines[i]["ir"]["tracks"][0];
        let num = |v: &serde_json::Value| v.as_f64().unwrap_or(f64::NAN);
        let close = |v: &serde_json::Value, want: f64| (num(v) - want).abs() < 1e-3;

        // before trust nothing is measured
        assert_eq!(track(0)["verdict"]["kind"], "Unknown");
        assert_eq!(track(0)["evidence"]["kind"], "Absent");
        assert!(track(0)["cues"].is_null() && track(0)["ir_face"].is_null());
        // trusted: every cue, the per-frame verdict and the pipeline's evidence
        let t = track(4);
        assert_eq!(lines[4]["ir"]["ambient"], "missing");
        assert_eq!(
            t["verdict"],
            serde_json::json!({"kind": "Live", "score": 0.8, "reason": null})
        );
        assert_eq!(
            (&t["evidence"]["kind"], &t["tier"]),
            (&"Live".into(), &"Ir".into())
        );
        assert_eq!(
            t["ir_face"],
            serde_json::json!({"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.4})
        );
        let c = &t["cues"];
        assert_eq!(
            (&c["face_present"], &c["glints"]),
            (&true.into(), &0.into())
        );
        assert!(
            close(&c["lit_face"], 149.2) && close(&c["lit_background"], 20.0),
            "{c}"
        );
        assert!(
            close(&c["face_vs_background"], 7.46) && close(&c["relief"], 130.0 / 150.0),
            "{c}"
        );
        // no ambient frame: these are null, not numbers
        assert!(c["lit_minus_ambient"].is_null() && c["ambient_background"].is_null());

        assert_eq!(lines[5]["ir"]["ambient"], "usable");
        let c = &track(5)["cues"];
        assert!(close(&c["lit_minus_ambient"], 139.2), "{c}");
        assert!(close(&c["ambient_background"], 10.0), "{c}");
        assert_eq!(lines[6]["ir"]["ambient"], "blank");
        assert!(track(6)["cues"]["lit_minus_ambient"].is_null());
        // a flat face: a spoof on every lit frame, the pipeline's evidence after three in a row
        for (i, evidence) in [(7, "Absent"), (8, "Absent"), (9, "Spoof")] {
            assert_eq!(track(i)["verdict"]["reason"], "FlatReflectance", "{i}");
            assert!(close(&track(i)["cues"]["relief"], 1.0), "{i}");
            assert_eq!(track(i)["evidence"]["kind"], evidence, "{i}");
        }
        assert_eq!(track(9)["evidence"]["reason"], "FlatReflectance");
    }

    /// A face on any frame bright at its centre: the RGB frames below and a lit IR face, never
    /// the dark IR frame of a phone.
    struct BrightFace;
    impl FaceDetector for BrightFace {
        fn detect(&mut self, f: &Frame) -> Result<Vec<Detection>, VisionError> {
            let centre = (50 * f.width as usize + 50) * 3;
            Ok(if f.data[centre] > 100 {
                vec![face()]
            } else {
                vec![]
            })
        }
    }

    #[test]
    fn several_recordings_replay_through_one_pipeline() {
        let root = tempfile::tempdir().unwrap();
        // two app runs: each recording starts at the same RGB time and IR seq
        let record = |label: &str, nose: Option<u8>| {
            let mut rec = Recorder::create(root.path(), label, std::time::UNIX_EPOCH).unwrap();
            for i in 0..6u64 {
                let seq = 2 * i + 1;
                let lit = nose.map_or_else(|| unlit_as_lit(seq), |n| lit_face(seq, n));
                let rgb = Frame::new(100 + i, 5000 + i * 130, W, W, vec![200; 30_000]).unwrap();
                assert!(rec.write(&rgb, &lit, None).unwrap());
            }
            rec.dir().to_owned()
        };
        let dirs = [record("real-50cm", Some(130)), record("phone-photo", None)];
        let frames = input::open_recordings(&dirs).unwrap();
        let cfg = PipelineConfig::default();
        let mut p = Pipeline::new(
            Box::new(BrightFace),
            Box::new(StubEmb),
            Box::new(StubLive),
            Box::new(StubPose),
            cfg,
        );
        let mut probe = IrProbe::new(Box::new(BrightFace), cfg.ir);
        let mut out = Vec::new();
        let opts = OutputOptions::default();
        assert_eq!(
            run(frames, &mut p, Some(&mut probe), 15, opts, &mut out, None).unwrap(),
            12
        );
        let lines: Vec<serde_json::Value> = String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let ts: Vec<u64> = lines.iter().map(|l| l["ts_ms"].as_u64().unwrap()).collect();
        // the second recording continues 500 ms after the first
        assert_eq!(ts[..7], [5000, 5130, 5260, 5390, 5520, 5650, 6150]);
        assert_eq!(ts[11], 6150 + 5 * 130);
        let ir = |i: usize| &lines[i]["ir"];
        for (i, l) in lines.iter().enumerate() {
            assert_eq!(l["ir"]["session"], i / 6, "{i}");
            assert_eq!(l["ir"]["n"], i % 6, "{i}");
        }
        assert_eq!(ir(5)["tracks"][0]["evidence"]["kind"], "Live");
        // trust earned on the face carries into the phone recording: no IR face is a spoof,
        // the pipeline's evidence after three lit frames in a row, beating the RGB Live
        for (i, l) in lines.iter().enumerate().skip(6) {
            let ir = &l["ir"];
            assert_eq!((&ir["trusted"], &ir["ir_faces"]), (&true.into(), &0.into()));
            let t = &ir["tracks"][0];
            assert_eq!(t["verdict"]["reason"], "NoIrFace", "{i}");
            let want = if i < 8 { "Absent" } else { "Spoof" };
            assert_eq!(t["evidence"]["kind"], want, "{i}");
            let live = &l["tracks"][0]["liveness"];
            let want = if i < 8 { "Live" } else { "Spoof" };
            assert!(live.get(want).is_some(), "{i}: {live}");
        }
    }

    /// Task 7's exit criteria on the owner's recordings in `%LOCALAPPDATA%\fasec\recordings`
    /// (`<label>-<yyyymmdd-hhmmss>` from `ir_debug`), per label, with the numbers printed:
    /// - `real-*`, each replayed alone: Live on the IR tier on ≥ 95 % of the tracks on frames
    ///   whose IR mapping is trusted (from the fifth lit frame on);
    /// - `phone-*`, each replayed after each `real-*` through one pipeline: Spoof on ≥ 99 % of
    ///   its tracks once the mapping is trusted and the spoof streak can be met (from its
    ///   `spoof_frames`-th trusted frame with a face on).
    #[test]
    #[ignore = "needs local recordings"]
    fn exit_criteria_on_the_owner_recordings() {
        use fasec_replay::recording::{FRAMES_FILE, recordings_root};
        type Line = serde_json::Value;
        let root = recordings_root().expect("LOCALAPPDATA");
        let mut dirs: Vec<(String, PathBuf)> = std::fs::read_dir(&root)
            .into_iter()
            .flatten()
            .filter_map(|e| {
                let path = e.ok()?.path();
                let name = path.file_name()?.to_str()?;
                let label = name
                    .get(..name.len().checked_sub("-yyyymmdd-hhmmss".len())?)?
                    .to_owned();
                let steps = std::fs::metadata(path.join(FRAMES_FILE)).is_ok_and(|m| m.len() > 0);
                steps.then_some((label, path))
            })
            .collect();
        dirs.sort();
        let of = |prefix: &str| -> Vec<(String, PathBuf)> {
            let mut v = dirs.clone();
            v.retain(|(l, _)| l.starts_with(prefix));
            v
        };
        // The owner's face is in view beside the phone in this take (an IR face with live cues at
        // the real-50cm box on its first 5 steps, a second large face later); re-recorded as
        // phone-photo-20261122-103933.
        const NOT_PHONE_ONLY: [&str; 1] = ["phone-photo-20261122-103919"];
        let (real, mut phone) = (of("real-"), of("phone-"));
        phone.retain(|(_, p)| !NOT_PHONE_ONLY.iter().any(|n| p.ends_with(n)));
        assert!(
            !real.is_empty() && !phone.is_empty(),
            "no real-* and phone-* recordings in {}",
            root.display()
        );

        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let cfg = PipelineConfig::default();
        let replay = |dirs: &[PathBuf]| -> Vec<Line> {
            let mut p = Pipeline::from_models(&rt, &models, cfg).unwrap();
            let mut probe = IrProbe::from_models(&rt, &models, cfg.ir).unwrap();
            let frames = input::open_recordings(dirs).unwrap();
            let mut out = Vec::new();
            let opts = OutputOptions::default();
            run(frames, &mut p, Some(&mut probe), 15, opts, &mut out, None).unwrap();
            let text = String::from_utf8(out).unwrap();
            text.lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        };
        let trusted = |l: &&Line| l["ir"]["trusted"] == true && l["tracks"][0].is_object();
        let tracks = |l: &Line| l["tracks"].as_array().unwrap().clone();
        let pct = |k: usize, n: usize| 100.0 * k as f64 / n as f64;

        // label -> (tracks, Live on the IR tier)
        let mut live: std::collections::BTreeMap<String, (usize, usize)> = Default::default();
        for (label, dir) in &real {
            let lines = replay(std::slice::from_ref(dir));
            let t: Vec<Line> = lines.iter().filter(trusted).flat_map(tracks).collect();
            let ok = t
                .iter()
                .filter(|t| t["liveness"]["Live"]["tier"] == "Ir")
                .count();
            eprintln!(
                "{}: {} lines, {} trusted tracks, {ok} Live/Ir",
                dir.file_name().unwrap().display(),
                lines.len(),
                t.len()
            );
            let e = live.entry(label.clone()).or_default();
            (e.0, e.1) = (e.0 + t.len(), e.1 + ok);
        }
        // label -> (tracks, Spoof, of which by IR evidence)
        let mut spoof: std::collections::BTreeMap<String, (usize, usize, usize)> =
            Default::default();
        let skip = usize::from(cfg.ir.spoof_frames) - 1;
        for (_, r) in &real {
            for (label, p) in &phone {
                let lines = replay(&[r.clone(), p.clone()]);
                let t: Vec<(Line, Line)> = lines
                    .iter()
                    .filter(|l| l["ir"]["session"] == 1)
                    .filter(trusted)
                    .skip(skip)
                    .flat_map(|l| {
                        let ir = l["ir"]["tracks"].as_array().unwrap().clone();
                        tracks(l).into_iter().zip(ir)
                    })
                    .collect();
                let s = t.iter().filter(|(t, _)| t["liveness"]["Spoof"].is_object());
                let by_ir = s.clone().filter(|(_, i)| i["evidence"]["kind"] == "Spoof");
                let (s, by_ir) = (s.count(), by_ir.count());
                eprintln!(
                    "{} -> {}: {} tracks, {s} Spoof ({by_ir} by IR)",
                    r.file_name().unwrap().display(),
                    p.file_name().unwrap().display(),
                    t.len()
                );
                let e = spoof.entry(label.clone()).or_default();
                (e.0, e.1, e.2) = (e.0 + t.len(), e.1 + s, e.2 + by_ir);
            }
        }
        for (label, (n, ok)) in &live {
            eprintln!("{label}: Live/Ir {ok} of {n} = {:.1} %", pct(*ok, *n));
        }
        for (label, (n, s, by_ir)) in &spoof {
            eprintln!(
                "{label}: Spoof {s} of {n} = {:.1} % (by IR {by_ir} = {:.1} %)",
                pct(*s, *n),
                pct(*by_ir, *n)
            );
        }
        for (label, (n, ok)) in &live {
            assert!(*n > 0 && pct(*ok, *n) >= 95.0, "{label}: {ok} of {n}");
        }
        for (label, (n, s, _)) in &spoof {
            assert!(*n > 0 && pct(*s, *n) >= 99.0, "{label}: {s} of {n}");
        }
    }

    /// What IR sees of a phone: a dark lit frame with no face.
    fn unlit_as_lit(seq: u64) -> IrFrame {
        IrFrame::new(
            seq,
            40_000 + seq * 64,
            W,
            W,
            true,
            vec![20; (W * W) as usize],
        )
        .unwrap()
    }
}
