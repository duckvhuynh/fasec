//! Offline replay CLI: run the vision pipeline on recorded frames or video with a simulated
//! clock and write one `VisionReport` per line (JSONL). Embeddings, when asked for, go to a
//! separate `<out>.embeddings.jsonl` file, never into the report lines.

mod input;
mod output;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fasec_config::ExecutionProvider;
use fasec_core::{CameraHealth, Frame};
use fasec_vision::{Pipeline, PipelineConfig, ProcessOutput, Runtime};

use crate::input::{Ffmpeg, RawFrame};
use crate::output::OutputOptions;

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
    #[arg(long)]
    input: PathBuf,
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
    fn process(&mut self, frame: &Frame) -> Result<ProcessOutput>;
    fn desired_interval_ms(&self) -> u64;
}

impl Process for Pipeline {
    fn process(&mut self, frame: &Frame) -> Result<ProcessOutput> {
        Ok(Pipeline::process(self, frame, CameraHealth::Ok)?)
    }
    fn desired_interval_ms(&self) -> u64 {
        Pipeline::desired_interval_ms(self)
    }
}

/// Feeds every frame to `p` (no skipping) and writes one line each to `w`, plus each embedding
/// computed on it to `emb` if given; returns the frame count.
fn run(
    frames: impl Iterator<Item = Result<RawFrame>>,
    p: &mut dyn Process,
    fps: u32,
    opts: OutputOptions,
    w: &mut dyn Write,
    mut emb: Option<&mut dyn Write>,
) -> Result<u64> {
    anyhow::ensure!(fps > 0, "fps must be at least 1");
    let mut n = 0;
    for (seq, raw) in (0u64..).zip(frames) {
        let raw = raw.with_context(|| format!("frame {seq}"))?;
        let ts_ms = seq * 1000 / u64::from(fps);
        let frame = Frame::new(seq, ts_ms, raw.width, raw.height, raw.data)?;
        let out = p.process(&frame).with_context(|| format!("frame {seq}"))?;
        if let Some(e) = emb.as_deref_mut() {
            output::write_embeddings(e, out.report.seq, &out.embeddings)?;
        }
        output::write_line(w, out.report, p.desired_interval_ms(), opts)?;
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
    // Opened first so a bad input fails before the models load.
    let frames = input::open(&args.input, &Ffmpeg::default())?;
    let rt = Runtime::new(args.ep.into())?;
    let mut p = Pipeline::from_models(&rt, &args.models, PipelineConfig::default())
        .with_context(|| format!("loading models from {}", args.models.display()))?;
    write_outputs(args, frames, &mut p)
}

/// Creates the output file(s) `args` asks for and runs `p` over `frames` into them.
fn write_outputs(
    args: &RunArgs,
    frames: impl Iterator<Item = Result<RawFrame>>,
    p: &mut dyn Process,
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
        return run(frames, p, args.fps, opts, &mut w, None);
    };
    let mut emb = if args.with_embeddings {
        Some(create(&output::embeddings_path(path))?)
    } else {
        None
    };
    let emb = emb.as_mut().map(|w| w as &mut dyn Write);
    run(frames, p, args.fps, opts, &mut create(path)?, emb)
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
        IdentityState, LivenessState, LivenessTier, PointF, RectF, StageTimings, TrackEmbedding,
        TrackId, TrackObs, VisionReport,
    };

    use super::*;

    /// Records every frame it sees and returns one track with an embedding.
    #[derive(Default)]
    struct Mock {
        seen: Vec<(u64, u64, u32, u32, u8)>,
    }

    impl Process for Mock {
        fn process(&mut self, f: &Frame) -> Result<ProcessOutput> {
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
        let n = run(frames, &mut mock, 4, opts, &mut out, None).unwrap();
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
        run(frames(), &mut Mock::default(), 10, opts, &mut out, None).unwrap();
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
        run(frames(), &mut Mock::default(), 10, opts, &mut out, None).unwrap();
        let first: serde_json::Value =
            serde_json::from_str(String::from_utf8(out).unwrap().lines().next().unwrap()).unwrap();
        assert_eq!(first["timings"]["detect_ms"], 1.5);
    }

    fn args(out: Option<PathBuf>, with_embeddings: bool) -> RunArgs {
        RunArgs {
            input: PathBuf::from("does-not-exist"),
            fps: 10,
            models: PathBuf::from("does-not-exist"),
            ep: Ep::Cpu,
            out,
            with_embeddings,
            with_timings: false,
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
        )
        .unwrap();
        assert!(!side.exists(), "sidecar written without --with-embeddings");

        write_outputs(
            &args(Some(out.clone()), true),
            frames(),
            &mut Mock::default(),
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
    fn with_embeddings_requires_out() {
        // Refused before the input is opened or any model loads.
        let e = replay(&args(None, true)).unwrap_err();
        assert!(format!("{e:#}").contains("--out"), "{e:#}");
        let e = write_outputs(&args(None, true), std::iter::empty(), &mut Mock::default());
        assert!(format!("{:#}", e.unwrap_err()).contains("--out"));
    }

    #[test]
    fn frame_error_stops_run() {
        let frames = vec![raw(2, 2, 0), Err(anyhow::anyhow!("decode failed"))].into_iter();
        let mut out = Vec::new();
        let err = run(
            frames,
            &mut Mock::default(),
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
                input: root.join(".assets"),
                fps: 15,
                models: root.join("models"),
                ep: Ep::Cpu,
                out: Some(out.clone()),
                with_embeddings: true,
                with_timings: false,
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
}
