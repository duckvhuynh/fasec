//! Offline replay CLI: run the vision pipeline on recorded frames or video with a simulated
//! clock and write one `VisionReport` per line (JSONL).

mod input;
mod output;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use fasec_config::ExecutionProvider;
use fasec_core::{CameraHealth, Frame, VisionReport};
use fasec_vision::{Pipeline, PipelineConfig, Runtime};

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
    /// Directory holding the model files and `manifest.toml`.
    #[arg(long, default_value = "models")]
    models: PathBuf,
    #[arg(long, value_enum, default_value_t = Ep::Auto)]
    ep: Ep,
    /// Output file (default: stdout).
    #[arg(long)]
    out: Option<PathBuf>,
    /// Include face embeddings in the output.
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
    fn process(&mut self, frame: &Frame) -> Result<VisionReport>;
    fn desired_interval_ms(&self) -> u64;
}

impl Process for Pipeline {
    fn process(&mut self, frame: &Frame) -> Result<VisionReport> {
        Ok(Pipeline::process(self, frame, CameraHealth::Ok)?)
    }
    fn desired_interval_ms(&self) -> u64 {
        Pipeline::desired_interval_ms(self)
    }
}

/// Feeds every frame to `p` (no skipping) and writes one line each; returns the frame count.
fn run(
    frames: impl Iterator<Item = Result<RawFrame>>,
    p: &mut dyn Process,
    fps: u32,
    opts: OutputOptions,
    w: &mut dyn Write,
) -> Result<u64> {
    anyhow::ensure!(fps > 0, "fps must be at least 1");
    let mut n = 0;
    for (seq, raw) in (0u64..).zip(frames) {
        let raw = raw.with_context(|| format!("frame {seq}"))?;
        let ts_ms = seq * 1000 / u64::from(fps);
        let frame = Frame::new(seq, ts_ms, raw.width, raw.height, raw.data)?;
        let report = p.process(&frame).with_context(|| format!("frame {seq}"))?;
        output::write_line(w, report, p.desired_interval_ms(), opts)?;
        n += 1;
    }
    w.flush()?;
    Ok(n)
}

fn replay(args: &RunArgs) -> Result<u64> {
    // Opened first so a bad input fails before the models load.
    let frames = input::open(&args.input, &Ffmpeg::default())?;
    let rt = Runtime::new(args.ep.into())?;
    let mut p = Pipeline::from_models(&rt, &args.models, PipelineConfig::default())
        .with_context(|| format!("loading models from {}", args.models.display()))?;
    let opts = OutputOptions {
        with_embeddings: args.with_embeddings,
        with_timings: args.with_timings,
    };
    match &args.out {
        Some(path) => {
            let f = File::create(path).with_context(|| format!("creating {}", path.display()))?;
            run(frames, &mut p, args.fps, opts, &mut BufWriter::new(f))
        }
        None => run(
            frames,
            &mut p,
            args.fps,
            opts,
            &mut std::io::stdout().lock(),
        ),
    }
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
        IdentityState, LivenessState, LivenessTier, PointF, RectF, StageTimings, TrackId, TrackObs,
    };

    use std::path::Path;

    use super::*;

    /// Records every frame it sees and returns one track with an embedding.
    #[derive(Default)]
    struct Mock {
        seen: Vec<(u64, u64, u32, u32, u8)>,
    }

    impl Process for Mock {
        fn process(&mut self, f: &Frame) -> Result<VisionReport> {
            self.seen
                .push((f.seq, f.ts_ms, f.width, f.height, f.data[0]));
            let p = PointF { x: 0.5, y: 0.5 };
            Ok(VisionReport {
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
                    embedding: Some(vec![0.25, -0.5]),
                }],
                camera_health: CameraHealth::Ok,
                liveness_tier: LivenessTier::Rgb,
                timings: StageTimings {
                    detect_ms: 1.5,
                    recognize_ms: 2.5,
                    liveness_ms: 3.5,
                    pose_ms: 4.5,
                },
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
        let n = run(frames, &mut mock, 4, OutputOptions::default(), &mut out).unwrap();
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
        run(frames(), &mut Mock::default(), 10, opts, &mut out).unwrap();
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
            assert!(v["tracks"][0]["embedding"].is_null());
            assert_eq!(v["timings"]["detect_ms"], 0.0);
            // The rest of the line is still a `VisionReport`.
            let r: VisionReport = serde_json::from_value(v.clone()).unwrap();
            assert_eq!(r.tracks.len(), 1);
        }

        let mut out = Vec::new();
        let opts = OutputOptions {
            with_embeddings: true,
            with_timings: true,
        };
        run(frames(), &mut Mock::default(), 10, opts, &mut out).unwrap();
        let first: serde_json::Value =
            serde_json::from_str(String::from_utf8(out).unwrap().lines().next().unwrap()).unwrap();
        assert_eq!(
            first["tracks"][0]["embedding"],
            serde_json::json!([0.25, -0.5])
        );
        assert_eq!(first["timings"]["detect_ms"], 1.5);
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
            std::fs::read(out).unwrap()
        };
        let a = go("a.jsonl");
        assert_eq!(a, go("b.jsonl"));
        let text = String::from_utf8(a).unwrap();
        assert!(text.contains("\"embedding\":["), "no face found: {text}");
    }
}
