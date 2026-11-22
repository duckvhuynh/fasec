//! JSONL output: one `VisionReport` per line, plus the scheduler's `desired_interval_ms`;
//! embeddings, when asked for, in a separate file.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use fasec_core::{StageTimings, TrackEmbedding, VisionReport};
use serde::Serialize;

/// What to keep in each line beyond the deterministic core.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputOptions {
    /// Keep wall-clock stage timings (else zeroed, so output is byte-for-byte reproducible).
    pub with_timings: bool,
}

/// Writes `report` as one JSON line.
pub fn write_line(
    w: &mut dyn Write,
    report: VisionReport,
    desired_interval_ms: u64,
    opts: OutputOptions,
) -> Result<()> {
    #[derive(Serialize)]
    struct Line {
        #[serde(flatten)]
        report: VisionReport,
        desired_interval_ms: u64,
    }
    let mut report = report;
    if !opts.with_timings {
        report.timings = StageTimings {
            detect_ms: 0.0,
            recognize_ms: 0.0,
            liveness_ms: 0.0,
            pose_ms: 0.0,
        };
    }
    let line = Line {
        report,
        desired_interval_ms,
    };
    serde_json::to_writer(&mut *w, &line)?;
    w.write_all(b"\n")?;
    Ok(())
}

/// `<out>.embeddings.jsonl`: where `--with-embeddings` writes.
pub fn embeddings_path(out: &Path) -> PathBuf {
    let mut p = out.as_os_str().to_owned();
    p.push(".embeddings.jsonl");
    p.into()
}

/// Writes each embedding of frame `seq` as `{"seq":..,"track":..,"embedding":[..]}`.
pub fn write_embeddings(w: &mut dyn Write, seq: u64, embeddings: &[TrackEmbedding]) -> Result<()> {
    #[derive(Serialize)]
    struct Line<'a> {
        seq: u64,
        track: u64,
        embedding: &'a [f32],
    }
    for e in embeddings {
        let line = Line {
            seq,
            track: e.track.0,
            embedding: e.embedding.as_slice(),
        };
        serde_json::to_writer(&mut *w, &line)?;
        w.write_all(b"\n")?;
    }
    Ok(())
}
