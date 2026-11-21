//! JSONL output: one `VisionReport` per line, plus the scheduler's `desired_interval_ms`.

use std::io::Write;

use anyhow::Result;
use fasec_core::{StageTimings, VisionReport};
use serde::Serialize;

/// What to keep in each line beyond the deterministic core.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputOptions {
    /// Keep `TrackObs.embedding` (else written as `null`).
    pub with_embeddings: bool,
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
    if !opts.with_embeddings {
        for t in &mut report.tracks {
            t.embedding = None;
        }
    }
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
