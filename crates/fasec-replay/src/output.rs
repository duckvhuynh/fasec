//! JSONL output: one `VisionReport` per line, plus the scheduler's `desired_interval_ms` and,
//! when replaying a recording, an `ir` object with every IR cue per track; embeddings, when
//! asked for, in a separate file.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::Result;
use fasec_core::{LivenessTier, RectF, StageTimings, TrackEmbedding, VisionReport};
use fasec_replay::probe::{Ambient, ProbeFrame};
use fasec_replay::recording::Step;
use fasec_vision::ir::{IrSpoofReason, IrVerdict};
use fasec_vision::{IrEvidence, TrackIr};
use serde::Serialize;

/// What to keep in each line beyond the deterministic core.
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputOptions {
    /// Keep wall-clock stage timings (else zeroed, so output is byte-for-byte reproducible).
    pub with_timings: bool,
}

/// The IR half of a line replayed from a recording. Cues that could not be measured (NaN) are
/// written as `null`.
#[derive(Debug, Serialize)]
pub struct IrLine {
    /// The recording step: its files are `rgb-<n>.png`, `ir-lit-<n>.png`, `ir-amb-<n>.png`.
    n: u64,
    ir_seq: u64,
    ir_w: u32,
    ir_h: u32,
    /// `missing`, `blank` (near-black, the driver blanks unlit frames) or `usable`.
    ambient: &'static str,
    /// The IR↔RGB mapping was trusted when the tracks were matched.
    trusted: bool,
    /// IR faces found in the lit frame.
    ir_faces: usize,
    tracks: Vec<IrTrack>,
}

#[derive(Debug, Serialize)]
struct IrTrack {
    track: u64,
    /// The probe's verdict on this lit frame alone.
    verdict: Verdict,
    /// The pipeline's IR evidence after its spoof streak and staleness rules, and the tier the
    /// track's fused liveness rests on.
    evidence: Verdict,
    tier: Option<LivenessTier>,
    /// The IR face matched to the track, in IR-normalised coordinates.
    ir_face: Option<RectF>,
    /// `null` when nothing was measured (see `fasec_vision::ir::ir_cues`).
    cues: Option<Cues>,
}

/// `kind` is `Live`, `Spoof`, `Unknown` (a verdict) or `Absent` (no evidence).
#[derive(Debug, Serialize)]
struct Verdict {
    kind: &'static str,
    score: Option<f32>,
    reason: Option<String>,
}

impl Verdict {
    fn new(kind: &'static str, score: Option<f32>, reason: Option<IrSpoofReason>) -> Verdict {
        let reason = reason.map(|r| format!("{r:?}"));
        Verdict {
            kind,
            score,
            reason,
        }
    }

    fn of_verdict(v: IrVerdict) -> Verdict {
        match v {
            IrVerdict::Live { score } => Verdict::new("Live", Some(score), None),
            IrVerdict::Spoof { score, reason } => Verdict::new("Spoof", Some(score), Some(reason)),
            IrVerdict::Unknown => Verdict::new("Unknown", None, None),
        }
    }

    fn of_evidence(e: IrEvidence) -> Verdict {
        match e {
            IrEvidence::Live { score } => Verdict::new("Live", Some(score), None),
            IrEvidence::Spoof { score, reason } => Verdict::new("Spoof", Some(score), Some(reason)),
            IrEvidence::Absent => Verdict::new("Absent", None, None),
        }
    }
}

/// `fasec_vision::ir::IrCues`, field for field.
#[derive(Debug, Serialize)]
struct Cues {
    face_present: bool,
    face_vs_background: f32,
    relief: f32,
    lit_face: f32,
    lit_background: f32,
    lit_minus_ambient: f32,
    ambient_background: f32,
    glints: u8,
}

impl IrLine {
    /// The probe's measurements of `step`, with the pipeline's evidence `ir` for each track.
    pub fn new(step: &Step, f: &ProbeFrame, ir: &[TrackIr]) -> IrLine {
        let tracks = f
            .tracks
            .iter()
            .map(|t| {
                let fused = ir.iter().find(|i| i.track == t.track);
                IrTrack {
                    track: t.track.0,
                    verdict: Verdict::of_verdict(t.verdict),
                    evidence: Verdict::of_evidence(
                        fused.map_or(IrEvidence::Absent, |i| i.evidence),
                    ),
                    tier: fused.map(|i| i.tier),
                    ir_face: t.matched.map(|d| d.bbox),
                    cues: t.cues.map(|c| Cues {
                        face_present: c.face_present,
                        face_vs_background: c.face_vs_background,
                        relief: c.relief,
                        lit_face: c.lit_face,
                        lit_background: c.lit_background,
                        lit_minus_ambient: c.lit_minus_ambient,
                        ambient_background: c.ambient_background,
                        glints: c.glints,
                    }),
                }
            })
            .collect();
        IrLine {
            n: step.n,
            ir_seq: step.ir_seq,
            ir_w: step.ir_w,
            ir_h: step.ir_h,
            ambient: match f.ambient {
                Ambient::Missing => "missing",
                Ambient::Blank => "blank",
                Ambient::Usable => "usable",
            },
            trusted: f.trusted,
            ir_faces: f.faces.len(),
            tracks,
        }
    }
}

/// Writes `report` (and `ir`, when replaying a recording) as one JSON line.
pub fn write_line(
    w: &mut dyn Write,
    report: VisionReport,
    ir: Option<IrLine>,
    desired_interval_ms: u64,
    opts: OutputOptions,
) -> Result<()> {
    #[derive(Serialize)]
    struct Line {
        #[serde(flatten)]
        report: VisionReport,
        desired_interval_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        ir: Option<IrLine>,
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
        ir,
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
