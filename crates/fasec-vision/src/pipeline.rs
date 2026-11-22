//! One frame in, one [`ProcessOutput`] out: detect + track every frame, attention every frame,
//! embedding and liveness on a per-track schedule (SRS §3.3).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use fasec_core::{
    CameraHealth, Frame, IdentityState, IrFrame, LivenessState, LivenessTier, PointF, RectF,
    StageTimings, TrackEmbedding, TrackId, TrackObs, VisionReport,
};
use fasec_models::{Manifest, VerifiedModel};
use zeroize::Zeroize;

use crate::attention::{CoarsePose, face_width_px};
use crate::ir::{
    IrAssociator, IrLivenessConfig, IrSpoofReason, IrVerdict, ir_cues, ir_faces, judge,
};
use crate::liveness::MiniFasNet;
use crate::recognize::SFace;
use crate::track::{Tracker, TrackerConfig};
use crate::{
    FaceDetector, FaceEmbedder, LivenessModel, PoseEstimator, Runtime, VisionError, YuNet,
    YuNetConfig,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PipelineConfig {
    /// Re-embed a track at most this often unless its box jumps (1000 ms).
    pub recog_interval_ms: u64,
    /// Re-score liveness at most this often unless the box jumps (2000 ms).
    pub liveness_interval_ms: u64,
    /// Frame rate asked for while no face is tracked (5).
    pub idle_fps: u32,
    /// Frame rate asked for while at least one face is tracked (15).
    pub active_fps: u32,
    /// IoU between the current box and the box at the last run below which the track is
    /// re-recognised immediately (0.5).
    pub rerecog_iou: f32,
    /// Liveness score at or above which a track is `Live` (0.75, Balanced, SRS §5.5).
    pub liveness_threshold: f32,
    pub tracker: TrackerConfig,
    pub ir: IrLivenessConfig,
    /// IR evidence counts for this long of RGB time after the call that took its lit frame
    /// (500 ms: about four lit frames on the ASUS IR camera, ~128 ms lit period).
    pub ir_max_age_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            recog_interval_ms: 1000,
            liveness_interval_ms: 2000,
            idle_fps: 5,
            active_fps: 15,
            rerecog_iou: 0.5,
            liveness_threshold: 0.75,
            tracker: TrackerConfig::default(),
            ir: IrLivenessConfig::default(),
            ir_max_age_ms: 500,
        }
    }
}

/// One processed frame. Embeddings are biometric templates, so they travel beside the
/// serialisable report, never inside it (NFR-PRIV-003).
#[derive(Clone, Debug, PartialEq)]
pub struct ProcessOutput {
    pub report: VisionReport,
    /// Only the embeddings computed on this frame (none carried over), so each is seen once.
    pub embeddings: Vec<TrackEmbedding>,
    /// One entry per reported track, in report order: the IR evidence its liveness was fused
    /// with. Empty when the RGB camera is not healthy.
    pub ir: Vec<TrackIr>,
}

/// IR evidence for one RGB track on one frame, after the spoof streak and staleness rules.
/// `Live` already implies IR healthy, the mapping trusted and the track matched to an IR face.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IrEvidence {
    /// IR unhealthy, absent or stale, an `Unknown` verdict, or a spoof streak still short of
    /// `IrLivenessConfig::spoof_frames`.
    Absent,
    Live {
        score: f32,
    },
    Spoof {
        score: f32,
        reason: IrSpoofReason,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackIr {
    pub track: TrackId,
    pub evidence: IrEvidence,
    /// The tier the track's fused liveness rests on.
    pub tier: LivenessTier,
    /// The cues behind the last IR verdict measured lit − ambient (a usable unlit frame).
    /// `false` where the driver blanks unlit frames: IR `Live` then rests on the lit frame alone.
    pub ambient: bool,
}

/// Detector, tracker and per-track stages, plus each track's carried results.
pub struct Pipeline {
    det: Box<dyn FaceDetector>,
    emb: Box<dyn FaceEmbedder>,
    live: Box<dyn LivenessModel>,
    pose: Box<dyn PoseEstimator>,
    cfg: PipelineConfig,
    tracker: Tracker,
    tracks: HashMap<TrackId, TrackState>,
    /// The last report had at least one track.
    active: bool,
    /// Learned IR↔RGB mapping; replaced by a fresh one whenever IR stops being healthy or the
    /// IR frame size changes, so trust is never carried across an IR restart.
    ir_assoc: IrAssociator,
    /// Number of lit frames used so far: the spoof streak counts lit frames, not calls.
    lit_index: u64,
    /// The last lit frame used (since the last IR reset).
    last_lit: Option<LitSeen>,
}

#[derive(Clone, Copy)]
struct LitSeen {
    seq: u64,
    /// RGB `ts_ms` of the call that used it: IR age is measured on the RGB clock only.
    at_ms: u64,
    size: (u32, u32),
}

/// A track's last IR verdict and the `lit_index` of the lit frame that gave it.
#[derive(Clone, Copy)]
struct IrSeen {
    lit: u64,
    verdict: IrVerdict,
    ambient: bool,
}

/// What a track carries between scheduled runs.
struct TrackState {
    last_seen_ms: u64,
    /// `ts_ms` and box of the last successful embed.
    embedded_at: Option<(u64, RectF)>,
    liveness: LivenessState,
    /// `ts_ms` and box of the last successful liveness score.
    scored_at: Option<(u64, RectF)>,
    ir: Option<IrSeen>,
    /// Consecutive lit frames with an IR `Spoof` verdict.
    ir_spoof_streak: u8,
}

impl Pipeline {
    pub fn new(
        det: Box<dyn FaceDetector>,
        emb: Box<dyn FaceEmbedder>,
        live: Box<dyn LivenessModel>,
        pose: Box<dyn PoseEstimator>,
        cfg: PipelineConfig,
    ) -> Pipeline {
        Pipeline {
            det,
            emb,
            live,
            pose,
            tracker: Tracker::new(cfg.tracker),
            cfg,
            tracks: HashMap::new(),
            active: false,
            ir_assoc: IrAssociator::new(),
            lit_index: 0,
            last_lit: None,
        }
    }

    /// YuNet (640 px long side) + SFace + MiniFASNet + CoarsePose from `models_dir`. Every
    /// file is read once and SHA-256 verified against the manifest embedded in the binary
    /// (`models_dir/manifest.toml` is ignored) before any session is created, and sessions are
    /// built from those verified bytes; a missing or mismatched file is an error naming the id.
    pub fn from_models(
        rt: &Runtime,
        models_dir: &Path,
        cfg: PipelineConfig,
    ) -> Result<Pipeline, VisionError> {
        let manifest = Manifest::embedded();
        let verified = |id: &str| {
            let bad = |e: &dyn std::fmt::Display| VisionError::Model(format!("{id}: {e}"));
            let entry = manifest.get(id).ok_or_else(|| bad(&"not in manifest"))?;
            VerifiedModel::load(models_dir, entry).map_err(|e| bad(&e))
        };
        // All four verified before the first session is created.
        let yunet = verified("yunet-2023mar")?;
        let sface = verified(crate::RECOGNISER_MODEL_ID)?;
        let v2 = verified("minifasnet-v2")?;
        let v1se = verified("minifasnet-v1se")?;
        Ok(Pipeline::new(
            Box::new(YuNet::new(rt, &yunet, YuNetConfig::default())?),
            Box::new(SFace::new(rt, &sface)?),
            Box::new(MiniFasNet::new(rt, &v2, &v1se)?),
            Box::new(CoarsePose),
            cfg,
        ))
    }

    /// RGB only: [`Pipeline::process_with_ir`] with no IR frame and IR `Lost`, so a pipeline
    /// that had IR drops its IR trust.
    pub fn process(
        &mut self,
        frame: &Frame,
        health: CameraHealth,
    ) -> Result<ProcessOutput, VisionError> {
        self.process_with_ir(frame, None, health, CameraHealth::Lost)
    }

    /// Runs the stages due on `frame`. A detector error fails the frame; an error of a
    /// per-track stage only leaves that track's field `None`/`Unknown` (retried next frame).
    /// With `health != Ok` no model runs and the report has no tracks.
    ///
    /// `ir` is the lit IR frame newly taken for this call and its ambient (unlit) partner, if
    /// any. It is used only when `ir_health` is `Ok`, the frame is lit and its seq is newer than
    /// the last one used. IR cues then run for every track, and each track's liveness is its RGB
    /// verdict fused with its IR evidence (`fuse`). A track's IR evidence counts only if it came
    /// from the latest lit frame used and that frame was used at most `ir_max_age_ms` of RGB time
    /// (`frame.ts_ms`) ago, so a stalled IR camera falls back to the RGB tier. The spoof streak
    /// counts consecutive lit frames in which the track was judged, however many calls lie
    /// between them. IR timestamps are on the IR camera's clock and are never compared with
    /// `frame.ts_ms`.
    ///
    /// Caller contract: pass each newly taken lit pair once, with the RGB frame closest to it;
    /// calls without a lit frame between them are fine. A restarted IR capture must be signalled
    /// by `ir_health != Ok` for at least one call; otherwise its seq may go backwards and IR is
    /// ignored until it passes the old maximum.
    pub fn process_with_ir(
        &mut self,
        frame: &Frame,
        ir: Option<(&IrFrame, Option<&IrFrame>)>,
        health: CameraHealth,
        ir_health: CameraHealth,
    ) -> Result<ProcessOutput, VisionError> {
        // before the frame check: a restart signalled with a bad frame must not be lost
        let ir_ok = ir_health == CameraHealth::Ok;
        if !ir_ok {
            self.reset_ir();
        }
        crate::image_ops::check_frame(frame)?;
        let mut timings = StageTimings {
            detect_ms: 0.0,
            recognize_ms: 0.0,
            liveness_ms: 0.0,
            pose_ms: 0.0,
        };
        let mut out = ProcessOutput {
            report: VisionReport {
                seq: frame.seq,
                ts_ms: frame.ts_ms,
                tracks: Vec::new(),
                camera_health: health,
                liveness_tier: self.live.tier(),
                timings,
            },
            embeddings: Vec::new(),
            ir: Vec::new(),
        };
        if out.report.camera_health != CameraHealth::Ok {
            self.active = false;
            return Ok(out);
        }

        let t0 = Instant::now();
        let mut dets = self.det.detect(frame)?;
        timings.detect_ms = ms(t0);
        // Non-finite coordinates would poison the tracker and the report.
        let n_det = dets.len();
        dets.retain(|d| {
            let b = d.bbox;
            let lm = d.landmarks5.iter().flat_map(|p| [p.x, p.y]);
            [b.x, b.y, b.w, b.h]
                .into_iter()
                .chain(lm)
                .all(f32::is_finite)
        });
        if dets.len() < n_det {
            tracing::warn!(
                dropped = n_det - dets.len(),
                "dropped detections with non-finite coordinates"
            );
        }

        let now = frame.ts_ms;
        let cfg = self.cfg;
        // Tracker ids are never reused, so state of tracks it has dropped can go.
        self.tracks
            .retain(|_, s| now.saturating_sub(s.last_seen_ms) <= cfg.tracker.track_ttl_ms);
        for t in self.tracker.update(now, &dets) {
            let d = &t.det;
            let st = self.tracks.entry(t.id).or_insert_with(|| TrackState {
                last_seen_ms: now,
                embedded_at: None,
                liveness: LivenessState::Unknown,
                scored_at: None,
                ir: None,
                ir_spoof_streak: 0,
            });
            st.last_seen_ms = now;
            // A new track has no state yet, so `None` covers `is_new`.
            // A timestamp before the last run (camera restart) makes the stage due: a result
            // from the old timeline cannot be trusted as fresh.
            let due = |at: Option<(u64, RectF)>, interval: u64| match at {
                None => true,
                Some((ts, b)) => {
                    now < ts || now - ts >= interval || d.bbox.iou(&b) < cfg.rerecog_iou
                }
            };

            if due(st.embedded_at, cfg.recog_interval_ms) {
                let t0 = Instant::now();
                let res = self.emb.embed(frame, d).and_then(|mut e| {
                    if e.iter().all(|v| v.is_finite()) {
                        Ok(e)
                    } else {
                        e.zeroize();
                        Err(VisionError::Model("non-finite embedding".into()))
                    }
                });
                match res {
                    Ok(e) => {
                        out.embeddings.push(TrackEmbedding {
                            track: t.id,
                            ts_ms: now,
                            embedding: e.into(),
                        });
                        st.embedded_at = Some((now, d.bbox));
                    }
                    Err(e) => {
                        tracing::warn!(track = t.id.0, error = %e, "embedding failed");
                        st.embedded_at = None;
                    }
                }
                timings.recognize_ms += ms(t0);
            }

            if due(st.scored_at, cfg.liveness_interval_ms) {
                let c = d.bbox.center();
                st.liveness = LivenessState::Unknown;
                st.scored_at = None;
                // A YuNet box can extend past the frame; with its centre outside, the crop
                // would score unrelated edge pixels. A face too large for an in-distribution
                // crop scores as a spoof, so it is not scored at all (`Unknown`, retried).
                if (0.0..=1.0).contains(&c.x)
                    && (0.0..=1.0).contains(&c.y)
                    && crate::liveness_fits(frame.width, frame.height, d.bbox)
                {
                    let t0 = Instant::now();
                    let res = self.live.score(frame, d).and_then(|s| {
                        if s.is_finite() {
                            Ok(s)
                        } else {
                            Err(VisionError::Model(format!("non-finite liveness score {s}")))
                        }
                    });
                    match res {
                        Ok(score) => {
                            let tier = self.live.tier();
                            st.liveness = if score >= cfg.liveness_threshold {
                                LivenessState::Live { score, tier }
                            } else {
                                LivenessState::Spoof { score, tier }
                            };
                            st.scored_at = Some((now, d.bbox));
                        }
                        Err(e) => tracing::warn!(track = t.id.0, error = %e, "liveness failed"),
                    }
                    timings.liveness_ms += ms(t0);
                }
            }

            let t0 = Instant::now();
            let head_pose = self
                .pose
                .estimate(frame, d)
                .inspect_err(|e| tracing::warn!(track = t.id.0, error = %e, "head pose failed"))
                .ok();
            timings.pose_ms += ms(t0);

            out.report.tracks.push(TrackObs {
                id: t.id,
                bbox: d.bbox.clamp_unit(),
                landmarks5: d.landmarks5.map(|p| PointF {
                    x: p.x.clamp(0.0, 1.0),
                    y: p.y.clamp(0.0, 1.0),
                }),
                det_score: d.score,
                identity: IdentityState::Pending,
                // the RGB verdict, fused with IR below
                liveness: st.liveness.clone(),
                head_pose,
                face_width_px: face_width_px(d, frame),
                age_ms: t.age_ms,
            });
        }

        let lit = ir.filter(|(l, _)| ir_ok && self.last_lit.is_none_or(|s| l.seq > s.seq));
        if let Some((lit, ambient)) = lit {
            let t0 = Instant::now();
            self.judge_ir(lit, ambient, now, &out.report.tracks);
            timings.liveness_ms += ms(t0);
        }

        // An RGB timestamp before the lit frame's call (RGB restart) is not fresh either.
        let lit_fresh = self.last_lit.is_some_and(|l| {
            now.checked_sub(l.at_ms)
                .is_some_and(|age| age <= cfg.ir_max_age_ms)
        });
        for obs in &mut out.report.tracks {
            let st = &self.tracks[&obs.id];
            let seen = st.ir.filter(|s| lit_fresh && s.lit == self.lit_index);
            let evidence = match seen.map(|s| s.verdict) {
                Some(IrVerdict::Live { score }) => IrEvidence::Live { score },
                Some(IrVerdict::Spoof { score, reason })
                    if st.ir_spoof_streak >= cfg.ir.spoof_frames =>
                {
                    IrEvidence::Spoof { score, reason }
                }
                _ => IrEvidence::Absent,
            };
            let (liveness, tier) = fuse(&st.liveness, evidence);
            obs.liveness = liveness;
            out.ir.push(TrackIr {
                track: obs.id,
                evidence,
                tier,
                ambient: seen.is_some_and(|s| s.ambient),
            });
        }
        if self.ir_assoc.is_trusted() && lit_fresh {
            out.report.liveness_tier = LivenessTier::Ir;
        }

        out.report.timings = timings;
        self.active = !out.report.tracks.is_empty();
        Ok(out)
    }

    /// Detects faces in `lit`, matches them to `tracks` and records each track's IR verdict.
    fn judge_ir(
        &mut self,
        lit: &IrFrame,
        ambient: Option<&IrFrame>,
        now: u64,
        tracks: &[TrackObs],
    ) {
        let size = (lit.width, lit.height);
        if self.last_lit.is_some_and(|s| s.size != size) {
            self.reset_ir();
        }
        // an unlit frame is an error here, so it never counts as a fresh lit frame
        let faces = match ir_faces(&mut *self.det, lit) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(error = %e, "IR face detection failed");
                return;
            }
        };
        self.lit_index += 1;
        self.last_lit = Some(LitSeen {
            seq: lit.seq,
            at_ms: now,
            size,
        });
        let cfg = self.cfg.ir;
        let matches = self.ir_assoc.associate(tracks, &faces);
        for (obs, (id, matched)) in tracks.iter().zip(matches) {
            let cues = ir_cues(
                lit,
                ambient,
                &self.ir_assoc,
                obs.bbox,
                &faces,
                matched.as_ref(),
            );
            let verdict = cues.map_or(IrVerdict::Unknown, |c| judge(&c, &cfg));
            let Some(st) = self.tracks.get_mut(&id) else {
                continue;
            };
            // the track judged on the previous lit frame too, or else a new streak
            let follows = st.ir.is_some_and(|s| s.lit + 1 == self.lit_index);
            st.ir_spoof_streak = match verdict {
                IrVerdict::Spoof { .. } if follows => st.ir_spoof_streak.saturating_add(1),
                IrVerdict::Spoof { .. } => 1,
                _ => 0,
            };
            st.ir = Some(IrSeen {
                lit: self.lit_index,
                verdict,
                ambient: cues.is_some_and(|c| c.lit_minus_ambient.is_finite()),
            });
        }
    }

    /// Forgets the IR mapping and every track's IR state.
    fn reset_ir(&mut self) {
        self.ir_assoc = IrAssociator::new();
        self.last_lit = None;
        for st in self.tracks.values_mut() {
            st.ir = None;
            st.ir_spoof_streak = 0;
        }
    }

    /// Frame interval to ask the camera for: the idle rate while the last report had no tracks.
    pub fn desired_interval_ms(&self) -> u64 {
        let fps = if self.active {
            self.cfg.active_fps
        } else {
            self.cfg.idle_fps
        };
        1000 / u64::from(fps.max(1))
    }
}

/// The fusion table. Fusion runs per RGB track, so IR alone never makes anything `Live`.
///
/// | IR evidence | RGB | result           |
/// |-------------|-----|------------------|
/// | Spoof       | any | Spoof, tier `Ir` |
/// | Live        | any | Live, tier `Ir`  |
/// | Absent      | any | the RGB verdict  |
///
/// IR `Live` overrides an RGB `Spoof` (owner decision, Task 7): MiniFASNet calls the owner
/// `Spoof` on most frames at laptop distance (real-30cm 15 of 22, real-80cm 17 of 17), while
/// Windows Hello's bar since CVE-2025-26644 is an NIR face plus an RGB camera that sees the same
/// face. With IR absent (untrusted, stale or unhealthy) the RGB verdict, `Spoof` included, still
/// decides. Residual risk: a screen that emits the owner's NIR image as well as a visible face,
/// on hardware without a usable ambient frame (the self-emission cue needs one).
fn fuse(rgb: &LivenessState, ir: IrEvidence) -> (LivenessState, LivenessTier) {
    let tier = LivenessTier::Ir;
    match ir {
        IrEvidence::Spoof { score, .. } => (LivenessState::Spoof { score, tier }, tier),
        IrEvidence::Live { score } => (LivenessState::Live { score, tier }, tier),
        IrEvidence::Absent => match rgb {
            LivenessState::Live { tier, .. } | LivenessState::Spoof { tier, .. } => {
                (rgb.clone(), *tier)
            }
            LivenessState::Unknown => (rgb.clone(), LivenessTier::Rgb),
        },
    }
}

fn ms(since: Instant) -> f32 {
    since.elapsed().as_secs_f32() * 1000.0
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use fasec_config::ExecutionProvider;
    use fasec_core::{HeadPose, LivenessTier};

    use super::*;
    use crate::Detection;

    /// Calls per stage; mocks sleep 1 ms so timings are measurable.
    #[derive(Clone, Default)]
    struct Calls {
        det: Arc<AtomicUsize>,
        emb: Arc<AtomicUsize>,
        live: Arc<AtomicUsize>,
        pose: Arc<AtomicUsize>,
    }

    impl Calls {
        fn get(&self) -> [usize; 4] {
            [&self.det, &self.emb, &self.live, &self.pose].map(|c| c.load(Ordering::Relaxed))
        }
    }

    fn tick(c: &AtomicUsize) {
        c.fetch_add(1, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(1));
    }

    /// Detections the mock detector returns; `None` = detector error.
    type Script = Arc<Mutex<Option<Vec<Detection>>>>;

    struct MockDet(Calls, Script);
    impl FaceDetector for MockDet {
        fn detect(&mut self, _: &Frame) -> Result<Vec<Detection>, VisionError> {
            tick(&self.0.det);
            self.1
                .lock()
                .unwrap()
                .clone()
                .ok_or_else(|| VisionError::Ort("detector down".into()))
        }
    }

    /// Fails on boxes narrower than 0.05 (stands in for a degenerate face).
    struct MockEmb(Calls);
    impl FaceEmbedder for MockEmb {
        fn embed(&mut self, _: &Frame, det: &Detection) -> Result<Vec<f32>, VisionError> {
            tick(&self.0.emb);
            if det.bbox.w < 0.05 {
                return Err(VisionError::Input("degenerate face".into()));
            }
            Ok(vec![det.bbox.x; 4])
        }
        fn model_id(&self) -> &str {
            "mock"
        }
    }

    /// Score = the detection score, so tests choose live or spoof.
    struct MockLive(Calls);
    impl LivenessModel for MockLive {
        fn score(&mut self, _: &Frame, det: &Detection) -> Result<f32, VisionError> {
            tick(&self.0.live);
            Ok(det.score)
        }
        fn tier(&self) -> LivenessTier {
            LivenessTier::Rgb
        }
    }

    struct MockPose(Calls);
    impl PoseEstimator for MockPose {
        fn estimate(&mut self, _: &Frame, det: &Detection) -> Result<HeadPose, VisionError> {
            tick(&self.0.pose);
            Ok(HeadPose {
                yaw: 0.0,
                pitch: 0.0,
                roll: 0.0,
                confidence: det.score,
            })
        }
    }

    struct Harness {
        p: Pipeline,
        calls: Calls,
        script: Script,
    }

    impl Harness {
        fn new() -> Harness {
            let calls = Calls::default();
            let script: Script = Arc::new(Mutex::new(Some(Vec::new())));
            let p = Pipeline::new(
                Box::new(MockDet(calls.clone(), script.clone())),
                Box::new(MockEmb(calls.clone())),
                Box::new(MockLive(calls.clone())),
                Box::new(MockPose(calls.clone())),
                PipelineConfig::default(),
            );
            Harness { p, calls, script }
        }

        fn set(&self, dets: Option<Vec<Detection>>) {
            *self.script.lock().unwrap() = dets;
        }

        fn run(&mut self, ts_ms: u64) -> VisionReport {
            self.run_out(ts_ms).report
        }

        fn run_out(&mut self, ts_ms: u64) -> ProcessOutput {
            self.p.process(&frame(ts_ms), CameraHealth::Ok).unwrap()
        }
    }

    fn frame(ts_ms: u64) -> Frame {
        Frame::new(ts_ms, ts_ms, 64, 48, vec![0; 64 * 48 * 3]).unwrap()
    }

    fn det(x: f32, y: f32, w: f32, score: f32) -> Detection {
        Detection {
            bbox: RectF { x, y, w, h: w },
            landmarks5: [PointF {
                x: x + w / 2.0,
                y: y + w / 2.0,
            }; 5],
            score,
        }
    }

    fn face() -> Detection {
        det(0.3, 0.3, 0.2, 0.9)
    }

    #[test]
    fn new_track_runs_embed_and_liveness_once() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        let r = h.run(0);
        assert_eq!(h.calls.get(), [1, 1, 1, 1]);
        assert_eq!(r.tracks.len(), 1);
        let t = &r.tracks[0];
        assert_eq!(t.identity, IdentityState::Pending);
        assert_eq!(
            t.liveness,
            LivenessState::Live {
                score: 0.9,
                tier: LivenessTier::Rgb
            }
        );
        assert!(t.head_pose.is_some());
        assert_eq!(t.age_ms, 0);
        assert!((t.face_width_px - 0.2 * 64.0).abs() < 1e-4);
        assert_eq!(r.liveness_tier, LivenessTier::Rgb);
        // next frame: nothing due, values carried
        let r2 = h.run(33);
        assert_eq!(h.calls.get(), [2, 1, 1, 2]);
        let t2 = &r2.tracks[0];
        assert_eq!(t2.id, t.id);
        assert_eq!(t2.liveness, t.liveness);
        assert_eq!(t2.age_ms, 33);
    }

    #[test]
    fn process_returns_embeddings_only_when_computed() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        let out = h.run_out(0);
        let id = out.report.tracks[0].id;
        assert_eq!(
            out.embeddings,
            [TrackEmbedding {
                track: id,
                ts_ms: 0,
                embedding: vec![0.3f32; 4].into(),
            }]
        );
        // Within the interval: nothing computed, nothing carried over.
        assert!(h.run_out(33).embeddings.is_empty());
        assert!(h.run_out(999).embeddings.is_empty());
        let out = h.run_out(1000);
        assert_eq!(out.embeddings.len(), 1);
        assert_eq!(
            (out.embeddings[0].track, out.embeddings[0].ts_ms),
            (id, 1000)
        );
    }

    #[test]
    fn embed_rate_limited_by_interval() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        for k in 0..30 {
            assert_eq!(h.run(k * 100).tracks.len(), 1);
        }
        // embed at 0, 1000, 2000; liveness at 0, 2000; pose every frame
        assert_eq!(h.calls.get(), [30, 3, 2, 30]);
    }

    #[test]
    fn big_box_jump_forces_rerecognition() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        let id = h.run(0).tracks[0].id;
        // small move: IoU ~0.82 vs the recognised box, nothing re-run
        h.set(Some(vec![det(0.32, 0.3, 0.2, 0.9)]));
        h.run(33);
        assert_eq!(h.calls.get()[1..3], [1, 1]);
        // jump: IoU 0.43 vs the recognised box (< 0.5), still the same track (>= 0.3)
        h.set(Some(vec![det(0.38, 0.3, 0.2, 0.9)]));
        let out = h.run_out(66);
        assert_eq!(out.report.tracks[0].id, id);
        assert_eq!(h.calls.get()[1..3], [2, 2]);
        assert_eq!(out.embeddings[0].embedding.as_slice(), &[0.38f32; 4][..]);
    }

    #[test]
    fn health_not_ok_skips_models() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        h.run(0);
        let out = h.p.process(&frame(33), CameraHealth::Covered).unwrap();
        assert!(out.embeddings.is_empty());
        let r = out.report;
        assert!(r.tracks.is_empty());
        assert_eq!(r.camera_health, CameraHealth::Covered);
        assert_eq!(r.ts_ms, 33);
        assert_eq!(h.calls.get(), [1, 1, 1, 1]);
    }

    #[test]
    fn idle_then_face_switches_interval() {
        let mut h = Harness::new();
        for k in 0..20 {
            assert!(h.run(k * 200).tracks.is_empty());
        }
        assert_eq!(h.p.desired_interval_ms(), 200);
        h.set(Some(vec![face()]));
        let out = h.run_out(4000);
        assert_eq!(h.p.desired_interval_ms(), 66);
        assert_eq!(out.report.tracks.len(), 1);
        assert_eq!(out.embeddings.len(), 1);
        assert_ne!(out.report.tracks[0].liveness, LivenessState::Unknown);
        assert_eq!(h.calls.get(), [21, 1, 1, 1]);
        // face gone again: back to idle
        h.set(Some(Vec::new()));
        h.run(4066);
        assert_eq!(h.p.desired_interval_ms(), 200);
    }

    #[test]
    fn report_timings_populated() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        let t = h.run(0).timings;
        for (name, v) in [
            ("detect", t.detect_ms),
            ("recognize", t.recognize_ms),
            ("liveness", t.liveness_ms),
            ("pose", t.pose_ms),
        ] {
            assert!(v >= 0.5, "{name}: {v}");
        }
        let t = h.run(33).timings;
        assert!(t.detect_ms >= 0.5 && t.pose_ms >= 0.5);
        assert_eq!((t.recognize_ms, t.liveness_ms), (0.0, 0.0));
    }

    #[test]
    fn report_carries_frame_seq_and_ts() {
        let mut h = Harness::new();
        let r =
            h.p.process(
                &Frame::new(7, 1234, 64, 48, vec![0; 64 * 48 * 3]).unwrap(),
                CameraHealth::Ok,
            )
            .unwrap()
            .report;
        assert_eq!((r.seq, r.ts_ms), (7, 1234));
        assert_eq!(r.camera_health, CameraHealth::Ok);
    }

    #[test]
    fn detector_error_aborts_frame() {
        let mut h = Harness::new();
        h.set(None);
        assert!(h.p.process(&frame(0), CameraHealth::Ok).is_err());
    }

    #[test]
    fn malformed_frame_is_an_error_before_any_stage() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        let mut f = frame(0);
        f.data = vec![0; 64 * 48 * 3 - 3].into();
        assert!(matches!(
            h.p.process(&f, CameraHealth::Ok),
            Err(VisionError::Input(_))
        ));
        assert_eq!(h.calls.get(), [0, 0, 0, 0]);
    }

    #[test]
    fn stage_error_only_affects_that_track() {
        let mut h = Harness::new();
        // second face is too narrow for the mock embedder
        h.set(Some(vec![face(), det(0.7, 0.1, 0.04, 0.9)]));
        let out = h.run_out(0);
        let r = &out.report;
        assert_eq!(r.tracks.len(), 2);
        let embedded: Vec<_> = out.embeddings.iter().map(|e| e.track).collect();
        assert_eq!(embedded, [r.tracks[0].id]);
        assert_ne!(r.tracks[1].liveness, LivenessState::Unknown);
        // the failed embed is retried on the next frame
        h.run(33);
        assert_eq!(h.calls.get()[1], 3);
    }

    #[test]
    fn low_liveness_is_spoof() {
        let mut h = Harness::new();
        h.set(Some(vec![det(0.3, 0.3, 0.2, 0.6)]));
        let r = h.run(0);
        assert_eq!(
            r.tracks[0].liveness,
            LivenessState::Spoof {
                score: 0.6,
                tier: LivenessTier::Rgb
            }
        );
    }

    #[test]
    fn non_finite_box_dropped() {
        let mut h = Harness::new();
        let mut bad_landmark = det(0.6, 0.6, 0.2, 0.9);
        bad_landmark.landmarks5[2].y = f32::INFINITY;
        h.set(Some(vec![
            det(f32::NAN, 0.3, 0.2, 0.9),
            face(),
            bad_landmark,
        ]));
        let r = h.run(0);
        assert_eq!(r.tracks.len(), 1);
        assert_eq!(r.tracks[0].bbox.x, face().bbox.x);
        assert_eq!(h.calls.get(), [1, 1, 1, 1]);
    }

    #[test]
    fn edge_face_clamped_and_liveness_skipped_when_centre_outside() {
        let mut h = Harness::new();
        // centre at x = -0.05: outside the frame
        let mut d = det(-0.2, 0.3, 0.3, 0.9);
        d.landmarks5[0] = PointF { x: -0.1, y: 1.2 };
        h.set(Some(vec![d]));
        let out = h.run_out(0);
        let t = &out.report.tracks[0];
        assert_eq!(t.liveness, LivenessState::Unknown);
        assert_eq!(h.calls.get()[2], 0);
        assert_eq!(out.embeddings.len(), 1);
        let b = t.bbox;
        assert!(b.x == 0.0 && (b.w - 0.1).abs() < 1e-6, "{b:?}");
        assert_eq!(t.landmarks5[0], PointF { x: 0.0, y: 1.0 });
        // face width comes from the unclamped box
        assert!((t.face_width_px - 0.3 * 64.0).abs() < 1e-4);
    }

    #[test]
    fn face_too_close_for_the_liveness_crop_is_unknown_and_not_scored() {
        let mut h = Harness::new();
        // 64×48 frame: a 24 px tall box caps the 2.7× crop at 47/24 ≈ 1.96 (< 0.9 · 2.7)
        let close = det(0.3, 0.1, 0.5, 0.9);
        assert!(!crate::liveness_fits(64, 48, close.bbox));
        h.set(Some(vec![close]));
        for ts in [0, 33, 2500] {
            let out = h.run_out(ts);
            assert_eq!(out.report.tracks[0].liveness, LivenessState::Unknown);
        }
        assert_eq!(
            h.calls.get()[2],
            0,
            "liveness model ran on a too-close face"
        );
        // the same face stepping back is scored on the next frame
        h.set(Some(vec![face()]));
        let r = h.run(2533);
        assert!(matches!(r.tracks[0].liveness, LivenessState::Live { .. }));
        assert_eq!(h.calls.get()[2], 1);
    }

    /// Liveness and embedding outputs the audit tests steer per frame.
    #[derive(Clone)]
    struct Ctl {
        live: Arc<Mutex<Result<f32, ()>>>,
        emb: Arc<Mutex<Result<Vec<f32>, ()>>>,
    }

    struct CtlEmb(Ctl);
    impl FaceEmbedder for CtlEmb {
        fn embed(&mut self, _: &Frame, _: &Detection) -> Result<Vec<f32>, VisionError> {
            self.0
                .emb
                .lock()
                .unwrap()
                .clone()
                .map_err(|()| VisionError::Input("emb".into()))
        }
        fn model_id(&self) -> &str {
            "ctl"
        }
    }

    struct CtlLive(Ctl);
    impl LivenessModel for CtlLive {
        fn score(&mut self, _: &Frame, _: &Detection) -> Result<f32, VisionError> {
            (*self.0.live.lock().unwrap()).map_err(|()| VisionError::Input("live".into()))
        }
        fn tier(&self) -> LivenessTier {
            LivenessTier::Rgb
        }
    }

    /// Pipeline with a scripted detector and steerable embedder/liveness.
    fn ctl_harness() -> (Pipeline, Script, Ctl) {
        let script: Script = Arc::new(Mutex::new(Some(vec![face()])));
        let ctl = Ctl {
            live: Arc::new(Mutex::new(Ok(0.9))),
            emb: Arc::new(Mutex::new(Ok(vec![1.0; 4]))),
        };
        let p = Pipeline::new(
            Box::new(MockDet(Calls::default(), script.clone())),
            Box::new(CtlEmb(ctl.clone())),
            Box::new(CtlLive(ctl.clone())),
            Box::new(MockPose(Calls::default())),
            PipelineConfig::default(),
        );
        (p, script, ctl)
    }

    fn step(p: &mut Pipeline, ts: u64) -> TrackObs {
        p.process(&frame(ts), CameraHealth::Ok)
            .unwrap()
            .report
            .tracks
            .remove(0)
    }

    /// The embeddings computed on the frame at `ts`.
    fn step_embeddings(p: &mut Pipeline, ts: u64) -> Vec<Vec<f32>> {
        p.process(&frame(ts), CameraHealth::Ok)
            .unwrap()
            .embeddings
            .iter()
            .map(|e| e.embedding.as_slice().to_vec())
            .collect()
    }

    #[test]
    fn liveness_threshold_boundary_is_inclusive() {
        let (mut p, _, ctl) = ctl_harness();
        *ctl.live.lock().unwrap() = Ok(0.75);
        assert!(matches!(
            step(&mut p, 0).liveness,
            LivenessState::Live { .. }
        ));
        let (mut p, _, ctl) = ctl_harness();
        *ctl.live.lock().unwrap() = Ok(0.749_999);
        assert!(matches!(
            step(&mut p, 0).liveness,
            LivenessState::Spoof { .. }
        ));
    }

    #[test]
    fn live_track_turning_spoof_flips_after_interval() {
        let (mut p, _, ctl) = ctl_harness();
        assert!(matches!(
            step(&mut p, 0).liveness,
            LivenessState::Live { .. }
        ));
        *ctl.live.lock().unwrap() = Ok(0.1);
        assert!(matches!(
            step(&mut p, 1000).liveness,
            LivenessState::Live { .. }
        ));
        assert!(matches!(
            step(&mut p, 2000).liveness,
            LivenessState::Spoof { .. }
        ));
    }

    #[test]
    fn liveness_error_drops_previous_live_verdict() {
        let (mut p, _, ctl) = ctl_harness();
        assert!(matches!(
            step(&mut p, 0).liveness,
            LivenessState::Live { .. }
        ));
        *ctl.live.lock().unwrap() = Err(());
        // FR-LV-001: a failed re-score must not leave the stale Live verdict standing.
        assert_eq!(step(&mut p, 2000).liveness, LivenessState::Unknown);
        *ctl.live.lock().unwrap() = Ok(0.9);
        assert!(matches!(
            step(&mut p, 2033).liveness,
            LivenessState::Live { .. }
        ));
    }

    #[test]
    fn embed_error_yields_no_embedding_and_retries() {
        let (mut p, _, ctl) = ctl_harness();
        assert_eq!(step_embeddings(&mut p, 0).len(), 1);
        *ctl.emb.lock().unwrap() = Err(());
        assert!(step_embeddings(&mut p, 1000).is_empty());
        *ctl.emb.lock().unwrap() = Ok(vec![2.0; 4]);
        assert_eq!(step_embeddings(&mut p, 1033), [vec![2.0; 4]]);
    }

    #[test]
    fn backwards_timestamp_does_not_keep_stale_live() {
        let (mut p, _, ctl) = ctl_harness();
        assert!(matches!(
            step(&mut p, 10_000).liveness,
            LivenessState::Live { .. }
        ));
        // Camera restarts at ts 0; the same-position face is now a spoof.
        *ctl.live.lock().unwrap() = Ok(0.1);
        let mut last = step(&mut p, 0);
        for ts in (100..=2100).step_by(100) {
            last = step(&mut p, ts);
        }
        assert!(
            matches!(last.liveness, LivenessState::Spoof { .. }),
            "{:?}",
            last.liveness
        );
    }

    #[test]
    fn backwards_timestamp_reembeds_immediately() {
        let (mut p, _, ctl) = ctl_harness();
        assert_eq!(step_embeddings(&mut p, 10_000), [vec![1.0; 4]]);
        *ctl.emb.lock().unwrap() = Ok(vec![2.0; 4]);
        // Camera restarts at ts 0: the old timeline's schedule does not suppress the embed.
        assert_eq!(step_embeddings(&mut p, 0), [vec![2.0; 4]]);
        assert!(step_embeddings(&mut p, 100).is_empty());
    }

    #[test]
    fn nan_liveness_score_is_unknown_not_a_verdict() {
        let (mut p, _, ctl) = ctl_harness();
        *ctl.live.lock().unwrap() = Ok(f32::NAN);
        assert_eq!(step(&mut p, 0).liveness, LivenessState::Unknown);
    }

    #[test]
    fn non_finite_embedding_is_dropped() {
        let (mut p, _, ctl) = ctl_harness();
        *ctl.emb.lock().unwrap() = Ok(vec![1.0, f32::NAN, 0.0, 0.0]);
        assert!(step_embeddings(&mut p, 0).is_empty());
    }

    #[test]
    fn expired_track_state_is_pruned() {
        let (mut p, script, _) = ctl_harness();
        step(&mut p, 0);
        assert_eq!(p.tracks.len(), 1);
        *script.lock().unwrap() = Some(Vec::new());
        p.process(&frame(500), CameraHealth::Ok).unwrap();
        assert_eq!(p.tracks.len(), 1, "kept within the ttl");
        p.process(&frame(1600), CameraHealth::Ok).unwrap();
        assert!(p.tracks.is_empty(), "state outlived the tracker's ttl");
    }

    #[test]
    fn face_outside_frame_is_never_live() {
        for d in [
            det(1.2, 0.3, 0.2, 0.9),
            det(0.3, 1.2, 0.2, 0.9),
            det(-0.5, 0.3, 0.2, 0.9),
        ] {
            let (mut p, script, _) = ctl_harness();
            *script.lock().unwrap() = Some(vec![d]);
            let t = step(&mut p, 0);
            assert_eq!(t.liveness, LivenessState::Unknown, "{d:?}");
            assert_eq!(t.bbox.w * t.bbox.h, 0.0, "{d:?}");
        }
    }

    #[test]
    fn zero_area_box_is_reported_finite_and_rechecked_each_frame() {
        let mut h = Harness::new();
        h.set(Some(vec![det(0.5, 0.5, 0.0, 0.9)]));
        for k in 0..3 {
            let r = h.run(k * 33);
            let b = r.tracks[0].bbox;
            assert!([b.x, b.y, b.w, b.h].iter().all(|v| v.is_finite()));
        }
        // IoU of empty boxes is 0 (< rerecog_iou): the re-recognition trigger is not
        // suppressed, so the track is re-embedded every frame (fail-safe direction).
        assert_eq!(h.calls.get()[1], 3);
    }

    // ---- IR fusion ----

    use crate::ir::IrSpoofReason;
    use fasec_core::IrFrame;

    /// IR frame width; the RGB test frame is 64 wide, so the mock detector tells them apart.
    const IR_W: u32 = 100;
    /// The face box on both cameras: the identity mapping sits inside the associator's prior.
    const FACE_BOX: RectF = RectF {
        x: 0.3,
        y: 0.3,
        w: 0.4,
        h: 0.4,
    };
    const RGB_LIVE: LivenessState = LivenessState::Live {
        score: 0.9,
        tier: LivenessTier::Rgb,
    };
    /// IR live without glints scores 0.8.
    const IR_LIVE: LivenessState = LivenessState::Live {
        score: 0.8,
        tier: LivenessTier::Ir,
    };
    const IR_SPOOF: LivenessState = LivenessState::Spoof {
        score: 0.0,
        tier: LivenessTier::Ir,
    };

    type Dets = Arc<Mutex<Vec<Detection>>>;

    /// RGB faces for RGB frames, IR faces for IR frames (which arrive as grey replicated to RGB).
    struct DualDet {
        rgb: Dets,
        ir: Dets,
    }
    impl FaceDetector for DualDet {
        fn detect(&mut self, f: &Frame) -> Result<Vec<Detection>, VisionError> {
            let s = if f.width == IR_W { &self.ir } else { &self.rgb };
            Ok(s.lock().unwrap().clone())
        }
    }

    #[derive(Clone, Copy)]
    enum Scene {
        /// A face near the emitter: bright on a dark background, the nose a little darker (relief
        /// 130 ÷ 150, as recorded on the owner).
        Face,
        /// A face whose nose is darker than any recorded real face (relief 0.4): no IR verdict.
        DarkNose,
        /// An evenly bright face-sized patch (printed photo).
        Flat,
        /// Dark background only (what IR sees of a phone replay).
        Empty,
        /// A dim, real ambient (unlit) frame.
        Dim,
    }

    /// Pixels are placed in normalised coordinates, so any frame size shows the same scene.
    fn ir_frame(seq: u64, lit: bool, scene: Scene, (w, h): (u32, u32)) -> IrFrame {
        let px = |x: u32, y: u32| {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let face = (0.3..0.7).contains(&u) && (0.3..0.7).contains(&v);
            let nose = (0.46..0.54).contains(&u) && (0.51..0.59).contains(&v);
            match scene {
                Scene::Face if nose => 130,
                Scene::DarkNose if nose => 60,
                Scene::Face | Scene::DarkNose | Scene::Flat if face => 150,
                Scene::Dim => 10,
                _ => 20,
            }
        };
        let data = (0..w * h).map(|i| px(i % w, i / w)).collect();
        IrFrame::new(seq, seq * 33, w, h, lit, data).unwrap()
    }

    fn rgb_face(x: f32) -> Detection {
        Detection {
            bbox: RectF { x, ..FACE_BOX },
            landmarks5: [RectF { x, ..FACE_BOX }.center(); 5],
            score: 0.9,
        }
    }

    /// The IR face at `FACE_BOX`, landmarks where `Scene::Face` puts the eyes, nose and mouth.
    fn ir_face() -> Detection {
        let p = |x, y| PointF { x, y };
        Detection {
            bbox: FACE_BOX,
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

    /// A pipeline whose RGB liveness runs every frame and is steered by `live` (`Err` = the
    /// model gave no verdict, as for a face too close for MiniFASNet).
    struct IrRig {
        p: Pipeline,
        rgb: Dets,
        ir: Dets,
        live: Arc<Mutex<Result<f32, ()>>>,
        ts: u64,
        seq: u64,
        size: (u32, u32),
        /// Send a dim ambient frame with each lit frame (else none, as on blanking hardware).
        ambient: bool,
    }

    impl IrRig {
        fn new() -> IrRig {
            let rgb: Dets = Arc::new(Mutex::new(vec![rgb_face(0.3)]));
            let ir: Dets = Arc::new(Mutex::new(vec![ir_face()]));
            let ctl = Ctl {
                live: Arc::new(Mutex::new(Ok(0.9))),
                emb: Arc::new(Mutex::new(Ok(vec![1.0; 4]))),
            };
            let p = Pipeline::new(
                Box::new(DualDet {
                    rgb: rgb.clone(),
                    ir: ir.clone(),
                }),
                Box::new(CtlEmb(ctl.clone())),
                Box::new(CtlLive(ctl.clone())),
                Box::new(MockPose(Calls::default())),
                PipelineConfig {
                    liveness_interval_ms: 0,
                    ..PipelineConfig::default()
                },
            );
            IrRig {
                p,
                rgb,
                ir,
                live: ctl.live,
                ts: 0,
                seq: 0,
                size: (IR_W, IR_W),
                ambient: false,
            }
        }

        /// One RGB frame 33 ms after the last, with lit frame `lit` (seq, scene) if any.
        fn call(&mut self, lit: Option<(u64, Scene)>, ir_health: CameraHealth) -> ProcessOutput {
            self.ts += 33;
            let frames = lit.map(|(seq, s)| {
                let amb = self
                    .ambient
                    .then(|| ir_frame(seq, false, Scene::Dim, self.size));
                (ir_frame(seq, true, s, self.size), amb)
            });
            let ir = frames.as_ref().map(|(l, a)| (l, a.as_ref()));
            self.p
                .process_with_ir(&frame(self.ts), ir, CameraHealth::Ok, ir_health)
                .unwrap()
        }

        /// A new lit frame of `scene`, IR healthy.
        fn lit(&mut self, scene: Scene) -> ProcessOutput {
            self.seq += 1;
            self.call(Some((self.seq, scene)), CameraHealth::Ok)
        }

        /// No new lit frame this call, IR healthy.
        fn dark(&mut self) -> ProcessOutput {
            self.call(None, CameraHealth::Ok)
        }

        /// Five lit frames of the face: the associator trusts the mapping on the fifth.
        fn warm(&mut self) -> ProcessOutput {
            let mut out = self.lit(Scene::Face);
            for _ in 0..4 {
                out = self.lit(Scene::Face);
            }
            out
        }
    }

    fn live0(out: &ProcessOutput) -> LivenessState {
        out.report.tracks[0].liveness.clone()
    }

    #[test]
    fn fusion_table_row_by_row() {
        let rgb_spoof = LivenessState::Spoof {
            score: 0.2,
            tier: LivenessTier::Rgb,
        };
        let all = [RGB_LIVE, rgb_spoof.clone(), LivenessState::Unknown];
        let ir_spoof = IrEvidence::Spoof {
            score: 0.0,
            reason: IrSpoofReason::NoIrFace,
        };
        let ir_live = IrEvidence::Live { score: 0.8 };
        // IR spoof (streak met) wins over any RGB verdict
        for rgb in &all {
            assert_eq!(fuse(rgb, ir_spoof), (IR_SPOOF, LivenessTier::Ir), "{rgb:?}");
        }
        // IR live overrides any RGB verdict: MiniFASNet calls the owner Spoof at a laptop
        for rgb in &all {
            assert_eq!(fuse(rgb, ir_live), (IR_LIVE, LivenessTier::Ir), "{rgb:?}");
        }
        // no IR evidence: the RGB verdict as today
        for rgb in &all {
            assert_eq!(
                fuse(rgb, IrEvidence::Absent),
                (rgb.clone(), LivenessTier::Rgb),
                "{rgb:?}"
            );
        }
    }

    #[test]
    fn owner_too_close_for_rgb_is_live_on_ir_once_trusted() {
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        // the mapping is not trusted for the first four lit frames: RGB alone, Unknown
        for k in 0..4 {
            let out = r.lit(Scene::Face);
            assert_eq!(live0(&out), LivenessState::Unknown, "frame {k}");
            assert_eq!(out.report.liveness_tier, LivenessTier::Rgb, "frame {k}");
            assert_eq!(out.ir[0].evidence, IrEvidence::Absent);
        }
        let out = r.lit(Scene::Face);
        assert_eq!(live0(&out), IR_LIVE);
        assert_eq!(out.report.liveness_tier, LivenessTier::Ir);
        let id = out.report.tracks[0].id;
        assert_eq!(
            out.ir,
            [TrackIr {
                track: id,
                evidence: IrEvidence::Live { score: 0.8 },
                tier: LivenessTier::Ir,
                ambient: false,
            }]
        );
        // with a usable ambient frame the evidence says so
        r.ambient = true;
        let out = r.lit(Scene::Face);
        assert_eq!(live0(&out), IR_LIVE);
        assert!(out.ir[0].ambient);
    }

    #[test]
    fn ir_live_overrides_rgb_spoof_but_rgb_spoof_decides_without_ir() {
        let rgb_spoof = LivenessState::Spoof {
            score: 0.2,
            tier: LivenessTier::Rgb,
        };
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Ok(0.2);
        // before trust the RGB spoof stands
        for k in 0..4 {
            assert_eq!(live0(&r.lit(Scene::Face)), rgb_spoof, "frame {k}");
        }
        let out = r.lit(Scene::Face);
        assert_eq!(live0(&out), IR_LIVE);
        assert_eq!(out.ir[0].tier, LivenessTier::Ir);
        // between lit frames the IR evidence still holds
        assert_eq!(live0(&r.dark()), IR_LIVE);
        // IR no verdict (a flat-dark nose): the RGB spoof decides again
        let out = r.lit(Scene::DarkNose);
        assert_eq!(out.ir[0].evidence, IrEvidence::Absent);
        assert_eq!(live0(&out), rgb_spoof);
        // IR unhealthy: the RGB spoof decides
        r.seq += 1;
        let out = r.call(Some((r.seq, Scene::Face)), CameraHealth::Lost);
        assert_eq!(
            (live0(&out), out.ir[0].tier),
            (rgb_spoof, LivenessTier::Rgb)
        );
    }

    #[test]
    fn ir_spoof_needs_three_consecutive_lit_frames_and_beats_rgb_live() {
        for (scene, faces) in [(Scene::Empty, vec![]), (Scene::Flat, vec![ir_face()])] {
            let mut r = IrRig::new();
            assert_eq!(live0(&r.warm()), IR_LIVE);
            *r.ir.lock().unwrap() = faces.clone();
            // two spoof frames: below the streak, the RGB verdict stands
            for _ in 0..2 {
                let out = r.lit(scene);
                assert_eq!(live0(&out), RGB_LIVE);
                assert_eq!(out.ir[0].evidence, IrEvidence::Absent);
            }
            let out = r.lit(scene);
            assert_eq!(live0(&out), IR_SPOOF);
            assert!(matches!(out.ir[0].evidence, IrEvidence::Spoof { .. }));
            // a non-spoof frame resets the streak
            *r.ir.lock().unwrap() = vec![ir_face()];
            assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
            for _ in 0..3 {
                *r.ir.lock().unwrap() = faces.clone();
                for _ in 0..2 {
                    assert_eq!(live0(&r.lit(scene)), RGB_LIVE);
                }
                *r.ir.lock().unwrap() = vec![ir_face()];
                assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
            }
        }
    }

    #[test]
    fn ir_counts_lit_frames_when_rgb_runs_faster() {
        let mut r = IrRig::new();
        assert_eq!(live0(&r.warm()), IR_LIVE);
        // RGB three times faster than lit: IR Live holds between lit frames
        for _ in 0..2 {
            assert_eq!(live0(&r.dark()), IR_LIVE);
        }
        // and three spoof lit frames make a streak, whatever the calls between them
        *r.ir.lock().unwrap() = Vec::new();
        for k in 0..3 {
            let want = if k < 2 { RGB_LIVE } else { IR_SPOOF };
            assert_eq!(live0(&r.lit(Scene::Empty)), want, "lit frame {k}");
            for _ in 0..2 {
                assert_eq!(live0(&r.dark()), want, "after lit frame {k}");
            }
        }
    }

    #[test]
    fn ir_spoof_streak_breaks_when_the_track_misses_a_lit_frame() {
        let mut r = IrRig::new();
        assert_eq!(live0(&r.warm()), IR_LIVE);
        *r.ir.lock().unwrap() = Vec::new();
        for _ in 0..2 {
            assert_eq!(live0(&r.lit(Scene::Empty)), RGB_LIVE);
        }
        *r.rgb.lock().unwrap() = Vec::new();
        assert!(r.lit(Scene::Empty).report.tracks.is_empty());
        *r.rgb.lock().unwrap() = vec![rgb_face(0.3)];
        // scattered spoof frames never add up: this is the first of a new streak
        assert_eq!(live0(&r.lit(Scene::Empty)), RGB_LIVE);
        assert_eq!(live0(&r.lit(Scene::Empty)), RGB_LIVE);
        assert_eq!(live0(&r.lit(Scene::Empty)), IR_SPOOF);
    }

    #[test]
    fn stale_ir_falls_back_to_rgb() {
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        assert_eq!(live0(&r.warm()), IR_LIVE);
        // the last lit frame counts for 500 ms of RGB time
        r.ts += 500 - 33;
        let out = r.dark();
        assert_eq!(live0(&out), IR_LIVE);
        assert_eq!(out.report.liveness_tier, LivenessTier::Ir);
        // then it is stale
        let out = r.dark();
        assert_eq!(live0(&out), LivenessState::Unknown);
        assert_eq!(out.report.liveness_tier, LivenessTier::Rgb);
        assert_eq!(out.ir[0].evidence, IrEvidence::Absent);
        // a new lit frame brings it back
        assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
        // RGB time before the lit frame (an RGB restart) is not fresh
        r.ts -= 100;
        assert_eq!(live0(&r.dark()), LivenessState::Unknown);
        // a track missing from the latest lit frame has no IR evidence, however recent
        assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
        *r.rgb.lock().unwrap() = Vec::new();
        r.lit(Scene::Face);
        *r.rgb.lock().unwrap() = vec![rgb_face(0.3)];
        let out = r.dark();
        assert_eq!(live0(&out), LivenessState::Unknown);
        assert_eq!(out.report.liveness_tier, LivenessTier::Ir);
        // a lit frame seen before (same or older seq) is not new evidence
        for back in [0, 1] {
            let mut r = IrRig::new();
            *r.live.lock().unwrap() = Err(());
            r.warm();
            r.ts += 500;
            let out = r.call(Some((r.seq - back, Scene::Face)), CameraHealth::Ok);
            assert_eq!(live0(&out), LivenessState::Unknown, "seq - {back}");
        }
        // an unlit frame passed as the lit one is not a lit frame
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        r.warm();
        let unlit = ir_frame(r.seq + 1, false, Scene::Face, r.size);
        r.ts += 533;
        let out =
            r.p.process_with_ir(
                &frame(r.ts),
                Some((&unlit, None)),
                CameraHealth::Ok,
                CameraHealth::Ok,
            )
            .unwrap();
        assert_eq!(
            (live0(&out), out.report.liveness_tier),
            (LivenessState::Unknown, LivenessTier::Rgb)
        );
    }

    #[test]
    fn ir_loss_falls_back_to_rgb_and_trust_is_re_earned() {
        for lost in [
            CameraHealth::Lost,
            CameraHealth::Busy,
            CameraHealth::Error("x".into()),
        ] {
            let mut r = IrRig::new();
            *r.live.lock().unwrap() = Err(());
            assert_eq!(live0(&r.warm()), IR_LIVE);
            // a lit frame passed with unhealthy IR is ignored
            r.seq += 1;
            let out = r.call(Some((r.seq, Scene::Face)), lost.clone());
            assert_eq!(live0(&out), LivenessState::Unknown, "{lost:?}");
            assert_eq!(out.report.liveness_tier, LivenessTier::Rgb);
            // IR back: four lit frames before the mapping is trusted again
            for k in 0..4 {
                let out = r.lit(Scene::Face);
                assert_eq!(live0(&out), LivenessState::Unknown, "{lost:?} frame {k}");
                assert_eq!(out.report.liveness_tier, LivenessTier::Rgb);
            }
            assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
        }
    }

    #[test]
    fn ir_loss_drops_trust_even_on_a_malformed_frame() {
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        assert_eq!(live0(&r.warm()), IR_LIVE);
        let mut bad = frame(r.ts + 33);
        bad.data = vec![0; 3].into();
        let res =
            r.p.process_with_ir(&bad, None, CameraHealth::Ok, CameraHealth::Lost);
        assert!(matches!(res, Err(VisionError::Input(_))));
        // the restart signal was not lost with the frame: trust is earned again
        for k in 0..4 {
            let out = r.lit(Scene::Face);
            assert_eq!(live0(&out), LivenessState::Unknown, "frame {k}");
        }
        assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
    }

    #[test]
    fn process_is_rgb_only_and_drops_ir_trust() {
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        r.warm();
        let out = r.p.process(&frame(r.ts + 33), CameraHealth::Ok).unwrap();
        r.ts += 33;
        assert_eq!(live0(&out), LivenessState::Unknown);
        assert_eq!(out.report.liveness_tier, LivenessTier::Rgb);
        assert_eq!(out.ir[0].evidence, IrEvidence::Absent);
        assert_eq!(live0(&r.lit(Scene::Face)), LivenessState::Unknown);
    }

    #[test]
    fn ir_frame_size_change_resets_trust() {
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        assert_eq!(live0(&r.warm()), IR_LIVE);
        r.size = (IR_W, 80);
        for k in 0..4 {
            assert_eq!(
                live0(&r.lit(Scene::Face)),
                LivenessState::Unknown,
                "frame {k}"
            );
        }
        assert_eq!(live0(&r.lit(Scene::Face)), IR_LIVE);
    }

    #[test]
    fn ir_face_without_rgb_face_is_never_live() {
        let mut r = IrRig::new();
        *r.rgb.lock().unwrap() = Vec::new();
        for _ in 0..10 {
            let out = r.lit(Scene::Face);
            assert!(out.report.tracks.is_empty() && out.ir.is_empty());
            assert_eq!(out.report.liveness_tier, LivenessTier::Rgb);
        }
        // trusted, then the RGB face goes: nothing to be live
        *r.rgb.lock().unwrap() = vec![rgb_face(0.3)];
        r.warm();
        *r.rgb.lock().unwrap() = Vec::new();
        let out = r.lit(Scene::Face);
        assert!(out.report.tracks.is_empty() && out.ir.is_empty());
    }

    #[test]
    fn one_ir_face_two_tracks_only_the_matched_track_gets_ir_evidence() {
        let mut r = IrRig::new();
        *r.live.lock().unwrap() = Err(());
        assert_eq!(live0(&r.warm()), IR_LIVE);
        // a second RGB face overlapping the first; IR sees one face
        for dets in [
            vec![rgb_face(0.3), rgb_face(0.36)],
            vec![rgb_face(0.36), rgb_face(0.3)],
        ] {
            *r.rgb.lock().unwrap() = dets;
            for _ in 0..5 {
                let out = r.lit(Scene::Face);
                assert_eq!(out.report.tracks.len(), 2);
                for (t, ir) in out.report.tracks.iter().zip(&out.ir) {
                    assert_eq!(t.id, ir.track);
                    if t.bbox.x == 0.3 {
                        assert_eq!(t.liveness, IR_LIVE);
                    } else {
                        assert_eq!(t.liveness, LivenessState::Unknown);
                        assert_eq!(ir.evidence, IrEvidence::Absent);
                    }
                }
            }
        }
    }

    fn models_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }

    fn temp_models(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fasec-pipeline-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn from_models_err(dir: &Path) -> String {
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        match Pipeline::from_models(&rt, dir, PipelineConfig::default()) {
            Ok(_) => panic!("loaded unverified models"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn from_models_missing_file_names_model() {
        let e = from_models_err(&temp_models("missing"));
        assert!(e.contains("yunet-2023mar"), "{e}");
    }

    #[test]
    fn from_models_hash_mismatch_names_model() {
        let dir = temp_models("mismatch");
        let size = Manifest::embedded().get("yunet-2023mar").unwrap().size;
        // The manifest size, so the hash (not the size check) rejects it.
        let junk = vec![0u8; usize::try_from(size.unwrap()).unwrap()];
        std::fs::write(dir.join("face_detection_yunet_2023mar.onnx"), junk).unwrap();
        let e = from_models_err(&dir);
        assert!(e.contains("yunet-2023mar") && e.contains("mismatch"), "{e}");
    }

    #[test]
    fn from_models_wrong_size_names_model() {
        let dir = temp_models("wrong_size");
        std::fs::write(
            dir.join("face_detection_yunet_2023mar.onnx"),
            b"not a model",
        )
        .unwrap();
        let e = from_models_err(&dir);
        assert!(e.contains("yunet-2023mar") && e.contains("size"), "{e}");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn pipeline_ignores_on_disk_manifest() {
        let dir = temp_models("tampered_manifest");
        let ids = [
            "yunet-2023mar",
            "sface-2021dec",
            "minifasnet-v2",
            "minifasnet-v1se",
        ];
        for id in ids {
            let entry = Manifest::embedded().get(id).unwrap();
            let name = fasec_models::model_path(Path::new(""), entry).unwrap();
            std::fs::copy(models_dir().join(&name), dir.join(&name)).unwrap();
        }
        // An on-disk manifest that would reject every real file (and could bless swapped ones).
        let tampered = std::fs::read_to_string(models_dir().join("manifest.toml"))
            .unwrap()
            .lines()
            .map(|l| {
                if l.starts_with("sha256 = \"") && l.len() > 12 {
                    format!("sha256 = \"{}\"", "0".repeat(64))
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join("manifest.toml"), tampered).unwrap();
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        Pipeline::from_models(&rt, &dir, PipelineConfig::default()).unwrap();

        // A tampered model file is still rejected, naming its id.
        let v1se = Manifest::embedded().get("minifasnet-v1se").unwrap();
        let path = fasec_models::model_path(&dir, v1se).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&path, bytes).unwrap();
        let e = from_models_err(&dir);
        assert!(
            e.contains("minifasnet-v1se") && e.contains("mismatch"),
            "{e}"
        );
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn pipeline_on_face_a1() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.assets/face-a-1.jpg");
        let img = image::open(p).unwrap().to_rgb8();
        let frame = Frame::new(1, 0, img.width(), img.height(), img.into_raw()).unwrap();
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let mut pipe =
            Pipeline::from_models(&rt, &models_dir(), PipelineConfig::default()).unwrap();
        let out = pipe.process(&frame, CameraHealth::Ok).unwrap();
        let r = &out.report;
        println!("{:?}", r.timings);
        assert_eq!(r.tracks.len(), 1, "{:?}", r.tracks);
        let t = &r.tracks[0];
        println!("liveness {:?}, pose {:?}", t.liveness, t.head_pose);
        assert_eq!(out.embeddings.len(), 1);
        let e = &out.embeddings[0];
        assert_eq!((e.track, e.embedding.dim()), (t.id, 128));
        assert_ne!(t.liveness, LivenessState::Unknown);
        assert!(t.head_pose.is_some());
        assert!(t.face_width_px > 0.0);
    }
}
