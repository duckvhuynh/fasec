//! One frame in, one [`VisionReport`] out: detect + track every frame, attention every frame,
//! embedding and liveness on a per-track schedule (SRS §3.3).

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use fasec_core::{
    CameraHealth, Frame, IdentityState, LivenessState, PointF, RectF, StageTimings, TrackId,
    TrackObs, VisionReport,
};
use fasec_models::Manifest;

use crate::attention::{CoarsePose, face_width_px};
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
        }
    }
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
}

/// What a track carries between scheduled runs.
struct TrackState {
    last_seen_ms: u64,
    embedding: Option<Vec<f32>>,
    /// `ts_ms` and box of the last successful embed.
    embedded_at: Option<(u64, RectF)>,
    liveness: LivenessState,
    /// `ts_ms` and box of the last successful liveness score.
    scored_at: Option<(u64, RectF)>,
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
        }
    }

    /// YuNet (640 px long side) + SFace + MiniFASNet + CoarsePose from `models_dir`. Every
    /// file is SHA-256 verified against `models_dir/manifest.toml` before any session is
    /// created; a missing or mismatched file is an error naming the model id.
    pub fn from_models(
        rt: &Runtime,
        models_dir: &Path,
        cfg: PipelineConfig,
    ) -> Result<Pipeline, VisionError> {
        let manifest = Manifest::load(models_dir.join("manifest.toml"))
            .map_err(|e| VisionError::Model(format!("models manifest: {e}")))?;
        let verified = |id: &str| {
            let bad = |e: &dyn std::fmt::Display| VisionError::Model(format!("{id}: {e}"));
            let entry = manifest.get(id).ok_or_else(|| bad(&"not in manifest"))?;
            let path = fasec_models::model_path(models_dir, entry).map_err(|e| bad(&e))?;
            fasec_models::verify_file(&path, &entry.sha256).map_err(|e| bad(&e))?;
            Ok::<_, VisionError>(path)
        };
        // All four verified before the first session is created.
        let yunet = verified("yunet-2023mar")?;
        let sface = verified("sface-2021dec")?;
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

    /// Runs the stages due on `frame`. A detector error fails the frame; an error of a
    /// per-track stage only leaves that track's field `None`/`Unknown` (retried next frame).
    /// With `health != Ok` no model runs and the report has no tracks.
    pub fn process(
        &mut self,
        frame: &Frame,
        health: CameraHealth,
    ) -> Result<VisionReport, VisionError> {
        crate::image_ops::check_frame(frame)?;
        let mut timings = StageTimings {
            detect_ms: 0.0,
            recognize_ms: 0.0,
            liveness_ms: 0.0,
            pose_ms: 0.0,
        };
        let mut report = VisionReport {
            seq: frame.seq,
            ts_ms: frame.ts_ms,
            tracks: Vec::new(),
            camera_health: health,
            liveness_tier: self.live.tier(),
            timings,
        };
        if report.camera_health != CameraHealth::Ok {
            self.active = false;
            return Ok(report);
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
                embedding: None,
                embedded_at: None,
                liveness: LivenessState::Unknown,
                scored_at: None,
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
                let res = self.emb.embed(frame, d).and_then(|e| {
                    if e.iter().all(|v| v.is_finite()) {
                        Ok(e)
                    } else {
                        Err(VisionError::Model("non-finite embedding".into()))
                    }
                });
                match res {
                    Ok(e) => {
                        st.embedding = Some(e);
                        st.embedded_at = Some((now, d.bbox));
                    }
                    Err(e) => {
                        tracing::warn!(track = t.id.0, error = %e, "embedding failed");
                        st.embedding = None;
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
                // would score unrelated edge pixels.
                if (0.0..=1.0).contains(&c.x) && (0.0..=1.0).contains(&c.y) {
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

            report.tracks.push(TrackObs {
                id: t.id,
                bbox: d.bbox.clamp_unit(),
                landmarks5: d.landmarks5.map(|p| PointF {
                    x: p.x.clamp(0.0, 1.0),
                    y: p.y.clamp(0.0, 1.0),
                }),
                det_score: d.score,
                identity: IdentityState::Pending,
                liveness: st.liveness.clone(),
                head_pose,
                face_width_px: face_width_px(d, frame),
                age_ms: t.age_ms,
                embedding: st.embedding.clone(),
            });
        }

        report.timings = timings;
        self.active = !report.tracks.is_empty();
        Ok(report)
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
        assert_eq!(t.embedding.as_deref(), Some(&[0.3f32; 4][..]));
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
        assert_eq!(t2.embedding, t.embedding);
        assert_eq!(t2.liveness, t.liveness);
        assert_eq!(t2.age_ms, 33);
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
        let r = h.run(66);
        assert_eq!(r.tracks[0].id, id);
        assert_eq!(h.calls.get()[1..3], [2, 2]);
        assert_eq!(r.tracks[0].embedding.as_deref(), Some(&[0.38f32; 4][..]));
    }

    #[test]
    fn health_not_ok_skips_models() {
        let mut h = Harness::new();
        h.set(Some(vec![face()]));
        h.run(0);
        let r = h.p.process(&frame(33), CameraHealth::Covered).unwrap();
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
        let r = h.run(4000);
        assert_eq!(h.p.desired_interval_ms(), 66);
        assert_eq!(r.tracks.len(), 1);
        assert!(r.tracks[0].embedding.is_some());
        assert_ne!(r.tracks[0].liveness, LivenessState::Unknown);
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
            .unwrap();
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
        let r = h.run(0);
        assert_eq!(r.tracks.len(), 2);
        assert!(r.tracks[0].embedding.is_some());
        assert_eq!(r.tracks[1].embedding, None);
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
        let r = h.run(0);
        let t = &r.tracks[0];
        assert_eq!(t.liveness, LivenessState::Unknown);
        assert_eq!(h.calls.get()[2], 0);
        assert!(t.embedding.is_some());
        let b = t.bbox;
        assert!(b.x == 0.0 && (b.w - 0.1).abs() < 1e-6, "{b:?}");
        assert_eq!(t.landmarks5[0], PointF { x: 0.0, y: 1.0 });
        // face width comes from the unclamped box
        assert!((t.face_width_px - 0.3 * 64.0).abs() < 1e-4);
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
            .tracks
            .remove(0)
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
    fn embed_error_drops_previous_embedding() {
        let (mut p, _, ctl) = ctl_harness();
        assert!(step(&mut p, 0).embedding.is_some());
        *ctl.emb.lock().unwrap() = Err(());
        assert_eq!(step(&mut p, 1000).embedding, None);
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
    fn backwards_timestamp_does_not_keep_stale_embedding() {
        let (mut p, _, ctl) = ctl_harness();
        assert_eq!(step(&mut p, 10_000).embedding, Some(vec![1.0; 4]));
        *ctl.emb.lock().unwrap() = Ok(vec![2.0; 4]);
        let mut last = step(&mut p, 0);
        for ts in (100..=1100).step_by(100) {
            last = step(&mut p, ts);
        }
        assert_eq!(last.embedding, Some(vec![2.0; 4]));
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
        assert_eq!(step(&mut p, 0).embedding, None);
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

    fn models_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models")
    }

    fn temp_models(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("fasec-pipeline-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(
            models_dir().join("manifest.toml"),
            dir.join("manifest.toml"),
        )
        .unwrap();
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
        std::fs::write(
            dir.join("face_detection_yunet_2023mar.onnx"),
            b"not a model",
        )
        .unwrap();
        let e = from_models_err(&dir);
        assert!(e.contains("yunet-2023mar") && e.contains("mismatch"), "{e}");
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
        let r = pipe.process(&frame, CameraHealth::Ok).unwrap();
        println!("{:?}", r.timings);
        assert_eq!(r.tracks.len(), 1, "{:?}", r.tracks);
        let t = &r.tracks[0];
        println!("liveness {:?}, pose {:?}", t.liveness, t.head_pose);
        assert_eq!(t.embedding.as_ref().map(Vec::len), Some(128));
        assert_ne!(t.liveness, LivenessState::Unknown);
        assert!(t.head_pose.is_some());
        assert!(t.face_width_px > 0.0);
    }
}
