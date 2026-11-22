//! Guided 5-pose enrolment (FR-ID-002): only good samples are accepted, 2 per pose.
//!
//! Pose sign convention follows M1 `CoarsePose`: +yaw = nose towards image right,
//! +pitch = looking down. Front is judged on the absolute pose; every other pose on the pose
//! relative to the mean of the accepted Front samples (the user's frontal baseline). No frame or crop is kept: only accepted embeddings (FR-ID-005).

use fasec_core::{
    CameraHealth, Embedding, Frame, LivenessState, RectF, TrackEmbedding, VisionReport,
    liveness_fits,
};
use zeroize::Zeroizing;

use crate::matcher::{DEFAULT_THRESHOLD, normalised};
use crate::store::Pose;

/// Default Laplacian-variance floor (grey 0..255). Measured on the `.assets` portraits scaled
/// to a 160..320 px face: sharp faces scored 48.7..608, the same crops box-blurred (radius 3)
/// 2.2..9.7; the log-space midpoint of 48.7 and 9.7 is 21.7, rounded down to 20.
pub const DEFAULT_MIN_SHARPNESS: f32 = 20.0;

/// Accepted head-pose range in degrees, inclusive on both ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseWindow {
    pub yaw: (f32, f32),
    pub pitch: (f32, f32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EnrolConfig {
    pub samples_per_pose: usize,
    pub min_face_px: f32,
    pub min_sharpness: f32,
    /// Targets in the order they are asked for. Front is absolute; the others are relative
    /// to the frontal baseline (absolute while no Front sample is accepted yet).
    pub pose_windows: [(Pose, PoseWindow); 5],
}

impl PoseWindow {
    fn contains(&self, yaw: f32, pitch: f32) -> bool {
        // NaN angles fall outside every window
        (self.yaw.0..=self.yaw.1).contains(&yaw) && (self.pitch.0..=self.pitch.1).contains(&pitch)
    }
}

impl Default for EnrolConfig {
    fn default() -> Self {
        let w = |yaw, pitch| PoseWindow { yaw, pitch };
        // The brief bounds one axis per turned pose; the other is held to ±20° so a diagonal
        // head turn is not accepted as a clean left/right/up/down sample.
        let free = (-20.0, 20.0);
        // Up/Down ±6°: on real hardware (one user, 2026-11-22) CoarsePose read pitch +11° looking
        // straight and only +1° on a >30° chin-up tilt (biased, low gain), so ±12° never passed.
        EnrolConfig {
            samples_per_pose: 2,
            min_face_px: 112.0,
            min_sharpness: DEFAULT_MIN_SHARPNESS,
            pose_windows: [
                (Pose::Front, w((-10.0, 10.0), (-10.0, 10.0))),
                (Pose::Left, w((-45.0, -15.0), free)),
                (Pose::Right, w((15.0, 45.0), free)),
                (Pose::Up, w(free, (-40.0, -6.0))),
                (Pose::Down, w(free, (6.0, 40.0))),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    NoFace,
    MultipleFaces,
    TooSmall,
    /// The face is too large for an in-distribution liveness crop (`liveness_fits`).
    TooClose,
    Blurry,
    NotLive,
    SpoofDetected,
    WrongPose {
        want: Pose,
    },
    NoEmbeddingYet,
    CameraUnhealthy,
    /// `frame` is not the frame `report` describes, so its gates cannot vouch for it.
    FrameMismatch,
    /// Another track than the first accepted sample's, or an embedding unlike the accepted
    /// ones: one enrolment never mixes two people.
    DifferentPerson,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleVerdict {
    Accepted { pose: Pose },
    Rejected(RejectReason),
    Complete,
}

/// One accepted sample. `Debug` stays redacted because `Embedding`'s is.
#[derive(Clone, Debug, PartialEq)]
pub struct EnrolledSample {
    pub pose: Pose,
    /// Detector score of the accepted face, 0..1.
    pub quality: f32,
    pub embedding: Embedding,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum EnrolError {
    #[error("enrolment incomplete: {have} of {need} samples")]
    Incomplete { have: usize, need: usize },
}

pub struct Enrolment {
    cfg: EnrolConfig,
    samples: Vec<EnrolledSample>,
    /// `ts_ms` of the last accepted embedding, so one frame is never stored twice.
    last_ts: Option<u64>,
    /// The accepted embeddings, L2-normalised. A new sample must look like at least one of them.
    /// Identity is judged by embedding, not by track id: the tracker hands out a new id when it
    /// loses the face for a moment (common while turning), and every sample has exactly one face.
    units: Vec<Zeroizing<Vec<f32>>>,
    /// (yaw, pitch) of each accepted Front sample.
    front: Vec<(f32, f32)>,
}

impl Enrolment {
    /// `samples_per_pose` below 1 is raised to 1: an enrolment can never complete empty.
    pub fn new(mut cfg: EnrolConfig) -> Self {
        cfg.samples_per_pose = cfg.samples_per_pose.max(1);
        Enrolment {
            cfg,
            samples: Vec::new(),
            last_ts: None,
            units: Vec::new(),
            front: Vec::new(),
        }
    }

    /// Mean (yaw, pitch) in degrees of the accepted Front samples; `None` before the first.
    pub fn frontal_baseline(&self) -> Option<(f32, f32)> {
        let n = self.front.len() as f32;
        let (y, p) = self
            .front
            .iter()
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        (n > 0.0).then(|| (y / n, p / n))
    }

    /// The pose being collected now; `None` once every pose has its samples.
    pub fn current_target(&self) -> Option<Pose> {
        let i = self.samples.len().checked_div(self.cfg.samples_per_pose)?;
        self.cfg.pose_windows.get(i).map(|(p, _)| *p)
    }

    /// `(accepted, needed)`.
    pub fn progress(&self) -> (usize, usize) {
        let need = self.cfg.samples_per_pose * self.cfg.pose_windows.len();
        (self.samples.len(), need)
    }

    /// Judges one frame. `embedding` is the one computed on this frame for its single track
    /// (from `ProcessOutput::embeddings`), if any. Only `Accepted` stores anything.
    pub fn offer(
        &mut self,
        frame: &Frame,
        report: &VisionReport,
        embedding: Option<&TrackEmbedding>,
    ) -> SampleVerdict {
        let Some(want) = self.current_target() else {
            return SampleVerdict::Complete;
        };
        match self.judge(frame, report, embedding, want) {
            Ok((quality, te, unit)) => {
                // `judge` accepted exactly one track, with a pose
                if let (Pose::Front, Some(hp)) = (want, report.tracks[0].head_pose) {
                    self.front.push((hp.yaw, hp.pitch));
                }
                self.last_ts = Some(te.ts_ms);
                self.units.push(unit);
                self.samples.push(EnrolledSample {
                    pose: want,
                    quality,
                    embedding: te.embedding.clone(),
                });
                SampleVerdict::Accepted { pose: want }
            }
            Err(r) => SampleVerdict::Rejected(r),
        }
    }

    fn judge<'a>(
        &self,
        frame: &Frame,
        report: &VisionReport,
        embedding: Option<&'a TrackEmbedding>,
        want: Pose,
    ) -> Result<(f32, &'a TrackEmbedding, Zeroizing<Vec<f32>>), RejectReason> {
        use RejectReason::*;
        if frame.ts_ms != report.ts_ms {
            return Err(FrameMismatch);
        }
        if report.camera_health != CameraHealth::Ok {
            return Err(CameraUnhealthy);
        }
        let t = match report.tracks.as_slice() {
            [] => return Err(NoFace),
            [t] => t,
            _ => return Err(MultipleFaces),
        };
        // `RangeFrom::contains` is false for NaN, so a NaN measurement fails the gate
        if !(self.cfg.min_face_px..).contains(&t.face_width_px) {
            return Err(TooSmall);
        }
        // before the liveness gate: a too-close face is `Unknown`, and "move back" is the fix
        if !liveness_fits(frame.width, frame.height, t.bbox) {
            return Err(TooClose);
        }
        if !(self.cfg.min_sharpness..).contains(&sharpness(frame, t.bbox)) {
            return Err(Blurry);
        }
        match t.liveness {
            LivenessState::Live { .. } => {}
            LivenessState::Spoof { .. } => return Err(SpoofDetected),
            LivenessState::Unknown => return Err(NotLive),
        }
        let window = self.cfg.pose_windows.iter().find(|(p, _)| *p == want);
        let (by, bp) = match want {
            Pose::Front => (0.0, 0.0),
            _ => self.frontal_baseline().unwrap_or((0.0, 0.0)),
        };
        let in_pose = match (t.head_pose, window) {
            (Some(hp), Some((_, w))) => w.contains(hp.yaw - by, hp.pitch - bp),
            _ => false,
        };
        if !in_pose {
            return Err(WrongPose { want });
        }
        let te = match embedding {
            Some(te)
                if te.track == t.id
                    && te.ts_ms == report.ts_ms
                    && Some(te.ts_ms) > self.last_ts =>
            {
                te
            }
            _ => return Err(NoEmbeddingYet),
        };
        // empty, zero or non-finite: nothing usable to store
        let unit = normalised(te.embedding.as_slice()).ok_or(NoEmbeddingYet)?;
        if !self.units.is_empty() && !self.like_accepted(&unit) {
            return Err(DifferentPerson);
        }
        Ok((t.det_score, te, unit))
    }

    /// Cosine of `unit` against its most similar accepted sample is at least the matcher's
    /// acceptance threshold. The closest sample (not the mean) tolerates the pose change from
    /// one side to the other; another person stays far below the threshold for every sample.
    fn like_accepted(&self, unit: &[f32]) -> bool {
        self.units.iter().any(|u| {
            let cos: f32 = u.iter().zip(unit).map(|(a, b)| a * b).sum();
            // `>=` is false for NaN
            u.len() == unit.len() && cos >= DEFAULT_THRESHOLD
        })
    }

    /// All samples, in target order; an error unless every pose has its samples.
    pub fn finish(self) -> Result<Vec<EnrolledSample>, EnrolError> {
        let (have, need) = self.progress();
        if have < need {
            return Err(EnrolError::Incomplete { have, need });
        }
        Ok(self.samples)
    }
}

/// Variance of the 3×3 Laplacian over the grey (Rec.601) crop of `bbox` (normalised), clamped
/// to the frame; 0.0 when the crop is smaller than 3×3.
pub fn sharpness(frame: &Frame, bbox: RectF) -> f32 {
    let b = bbox.clamp_unit();
    let (w, h) = (frame.width as usize, frame.height as usize);
    // `as usize` saturates and maps NaN to 0, so a NaN box gives an empty crop
    let x0 = (b.x * w as f32).floor() as usize;
    let y0 = (b.y * h as f32).floor() as usize;
    let x1 = (((b.x + b.w) * w as f32).ceil() as usize).min(w);
    let y1 = (((b.y + b.h) * h as f32).ceil() as usize).min(h);
    if x1 < x0 + 3 || y1 < y0 + 3 {
        return 0.0;
    }
    let cw = x1 - x0;
    let grey: Vec<f32> = (y0..y1)
        .flat_map(|y| (x0..x1).map(move |x| (y * w + x) * 3))
        .map(|i| {
            let p = &frame.data[i..i + 3];
            0.299 * f32::from(p[0]) + 0.587 * f32::from(p[1]) + 0.114 * f32::from(p[2])
        })
        .collect();
    let (mut n, mut sum, mut sq) = (0.0f64, 0.0f64, 0.0f64);
    for y in 1..(y1 - y0 - 1) {
        for x in 1..(cw - 1) {
            let g = |dx: usize, dy: usize| grey[(y + dy - 1) * cw + x + dx - 1];
            let lap = f64::from(g(1, 0) + g(0, 1) + g(2, 1) + g(1, 2) - 4.0 * g(1, 1));
            n += 1.0;
            sum += lap;
            sq += lap * lap;
        }
    }
    let mean = sum / n;
    (sq / n - mean * mean).max(0.0) as f32
}

#[cfg(test)]
mod tests {
    use fasec_core::{
        HeadPose, IdentityState, LivenessTier, PointF, StageTimings, TrackId, TrackObs,
    };

    use super::*;

    /// 22 px in the 64×64 test frame: small enough for the liveness crop.
    const FACE: RectF = RectF {
        x: 0.3,
        y: 0.3,
        w: 0.35,
        h: 0.35,
    };

    /// 64×64 frame: 1-px black/white checkerboard (sharp) or flat grey (blurry).
    fn frame(ts_ms: u64, sharp: bool) -> Frame {
        let mut data = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let v = match sharp {
                    true if (x + y) % 2 == 0 => 255,
                    true => 0,
                    false => 128,
                };
                data.extend([v; 3]);
            }
        }
        Frame::new(ts_ms, ts_ms, 64, 64, data).unwrap()
    }

    fn track(id: u64, yaw: f32, pitch: f32) -> TrackObs {
        TrackObs {
            id: TrackId(id),
            bbox: FACE,
            landmarks5: [PointF { x: 0.5, y: 0.5 }; 5],
            det_score: 0.9,
            identity: IdentityState::Pending,
            liveness: LivenessState::Live {
                score: 0.95,
                tier: LivenessTier::Rgb,
            },
            head_pose: Some(HeadPose {
                yaw,
                pitch,
                roll: 0.0,
                confidence: 0.8,
            }),
            face_width_px: 150.0,
            age_ms: 1000,
        }
    }

    fn report(ts_ms: u64, tracks: Vec<TrackObs>) -> VisionReport {
        VisionReport {
            seq: ts_ms,
            ts_ms,
            tracks,
            camera_health: CameraHealth::Ok,
            liveness_tier: LivenessTier::Rgb,
            timings: StageTimings {
                detect_ms: 0.0,
                recognize_ms: 0.0,
                liveness_ms: 0.0,
                pose_ms: 0.0,
            },
        }
    }

    fn emb(id: u64, ts_ms: u64) -> TrackEmbedding {
        TrackEmbedding {
            track: TrackId(id),
            ts_ms,
            embedding: Embedding::from(vec![ts_ms as f32; 128]),
        }
    }

    /// (yaw, pitch) in the middle of each default target window.
    fn angles(p: Pose) -> (f32, f32) {
        match p {
            Pose::Front => (0.0, 0.0),
            Pose::Left => (-25.0, 0.0),
            Pose::Right => (25.0, 0.0),
            Pose::Up => (0.0, -20.0),
            Pose::Down => (0.0, 20.0),
        }
    }

    /// Offers a good front sample at `ts_ms` after `edit` spoils it; asserts `want` and that
    /// nothing was stored (Review Focus 4).
    fn assert_rejected(
        edit: impl FnOnce(&mut Frame, &mut VisionReport, &mut Option<TrackEmbedding>),
        want: RejectReason,
    ) {
        let mut e = Enrolment::new(EnrolConfig::default());
        let (mut f, mut r, mut te) = (
            frame(10, true),
            report(10, vec![track(1, 0.0, 0.0)]),
            Some(emb(1, 10)),
        );
        edit(&mut f, &mut r, &mut te);
        assert_eq!(e.offer(&f, &r, te.as_ref()), SampleVerdict::Rejected(want));
        assert_eq!(e.progress(), (0, 10));
        assert_eq!(e.current_target(), Some(Pose::Front));
        assert_eq!(
            e.finish(),
            Err(EnrolError::Incomplete { have: 0, need: 10 })
        );
    }

    #[test]
    fn default_config_matches_fr_id_002() {
        let c = EnrolConfig::default();
        assert_eq!((c.samples_per_pose, c.min_face_px), (2, 112.0));
        let order: Vec<Pose> = c.pose_windows.iter().map(|(p, _)| *p).collect();
        assert_eq!(
            order,
            [Pose::Front, Pose::Left, Pose::Right, Pose::Up, Pose::Down]
        );
        assert_eq!(c.pose_windows[1].1.yaw, (-45.0, -15.0));
        assert_eq!(c.pose_windows[3].1.pitch, (-40.0, -6.0));
        assert_eq!(c.pose_windows[4].1.pitch, (6.0, 40.0));
        assert!(c.min_sharpness > 0.0);
    }

    #[test]
    fn rejects_two_faces() {
        assert_rejected(
            |_, r, _| r.tracks.push(track(2, 0.0, 0.0)),
            RejectReason::MultipleFaces,
        );
        assert_rejected(|_, r, _| r.tracks.clear(), RejectReason::NoFace);
    }

    #[test]
    fn rejects_small_face() {
        assert_rejected(
            |_, r, _| r.tracks[0].face_width_px = 111.0,
            RejectReason::TooSmall,
        );
        assert_rejected(
            |_, r, _| r.tracks[0].face_width_px = f32::NAN,
            RejectReason::TooSmall,
        );
    }

    #[test]
    fn rejects_face_too_close_for_liveness() {
        // 0.5 of 64 px: the 2.7× liveness crop is capped at 63/32 ≈ 1.97
        let close = RectF {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        assert_rejected(|_, r, _| r.tracks[0].bbox = close, RejectReason::TooClose);
        // checked before liveness: the pipeline reports such a face as Unknown
        assert_rejected(
            |_, r, _| {
                r.tracks[0].bbox = close;
                r.tracks[0].liveness = LivenessState::Unknown;
            },
            RejectReason::TooClose,
        );
    }

    #[test]
    fn rejects_blurred_frame() {
        assert_rejected(|f, _, _| *f = frame(10, false), RejectReason::Blurry);
    }

    #[test]
    fn rejects_spoof_and_unknown_liveness() {
        assert_rejected(
            |_, r, _| {
                r.tracks[0].liveness = LivenessState::Spoof {
                    score: 0.1,
                    tier: LivenessTier::RgbTemporal,
                }
            },
            RejectReason::SpoofDetected,
        );
        assert_rejected(
            |_, r, _| r.tracks[0].liveness = LivenessState::Unknown,
            RejectReason::NotLive,
        );
    }

    #[test]
    fn rejects_wrong_pose_with_target() {
        let want = RejectReason::WrongPose { want: Pose::Front };
        assert_rejected(|_, r, _| r.tracks[0] = track(1, -25.0, 0.0), want);
        assert_rejected(|_, r, _| r.tracks[0] = track(1, 0.0, 10.5), want);
        assert_rejected(|_, r, _| r.tracks[0] = track(1, f32::NAN, 0.0), want);
        assert_rejected(|_, r, _| r.tracks[0].head_pose = None, want);
    }

    #[test]
    fn rejects_without_fresh_embedding() {
        let want = RejectReason::NoEmbeddingYet;
        assert_rejected(|_, _, te| *te = None, want);
        assert_rejected(|_, _, te| *te = Some(emb(2, 10)), want);
        assert_rejected(|_, _, te| *te = Some(emb(1, 9)), want);

        // the same frame offered twice is stored once
        let mut e = Enrolment::new(EnrolConfig::default());
        let (f, r, te) = (
            frame(10, true),
            report(10, vec![track(1, 0.0, 0.0)]),
            emb(1, 10),
        );
        assert_eq!(
            e.offer(&f, &r, Some(&te)),
            SampleVerdict::Accepted { pose: Pose::Front }
        );
        assert_eq!(e.offer(&f, &r, Some(&te)), SampleVerdict::Rejected(want));
        assert_eq!(e.progress(), (1, 10));
    }

    #[test]
    fn rejects_when_camera_unhealthy() {
        assert_rejected(
            |_, r, _| r.camera_health = CameraHealth::Covered,
            RejectReason::CameraUnhealthy,
        );
        assert_rejected(
            |_, r, _| r.camera_health = CameraHealth::Error("gone".into()),
            RejectReason::CameraUnhealthy,
        );
    }

    #[test]
    fn accepts_and_advances_in_order() {
        let mut e = Enrolment::new(EnrolConfig::default());
        let order = [Pose::Front, Pose::Left, Pose::Right, Pose::Up, Pose::Down];
        let mut ts = 0;
        for (i, pose) in order.into_iter().enumerate() {
            // the next pose early is refused with the current target
            if let Some(next) = order.get(i + 1) {
                ts += 1;
                let (y, p) = angles(*next);
                let v = e.offer(
                    &frame(ts, true),
                    &report(ts, vec![track(1, y, p)]),
                    Some(&emb(1, ts)),
                );
                assert_eq!(
                    v,
                    SampleVerdict::Rejected(RejectReason::WrongPose { want: pose })
                );
            }
            for k in 0..2 {
                assert_eq!(e.current_target(), Some(pose));
                assert_eq!(e.progress(), (i * 2 + k, 10));
                ts += 1;
                let (y, p) = angles(pose);
                let v = e.offer(
                    &frame(ts, true),
                    &report(ts, vec![track(1, y, p)]),
                    Some(&emb(1, ts)),
                );
                assert_eq!(v, SampleVerdict::Accepted { pose });
            }
        }
        assert_eq!((e.current_target(), e.progress()), (None, (10, 10)));
        ts += 1;
        let v = e.offer(
            &frame(ts, true),
            &report(ts, vec![track(1, 0.0, 0.0)]),
            Some(&emb(1, ts)),
        );
        assert_eq!(v, SampleVerdict::Complete);

        let samples = e.finish().unwrap();
        let poses: Vec<Pose> = samples.iter().map(|s| s.pose).collect();
        let want: Vec<Pose> = order.iter().flat_map(|p| [*p, *p]).collect();
        assert_eq!(poses, want);
        assert!(
            samples
                .iter()
                .all(|s| s.embedding.dim() == 128 && s.quality == 0.9)
        );
        // each sample keeps the embedding of the frame that was accepted
        assert_ne!(samples[0].embedding, samples[1].embedding);
    }

    #[test]
    fn finish_requires_all_poses() {
        let mut e = Enrolment::new(EnrolConfig::default());
        for ts in 1..=9 {
            let pose =
                [Pose::Front, Pose::Left, Pose::Right, Pose::Up, Pose::Down][(ts as usize - 1) / 2];
            let (y, p) = angles(pose);
            let v = e.offer(
                &frame(ts, true),
                &report(ts, vec![track(1, y, p)]),
                Some(&emb(1, ts)),
            );
            assert_eq!(v, SampleVerdict::Accepted { pose });
        }
        assert_eq!(e.current_target(), Some(Pose::Down));
        assert_eq!(
            e.finish(),
            Err(EnrolError::Incomplete { have: 9, need: 10 })
        );
    }

    #[test]
    fn sharpness_orders_sharp_above_blurred() {
        let all = RectF {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
        let sharp = sharpness(&frame(0, true), all);
        // 1-px checkerboard: every interior Laplacian is ±4·255
        assert!((sharp - (4.0f32 * 255.0).powi(2)).abs() < 1.0, "{sharp}");
        assert_eq!(sharpness(&frame(0, false), all), 0.0);

        // a blurred step edge is less sharp than the crisp one
        let edge = |soft: bool| {
            let mut d = Vec::new();
            for _y in 0..16 {
                for x in 0..16u32 {
                    let v = match (x, soft) {
                        (0..7, _) => 0u8,
                        (7, true) => 64,
                        (8, true) => 192,
                        _ => 255,
                    };
                    d.extend([v; 3]);
                }
            }
            Frame::new(0, 0, 16, 16, d).unwrap()
        };
        assert!(sharpness(&edge(false), all) > sharpness(&edge(true), all));
    }

    #[test]
    fn sharpness_of_empty_or_outside_crop_is_zero() {
        let f = frame(0, true);
        assert_eq!(
            sharpness(
                &f,
                RectF {
                    x: 2.0,
                    y: 0.0,
                    w: 0.5,
                    h: 0.5
                }
            ),
            0.0
        );
        assert_eq!(
            sharpness(
                &f,
                RectF {
                    x: 0.5,
                    y: 0.5,
                    w: 0.0,
                    h: 0.5
                }
            ),
            0.0
        );
        assert_eq!(
            sharpness(
                &f,
                RectF {
                    x: 0.5,
                    y: 0.5,
                    w: 0.02,
                    h: 0.5
                }
            ),
            0.0
        ); // 2 px wide
        assert_eq!(
            sharpness(
                &f,
                RectF {
                    x: f32::NAN,
                    y: 0.0,
                    w: 0.5,
                    h: 0.5
                }
            ),
            0.0
        );
        // a box hanging off the frame is clamped, not rejected
        assert!(
            sharpness(
                &f,
                RectF {
                    x: 0.9,
                    y: -0.5,
                    w: 0.5,
                    h: 1.0
                }
            ) > 0.0
        );
        let empty = Frame::new(0, 0, 0, 0, vec![]).unwrap();
        assert_eq!(sharpness(&empty, FACE), 0.0);
    }

    /// RGB box blur of radius `r` (separable, edges clamped).
    fn box_blur(f: &Frame, r: i64) -> Frame {
        let (w, h) = (i64::from(f.width), i64::from(f.height));
        let pass = |src: &[u8], horizontal: bool| -> Vec<u8> {
            let mut out = vec![0u8; src.len()];
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let mut sum = 0u32;
                        for d in -r..=r {
                            let (sx, sy) = match horizontal {
                                true => ((x + d).clamp(0, w - 1), y),
                                false => (x, (y + d).clamp(0, h - 1)),
                            };
                            sum += u32::from(src[((sy * w + sx) * 3 + c) as usize]);
                        }
                        out[((y * w + x) * 3 + c) as usize] = (sum / (2 * r as u32 + 1)) as u8;
                    }
                }
            }
            out
        };
        let data = pass(&pass(&f.data, true), false);
        Frame::new(0, 0, f.width, f.height, data).unwrap()
    }

    #[test]
    #[ignore = "needs .assets portraits (gitignored)"]
    fn sharpness_threshold_separates_assets() {
        use image::imageops::{FilterType, resize};
        // hand-picked face boxes, normalised to each portrait
        let faces = [
            (
                "face-a-1",
                RectF {
                    x: 0.38,
                    y: 0.18,
                    w: 0.24,
                    h: 0.25,
                },
            ),
            (
                "face-a-2",
                RectF {
                    x: 0.38,
                    y: 0.20,
                    w: 0.20,
                    h: 0.22,
                },
            ),
            (
                "face-b-1",
                RectF {
                    x: 0.35,
                    y: 0.18,
                    w: 0.31,
                    h: 0.27,
                },
            ),
        ];
        let (mut sharp, mut blurred) = (Vec::new(), Vec::new());
        for (name, bbox) in faces {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../.assets/{name}.jpg"));
            let img = image::open(p).unwrap().to_rgb8();
            // webcam-like scale: face box FACE_PX wide, as a 1080p enrolment frame would hold it
            for face_px in [160.0f32, 240.0, 320.0] {
                let k = face_px / (bbox.w * img.width() as f32);
                let (w, h) = (
                    (img.width() as f32 * k) as u32,
                    (img.height() as f32 * k) as u32,
                );
                let small = resize(&img, w, h, FilterType::Triangle);
                let f = Frame::new(0, 0, w, h, small.into_raw()).unwrap();
                let (s, b) = (sharpness(&f, bbox), sharpness(&box_blur(&f, 3), bbox));
                println!("{name} face {face_px}px: sharp {s:.1} blurred(r=3) {b:.2}");
                sharp.push(s);
                blurred.push(b);
            }
        }
        let min_sharp = sharp.iter().copied().fold(f32::INFINITY, f32::min);
        let max_blur = blurred.iter().copied().fold(0.0, f32::max);
        println!(
            "min sharp {min_sharp:.1}, max blurred {max_blur:.2}, log midpoint {:.1}, default {}",
            (min_sharp * max_blur).sqrt(),
            DEFAULT_MIN_SHARPNESS
        );
        assert!(min_sharp > DEFAULT_MIN_SHARPNESS && DEFAULT_MIN_SHARPNESS > max_blur);
    }

    // ---- audit additions ----

    #[test]
    fn zero_samples_per_pose_is_never_a_complete_enrolment() {
        let mut e = Enrolment::new(EnrolConfig {
            samples_per_pose: 0,
            ..EnrolConfig::default()
        });
        let _ = e.offer(
            &frame(10, true),
            &report(10, vec![track(1, 0.0, 0.0)]),
            Some(&emb(1, 10)),
        );
        assert_eq!(e.progress(), (1, 5));
        assert!(
            !matches!(e.finish(), Ok(s) if s.is_empty()),
            "empty enrolment reported success"
        );
    }

    /// Sharpness is measured on `frame`, the embedding on the frame at `report.ts_ms`;
    /// a sharp unrelated frame must not vouch for a blurry embedding frame.
    #[test]
    fn frame_from_another_timestamp_is_rejected() {
        let mut e = Enrolment::new(EnrolConfig::default());
        let v = e.offer(
            &frame(5, true), // sharp frame from t=5
            &report(10, vec![track(1, 0.0, 0.0)]),
            Some(&emb(1, 10)), // embedding/report from t=10
        );
        assert!(matches!(v, SampleVerdict::Rejected(_)), "{v:?}");
        assert_eq!(e.progress().0, 0);
        // checked before any gate: a mismatched frame reports the mismatch, not blur
        let v = e.offer(
            &frame(5, false),
            &report(10, vec![track(1, 0.0, 0.0)]),
            Some(&emb(1, 10)),
        );
        assert_eq!(v, SampleVerdict::Rejected(RejectReason::FrameMismatch));
    }

    #[test]
    fn rejected_frames_between_good_ones_never_store_or_advance() {
        let mut e = Enrolment::new(EnrolConfig::default());
        let good = |ts: u64| {
            (
                frame(ts, true),
                report(ts, vec![track(1, 0.0, 0.0)]),
                emb(1, ts),
            )
        };
        let (f, r, te) = good(1);
        assert!(matches!(
            e.offer(&f, &r, Some(&te)),
            SampleVerdict::Accepted { .. }
        ));
        // second face, blur, spoof, stale embedding: progress stays at 1
        let mut two = report(2, vec![track(1, 0.0, 0.0)]);
        two.tracks.push(track(2, 0.0, 0.0));
        let mut spoof = report(3, vec![track(1, 0.0, 0.0)]);
        spoof.tracks[0].liveness = LivenessState::Spoof {
            score: 0.1,
            tier: LivenessTier::Rgb,
        };
        for (f, r, te) in [
            (frame(2, true), two, emb(1, 2)),
            (
                frame(3, false),
                report(3, vec![track(1, 0.0, 0.0)]),
                emb(1, 3),
            ),
            (frame(3, true), spoof, emb(1, 3)),
            (
                frame(1, true),
                report(1, vec![track(1, 0.0, 0.0)]),
                emb(1, 1),
            ),
        ] {
            assert!(matches!(
                e.offer(&f, &r, Some(&te)),
                SampleVerdict::Rejected(_)
            ));
            assert_eq!(e.progress(), (1, 10));
        }
        let (f, r, te) = good(4);
        assert!(matches!(
            e.offer(&f, &r, Some(&te)),
            SampleVerdict::Accepted { pose: Pose::Front }
        ));
        assert_eq!(e.progress(), (2, 10));
    }

    /// Accepts one front sample from track 1 with `emb(1, 1)`.
    fn started() -> Enrolment {
        let mut e = Enrolment::new(EnrolConfig::default());
        let v = e.offer(
            &frame(1, true),
            &report(1, vec![track(1, 0.0, 0.0)]),
            Some(&emb(1, 1)),
        );
        assert_eq!(v, SampleVerdict::Accepted { pose: Pose::Front });
        e
    }

    #[test]
    fn a_new_track_of_the_same_person_continues() {
        // regression: the tracker re-assigns the id after losing the face for a moment (e.g.
        // while turning); pinning the first id rejected every later frame and stuck enrolment
        let mut e = started();
        let v = e.offer(
            &frame(2, true),
            &report(2, vec![track(2, 0.0, 0.0)]),
            Some(&emb(2, 2)),
        );
        assert_eq!(v, SampleVerdict::Accepted { pose: Pose::Front });
        assert_eq!(e.progress(), (2, 10));
    }

    #[test]
    fn a_new_track_of_another_person_is_rejected() {
        let mut e = started();
        let mut v = vec![0.0f32; 128];
        v[0] = 1.0;
        let te = TrackEmbedding {
            track: TrackId(2),
            ts_ms: 2,
            embedding: Embedding::from(v),
        };
        let v = e.offer(
            &frame(2, true),
            &report(2, vec![track(2, 0.0, 0.0)]),
            Some(&te),
        );
        assert_eq!(v, SampleVerdict::Rejected(RejectReason::DifferentPerson));
        assert_eq!(e.progress(), (1, 10));
    }

    #[test]
    fn rejects_an_unlike_embedding_on_the_same_track() {
        let mut e = started();
        // e_0 against the accepted all-equal direction: cosine 1/sqrt(128) ≈ 0.088 < 0.363
        let mut v = vec![0.0f32; 128];
        v[0] = 1.0;
        let te = TrackEmbedding {
            track: TrackId(1),
            ts_ms: 2,
            embedding: Embedding::from(v),
        };
        let v = e.offer(
            &frame(2, true),
            &report(2, vec![track(1, 0.0, 0.0)]),
            Some(&te),
        );
        assert_eq!(v, SampleVerdict::Rejected(RejectReason::DifferentPerson));
        assert_eq!(e.progress(), (1, 10));
        assert_eq!(e.current_target(), Some(Pose::Front));
        // a similar embedding is still accepted afterwards
        let v = e.offer(
            &frame(3, true),
            &report(3, vec![track(1, 0.0, 0.0)]),
            Some(&emb(1, 3)),
        );
        assert_eq!(v, SampleVerdict::Accepted { pose: Pose::Front });
        assert_eq!(
            e.finish().unwrap_err(),
            EnrolError::Incomplete { have: 2, need: 10 }
        );
    }

    #[test]
    fn similarity_threshold_is_the_matchers() {
        // a unit embedding at cosine `cos` against the accepted direction
        let at = |ts: u64, cos: f32| {
            let mean = 1.0 / (128f32).sqrt();
            // unit vector u = cos·m + sin·w, with w ⟂ m (m = all-equal unit vector)
            let sin = (1.0 - cos * cos).sqrt();
            let mut w = vec![0.0f32; 128];
            w[0] = 1.0;
            w[1] = -1.0;
            let wn = 2f32.sqrt();
            let v: Vec<f32> = (0..128).map(|i| cos * mean + sin * w[i] / wn).collect();
            TrackEmbedding {
                track: TrackId(1),
                ts_ms: ts,
                embedding: Embedding::from(v),
            }
        };
        let mut e = started();
        let offer = |e: &mut Enrolment, te: &TrackEmbedding| {
            e.offer(
                &frame(te.ts_ms, true),
                &report(te.ts_ms, vec![track(1, 0.0, 0.0)]),
                Some(te),
            )
        };
        assert_eq!(
            offer(&mut e, &at(2, DEFAULT_THRESHOLD - 0.01)),
            SampleVerdict::Rejected(RejectReason::DifferentPerson)
        );
        assert_eq!(
            offer(&mut e, &at(3, DEFAULT_THRESHOLD + 0.01)),
            SampleVerdict::Accepted { pose: Pose::Front }
        );
    }

    fn offer_at(e: &mut Enrolment, ts: u64, yaw: f32, pitch: f32) -> SampleVerdict {
        e.offer(
            &frame(ts, true),
            &report(ts, vec![track(1, yaw, pitch)]),
            Some(&emb(1, ts)),
        )
    }

    /// Two Front samples at `(yaw, pitch)`, then `poses` (absolute angles), all accepted.
    fn with_baseline(yaw: f32, pitch: f32, poses: &[(f32, f32)]) -> (Enrolment, u64) {
        let mut e = Enrolment::new(EnrolConfig::default());
        let mut ts = 0;
        for &(y, p) in [(yaw, pitch), (yaw, pitch)].iter().chain(poses) {
            ts += 1;
            let v = offer_at(&mut e, ts, y, p);
            assert!(
                matches!(v, SampleVerdict::Accepted { .. }),
                "{y} {p}: {v:?}"
            );
        }
        (e, ts)
    }

    /// Up is next after Front, Left and Right at relative yaw ∓25.
    fn at_up(pitch: f32) -> (Enrolment, u64) {
        let side = [(-25.0, pitch), (-25.0, pitch), (25.0, pitch), (25.0, pitch)];
        let (e, ts) = with_baseline(0.0, pitch, &side);
        assert_eq!(e.current_target(), Some(Pose::Up));
        (e, ts)
    }

    fn accepted(pose: Pose) -> SampleVerdict {
        SampleVerdict::Accepted { pose }
    }

    fn wrong(want: Pose) -> SampleVerdict {
        SampleVerdict::Rejected(RejectReason::WrongPose { want })
    }

    #[test]
    fn frontal_baseline_is_the_mean_of_front_samples() {
        let mut e = Enrolment::new(EnrolConfig::default());
        assert_eq!(e.frontal_baseline(), None);
        assert_eq!(offer_at(&mut e, 1, 2.0, 8.0), accepted(Pose::Front));
        assert_eq!(offer_at(&mut e, 2, 4.0, 10.0), accepted(Pose::Front));
        assert_eq!(e.frontal_baseline(), Some((3.0, 9.0)));
        // later poses do not move it
        assert_eq!(offer_at(&mut e, 3, -20.0, 9.0), accepted(Pose::Left));
        assert_eq!(e.frontal_baseline(), Some((3.0, 9.0)));
    }

    #[test]
    fn front_stays_absolute() {
        let mut e = Enrolment::new(EnrolConfig::default());
        assert_eq!(offer_at(&mut e, 1, 0.0, 10.5), wrong(Pose::Front));
        assert_eq!(offer_at(&mut e, 2, 0.0, 10.0), accepted(Pose::Front));
        // a baseline from the first sample does not shift the second Front sample's window
        assert_eq!(offer_at(&mut e, 3, 0.0, -10.5), wrong(Pose::Front));
    }

    #[test]
    fn up_and_down_are_relative_to_the_frontal_baseline() {
        // baseline pitch +10
        let (mut e, ts) = at_up(10.0);
        // +5 is relative −5: not up enough
        assert_eq!(offer_at(&mut e, ts + 1, 0.0, 5.0), wrong(Pose::Up));
        // −3 is relative −13, −8 relative −18: both up
        assert_eq!(offer_at(&mut e, ts + 2, 0.0, -3.0), accepted(Pose::Up));
        assert_eq!(offer_at(&mut e, ts + 3, 0.0, -8.0), accepted(Pose::Up));
        // Down: +15 is relative +5, +16 relative +6 (the window edge)
        assert_eq!(offer_at(&mut e, ts + 4, 0.0, 15.0), wrong(Pose::Down));
        assert_eq!(offer_at(&mut e, ts + 5, 0.0, 16.0), accepted(Pose::Down));
    }

    #[test]
    fn up_passes_on_the_measured_hardware_readings() {
        // looking straight read pitch +11, a full chin-up tilt about +1 (2026-11-22). The
        // absolute Front window stops at +10, so +11 itself never becomes the baseline: the
        // Front samples land at +10 (+9 and +11 averaged the same) and +11 is refused.
        let mut e = Enrolment::new(EnrolConfig::default());
        assert_eq!(offer_at(&mut e, 1, 0.0, 11.0), wrong(Pose::Front));
        assert_eq!(offer_at(&mut e, 2, -3.0, 9.0), accepted(Pose::Front));
        assert_eq!(offer_at(&mut e, 3, 3.0, 11.0), wrong(Pose::Front));
        assert_eq!(offer_at(&mut e, 4, 3.0, 10.0), accepted(Pose::Front));
        assert_eq!(e.frontal_baseline(), Some((0.0, 9.5)));
        for (ts, yaw) in [(5, -25.0), (6, -25.0), (7, 25.0), (8, 25.0)] {
            assert!(matches!(
                offer_at(&mut e, ts, yaw, 9.5),
                SampleVerdict::Accepted { .. }
            ));
        }
        // +3 is relative −6.5 and the full-tilt +1 relative −8.5: accepted for Up;
        // +4 (relative −5.5) is not
        assert_eq!(offer_at(&mut e, 9, 0.0, 4.0), wrong(Pose::Up));
        assert_eq!(offer_at(&mut e, 10, 0.0, 3.0), accepted(Pose::Up));
        assert_eq!(offer_at(&mut e, 11, -3.0, 1.0), accepted(Pose::Up));

        // the exact readings, with a Front window that admits +11: +4 is relative −7
        let mut cfg = EnrolConfig::default();
        cfg.pose_windows[0].1.pitch = (-10.0, 15.0);
        let mut e = Enrolment::new(cfg);
        for (ts, yaw) in [
            (1, 0.0),
            (2, 0.0),
            (3, -25.0),
            (4, -25.0),
            (5, 25.0),
            (6, 25.0),
        ] {
            assert!(matches!(
                offer_at(&mut e, ts, yaw, 11.0),
                SampleVerdict::Accepted { .. }
            ));
        }
        assert_eq!(e.frontal_baseline(), Some((0.0, 11.0)));
        assert_eq!(offer_at(&mut e, 7, 0.0, 4.0), accepted(Pose::Up));
    }

    #[test]
    fn left_and_right_are_relative_to_the_frontal_baseline() {
        // baseline yaw +8, pitch +10
        let (mut e, ts) = with_baseline(8.0, 10.0, &[]);
        // −5 is relative −13: not turned enough; −10 is relative −18
        assert_eq!(offer_at(&mut e, ts + 1, -5.0, 10.0), wrong(Pose::Left));
        assert_eq!(offer_at(&mut e, ts + 2, -10.0, 10.0), accepted(Pose::Left));
        // the off-axis pitch is relative too: −12 is relative −22, +28 relative +18
        assert_eq!(offer_at(&mut e, ts + 3, -10.0, -12.0), wrong(Pose::Left));
        assert_eq!(offer_at(&mut e, ts + 4, -10.0, 28.0), accepted(Pose::Left));
        // +20 is in the absolute window but relative +12; +25 is relative +17
        assert_eq!(offer_at(&mut e, ts + 5, 20.0, 10.0), wrong(Pose::Right));
        assert_eq!(offer_at(&mut e, ts + 6, 25.0, 10.0), accepted(Pose::Right));
        // the widened outer bound: relative +46 rejected, +45 accepted
        assert_eq!(offer_at(&mut e, ts + 7, 54.0, 10.0), wrong(Pose::Right));
        assert_eq!(offer_at(&mut e, ts + 8, 53.0, 10.0), accepted(Pose::Right));
        assert_eq!(e.current_target(), Some(Pose::Up));
    }
}
