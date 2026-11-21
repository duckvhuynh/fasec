//! Per-frame vision output.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::geom::{PointF, RectF};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TrackId(pub u64);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum IdentityState {
    Unknown,
    Pending,
    Known { person: String, score: f32 },
    Stranger { best: f32 },
}

/// Liveness evidence tier, ordered weakest to strongest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LivenessTier {
    Rgb,
    RgbTemporal,
    Ir,
    Depth,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LivenessState {
    Unknown,
    Live { score: f32, tier: LivenessTier },
    Spoof { score: f32, tier: LivenessTier },
}

/// Head pose in degrees.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadPose {
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub confidence: f32,
}

/// One tracked face in one frame. Coordinates are normalised 0..1. `Debug` redacts the
/// embedding (a biometric template, NFR-PRIV-003) so `?report` logs never carry it.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackObs {
    pub id: TrackId,
    pub bbox: RectF,
    /// YuNet order: right eye, left eye, nose tip, right mouth corner, left mouth corner.
    pub landmarks5: [PointF; 5],
    pub det_score: f32,
    pub identity: IdentityState,
    pub liveness: LivenessState,
    pub head_pose: Option<HeadPose>,
    pub face_width_px: f32,
    pub age_ms: u64,
    pub embedding: Option<Vec<f32>>,
}

/// Stands in for an embedding in `Debug` output.
struct Redacted(usize);

impl fmt::Debug for Redacted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{} floats redacted>", self.0)
    }
}

impl fmt::Debug for TrackObs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Destructured so a new field cannot be left out of the output by accident.
        let TrackObs {
            id,
            bbox,
            landmarks5,
            det_score,
            identity,
            liveness,
            head_pose,
            face_width_px,
            age_ms,
            embedding,
        } = self;
        f.debug_struct("TrackObs")
            .field("id", id)
            .field("bbox", bbox)
            .field("landmarks5", landmarks5)
            .field("det_score", det_score)
            .field("identity", identity)
            .field("liveness", liveness)
            .field("head_pose", head_pose)
            .field("face_width_px", face_width_px)
            .field("age_ms", age_ms)
            .field("embedding", &embedding.as_ref().map(|e| Redacted(e.len())))
            .finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CameraHealth {
    Ok,
    Lost,
    Covered,
    PermissionDenied,
    Busy,
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageTimings {
    pub detect_ms: f32,
    pub recognize_ms: f32,
    pub liveness_ms: f32,
    pub pose_ms: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VisionReport {
    pub seq: u64,
    pub ts_ms: u64,
    pub tracks: Vec<TrackObs>,
    pub camera_health: CameraHealth,
    pub liveness_tier: LivenessTier,
    pub timings: StageTimings,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{PointF, RectF};

    #[test]
    fn vision_report_serde_roundtrip() {
        let p = |x, y| PointF { x, y };
        let track = |id, identity, liveness, head_pose, embedding| TrackObs {
            id: TrackId(id),
            bbox: RectF {
                x: 0.1,
                y: 0.2,
                w: 0.3,
                h: 0.4,
            },
            landmarks5: [
                p(0.2, 0.3),
                p(0.3, 0.3),
                p(0.25, 0.35),
                p(0.21, 0.4),
                p(0.29, 0.4),
            ],
            det_score: 0.9,
            identity,
            liveness,
            head_pose,
            face_width_px: 120.0,
            age_ms: 1500,
            embedding,
        };
        let report = VisionReport {
            seq: 42,
            ts_ms: 123_456,
            tracks: vec![
                track(
                    1,
                    IdentityState::Known {
                        person: "owner".into(),
                        score: 0.71,
                    },
                    LivenessState::Live {
                        score: 0.95,
                        tier: LivenessTier::RgbTemporal,
                    },
                    Some(HeadPose {
                        yaw: -10.0,
                        pitch: 5.0,
                        roll: 1.5,
                        confidence: 0.8,
                    }),
                    Some(vec![0.1, -0.2, 0.3]),
                ),
                track(
                    2,
                    IdentityState::Stranger { best: 0.12 },
                    LivenessState::Spoof {
                        score: 0.2,
                        tier: LivenessTier::Rgb,
                    },
                    None,
                    None,
                ),
                track(
                    3,
                    IdentityState::Pending,
                    LivenessState::Unknown,
                    None,
                    None,
                ),
                track(
                    4,
                    IdentityState::Unknown,
                    LivenessState::Unknown,
                    None,
                    None,
                ),
            ],
            camera_health: CameraHealth::Error("device gone".into()),
            liveness_tier: LivenessTier::Rgb,
            timings: StageTimings {
                detect_ms: 4.5,
                recognize_ms: 3.0,
                liveness_ms: 1.0,
                pose_ms: 0.5,
            },
        };
        let json = serde_json::to_string(&report).unwrap();
        let back: VisionReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, report);
    }

    #[test]
    fn track_obs_debug_redacts_embedding() {
        let obs = TrackObs {
            id: TrackId(7),
            bbox: RectF {
                x: 0.1,
                y: 0.2,
                w: 0.3,
                h: 0.4,
            },
            landmarks5: [PointF { x: 0.5, y: 0.5 }; 5],
            det_score: 0.9,
            identity: IdentityState::Pending,
            liveness: LivenessState::Unknown,
            head_pose: None,
            face_width_px: 120.0,
            age_ms: 0,
            embedding: Some(vec![0.123_456_7, -7.654_321, 31.4159]),
        };
        let dbg = format!("{obs:?}");
        for n in ["123456", "654321", "31.41"] {
            assert!(!dbg.contains(n), "{dbg}");
        }
        assert!(
            dbg.contains("embedding: Some(<3 floats redacted>)"),
            "{dbg}"
        );
        assert!(dbg.contains("TrackId(7)"), "{dbg}");
        let report = format!("{:#?}", vec![obs.clone()]);
        assert!(!report.contains("123456"), "{report}");
        let none = TrackObs {
            embedding: None,
            ..obs
        };
        assert!(format!("{none:?}").contains("embedding: None"));
    }

    #[test]
    fn track_id_is_ordered_and_hashable() {
        let set: std::collections::HashSet<TrackId> = [TrackId(2), TrackId(1), TrackId(2)].into();
        assert_eq!(set.len(), 2);
        assert!(TrackId(1) < TrackId(2));
    }
}
