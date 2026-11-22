//! Per-frame vision output.

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

/// One tracked face in one frame. Coordinates are normalised 0..1. Embeddings never travel
/// here (NFR-PRIV-003): see `fasec_core::TrackEmbedding`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
        let track = |id, identity, liveness, head_pose| TrackObs {
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
                ),
                track(
                    2,
                    IdentityState::Stranger { best: 0.12 },
                    LivenessState::Spoof {
                        score: 0.2,
                        tier: LivenessTier::Rgb,
                    },
                    None,
                ),
                track(3, IdentityState::Pending, LivenessState::Unknown, None),
                track(4, IdentityState::Unknown, LivenessState::Unknown, None),
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

    fn sample_track() -> TrackObs {
        TrackObs {
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
        }
    }

    #[test]
    fn vision_report_json_has_no_embedding_key() {
        let report = VisionReport {
            seq: 1,
            ts_ms: 2,
            tracks: vec![sample_track()],
            camera_health: CameraHealth::Ok,
            liveness_tier: LivenessTier::Rgb,
            timings: StageTimings {
                detect_ms: 0.0,
                recognize_ms: 0.0,
                liveness_ms: 0.0,
                pose_ms: 0.0,
            },
        };
        let v = serde_json::to_value(&report).unwrap();
        let track = v["tracks"][0].as_object().unwrap();
        assert!(track.contains_key("bbox"), "{v}");
        assert!(
            !serde_json::to_string(&report)
                .unwrap()
                .contains("embedding"),
            "{v}"
        );
    }

    #[test]
    fn track_id_is_ordered_and_hashable() {
        let set: std::collections::HashSet<TrackId> = [TrackId(2), TrackId(1), TrackId(2)].into();
        assert_eq!(set.len(), 2);
        assert!(TrackId(1) < TrackId(2));
    }
}
