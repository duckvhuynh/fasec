//! Per-frame identity: each new embedding goes matcher → voter, and every track in the
//! report gets the voter's current state (FR-ID-009, SRS §5.1).

use std::collections::HashMap;

use fasec_core::{IdentityState, TrackEmbedding, TrackId, VisionReport};

use crate::error::IdentityError;
use crate::matcher::{Matcher, MatcherConfig};
use crate::store::Gallery;
use crate::voter::{Voter, VoterConfig};

#[derive(Debug)]
pub struct IdentityTracker {
    matcher: Matcher,
    voter: Voter,
    model_id: String,
    /// The voter's latest state per live track, for frames without a new embedding.
    states: HashMap<TrackId, IdentityState>,
}

impl IdentityTracker {
    /// Loads the templates enrolled with `current_model_id`. A gallery holding templates only
    /// for other models is `ModelMismatch` (FR-ID-008); an empty gallery is fine (everyone
    /// becomes `Stranger`). Any load or decrypt error is returned, never a partial matcher.
    pub fn from_gallery(
        gallery: &Gallery,
        current_model_id: &str,
        matcher: MatcherConfig,
        voter: VoterConfig,
    ) -> Result<IdentityTracker, IdentityError> {
        let in_use = gallery.model_ids_in_use()?;
        if !in_use.is_empty() && !in_use.iter().any(|m| m == current_model_id) {
            return Err(IdentityError::ModelMismatch {
                stored: in_use,
                current: current_model_id.to_owned(),
            });
        }
        let templates = gallery
            .templates_for_model(current_model_id)?
            .into_iter()
            // `take` hands the buffer over without making another plaintext copy.
            .map(|(p, r, mut e)| (p, r, std::mem::take(&mut *e)))
            .collect();
        Ok(IdentityTracker {
            matcher: Matcher::new(templates, matcher)?,
            voter: Voter::new(voter),
            model_id: current_model_id.to_owned(),
            states: HashMap::new(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Feeds `embeddings` (those computed on this frame) through matcher and voter, then
    /// writes `identity` for every track in `report`: `Pending` until the voter decides.
    /// Voter state of tracks no longer in `report` is dropped.
    pub fn apply(&mut self, report: &mut VisionReport, embeddings: &[TrackEmbedding]) {
        let live: Vec<TrackId> = report.tracks.iter().map(|t| t.id).collect();
        self.voter.retain(&live);
        self.states.retain(|t, _| live.contains(t));
        for te in embeddings.iter().filter(|te| live.contains(&te.track)) {
            let result = self.matcher.best(te.embedding.as_slice());
            let state = self.voter.observe(te.track, &result, te.ts_ms);
            self.states.insert(te.track, state);
        }
        for t in &mut report.tracks {
            t.identity = self
                .states
                .get(&t.id)
                .cloned()
                .unwrap_or(IdentityState::Pending);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use fasec_core::{
        CameraHealth, LivenessState, LivenessTier, PointF, RectF, StageTimings, TrackObs,
    };

    use super::*;
    use crate::protect::testing::XorProtector;
    use crate::store::{EMBEDDING_DIM, Pose, Role};

    const MODEL: &str = "sface-2021dec";

    fn unit(i: usize) -> Vec<f32> {
        let mut v = vec![0.0; EMBEDDING_DIM];
        v[i] = 1.0;
        v
    }

    /// Gallery with "owner" enrolled as `unit(0)` under `model`.
    fn gallery(model: &str) -> Gallery {
        let mut g = Gallery::open_in_memory(Arc::new(XorProtector::default())).unwrap();
        let p = g.add_person("owner", Role::Owner).unwrap();
        g.add_template(&p, model, Pose::Front, 0.9, &unit(0))
            .unwrap();
        g
    }

    fn tracker(g: &Gallery) -> IdentityTracker {
        IdentityTracker::from_gallery(g, MODEL, MatcherConfig::default(), VoterConfig::default())
            .unwrap()
    }

    fn report(ts_ms: u64, ids: &[u64]) -> VisionReport {
        let p = PointF { x: 0.5, y: 0.5 };
        VisionReport {
            seq: ts_ms,
            ts_ms,
            tracks: ids
                .iter()
                .map(|&id| TrackObs {
                    id: TrackId(id),
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
                })
                .collect(),
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

    fn emb(track: u64, ts_ms: u64, v: Vec<f32>) -> TrackEmbedding {
        TrackEmbedding {
            track: TrackId(track),
            ts_ms,
            embedding: v.into(),
        }
    }

    fn ids(r: &VisionReport) -> Vec<IdentityState> {
        r.tracks.iter().map(|t| t.identity.clone()).collect()
    }

    fn is_known_owner(s: &IdentityState) -> bool {
        matches!(s, IdentityState::Known { score, .. } if (*score - 1.0).abs() < 1e-5)
    }

    #[test]
    fn tracker_marks_known_after_vote() {
        let g = gallery(MODEL);
        let mut t = tracker(&g);
        let owner = g.persons().unwrap()[0].id.0.clone();
        // track 1 is the owner; track 2 is orthogonal to every template
        for i in 0..5u64 {
            let ts = i * 1000;
            let mut r = report(ts, &[1, 2]);
            t.apply(&mut r, &[emb(1, ts, unit(0)), emb(2, ts, unit(1))]);
            let s = ids(&r);
            assert!(is_known_owner(&s[0]), "frame {i}: {s:?}");
            assert!(
                matches!(&s[0], IdentityState::Known { person, .. } if *person == owner),
                "{s:?}"
            );
            let want_stranger = i == 4; // voter window = 5 non-matches
            assert_eq!(
                matches!(s[1], IdentityState::Stranger { .. }),
                want_stranger,
                "frame {i}: {s:?}"
            );
            if !want_stranger {
                assert_eq!(s[1], IdentityState::Pending, "frame {i}");
            }
        }
    }

    #[test]
    fn tracker_rejects_other_model_gallery() {
        let g = gallery("other-model");
        let e = IdentityTracker::from_gallery(
            &g,
            MODEL,
            MatcherConfig::default(),
            VoterConfig::default(),
        )
        .unwrap_err();
        match e {
            IdentityError::ModelMismatch { stored, current } => {
                assert_eq!(
                    (stored, current),
                    (vec!["other-model".into()], MODEL.into())
                );
            }
            e => panic!("{e:?}"),
        }
        // An empty gallery is not a mismatch: everyone ends up a stranger.
        let empty = Gallery::open_in_memory(Arc::new(XorProtector::default())).unwrap();
        let mut t = tracker(&empty);
        let mut r = report(0, &[1]);
        for ts in 0..5 {
            r = report(ts, &[1]);
            t.apply(&mut r, &[emb(1, ts, unit(0))]);
        }
        assert!(matches!(ids(&r)[0], IdentityState::Stranger { .. }));
    }

    #[test]
    fn tracker_fails_on_undecryptable_gallery() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.db");
        {
            let mut g = Gallery::open(&path, Arc::new(XorProtector(1))).unwrap();
            let p = g.add_person("owner", Role::Owner).unwrap();
            g.add_template(&p, MODEL, Pose::Front, 0.9, &unit(0))
                .unwrap();
        }
        let g = Gallery::open(&path, Arc::new(XorProtector(2))).unwrap();
        let e = IdentityTracker::from_gallery(
            &g,
            MODEL,
            MatcherConfig::default(),
            VoterConfig::default(),
        );
        assert!(
            matches!(e, Err(IdentityError::CorruptTemplate { .. })),
            "{e:?}"
        );
        let bad_cfg = MatcherConfig {
            threshold: 0.0,
            margin: 0.05,
        };
        let e =
            IdentityTracker::from_gallery(&gallery(MODEL), MODEL, bad_cfg, VoterConfig::default());
        assert!(matches!(e, Err(IdentityError::InvalidConfig(_))), "{e:?}");
    }

    #[test]
    fn tracker_keeps_pending_without_embedding() {
        let g = gallery(MODEL);
        let mut t = tracker(&g);
        let mut r = report(0, &[1, 2]);
        t.apply(&mut r, &[]);
        assert_eq!(ids(&r), [IdentityState::Pending, IdentityState::Pending]);
        // track 1 is recognised once; later frames without embeddings keep its decision
        let mut r = report(100, &[1, 2]);
        t.apply(&mut r, &[emb(1, 100, unit(0))]);
        assert!(is_known_owner(&ids(&r)[0]));
        assert_eq!(ids(&r)[1], IdentityState::Pending);
        let mut r = report(200, &[1, 2]);
        t.apply(&mut r, &[]);
        assert!(is_known_owner(&ids(&r)[0]), "{:?}", ids(&r));
        assert_eq!(ids(&r)[1], IdentityState::Pending);
    }

    #[test]
    fn tracker_forgets_dead_tracks() {
        let g = gallery(MODEL);
        let mut t = tracker(&g);
        let mut r = report(0, &[1]);
        t.apply(&mut r, &[emb(1, 0, unit(0))]);
        assert!(is_known_owner(&ids(&r)[0]));
        // track 1 leaves; if the same id came back it must start over
        t.apply(&mut report(100, &[2]), &[]);
        let mut r = report(200, &[1]);
        t.apply(&mut r, &[]);
        assert_eq!(ids(&r), [IdentityState::Pending]);
        // and the voter starts over too: one non-match does not decide anything
        let mut r = report(300, &[1]);
        t.apply(&mut r, &[emb(1, 300, unit(1))]);
        assert_eq!(ids(&r), [IdentityState::Pending]);
    }
}
