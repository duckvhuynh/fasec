//! Per-track identity vote (SRS §5.1): one bad frame never flips identity.

use std::collections::HashMap;

use fasec_core::{IdentityState, TrackId};

use crate::matcher::MatchResult;
use crate::store::PersonId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoterConfig {
    /// Results without an accepted match before a new track becomes `Stranger`.
    pub window: usize,
    /// Consecutive contrary results needed to change a decided identity.
    pub flip_after: usize,
}

impl Default for VoterConfig {
    fn default() -> Self {
        Self {
            window: 5,
            flip_after: 3,
        }
    }
}

#[derive(Debug, Default)]
pub struct Voter {
    cfg: VoterConfig,
    tracks: HashMap<TrackId, TrackVote>,
}

#[derive(Debug, Default)]
enum Decision {
    #[default]
    Pending,
    Known(PersonId, f32),
    Stranger(f32),
}

#[derive(Debug, Default)]
struct TrackVote {
    decision: Decision,
    /// Consecutive results that disagree with a `Known` decision.
    contrary: usize,
    /// Latest verdict (`None` = no accepted match) and how many times in a row it occurred.
    run: (Option<PersonId>, usize),
}

impl Voter {
    pub fn new(cfg: VoterConfig) -> Self {
        Self {
            cfg,
            tracks: HashMap::new(),
        }
    }

    /// `_ts_ms` is unused for now: track expiry is driven by `retain`/`forget`.
    pub fn observe(&mut self, track: TrackId, result: &MatchResult, _ts_ms: u64) -> IdentityState {
        let window = self.cfg.window.max(1);
        let flip = self.cfg.flip_after.max(1);
        let (verdict, score) = match result {
            MatchResult::Known { person, score, .. } => (Some(person), *score),
            MatchResult::Stranger { best } | MatchResult::Ambiguous { best, .. } => (None, *best),
        };
        let v = self.tracks.entry(track).or_default();
        if v.run.0.as_ref() == verdict {
            v.run.1 += 1;
        } else {
            v.run = (verdict.cloned(), 1);
        }
        // A person who has won `flip` results in a row, if any.
        let streak_winner = match &v.run {
            (Some(p), n) if *n >= flip => Some(p.clone()),
            _ => None,
        };
        v.decision = match std::mem::take(&mut v.decision) {
            Decision::Pending => match verdict {
                Some(p) => Decision::Known(p.clone(), score),
                // In Pending every result so far was a non-match, so `run.1` counts them all.
                None if v.run.1 >= window => Decision::Stranger(score),
                None => Decision::Pending,
            },
            Decision::Known(p, _) if verdict == Some(&p) => {
                v.contrary = 0;
                Decision::Known(p, score)
            }
            Decision::Known(p, s) => {
                v.contrary += 1;
                if v.contrary < flip {
                    Decision::Known(p, s)
                } else {
                    v.contrary = 0;
                    match streak_winner {
                        Some(q) => Decision::Known(q, score),
                        None => Decision::Stranger(score),
                    }
                }
            }
            Decision::Stranger(_) => match streak_winner {
                Some(q) => Decision::Known(q, score),
                None => Decision::Stranger(score),
            },
        };
        match &v.decision {
            Decision::Pending => IdentityState::Pending,
            Decision::Known(p, s) => IdentityState::Known {
                person: p.0.clone(),
                score: *s,
            },
            Decision::Stranger(best) => IdentityState::Stranger { best: *best },
        }
    }

    pub fn forget(&mut self, track: TrackId) {
        self.tracks.remove(&track);
    }

    pub fn retain(&mut self, live: &[TrackId]) {
        self.tracks.retain(|t, _| live.contains(t));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Role;

    const T: TrackId = TrackId(1);

    fn known(p: &str, score: f32) -> MatchResult {
        MatchResult::Known {
            person: PersonId(p.into()),
            role: Role::Owner,
            score,
        }
    }

    fn stranger(best: f32) -> MatchResult {
        MatchResult::Stranger { best }
    }

    fn is_known(s: &IdentityState, p: &str) -> bool {
        matches!(s, IdentityState::Known { person, .. } if person == p)
    }

    fn feed(v: &mut Voter, t: TrackId, rs: &[MatchResult]) -> IdentityState {
        let mut last = IdentityState::Pending;
        for (i, r) in rs.iter().enumerate() {
            last = v.observe(t, r, i as u64 * 33);
        }
        last
    }

    #[test]
    fn first_known_is_known() {
        let mut v = Voter::default();
        assert_eq!(
            v.observe(T, &known("a", 0.7), 0),
            IdentityState::Known {
                person: "a".into(),
                score: 0.7
            }
        );
    }

    #[test]
    fn single_contrary_frame_does_not_flip() {
        let mut v = Voter::default();
        feed(&mut v, T, &[known("a", 0.7), known("a", 0.8)]);
        for bad in [
            stranger(0.1),
            known("b", 0.9),
            MatchResult::Ambiguous {
                best: 0.5,
                second: 0.48,
            },
        ] {
            let s = v.observe(T, &bad, 100);
            // Score stays the latest *accepted* one.
            assert_eq!(
                s,
                IdentityState::Known {
                    person: "a".into(),
                    score: 0.8
                }
            );
            assert!(is_known(&v.observe(T, &known("a", 0.8), 133), "a"));
        }
        // Two contrary then a good one resets the streak.
        let s = feed(
            &mut v,
            T,
            &[
                stranger(0.1),
                stranger(0.1),
                known("a", 0.75),
                stranger(0.1),
                stranger(0.1),
            ],
        );
        assert!(is_known(&s, "a"));
    }

    #[test]
    fn flips_after_three_consecutive() {
        let mut v = Voter::default();
        feed(&mut v, T, &[known("a", 0.7)]);
        assert!(is_known(
            &feed(&mut v, T, &[stranger(0.1), stranger(0.2)]),
            "a"
        ));
        assert_eq!(
            v.observe(T, &stranger(0.15), 0),
            IdentityState::Stranger { best: 0.15 }
        );

        // Three consecutive results for another person hand the track over.
        let mut v = Voter::default();
        feed(&mut v, T, &[known("a", 0.7)]);
        let s = feed(
            &mut v,
            T,
            &[known("b", 0.6), known("b", 0.65), known("b", 0.62)],
        );
        assert_eq!(
            s,
            IdentityState::Known {
                person: "b".into(),
                score: 0.62
            }
        );

        // Mixed contrary results flip to Stranger, never to the last person seen.
        let mut v = Voter::default();
        feed(&mut v, T, &[known("a", 0.7)]);
        let s = feed(
            &mut v,
            T,
            &[stranger(0.1), known("b", 0.6), known("b", 0.6)],
        );
        assert_eq!(s, IdentityState::Stranger { best: 0.6 });
    }

    #[test]
    fn stranger_after_window_without_match() {
        let mut v = Voter::default();
        let s = feed(
            &mut v,
            T,
            &[stranger(0.1), stranger(0.2), stranger(0.1), stranger(0.3)],
        );
        assert_eq!(s, IdentityState::Pending);
        assert_eq!(
            v.observe(T, &stranger(0.25), 0),
            IdentityState::Stranger { best: 0.25 }
        );
        // A decided Stranger needs a full flip streak to become Known.
        assert!(!is_known(
            &feed(&mut v, T, &[known("a", 0.7), known("a", 0.7)]),
            "a"
        ));
        assert!(is_known(&v.observe(T, &known("a", 0.7), 0), "a"));
    }

    #[test]
    fn ambiguous_never_known() {
        let amb = MatchResult::Ambiguous {
            best: 0.5,
            second: 0.47,
        };
        let mut v = Voter::default();
        for _ in 0..4 {
            assert_eq!(v.observe(T, &amb, 0), IdentityState::Pending);
        }
        for _ in 0..10 {
            assert_eq!(v.observe(T, &amb, 0), IdentityState::Stranger { best: 0.5 });
        }
    }

    #[test]
    fn forget_resets_track() {
        let mut v = Voter::default();
        feed(&mut v, T, &vec![stranger(0.1); 5]);
        v.forget(T);
        assert!(is_known(&v.observe(T, &known("a", 0.7), 0), "a"));
    }

    #[test]
    fn tracks_are_independent() {
        let mut v = Voter::default();
        feed(&mut v, T, &[known("a", 0.7)]);
        assert_eq!(
            v.observe(TrackId(2), &stranger(0.1), 0),
            IdentityState::Pending
        );
        assert!(is_known(&v.observe(T, &stranger(0.1), 0), "a"));
    }

    #[test]
    fn retain_drops_dead_tracks() {
        let mut v = Voter::default();
        feed(&mut v, TrackId(1), &vec![stranger(0.1); 5]);
        feed(&mut v, TrackId(2), &vec![stranger(0.1); 5]);
        v.retain(&[TrackId(2)]);
        assert!(is_known(&v.observe(TrackId(1), &known("a", 0.7), 0), "a"));
        assert!(!is_known(&v.observe(TrackId(2), &known("a", 0.7), 0), "a"));
    }

    #[test]
    fn zero_config_is_clamped() {
        let mut v = Voter::new(VoterConfig {
            window: 0,
            flip_after: 0,
        });
        feed(&mut v, T, &[known("a", 0.7)]);
        assert!(is_known(&v.observe(T, &known("a", 0.7), 0), "a"));
        assert_eq!(
            v.observe(T, &stranger(0.1), 0),
            IdentityState::Stranger { best: 0.1 }
        );
    }
}
