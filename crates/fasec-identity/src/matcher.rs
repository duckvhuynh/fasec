//! Margin-gated cosine matcher over the enrolled gallery (FR-ID-009).

use crate::store::{PersonId, Role};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatcherConfig {
    pub threshold: f32,
    pub margin: f32,
}

impl Default for MatcherConfig {
    fn default() -> Self {
        Self {
            threshold: 0.363,
            margin: 0.05,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatchResult {
    Known {
        person: PersonId,
        role: Role,
        score: f32,
    },
    Stranger {
        best: f32,
    },
    Ambiguous {
        best: f32,
        second: f32,
    },
}

/// Holds normalised templates; `Debug` deliberately omits them (NFR-PRIV-003).
pub struct Matcher {
    persons: Vec<(PersonId, Role, Vec<Vec<f32>>)>,
    cfg: MatcherConfig,
    /// False if any template was non-finite, zero or of a different length.
    valid: bool,
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matcher")
            .field("persons", &self.persons.len())
            .field("cfg", &self.cfg)
            .field("valid", &self.valid)
            .finish_non_exhaustive()
    }
}

impl Matcher {
    /// One entry per template. Any unusable template disables matching entirely:
    /// silently dropping it could widen another person's margin.
    pub fn new(templates: Vec<(PersonId, Role, Vec<f32>)>, cfg: MatcherConfig) -> Self {
        let dim = templates.first().map(|t| t.2.len());
        let mut valid = true;
        let mut persons: Vec<(PersonId, Role, Vec<Vec<f32>>)> = Vec::new();
        for (person, role, t) in templates {
            let Some(t) = normalised(&t).filter(|t| Some(t.len()) == dim) else {
                tracing::warn!(?person, "unusable template; matching disabled");
                valid = false;
                continue;
            };
            match persons.iter_mut().find(|p| p.0 == person) {
                Some(p) => p.2.push(t),
                None => persons.push((person, role, vec![t])),
            }
        }
        Self {
            persons,
            cfg,
            valid,
        }
    }

    pub fn best(&self, embedding: &[f32]) -> MatchResult {
        const NONE: MatchResult = MatchResult::Stranger { best: 0.0 };
        if !self.valid {
            return NONE;
        }
        let Some(q) = normalised(embedding) else {
            return NONE;
        };
        // (index, score) of the best and second-best person.
        let mut top: Option<(usize, f32)> = None;
        let mut second: Option<f32> = None;
        for (i, (_, _, ts)) in self.persons.iter().enumerate() {
            let mut score = f32::NEG_INFINITY;
            for t in ts {
                if t.len() != q.len() {
                    return NONE;
                }
                score = score.max(t.iter().zip(&q).map(|(a, b)| a * b).sum());
            }
            if !score.is_finite() {
                return NONE;
            }
            match top {
                Some((_, b)) if score <= b => second = Some(second.map_or(score, |s| s.max(score))),
                _ => {
                    second = top.map(|t| t.1);
                    top = Some((i, score));
                }
            }
        }
        let Some((i, best)) = top else {
            return NONE;
        };
        if best < self.cfg.threshold {
            return MatchResult::Stranger { best };
        }
        match second {
            Some(second) if best - second < self.cfg.margin => {
                MatchResult::Ambiguous { best, second }
            }
            _ => {
                let (person, role, _) = &self.persons[i];
                MatchResult::Known {
                    person: person.clone(),
                    role: *role,
                    score: best,
                }
            }
        }
    }
}

/// L2-normalised copy, or `None` if empty, non-finite or zero.
fn normalised(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm.is_finite() && norm > 0.0).then(|| v.iter().map(|x| x / norm).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 128;

    /// Unit vector with cosine `cos` to axis 0, rotated towards `axis`.
    fn at(cos: f32, axis: usize) -> Vec<f32> {
        let mut v = vec![0.0; DIM];
        v[0] = cos;
        v[axis] = (1.0 - cos * cos).sqrt();
        v
    }

    fn pid(s: &str) -> PersonId {
        PersonId(s.into())
    }

    fn query() -> Vec<f32> {
        at(1.0, 1)
    }

    #[test]
    fn exact_template_is_known() {
        let m = Matcher::new(
            vec![(pid("a"), Role::Owner, at(1.0, 1))],
            MatcherConfig::default(),
        );
        match m.best(&query()) {
            MatchResult::Known {
                person,
                role,
                score,
            } => {
                assert_eq!(person, pid("a"));
                assert_eq!(role, Role::Owner);
                assert!((score - 1.0).abs() < 1e-5);
            }
            other => panic!("expected Known, got {other:?}"),
        }
    }

    #[test]
    fn unnormalised_template_and_query_still_match() {
        let t: Vec<f32> = at(0.9, 1).iter().map(|x| x * 7.0).collect();
        let q: Vec<f32> = query().iter().map(|x| x * 0.01).collect();
        let m = Matcher::new(vec![(pid("a"), Role::Owner, t)], MatcherConfig::default());
        match m.best(&q) {
            MatchResult::Known { score, .. } => assert!((score - 0.9).abs() < 1e-5),
            other => panic!("expected Known, got {other:?}"),
        }
    }

    #[test]
    fn below_threshold_is_stranger() {
        let m = Matcher::new(
            vec![(pid("a"), Role::Owner, at(0.30, 1))],
            MatcherConfig::default(),
        );
        match m.best(&query()) {
            MatchResult::Stranger { best } => assert!((best - 0.30).abs() < 1e-5),
            other => panic!("expected Stranger, got {other:?}"),
        }
    }

    #[test]
    fn margin_too_small_is_ambiguous() {
        let m = Matcher::new(
            vec![
                (pid("a"), Role::Owner, at(0.50, 1)),
                (pid("b"), Role::Authorised, at(0.47, 2)),
            ],
            MatcherConfig::default(),
        );
        match m.best(&query()) {
            MatchResult::Ambiguous { best, second } => {
                assert!((best - 0.50).abs() < 1e-5);
                assert!((second - 0.47).abs() < 1e-5);
            }
            other => panic!("expected Ambiguous, got {other:?}"),
        }
    }

    #[test]
    fn margin_large_enough_is_known() {
        let m = Matcher::new(
            vec![
                (pid("a"), Role::Owner, at(0.50, 1)),
                (pid("b"), Role::Authorised, at(0.40, 2)),
            ],
            MatcherConfig::default(),
        );
        assert!(
            matches!(m.best(&query()), MatchResult::Known { person, .. } if person == pid("a"))
        );
    }

    #[test]
    fn max_over_templates_per_person() {
        // a: weak + strong template; b: one template between them.
        // Margin is measured between persons, so a's two templates never compete.
        let m = Matcher::new(
            vec![
                (pid("a"), Role::Owner, at(0.20, 1)),
                (pid("b"), Role::Authorised, at(0.60, 2)),
                (pid("a"), Role::Owner, at(0.90, 3)),
                (pid("a"), Role::Owner, at(0.88, 4)),
            ],
            MatcherConfig::default(),
        );
        match m.best(&query()) {
            MatchResult::Known { person, score, .. } => {
                assert_eq!(person, pid("a"));
                assert!((score - 0.90).abs() < 1e-5);
            }
            other => panic!("expected Known, got {other:?}"),
        }
    }

    #[test]
    fn empty_gallery_is_stranger() {
        let m = Matcher::new(vec![], MatcherConfig::default());
        assert_eq!(m.best(&query()), MatchResult::Stranger { best: 0.0 });
    }

    #[test]
    fn nan_embedding_is_stranger() {
        let m = Matcher::new(
            vec![(pid("a"), Role::Owner, at(1.0, 1))],
            MatcherConfig::default(),
        );
        let mut q = query();
        q[5] = f32::NAN;
        assert_eq!(m.best(&q), MatchResult::Stranger { best: 0.0 });
        q[5] = f32::INFINITY;
        assert_eq!(m.best(&q), MatchResult::Stranger { best: 0.0 });
        assert_eq!(m.best(&[0.0; DIM]), MatchResult::Stranger { best: 0.0 });
        assert_eq!(m.best(&[]), MatchResult::Stranger { best: 0.0 });
    }

    #[test]
    fn wrong_length_embedding_is_stranger() {
        let m = Matcher::new(
            vec![(pid("a"), Role::Owner, at(1.0, 1))],
            MatcherConfig::default(),
        );
        assert_eq!(m.best(&query()[..64]), MatchResult::Stranger { best: 0.0 });
    }

    #[test]
    fn bad_template_disables_matching() {
        // Dropping b's corrupt template would widen a's margin; refuse instead.
        let mut bad = at(0.47, 2);
        bad[3] = f32::NAN;
        for b in [bad, vec![0.0; DIM], at(0.47, 2)[..64].to_vec()] {
            let m = Matcher::new(
                vec![
                    (pid("a"), Role::Owner, at(0.50, 1)),
                    (pid("b"), Role::Authorised, b),
                ],
                MatcherConfig::default(),
            );
            assert_eq!(m.best(&query()), MatchResult::Stranger { best: 0.0 });
        }
    }

    #[test]
    fn debug_omits_templates() {
        let m = Matcher::new(
            vec![(pid("a"), Role::Owner, at(0.123_456, 1))],
            MatcherConfig::default(),
        );
        assert!(!format!("{m:?}").contains("0.123"));
    }
}
