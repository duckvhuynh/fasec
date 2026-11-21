//! SORT-style IoU tracker: stable ids across frames via constant-velocity prediction
//! and Hungarian assignment.

use fasec_core::{RectF, TrackId};
use pathfinding::kuhn_munkres::kuhn_munkres;
use pathfinding::matrix::Matrix;

use crate::Detection;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackerConfig {
    /// Minimum IoU between a predicted track box and a detection to match (0.3).
    pub iou_match: f32,
    /// A track without a match for longer than this is dropped (1000 ms, SRS §5.1).
    pub track_ttl_ms: u64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            iou_match: 0.3,
            track_ttl_ms: 1000,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackedDet {
    pub id: TrackId,
    pub det: Detection,
    /// Time since the track was first seen.
    pub age_ms: u64,
    /// True only on the update that created the track.
    pub is_new: bool,
}

/// IoU is scaled to an integer weight for `kuhn_munkres`.
const IOU_SCALE: f32 = 10_000.0;

struct Track {
    id: TrackId,
    /// Last matched box.
    bbox: RectF,
    /// Per-ms velocity of (cx, cy, w, h), from the last two matches.
    vel: [f32; 4],
    first_ts: u64,
    last_ts: u64,
}

impl Track {
    /// Constant-velocity prediction of the box at `ts_ms`.
    fn predict(&self, ts_ms: u64) -> RectF {
        let dt = ts_ms.saturating_sub(self.last_ts) as f32;
        let c = self.bbox.center();
        let w = (self.bbox.w + self.vel[2] * dt).max(0.0);
        let h = (self.bbox.h + self.vel[3] * dt).max(0.0);
        RectF {
            x: c.x + self.vel[0] * dt - w / 2.0,
            y: c.y + self.vel[1] * dt - h / 2.0,
            w,
            h,
        }
    }

    fn observe(&mut self, ts_ms: u64, bbox: RectF) {
        let dt = ts_ms.saturating_sub(self.last_ts);
        if dt > 0 {
            let dt = dt as f32;
            let (a, b) = (self.bbox.center(), bbox.center());
            self.vel = [
                (b.x - a.x) / dt,
                (b.y - a.y) / dt,
                (bbox.w - self.bbox.w) / dt,
                (bbox.h - self.bbox.h) / dt,
            ];
        }
        self.bbox = bbox;
        self.last_ts = ts_ms;
    }
}

pub struct Tracker {
    cfg: TrackerConfig,
    tracks: Vec<Track>,
    next_id: u64,
}

impl Tracker {
    pub fn new(cfg: TrackerConfig) -> Self {
        Self {
            cfg,
            tracks: Vec::new(),
            next_id: 1,
        }
    }

    /// Matches `dets` to live tracks and returns one entry per detection, in `dets` order.
    /// `ts_ms` must be non-decreasing across calls.
    pub fn update(&mut self, ts_ms: u64, dets: &[Detection]) -> Vec<TrackedDet> {
        let ttl = self.cfg.track_ttl_ms;
        self.tracks
            .retain(|t| ts_ms.saturating_sub(t.last_ts) <= ttl);

        let mut det_track: Vec<Option<usize>> = vec![None; dets.len()];
        for (ti, di) in self.assign(ts_ms, dets) {
            det_track[di] = Some(ti);
        }

        let mut out = Vec::with_capacity(dets.len());
        for (det, slot) in dets.iter().zip(det_track) {
            let (ti, is_new) = match slot {
                Some(ti) => {
                    self.tracks[ti].observe(ts_ms, det.bbox);
                    (ti, false)
                }
                None => {
                    self.tracks.push(Track {
                        id: TrackId(self.next_id),
                        bbox: det.bbox,
                        vel: [0.0; 4],
                        first_ts: ts_ms,
                        last_ts: ts_ms,
                    });
                    self.next_id += 1;
                    (self.tracks.len() - 1, true)
                }
            };
            let t = &self.tracks[ti];
            out.push(TrackedDet {
                id: t.id,
                det: *det,
                age_ms: ts_ms.saturating_sub(t.first_ts),
                is_new,
            });
        }
        out
    }

    /// Hungarian assignment maximising IoU of predicted track boxes vs detections.
    /// Returns accepted `(track index, detection index)` pairs.
    fn assign(&self, ts_ms: u64, dets: &[Detection]) -> Vec<(usize, usize)> {
        if self.tracks.is_empty() || dets.is_empty() {
            return Vec::new();
        }
        let preds: Vec<RectF> = self.tracks.iter().map(|t| t.predict(ts_ms)).collect();
        let iou = |ti: usize, di: usize| preds[ti].iou(&dets[di].bbox);
        let weight = |ti: usize, di: usize| {
            let v = iou(ti, di);
            // Sub-threshold pairs weigh nothing so they never outbid a valid pairing.
            if v >= self.cfg.iou_match {
                (v * IOU_SCALE).round() as i64
            } else {
                0
            }
        };
        // kuhn_munkres needs rows <= columns: put the shorter side on the rows.
        let pairs: Vec<(usize, usize)> = if self.tracks.len() <= dets.len() {
            let m = Matrix::from_fn(self.tracks.len(), dets.len(), |(t, d)| weight(t, d));
            let (_, cols) = kuhn_munkres(&m);
            cols.into_iter().enumerate().collect()
        } else {
            let m = Matrix::from_fn(dets.len(), self.tracks.len(), |(d, t)| weight(t, d));
            let (_, cols) = kuhn_munkres(&m);
            cols.into_iter().enumerate().map(|(d, t)| (t, d)).collect()
        };
        pairs
            .into_iter()
            .filter(|&(t, d)| iou(t, d) >= self.cfg.iou_match)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use fasec_core::PointF;

    use super::*;

    fn det(x: f32, y: f32) -> Detection {
        Detection {
            bbox: RectF {
                x,
                y,
                w: 0.2,
                h: 0.2,
            },
            landmarks5: [PointF { x, y }; 5],
            score: 0.9,
        }
    }

    fn tracker() -> Tracker {
        Tracker::new(TrackerConfig::default())
    }

    fn ids(out: &[TrackedDet]) -> Vec<u64> {
        out.iter().map(|t| t.id.0).collect()
    }

    #[test]
    fn single_box_keeps_id_while_moving() {
        let mut t = tracker();
        let first = t.update(0, &[det(0.1, 0.3)]);
        assert_eq!(first.len(), 1);
        assert!(first[0].is_new);
        let id = first[0].id;
        for k in 1..10u64 {
            let out = t.update(k * 33, &[det(0.1 + 0.05 * k as f32, 0.3)]);
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].id, id, "step {k}");
            assert!(!out[0].is_new);
            assert_eq!(out[0].age_ms, k * 33);
        }
    }

    #[test]
    fn two_crossing_boxes_keep_ids() {
        // Same row, they pass through each other between steps 4 and 5. Without
        // velocity prediction plain IoU would swap them at step 5.
        let mut t = tracker();
        let mut id_a = None;
        let mut id_b = None;
        for k in 0..10u64 {
            let a = det(0.02 + 0.08 * k as f32, 0.4);
            let b = det(0.78 - 0.08 * k as f32, 0.4);
            let out = t.update(k * 33, &[a, b]);
            assert_eq!(out.len(), 2, "step {k}");
            let got_a = out.iter().find(|o| o.det == a).map(|o| o.id);
            let got_b = out.iter().find(|o| o.det == b).map(|o| o.id);
            if k == 0 {
                id_a = got_a;
                id_b = got_b;
                assert_ne!(id_a, id_b);
            } else {
                assert_eq!(got_a, id_a, "A at step {k}");
                assert_eq!(got_b, id_b, "B at step {k}");
            }
        }
    }

    #[test]
    fn new_box_gets_new_id_and_is_new() {
        let mut t = tracker();
        let first = t.update(0, &[det(0.1, 0.1)]);
        let out = t.update(33, &[det(0.1, 0.1), det(0.6, 0.6)]);
        assert_eq!(out.len(), 2);
        let old = out.iter().find(|o| o.det.bbox.x == 0.1).unwrap();
        let new = out.iter().find(|o| o.det.bbox.x == 0.6).unwrap();
        assert_eq!(old.id, first[0].id);
        assert!(!old.is_new);
        assert!(new.is_new);
        assert_eq!(new.age_ms, 0);
        assert!(new.id > old.id);
    }

    #[test]
    fn track_expires_after_ttl() {
        let mut t = tracker();
        let first = t.update(0, &[det(0.3, 0.3)]);
        assert!(
            t.update(500, &[]).is_empty(),
            "unmatched tracks are not returned"
        );
        let out = t.update(1001, &[det(0.3, 0.3)]);
        assert_eq!(out.len(), 1);
        assert_ne!(out[0].id, first[0].id);
        assert!(out[0].is_new);
    }

    #[test]
    fn track_survives_short_gap() {
        let mut t = tracker();
        let first = t.update(0, &[det(0.10, 0.3)]);
        t.update(33, &[det(0.12, 0.3)]);
        assert!(t.update(66, &[]).is_empty());
        let out = t.update(99, &[det(0.16, 0.3)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, first[0].id);
        assert!(!out[0].is_new);
        assert_eq!(out[0].age_ms, 99);
    }

    #[test]
    fn stationary_track_alive_at_exactly_ttl() {
        let mut t = tracker();
        let first = t.update(0, &[det(0.3, 0.3)]);
        let out = t.update(1000, &[det(0.3, 0.3)]);
        assert_eq!(out[0].id, first[0].id);
        assert!(!out[0].is_new);
    }

    #[test]
    fn ids_never_reused() {
        let mut t = tracker();
        let mut seen = std::collections::HashSet::new();
        let mut last = 0;
        for round in 0..5u64 {
            let ts = round * 2000; // every round outlives the ttl
            for o in t.update(ts, &[det(0.1, 0.1), det(0.5, 0.5)]) {
                assert!(o.is_new);
                assert!(seen.insert(o.id), "id {:?} reused", o.id);
                assert!(o.id.0 > last);
                last = o.id.0;
            }
        }
        assert_eq!(seen.len(), 10);
    }

    #[test]
    fn more_dets_than_tracks_and_vice_versa() {
        let mut t = tracker();
        t.update(0, &[det(0.1, 0.1)]);
        let out = t.update(33, &[det(0.1, 0.1), det(0.4, 0.4), det(0.7, 0.7)]);
        assert_eq!(ids(&out), vec![1, 2, 3]);
        let out = t.update(66, &[det(0.4, 0.4)]);
        assert_eq!(ids(&out), vec![2]);
        assert!(t.update(99, &[]).is_empty());
    }
}
