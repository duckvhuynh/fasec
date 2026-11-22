//! One-to-one association of IR faces with RGB tracks.
//!
//! The IR and RGB cameras sit side by side, so a face's normalised box differs between them by
//! a per-axis scale (field of view) and a shift (baseline parallax, which varies with distance).
//! The mapping is learned online from frames with exactly one confident face on each camera,
//! and no IR face is matched until the mapping is trusted.

use std::ops::RangeInclusive;

use fasec_core::{RectF, TrackId, TrackObs};

use crate::Detection;

/// Same match threshold as the RGB tracker.
pub(crate) const IOU_MATCH: f32 = 0.3;
/// A track or IR face whose two best IoUs are closer than this is ambiguous and left unmatched.
const AMBIGUITY_MARGIN: f32 = 0.1;
/// A 1↔1 pair agrees with the mapping when the mapped RGB box overlaps the IR face this well.
const IOU_CONSISTENT: f32 = 0.5;
/// Consecutive consistent bootstrap pairs before the mapping is trusted and used for matching.
const TRUST_STREAK: u8 = 5;
/// EMA weight of one single-face frame: about 10 frames to settle after a move.
const ALPHA: f32 = 0.2;
/// Bezel geometry as per-axis IR/RGB box-size ratio and face-centre displacement (normalised).
/// Placeholders until Task 7 measures the pair. The bootstrap prior is tight so that a second
/// face beside the RGB one (about a face width away) can never seed the mapping.
const BOOT_SCALE: RangeInclusive<f32> = 0.85..=1.18;
const BOOT_SHIFT: f32 = 0.08;
/// Outer bound for learning once trusted, which leaves room for parallax up close.
const SCALE_RANGE: RangeInclusive<f32> = 0.75..=1.33;
const MAX_SHIFT: f32 = 0.25;
/// Only clear faces on both cameras teach the mapping.
const MIN_LEARN_SCORE: f32 = 0.8;

pub struct IrAssociator {
    /// IR = `scale` · RGB + `shift`, per axis (x, y), in normalised coordinates.
    scale: [f32; 2],
    shift: [f32; 2],
    /// Consistent bootstrap pairs in a row, counting up to `TRUST_STREAK`.
    streak: u8,
}

impl Default for IrAssociator {
    fn default() -> Self {
        IrAssociator {
            scale: [1.0; 2],
            shift: [0.0; 2],
            streak: 0,
        }
    }
}

impl IrAssociator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the mapping has been learned (consecutive consistent pairs). Until then a track
    /// without an IR face says nothing about liveness.
    pub fn is_trusted(&self) -> bool {
        self.streak >= TRUST_STREAK
    }

    /// `rgb` box mapped into IR normalised coordinates.
    pub fn map(&self, rgb: RectF) -> RectF {
        let ([sx, sy], [tx, ty]) = (self.scale, self.shift);
        RectF {
            x: sx * rgb.x + tx,
            y: sy * rgb.y + ty,
            w: sx * rgb.w,
            h: sy * rgb.h,
        }
    }

    /// One entry per RGB track, in `rgb` order, holding the IR face matched to it (in IR
    /// coordinates): the pair with the highest IoU after mapping goes first, and each IR face
    /// goes to at most one track, so one person's IR evidence never reaches another's track.
    /// A track or IR face whose best two IoUs are within `AMBIGUITY_MARGIN` gets nothing, and
    /// until the mapping is trusted every track gets `None`. A frame with exactly one face on
    /// each camera first refines the mapping.
    pub fn associate(
        &mut self,
        rgb: &[TrackObs],
        ir: &[Detection],
    ) -> Vec<(TrackId, Option<Detection>)> {
        if let ([t], [d]) = (rgb, ir) {
            self.learn(t, d);
        }
        let mut out: Vec<(TrackId, Option<Detection>)> = rgb.iter().map(|t| (t.id, None)).collect();
        if !self.is_trusted() {
            return out;
        }
        let iou: Vec<Vec<f32>> = rgb
            .iter()
            .map(|t| {
                let m = self.map(t.bbox);
                ir.iter()
                    .map(|d| {
                        if usable(&t.bbox) && usable(&d.bbox) {
                            m.iou(&d.bbox)
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();
        let face_clear: Vec<bool> = (0..ir.len())
            .map(|di| clear(iou.iter().map(|row| row[di])))
            .collect();
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, row) in iou.iter().enumerate() {
            if !clear(row.iter().copied()) {
                continue;
            }
            for (di, &v) in row.iter().enumerate() {
                if v >= IOU_MATCH && face_clear[di] {
                    pairs.push((v, ti, di));
                }
            }
        }
        // Greedy, not Hungarian: with near-ties dropped above, each IR face goes to its clearly
        // closest track, where Hungarian could trade that for a higher IoU total.
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut taken = vec![false; ir.len()];
        for (_, ti, di) in pairs {
            if out[ti].1.is_none() && !taken[di] {
                out[ti].1 = Some(ir[di]);
                taken[di] = true;
            }
        }
        out
    }

    /// Moves the mapping toward the one implied by this face pair.
    ///
    /// Before trust, a pair must sit inside the tight bootstrap prior: the first of a streak
    /// seeds the mapping and each next one must agree with it (`IOU_CONSISTENT`), and any other
    /// 1↔1 pair breaks the streak. Once trusted, only pairs that already match the mapping that
    /// well (and stay inside the outer bound) teach it, so a different face cannot drag it.
    fn learn(&mut self, t: &TrackObs, d: &Detection) {
        let (r, i) = (t.bbox, d.bbox);
        // `>=` is false for a NaN score
        let confident = t.det_score >= MIN_LEARN_SCORE && d.score >= MIN_LEARN_SCORE;
        if !(confident && usable(&r) && usable(&i)) {
            if !self.is_trusted() {
                self.streak = 0;
            }
            return;
        }
        let (cr, ci) = (r.center(), i.center());
        let scale = [i.w / r.w, i.h / r.h];
        let shift = [ci.x - scale[0] * cr.x, ci.y - scale[1] * cr.y];
        let within = |range: &RangeInclusive<f32>, max: f32| {
            scale.iter().all(|s| range.contains(s))
                && [ci.x - cr.x, ci.y - cr.y].iter().all(|o| o.abs() <= max)
        };
        let consistent = self.map(r).iou(&i) >= IOU_CONSISTENT;
        if self.is_trusted() {
            if !(consistent && within(&SCALE_RANGE, MAX_SHIFT)) {
                return;
            }
        } else if !within(&BOOT_SCALE, BOOT_SHIFT) {
            self.streak = 0;
            return;
        } else if self.streak == 0 || !consistent {
            (self.scale, self.shift, self.streak) = (scale, shift, 1);
            return;
        } else {
            self.streak += 1;
        }
        for k in 0..2 {
            self.scale[k] += ALPHA * (scale[k] - self.scale[k]);
            self.shift[k] += ALPHA * (shift[k] - self.shift[k]);
        }
    }
}

/// Whether the best of `ious` beats the second best by at least `AMBIGUITY_MARGIN`.
fn clear(ious: impl Iterator<Item = f32>) -> bool {
    let (mut best, mut second) = (0.0f32, 0.0f32);
    for v in ious {
        if v > best {
            (best, second) = (v, best);
        } else if v > second {
            second = v;
        }
    }
    best - second >= AMBIGUITY_MARGIN
}

/// Finite with a positive size: `RectF::iou` drops NaN in its min/max, so a NaN box can
/// otherwise score a real overlap.
fn usable(r: &RectF) -> bool {
    [r.x, r.y, r.w, r.h].iter().all(|v| v.is_finite()) && r.w > 0.0 && r.h > 0.0
}

#[cfg(test)]
mod tests {
    use fasec_core::{IdentityState, LivenessState, PointF};

    use super::*;

    fn rect(x: f32, y: f32, w: f32, h: f32) -> RectF {
        RectF { x, y, w, h }
    }

    fn track(id: u64, bbox: RectF) -> TrackObs {
        TrackObs {
            id: TrackId(id),
            bbox,
            landmarks5: [bbox.center(); 5],
            det_score: 0.9,
            identity: IdentityState::Pending,
            liveness: LivenessState::Unknown,
            head_pose: None,
            face_width_px: 100.0,
            age_ms: 0,
        }
    }

    fn det(bbox: RectF) -> Detection {
        Detection {
            bbox,
            landmarks5: [PointF { x: 0.0, y: 0.0 }; 5],
            score: 0.9,
        }
    }

    /// The bezel offset used by the tests, inside the bootstrap prior: IR is slightly zoomed
    /// out and shifted.
    const SCALE: [f32; 2] = [0.9, 0.95];
    const SHIFT: [f32; 2] = [0.1, -0.01];

    fn true_map(r: RectF) -> RectF {
        rect(
            SCALE[0] * r.x + SHIFT[0],
            SCALE[1] * r.y + SHIFT[1],
            SCALE[0] * r.w,
            SCALE[1] * r.h,
        )
    }

    fn trusted(scale: [f32; 2], shift: [f32; 2]) -> IrAssociator {
        IrAssociator {
            scale,
            shift,
            streak: TRUST_STREAK,
        }
    }

    fn ids(out: &[(TrackId, Option<Detection>)]) -> Vec<(u64, Option<RectF>)> {
        out.iter().map(|(t, d)| (t.0, d.map(|d| d.bbox))).collect()
    }

    fn assert_close(a: RectF, b: RectF, eps: f32) {
        let ok = [(a.x, b.x), (a.y, b.y), (a.w, b.w), (a.h, b.h)]
            .iter()
            .all(|(p, q)| (p - q).abs() < eps);
        assert!(ok, "{a:?} != {b:?}");
    }

    #[test]
    fn starts_at_identity() {
        let r = rect(0.2, 0.3, 0.15, 0.25);
        assert_close(IrAssociator::new().map(r), r, 1e-6);
    }

    #[test]
    fn learns_known_offset_from_single_face_frames() {
        let mut a = IrAssociator::new();
        // one face wandering across the frame, alone on both cameras
        for i in 0..60 {
            let t = i as f32 / 60.0;
            let r = rect(
                0.2 + 0.4 * t,
                0.3 - 0.1 * t,
                0.15 + 0.05 * t,
                0.27 + 0.09 * t,
            );
            let out = a.associate(&[track(1, r)], &[det(true_map(r))]);
            // nothing attaches until five consistent pairs have built trust
            assert_eq!(out[0].1.is_some(), i >= 4, "frame {i}");
        }
        let probe = rect(0.4, 0.35, 0.15, 0.27);
        assert_close(a.map(probe), true_map(probe), 1e-3);

        // two faces (no learning): each RGB track gets its own IR face
        let other = rect(0.05, 0.3, 0.12, 0.22);
        let out = a.associate(
            &[track(1, probe), track(2, other)],
            &[det(true_map(other)), det(true_map(probe))],
        );
        assert_eq!(
            ids(&out),
            vec![(1, Some(true_map(probe))), (2, Some(true_map(other)))]
        );
    }

    #[test]
    fn trust_needs_five_consecutive_consistent_pairs() {
        let r = rect(0.3, 0.3, 0.2, 0.3);
        let good = det(true_map(r));
        let mut weak = good;
        weak.score = 0.5;
        let mut a = IrAssociator::new();
        // four good pairs, then one that breaks the streak, three times over
        for _ in 0..3 {
            for _ in 0..TRUST_STREAK - 1 {
                let out = a.associate(&[track(1, r)], &[good]);
                assert_eq!(ids(&out), vec![(1, None)]);
            }
            a.associate(&[track(1, r)], &[weak]);
        }
        for k in 0..TRUST_STREAK {
            let out = a.associate(&[track(1, r)], &[good]);
            assert_eq!(out[0].1.is_some(), k == TRUST_STREAK - 1, "pair {k}");
        }
    }

    #[test]
    fn face_beside_a_replay_never_attaches_or_teaches() {
        // RGB sees only a replayed face `r`; IR sees only a real face beside it (shift 0.2)
        let r = rect(0.3, 0.3, 0.2, 0.3);
        let beside = det(rect(r.x + 0.2, r.y, r.w, r.h));
        for mut a in [IrAssociator::new(), trusted(SCALE, SHIFT)] {
            let before = a.map(r);
            for _ in 0..50 {
                let out = a.associate(&[track(1, r)], &[beside]);
                assert_eq!(ids(&out), vec![(1, None)]);
            }
            assert_close(a.map(r), before, 1e-6);
            // the matching face still converges and attaches
            let mut out = Vec::new();
            for _ in 0..TRUST_STREAK {
                out = a.associate(&[track(1, r)], &[det(true_map(r))]);
            }
            assert_eq!(ids(&out), vec![(1, Some(true_map(r)))]);
            assert_close(a.map(r), true_map(r), 1e-3);
        }
    }

    #[test]
    fn ambiguous_ir_face_is_left_unmatched() {
        let ir = rect(0.4, 0.3, 0.2, 0.3);
        // a same-size box slid sideways (`sign`) until its IoU with `ir` is `q`
        let slid =
            |q: f32, sign: f32| rect(ir.x + sign * ir.w * (1.0 - q) / (1.0 + q), ir.y, ir.w, ir.h);
        let mut a = trusted([1.0; 2], [0.0; 2]);
        for (q1, q2, want) in [(0.55, 0.50, None), (0.8, 0.4, Some(ir))] {
            let (t1, t2) = (slid(q1, -1.0), slid(q2, 1.0));
            assert!((t1.iou(&ir) - q1).abs() < 1e-4 && (t2.iou(&ir) - q2).abs() < 1e-4);
            let out = a.associate(&[track(1, t1), track(2, t2)], &[det(ir)]);
            assert_eq!(ids(&out), vec![(1, want), (2, None)], "{q1} vs {q2}");
        }
    }

    #[test]
    fn crossing_faces_keep_their_own_ir_face() {
        // two overlapping faces passing each other; IR lists them in the opposite order
        let a = rect(0.30, 0.3, 0.2, 0.3);
        let b = rect(0.38, 0.32, 0.2, 0.3);
        let mut assoc = trusted(SCALE, SHIFT);
        let out = assoc.associate(
            &[track(1, a), track(2, b)],
            &[det(true_map(b)), det(true_map(a))],
        );
        assert_eq!(
            ids(&out),
            vec![(1, Some(true_map(a))), (2, Some(true_map(b)))]
        );
        // and after they swap sides
        let out = assoc.associate(
            &[track(1, b), track(2, a)],
            &[det(true_map(a)), det(true_map(b))],
        );
        assert_eq!(
            ids(&out),
            vec![(1, Some(true_map(b))), (2, Some(true_map(a)))]
        );
    }

    #[test]
    fn one_ir_face_goes_to_one_track_only() {
        // both RGB tracks overlap the single IR face above the threshold
        let near = rect(0.30, 0.3, 0.2, 0.3);
        let close_by = rect(0.36, 0.3, 0.2, 0.3);
        let ir = det(near);
        assert!(close_by.iou(&ir.bbox) >= 0.3);
        for order in [[1, 2], [2, 1]] {
            let tracks: Vec<TrackObs> = order
                .iter()
                .map(|&id| track(id, if id == 1 { near } else { close_by }))
                .collect();
            let out = trusted([1.0; 2], [0.0; 2]).associate(&tracks, &[ir]);
            let mut got = ids(&out);
            got.sort_by_key(|g| g.0);
            assert_eq!(got, vec![(1, Some(near)), (2, None)], "order {order:?}");
        }
    }

    #[test]
    fn empty_inputs() {
        let mut a = IrAssociator::new();
        assert!(
            a.associate(&[], &[det(rect(0.1, 0.1, 0.2, 0.2))])
                .is_empty()
        );
        let out = a.associate(&[track(3, rect(0.1, 0.1, 0.2, 0.2))], &[]);
        assert_eq!(ids(&out), vec![(3, None)]);
        assert!(a.associate(&[], &[]).is_empty());
        let r = rect(0.2, 0.3, 0.15, 0.25);
        assert_close(a.map(r), r, 1e-6);
    }

    #[test]
    fn nan_boxes_are_ignored() {
        let good = rect(0.3, 0.3, 0.2, 0.3);
        let mut a = trusted([1.0; 2], [0.0; 2]);
        // a NaN corner still overlaps under `RectF::iou`'s NaN-dropping min/max
        for bad in [
            rect(f32::NAN, 0.3, 0.2, 0.3),
            rect(0.3, 0.3, f32::NAN, 0.3),
            rect(0.3, f32::INFINITY, 0.2, 0.3),
        ] {
            let out = a.associate(&[track(1, bad)], &[det(good)]);
            assert_eq!(ids(&out), vec![(1, None)], "{bad:?}");
            let out = a.associate(&[track(1, good)], &[det(bad)]);
            assert_eq!(ids(&out), vec![(1, None)], "{bad:?}");
            let out = a.associate(&[track(1, good), track(2, bad)], &[det(good)]);
            assert_eq!(ids(&out), vec![(1, Some(good)), (2, None)], "{bad:?}");
        }
        // the mapping learned nothing from them
        let r = rect(0.2, 0.3, 0.15, 0.25);
        assert_close(a.map(r), r, 1e-6);
    }

    #[test]
    fn learning_skips_outliers_and_weak_faces() {
        let r = rect(0.3, 0.3, 0.2, 0.3);
        for mut a in [IrAssociator::new(), trusted(SCALE, SHIFT)] {
            let before = a.map(r);
            // a lone IR face on the other side of the frame (another source, not this face)
            for _ in 0..20 {
                a.associate(&[track(1, r)], &[det(rect(0.75, 0.3, 0.2, 0.3))]);
            }
            // a lone IR face of implausible size
            for _ in 0..20 {
                a.associate(&[track(1, r)], &[det(rect(0.3, 0.3, 0.05, 0.08))]);
            }
            // a plausible pair, but one side is a weak or NaN-scored detection
            for score in [0.65, f32::NAN] {
                for _ in 0..20 {
                    let mut weak = det(true_map(r));
                    weak.score = score;
                    let out = a.associate(&[track(1, r)], &[weak]);
                    assert_eq!(out[0].1.is_some(), a.is_trusted());
                    let mut t = track(1, r);
                    t.det_score = score;
                    a.associate(&[t], &[det(true_map(r))]);
                }
            }
            assert_close(a.map(r), before, 1e-6);
        }
    }
}
