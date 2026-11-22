//! One-to-one association of IR faces with RGB tracks.
//!
//! The IR and RGB cameras sit side by side, so a face's normalised box differs between them by
//! a per-axis scale (field of view) and a shift (baseline parallax, which varies with distance).
//! The mapping is learned online from frames with exactly one confident face on each camera.

use fasec_core::{RectF, TrackId, TrackObs};

use crate::Detection;

/// Same match threshold as the RGB tracker.
const IOU_MATCH: f32 = 0.3;
/// EMA weight of one single-face frame: about 10 frames to settle after a move.
const ALPHA: f32 = 0.2;
/// Plausible bezel geometry, placeholders until the recorder (Task 6) measures the pair:
/// per-axis IR/RGB box-size ratio and normalised shift.
const SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.75..=1.33;
const MAX_SHIFT: f32 = 0.25;
/// Only clear faces on both cameras teach the mapping.
const MIN_LEARN_SCORE: f32 = 0.8;

pub struct IrAssociator {
    /// IR = `scale` · RGB + `shift`, per axis (x, y), in normalised coordinates.
    scale: [f32; 2],
    shift: [f32; 2],
}

impl Default for IrAssociator {
    fn default() -> Self {
        IrAssociator {
            scale: [1.0; 2],
            shift: [0.0; 2],
        }
    }
}

impl IrAssociator {
    pub fn new() -> Self {
        Self::default()
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
    /// A frame with exactly one face on each camera first refines the mapping.
    pub fn associate(
        &mut self,
        rgb: &[TrackObs],
        ir: &[Detection],
    ) -> Vec<(TrackId, Option<Detection>)> {
        if let ([t], [d]) = (rgb, ir) {
            self.learn(t, d);
        }
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (ti, t) in rgb.iter().enumerate().filter(|(_, t)| usable(&t.bbox)) {
            let m = self.map(t.bbox);
            for (di, d) in ir.iter().enumerate().filter(|(_, d)| usable(&d.bbox)) {
                let iou = m.iou(&d.bbox);
                if iou >= IOU_MATCH {
                    pairs.push((iou, ti, di));
                }
            }
        }
        // stable sort: ties go to the earlier track
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut out: Vec<(TrackId, Option<Detection>)> = rgb.iter().map(|t| (t.id, None)).collect();
        let mut taken = vec![false; ir.len()];
        for (_, ti, di) in pairs {
            if out[ti].1.is_none() && !taken[di] {
                out[ti].1 = Some(ir[di]);
                taken[di] = true;
            }
        }
        out
    }

    /// Moves the mapping toward the one implied by this face pair. A pair implying a scale or
    /// shift outside the plausible bezel geometry is another IR source, not this face, and is
    /// skipped; the average of in-range samples stays in range, which is the clamp.
    fn learn(&mut self, t: &TrackObs, d: &Detection) {
        let (r, i) = (t.bbox, d.bbox);
        if t.det_score < MIN_LEARN_SCORE || d.score < MIN_LEARN_SCORE || !usable(&r) || !usable(&i)
        {
            return;
        }
        let (cr, ci) = (r.center(), i.center());
        let scale = [i.w / r.w, i.h / r.h];
        let shift = [ci.x - scale[0] * cr.x, ci.y - scale[1] * cr.y];
        if scale.iter().any(|s| !SCALE_RANGE.contains(s))
            || shift.iter().any(|s| s.abs() > MAX_SHIFT)
        {
            return;
        }
        for k in 0..2 {
            self.scale[k] += ALPHA * (scale[k] - self.scale[k]);
            self.shift[k] += ALPHA * (shift[k] - self.shift[k]);
        }
    }
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

    /// The bezel offset used by the tests: IR is slightly zoomed out and shifted.
    const SCALE: [f32; 2] = [0.9, 0.95];
    const SHIFT: [f32; 2] = [0.12, -0.04];

    fn true_map(r: RectF) -> RectF {
        rect(
            SCALE[0] * r.x + SHIFT[0],
            SCALE[1] * r.y + SHIFT[1],
            SCALE[0] * r.w,
            SCALE[1] * r.h,
        )
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
        // a face that, unmapped, overlaps its IR box too little to match
        let probe = rect(0.4, 0.35, 0.15, 0.27);
        assert!(probe.iou(&true_map(probe)) < 0.3);

        // one face wandering across the frame, alone on both cameras
        for i in 0..60 {
            let t = i as f32 / 60.0;
            let r = rect(
                0.2 + 0.4 * t,
                0.3 - 0.1 * t,
                0.15 + 0.05 * t,
                0.27 + 0.09 * t,
            );
            a.associate(&[track(1, r)], &[det(true_map(r))]);
        }
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
    fn crossing_faces_keep_their_own_ir_face() {
        // two overlapping faces passing each other; IR lists them in the opposite order
        let a = rect(0.30, 0.3, 0.2, 0.3);
        let b = rect(0.38, 0.32, 0.2, 0.3);
        let mut assoc = IrAssociator {
            scale: SCALE,
            shift: SHIFT,
        };
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
            let out = IrAssociator::new().associate(&tracks, &[ir]);
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
        let mut a = IrAssociator::new();
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
        // the mapping learned nothing from them (the good 1↔1 pairs were identity anyway)
        let r = rect(0.2, 0.3, 0.15, 0.25);
        assert_close(a.map(r), r, 1e-6);
    }

    #[test]
    fn learning_skips_outliers_and_weak_faces() {
        let mut a = IrAssociator::new();
        let r = rect(0.3, 0.3, 0.2, 0.3);
        // a lone IR face on the other side of the frame (another source, not this face)
        for _ in 0..20 {
            a.associate(&[track(1, r)], &[det(rect(0.75, 0.3, 0.2, 0.3))]);
        }
        // a lone IR face of implausible size
        for _ in 0..20 {
            a.associate(&[track(1, r)], &[det(rect(0.3, 0.3, 0.05, 0.08))]);
        }
        // a plausible pair, but one side is a weak detection
        for _ in 0..20 {
            let mut weak = det(true_map(r));
            weak.score = 0.65;
            a.associate(&[track(1, r)], &[weak]);
            let mut t = track(1, r);
            t.det_score = 0.65;
            a.associate(&[t], &[det(true_map(r))]);
        }
        assert_close(a.map(r), r, 1e-6);
    }
}
