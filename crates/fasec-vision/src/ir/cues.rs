//! IR liveness cues and the per-frame IR verdict.
//!
//! Skin close to the Windows Hello emitter reflects its 850 nm light strongly, with 3-D shading
//! (on the recorded ASUS camera the nose reads 0.81–0.91 of the cheeks, not brighter) and corneal
//! glints, and it is dark in the unlit frame. A phone or monitor reflects little of it (no IR face), paper reflects it evenly,
//! and an NIR-emitting screen is as bright unlit as lit. Every cue is a ratio, so absolute
//! brightness (sunlight) never makes a face live.

use fasec_core::{IrFrame, PointF, RectF};

use crate::Detection;
use crate::ir::IrAssociator;
use crate::ir::assoc::IOU_MATCH;

/// The ring around the face box extends this fraction of the box size on each side.
const RING_MARGIN: f32 = 0.25;
/// Below this visible fraction of the face box (the rest outside the IR frame), no cues.
const MIN_VISIBLE: f32 = 0.5;
/// Landmarks may lie this fraction of the box size outside the IR face box (a turned head puts
/// an eye at the edge); further out the detection is inconsistent and gives no cues.
const LANDMARK_SLACK: f32 = 0.1;
/// Half side of the nose and cheek squares, as a fraction of the face box width.
const REGION_HALF: f32 = 0.07;
/// A glint is a pixel brighter than this in the lit frame (and not in the ambient one).
const GLINT_LEVEL: u8 = 240;
/// Glints are searched within this radius of an eye landmark (fraction of box width, ≥ 2 px)...
const GLINT_RADIUS: f32 = 0.08;
/// ...and must be small: at most this fraction of that disc, so a saturated patch is not one.
const GLINT_MAX_FRACTION: f32 = 0.1;

/// Per-frame IR measurements for one RGB track. Built by [`ir_cues`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrCues {
    /// The track has an associated IR face. `false` only from a trusted mapping with no IR face
    /// anywhere near the mapped RGB box; the other cues are then over that mapped box.
    pub face_present: bool,
    /// Mean of lit − ambient over the face box. NaN when the ambient frame is missing or blank
    /// (some drivers blank the unlit frames), so it is not evidence.
    pub lit_minus_ambient: f32,
    /// Face mean ÷ background-ring mean, on lit − ambient when the ambient frame is usable and on
    /// the lit frame otherwise. The ring mean is floored at one grey level.
    pub face_vs_background: f32,
    /// Nose-region mean ÷ mean of the two cheek regions (same image and floor as above). NaN when
    /// there is no IR face or a region falls outside the frame.
    pub relief: f32,
    /// Eyes (0–2) with a small saturated spot in the lit frame that the ambient frame lacks.
    pub glints: u8,
    /// Lit-frame means over the face box and over the ring, for the signal floor, self-emission
    /// and flooding checks only.
    pub lit_face: f32,
    pub lit_background: f32,
    /// Ambient-frame mean over the ring; NaN when the ambient frame is missing or blank. The
    /// ambient face mean is `lit_face - lit_minus_ambient`.
    pub ambient_background: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IrSpoofReason {
    /// An RGB face with no IR face where the mapping puts it (phone or monitor replay).
    NoIrFace,
    /// An IR face flatter than any recorded real face (meant for a printed photo; not verified
    /// against one, as none was recorded).
    FlatReflectance,
    /// The face is about as bright unlit as lit (NIR-emitting screen).
    SelfEmitting,
}

/// `score` follows the RGB tier: live-likeness in [0, 1].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IrVerdict {
    Live { score: f32 },
    Spoof { score: f32, reason: IrSpoofReason },
    Unknown,
}

/// Thresholds of [`judge`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrLivenessConfig {
    /// Consecutive lit frames with an IR `Spoof` verdict (any reason) before a track is called
    /// a spoof. `judge` is per frame; the pipeline counts this per track.
    pub spoof_frames: u8,
    /// Live needs `face_vs_background` at least this; below it the emitter contrast is too weak
    /// to judge (`Unknown`).
    pub min_face_vs_background: f32,
    /// Live needs `relief` in `min_relief..=max_relief`. Below it the nose is darker than on any
    /// recorded real face (`Unknown`); above it the face is flatter than any (`FlatReflectance`).
    pub min_relief: f32,
    pub max_relief: f32,
    /// A lit background ring brighter than this means the scene is flooded with ambient IR and
    /// the emitter contrast is lost (`Unknown`).
    pub max_background: f32,
    /// With a usable ambient frame, a face whose lit − ambient is below this fraction of its lit
    /// brightness gets too little of its light from the emitter: `SelfEmitting` if it also glows
    /// in the ambient frame (see `self_emit_contrast`), else `Unknown` (sun or window NIR).
    pub min_emitter_fraction: f32,
    /// A face is self-lit when its ambient mean is at least this multiple of the ambient ring
    /// mean; a face lit by the room is about as bright as its surroundings.
    pub self_emit_contrast: f32,
    /// Below this lit face mean (grey levels) the frame is noise, not a measurement (`Unknown`).
    pub min_lit_face: f32,
}

impl Default for IrLivenessConfig {
    fn default() -> Self {
        // Measured on the owner's recordings (Task 7, ASUS IR camera 640×360, trusted frames of
        // a real face at 30, 50 and 80 cm and turning; docs/spikes/SP-7-ir.md). No printed photo
        // or monitor was recorded, and the unlit frame was blank on every frame.
        IrLivenessConfig {
            // 3 consecutive lit frames ≈ 1.4 s at the recorded ~2.2 lit frames/s
            spoof_frames: 3,
            // real min 1.72 (80 cm; median 2.35 at 30 cm): margin 0.22
            min_face_vs_background: 1.5,
            // real 0.81–0.91: on this camera the nose reads darker than the cheeks
            min_relief: 0.70,
            // real max 0.91; a flat print should read ≈ 1.0 (unverified: no printer)
            max_relief: 0.97,
            // real lit_background max 76 (80 cm): margin 24
            max_background: 100.0,
            // not measurable: no usable ambient frame was recorded; placeholder
            min_emitter_fraction: 0.5,
            // not measurable: no usable ambient frame was recorded; placeholder
            self_emit_contrast: 2.0,
            // real lit_face min 109 (30 cm): far above the noise floor
            min_lit_face: 20.0,
        }
    }
}

/// Live score without glints, and the bonus per glinting eye. Placeholders until Task 7.
const LIVE_SCORE: f32 = 0.8;
const GLINT_BONUS: f32 = 0.1;

/// The IR verdict for one frame. Glints only raise a `Live` score; `Live` rests on the ratios.
pub fn judge(cues: &IrCues, cfg: &IrLivenessConfig) -> IrVerdict {
    let spoof = |reason| IrVerdict::Spoof { score: 0.0, reason };
    // before NoIrFace: in a flooded scene the IR detector may miss a real face
    if cues.lit_background > cfg.max_background {
        return IrVerdict::Unknown;
    }
    if !cues.face_present {
        return spoof(IrSpoofReason::NoIrFace);
    }
    let finite = [
        cues.face_vs_background,
        cues.relief,
        cues.lit_face,
        cues.lit_background,
    ]
    .iter()
    .all(|v| v.is_finite());
    if !finite || cues.lit_face < cfg.min_lit_face {
        return IrVerdict::Unknown;
    }
    // false for a NaN (unavailable) lit − ambient
    if cues.lit_minus_ambient < cfg.min_emitter_fraction * cues.lit_face {
        let ambient_face = cues.lit_face - cues.lit_minus_ambient;
        // false for a NaN ambient ring
        return if ambient_face >= cfg.self_emit_contrast * cues.ambient_background {
            spoof(IrSpoofReason::SelfEmitting)
        } else {
            IrVerdict::Unknown
        };
    }
    if cues.face_vs_background < cfg.min_face_vs_background {
        return IrVerdict::Unknown;
    }
    if cues.relief > cfg.max_relief {
        return spoof(IrSpoofReason::FlatReflectance);
    }
    if cues.relief < cfg.min_relief {
        return IrVerdict::Unknown;
    }
    IrVerdict::Live {
        score: LIVE_SCORE + GLINT_BONUS * f32::from(cues.glints.min(2)),
    }
}

/// Cues for the RGB track with box `rgb_box`, from the lit IR frame, the paired ambient frame and
/// `matched`, the IR face `IrAssociator::associate` gave the track (one of `ir_faces`, all IR
/// faces in `lit`).
///
/// `None` (no evidence, `Unknown`) when the frames are inconsistent, the face box or its ring is
/// degenerate or mostly outside the frame, a landmark of `matched` is non-finite or outside its
/// box (beyond `LANDMARK_SLACK`), or the track has no IR face and that cannot be told
/// apart from a missed match: the mapping is untrusted, or an IR face overlaps the mapped box
/// enough to match but went elsewhere (ambiguous or contested).
pub fn ir_cues(
    lit: &IrFrame,
    ambient: Option<&IrFrame>,
    assoc: &IrAssociator,
    rgb_box: RectF,
    ir_faces: &[Detection],
    matched: Option<&Detection>,
) -> Option<IrCues> {
    let (w, h) = (lit.width, lit.height);
    let n = w as usize * h as usize;
    if !lit.illuminated || lit.data.len() != n {
        return None;
    }
    if let Some(a) = ambient
        && (a.illuminated || (a.width, a.height) != (w, h) || a.data.len() != n)
    {
        return None;
    }
    let (bbox, landmarks) = match matched {
        Some(d) => {
            let b = d.bbox;
            let (tx, ty) = (LANDMARK_SLACK * b.w, LANDMARK_SLACK * b.h);
            let on_face = |p: &PointF| {
                (b.x - tx..=b.x + b.w + tx).contains(&p.x)
                    && (b.y - ty..=b.y + b.h + ty).contains(&p.y)
            };
            if !d.landmarks5.iter().all(on_face) {
                return None;
            }
            (b, Some(d.landmarks5))
        }
        None => {
            let mapped = assoc.map(rgb_box);
            if !assoc.is_trusted() || ir_faces.iter().any(|d| mapped.iou(&d.bbox) >= IOU_MATCH) {
                return None;
            }
            (mapped, None)
        }
    };
    let face = to_px(bbox, w, h)?;
    let full_area = bbox.w * w as f32 * bbox.h * h as f32;
    if face.area() == 0 || (face.area() as f32) < MIN_VISIBLE * full_area {
        return None;
    }
    let ring = to_px(
        RectF {
            x: bbox.x - RING_MARGIN * bbox.w,
            y: bbox.y - RING_MARGIN * bbox.h,
            w: bbox.w * (1.0 + 2.0 * RING_MARGIN),
            h: bbox.h * (1.0 + 2.0 * RING_MARGIN),
        },
        w,
        h,
    )?;
    let ring_area = ring.area() - face.area();
    if ring_area == 0 {
        return None;
    }

    let width = w as usize;
    let amb = ambient
        .filter(|a| !ambient_is_blank(a))
        .map(|a| &a.data[..]);
    let lit_px = |i: usize| f64::from(lit.data[i]);
    let signal = |i: usize| lit_px(i) - amb.map_or(0.0, |a| f64::from(a[i]));
    let ring_mean =
        |f: &dyn Fn(usize) -> f64| (sum(ring, width, f) - sum(face, width, f)) / ring_area as f64;
    let face_mean = mean(face, width, &signal);
    // `max` would turn a NaN (unmeasurable) region into 1
    let floored = |v: f64| if v < 1.0 { 1.0 } else { v };

    let face_w_px = bbox.w * w as f32;
    let region = |p: PointF| {
        let (hx, hy) = (REGION_HALF * bbox.w, REGION_HALF * face_w_px / h as f32);
        to_px(
            RectF {
                x: p.x - hx,
                y: p.y - hy,
                w: 2.0 * hx,
                h: 2.0 * hy,
            },
            w,
            h,
        )
        .map_or(f64::NAN, |r| mean(r, width, &signal))
    };
    let (relief, glints) = match landmarks {
        Some([re, le, nose, rm, lm]) => {
            let mid = |a: PointF, b: PointF| PointF {
                x: (a.x + b.x) / 2.0,
                y: (a.y + b.y) / 2.0,
            };
            let cheeks = (region(mid(re, rm)) + region(mid(le, lm))) / 2.0;
            let radius = (GLINT_RADIUS * face_w_px).max(2.0);
            let glint = |p: PointF| u8::from(has_glint(lit, amb, p, radius));
            (region(nose) / floored(cheeks), glint(re) + glint(le))
        }
        None => (f64::NAN, 0),
    };

    Some(IrCues {
        face_present: matched.is_some(),
        lit_minus_ambient: if amb.is_some() {
            face_mean as f32
        } else {
            f32::NAN
        },
        face_vs_background: (face_mean / floored(ring_mean(&signal))) as f32,
        relief: relief as f32,
        glints,
        lit_face: mean(face, width, &lit_px) as f32,
        lit_background: ring_mean(&lit_px) as f32,
        ambient_background: amb.map_or(f32::NAN, |a| ring_mean(&|i: usize| f64::from(a[i])) as f32),
    })
}

/// A near-black unlit frame (mean < 1 and 99th percentile ≤ 8): the driver blanked it, so it
/// carries no ambient level. Not `IrFrame::is_blank`, which needs every pixel at 0.
pub fn ambient_is_blank(f: &IrFrame) -> bool {
    let n = f.data.len();
    let total: u64 = f.data.iter().map(|&p| u64::from(p)).sum();
    let above = f.data.iter().filter(|&&p| p > 8).count();
    (total as f64) < n as f64 && above * 100 <= n
}

/// Whether the disc of `radius` px around `p` holds a small spot brighter than `GLINT_LEVEL` in
/// `lit` but not in `amb`.
fn has_glint(lit: &IrFrame, amb: Option<&[u8]>, p: PointF, radius: f32) -> bool {
    let (w, h) = (lit.width as f32, lit.height as f32);
    let (cx, cy) = (p.x * w, p.y * h);
    let square = RectF {
        x: p.x - radius / w,
        y: p.y - radius / h,
        w: 2.0 * radius / w,
        h: 2.0 * radius / h,
    };
    let Some(r) = to_px(square, lit.width, lit.height) else {
        return false;
    };
    let (mut disc, mut spot) = (0usize, 0usize);
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            disc += 1;
            let i = y * lit.width as usize + x;
            if lit.data[i] > GLINT_LEVEL && amb.is_none_or(|a| a[i] <= GLINT_LEVEL) {
                spot += 1;
            }
        }
    }
    spot > 0 && spot as f32 <= GLINT_MAX_FRACTION * disc as f32
}

/// Pixel rectangle [x0, x1) × [y0, y1).
#[derive(Clone, Copy)]
struct Px {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Px {
    fn area(&self) -> usize {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

/// `r` (normalised) in pixels of a `w`×`h` frame, rounded to whole pixels and clipped to the
/// frame (possibly empty). `None` for a non-finite or non-positive box.
fn to_px(r: RectF, w: u32, h: u32) -> Option<Px> {
    if ![r.x, r.y, r.w, r.h].iter().all(|v| v.is_finite()) || r.w <= 0.0 || r.h <= 0.0 {
        return None;
    }
    let (fw, fh) = (w as f32, h as f32);
    let clip = |v: f32, max: f32| v.clamp(0.0, max) as usize;
    let (x0, y0) = (clip((r.x * fw).round(), fw), clip((r.y * fh).round(), fh));
    let x1 = clip(((r.x + r.w) * fw).round(), fw).max(x0);
    let y1 = clip(((r.y + r.h) * fh).round(), fh).max(y0);
    Some(Px { x0, y0, x1, y1 })
}

fn sum(r: Px, width: usize, f: &dyn Fn(usize) -> f64) -> f64 {
    (r.y0..r.y1)
        .flat_map(|y| (r.x0..r.x1).map(move |x| y * width + x))
        .map(f)
        .sum()
}

/// NaN for an empty region.
fn mean(r: Px, width: usize, f: &dyn Fn(usize) -> f64) -> f64 {
    sum(r, width, f) / r.area() as f64
}

#[cfg(test)]
mod tests {
    use fasec_core::{IdentityState, LivenessState, TrackId, TrackObs};

    use super::*;

    const W: u32 = 100;
    /// IR face box of the fixtures: pixels 30..70 on both axes.
    const FACE: RectF = RectF {
        x: 0.3,
        y: 0.3,
        w: 0.4,
        h: 0.4,
    };

    fn rect(x: f32, y: f32, w: f32, h: f32) -> RectF {
        RectF { x, y, w, h }
    }

    fn frame(lit: bool, f: impl Fn(u32, u32) -> u8) -> IrFrame {
        let data = (0..W * W).map(|i| f(i % W, i / W)).collect();
        IrFrame::new(0, 0, W, W, lit, data).unwrap()
    }

    fn det(bbox: RectF) -> Detection {
        let p = |x, y| PointF { x, y };
        Detection {
            bbox,
            // eyes at px (42, 45) and (58, 45), nose (50, 55), mouth corners (44, 65), (56, 65)
            landmarks5: [
                p(0.42, 0.45),
                p(0.58, 0.45),
                p(0.5, 0.55),
                p(0.44, 0.65),
                p(0.56, 0.65),
            ],
            score: 0.9,
        }
    }

    fn in_face(x: u32, y: u32) -> bool {
        (30..70).contains(&x) && (30..70).contains(&y)
    }

    fn is_eye(x: u32, y: u32) -> bool {
        (x, y) == (42, 45) || (x, y) == (58, 45)
    }

    /// A face lit by the near-field emitter: brighter than the background, a brighter nose and
    /// a corneal glint in each eye.
    fn real_face(bg: u8, face: u8, nose: u8, glints: bool) -> impl Fn(u32, u32) -> u8 {
        move |x, y| {
            if glints && is_eye(x, y) {
                255
            } else if (46..54).contains(&x) && (51..59).contains(&y) {
                nose
            } else if in_face(x, y) {
                face
            } else {
                bg
            }
        }
    }

    /// An unlit frame blanked by the driver: near-black, not all-zero.
    fn blank_ambient() -> IrFrame {
        frame(false, |x, y| if (x + y).is_multiple_of(50) { 3 } else { 0 })
    }

    fn trusted_identity() -> IrAssociator {
        let t = TrackObs {
            id: TrackId(1),
            bbox: FACE,
            landmarks5: [FACE.center(); 5],
            det_score: 0.9,
            identity: IdentityState::Pending,
            liveness: LivenessState::Unknown,
            head_pose: None,
            face_width_px: 100.0,
            age_ms: 0,
        };
        let mut a = IrAssociator::new();
        for _ in 0..5 {
            a.associate(std::slice::from_ref(&t), &[det(FACE)]);
        }
        assert!(a.is_trusted());
        a
    }

    /// Cues and verdict for a track at `FACE` whose IR face is `FACE`.
    fn matched(lit: &IrFrame, ambient: Option<&IrFrame>) -> (IrCues, IrVerdict) {
        let d = det(FACE);
        let c = ir_cues(lit, ambient, &IrAssociator::new(), FACE, &[d], Some(&d)).unwrap();
        (c, judge(&c, &IrLivenessConfig::default()))
    }

    fn reason(v: IrVerdict) -> Option<IrSpoofReason> {
        match v {
            IrVerdict::Spoof { reason, .. } => Some(reason),
            _ => None,
        }
    }

    #[test]
    fn real_face_near_emitter_is_live() {
        let lit = frame(true, real_face(20, 150, 130, true));
        for ambient in [None, Some(blank_ambient())] {
            let (c, v) = matched(&lit, ambient.as_ref());
            assert!(c.face_present && c.lit_minus_ambient.is_nan(), "{c:?}");
            assert!(c.face_vs_background > 5.0 && (c.relief - 130.0 / 150.0).abs() < 1e-3);
            assert_eq!(c.glints, 2);
            assert_eq!(v, IrVerdict::Live { score: 1.0 }, "{c:?}");
        }
        // a dim but real ambient frame: lit − ambient is evidence and the ratios use it
        let amb = frame(false, |_, _| 10);
        let (c, v) = matched(&lit, Some(&amb));
        let face_mean = (1600.0 * 150.0 - 64.0 * 20.0 + 2.0 * 105.0) / 1600.0 - 10.0;
        assert!((c.lit_minus_ambient - face_mean).abs() < 1e-3, "{c:?}");
        assert!((c.relief - 120.0 / 140.0).abs() < 1e-3, "{c:?}");
        assert!(matches!(v, IrVerdict::Live { .. }), "{c:?}");
    }

    /// The owner's recorded real faces (Task 7, trusted frames at 30, 50 and 80 cm and turning):
    /// face_vs_background 1.72–2.61, relief 0.81–0.91, lit_face 109–137, lit_background 46–76.
    #[test]
    fn recorded_real_face_extremes_are_live() {
        let cfg = IrLivenessConfig::default();
        let (base, _) = matched(&frame(true, real_face(20, 150, 130, false)), None);
        for (fvb, relief, lit_face, lit_bg) in [
            (1.72, 0.81, 128.0, 76.0),
            (2.61, 0.91, 109.0, 46.0),
            (2.07, 0.83, 137.0, 62.0),
        ] {
            let c = IrCues {
                face_vs_background: fvb,
                relief,
                lit_face,
                lit_background: lit_bg,
                ..base
            };
            assert_eq!(judge(&c, &cfg), IrVerdict::Live { score: 0.8 }, "{c:?}");
        }
    }

    #[test]
    fn relief_outside_the_real_face_window() {
        let cfg = IrLivenessConfig::default();
        let (live, _) = matched(&frame(true, real_face(20, 150, 130, false)), None);
        let at = |relief: f32| judge(&IrCues { relief, ..live }, &cfg);
        // the window's edges are live
        for r in [0.70, 0.97] {
            assert_eq!(at(r), IrVerdict::Live { score: 0.8 }, "{r}");
        }
        // flatter than any recorded real face: a print or a screen (unverified against prints)
        for r in [0.971, 1.0, 1.4] {
            assert_eq!(reason(at(r)), Some(IrSpoofReason::FlatReflectance), "{r}");
        }
        // a darker nose than any recorded real face: no verdict
        for r in [0.0, 0.69] {
            assert_eq!(at(r), IrVerdict::Unknown, "{r}");
        }
    }

    #[test]
    fn glints_are_a_bonus_only() {
        let (c, v) = matched(&frame(true, real_face(20, 150, 130, false)), None);
        assert_eq!(c.glints, 0);
        assert_eq!(v, IrVerdict::Live { score: 0.8 });
        // glints do not rescue a flat patch
        let (c, v) = matched(&frame(true, real_face(20, 150, 150, true)), None);
        assert_eq!(c.glints, 2);
        assert_eq!(reason(v), Some(IrSpoofReason::FlatReflectance));
    }

    #[test]
    fn glints_must_be_small_and_absent_in_ambient() {
        // a saturated patch (paper glare) is not a glint
        let (c, _) = matched(&frame(true, real_face(20, 250, 250, false)), None);
        assert_eq!(c.glints, 0);
        // a spot just as bright unlit is not the emitter's reflection
        let lit = frame(true, real_face(20, 150, 130, true));
        let amb = frame(false, |x, y| if is_eye(x, y) { 250 } else { 10 });
        assert_eq!(matched(&lit, Some(&amb)).0.glints, 0);
        let amb = frame(false, |_, _| 10);
        assert_eq!(matched(&lit, Some(&amb)).0.glints, 2);
    }

    #[test]
    fn uniform_patch_is_flat_reflectance() {
        // a printed photo near the emitter: bright, but evenly so
        let (c, v) = matched(&frame(true, real_face(20, 150, 150, false)), None);
        assert!(
            (c.relief - 1.0).abs() < 1e-6 && c.face_vs_background > 5.0,
            "{c:?}"
        );
        assert_eq!(reason(v), Some(IrSpoofReason::FlatReflectance));
    }

    #[test]
    fn empty_region_is_no_ir_face_only_when_trusted() {
        let lit = frame(true, |_, _| 20);
        let a = trusted_identity();
        let beside = det(rect(0.75, 0.3, 0.2, 0.3));
        for faces in [&[][..], &[beside][..]] {
            let c = ir_cues(&lit, None, &a, FACE, faces, None).unwrap();
            assert!(!c.face_present && c.relief.is_nan(), "{c:?}");
            let v = judge(&c, &IrLivenessConfig::default());
            assert_eq!(reason(v), Some(IrSpoofReason::NoIrFace));
        }
        // untrusted mapping: no IR face says nothing
        assert_eq!(
            ir_cues(&lit, None, &IrAssociator::new(), FACE, &[], None),
            None
        );
        // an IR face that overlaps enough to match but went to another track (or was a near-tie)
        let near = det(rect(0.32, 0.3, 0.4, 0.4));
        assert_eq!(ir_cues(&lit, None, &a, FACE, &[near], None), None);
    }

    #[test]
    fn self_emitting_screen_is_spoof() {
        // as bright unlit as lit: the screen emits NIR itself
        let lit = frame(true, real_face(20, 150, 130, true));
        let amb = frame(false, |x, y| {
            (f32::from(real_face(20, 150, 130, true)(x, y)) * 0.9) as u8
        });
        let (c, v) = matched(&lit, Some(&amb));
        let cfg = IrLivenessConfig::default();
        // the ratios on lit − ambient alone would look live
        assert!(
            c.face_vs_background >= cfg.min_face_vs_background
                && (cfg.min_relief..=cfg.max_relief).contains(&c.relief),
            "{c:?}"
        );
        assert_eq!(reason(v), Some(IrSpoofReason::SelfEmitting), "{c:?}");
    }

    #[test]
    fn sunlit_photo_is_never_live() {
        // ambient IR floods the scene and there is no usable ambient frame: the face is not
        // much brighter than the background, yet shaded like a face
        let cfg = IrLivenessConfig::default();
        for ambient in [None, Some(blank_ambient())] {
            let lit = frame(true, real_face(130, 200, 170, true));
            let (c, v) = matched(&lit, ambient.as_ref());
            assert!(
                c.face_vs_background >= cfg.min_face_vs_background
                    && (cfg.min_relief..=cfg.max_relief).contains(&c.relief),
                "{c:?}"
            );
            assert_eq!(v, IrVerdict::Unknown, "{c:?}");
            // uniformly bright patch on a bright background
            let lit = frame(true, real_face(200, 255, 255, false));
            assert_eq!(matched(&lit, ambient.as_ref()).1, IrVerdict::Unknown);
        }
    }

    #[test]
    fn weak_emitter_contrast_is_unknown() {
        // a face far from the emitter: barely brighter than the background
        let (_, v) = matched(&frame(true, real_face(80, 100, 140, true)), None);
        assert_eq!(v, IrVerdict::Unknown);
    }

    #[test]
    fn degenerate_boxes_and_frames_give_no_cues() {
        let lit = frame(true, real_face(20, 150, 210, true));
        let none = IrAssociator::new();
        for bad in [
            rect(f32::NAN, 0.3, 0.4, 0.4),
            rect(0.3, 0.3, f32::INFINITY, 0.4),
            rect(0.3, 0.3, 0.0, 0.4),
            rect(0.3, 0.3, -0.4, 0.4),
            rect(1.5, 0.3, 0.4, 0.4),
            // mostly outside the frame
            rect(0.85, 0.3, 0.4, 0.4),
            // no room for a background ring
            rect(0.0, 0.0, 1.0, 1.0),
        ] {
            // landmarks inside the box, so the box itself is what is rejected
            let d = Detection {
                landmarks5: [bad.center(); 5],
                ..det(bad)
            };
            assert_eq!(
                ir_cues(&lit, None, &none, bad, &[d], Some(&d)),
                None,
                "{bad:?}"
            );
        }
        let d = det(FACE);
        let cues =
            |lit: &IrFrame, amb: Option<&IrFrame>| ir_cues(lit, amb, &none, FACE, &[d], Some(&d));
        let unlit = frame(false, real_face(20, 150, 210, true));
        assert_eq!(cues(&unlit, None), None);
        // an "ambient" frame that is lit, or of another size
        assert_eq!(cues(&lit, Some(&lit)), None);
        let small = IrFrame::new(0, 0, 10, 10, false, vec![0; 100]).unwrap();
        assert_eq!(cues(&lit, Some(&small)), None);
        let mut broken = lit.clone();
        broken.data = vec![0; 10].into();
        assert_eq!(cues(&broken, None), None);
        // a NaN landmark cannot be placed on the face
        for k in 0..5 {
            let mut d = det(FACE);
            d.landmarks5[k].x = f32::NAN;
            assert_eq!(
                ir_cues(&lit, None, &none, FACE, &[d], Some(&d)),
                None,
                "{k}"
            );
        }
    }

    #[test]
    fn a_cheek_off_the_frame_is_unknown() {
        // a face at the left edge: box x −0.15..0.25, the right cheek square entirely off-frame
        let lit = frame(true, |x, y| match (x, y) {
            (3..8, 52..58) => 210,
            (0..25, 30..70) => 150,
            _ => 20,
        });
        let p = |x, y| PointF { x, y };
        let d = Detection {
            bbox: rect(-0.15, 0.3, 0.4, 0.4),
            landmarks5: [
                p(-0.05, 0.45),
                p(0.12, 0.45),
                p(0.05, 0.55),
                p(-0.03, 0.65),
                p(0.1, 0.65),
            ],
            score: 0.9,
        };
        let c = ir_cues(&lit, None, &IrAssociator::new(), d.bbox, &[d], Some(&d)).unwrap();
        assert!(c.relief.is_nan(), "{c:?}");
        assert_eq!(judge(&c, &IrLivenessConfig::default()), IrVerdict::Unknown);
    }

    #[test]
    fn landmarks_outside_the_face_box_give_no_cues() {
        let lit = frame(true, real_face(20, 150, 210, true));
        let none = IrAssociator::new();
        let cues = |k: usize, p: PointF| {
            let mut d = det(FACE);
            d.landmarks5[k] = p;
            ir_cues(&lit, None, &none, FACE, &[d], Some(&d))
        };
        // the nose on the background ring would measure relief off the face
        assert_eq!(cues(2, PointF { x: 0.9, y: 0.55 }), None);
        assert_eq!(cues(0, PointF { x: 0.42, y: 0.2 }), None);
        // just past the box edge, within the tolerance for a turned head
        assert!(cues(0, PointF { x: 0.29, y: 0.45 }).is_some());
    }

    #[test]
    fn owner_by_a_window_is_never_self_emitting() {
        // sun or window NIR lights face and room alike: small lit − ambient, but the face does
        // not glow against its surroundings in the ambient frame
        let lit = frame(true, real_face(95, 120, 160, false));
        let amb = frame(false, |_, _| 90);
        let (c, v) = matched(&lit, Some(&amb));
        let cfg = IrLivenessConfig::default();
        assert!(
            c.lit_minus_ambient < cfg.min_emitter_fraction * c.lit_face
                && c.lit_background <= cfg.max_background
                && (c.ambient_background - 90.0).abs() < 1e-3,
            "{c:?}"
        );
        assert_eq!(v, IrVerdict::Unknown, "{c:?}");
    }

    #[test]
    fn screen_in_a_dark_room_is_self_emitting() {
        let lit = frame(true, real_face(10, 110, 140, false));
        let amb = frame(false, |x, y| if in_face(x, y) { 100 } else { 5 });
        let (c, v) = matched(&lit, Some(&amb));
        assert!((c.ambient_background - 5.0).abs() < 1e-3, "{c:?}");
        assert_eq!(reason(v), Some(IrSpoofReason::SelfEmitting), "{c:?}");
    }

    #[test]
    fn flood_is_checked_before_no_ir_face() {
        // a real face in sunlight that the IR detector missed
        let lit = frame(true, real_face(150, 200, 250, false));
        let c = ir_cues(&lit, None, &trusted_identity(), FACE, &[], None).unwrap();
        assert!(!c.face_present, "{c:?}");
        assert_eq!(judge(&c, &IrLivenessConfig::default()), IrVerdict::Unknown);
    }

    #[test]
    fn near_black_noise_is_unknown() {
        // 0–4 grey levels: the ratios alone would read a shaded face
        let lit = frame(true, |x, y| match real_face(9, 4, 3, false)(x, y) {
            9 => ((x * 7 + y * 3) % 2) as u8,
            v => v,
        });
        let (c, v) = matched(&lit, None);
        let cfg = IrLivenessConfig::default();
        assert!(
            c.face_vs_background >= cfg.min_face_vs_background
                && (cfg.min_relief..=cfg.max_relief).contains(&c.relief),
            "{c:?}"
        );
        assert_eq!(v, IrVerdict::Unknown, "{c:?}");
    }

    #[test]
    fn decision_order_is_flood_then_self_emission_then_contrast() {
        let cfg = IrLivenessConfig::default();
        let self_lit = |c: &IrCues| {
            c.lit_minus_ambient < cfg.min_emitter_fraction * c.lit_face
                && c.lit_face - c.lit_minus_ambient >= cfg.self_emit_contrast * c.ambient_background
        };
        let amb = frame(false, |x, y| if in_face(x, y) { 140 } else { 60 });
        // flooded and self-lit: flooding wins
        let (c, v) = matched(&frame(true, real_face(110, 170, 200, false)), Some(&amb));
        assert!(
            self_lit(&c) && c.lit_background > cfg.max_background,
            "{c:?}"
        );
        assert_eq!(v, IrVerdict::Unknown, "{c:?}");
        // self-lit with weak emitter contrast: self-emission wins
        let (c, v) = matched(&frame(true, real_face(70, 152, 182, false)), Some(&amb));
        assert!(
            self_lit(&c) && c.face_vs_background < cfg.min_face_vs_background,
            "{c:?}"
        );
        assert_eq!(reason(v), Some(IrSpoofReason::SelfEmitting), "{c:?}");
    }

    #[test]
    fn trusted_mapping_with_a_matched_ir_face_is_live() {
        let lit = frame(true, real_face(20, 150, 130, true));
        let d = det(FACE);
        let c = ir_cues(&lit, None, &trusted_identity(), FACE, &[d], Some(&d)).unwrap();
        assert!(c.face_present, "{c:?}");
        assert_eq!(
            judge(&c, &IrLivenessConfig::default()),
            IrVerdict::Live { score: 1.0 }
        );
    }

    #[test]
    fn non_finite_cues_are_unknown() {
        let (live, v) = matched(&frame(true, real_face(20, 150, 130, true)), None);
        assert!(matches!(v, IrVerdict::Live { .. }));
        let cfg = IrLivenessConfig::default();
        for f in [
            |c: &mut IrCues| c.face_vs_background = f32::NAN,
            |c: &mut IrCues| c.relief = f32::NAN,
            |c: &mut IrCues| c.lit_face = f32::NAN,
            |c: &mut IrCues| c.lit_background = f32::NAN,
            |c: &mut IrCues| c.face_vs_background = f32::INFINITY,
        ] {
            let mut c = live;
            f(&mut c);
            assert_eq!(judge(&c, &cfg), IrVerdict::Unknown, "{c:?}");
        }
    }

    #[test]
    fn blank_ambient_needs_a_tolerance() {
        assert!(ambient_is_blank(&blank_ambient()));
        assert!(ambient_is_blank(&frame(false, |_, _| 0)));
        // a dark but real ambient frame
        assert!(!ambient_is_blank(&frame(false, |_, _| 1)));
        // mean 0.5, but 2 % of pixels at 25: a real (dim) scene
        assert!(!ambient_is_blank(&frame(false, |x, y| {
            if (x + y * W).is_multiple_of(50) {
                25
            } else {
                0
            }
        })));
    }
}
