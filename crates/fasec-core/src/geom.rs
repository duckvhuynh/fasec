//! Normalised (0..1) 2-D geometry.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PointF {
    pub x: f32,
    pub y: f32,
}

/// Axis-aligned rectangle: top-left `(x, y)`, size `(w, h)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RectF {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl RectF {
    /// Area; zero for degenerate (non-positive) sizes.
    pub fn area(&self) -> f32 {
        self.w.max(0.0) * self.h.max(0.0)
    }

    pub fn center(&self) -> PointF {
        PointF {
            x: self.x + self.w / 2.0,
            y: self.y + self.h / 2.0,
        }
    }

    /// Intersection over union; 0 when both rects are empty.
    pub fn iou(&self, other: &RectF) -> f32 {
        let iw = ((self.x + self.w).min(other.x + other.w) - self.x.max(other.x)).max(0.0);
        let ih = ((self.y + self.h).min(other.y + other.h) - self.y.max(other.y)).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }

    /// Intersection with the unit square `[0,1]x[0,1]`; fully outside gives a zero size.
    pub fn clamp_unit(&self) -> RectF {
        let x0 = self.x.clamp(0.0, 1.0);
        let y0 = self.y.clamp(0.0, 1.0);
        let x1 = (self.x + self.w).clamp(0.0, 1.0);
        let y1 = (self.y + self.h).clamp(0.0, 1.0);
        RectF {
            x: x0,
            y: y0,
            w: (x1 - x0).max(0.0),
            h: (y1 - y0).max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(x: f32, y: f32, w: f32, h: f32) -> RectF {
        RectF { x, y, w, h }
    }

    fn assert_rect_eq(a: RectF, b: RectF) {
        let close = |p: f32, q: f32| (p - q).abs() < 1e-6;
        assert!(
            close(a.x, b.x) && close(a.y, b.y) && close(a.w, b.w) && close(a.h, b.h),
            "{a:?} != {b:?}"
        );
    }

    #[test]
    fn iou_identical_is_one() {
        let a = r(0.1, 0.2, 0.3, 0.4);
        assert!((a.iou(&a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_disjoint_is_zero() {
        assert_eq!(r(0.0, 0.0, 0.1, 0.1).iou(&r(0.5, 0.5, 0.1, 0.1)), 0.0);
    }

    #[test]
    fn iou_half_overlap() {
        let iou = r(0.0, 0.0, 1.0, 1.0).iou(&r(0.5, 0.0, 1.0, 1.0));
        assert!((iou - 1.0 / 3.0).abs() < 1e-6, "{iou}");
    }

    #[test]
    fn iou_of_empty_rects_is_zero() {
        let z = r(0.5, 0.5, 0.0, 0.0);
        assert_eq!(z.iou(&z), 0.0);
    }

    #[test]
    fn area_and_center() {
        let a = r(0.2, 0.4, 0.2, 0.4);
        assert!((a.area() - 0.08).abs() < 1e-6);
        let c = a.center();
        assert!((c.x - 0.3).abs() < 1e-6 && (c.y - 0.6).abs() < 1e-6);
    }

    #[test]
    fn clamp_unit_clips_to_0_1() {
        assert_rect_eq(r(-0.2, 0.5, 0.5, 0.8).clamp_unit(), r(0.0, 0.5, 0.3, 0.5));
        assert_rect_eq(r(0.9, 1.2, 0.5, 0.5).clamp_unit(), r(0.9, 1.0, 0.1, 0.0));
        assert_rect_eq(r(0.1, 0.1, 0.2, 0.2).clamp_unit(), r(0.1, 0.1, 0.2, 0.2));
    }
}
