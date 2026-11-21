//! 5-point similarity alignment, as OpenCV's `FaceRecognizerSF::alignCrop`.

use fasec_core::{Frame, PointF};
use nalgebra::{Matrix2x3, Vector2};

/// ArcFace 112×112 template in pixels, in YuNet landmark order.
pub const ARCFACE_112: [PointF; 5] = [
    PointF {
        x: 38.2946,
        y: 51.6963,
    },
    PointF {
        x: 73.5318,
        y: 51.5014,
    },
    PointF {
        x: 56.0252,
        y: 71.7366,
    },
    PointF {
        x: 41.5493,
        y: 92.3655,
    },
    PointF {
        x: 70.7299,
        y: 92.2041,
    },
];

/// Least-squares similarity (rotation, uniform scale, translation; never a reflection) mapping
/// `src` onto `dst`, as a forward `dst ≈ M·[x, y, 1]` matrix (Umeyama 1991).
///
/// In 2-D the Umeyama solution restricted to proper rotations has this closed form: with both
/// point sets centred, `c·R = [[a, -b], [b, a]]`, `a = Σ s·d / Σ|s|²`, `b = Σ s×d / Σ|s|²`.
/// Degenerate `src` (all points equal) or non-finite input gives a non-finite matrix.
pub fn similarity_transform(src: &[PointF; 5], dst: &[PointF; 5]) -> Matrix2x3<f32> {
    let mean = |p: &[PointF; 5]| {
        let n = p.len() as f32;
        (
            p.iter().map(|q| q.x).sum::<f32>() / n,
            p.iter().map(|q| q.y).sum::<f32>() / n,
        )
    };
    let ((sx, sy), (dx, dy)) = (mean(src), mean(dst));
    let (mut dot, mut cross, mut var) = (0.0, 0.0, 0.0);
    for (s, d) in src.iter().zip(dst) {
        let (s, d) = ((s.x - sx, s.y - sy), (d.x - dx, d.y - dy));
        dot += s.0 * d.0 + s.1 * d.1;
        cross += s.0 * d.1 - s.1 * d.0;
        var += s.0 * s.0 + s.1 * s.1;
    }
    let (a, b) = (dot / var, cross / var);
    Matrix2x3::new(a, -b, dx - (a * sx - b * sy), b, a, dy - (b * sx + a * sy))
}

/// Warps `frame` by the forward transform `m` (frame pixels → output pixels) into an
/// `out_w`×`out_h` RGB8 image, as `cv::warpAffine` with `INTER_LINEAR` and a black constant
/// border: samples outside the frame read as black. A singular or non-finite `m` gives black.
pub fn warp_affine_rgb(frame: &Frame, m: &Matrix2x3<f32>, out_w: u32, out_h: u32) -> Vec<u8> {
    let mut out = vec![0u8; out_w as usize * out_h as usize * 3];
    let Some(inv) = m.fixed_view::<2, 2>(0, 0).into_owned().try_inverse() else {
        return out;
    };
    let t = -(inv * m.column(2));
    if !inv.iter().chain(t.iter()).all(|v| v.is_finite()) {
        return out;
    }
    let (fw, fh) = (i64::from(frame.width), i64::from(frame.height));
    let at = |x: i64, y: i64, c: usize| -> f32 {
        if x < 0 || y < 0 || x >= fw || y >= fh {
            return 0.0;
        }
        let i = ((y * fw + x) * 3) as usize + c;
        frame.data.get(i).copied().map_or(0.0, f32::from)
    };
    for (i, px) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {
        let o = Vector2::new((i % out_w as usize) as f32, (i / out_w as usize) as f32);
        let s = inv * o + t;
        // Fully outside (or NaN): stays black. Also keeps the casts below far from overflow.
        if !(s.x > -1.0 && s.y > -1.0 && s.x < fw as f32 && s.y < fh as f32) {
            continue;
        }
        let (x0, y0) = (s.x.floor(), s.y.floor());
        let (fx, fy) = (s.x - x0, s.y - y0);
        let (x0, y0) = (x0 as i64, y0 as i64);
        for (c, v) in px.iter_mut().enumerate() {
            let top = at(x0, y0, c) * (1.0 - fx) + at(x0 + 1, y0, c) * fx;
            let bottom = at(x0, y0 + 1, c) * (1.0 - fx) + at(x0 + 1, y0 + 1, c) * fx;
            *v = (top * (1.0 - fy) + bottom * fy).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Frame {
        let data = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .flat_map(|(x, y)| f(x, y))
            .collect();
        Frame::new(0, 0, w, h, data).unwrap()
    }

    #[test]
    fn umeyama_recovers_known_similarity() {
        let (s, th, tx, ty) = (1.5f32, 10f32.to_radians(), 20.0, -7.0);
        let (a, b) = (s * th.cos(), s * th.sin());
        let want = Matrix2x3::new(a, -b, tx, b, a, ty);
        let dst = ARCFACE_112.map(|p| PointF {
            x: a * p.x - b * p.y + tx,
            y: b * p.x + a * p.y + ty,
        });
        let got = similarity_transform(&ARCFACE_112, &dst);
        assert!((got - want).abs().max() < 1e-3, "{got} vs {want}");

        // mirrored target: best fit is still a rotation + scale (positive determinant)
        let mirrored = ARCFACE_112.map(|p| PointF { x: -p.x, y: p.y });
        let m = similarity_transform(&ARCFACE_112, &mirrored);
        assert!(m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)] >= 0.0, "{m}");
    }

    #[test]
    fn warp_identity_preserves_pixels() {
        let f = frame(7, 5, |x, y| [x as u8 * 30, y as u8 * 40, 9]);
        let id = Matrix2x3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        assert_eq!(warp_affine_rgb(&f, &id, 7, 5), &f.data[..]);
        // pure integer translation: output (x, y) = frame (x + 2, y + 1)
        let shift = Matrix2x3::new(1.0, 0.0, -2.0, 0.0, 1.0, -1.0);
        let out = warp_affine_rgb(&f, &shift, 3, 2);
        assert_eq!(&out[..3], [60, 40, 9]);
        assert_eq!(&out[out.len() - 3..], [120, 80, 9]);
    }

    #[test]
    fn warp_outside_frame_is_black_no_panic() {
        let (w, h) = (64u32, 48u32);
        let f = frame(w, h, |_, _| [200, 200, 200]);
        // a face cut by the left edge: YuNet reports landmarks with x < 0 (normalised)
        let norm = [
            (-0.08, 0.40),
            (0.10, 0.40),
            (0.01, 0.55),
            (-0.06, 0.70),
            (0.08, 0.70),
        ];
        let src = norm.map(|(x, y)| PointF {
            x: x * w as f32,
            y: y * h as f32,
        });
        let m = similarity_transform(&src, &ARCFACE_112);
        let out = warp_affine_rgb(&f, &m, 112, 112);
        assert_eq!(out.len(), 112 * 112 * 3);
        let px = |x: usize, y: usize| &out[(y * 112 + x) * 3..(y * 112 + x) * 3 + 3];
        assert_eq!(px(0, 56), [0, 0, 0], "left of the frame edge is black");
        assert_eq!(px(111, 56), [200, 200, 200], "inside the frame is copied");

        // fully outside, degenerate (all points equal) and non-finite inputs: black, no panic
        let far = src.map(|p| PointF {
            x: p.x + 1e6,
            y: p.y,
        });
        let nan = [PointF {
            x: f32::NAN,
            y: 0.0,
        }; 5];
        let same = [PointF { x: 3.0, y: 3.0 }; 5];
        for src in [far, nan, same] {
            let m = similarity_transform(&src, &ARCFACE_112);
            let out = warp_affine_rgb(&f, &m, 112, 112);
            assert_eq!(out.len(), 112 * 112 * 3);
            assert!(out.iter().all(|&v| v == 0), "{m}");
        }
        assert!(warp_affine_rgb(&f, &Matrix2x3::from_element(f32::MAX), 4, 4).len() == 48);
        assert!(warp_affine_rgb(&f, &m, 0, 0).is_empty());
    }
}
