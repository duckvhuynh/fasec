//! Rotation (clockwise) and horizontal mirror of packed RGB8 frames.

/// Rotates `rgb` (`w`x`h`, RGB8) clockwise by `rotation` degrees (0/90/180/270; anything else
/// is treated as 0) and then mirrors horizontally if `mirror`. Returns `(data, width, height)`.
/// `rgb.len()` must equal `w * h * 3`.
pub fn apply(rgb: &[u8], w: u32, h: u32, rotation: u16, mirror: bool) -> (Vec<u8>, u32, u32) {
    let rotation = if matches!(rotation, 90 | 180 | 270) {
        rotation
    } else {
        0
    };
    if rotation == 0 && !mirror {
        return (rgb.to_vec(), w, h);
    }
    let (ow, oh) = if rotation % 180 == 90 { (h, w) } else { (w, h) };
    let mut out = Vec::with_capacity(rgb.len());
    for oy in 0..oh {
        for ox in 0..ow {
            let mx = if mirror { ow - 1 - ox } else { ox };
            let (sx, sy) = match rotation {
                90 => (oy, h - 1 - mx),
                180 => (w - 1 - mx, h - 1 - oy),
                270 => (w - 1 - oy, mx),
                _ => (mx, oy),
            };
            let i = (sy as usize * w as usize + sx as usize) * 3;
            out.extend_from_slice(&rgb[i..i + 3]);
        }
    }
    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 3x2 image whose pixel (x, y) is `[x, y, 9]`.
    fn img3x2() -> Vec<u8> {
        let mut v = Vec::new();
        for y in 0..2u8 {
            for x in 0..3u8 {
                v.extend_from_slice(&[x, y, 9]);
            }
        }
        v
    }

    fn px(data: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * w + x) * 3) as usize;
        [data[i], data[i + 1], data[i + 2]]
    }

    #[test]
    fn transform_identity() {
        assert_eq!(apply(&img3x2(), 3, 2, 0, false), (img3x2(), 3, 2));
    }

    #[test]
    fn transform_rotate_90_dims_and_pixel() {
        let (out, w, h) = apply(&img3x2(), 3, 2, 90, false);
        assert_eq!((w, h), (2, 3));
        // Clockwise: the source bottom-left lands top-left, the source top-left lands top-right.
        assert_eq!(px(&out, w, 0, 0), [0, 1, 9]);
        assert_eq!(px(&out, w, 1, 0), [0, 0, 9]);
        assert_eq!(px(&out, w, 1, 2), [2, 0, 9]);
    }

    #[test]
    fn transform_rotate_180_and_270() {
        let (out, w, h) = apply(&img3x2(), 3, 2, 180, false);
        assert_eq!((w, h), (3, 2));
        assert_eq!(px(&out, w, 0, 0), [2, 1, 9]);
        let (out, w, h) = apply(&img3x2(), 3, 2, 270, false);
        assert_eq!((w, h), (2, 3));
        // Counter-clockwise: the source top-right lands top-left.
        assert_eq!(px(&out, w, 0, 0), [2, 0, 9]);
        assert_eq!(px(&out, w, 1, 2), [0, 1, 9]);
    }

    #[test]
    fn transform_mirror() {
        let (out, w, h) = apply(&img3x2(), 3, 2, 0, true);
        assert_eq!((w, h), (3, 2));
        assert_eq!(px(&out, w, 0, 0), [2, 0, 9]);
        assert_eq!(px(&out, w, 2, 1), [0, 1, 9]);
        // Mirror is applied after rotation.
        let (out, w, _) = apply(&img3x2(), 3, 2, 90, true);
        assert_eq!(px(&out, w, 0, 0), [0, 0, 9]);
    }

    type Grid = Vec<Vec<[u8; 3]>>;

    fn to_grid(data: &[u8], w: usize) -> Grid {
        data.chunks(w * 3)
            .map(|row| row.chunks(3).map(|p| [p[0], p[1], p[2]]).collect())
            .collect()
    }

    /// Independent oracle: clockwise quarter turn, `new[r][c] = old[h - 1 - c][r]`.
    fn rot90(g: &Grid) -> Grid {
        let (h, w) = (g.len(), g[0].len());
        (0..w)
            .map(|r| (0..h).map(|c| g[h - 1 - c][r]).collect())
            .collect()
    }

    /// Distinct pixel per cell.
    fn img(w: u32, h: u32) -> Vec<u8> {
        (0..h)
            .flat_map(|y| (0..w).flat_map(move |x| [x as u8, y as u8, (y * w + x) as u8]))
            .collect()
    }

    #[test]
    fn transform_matches_reference_for_every_rotation_mirror_and_shape() {
        for (w, h) in [(3u32, 2u32), (1, 4), (4, 1), (5, 3), (1, 1)] {
            let src = img(w, h);
            for rotation in [0u16, 90, 180, 270] {
                for mirror in [false, true] {
                    let mut want = to_grid(&src, w as usize);
                    for _ in 0..rotation / 90 {
                        want = rot90(&want);
                    }
                    if mirror {
                        want.iter_mut().for_each(|row| row.reverse());
                    }
                    let (out, ow, oh) = apply(&src, w, h, rotation, mirror);
                    let case = format!("{w}x{h} rot {rotation} mirror {mirror}");
                    assert_eq!(
                        (ow as usize, oh as usize),
                        (want[0].len(), want.len()),
                        "{case}"
                    );
                    assert_eq!(to_grid(&out, ow as usize), want, "{case}");
                }
            }
        }
    }

    #[test]
    fn transform_unsupported_rotation_acts_as_zero_and_four_turns_are_identity() {
        let src = img(3, 2);
        for rotation in [45u16, 360, 450, u16::MAX] {
            assert_eq!(apply(&src, 3, 2, rotation, false), (src.clone(), 3, 2));
            assert_eq!(
                apply(&src, 3, 2, rotation, true),
                apply(&src, 3, 2, 0, true)
            );
        }
        let (mut data, mut w, mut h) = (src.clone(), 3, 2);
        for _ in 0..4 {
            (data, w, h) = apply(&data, w, h, 90, false);
        }
        assert_eq!((data, w, h), (src, 3, 2));
    }
}
