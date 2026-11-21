//! Image helpers shared by the model stages. All buffers are RGB8, row-major.

use fasec_core::Frame;
use image::imageops::{self, FilterType};
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::Array4;

/// Borrowed view of the frame; `None` only if `data` does not match `width*height*3`.
fn view(frame: &Frame) -> Option<ImageBuffer<Rgb<u8>, &[u8]>> {
    ImageBuffer::from_raw(frame.width, frame.height, &frame.data[..])
}

fn black(w: u32, h: u32) -> Vec<u8> {
    vec![0; w as usize * h as usize * 3]
}

/// Resizes the whole frame to `w`×`h` (bilinear).
pub fn resize_rgb(frame: &Frame, w: u32, h: u32) -> Vec<u8> {
    match view(frame) {
        Some(img) => imageops::resize(&img, w, h, FilterType::Triangle).into_raw(),
        None => black(w, h),
    }
}

/// Crops `bbox_px` (`[x, y, w, h]` in pixels) expanded around its centre by `scale`, pads with
/// black outside the frame, and resizes to `out`×`out`.
pub fn crop_scaled(frame: &Frame, bbox_px: [f32; 4], scale: f32, out: u32) -> Vec<u8> {
    let mut dst = RgbImage::new(out, out);
    let [x, y, w, h] = bbox_px;
    let (cw, ch) = (w * scale, h * scale);
    let (x0, y0) = (x + w / 2.0 - cw / 2.0, y + h / 2.0 - ch / 2.0);
    let Some(src) = view(frame) else {
        return dst.into_raw();
    };
    if out == 0 || !(cw > 0.0 && ch > 0.0 && x0.is_finite() && y0.is_finite()) {
        return dst.into_raw();
    }
    // Only the in-frame part is resampled and pasted; the rest of `dst` stays black, so huge
    // scales or boxes far outside the frame cost nothing extra.
    let (fw, fh) = (frame.width as f32, frame.height as f32);
    let (sx0, sx1) = (x0.round().clamp(0.0, fw), (x0 + cw).round().clamp(0.0, fw));
    let (sy0, sy1) = (y0.round().clamp(0.0, fh), (y0 + ch).round().clamp(0.0, fh));
    let (kx, ky) = (out as f32 / cw, out as f32 / ch);
    let o = out as f32;
    let (dx0, dx1) = (
        ((sx0 - x0) * kx).round().clamp(0.0, o),
        ((sx1 - x0) * kx).round().clamp(0.0, o),
    );
    let (dy0, dy1) = (
        ((sy0 - y0) * ky).round().clamp(0.0, o),
        ((sy1 - y0) * ky).round().clamp(0.0, o),
    );
    if sx1 > sx0 && sy1 > sy0 && dx1 > dx0 && dy1 > dy0 {
        let part = imageops::crop_imm(
            &src,
            sx0 as u32,
            sy0 as u32,
            (sx1 - sx0) as u32,
            (sy1 - sy0) as u32,
        );
        let part = imageops::resize(
            &*part,
            (dx1 - dx0) as u32,
            (dy1 - dy0) as u32,
            FilterType::Triangle,
        );
        imageops::replace(&mut dst, &part, dx0 as i64, dy0 as i64);
    }
    dst.into_raw()
}

/// Packs an RGB8 image into a `1×3×h×w` tensor with values 0..255 (no normalisation);
/// `bgr` swaps the channel order to B, G, R. A too-short `rgb` reads as 0 instead of panicking.
pub fn to_nchw_f32(rgb: &[u8], w: u32, h: u32, bgr: bool) -> Array4<f32> {
    let (w, h) = (w as usize, h as usize);
    Array4::from_shape_fn((1, 3, h, w), |(_, c, y, x)| {
        let c = if bgr { 2 - c } else { c };
        f32::from(rgb.get((y * w + x) * 3 + c).copied().unwrap_or(0))
    })
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
    fn to_nchw_shape_and_channel_order() {
        let rgb = [1, 2, 3, 4, 5, 6]; // 2×1: (1,2,3) (4,5,6)
        let t = to_nchw_f32(&rgb, 2, 1, false);
        assert_eq!(t.shape(), &[1, 3, 1, 2]);
        assert_eq!(t[[0, 0, 0, 0]], 1.0);
        assert_eq!(t[[0, 1, 0, 0]], 2.0);
        assert_eq!(t[[0, 2, 0, 1]], 6.0);
        let t = to_nchw_f32(&rgb, 2, 1, true);
        assert_eq!(t[[0, 0, 0, 0]], 3.0);
        assert_eq!(t[[0, 2, 0, 0]], 1.0);
        assert_eq!(t[[0, 0, 0, 1]], 6.0);
        assert_eq!(t[[0, 1, 0, 1]], 5.0);
        assert_eq!(t[[0, 2, 0, 1]], 4.0);
    }

    #[test]
    fn to_nchw_short_input_does_not_panic() {
        let t = to_nchw_f32(&[9, 9], 2, 2, false);
        assert_eq!(t.shape(), &[1, 3, 2, 2]);
        assert_eq!(t[[0, 0, 0, 0]], 9.0);
        assert_eq!(t[[0, 2, 1, 1]], 0.0);
    }

    #[test]
    fn resize_rgb_output_len_and_uniform_colour() {
        let f = frame(8, 6, |_, _| [10, 20, 30]);
        let out = resize_rgb(&f, 5, 3);
        assert_eq!(out.len(), 5 * 3 * 3);
        assert!(out.as_chunks::<3>().0.iter().all(|p| *p == [10, 20, 30]));
    }

    #[test]
    fn crop_scaled_pads_outside_frame() {
        let f = frame(10, 10, |_, _| [200, 200, 200]);
        // 4×4 box at the top-left corner, doubled: crop spans -2..6, so the first 2 rows and
        // columns of the 8×8 output are outside the frame.
        let out = crop_scaled(&f, [0.0, 0.0, 4.0, 4.0], 2.0, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
        let px = |x: usize, y: usize| &out[(y * 8 + x) * 3..(y * 8 + x) * 3 + 3];
        assert_eq!(px(0, 0), [0, 0, 0]);
        assert_eq!(px(7, 0), [0, 0, 0]);
        assert_eq!(px(0, 7), [0, 0, 0]);
        assert_eq!(px(7, 7), [200, 200, 200]);
        // fully outside, degenerate and non-finite boxes: all black, no panic
        let out = crop_scaled(&f, [50.0, 50.0, 4.0, 4.0], 1.0, 4);
        assert!(out.iter().all(|&v| v == 0));
        assert_eq!(
            crop_scaled(&f, [1.0, 1.0, 0.0, 0.0], 1.0, 4).len(),
            4 * 4 * 3
        );
        assert_eq!(
            crop_scaled(&f, [f32::NAN, 1.0, 3.0, 3.0], 1.0, 4).len(),
            4 * 4 * 3
        );
        assert!(crop_scaled(&f, [1.0, 1.0, 3.0, 3.0], 1.0, 0).is_empty());
    }

    #[test]
    fn crop_scaled_centre_matches_source() {
        let f = frame(16, 12, |x, y| [x as u8 * 10, y as u8 * 10, 7]);
        // box 4..8 × 3..7, scale 1, out 4: identity copy of that region
        let out = crop_scaled(&f, [4.0, 3.0, 4.0, 4.0], 1.0, 4);
        for y in 0..4u32 {
            for x in 0..4u32 {
                let i = ((y * 4 + x) * 3) as usize;
                assert_eq!(&out[i..i + 3], [(x + 4) as u8 * 10, (y + 3) as u8 * 10, 7]);
            }
        }
        // a box whose centre pixel is distinct keeps that pixel at the output centre
        let f = frame(9, 9, |x, y| {
            if (x, y) == (4, 4) {
                [255, 0, 0]
            } else {
                [0, 0, 255]
            }
        });
        let out = crop_scaled(&f, [3.0, 3.0, 3.0, 3.0], 1.0, 3);
        assert_eq!(&out[(3 + 1) * 3..(3 + 1) * 3 + 3], [255, 0, 0]);
    }
}
