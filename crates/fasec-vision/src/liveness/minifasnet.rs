//! MiniFASNet RGB passive anti-spoofing (minivision-ai/Silent-Face-Anti-Spoofing, Apache-2.0).
//!
//! Matches upstream `test.py` at commit b6d5f04: per model, `CropImage.crop` (box scaled about its
//! centre, scale capped and the window *shifted* to stay inside the frame, no padding), OpenCV
//! bilinear resize to 80×80, BGR float 0..255 NCHW (the repo's own `ToTensor` does not divide by
//! 255), softmax; the two softmaxes are summed and `live = sum[1] / 2`.

use std::path::Path;

use fasec_core::{Frame, LivenessTier};
use ort::session::Session;
use ort::value::TensorRef;

use crate::image_ops::to_nchw_f32;
use crate::runtime::StageSession;
use crate::{Detection, LivenessModel, Runtime, VisionError};

/// Input side of both models.
const SIZE: u32 = 80;
/// Class index of "real face" in the 3-class output (upstream `label == 1`).
const LIVE: usize = 1;

struct Stage {
    session: StageSession,
    input: String,
    output: String,
    /// Crop scale from the upstream file name (`2.7_80x80_…`, `4_0_0_80x80_…`).
    scale: f32,
}

impl Stage {
    fn new(rt: &Runtime, model: &Path, scale: f32) -> Result<Stage, VisionError> {
        let session = rt.stage_session(model)?;
        let s: &Session = session.session();
        let bad = |why: &str| VisionError::Model(format!("{}: {why}", model.display()));
        let ([input], [output]) = (s.inputs(), s.outputs()) else {
            return Err(bad("expected one input and one output"));
        };
        // batch is dynamic (-1) in the exports
        match input.dtype().tensor_shape().map(|s| &s[..]) {
            Some(&[b, 3, 80, 80]) if b == 1 || b == -1 => {}
            other => return Err(bad(&format!("unsupported input shape {other:?}"))),
        }
        tracing::info!(ep = session.ep(), model = %model.display(), "MiniFASNet loaded");
        Ok(Stage {
            input: input.name().to_string(),
            output: output.name().to_string(),
            session,
            scale,
        })
    }

    /// Softmax of the 3-class output for this stage's crop of `bbox_px`.
    fn probs(&mut self, frame: &Frame, bbox_px: [f32; 4]) -> Result<[f32; 3], VisionError> {
        let crop = crop_upstream(frame, bbox_px, self.scale, SIZE);
        let x = to_nchw_f32(&crop, SIZE, SIZE, true);
        let (input, output) = (self.input.as_str(), self.output.as_str());
        let logits = self.session.run(|s| {
            let outs = s.run(ort::inputs![input => TensorRef::from_array_view(&x)?])?;
            let v = outs
                .get(output)
                .ok_or_else(|| VisionError::Model(format!("MiniFASNet output {output} missing")))?;
            let (_, v) = v.try_extract_tensor::<f32>()?;
            <[f32; 3]>::try_from(v)
                .map_err(|_| VisionError::Model(format!("MiniFASNet output len {}", v.len())))
        })?;
        Ok(softmax(logits))
    }
}

/// Both upstream models; the sum of their softmaxes decides (upstream `test.py`).
pub struct MiniFasNet {
    v2: Stage,
    v1se: Stage,
}

impl MiniFasNet {
    /// `v2` = `minifasnet-v2` (crop scale 2.7), `v1se` = `minifasnet-v1se` (crop scale 4.0); both
    /// must be SHA-256 verified by the caller (`fasec_models::verify_file`).
    pub fn new(rt: &Runtime, v2: &Path, v1se: &Path) -> Result<MiniFasNet, VisionError> {
        Ok(MiniFasNet {
            v2: Stage::new(rt, v2, 2.7)?,
            v1se: Stage::new(rt, v1se, 4.0)?,
        })
    }

    /// EPs of the v2 and v1se sessions (`"directml"` or `"cpu"`); each may fall back on its own.
    pub fn ep(&self) -> [&'static str; 2] {
        [self.v2.session.ep(), self.v1se.session.ep()]
    }
}

impl LivenessModel for MiniFasNet {
    fn score(&mut self, frame: &Frame, det: &Detection) -> Result<f32, VisionError> {
        let (fw, fh) = (frame.width, frame.height);
        if fw == 0 || fh == 0 || frame.data.len() != fw as usize * fh as usize * 3 {
            return Err(VisionError::Input(format!("bad frame {fw}x{fh}")));
        }
        let b = det.bbox;
        let (fw, fh) = (fw as f32, fh as f32);
        let px = [b.x * fw, b.y * fh, b.w * fw, b.h * fh];
        if !px.iter().all(|v| v.is_finite()) || px[2] <= 0.0 || px[3] <= 0.0 {
            return Err(VisionError::Input(format!("unusable face box {b:?}")));
        }
        let (a, c) = (self.v2.probs(frame, px)?, self.v1se.probs(frame, px)?);
        let live = (a[LIVE] + c[LIVE]) / 2.0;
        if !live.is_finite() {
            return Err(VisionError::Model(format!(
                "MiniFASNet non-finite output {a:?} {c:?}"
            )));
        }
        Ok(live.clamp(0.0, 1.0))
    }

    fn tier(&self) -> LivenessTier {
        LivenessTier::Rgb
    }
}

fn softmax(v: [f32; 3]) -> [f32; 3] {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e = v.map(|x| (x - m).exp());
    let sum: f32 = e.iter().sum();
    e.map(|x| x / sum)
}

/// Upstream `CropImage._get_new_box`: inclusive pixel box `[left, top, right, bottom]`.
fn new_box(sw: usize, sh: usize, [x, y, bw, bh]: [f64; 4], scale: f64) -> [usize; 4] {
    let (sw1, sh1) = ((sw - 1) as f64, (sh - 1) as f64);
    let scale = scale.min(sh1 / bh).min(sw1 / bw);
    let (nw, nh) = (bw * scale, bh * scale);
    let (cx, cy) = (x + bw / 2.0, y + bh / 2.0);
    let (mut l, mut t) = (cx - nw / 2.0, cy - nh / 2.0);
    let (mut r, mut b) = (cx + nw / 2.0, cy + nh / 2.0);
    if l < 0.0 {
        r -= l;
        l = 0.0;
    }
    if t < 0.0 {
        b -= t;
        t = 0.0;
    }
    if r > sw1 {
        l -= r - sw1;
        r = sw1;
    }
    if b > sh1 {
        t -= b - sh1;
        b = sh1;
    }
    // `int()` truncates; the clamp only catches round-off on huge or far-away boxes
    let fix = |v: f64, max: f64| v.trunc().clamp(0.0, max) as usize;
    let (l, t) = (fix(l, sw1), fix(t, sh1));
    [l, t, fix(r, sw1).max(l), fix(b, sh1).max(t)]
}

/// Upstream `CropImage.crop` (crop=True): the [`new_box`] region resized to `out`×`out` like
/// `cv2.resize` INTER_LINEAR (half-pixel centres, edge replicate, no antialiasing, so print and
/// screen texture survive the downscale). RGB8 in and out; `bbox_px` is `[x, y, w, h]` pixels.
fn crop_upstream(frame: &Frame, bbox_px: [f32; 4], scale: f32, out: u32) -> Vec<u8> {
    let (sw, sh, n) = (frame.width as usize, frame.height as usize, out as usize);
    if sw == 0 || sh == 0 || frame.data.len() != sw * sh * 3 {
        return vec![0; n * n * 3];
    }
    let [l, t, r, b] = new_box(sw, sh, bbox_px.map(f64::from), f64::from(scale));
    // per output index: the two source taps and the second one's weight (OpenCV's clamping)
    let taps = |len: usize| -> Vec<(usize, usize, f32)> {
        let k = len as f64 / n as f64;
        (0..n)
            .map(|d| {
                let f = (d as f64 + 0.5) * k - 0.5;
                let i = f.floor();
                let (i, w) = if i < 0.0 {
                    (0, 0.0)
                } else if i as usize >= len - 1 {
                    (len - 1, 0.0)
                } else {
                    (i as usize, (f - i) as f32)
                };
                (i, (i + 1).min(len - 1), w)
            })
            .collect()
    };
    let (xs, ys) = (taps(r - l + 1), taps(b - t + 1));
    let px = |x: usize, y: usize, c: usize| f32::from(frame.data[((t + y) * sw + l + x) * 3 + c]);
    let mut dst = Vec::with_capacity(n * n * 3);
    for &(y0, y1, wy) in &ys {
        for &(x0, x1, wx) in &xs {
            for c in 0..3 {
                let top = px(x0, y0, c) * (1.0 - wx) + px(x1, y0, c) * wx;
                let bot = px(x0, y1, c) * (1.0 - wx) + px(x1, y1, c) * wx;
                dst.push((top * (1.0 - wy) + bot * wy).round() as u8);
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Instant;

    use fasec_config::ExecutionProvider;
    use fasec_core::{PointF, RectF};
    use image::RgbImage;

    use super::*;
    use crate::runtime::tests::verified_yunet;
    use crate::{FaceDetector, YuNet, YuNetConfig};

    fn frame(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Frame {
        let data = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .flat_map(|(x, y)| f(x, y))
            .collect();
        Frame::new(0, 0, w, h, data).unwrap()
    }

    #[test]
    fn crop_shifts_box_inside_frame_like_upstream() {
        // x-gradient frame, 100×50; a 10×10 box at the left edge, scale 2.7 → 27×27 crop
        // shifted right so it starts at x = 0 (no black padding)
        let f = frame(100, 50, |x, _| [x as u8, 0, 0]);
        let out = crop_upstream(&f, [-3.0, 20.0, 10.0, 10.0], 2.7, 27);
        assert_eq!(out.len(), 27 * 27 * 3);
        assert!(out.iter().any(|&v| v > 0));
        // upstream shifts the 27-px window to x 0..=27 (inclusive right edge, 28 px) and
        // resizes to 27: first column ≈ 0, last ≈ 27, never padded black on the right
        assert!(out[0] <= 1, "{}", out[0]);
        let last = out[(26) * 3] as i32;
        assert!((last - 27).abs() <= 1, "{last}");
        // scale is capped so the crop fits the frame: (h-1)/box_h = 49/10 < 5.0
        let out = crop_upstream(&f, [45.0, 20.0, 10.0, 10.0], 5.0, 10);
        assert_eq!(out.len(), 10 * 10 * 3);
    }

    #[test]
    fn crop_degenerate_boxes_do_not_panic() {
        let f = frame(8, 6, |_, _| [9, 9, 9]);
        for b in [
            [100.0, 100.0, 4.0, 4.0],
            [-100.0, -100.0, 4.0, 4.0],
            [0.0, 0.0, 1e30, 1e30],
            [2.0, 2.0, 0.5, 0.5],
        ] {
            let out = crop_upstream(&f, b, 4.0, 80);
            assert_eq!(out.len(), 80 * 80 * 3, "{b:?}");
            assert!(out.iter().all(|&v| v == 9), "{b:?}");
        }
        let one = frame(1, 1, |_, _| [5, 6, 7]);
        assert!(
            crop_upstream(&one, [0.0, 0.0, 1.0, 1.0], 2.7, 4)
                .chunks(3)
                .all(|p| p == [5, 6, 7])
        );
    }

    fn verified(id: &str) -> PathBuf {
        let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
        let manifest = fasec_models::Manifest::load(models.join("manifest.toml")).unwrap();
        fasec_models::ensure(&models, manifest.get(id).unwrap()).unwrap()
    }

    fn minifasnet(ep: ExecutionProvider) -> MiniFasNet {
        let rt = Runtime::new(ep).unwrap();
        MiniFasNet::new(
            &rt,
            &verified("minifasnet-v2"),
            &verified("minifasnet-v1se"),
        )
        .unwrap()
    }

    fn load(name: &str) -> Frame {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../.assets/{name}.jpg"));
        let img: RgbImage = image::open(p).unwrap().to_rgb8();
        Frame::new(0, 0, img.width(), img.height(), img.into_raw()).unwrap()
    }

    fn detect(f: &Frame) -> Detection {
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let mut yunet = YuNet::new(&rt, &verified_yunet(), YuNetConfig::default()).unwrap();
        let dets = yunet.detect(f).unwrap();
        assert_eq!(dets.len(), 1, "{dets:?}");
        dets[0]
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn liveness_score_in_unit_range() {
        let mut m = minifasnet(ExecutionProvider::Cpu);
        assert_eq!(m.tier(), LivenessTier::Rgb);
        for name in ["face-a-1", "face-a-2", "face-b-1"] {
            let f = load(name);
            let det = detect(&f);
            let s = m.score(&f, &det).unwrap();
            println!("{name}: live = {s:.4}");
            assert!((0.0..=1.0).contains(&s), "{name}: {s}");
        }
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn liveness_deterministic() {
        let mut m = minifasnet(ExecutionProvider::Cpu);
        let f = load("face-a-1");
        let det = detect(&f);
        let a = m.score(&f, &det).unwrap();
        let b = m.score(&f, &det).unwrap();
        assert_eq!(a.to_bits(), b.to_bits());
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn liveness_edge_face_no_panic() {
        let mut m = minifasnet(ExecutionProvider::Cpu);
        let f = load("face-a-1");
        let mut det = detect(&f);
        // box and landmarks pushed half past the left / top edges
        det.bbox.x = -det.bbox.w / 2.0;
        det.bbox.y = -det.bbox.h / 2.0;
        let s = m.score(&f, &det).unwrap();
        assert!((0.0..=1.0).contains(&s), "{s}");
        // past the right / bottom edges too
        det.bbox.x = 1.0 - det.bbox.w / 2.0;
        det.bbox.y = 1.0 - det.bbox.h / 2.0;
        let s = m.score(&f, &det).unwrap();
        assert!((0.0..=1.0).contains(&s), "{s}");
        // non-finite or degenerate boxes are input errors, never a score
        for bbox in [
            RectF {
                x: 0.1,
                y: 0.1,
                w: f32::INFINITY,
                h: 0.2,
            },
            RectF {
                x: f32::NAN,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            },
            RectF {
                x: 0.1,
                y: 0.1,
                w: 0.0,
                h: 0.2,
            },
        ] {
            let det = Detection { bbox, ..det };
            assert!(
                matches!(m.score(&f, &det), Err(VisionError::Input(_))),
                "{bbox:?}"
            );
        }
        let empty = Frame::new(0, 0, 0, 0, Vec::new()).unwrap();
        let det = Detection {
            bbox: RectF {
                x: 0.1,
                y: 0.1,
                w: 0.2,
                h: 0.2,
            },
            landmarks5: [PointF { x: 0.2, y: 0.2 }; 5],
            score: 0.9,
        };
        assert!(matches!(m.score(&empty, &det), Err(VisionError::Input(_))));
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn liveness_and_pose_ms() {
        use crate::PoseEstimator;
        use crate::attention::CoarsePose;
        let f = load("face-a-1");
        let det = detect(&f);
        for ep in [ExecutionProvider::Cpu, ExecutionProvider::Auto] {
            let mut m = minifasnet(ep);
            m.score(&f, &det).unwrap(); // warm-up
            let n = 50;
            let t = Instant::now();
            for _ in 0..n {
                m.score(&f, &det).unwrap();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
            println!("liveness (2 models) on {:?}: {ms:.2} ms", m.ep());
        }
        let t = Instant::now();
        for _ in 0..1000 {
            CoarsePose.estimate(&f, &det).unwrap();
        }
        println!(
            "coarse pose: {:.4} ms",
            t.elapsed().as_secs_f64() * 1000.0 / 1000.0
        );
    }
}
