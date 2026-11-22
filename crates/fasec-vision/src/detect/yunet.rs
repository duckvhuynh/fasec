//! YuNet (`face_detection_yunet_2023mar`) face detector.
//!
//! Pre/post-processing follows OpenCV's `FaceDetectorYN` (`modules/objdetect/src/face_detect.cpp`,
//! OpenCV 4.x): BGR float 0..255 NCHW input with the image in the top-left and zero padding to
//! the right/bottom, then per-stride anchor decoding and NMS.

use fasec_core::{Frame, PointF, RectF};
use fasec_models::VerifiedModel;
use ndarray::{Array4, s};
use ort::session::Session;
use ort::value::TensorRef;

use super::nms::nms;
use crate::image_ops::{check_frame, resize_rgb, to_nchw_f32};
use crate::runtime::StageSession;
use crate::{Detection, FaceDetector, Runtime, VisionError};

const STRIDES: [usize; 3] = [8, 16, 32];
/// Output order: `cls`, `obj`, `bbox`, `kps`, each for strides 8, 16, 32.
const OUTPUTS: [&str; 12] = [
    "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32",
    "kps_8", "kps_16", "kps_32",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct YuNetConfig {
    /// The frame is scaled so its longer side is this many pixels (capped at the model input).
    pub input_long_side: u32,
    pub score_threshold: f32,
    pub nms_iou: f32,
    pub top_k: usize,
}

impl Default for YuNetConfig {
    fn default() -> Self {
        YuNetConfig {
            // The 2023mar input is fixed 640×640, so a smaller side only adds padding and loses range.
            input_long_side: 640,
            score_threshold: 0.6,
            nms_iou: 0.3,
            top_k: 50,
        }
    }
}

/// Boxes and landmarks are not clamped: a face cut by the frame edge can extend past 0..1, as in
/// OpenCV; consumers clamp or pad when cropping.
pub struct YuNet {
    session: StageSession,
    cfg: YuNetConfig,
    input_name: String,
    /// Model input width and height (the 2023mar export is fixed at 640×640).
    in_w: usize,
    in_h: usize,
}

impl YuNet {
    pub fn new(
        rt: &Runtime,
        model: &VerifiedModel,
        cfg: YuNetConfig,
    ) -> Result<YuNet, VisionError> {
        let session = rt.stage_session(model)?;
        let s: &Session = session.session();
        let bad = |why: &str| VisionError::Model(format!("{}: {why}", model.id()));
        let [input] = s.inputs() else {
            return Err(bad("expected exactly one input"));
        };
        // The 2023mar export has a static input shape, so ORT rejects any other size; frames are
        // scaled into it instead of padding to the next multiple of 32 as OpenCV does.
        let (in_w, in_h) = match input.dtype().tensor_shape().map(|s| &s[..]) {
            Some(&[1, 3, h, w]) if h > 0 && w > 0 && h % 32 == 0 && w % 32 == 0 => {
                (w as usize, h as usize)
            }
            other => return Err(bad(&format!("unsupported input shape {other:?}"))),
        };
        for name in OUTPUTS {
            if !s.outputs().iter().any(|o| o.name() == name) {
                return Err(bad(&format!("missing output {name}")));
            }
        }
        tracing::info!(ep = session.ep(), in_w, in_h, "YuNet loaded");
        Ok(YuNet {
            input_name: input.name().to_string(),
            session,
            cfg,
            in_w,
            in_h,
        })
    }

    /// EP this detector's session runs on (`"directml"` or `"cpu"`); may differ from
    /// [`Runtime::active_ep`] after a first-run fallback.
    pub fn ep(&self) -> &'static str {
        self.session.ep()
    }
}

impl FaceDetector for YuNet {
    fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>, VisionError> {
        check_frame(frame)?;
        let (fw, fh) = (frame.width, frame.height);
        let (in_w, in_h) = (self.in_w, self.in_h);
        let scale = (self.cfg.input_long_side.max(1) as f32 / fw.max(fh) as f32)
            .min(in_w as f32 / fw as f32)
            .min(in_h as f32 / fh as f32);
        let sw = ((fw as f32 * scale).round() as usize).clamp(1, in_w);
        let sh = ((fh as f32 * scale).round() as usize).clamp(1, in_h);
        let rgb = resize_rgb(frame, sw as u32, sh as u32);
        let mut input = Array4::<f32>::zeros((1, 3, in_h, in_w));
        input
            .slice_mut(s![.., .., ..sh, ..sw])
            .assign(&to_nchw_f32(&rgb, sw as u32, sh as u32, true));

        let (name, thr) = (self.input_name.as_str(), self.cfg.score_threshold);
        let dets = self.session.run(|session| {
            let outs = session.run(ort::inputs![name => TensorRef::from_array_view(&input)?])?;
            let mut raw: [&[f32]; 12] = [&[]; 12];
            for (slot, out) in raw.iter_mut().zip(OUTPUTS) {
                let v = outs
                    .get(out)
                    .ok_or_else(|| VisionError::Model(format!("YuNet output {out} missing")))?;
                *slot = v.try_extract_tensor::<f32>()?.1;
            }
            decode(&raw, in_w, in_h, sw as f32, sh as f32, thr)
        })?;
        Ok(nms(dets, self.cfg.nms_iou, self.cfg.top_k))
    }
}

/// Decodes the raw outputs (in [`OUTPUTS`] order) of an `in_w`×`in_h` input whose top-left
/// `sw`×`sh` pixels hold the frame, into detections normalised to that frame. Any non-finite
/// output value is a model fault and an error, never "no faces".
fn decode(
    outs: &[&[f32]; 12],
    in_w: usize,
    in_h: usize,
    sw: f32,
    sh: f32,
    score_threshold: f32,
) -> Result<Vec<Detection>, VisionError> {
    let mut dets = Vec::new();
    for (i, stride) in STRIDES.into_iter().enumerate() {
        let (cols, rows) = (in_w / stride, in_h / stride);
        let n = cols * rows;
        let [cls, obj, bbox, kps] = [0, 1, 2, 3].map(|k| outs[i + 3 * k]);
        if cls.len() != n || obj.len() != n || bbox.len() != n * 4 || kps.len() != n * 10 {
            return Err(VisionError::Model(format!(
                "YuNet stride {stride}: unexpected output lengths for {n} anchors"
            )));
        }
        if [cls, obj, bbox, kps]
            .iter()
            .any(|o| o.iter().any(|v| !v.is_finite()))
        {
            return Err(VisionError::Model(format!(
                "YuNet stride {stride}: non-finite output"
            )));
        }
        let st = stride as f32;
        for idx in 0..n {
            let score = (cls[idx].clamp(0.0, 1.0) * obj[idx].clamp(0.0, 1.0)).sqrt();
            if score < score_threshold {
                continue;
            }
            let (c, r) = ((idx % cols) as f32, (idx / cols) as f32);
            let b = &bbox[idx * 4..idx * 4 + 4];
            let (cx, cy) = ((c + b[0]) * st, (r + b[1]) * st);
            let (w, h) = (b[2].exp() * st, b[3].exp() * st);
            let k = &kps[idx * 10..idx * 10 + 10];
            dets.push(Detection {
                bbox: RectF {
                    x: (cx - w / 2.0) / sw,
                    y: (cy - h / 2.0) / sh,
                    w: w / sw,
                    h: h / sh,
                },
                landmarks5: std::array::from_fn(|j| PointF {
                    x: (c + k[2 * j]) * st / sw,
                    y: (r + k[2 * j + 1]) * st / sh,
                }),
                score,
            });
        }
    }
    Ok(dets)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Instant;

    use fasec_config::ExecutionProvider;
    use image::{RgbImage, imageops};

    use super::*;
    use crate::runtime::tests::verified_yunet;

    /// 64×64 input: 8×8, 4×4 and 2×2 anchors.
    fn zero_outputs() -> Vec<Vec<f32>> {
        let n = |st: usize| (64 / st) * (64 / st);
        let mut v = Vec::new();
        for per in [1, 1, 4, 10] {
            for st in STRIDES {
                v.push(vec![0.0; n(st) * per]);
            }
        }
        v
    }

    fn as_refs(v: &[Vec<f32>]) -> [&[f32]; 12] {
        std::array::from_fn(|i| v[i].as_slice())
    }

    #[test]
    fn decode_maps_anchor_to_frame() {
        let mut v = zero_outputs();
        // stride 16, row 1, col 2 → anchor index 6
        v[1][6] = 0.81; // cls_16
        v[4][6] = 0.81; // obj_16
        v[7][6 * 4..6 * 4 + 4].copy_from_slice(&[0.5, 0.25, 2f32.ln(), 0.0]);
        v[10][6 * 10..6 * 10 + 2].copy_from_slice(&[0.0, 0.5]); // first landmark
        // below-threshold anchors are dropped
        v[0][0] = 0.5;
        v[3][0] = 0.5;
        // the frame fills the left 32×64 of the input
        let dets = decode(&as_refs(&v), 64, 64, 32.0, 64.0, 0.6).unwrap();
        assert_eq!(dets.len(), 1);
        let d = dets[0];
        assert!((d.score - 0.81).abs() < 1e-6);
        // centre (2.5·16, 1.25·16) = (40, 20), size (32, 16) in input pixels
        let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
        assert!(close(d.bbox.x, (40.0 - 16.0) / 32.0), "{d:?}");
        assert!(close(d.bbox.y, (20.0 - 8.0) / 64.0), "{d:?}");
        assert!(close(d.bbox.w, 32.0 / 32.0), "{d:?}");
        assert!(close(d.bbox.h, 16.0 / 64.0), "{d:?}");
        assert!(close(d.landmarks5[0].x, 32.0 / 32.0), "{d:?}");
        assert!(close(d.landmarks5[0].y, 24.0 / 64.0), "{d:?}");
        assert!(close(d.landmarks5[1].x, 32.0 / 32.0), "{d:?}");
        assert!(close(d.landmarks5[1].y, 16.0 / 64.0), "{d:?}");
    }

    #[test]
    fn decode_rejects_non_finite_outputs() {
        // every output tensor, NaN and ±inf: a numeric fault is an error, not "no faces"
        for out in 0..12 {
            for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let mut v = zero_outputs();
                v[out][0] = bad;
                let r = decode(&as_refs(&v), 64, 64, 64.0, 64.0, 0.6);
                assert!(
                    matches!(r, Err(VisionError::Model(_))),
                    "output {out}, {bad}: {r:?}"
                );
            }
        }
    }

    #[test]
    fn decode_rejects_wrong_output_length() {
        let mut v = zero_outputs();
        v[8].pop();
        assert!(decode(&as_refs(&v), 64, 64, 64.0, 64.0, 0.6).is_err());
    }

    fn load(name: &str) -> RgbImage {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../.assets/{name}.jpg"));
        image::open(p).unwrap().to_rgb8()
    }

    fn frame(img: &RgbImage) -> Frame {
        Frame::new(0, 0, img.width(), img.height(), img.as_raw().clone()).unwrap()
    }

    /// `img` scaled to fit `w`×`h`, centred on black; returns it and the image's (x, y, scale).
    fn letterbox(img: &RgbImage, w: u32, h: u32) -> (RgbImage, [f32; 3]) {
        let k = (w as f32 / img.width() as f32).min(h as f32 / img.height() as f32);
        let (nw, nh) = (
            (img.width() as f32 * k) as u32,
            (img.height() as f32 * k) as u32,
        );
        let small = imageops::resize(img, nw, nh, imageops::FilterType::Triangle);
        let mut out = RgbImage::new(w, h);
        let (ox, oy) = ((w - nw) / 2, (h - nh) / 2);
        imageops::replace(&mut out, &small, i64::from(ox), i64::from(oy));
        (out, [ox as f32, oy as f32, k])
    }

    fn yunet(ep: ExecutionProvider) -> YuNet {
        let rt = Runtime::new(ep).unwrap();
        YuNet::new(&rt, &verified_yunet(), YuNetConfig::default()).unwrap()
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn detects_face_a1() {
        let mut det = yunet(ExecutionProvider::Cpu);
        let dets = det.detect(&frame(&load("face-a-1"))).unwrap();
        println!("face-a-1: {dets:?}");
        assert_eq!(dets.len(), 1, "{dets:?}");
        let d = dets[0];
        assert!(d.score >= 0.8, "{d:?}");
        let b = d.bbox;
        assert!(b.x >= 0.0 && b.y >= 0.0 && b.x + b.w <= 1.0 && b.y + b.h <= 1.0);
        assert!(b.w > 0.0 && b.h > 0.0);
        assert!(d.landmarks5[0].y < d.landmarks5[3].y, "{d:?}");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn detects_on_non_multiple_of_32_sizes() {
        let mut det = yunet(ExecutionProvider::Cpu);
        let img = load("face-a-1");
        let (iw, ih) = (img.width() as f32, img.height() as f32);
        let base = det.detect(&frame(&img)).unwrap();
        assert_eq!(base.len(), 1);
        let c = base[0].bbox.center();

        // 720×1280 is the rotated 1280×720 frame size
        for (w, h) in [(1280, 720), (640, 360), (720, 1280)] {
            let (lb, [ox, oy, k]) = letterbox(&img, w, h);
            let dets = det.detect(&frame(&lb)).unwrap();
            println!("{w}x{h}: {dets:?}");
            assert_eq!(dets.len(), 1, "{w}x{h}: {dets:?}");
            let got = dets[0].bbox.center();
            let want = (
                (ox + c.x * iw * k) / w as f32,
                (oy + c.y * ih * k) / h as f32,
            );
            assert!(
                (got.x - want.0).abs() < 0.05 && (got.y - want.1).abs() < 0.05,
                "{w}x{h}: centre {got:?}, want {want:?}"
            );
        }

        // image content rotated 90° clockwise: (x, y) → (1 - y, x)
        let rot = imageops::rotate90(&img);
        let dets = det.detect(&frame(&rot)).unwrap();
        println!("rotated: {dets:?}");
        assert_eq!(dets.len(), 1, "rotated: {dets:?}");
        let got = dets[0].bbox.center();
        assert!(
            (got.x - (1.0 - c.y)).abs() < 0.05 && (got.y - c.x).abs() < 0.05,
            "rotated: centre {got:?}, want ({}, {})",
            1.0 - c.y,
            c.x
        );
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn empty_on_black_frame() {
        let mut det = yunet(ExecutionProvider::Cpu);
        let black = Frame::new(0, 0, 640, 480, vec![0; 640 * 480 * 3]).unwrap();
        assert!(det.detect(&black).unwrap().is_empty());
        let empty = Frame::new(0, 0, 0, 0, Vec::new()).unwrap();
        assert!(matches!(det.detect(&empty), Err(VisionError::Input(_))));
        // `Frame` fields are public, so the length invariant can be broken after construction
        let mut short = black.clone();
        short.data = vec![0; 640 * 480 * 3 - 1].into();
        assert!(matches!(det.detect(&short), Err(VisionError::Input(_))));
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn detect_ms_cpu_vs_directml() {
        let (lb, _) = letterbox(&load("face-a-1"), 1280, 720);
        let f = frame(&lb);
        for ep in [ExecutionProvider::Cpu, ExecutionProvider::Auto] {
            let mut det = yunet(ep);
            assert_eq!(det.detect(&f).unwrap().len(), 1); // warm-up
            let n = 50;
            let t = Instant::now();
            for _ in 0..n {
                det.detect(&f).unwrap();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
            println!("detect 1280x720 on {}: {ms:.2} ms", det.ep());
        }
    }
}
