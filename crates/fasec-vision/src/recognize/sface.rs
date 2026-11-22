//! SFace (`face_recognition_sface_2021dec`) face embedder.
//!
//! Pre-processing follows OpenCV's `FaceRecognizerSF` (`modules/objdetect/src/face_recognize.cpp`,
//! OpenCV 4.x): 5-point similarity alignment to the ArcFace 112×112 template, then RGB float 0..255
//! NCHW (`blobFromImage(..., swapRB=true)` on a BGR image) with no mean or scale.

use fasec_core::{Frame, PointF};
use fasec_models::VerifiedModel;
use ort::session::Session;
use ort::value::TensorRef;

use super::align::{ARCFACE_112, similarity_transform, warp_affine_rgb};
use crate::image_ops::to_nchw_f32;
use crate::runtime::StageSession;
use crate::{Detection, FaceEmbedder, Runtime, VisionError};

/// Cosine similarity above which two SFace embeddings are the same person (OpenCV zoo).
pub const SFACE_COSINE_THRESHOLD: f32 = 0.363;

const MODEL_ID: &str = "sface-2021dec";
const SIZE: u32 = 112;
const DIM: usize = 128;
const INPUT: &str = "data";
const OUTPUT: &str = "fc1";

/// Cosine similarity; 0 for zero-length, mismatched or all-zero vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n = norm(a) * norm(b);
    if n > 0.0 { dot / n } else { 0.0 }
}

pub struct SFace {
    session: StageSession,
}

impl SFace {
    pub fn new(rt: &Runtime, model: &VerifiedModel) -> Result<SFace, VisionError> {
        let session = rt.stage_session(model)?;
        let s: &Session = session.session();
        let bad = |why: String| VisionError::Model(format!("{}: {why}", model.id()));
        let input = s
            .inputs()
            .iter()
            .find(|i| i.name() == INPUT)
            .ok_or_else(|| bad(format!("missing input {INPUT}")))?;
        let n = SIZE as i64;
        match input.dtype().tensor_shape().map(|s| &s[..]) {
            Some(&[1, 3, h, w]) if (h, w) == (n, n) => {}
            other => return Err(bad(format!("unsupported input shape {other:?}"))),
        }
        if !s.outputs().iter().any(|o| o.name() == OUTPUT) {
            return Err(bad(format!("missing output {OUTPUT}")));
        }
        tracing::info!(ep = session.ep(), "SFace loaded");
        Ok(SFace { session })
    }

    /// EP this embedder's session runs on (`"directml"` or `"cpu"`); may differ from
    /// [`Runtime::active_ep`] after a first-run fallback.
    pub fn ep(&self) -> &'static str {
        self.session.ep()
    }
}

impl FaceEmbedder for SFace {
    /// L2-normalised 128-d embedding of the face aligned from `det.landmarks5`. Landmarks may lie
    /// outside the frame (cut faces); that part of the crop is black.
    fn embed(&mut self, frame: &Frame, det: &Detection) -> Result<Vec<f32>, VisionError> {
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        if frame.width == 0 || frame.height == 0 {
            return Err(VisionError::Input(format!("empty frame {fw}x{fh}")));
        }
        let src = det.landmarks5.map(|p| PointF {
            x: p.x * fw,
            y: p.y * fh,
        });
        let m = similarity_transform(&src, &ARCFACE_112);
        if !m.iter().all(|v| v.is_finite()) || m.fixed_view::<2, 2>(0, 0).determinant() == 0.0 {
            return Err(VisionError::Input(format!(
                "degenerate landmarks {:?}",
                det.landmarks5
            )));
        }
        let crop = warp_affine_rgb(frame, &m, SIZE, SIZE);
        let input = to_nchw_f32(&crop, SIZE, SIZE, false);
        let mut e = self.session.run(|session| {
            let outs = session.run(ort::inputs![INPUT => TensorRef::from_array_view(&input)?])?;
            let v = outs
                .get(OUTPUT)
                .ok_or_else(|| VisionError::Model(format!("SFace output {OUTPUT} missing")))?;
            Ok(v.try_extract_tensor::<f32>()?.1.to_vec())
        })?;
        let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt();
        if e.len() != DIM || !norm.is_normal() {
            return Err(VisionError::Model(format!(
                "SFace output: {} values, norm {norm}",
                e.len()
            )));
        }
        e.iter_mut().for_each(|v| *v /= norm);
        Ok(e)
    }

    fn model_id(&self) -> &str {
        MODEL_ID
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::Instant;

    use fasec_config::ExecutionProvider;
    use image::RgbImage;

    use super::*;
    use crate::runtime::tests::{verified, verified_yunet};
    use crate::{FaceDetector, YuNet, YuNetConfig};

    #[test]
    fn cosine_basic() {
        let a = [1.0, 2.0, 3.0];
        assert!((cosine(&a, &a) - 1.0).abs() < 1e-6);
        assert!((cosine(&a, &[2.0, 4.0, 6.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 5.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
        // zero vectors and length mismatches never match
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 0.0]), 0.0);
    }

    fn load(name: &str) -> Frame {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../.assets/{name}.jpg"));
        let img: RgbImage = image::open(p).unwrap().to_rgb8();
        Frame::new(0, 0, img.width(), img.height(), img.into_raw()).unwrap()
    }

    struct Models {
        det: YuNet,
        sface: SFace,
    }

    impl Models {
        fn new(ep: ExecutionProvider) -> Models {
            let rt = Runtime::new(ep).unwrap();
            Models {
                det: YuNet::new(&rt, &verified_yunet(), YuNetConfig::default()).unwrap(),
                sface: SFace::new(&rt, &verified("sface-2021dec")).unwrap(),
            }
        }

        /// Embedding of the single face in `frame`.
        fn embed(&mut self, frame: &Frame) -> Vec<f32> {
            let dets = self.det.detect(frame).unwrap();
            assert_eq!(dets.len(), 1, "{dets:?}");
            self.sface.embed(frame, &dets[0]).unwrap()
        }

        fn embed_asset(&mut self, name: &str) -> Vec<f32> {
            self.embed(&load(name))
        }
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn embedding_is_unit_norm() {
        let mut m = Models::new(ExecutionProvider::Cpu);
        assert_eq!(m.sface.model_id(), "sface-2021dec");
        let e = m.embed_asset("face-a-1");
        assert_eq!(e.len(), 128);
        let norm = e.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn same_person_above_threshold() {
        let mut m = Models::new(ExecutionProvider::Cpu);
        let (a1, a2) = (m.embed_asset("face-a-1"), m.embed_asset("face-a-2"));
        let c = cosine(&a1, &a2);
        println!("cosine a1-a2 = {c:.4}");
        assert!(c > SFACE_COSINE_THRESHOLD, "a1-a2 cosine {c}");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn different_person_below_threshold() {
        let mut m = Models::new(ExecutionProvider::Cpu);
        let a1 = m.embed_asset("face-a-1");
        let a2 = m.embed_asset("face-a-2");
        let b1 = m.embed_asset("face-b-1");
        let (c1, c2) = (cosine(&a1, &b1), cosine(&a2, &b1));
        println!("cosine a1-b1 = {c1:.4}, a2-b1 = {c2:.4}");
        assert!(c1 < SFACE_COSINE_THRESHOLD, "a1-b1 cosine {c1}");
        assert!(c2 < SFACE_COSINE_THRESHOLD, "a2-b1 cosine {c2}");
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn edge_face_embeds_without_panic() {
        let mut m = Models::new(ExecutionProvider::Cpu);
        let f = load("face-a-1");
        let d = m.det.detect(&f).unwrap()[0];
        // shift the image so the face's left half falls off the frame edge
        let shift = ((d.bbox.x + d.bbox.w / 2.0) * f.width as f32) as u32;
        let (w, h) = (f.width as usize, f.height as usize);
        let mut data = vec![0u8; w * h * 3];
        for y in 0..h {
            let row = &f.data[y * w * 3..(y + 1) * w * 3];
            data[y * w * 3..(y * w + w - shift as usize) * 3]
                .copy_from_slice(&row[shift as usize * 3..]);
        }
        let cut = Frame::new(0, 0, f.width, f.height, data).unwrap();
        let mut det = d;
        for p in &mut det.landmarks5 {
            p.x -= shift as f32 / f.width as f32;
        }
        assert!(det.landmarks5.iter().any(|p| p.x < 0.0), "{det:?}");
        let e = m.sface.embed(&cut, &det).unwrap();
        assert_eq!(e.len(), 128);
        assert!(e.iter().all(|v| v.is_finite()));
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn embed_ms_cpu_vs_directml() {
        let f = load("face-a-1");
        for ep in [ExecutionProvider::Cpu, ExecutionProvider::Auto] {
            let mut m = Models::new(ep);
            let d = m.det.detect(&f).unwrap()[0];
            m.sface.embed(&f, &d).unwrap(); // warm-up
            let n = 50;
            let t = Instant::now();
            for _ in 0..n {
                m.sface.embed(&f, &d).unwrap();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
            println!("embed on {}: {ms:.2} ms", m.sface.ep());
        }
    }
}
