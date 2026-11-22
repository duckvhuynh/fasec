//! Face detection on IR frames with the RGB detector.

use fasec_core::{Frame, IrFrame};

use crate::{Detection, FaceDetector, VisionError};

/// Faces in `frame` (normalised to it), found by the RGB detector on the grey image replicated
/// to three channels, so one YuNet instance and config serve both cameras.
///
/// An unlit frame is an `Input` error, not an empty list: no face in a lit frame is spoof
/// evidence, so an unlit frame must never read as one.
pub fn ir_faces(
    det: &mut dyn FaceDetector,
    frame: &IrFrame,
) -> Result<Vec<Detection>, VisionError> {
    if !frame.illuminated {
        return Err(VisionError::Input("IR frame is unlit".into()));
    }
    let rgb: Vec<u8> = frame.data.iter().flat_map(|&p| [p; 3]).collect();
    let rgb = Frame::new(frame.seq, frame.ts_ms, frame.width, frame.height, rgb)
        .map_err(|e| VisionError::Input(format!("IR frame: {e}")))?;
    det.detect(&rgb)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use fasec_config::ExecutionProvider;
    use fasec_core::{PointF, RectF};

    use super::*;
    use crate::runtime::tests::verified_yunet;
    use crate::{Runtime, YuNet, YuNetConfig};

    /// Records the frame it was given and reports one face.
    struct Spy(Option<Frame>);

    impl FaceDetector for Spy {
        fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>, VisionError> {
            self.0 = Some(frame.clone());
            Ok(vec![Detection {
                bbox: RectF {
                    x: 0.1,
                    y: 0.2,
                    w: 0.3,
                    h: 0.4,
                },
                landmarks5: [PointF { x: 0.2, y: 0.3 }; 5],
                score: 0.9,
            }])
        }
    }

    #[test]
    fn grey_is_replicated_to_rgb() {
        let ir = IrFrame::new(5, 77, 3, 2, true, vec![0, 10, 20, 30, 40, 255]).unwrap();
        let mut spy = Spy(None);
        let dets = ir_faces(&mut spy, &ir).unwrap();
        assert_eq!(dets.len(), 1);
        let f = spy.0.unwrap();
        assert_eq!((f.seq, f.ts_ms, f.width, f.height), (5, 77, 3, 2));
        let want: Vec<u8> = ir.data.iter().flat_map(|&p| [p; 3]).collect();
        assert_eq!(&f.data[..], &want[..]);
    }

    #[test]
    fn broken_ir_frame_is_an_input_error() {
        // `IrFrame` fields are public, so the length invariant can be broken after construction
        let mut ir = IrFrame::new(0, 0, 4, 2, true, vec![1; 8]).unwrap();
        ir.data = vec![1; 7].into();
        let mut spy = Spy(None);
        assert!(matches!(
            ir_faces(&mut spy, &ir),
            Err(VisionError::Input(_))
        ));
        assert!(spy.0.is_none());
    }

    #[test]
    fn unlit_frame_is_an_input_error() {
        let ir = IrFrame::new(0, 0, 2, 2, false, vec![9; 4]).unwrap();
        let mut spy = Spy(None);
        assert!(matches!(
            ir_faces(&mut spy, &ir),
            Err(VisionError::Input(_))
        ));
        assert!(spy.0.is_none());
    }

    #[test]
    #[ignore = "needs .assets and models: run `cargo run -p fasec-models --example fetch`"]
    fn yunet_finds_face_in_grey_image() {
        // stand-in for a lit IR frame until the recorder (Task 6) gives real pairs
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.assets/face-a-1.jpg");
        let grey = image::open(p).unwrap().to_luma8();
        let ir = IrFrame::new(0, 0, grey.width(), grey.height(), true, grey.into_raw()).unwrap();
        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let mut det = YuNet::new(&rt, &verified_yunet(), YuNetConfig::default()).unwrap();
        let dets = ir_faces(&mut det, &ir).unwrap();
        println!("grey face-a-1: {dets:?}");
        assert_eq!(dets.len(), 1, "{dets:?}");
        assert!(dets[0].score >= 0.8, "{dets:?}");
    }
}
