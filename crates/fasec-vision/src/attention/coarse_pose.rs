//! Coarse head pose from the 5 detector landmarks alone (no model).
//!
//! M1 default; SP-4 may replace it with a model (6DRepNet / Face Landmarker) behind the same
//! [`PoseEstimator`] trait.

use fasec_core::{Frame, HeadPose};

use crate::{Detection, PoseEstimator, VisionError};

/// Degrees of rotation per degree of the landmark angle. shortcut: K = 1 / (nose depth /
/// projected span) with a generic nose depth of half the inter-eye / eye-mouth span, not fitted to
/// labelled data; upgrade with SP-4 (6DRepNet / Face Landmarker) if attention needs accuracy.
const K_YAW: f32 = 2.0;
const K_PITCH: f32 = 2.0;

/// Yaw/pitch/roll in degrees from [`Detection::landmarks5`]. Positive yaw = face turned to the
/// camera's right (nose towards image right), positive pitch = looking down, positive roll = eye
/// line rotated clockwise on screen. Yaw and pitch are measured after undoing the roll.
#[derive(Clone, Copy, Debug, Default)]
pub struct CoarsePose;

impl PoseEstimator for CoarsePose {
    fn estimate(&mut self, frame: &Frame, det: &Detection) -> Result<HeadPose, VisionError> {
        let (fw, fh) = (frame.width as f32, frame.height as f32);
        if frame.width == 0 || frame.height == 0 {
            return Err(VisionError::Input(format!("empty frame {fw}x{fh}")));
        }
        // Pixels, not normalised units, so angles are right on non-square frames.
        let px = det.landmarks5.map(|p| (p.x * fw, p.y * fh));
        if px.iter().any(|(x, y)| !x.is_finite() || !y.is_finite()) {
            return Err(VisionError::Input(format!(
                "non-finite landmarks {:?}",
                det.landmarks5
            )));
        }
        let [re, le, nose, rm, lm] = px;
        let mid = |a: (f32, f32), b: (f32, f32)| ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
        let eye = mid(re, le);
        let roll = (le.1 - re.1).atan2(le.0 - re.0);
        // rotate by -roll about the eye midpoint: eye line becomes horizontal
        let (s, c) = (-roll).sin_cos();
        let upright = |p: (f32, f32)| {
            let (dx, dy) = (p.0 - eye.0, p.1 - eye.1);
            (dx * c - dy * s, dx * s + dy * c)
        };
        let (re, le, nose, mouth) = (
            upright(re),
            upright(le),
            upright(nose),
            upright(mid(rm, lm)),
        );
        let inter_eye = (le.0 - re.0).hypot(le.1 - re.1);
        let eye_mouth = mouth.0.hypot(mouth.1); // eye midpoint is the origin now
        let yaw = nose.0.atan2(inter_eye).to_degrees() * K_YAW;
        let pitch = (nose.1 - mouth.1 / 2.0).atan2(eye_mouth).to_degrees() * K_PITCH;
        // finite but huge landmarks can still overflow the sums above
        if !(yaw.is_finite() && pitch.is_finite() && roll.is_finite()) {
            return Err(VisionError::Input(format!(
                "landmarks out of range {:?}",
                det.landmarks5
            )));
        }
        Ok(HeadPose {
            yaw: yaw.clamp(-90.0, 90.0),
            pitch: pitch.clamp(-90.0, 90.0),
            roll: roll.to_degrees(),
            confidence: det.score,
        })
    }
}

/// Face box width in frame pixels; 0 for a degenerate or non-finite box.
pub fn face_width_px(det: &Detection, frame: &Frame) -> f32 {
    let w = det.bbox.w * frame.width as f32;
    if w.is_finite() { w.max(0.0) } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use fasec_core::{PointF, RectF};

    use super::*;

    /// ArcFace 112×112 alignment template (insightface): a frontal face.
    const TEMPLATE: [(f32, f32); 5] = [
        (38.2946, 51.6963),
        (73.5318, 51.5014),
        (56.0252, 71.7366),
        (41.5493, 92.3655),
        (70.7299, 92.2041),
    ];

    /// 640×480 frame with the template scaled ×2 at (200, 100), edited by `f` (pixels).
    fn pose_of(f: impl Fn(&mut [(f32, f32); 5])) -> HeadPose {
        let frame = Frame::new(0, 0, 640, 480, vec![0; 640 * 480 * 3]).unwrap();
        let mut pts = TEMPLATE.map(|(x, y)| (200.0 + 2.0 * x, 100.0 + 2.0 * y));
        f(&mut pts);
        let det = Detection {
            bbox: RectF {
                x: 200.0 / 640.0,
                y: 100.0 / 480.0,
                w: 224.0 / 640.0,
                h: 224.0 / 480.0,
            },
            landmarks5: pts.map(|(x, y)| PointF {
                x: x / 640.0,
                y: y / 480.0,
            }),
            score: 0.9,
        };
        CoarsePose.estimate(&frame, &det).unwrap()
    }

    #[test]
    fn pose_frontal_synthetic_is_near_zero() {
        let p = pose_of(|_| {});
        assert!(
            p.yaw.abs() < 3.0 && p.pitch.abs() < 3.0 && p.roll.abs() < 3.0,
            "{p:?}"
        );
        assert_eq!(p.confidence, 0.9);
    }

    #[test]
    fn pose_nose_right_gives_positive_yaw() {
        let p = pose_of(|p| p[2].0 += 20.0);
        assert!(p.yaw > 10.0, "{p:?}");
        assert!(p.pitch.abs() < 3.0 && p.roll.abs() < 3.0, "{p:?}");
        let p = pose_of(|p| p[2].0 -= 20.0);
        assert!(p.yaw < -10.0, "{p:?}");
    }

    #[test]
    fn pose_nose_low_gives_positive_pitch() {
        let p = pose_of(|p| p[2].1 += 15.0);
        assert!(p.pitch > 10.0, "{p:?}");
        assert!(p.yaw.abs() < 3.0 && p.roll.abs() < 3.0, "{p:?}");
        let p = pose_of(|p| p[2].1 -= 15.0);
        assert!(p.pitch < -10.0, "{p:?}");
    }

    #[test]
    fn pose_tilted_eyes_gives_roll() {
        // whole face rotated 20° about the eye midpoint (clockwise on screen, y down)
        let p = pose_of(|p| {
            let (cx, cy) = ((p[0].0 + p[1].0) / 2.0, (p[0].1 + p[1].1) / 2.0);
            let (s, c) = 20f32.to_radians().sin_cos();
            for q in p.iter_mut() {
                let (dx, dy) = (q.0 - cx, q.1 - cy);
                *q = (cx + dx * c - dy * s, cy + dx * s + dy * c);
            }
        });
        assert!((p.roll - 20.0).abs() < 3.0, "{p:?}");
        // roll is removed before yaw/pitch, so they stay frontal
        assert!(p.yaw.abs() < 3.0 && p.pitch.abs() < 3.0, "{p:?}");
    }

    #[test]
    fn pose_edge_and_degenerate_inputs_are_finite_or_errors() {
        let frame = Frame::new(0, 0, 64, 48, vec![0; 64 * 48 * 3]).unwrap();
        let pt = |x, y| PointF { x, y };
        let mut det = Detection {
            bbox: RectF {
                x: -0.2,
                y: 0.5,
                w: 0.5,
                h: 0.7,
            },
            // landmarks past the frame edge, as YuNet gives for a face cut by the edge
            landmarks5: [
                pt(-0.1, 0.8),
                pt(0.2, 0.8),
                pt(0.05, 1.05),
                pt(-0.05, 1.2),
                pt(0.15, 1.2),
            ],
            score: 0.7,
        };
        let p = CoarsePose.estimate(&frame, &det).unwrap();
        assert!(
            [p.yaw, p.pitch, p.roll].iter().all(|v| v.is_finite()),
            "{p:?}"
        );
        // all landmarks on one point: finite, no NaN
        det.landmarks5 = [pt(0.5, 0.5); 5];
        let p = CoarsePose.estimate(&frame, &det).unwrap();
        assert!(
            [p.yaw, p.pitch, p.roll].iter().all(|v| v.is_finite()),
            "{p:?}"
        );
        // non-finite landmark or empty frame: input error
        det.landmarks5[2] = pt(f32::INFINITY, 0.5);
        assert!(matches!(
            CoarsePose.estimate(&frame, &det),
            Err(VisionError::Input(_))
        ));
        // finite but overflowing once scaled and summed
        det.landmarks5 = [pt(0.5, 0.5); 5];
        det.landmarks5[0] = pt(4.7e36, 0.5);
        det.landmarks5[1] = pt(4.7e36, 0.5);
        assert!(CoarsePose.estimate(&frame, &det).is_err());
        det.landmarks5 = [pt(0.5, 0.5); 5];
        let empty = Frame::new(0, 0, 0, 0, Vec::new()).unwrap();
        assert!(matches!(
            CoarsePose.estimate(&empty, &det),
            Err(VisionError::Input(_))
        ));
    }

    #[test]
    #[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]
    fn pose_on_face_images() {
        use std::path::Path;

        use fasec_config::ExecutionProvider;

        use crate::runtime::tests::verified_yunet;
        use crate::{FaceDetector, Runtime, YuNet, YuNetConfig};

        let rt = Runtime::new(ExecutionProvider::Cpu).unwrap();
        let mut yunet = YuNet::new(&rt, &verified_yunet(), YuNetConfig::default()).unwrap();
        for name in ["face-a-1", "face-a-2", "face-b-1"] {
            let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../../.assets/{name}.jpg"));
            let img = image::open(p).unwrap().to_rgb8();
            let frame = Frame::new(0, 0, img.width(), img.height(), img.into_raw()).unwrap();
            let dets = yunet.detect(&frame).unwrap();
            assert_eq!(dets.len(), 1, "{name}: {dets:?}");
            let pose = CoarsePose.estimate(&frame, &dets[0]).unwrap();
            let w = face_width_px(&dets[0], &frame);
            println!("{name}: {pose:?}, face width {w:.0} px");
            assert!(w > 0.0);
            assert!(
                [pose.yaw, pose.pitch, pose.roll]
                    .iter()
                    .all(|v| v.is_finite() && v.abs() <= 90.0),
                "{name}: {pose:?}"
            );
        }
    }

    #[test]
    fn face_width_px_scales_and_guards() {
        let frame = Frame::new(0, 0, 640, 480, vec![0; 640 * 480 * 3]).unwrap();
        let mut det = Detection {
            bbox: RectF {
                x: 0.1,
                y: 0.1,
                w: 0.25,
                h: 0.3,
            },
            landmarks5: [PointF { x: 0.0, y: 0.0 }; 5],
            score: 1.0,
        };
        assert_eq!(face_width_px(&det, &frame), 160.0);
        det.bbox.w = f32::INFINITY;
        assert_eq!(face_width_px(&det, &frame), 0.0);
        det.bbox.w = f32::NAN;
        assert_eq!(face_width_px(&det, &frame), 0.0);
        det.bbox.w = -0.1;
        assert_eq!(face_width_px(&det, &frame), 0.0);
    }
}
