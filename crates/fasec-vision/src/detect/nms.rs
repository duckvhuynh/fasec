//! Greedy non-maximum suppression.

use crate::Detection;

/// Keeps the highest-scoring detections, dropping any whose IoU with an already kept one is
/// above `iou`; returns at most `top_k`, best first. Non-finite scores are dropped.
///
/// Unlike OpenCV's `NMSBoxes`, `top_k` caps the output rather than the candidate list.
pub fn nms(mut dets: Vec<Detection>, iou: f32, top_k: usize) -> Vec<Detection> {
    dets.retain(|d| d.score.is_finite());
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for d in dets {
        if kept.len() == top_k {
            break;
        }
        if kept.iter().all(|k| k.bbox.iou(&d.bbox) <= iou) {
            kept.push(d);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use fasec_core::{PointF, RectF};

    use super::*;

    fn det(x: f32, y: f32, w: f32, score: f32) -> Detection {
        Detection {
            bbox: RectF { x, y, w, h: w },
            landmarks5: [PointF { x, y }; 5],
            score,
        }
    }

    #[test]
    fn nms_suppresses_overlaps() {
        let dets = vec![
            det(0.10, 0.10, 0.2, 0.7),
            det(0.11, 0.10, 0.2, 0.9), // overlaps the first, higher score
            det(0.60, 0.60, 0.2, 0.8), // disjoint
            det(0.12, 0.11, 0.2, f32::NAN),
        ];
        let kept = nms(dets, 0.3, 50);
        let scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
        assert_eq!(scores, [0.9, 0.8]);
    }

    #[test]
    fn nms_keeps_top_k() {
        let dets: Vec<Detection> = (0..10)
            .map(|i| det(i as f32 * 0.1, 0.0, 0.05, i as f32 / 10.0))
            .collect();
        let kept = nms(dets, 0.3, 3);
        let scores: Vec<f32> = kept.iter().map(|d| d.score).collect();
        assert_eq!(scores, [0.9, 0.8, 0.7]);
        assert!(nms(Vec::new(), 0.3, 3).is_empty());
    }
}
