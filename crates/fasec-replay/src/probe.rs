//! Every IR cue of every track, measured beside the pipeline from its public parts
//! (`ir_faces`, an own `IrAssociator`, `ir_cues`, `judge`). The pipeline reports only the fused
//! IR evidence; `ir_debug` shows these numbers and replay writes them for tuning (Task 7).
//!
//! It mirrors the pipeline's IR half: fed the same lit pairs and the same report tracks, its
//! associator learns the same mapping and its verdicts are the pipeline's per-frame verdicts
//! (before the spoof streak and staleness rules).

use std::path::Path;

use anyhow::Context;
use fasec_core::{IrFrame, RectF, TrackId, TrackObs};
use fasec_models::{Manifest, VerifiedModel};
use fasec_vision::ir::{
    IrAssociator, IrCues, IrLivenessConfig, IrVerdict, ambient_is_blank, ir_cues, ir_faces, judge,
};
use fasec_vision::{Detection, FaceDetector, Runtime, VisionError, YuNet, YuNetConfig};

/// What the lit frame's ambient partner gives the cues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ambient {
    /// No unlit frame came before the lit one.
    Missing,
    /// Near-black (the driver blanks unlit frames): no lit − ambient cue.
    Blank,
    Usable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackCues {
    pub track: TrackId,
    /// The track's RGB box in IR coordinates, by the mapping as it stood after this frame.
    pub mapped: RectF,
    /// The IR face the associator gave the track.
    pub matched: Option<Detection>,
    /// `None` when `ir_cues` measures nothing (untrusted mapping with no match, a box off the
    /// frame, ...): the verdict is then `Unknown`.
    pub cues: Option<IrCues>,
    pub verdict: IrVerdict,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeFrame {
    /// All IR faces in the lit frame.
    pub faces: Vec<Detection>,
    /// The mapping was trusted when the tracks were matched.
    pub trusted: bool,
    pub ambient: Ambient,
    /// One per track, in the order given.
    pub tracks: Vec<TrackCues>,
}

pub struct IrProbe {
    det: Box<dyn FaceDetector>,
    cfg: IrLivenessConfig,
    assoc: IrAssociator,
    /// Seq and size of the last lit frame measured.
    last: Option<(u64, (u32, u32))>,
}

impl IrProbe {
    pub fn new(det: Box<dyn FaceDetector>, cfg: IrLivenessConfig) -> IrProbe {
        IrProbe {
            det,
            cfg,
            assoc: IrAssociator::new(),
            last: None,
        }
    }

    /// With its own YuNet (the pipeline's detector and config), verified like the pipeline's.
    pub fn from_models(
        rt: &Runtime,
        models_dir: &Path,
        cfg: IrLivenessConfig,
    ) -> anyhow::Result<IrProbe> {
        let entry = Manifest::embedded()
            .get("yunet-2023mar")
            .context("yunet-2023mar not in manifest")?;
        let model = VerifiedModel::load(models_dir, entry).context("yunet-2023mar")?;
        let det = YuNet::new(rt, &model, YuNetConfig::default())?;
        Ok(IrProbe::new(Box::new(det), cfg))
    }

    /// Forgets the mapping. Call it whenever the pipeline got an IR health other than `Ok`.
    pub fn reset(&mut self) {
        (self.assoc, self.last) = (IrAssociator::new(), None);
    }

    /// Cues for `tracks` (the pipeline report's tracks for the same call) on the lit pair the
    /// pipeline was given. `Ok(None)` for a lit frame no newer than the last one, which the
    /// pipeline ignores too; a new frame size resets the mapping first, as in the pipeline.
    pub fn measure(
        &mut self,
        lit: &IrFrame,
        ambient: Option<&IrFrame>,
        tracks: &[TrackObs],
    ) -> Result<Option<ProbeFrame>, VisionError> {
        if self.last.is_some_and(|(seq, _)| lit.seq <= seq) {
            return Ok(None);
        }
        let size = (lit.width, lit.height);
        if self.last.is_some_and(|(_, s)| s != size) {
            self.reset();
        }
        let faces = ir_faces(&mut *self.det, lit)?;
        self.last = Some((lit.seq, size));
        let matches = self.assoc.associate(tracks, &faces);
        let tracks = tracks
            .iter()
            .zip(matches)
            .map(|(t, (track, matched))| {
                let cues = ir_cues(lit, ambient, &self.assoc, t.bbox, &faces, matched.as_ref());
                TrackCues {
                    track,
                    mapped: self.assoc.map(t.bbox),
                    matched,
                    cues,
                    verdict: cues.map_or(IrVerdict::Unknown, |c| judge(&c, &self.cfg)),
                }
            })
            .collect();
        let ambient = match ambient {
            None => Ambient::Missing,
            Some(a) if ambient_is_blank(a) => Ambient::Blank,
            Some(_) => Ambient::Usable,
        };
        Ok(Some(ProbeFrame {
            trusted: self.assoc.is_trusted(),
            faces,
            ambient,
            tracks,
        }))
    }
}

#[cfg(test)]
mod tests {
    use fasec_core::{IdentityState, LivenessState, PointF};
    use fasec_vision::ir::IrSpoofReason;

    use super::*;

    const W: u32 = 100;
    /// The face box both cameras report in the fixtures: pixels 30..70 on both axes.
    const FACE: RectF = RectF {
        x: 0.3,
        y: 0.3,
        w: 0.4,
        h: 0.4,
    };

    /// Reports `FACE` with YuNet-ordered landmarks on every frame, RGB or IR.
    struct OneFace;

    impl FaceDetector for OneFace {
        fn detect(&mut self, _: &fasec_core::Frame) -> Result<Vec<Detection>, VisionError> {
            Ok(vec![face()])
        }
    }

    fn face() -> Detection {
        let p = |x, y| PointF { x, y };
        Detection {
            bbox: FACE,
            // eyes at px (42, 45) and (58, 45), nose (50, 55), mouth corners (44, 65), (56, 65)
            landmarks5: [
                p(0.42, 0.45),
                p(0.58, 0.45),
                p(0.5, 0.55),
                p(0.44, 0.65),
                p(0.56, 0.65),
            ],
            score: 0.9,
        }
    }

    /// A lit frame of a face near the emitter: background 20, face 150, nose 210, no glints.
    /// Face mean 152.4, ring mean 20, nose ÷ cheeks 210 ÷ 150.
    fn lit_face(seq: u64, nose: u8) -> IrFrame {
        let data = (0..W * W)
            .map(|i| match (i % W, i / W) {
                (46..54, 51..59) => nose,
                (30..70, 30..70) => 150,
                _ => 20,
            })
            .collect();
        IrFrame::new(seq, seq * 64, W, W, true, data).unwrap()
    }

    fn unlit(seq: u64, level: u8) -> IrFrame {
        IrFrame::new(seq, seq * 64, W, W, false, vec![level; (W * W) as usize]).unwrap()
    }

    fn track() -> TrackObs {
        let d = face();
        TrackObs {
            id: TrackId(4),
            bbox: d.bbox,
            landmarks5: d.landmarks5,
            det_score: d.score,
            identity: IdentityState::Pending,
            liveness: LivenessState::Unknown,
            head_pose: None,
            face_width_px: 100.0,
            age_ms: 0,
        }
    }

    fn probe() -> IrProbe {
        IrProbe::new(Box::new(OneFace), IrLivenessConfig::default())
    }

    #[test]
    fn measures_every_cue_once_the_mapping_is_trusted() {
        let mut p = probe();
        let t = [track()];
        // the associator needs five consistent single-face pairs before it matches anything
        for seq in 1..5 {
            let f = p.measure(&lit_face(seq, 210), None, &t).unwrap().unwrap();
            assert!(!f.trusted && f.faces.len() == 1);
            assert_eq!(f.tracks[0].cues, None);
            assert_eq!(f.tracks[0].verdict, IrVerdict::Unknown);
        }
        let f = p
            .measure(&lit_face(5, 210), Some(&unlit(4, 10)), &t)
            .unwrap()
            .unwrap();
        assert!(f.trusted);
        assert_eq!(f.ambient, Ambient::Usable);
        let tc = &f.tracks[0];
        assert_eq!(tc.track, TrackId(4));
        assert_eq!(tc.matched, Some(face()));
        let c = tc.cues.unwrap();
        assert!(c.face_present && c.glints == 0, "{c:?}");
        let close = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert!(
            close(c.lit_face, 152.4) && close(c.lit_background, 20.0),
            "{c:?}"
        );
        assert!(close(c.lit_minus_ambient, 142.4), "{c:?}");
        assert!(close(c.ambient_background, 10.0), "{c:?}");
        assert!(close(c.face_vs_background, 142.4 / 10.0), "{c:?}");
        assert!(close(c.relief, 200.0 / 140.0), "{c:?}");
        assert_eq!(tc.verdict, IrVerdict::Live { score: 0.8 });

        // a flat face: the same numbers the pipeline would judge FlatReflectance on
        let f = p.measure(&lit_face(6, 150), None, &t).unwrap().unwrap();
        assert_eq!(f.ambient, Ambient::Missing);
        let c = f.tracks[0].cues.unwrap();
        assert!(c.lit_minus_ambient.is_nan() && c.ambient_background.is_nan());
        assert!((c.relief - 1.0).abs() < 1e-6, "{c:?}");
        assert!(matches!(
            f.tracks[0].verdict,
            IrVerdict::Spoof {
                reason: IrSpoofReason::FlatReflectance,
                ..
            }
        ));
        let f = p
            .measure(&lit_face(7, 210), Some(&unlit(6, 0)), &t)
            .unwrap()
            .unwrap();
        assert_eq!(f.ambient, Ambient::Blank);
        assert!(f.tracks[0].cues.unwrap().lit_minus_ambient.is_nan());
    }

    #[test]
    fn follows_the_pipeline_on_old_frames_resizes_and_resets() {
        let mut p = probe();
        let t = [track()];
        for seq in 1..=5 {
            p.measure(&lit_face(seq, 210), None, &t).unwrap();
        }
        // not newer: the pipeline ignores it, so the probe does too
        assert_eq!(p.measure(&lit_face(5, 210), None, &t).unwrap(), None);
        assert_eq!(p.measure(&lit_face(3, 210), None, &t).unwrap(), None);
        assert!(
            p.measure(&lit_face(6, 210), None, &t)
                .unwrap()
                .unwrap()
                .trusted
        );
        // a new frame size starts the mapping over
        let small = IrFrame::new(7, 0, 50, 50, true, vec![20; 2500]).unwrap();
        assert!(!p.measure(&small, None, &t).unwrap().unwrap().trusted);
        // so does an IR restart, and the seq may then start over
        p.reset();
        let f = p.measure(&lit_face(1, 210), None, &t).unwrap().unwrap();
        assert!(!f.trusted);
        // an unlit frame passed as lit is the detector's input error, not "no face"
        assert!(p.measure(&unlit(9, 10), None, &t).is_err());
    }
}
