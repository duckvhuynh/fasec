//! The examples' IR half: the IR camera paired with the RGB one, its lit pairs for
//! `Pipeline::process_with_ir`, and the plain-words reason IR is or is not in use.
//! No IR pixel leaves this module except into the pipeline.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use fasec_camera::{CameraError, IrCapture, LatestIrPair};
use fasec_config::{Config, IrMode};
use fasec_core::{CameraHealth, IrFrame, LivenessTier};
use fasec_vision::{IrEvidence, PipelineConfig, TrackIr};

/// Longer than the ASUS IR camera's lit-frame period (~128 ms, 24 lit frames in 3 s in Task 1),
/// so with healthy IR most calls carry a new lit pair with the RGB frame taken just after it.
/// Freshness does not depend on it: the pipeline counts lit frames and RGB time.
const LIT_WAIT: Duration = Duration::from_millis(200);

/// A lit IR frame and its ambient (unlit) partner.
pub type IrPair = (IrFrame, Option<IrFrame>);
/// What the next `process_with_ir` call gets: a lit pair (if any) and the IR health.
pub type IrOffer = (Option<IrPair>, CameraHealth);

const LOST: IrOffer = (None, CameraHealth::Lost);

/// An offer and when it was polled.
type Pending = Option<(IrOffer, Instant)>;

/// The pending offer, or a newly polled one that then stays pending until `delivered`. A held
/// lit pair older than the pipeline's `ir_max_age_ms` is polled afresh, since the pipeline would
/// count it as fresh from the call that takes it; an offer without a pair (the `Lost` of a
/// restart) is held however long.
fn offer(pending: &mut Pending, now: Instant, poll: impl FnOnce() -> IrOffer) -> IrOffer {
    let max_age = Duration::from_millis(PipelineConfig::default().ir_max_age_ms);
    if pending
        .as_ref()
        .is_some_and(|((pair, _), at)| pair.is_some() && now.duration_since(*at) > max_age)
    {
        *pending = None;
    }
    pending.get_or_insert_with(|| (poll(), now)).0.clone()
}

/// `[liveness] ir` from `%APPDATA%\fasec\config.toml` (SRS §6.2); no file means `prefer`.
pub fn ir_mode_from_config() -> Result<IrMode, String> {
    let Some(dir) = std::env::var_os("APPDATA") else {
        return Ok(IrMode::default());
    };
    let path = Path::new(&dir).join("fasec").join("config.toml");
    Config::load_or_default(&path)
        .map(|c| c.liveness.ir)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Why IR is or is not running.
#[derive(Clone, Debug, PartialEq)]
enum IrState {
    NotStarted,
    Off,
    NoPair,
    Failed(CameraError),
    Running,
}

/// What the UI shows about IR.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IrView {
    /// "IR: on (shared)", "IR: off in settings", ...
    pub status: String,
    /// IR capture running and healthy.
    pub on: bool,
    pub source: Option<String>,
    pub sharing: &'static str,
    pub health: Option<CameraHealth>,
    /// Lit pairs handed to the pipeline per second.
    pub fps: f32,
}

/// Owns the IR capture; lives on the worker thread next to the RGB `Capture`.
pub struct IrLink {
    mode: IrMode,
    /// One slot per app run: a new capture numbers its frames after the old one's.
    slot: Arc<LatestIrPair>,
    cap: Option<IrCapture>,
    source: Option<String>,
    state: IrState,
    /// The offer handed out until a pipeline call takes it. A (re)start sets it to `Lost`, so
    /// the pipeline drops IR trust from the previous run before it sees the new capture.
    pending: Pending,
    last_seq: u64,
    window: (Instant, u32),
    fps: f32,
}

impl IrLink {
    pub fn new(mode: IrMode) -> IrLink {
        IrLink {
            mode,
            slot: Arc::new(LatestIrPair::new()),
            cap: None,
            source: None,
            state: IrState::NotStarted,
            pending: Some((LOST, Instant::now())),
            last_seq: 0,
            window: (Instant::now(), 0),
            fps: 0.0,
        }
    }

    /// Stops any IR capture, then opens the IR camera paired with RGB camera `rgb_id`. Blocks
    /// up to ~5 s. Every failure only leaves IR off: RGB carries on.
    // shortcut: runs on the RGB worker, so a slow open delays RGB (5 s max); a stuck stop detaches
    pub fn start(&mut self, rgb_id: &str) {
        self.stop();
        if self.mode == IrMode::Off {
            self.state = IrState::Off;
            return;
        }
        self.state = match fasec_camera::paired_ir_for(rgb_id) {
            Ok(None) => IrState::NoPair,
            Err(e) => IrState::Failed(e),
            Ok(Some(src)) => {
                self.source = Some(src.name);
                match IrCapture::start(&src.id, Arc::clone(&self.slot)) {
                    Ok(c) => {
                        self.cap = Some(c);
                        IrState::Running
                    }
                    Err(e) => IrState::Failed(e),
                }
            }
        };
        if let IrState::Failed(e) = &self.state {
            eprintln!("IR not used: {e}");
        }
    }

    /// Stops the IR capture (bounded at ~2 s).
    pub fn stop(&mut self) {
        self.cap = None;
        let lost = Some((LOST, Instant::now()));
        (self.source, self.state, self.pending) = (None, IrState::NotStarted, lost);
        (self.window, self.fps) = ((Instant::now(), 0), 0.0);
    }

    /// The lit pair for the next `process_with_ir` call and the IR health to pass with it.
    /// The same offer comes back until `delivered`, so neither a taken pair nor the `Lost` of a
    /// restart is lost on an iteration without an RGB frame. With healthy IR a new poll waits up
    /// to `LIT_WAIT` for a new lit pair.
    pub fn next(&mut self) -> IrOffer {
        offer(&mut self.pending, Instant::now(), || {
            let Some(cap) = &self.cap else {
                return LOST;
            };
            let health = cap.health();
            if health != CameraHealth::Ok {
                return (None, health);
            }
            let pair = self.slot.take_newer(self.last_seq, LIT_WAIT);
            if let Some((lit, _)) = &pair {
                self.last_seq = lit.seq;
                self.window.1 += 1;
            }
            let secs = self.window.0.elapsed().as_secs_f32();
            if secs >= 1.0 {
                self.fps = self.window.1 as f32 / secs;
                self.window = (Instant::now(), 0);
            }
            (pair, CameraHealth::Ok)
        })
    }

    /// The offer from `next` reached a `process_with_ir` call: the next `next` polls afresh.
    pub fn delivered(&mut self) {
        self.pending = None;
    }

    pub fn view(&self) -> IrView {
        let health = self.cap.as_ref().map(IrCapture::health);
        let sharing = self.cap.as_ref().map_or("", IrCapture::sharing_mode);
        IrView {
            status: status(&self.state, health.as_ref(), sharing),
            on: health == Some(CameraHealth::Ok),
            source: self.source.clone(),
            sharing,
            fps: self.fps,
            health,
        }
    }
}

fn status(state: &IrState, health: Option<&CameraHealth>, sharing: &str) -> String {
    let busy = "IR: busy (Windows Hello?)";
    let denied = "IR: access denied (Windows privacy settings)";
    match (state, health) {
        (IrState::NotStarted, _) => "IR: not started".into(),
        (IrState::Off, _) => "IR: off in settings".into(),
        (IrState::NoPair, _) => "IR: no IR camera paired".into(),
        (IrState::Failed(CameraError::Busy), _) => busy.into(),
        (IrState::Failed(CameraError::PermissionDenied), _) => denied.into(),
        (IrState::Failed(e), _) => format!("IR: failed to start ({e})"),
        (IrState::Running, None | Some(CameraHealth::Ok)) if sharing == "SharedReadOnly" => {
            "IR: on (shared)".into()
        }
        (IrState::Running, None | Some(CameraHealth::Ok)) => "IR: on (exclusive)".into(),
        (IrState::Running, Some(CameraHealth::Lost)) => "IR: lost".into(),
        (IrState::Running, Some(CameraHealth::Covered)) => "IR: covered".into(),
        (IrState::Running, Some(CameraHealth::Busy)) => busy.into(),
        (IrState::Running, Some(CameraHealth::PermissionDenied)) => denied.into(),
        (IrState::Running, Some(CameraHealth::Error(e))) => format!("IR: error ({e})"),
    }
}

/// The report's liveness tier and why, in plain words.
pub fn tier_line(tier: LivenessTier, ir: &IrView) -> String {
    let why = match tier {
        LivenessTier::Ir => "IR and RGB see the same face".to_owned(),
        _ if ir.on => "IR not trusted yet: needs one face seen by both cameras".to_owned(),
        _ => ir.status.clone(),
    };
    format!("liveness tier: {tier:?} ({why})")
}

/// The IR verdict for one box; "no ambient" marks a verdict from the lit frame alone.
pub fn track_ir_label(ir: &TrackIr) -> String {
    let verdict = match ir.evidence {
        IrEvidence::Absent => return "IR -".into(),
        IrEvidence::Live { score } => format!("IR live {score:.2}"),
        IrEvidence::Spoof { score, reason } => format!("IR SPOOF {reason:?} {score:.2}"),
        IrEvidence::Suspect { score, reason } => format!("IR suspect {reason:?} {score:.2}"),
    };
    if ir.ambient {
        verdict
    } else {
        format!("{verdict} (no ambient)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fasec_core::TrackId;
    use fasec_vision::ir::IrSpoofReason;

    #[test]
    fn status_says_why_in_plain_words() {
        let ok = Some(&CameraHealth::Ok);
        let cases = [
            (IrState::Off, None, "", "IR: off in settings"),
            (IrState::NoPair, None, "", "IR: no IR camera paired"),
            (
                IrState::Failed(CameraError::Busy),
                None,
                "",
                "IR: busy (Windows Hello?)",
            ),
            (IrState::Running, ok, "SharedReadOnly", "IR: on (shared)"),
            (
                IrState::Running,
                ok,
                "ExclusiveControl",
                "IR: on (exclusive)",
            ),
            (
                IrState::Running,
                Some(&CameraHealth::Lost),
                "SharedReadOnly",
                "IR: lost",
            ),
            (
                IrState::Running,
                Some(&CameraHealth::Busy),
                "SharedReadOnly",
                "IR: busy (Windows Hello?)",
            ),
        ];
        for (state, health, sharing, want) in cases {
            assert_eq!(status(&state, health, sharing), want, "{state:?}");
        }
    }

    #[test]
    fn tier_line_names_the_reason() {
        let mut v = IrView {
            status: "IR: off in settings".into(),
            ..IrView::default()
        };
        assert_eq!(
            tier_line(LivenessTier::Rgb, &v),
            "liveness tier: Rgb (IR: off in settings)"
        );
        v.on = true;
        assert!(tier_line(LivenessTier::Rgb, &v).contains("not trusted yet"));
        assert!(tier_line(LivenessTier::Ir, &v).starts_with("liveness tier: Ir ("));
    }

    #[test]
    fn track_label_marks_blank_ambient() {
        let ir = |evidence, ambient| TrackIr {
            track: TrackId(1),
            evidence,
            tier: LivenessTier::Ir,
            ambient,
        };
        let live = IrEvidence::Live { score: 0.8 };
        assert_eq!(track_ir_label(&ir(live, true)), "IR live 0.80");
        assert_eq!(
            track_ir_label(&ir(live, false)),
            "IR live 0.80 (no ambient)"
        );
        assert_eq!(track_ir_label(&ir(IrEvidence::Absent, false)), "IR -");
        let spoof = IrEvidence::Spoof {
            score: 0.0,
            reason: IrSpoofReason::NoIrFace,
        };
        assert_eq!(track_ir_label(&ir(spoof, true)), "IR SPOOF NoIrFace 0.00");
        let suspect = IrEvidence::Suspect {
            score: 0.0,
            reason: IrSpoofReason::NoIrFace,
        };
        assert_eq!(
            track_ir_label(&ir(suspect, true)),
            "IR suspect NoIrFace 0.00"
        );
    }

    fn lit(seq: u64) -> IrFrame {
        IrFrame::new(seq, 0, 2, 2, true, vec![9; 4]).unwrap()
    }

    fn seq(o: &IrOffer) -> Option<u64> {
        o.0.as_ref().map(|(l, _)| l.seq)
    }

    #[test]
    fn restart_lost_and_a_taken_pair_wait_for_a_pipeline_call() {
        let now = Instant::now();
        let mut link = IrLink::new(IrMode::Prefer);
        link.stop(); // a restart
        // the RGB frame is missing on two iterations: Lost is still what the next call gets
        for _ in 0..2 {
            let o = offer(&mut link.pending, now, || {
                (Some((lit(1), None)), CameraHealth::Ok)
            });
            assert_eq!((seq(&o), o.1), (None, CameraHealth::Lost));
        }
        link.delivered();
        // a taken pair is offered again until a call takes it
        let o = offer(&mut link.pending, now, || {
            (Some((lit(2), None)), CameraHealth::Ok)
        });
        assert_eq!((seq(&o), o.1), (Some(2), CameraHealth::Ok));
        let o = offer(&mut link.pending, now, || (None, CameraHealth::Ok));
        assert_eq!(seq(&o), Some(2));
        link.delivered();
        let o = offer(&mut link.pending, now, || (None, CameraHealth::Ok));
        assert_eq!((seq(&o), o.1), (None, CameraHealth::Ok));
    }

    #[test]
    fn a_held_pair_older_than_the_ir_max_age_is_dropped_but_lost_is_kept() {
        let t0 = Instant::now();
        let max_age = Duration::from_millis(PipelineConfig::default().ir_max_age_ms);
        let mut link = IrLink::new(IrMode::Prefer);
        link.delivered();
        let new = |s| move || (Some((lit(s), None)), CameraHealth::Ok);
        assert_eq!(seq(&offer(&mut link.pending, t0, new(1))), Some(1));
        // still within the IR max age: the same pair
        assert_eq!(
            seq(&offer(&mut link.pending, t0 + max_age, new(2))),
            Some(1)
        );
        // older: the pipeline would take it as fresh, so a new poll replaces it
        let late = t0 + max_age + Duration::from_millis(1);
        assert_eq!(seq(&offer(&mut link.pending, late, new(3))), Some(3));
        // a restart's Lost is kept however long no RGB frame comes
        link.stop();
        let o = offer(&mut link.pending, late + 10 * max_age, new(4));
        assert_eq!((seq(&o), o.1), (None, CameraHealth::Lost));
    }

    #[test]
    fn a_link_without_ir_hands_out_lost_and_never_opens_a_camera() {
        let mut link = IrLink::new(IrMode::Off);
        link.start("any-rgb-id");
        assert_eq!(link.view().status, "IR: off in settings");
        assert!(matches!(link.next(), (None, CameraHealth::Lost)));
        assert!(link.cap.is_none());
    }
}
