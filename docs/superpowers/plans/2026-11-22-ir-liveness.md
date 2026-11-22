# IR liveness (Windows Hello IR camera) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the machine has a Windows Hello IR camera, fasec decides liveness from IR (tier `Ir`) instead of RGB-only MiniFASNet, so the owner is recognised at normal laptop distance and photos/phone screens are rejected far more reliably.

**Why now:** at normal laptop distance the face is ~320 px tall in 720p, MiniFASNet cannot get its trained 2.7× context crop, liveness becomes `Unknown`, and (by ruling R10 of M2) the owner's `Known` is hidden. IR has no such context requirement and is a much stronger signal (SRS §7.1, FR-LV-006, ROADMAP M7).

**What SP-7 established (dev laptop, 2026-11-21):** the IR source (`ASUS IR camera`, same USB device as the RGB webcam, also exposed in the combined group `Rts-DMFT-Group`) can be read by a normal app with WinRT `MediaFrameReader` in `SharedReadOnly` mode; frames are `Gray8 640×360 @ 15 fps`; frames alternate **illuminated** (IR emitter on, mean ≈ 40–70) and **ambient** (mean ≈ 0), and `InfraredMediaFrame.IsIlluminated` reports which.

**Architecture:** `fasec-camera` gets an IR capture thread (WinRT frame reader) with a latest lit/ambient pair slot and RGB↔IR source pairing. `fasec-vision` detects faces in the IR frame (YuNet on grey), associates them with RGB tracks, and computes IR liveness cues; the pipeline fuses IR and RGB into `LivenessState` with tier `Ir`. An `ir_debug` example records local paired clips for a human-in-the-loop tuning task.

**Tech Stack:** existing workspace; `windows` (WinRT `Windows.Media.Capture.Frames`, already used by `ir_probe`); no new ML model in this plan.

**Spec:** `docs/SRS.md` §4.3 (FR-LV-001…006), §5.1, §7.1–7.4; `docs/ROADMAP.md` §M7; `docs/spikes/SP-7-ir.md`; **`docs/research/liveness-landscape.md`** (how Windows Hello, Face ID, Android and the PAD literature do it — read its Summary first).

**Design decisions taken from the research (binding):**
- Windows Hello is 2D NIR with alternating lit/unlit IR, and since April 2025 (CVE-2025-26644, "Red Bleed" NIR-emitting LCD replay) it also requires the RGB camera to see the face. So in fasec **IR alone never yields `Live`**: `Live` (tier `Ir`) needs IR cues live **and** a geometrically consistent RGB face whose RGB verdict is not `Spoof`. An IR `Spoof` overrides any RGB `Live`. *Changed in Task 7 (owner decision D1, see SP-7):* IR `Live` now also overrides an RGB `Spoof`, because MiniFASNet called the real owner `Spoof` on most recorded frames; the RGB face is still required, and with IR absent the RGB verdict decides.
- No depth / 3D now (Hello and Android Class 3 phones reach their bar without it; Windows laptops lack depth sensors). Depth stays a P2 plugin via `MediaFrameSourceKind::Depth`.
- ESS is on for the dev laptop yet SharedReadOnly IR reads work while Hello is idle; behaviour during an active Hello sign-in is unknown → an IR stall is `Lost` and the RGB tier takes over.
- No permissively licensed RGB+IR anti-spoof model exists (all trained on research-only datasets) → hand-built cues now; a small model trained on the owner's opt-in recordings is a later option.
- The unlit frames on the dev laptop have mean 0.0 (possibly driver blanking) → the lit−unlit cue must detect a blanked unlit frame and switch itself off rather than treat it as evidence.
- Claims are stated in ISO/IEC 30107-3 terms (APCER per attack species, BPCER, attempts); the RGB-only tier is described as a deterrent.

## Global Constraints

- Same toolchain/lint/dependency rules as M2 (MSRV 1.92, `cargo add`, fmt + clippy `-D warnings` + tests before every commit, conventional commits, no attribution lines).
- `windows`/WinRT code only in `fasec-camera` (and examples). IR liveness *logic* lives in platform-neutral `fasec-vision` and takes plain `IrFrame` data, so it is unit-testable on any OS.
- Open the IR source in `SharedReadOnly` first; fall back to `ExclusiveControl` only if shared fails, and log it. fasec must not stop Windows Hello sign-in from working (human check).
- IR frames are biometric-adjacent: never written to disk except by the explicit, opt-in recorder of Task 6, which writes only under `%LOCALAPPDATA%\fasec\recordings\` and is never committed. No IR pixels in logs.
- Fail safe: missing/unpaired IR, IR errors, or IR stalls → fall back to the RGB tier (never report `Live` because IR is missing). IR may only *add* evidence: an IR spoof verdict wins over an RGB live verdict.
- Thresholds for IR cues are constants with a comment naming the measurement they came from (Task 7); until then they are conservative placeholders and the tier is reported honestly.

## Review Focus

1. **Phone/monitor replay of the owner** → RGB sees a face, IR sees none (screens don't reflect the 850 nm emitter like skin) → `Spoof`, even if MiniFASNet said Live. **NIR-emitting screen (Red Bleed style)** → IR shows a face but it is equally visible in the unlit frame / lacks NIR shading, or the RGB view shows no matching face → never `Live`.
6. **IR face with no matching RGB face** → never `Live` (IR alone is not enough).
2. **Real face close to the camera (~320–400 px tall in RGB)** → IR verdict `Live`, owner shown as `Known` (the bug that motivated this plan).
3. **IR camera present but stalled / unplugged mid-run / opened by Windows Hello exclusively** → IR health ≠ Ok → RGB tier used, no panic, no `Live` from stale IR.
4. **Two people, one IR face** → association never transfers one person's IR evidence to the other's RGB track.
5. **Strong sunlight (ambient IR high)** → cues use lit − ambient difference, not raw lit brightness; no false `Live` for a photo in sunlight.

---

## Tasks

### Task 0 (resolved by measurement): unlit frames are blanked on the dev laptop
SP-7 measured mean 0.0 on every unlit Gray8 frame; u8 pixels are non-negative, so every unlit pixel is 0 — the driver blanks them. Task 3's lit−unlit cue must detect an all-zero unlit frame at runtime and switch itself off (other devices may deliver real ambient frames), so on this laptop the decision rests on the lit-frame cues plus RGB↔IR consistency.

*Correction (2026-11-22, Tasks 1 and 7):* the claim above that the unlit frames are all-zero is wrong. They are near-black, not all-zero: mean ≈ 0.00 with a maximum pixel of 1–4 (Task 1). "Mean 0.0" was rounded to one decimal. The blank check is `ambient_is_blank` (mean < 1 and 99th percentile ≤ 8), not `IrFrame::is_blank`.

### Task 1: IR capture in `fasec-camera`

**Files:** Create `crates/fasec-camera/src/ir.rs`; modify `lib.rs`, `device.rs` (pairing), `Cargo.toml` (WinRT features as in `ir_probe`).

**Interfaces:**
- `fasec_core::IrFrame { pub seq: u64, pub ts_ms: u64, pub width: u32, pub height: u32, pub illuminated: bool, pub data: Arc<[u8]> /* Gray8 */ }` (add to `fasec-core`, `Frame`-like validation, no serde on pixels).
- `pub struct IrSourceInfo { pub id: String, pub name: String, pub group_id: String, pub device_id: String }`; `pub fn ir_sources() -> Result<Vec<IrSourceInfo>, CameraError>`; `pub fn paired_ir_for(rgb_id: &str) -> Result<Option<IrSourceInfo>, CameraError>` — same MediaFrameSourceGroup as the RGB source first, else same USB `VID_xxxx&PID_xxxx` prefix (case-insensitive, via `normalize_id`), else None.
- `pub struct IrCapture` with `IrCapture::start(source_id: &str, slot: Arc<LatestIrPair>) -> Result<IrCapture, CameraError>` (own thread; `.get()`/`.join()` on WinRT async ops on that thread; SharedReadOnly then ExclusiveControl), `stop(self)` bounded like `Capture` (≤ 2 s, then detach), `health() -> CameraHealth` (no frame for 1500 ms → Lost), `sharing_mode() -> &'static str`.
- `pub struct LatestIrPair` — keeps the latest illuminated frame and the latest ambient frame; `fn take_newer(after_seq, timeout) -> Option<(IrFrame, Option<IrFrame>)>` returning the newest lit frame and the ambient frame closest before it (or None).
- Tests: pairing logic on synthetic ids (pure function), LatestIrPair semantics (newest lit, matching ambient, never the same lit twice), bounded stop with a fake source; `#[ignore = "needs IR camera"]` live test: 30 frames, ~half illuminated.
- Commit `feat(camera): IR capture with lit/ambient pairing`.

### Task 2: IR face detection and RGB↔IR association (`fasec-vision`)

**Files:** Create `crates/fasec-vision/src/ir/{mod.rs,detect.rs,assoc.rs}`.

**Interfaces:**
- `pub fn ir_faces(det: &mut YuNet, lit: &IrFrame) -> Result<Vec<Detection>, VisionError>` — grey → 3-channel, same YuNet instance/config as RGB.
- `pub struct IrAssociator` — maps an RGB track's normalised box into IR normalised coords with an affine offset (scale + translation) estimated online from confident 1-face/1-face pairs (exponential average, clamped; starts at identity); `fn associate(&mut self, rgb: &[TrackObs], ir: &[Detection]) -> Vec<(TrackId, Option<Detection>)>` by IoU after mapping (threshold 0.3), one-to-one (Hungarian or greedy by IoU), never assigning one IR face to two tracks (Review Focus 4).
- Tests: synthetic boxes with a known offset converge; two faces cross-assign correctly; one IR face + two RGB tracks → only one gets it; `#[ignore]` test on a recorded pair if available (Task 6 output), else skipped.
- Commit `feat(vision): IR face detection and RGB↔IR association`.

### Task 3: IR liveness cues (`fasec-vision::liveness::ir`)

**Files:** Create `crates/fasec-vision/src/liveness/ir.rs`.

**Interfaces:**
- `pub struct IrCues { pub face_present: bool, pub lit_minus_ambient: f32, pub face_vs_background: f32, pub relief: f32, pub glints: u8 }` computed from the lit frame, the ambient frame (if any) and the associated IR detection: `lit_minus_ambient` = mean of (lit − ambient) in the face box; `face_vs_background` = that mean ÷ the same over a ring around the box (a real face near the emitter is much brighter than the background); `relief` = nose-region mean ÷ cheek-region mean from the 5 landmarks (3-D shading under a near-field emitter; flat paper is more uniform); `glints` = number of small saturated blobs (> 240) within a radius of each eye landmark in the lit frame that are absent in the ambient frame (corneal reflections of the emitter).
- `pub enum IrVerdict { Live { score: f32 }, Spoof { score: f32, reason: IrSpoofReason }, Unknown }`, `IrSpoofReason { NoIrFace, FlatReflectance, NoGlint }`; `pub fn judge(cues: &IrCues, cfg: &IrLivenessConfig) -> IrVerdict` with placeholder thresholds in `IrLivenessConfig::default()` (documented as placeholders until Task 7). Rules: RGB face with no associated IR face for `no_face_frames` (default 3) consecutive lit frames → `Spoof(NoIrFace)` (Review Focus 1); otherwise Live requires `face_vs_background ≥ t1` and `relief ≥ t2`, with glints as a bonus; ambient missing → use lit only but never decide `Live` on raw brightness alone when the ambient frame mean is high (Review Focus 5).
- Tests: synthetic IR images (face blob with bright nose + glints vs uniform flat patch vs empty region) produce Live / Spoof(FlatReflectance) / Spoof(NoIrFace); NaN/empty boxes → Unknown; sunlight case (high ambient) never Live from brightness alone.
- Commit `feat(vision): IR liveness cues and verdict`.

### Task 4: Pipeline fusion and tier `Ir`

**Files:** Modify `crates/fasec-vision/src/pipeline.rs`, `crates/fasec-core/src/vision.rs` (if a field is needed), `crates/fasec-config` (`ir: auto | off`).

**Interfaces:**
- `Pipeline::process_with_ir(&mut self, frame: &Frame, ir: Option<(&IrFrame, Option<&IrFrame>)>, health: CameraHealth, ir_health: CameraHealth) -> Result<ProcessOutput, VisionError>`; `process` stays and passes `None`.
- Fusion per track (encode as a small table/function, not scattered ifs): IR `Spoof` → `Spoof` (wins over RGB). IR `Live` **and** the track is associated to an IR face with consistent geometry **and** RGB verdict ≠ `Spoof` → `Live` tier `Ir` — RGB `Unknown` (e.g. too close for MiniFASNet) is acceptable here (Review Focus 2). IR `Live` but RGB `Spoof` → `Spoof`. IR `Unknown` → RGB result (tier `Rgb`). IR unhealthy/absent → RGB as today (Review Focus 3). `VisionReport.liveness_tier` reports `Ir` only while IR is healthy and associated.
- Config: `[liveness] ir = "prefer" | "off"` (default `prefer`); `prefer` uses the paired IR camera when present and healthy, else RGB; the UI always shows which tier is active and why (no IR found / IR busy / IR off).
- IR scheduling: IR cues evaluated on every new lit frame for tracks due for liveness (same interval as RGB liveness, plus immediately for new tracks).
- Tests with mock IR inputs covering each fusion row and the stale/unhealthy cases.
- Commit `feat(vision): fuse IR and RGB liveness (tier Ir)`.

### Task 5: Wire IR into `live_debug` and `enroll`

**Files:** Modify `crates/fasec-ui/examples/{live_debug.rs,enroll.rs}`.

- On camera start, `paired_ir_for(rgb_id)`; if found and config `ir != off`, start `IrCapture` on the worker thread; pass the IR pair to `process_with_ir`; stop it with the RGB capture. Side panel: IR source, sharing mode, IR health, IR fps, liveness tier, and per-track IR cues + verdict in the box label. `enroll`: the TooClose gate applies only when the tier is not `Ir` (IR has no context-crop limit).
- Commit `feat(ui): use the paired IR camera for liveness`.

### Task 6: `ir_debug` example + local recorder + replay support

**Files:** Create `crates/fasec-ui/examples/ir_debug.rs`; modify `crates/fasec-replay` (`--ir-input <dir>`).

- `ir_debug`: side-by-side RGB + IR (lit, and lit − ambient), detected boxes, association, all `IrCues` numbers and the verdict, live; a **Record** toggle writes paired frames (RGB PNG + IR lit/ambient PNG + a JSON line of timestamps) to `%LOCALAPPDATA%\fasec\recordings\<label>-<time>\` with a label chosen in the UI (`real-30cm`, `real-60cm`, `phone-replay`, `printed-photo`, …). The window states clearly that recordings stay on this PC and can be deleted.
- `fasec-replay --ir-input <dir>` replays a recorded IR stream alongside the RGB frames so tuning is deterministic.
- Commit `feat(ui,replay): ir_debug viewer, local paired recorder, IR replay`.

### Task 7 (human-in-the-loop): Measure and tune

- The owner records with `ir_debug`: real face at ~30, 50, 80 cm, turning, glasses on/off if relevant; phone showing the owner's photo and a video; a printed photo if a printer is available; a monitor showing the owner.
- The agent replays the recordings (`fasec-replay --ir-input`), tabulates the cue distributions per label, sets `IrLivenessConfig::default()` thresholds with margins, records the table in `docs/spikes/SP-7-ir.md` (decision section) and in code comments, and adds an `#[ignore = "needs local recordings"]` test that runs the labelled set from `%LOCALAPPDATA%\fasec\recordings` when present.
- Exit criteria: phone/monitor replay → `Spoof` on ≥ 99 % of frames with a face; real face 30–100 cm → `Live` on ≥ 95 % of frames; owner shown `Known` in `live_debug` at normal distance.
- Commit `tune(vision): IR liveness thresholds from recorded set`.

### Task 8: Docs and roadmap

- README (IR requirements and behaviour), SRS FR-LV-006 note on the chosen cues, ROADMAP M7 ticks for the IR items (blink/motion/active challenge stay open), SP-7 decision section.
- Human check: Windows Hello sign-in still works while fasec has the IR source open (SP-7 checks 2–3).
- Commit `docs: IR liveness`.

## Exit criteria

Review Focus 1–5 tested; Task 7 numbers met on the owner's recordings; owner recognised at normal laptop distance; Windows Hello unaffected.
