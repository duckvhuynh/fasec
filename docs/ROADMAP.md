# fasec: Roadmap

| | |
|---|---|
| Status | Draft v0.1 |
| Date | 2026-11-21 |
| Related | [PRD](PRD.md) · [SRS](SRS.md) · [MVP](MVP.md) |

Estimates assume **one developer working full time**. Halve the pace for part time.
They are sizing guesses to plan with, not commitments. Each milestone ends with
**exit criteria**; don't start the next milestone until they are met or explicitly waived.

## Overview

| Release | Milestone | Theme | Est. |
|---|---|---|---|
| — | **M0** | Spikes and project skeleton | 2 wk |
| — | **M1** | Camera + vision core + replay harness | 3 wk |
| — | **M2** | Identity, enrolment, secure storage | 2 wk |
| — | **M3** | Policy engine | 1.5 wk |
| — | **M4** | Overlays + action engine | 2.5 wk |
| — | **M5** | App shell, UI, wizard | 3 wk |
| **v0.1.0** | **M6** | MVP hardening, dogfooding, release | 2 wk |
| v0.2.0 | M7 | Strong liveness (temporal, IR, challenge, Hello unlock polish) | 4 wk |
| v0.3.0 | M8 | Attention+ and targeted actions (gaze, per-window/monitor, auto profiles, i18n) | 5 wk |
| v0.4.0 | M9 | Selective content masking: UIA + OCR | 6–8 wk |
| v0.5.0 | M10 | Browser extension | 4 wk |
| v0.6.0 | M11 | Depth camera plug-in | 3 wk |
| v1.0.0 | M12 | Hardening: `uiAccess`, signing, installer, updates, threat model | 4 wk |

MVP ≈ **16 weeks**. v1.0 ≈ **9–10 months** total.

```
Weeks  0    2    4    6    8    10   12   14   16
M0     ████
M1          ██████
M2                ████
M3                    ███
M4                       █████
M5                            ██████
M6                                  ████  → v0.1.0
```
M2 and M3 can overlap with the end of M1. M4 depends only on M0's SP-1/SP-2 and can
start in parallel with M2 if there are two contributors.

---

## M0: Spikes and skeleton (2 weeks)

**Goal:** retire the technical risks that could change the architecture, and set up the repo.

### Tasks
- [x] Repo: Cargo workspace with empty crates from [SRS §3.1](SRS.md#31-workspace-layout), MIT/Apache-2.0 licence files, `CONTRIBUTING.md`, issue templates.
- [x] CI (GitHub Actions): `fmt`, `clippy -D warnings`, `test` on `windows-latest` + `ubuntu-latest` (neutral crates), `cargo deny`. Also builds all examples and checks the MSRV. **MSRV ruling: 1.92** (raised from 1.85 so current `ort` 2.0 RCs and `sha2` 0.11 can be used).
- [x] `models/manifest.toml` with YuNet, SFace, MiniFASNet ×2, candidate pose models (URLs + SHA-256 + licence). Pose candidates are listed with empty URL/hash (not fetchable) until SP-4.
- [ ] **SP-1** Overlay blur prototype: top-most, non-activating, capture-excluded window per monitor with DWM backdrop. Test active/inactive, transparency off, Win10 vs Win11, RDP.
- [ ] **SP-1b** Capture-based blur prototype: WGC monitor capture → D3D11 downscale/blur → present in overlay (overlay excluded from capture, so no feedback loop). Measure GPU/CPU cost.
- [ ] **SP-2** Z-order test matrix: Start, Alt-Tab, Task View, Action Center, toasts, Task Manager on top, full-screen games/video, UAC prompt.
- [x] **SP-3** `ort` + DirectML: run YuNet, SFace, MiniFASNet on reference laptop; measure ms/frame CPU vs DirectML. Done as `cargo bench -p fasec-vision` (dev laptop, not yet the reference laptop): 720p, 1 face, YuNet@320, a frame with recognition + liveness due ≈ **15 ms DirectML / 30 ms CPU**; detect + pose only ≈ 10 / 12 ms. DirectML is the default (`ep = auto`). Since then the YuNet default is 640 (model input is fixed 640×640, so 320 only added padding); these numbers predate that.
- [ ] **SP-4** Head pose: compare 6DRepNet vs Face Landmarker + PnP vs 5-point coarse pose on a short labelled clip. *Not started:* M1 uses the 5-point coarse pose (uncalibrated, +4..13° pitch bias on test faces) behind the `PoseEstimator` trait.
- [ ] **SP-5** Camera sharing with Teams/Zoom/Camera app on 2–3 cameras.
- [ ] **SP-6** `nokhwa`: enumeration (stable IDs), format negotiation, hot-plug behaviour on built-in + 2 USB cameras.
- [ ] **SP-7** IR frame access on any available Windows Hello laptop (feeds v0.2).
- [ ] Write `docs/spikes/SP-x.md` for each: result, numbers, decision. Docs and programs exist for SP-1, 1b, 2, 5, 6, 7 with the agent-measured parts filled in ([index](spikes/README.md)). The rows that need a person at the screen or physical cameras (SP-1, SP-2, SP-5 with Teams/Zoom, SP-6 replug, SP-7 with Hello) and the decisions are still open, so SP-1/1b/2/5/6/7 stay unticked.

### Exit criteria
- Decisions recorded for: blur backend order, pose model, camera library (nokhwa vs direct MF), DirectML default.
- Measured inference budget fits the latency target (or a plan to make it fit).
- CI green on an empty workspace.

---

## M1: Camera + vision core + replay (3 weeks)

**Goal:** frames in → `VisionReport` out, testable offline.

### `fasec-camera`
- [x] `CameraInfo { id, name, formats }`, `list()`, `open(id, format)`.
- [x] Capture thread writing into a latest-frame slot (`ArcSwap`/mutex + seq number).
- [x] DeviceWatcher → `CameraEvent::{Added, Removed}`.
- [x] Health: frame timeout, black-frame detector.
- [x] Rotation/mirror applied at capture.
- [x] Privacy-permission-denied detection (mapped from the privacy setting and `0x80070005`; not reproduced on hardware).

### `fasec-vision`
- [x] `runtime`: session builder, EP selection (`auto|directml|cpu`), warm-up, input tensor helpers. No explicit warm-up call yet: the first `run` of each session pays it.
- [x] Traits: `FaceDetector`, `FaceEmbedder`, `LivenessModel`, `PoseEstimator`.
- [x] YuNet detector (+ NMS, score threshold, 5 landmarks).
- [x] SORT tracker (Kalman on box, IoU + Hungarian, `track_ttl_ms`).
- [x] 5-point similarity alignment to 112×112 (`nalgebra` Umeyama).
- [x] SFace embedder + L2 normalisation.
- [x] MiniFASNet liveness (two crops, softmax fusion).
- [x] Pose estimator: 5-point coarse pose for now (SP-4 not done, see M0).
- [x] Scheduler (per-track recognition/liveness intervals, idle FPS).
- [x] `VisionReport` assembly.
- [x] Criterion benches per stage.

### `fasec-replay`
- [x] Read MP4 / image folder (via `ffmpeg` CLI or `image` sequence; keep it simple).
- [x] Simulated clock driven by frame timestamps.
- [ ] JSONL per-frame output, optional annotated PNG/MP4 dump. *JSONL done* (`cargo run -p fasec-replay --release -- run --input <dir|video>`); the annotated dump is not.

### Exit criteria
- Live debug window (temporary, can be a tiny egui app) shows boxes, track IDs, pose, liveness at ≥ 15 FPS on reference laptop.
- `fasec-replay` produces stable output for the same input (deterministic).
- Per-stage latencies documented; total vision latency per frame ≤ 60 ms p95 with DirectML.

**Status (2026-11-22):** replay output is byte-identical across runs (CPU, `deterministic_output` test). Per-stage medians are in the SP-3 line above (benches give medians, not p95); worst case ≈ 15 ms/frame on DirectML. `live_debug` (`cargo run -p fasec-ui --example live_debug --release`) runs at **~10 FPS on the dev laptop, limited by the webcam** (1280×720 delivers ~10 FPS even with no vision; detect 7–13 ms), so ≥ 15 FPS is not yet met; it also still needs a human run with a face in view.

---

## M2: Identity, enrolment, secure storage (2 weeks)

**Goal:** know who is who, store it safely.

### Tasks
- [x] `fasec-platform::dpapi` (protect/unprotect bytes), `zeroize` on buffers.
- [x] `gallery.db` schema + migrations (`rusqlite`), owner uniqueness index.
- [x] Argon2id PIN set/verify (4–12 digits, m = 19 MiB, t = 2, p = 1).
- [x] Enrolment state machine: pose targets (front/left/right/up/down), per-sample quality gates (size, blur via Laplacian variance, single face, liveness). Guided UI: `cargo run -p fasec-ui --example enroll`.
- [x] Gallery matcher: cosine, max-over-templates, threshold + margin, per-track vote (`IdentityTracker`; shown live by `live_debug --gallery`).
- [x] Model-ID tagging; force re-enrol on recogniser change.
- [x] Delete-all-biometrics (gallery: persons, templates, calibration, PIN; intruder snapshots do not exist yet).
- [ ] Threshold tuning script: compute genuine/impostor score distributions from personal clips + LFW; pick threshold for target FAR. *Not done:* the SFace model-card default 0.363 / margin 0.05 is used.
- [x] Unit tests: matcher, vote, storage round-trip, tamper detection.

### Exit criteria
- [ ] Owner + 2 authorised users enrolled via a test harness; identity correct on personal replay set with ≥ 99 % frame-level accuracy for enrolled people at ≤ 1 m.
- [ ] No plaintext template on disk (checked by grepping the DB file for float patterns / inspection).

**Status (2026-11-22):** the harness is `enroll` + `live_debug --gallery` (see the README). Automated checks pass: `template_blob_is_not_plaintext`, tamper/wrong-key load failures, margin/vote tests, enrolment gate tests, and (ignored, local) the blur gate on the `.assets` portraits. Still needs a person: enrol the owner and 2 authorised users and check the names in `live_debug`, record the personal replay set and measure the ≥ 99 % rate, and inspect a real `gallery.db` with `sqlite3` (template blobs must be ciphertext only).

---

## M3: Policy engine (1.5 weeks)

**Goal:** turn observations into correct, calm threat levels.

### Tasks
- [ ] Per-track identity state machine ([SRS §5.1](SRS.md#51-per-track-identity-state)).
- [ ] Presence aggregates and onlooker rule (facing + range + dwell).
- [ ] Threat precedence, activation/recovery delays, min-protect time.
- [ ] Presets (Relaxed/Balanced/Paranoid) + custom overrides.
- [ ] Owner-absent rule: `off | blur | lock` with its own delay.
- [ ] Pause, panic, unlock inputs.
- [ ] Snapshot tests (`insta`) for scenario timelines S1–S10 using synthetic report streams.
- [ ] Replay labels format + metrics (latency, false/missed triggers).

### Exit criteria
- Line coverage ≥ 90 % for `fasec-policy`.
- On the personal replay set at Balanced: 0 false triggers in owner-alone clips, ≥ 95 % onlooker detection, p95 decision latency ≤ 400 ms (leaving ≤ 100 ms for actions).

---

## M4: Overlays + action engine (2.5 weeks)

**Goal:** protect and restore reliably on any monitor layout.

### `fasec-overlay`
- [ ] Overlay thread with own message loop; command channel.
- [ ] Per-monitor windows, pre-created hidden, PMv2 DPI, `WM_DISPLAYCHANGE` handling.
- [ ] Styles: top-most, tool, no-activate, layered, `WDA_EXCLUDEFROMCAPTURE`.
- [ ] Backends: DWM backdrop blur, capture-based blur, solid, image (fit/fill/tile), message (text rendering via Direct2D/DirectWrite).
- [ ] Runtime backend detection + fail-safe to opaque.
- [ ] Z-order re-assert timer + foreground hook while visible.
- [ ] Click-through vs input-blocking modes.

### `fasec-actions`
- [ ] `Action` trait (`activate`, `restore`, `is_active`, `reversible`).
- [ ] Built-ins: blur, solid, image, message, minimize_all (remember and restore window list where possible), lock_workstation.
- [ ] `snapshot` action: frame ring buffer in vision worker, JPEG + DPAPI, `snapshot` table, per-incident rate limit, retention purge.
- [ ] Chain executor with per-step delays, cancellation, reverse restore.
- [ ] Reconciliation on level change (diff current vs desired active set).
- [ ] Profiles model + serde.
- [ ] Tests with a mock overlay backend.

### Exit criteria
- AT-13, AT-14, AT-15, AT-16, AT-18, AT-20, AT-21, AT-27 from [MVP §6](MVP.md#6-acceptance-tests) pass.
- Overlay show ≤ 50 ms from command (measured).

---

## M5: App shell, UI, wizard (3 weeks)

**Goal:** a usable product.

### Tasks
- [ ] `fasec-app`: supervisor, thread wiring, heartbeats, panic restart policy, single instance (named mutex).
- [ ] Tray (`tray-icon`): status icons, menu, profile submenu, pause/resume.
- [ ] Global hotkeys (`global-hotkey`): panic, unlock/pause.
- [ ] Unlock flow: `UserConsentVerifier` → fallback PIN dialog (shown above overlays).
- [ ] Camera-loss popup (top-most, above overlays): cause detection (unplugged / busy / covered / permission / driver error), Pause / Switch camera / Keep protecting, auto-close on recovery.
- [ ] Windows build check at startup (≥ 19041) with a clear message.
- [ ] Session lock/unlock + display power notifications → pause/resume capture.
- [ ] `fasec-config`: schema, defaults, validation, versioned migration, hot reload.
- [ ] `fasec-models`: downloader (progress, resume, SHA-256), licence screen, manual import.
- [ ] `fasec-ui` (egui): tabs Status, Camera (picker, formats, calibration), People (list, enrol, delete), Protection (profiles + chain editor with drag reorder), Sensitivity (incl. owner-absent action), Privacy & Data (snapshot switch, retention, viewer behind owner verification, delete/export), Diagnostics (backend, EP, FPS, per-stage ms, event log viewer).
- [ ] Live preview with debug overlay.
- [ ] First-run wizard (J1 in [MVP §4](MVP.md#4-mvp-user-journeys)).
- [ ] Start with Windows toggle.
- [ ] Event log writer + viewer.
- [ ] In-app clip recorder (opt-in) for test sets.

### Exit criteria
- J1–J5 journeys work end to end.
- AT-01, AT-09–AT-12, AT-17, AT-19, AT-22–AT-25, AT-28–AT-33 pass.

---

## M6: MVP hardening and release v0.1.0 (2 weeks)

### Tasks
- [ ] Full acceptance run on Win10 22H2 + Win11 machines.
- [ ] 8-hour soak (AT-26), fix leaks.
- [ ] 5 days of owner dogfooding; tune presets from the event log.
- [ ] Profiling pass: CPU/RAM targets.
- [ ] README: install, screenshots, privacy statement, limitations, model licences, FAQ.
- [ ] `SECURITY.md` (how to report vulnerabilities).
- [ ] Release workflow: build `x86_64-pc-windows-msvc`, zip with manifest, checksums, GitHub Release.
- [ ] Tag `v0.1.0`.

### Exit criteria
- [MVP definition of done](MVP.md#5-definition-of-done-mvp) met.

---

## M7 → v0.2.0: Strong liveness (4 weeks)

- [ ] Dense landmarks (Face Landmarker ONNX) for eye aspect ratio → blink detection.
- [ ] Non-rigid motion cue (landmark residual after rigid fit) → static-photo rejection.
- [ ] Fusion model for liveness score (simple logistic fusion, tuned on replay set).
- [ ] IR source pairing UI (`MediaFrameSourceGroup`), RGB↔IR alignment (box mapping by calibration or same-group intrinsics).
- [ ] IR liveness checks: face present in IR, reflectance statistics, screen-replay rejection.
- [ ] Liveness tier indicator in tray/status.
- [ ] Active challenge (random head-turn sequence) for enrolment and optional unlock.
- [ ] Spoof attack replay set expanded (print, phone, tablet, monitor, cut-out with eye holes).
- **Exit:** with IR: ≥ 99 % rejection of print/phone/tablet attacks in test set; RGB-only: ≥ 95 % with temporal cues.

## M8 → v0.3.0: Attention+ and targeted actions (5 weeks)

- [ ] Gaze refinement from iris landmarks (and optionally L2CS-Net) for near faces.
- [ ] Per-monitor scope for blur/solid/image.
- [ ] Window/app-scoped overlays following the window (WinEvent hooks).
- [ ] `minimize_apps(list)`, `media_pause`, `mute`, `sound_alert`, `toast` actions.
- [ ] Automatic profile switching: foreground app, schedule, Wi-Fi SSID.
- [ ] Battery-aware adaptive FPS.
- [ ] i18n (English + Vietnamese) via `fluent`.
- [ ] ARM64 build.
- [ ] `docs/THREAT_MODEL.md`.
- **Exit:** false trigger rate unchanged or better vs v0.1; window-scoped blur tracks window moves with ≤ 1 frame visible lag at 60 Hz in normal drags.

## M9 → v0.4.0: Selective content masking (6–8 weeks, research-heavy)

Phase A, UIA (3 wk)
- [ ] `fasec-content` crate; UIA tree walker with caching and event subscriptions (structure/property changed).
- [ ] Rule language: app (exe), control type, automation ID / class / name regex, region of window.
- [ ] Built-in rule packs, verified one by one: Windows Mail/Outlook (new), Teams, Slack, Discord, Telegram Desktop, Zalo PC (where UIA exposes text).
- [ ] Mask renderer (many small capture-excluded rects or one per-window mask surface).
- [ ] Conservative mode: mask whole content pane when UIA data is stale.

Phase B, OCR (3 wk)
- [ ] WGC window capture → `Windows.Media.Ocr` (Vietnamese/English language packs) → line boxes.
- [ ] Optional PaddleOCR (`oar-ocr`) backend for better accuracy.
- [ ] Classifiers: regex packs (email, phone VN/intl, card+Luhn, IBAN, API-key/JWT shapes), keyword lists, optional GLiNER NER (`gline-rs`).
- [ ] Change detection (frame diff) to avoid re-OCR of static windows; over-mask on scroll.

Phase C, evaluation (1–2 wk)
- [ ] Per-app compatibility table in docs.
- [ ] Latency and CPU measurements; publish limitations honestly.
- **Exit:** for at least 3 popular apps, message bodies are masked while navigation stays visible, with no visible leak during normal (non-scroll) use.

## M10 → v0.5.0: Browser extension (4 weeks)

- [ ] Chromium MV3 extension (Chrome/Edge); Firefox later.
- [ ] Native messaging host inside `fasec-app` (`fasec-bridge`), stdio only.
- [ ] Threat level pushed to extension; extension masks DOM by site rules (Gmail, Outlook web, Messenger, WhatsApp Web, Zalo web, Slack web).
- [ ] Rule packs as JSON, user-editable.
- **Exit:** masking inside supported sites within 100 ms of threat change; extension store listing.

## M11 → v0.6.0: Depth camera plug-in (3 weeks)

- [ ] `fasec-depth` crate with feature flags: Intel RealSense (`realsense-rust`), Orbbec SDK.
- [ ] Aligned depth for face box; relief-based liveness (FR-LV-008).
- [ ] Liveness tier `depth`.
- **Exit:** 100 % rejection of flat attacks in test set on supported devices.

## M12 → v1.0.0: Hardening (4 weeks)

- [ ] Code-signing certificate; signed binaries.
- [ ] Installer (MSIX or WiX) with `uiAccess=true` variant installed to Program Files.
- [ ] Opt-in update check (GitHub Releases), signature-verified.
- [ ] External security review of storage and unlock paths.
- [ ] Accessibility pass on UI.
- [ ] Performance regression suite in CI (replay set timings).
- **Exit:** all P0/P1 requirements done; no open high-severity issues.

---

## Cross-cutting tracks (all milestones)

| Track | Practice |
|---|---|
| Testing | Unit tests per crate; policy snapshot tests; replay metrics before every release |
| Performance | Criterion benches per vision stage; perf numbers in each release notes |
| Docs | Each milestone updates README + relevant `docs/` page |
| Privacy | Every new feature reviewed against [PRD §11](PRD.md#11-privacy-and-trust-principles) |
| Licences | `cargo deny` + model manifest licence field reviewed on change |
| Community | Good-first-issue labels from M1 on; roadmap kept in GitHub Projects |

## Top risks and responses

| Risk | Milestone hit | Early signal | Response |
|---|---|---|---|
| DWM blur not usable for top-most inactive overlay | M0/M4 | SP-1 fails | Make capture-based blur primary |
| Inference too slow on CPU-only machines | M1 | SP-3 numbers | int8 models, lower FPS, skip liveness for far faces |
| Head pose unreliable with external cameras | M1/M3 | Replay misses | Mandatory calibration; widen dwell; gaze in M8 |
| False triggers too frequent | M3/M6 | Dogfooding log | Tune dwell/vote/threshold; Relaxed default |
| Overlay below system UI surfaces | M4 | SP-2 matrix | Chain with minimize; `uiAccess` in M12 |
| Camera exclusivity conflicts | M1/M5 | SP-5 | Pause-in-call option; clear status |
| UIA coverage too thin for target apps | M9 | Phase A results | Lean on OCR + extension; publish compat table |
| Model licence change upstream | any | Manifest review | Keep alternates per task in manifest |
