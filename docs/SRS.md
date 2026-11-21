# fasec: Software Requirements Specification (SRS)

| | |
|---|---|
| Status | Draft v0.1 |
| Date | 2026-11-21 |
| Related | [PRD](PRD.md) · [MVP](MVP.md) · [Roadmap](ROADMAP.md) |

Keywords **MUST**, **SHOULD** and **MAY** follow RFC 2119. Each requirement has an ID,
a priority (P0 = MVP, P1, P2) and a verification method: **T** test, **R** replay
harness, **M** manual test, **I** inspection.

---

## 1. Introduction

### 1.1 Purpose
This document specifies the behaviour, interfaces, architecture and quality
attributes of fasec, a Windows desktop application written in Rust that protects
on-screen content from unauthorised onlookers using on-device computer vision.

### 1.2 Scope
In scope: camera management, face detection/recognition, liveness, attention
estimation, decision policy, protection actions, overlays, configuration, local
storage, tray/settings UI, tooling for evaluation.
Out of scope for v1: macOS/Linux, cloud features, process injection, DRM.

### 1.3 Definitions
| Term | Meaning |
|---|---|
| Track | A face followed across frames with a stable ID |
| Template | A face embedding vector stored for an enrolled person |
| Gallery | All templates of all enrolled people |
| Owner | The single privileged enrolled person |
| Authorised user | An enrolled person allowed to see the screen |
| Stranger | A live face that matches nobody in the gallery |
| Spoof | A face judged to be a presentation attack (photo, screen, mask) |
| Onlooker | A stranger (or spoof) facing the screen, inside the distance gate, for at least the dwell time |
| Threat level | The policy engine's output (§5.4) |
| Action chain | Ordered list of actions bound to a threat level in a profile |
| Liveness tier | Strength of the liveness evidence available: `rgb`, `rgb_temporal`, `ir`, `depth` |
| Attention cone | Range of head yaw/pitch, calibrated per camera, that counts as "facing the screen" |

### 1.4 References
- ONNX Runtime and `ort` crate docs
- OpenCV Zoo: YuNet, SFace model cards
- Silent-Face-Anti-Spoofing (minivision-ai)
- 6DRepNet; MediaPipe Face Landmarker model card
- Win32: `SetWindowDisplayAffinity`, `SetWinEventHook`, `LockWorkStation`, DWM backdrop attributes
- WinRT: `Windows.Media.Capture.Frames`, `Windows.Devices.Enumeration.DeviceWatcher`,
  `Windows.Security.Credentials.UI.UserConsentVerifier`, `Windows.Media.Ocr`

---

## 2. Overall description

### 2.1 Product perspective
Standalone per-user desktop application. Runs as a normal (non-elevated) user
process started at login. No service, no driver, no injection.

### 2.2 Operating environment
| Item | Requirement |
|---|---|
| OS | Windows 10 version 2004 (build 19041) or later; Windows 11. Test matrix: Windows 10 22H2, Windows 11 current. The installer/startup MUST refuse older builds with a clear message (`WDA_EXCLUDEFROMCAPTURE` needs 19041) |
| Arch | x86-64 (MUST), ARM64 (SHOULD, P1) |
| GPU | Any DirectX 12 capable GPU for DirectML (SHOULD). CPU-only MUST work |
| Camera | Any Media Foundation camera, 640×480 @ 15 FPS minimum |
| Optional | Windows Hello IR camera, Intel RealSense / Orbbec depth camera |

### 2.3 Constraints
- C1. Rust stable toolchain, MSVC target.
- C2. No model weights in the git repository. Weights are fetched by manifest with SHA-256 verification.
- C3. Default models MUST have licences that allow redistribution and use by anyone
  (MIT / Apache-2.0 / BSD). Models with restrictive licences MAY be supported as
  user-supplied downloads only.
- C4. No network access except model download and (P2) opt-in update check.
- C5. No code injection, no hooks into other processes' rendering.

### 2.4 Assumptions
- A1. The camera faces the user and is roughly aligned with the screen, or the user runs placement calibration.
- A2. Lighting is enough for face detection (or IR is available).
- A3. The user can see the camera LED and has consented to camera use.

---

## 3. System architecture

### 3.1 Workspace layout

```
fasec/
├─ Cargo.toml                 # workspace
├─ crates/
│  ├─ fasec-core/             # shared types: Frame, Track, ThreatLevel, events, errors
│  ├─ fasec-config/           # serde config schema, load/save/migrate, validation
│  ├─ fasec-camera/           # enumeration, capture, hot-plug, IR source pairing
│  ├─ fasec-vision/           # model runtime + pipeline stages
│  │   ├─ runtime/            #   ort session mgmt, EP selection, pre/post-processing
│  │   ├─ detect/             #   YuNet face detector
│  │   ├─ track/              #   SORT-style tracker
│  │   ├─ recognize/          #   SFace embeddings + gallery matching
│  │   ├─ liveness/           #   MiniFASNet, temporal cues, IR checks
│  │   └─ attention/          #   head pose, distance, gaze (P1)
│  ├─ fasec-identity/         # enrolment flows, gallery, encrypted store
│  ├─ fasec-policy/           # per-track state, decision engine, threat levels
│  ├─ fasec-actions/          # action trait, chain executor, built-in actions
│  ├─ fasec-overlay/          # Win32 overlay windows, blur backends, per-monitor mgmt
│  ├─ fasec-platform/         # Win32/WinRT wrappers: hotkeys, session events, Hello, DPAPI, power
│  ├─ fasec-models/           # model manifest, downloader, hash check, licence info
│  ├─ fasec-ui/               # egui settings window, wizard, preview, debug view
│  ├─ fasec-app/              # binary: wiring, threads, tray, lifecycle
│  └─ fasec-replay/           # CLI: run pipeline on recorded video, emit timelines/metrics
├─ models/manifest.toml       # model URLs + SHA-256 + licence (no weights)
├─ docs/
└─ tests/fixtures/            # small synthetic fixtures only; real videos kept out of git
```

Later crates: `fasec-content` (UIA/OCR masking, P2), `fasec-bridge` (browser
extension native host, P2), `fasec-depth` (depth camera plug-in, P2).

Rule: only `fasec-platform`, `fasec-camera`, `fasec-overlay`, `fasec-actions` and
`fasec-app` may depend on the `windows` crate. Vision, policy, identity and config stay
platform-neutral so they can be tested on any OS and ported later.

### 3.2 Runtime topology

```
 ┌──────────────┐  latest-frame slot  ┌───────────────┐  VisionReport  ┌──────────────┐
 │ Camera thread│ ───────────────────▶│ Vision worker │ ──────────────▶│ Policy engine│
 │ (MF / nokhwa)│  (drops stale)      │ (ort sessions)│   (channel)    │  (tick 20 Hz)│
 └──────┬───────┘                     └───────────────┘                └──────┬───────┘
        │ device events                                                        │ ThreatChanged
        ▼                                                                      ▼
 ┌──────────────┐        commands / status (channels)                   ┌──────────────┐
 │ Supervisor   │◀─────────────────────────────────────────────────────▶│ Action engine│
 │ (fasec-app)  │                                                       └──────┬───────┘
 └──────┬───────┘                                                              │ overlay cmds
        │                                                                      ▼
 ┌──────┴───────┐                                                       ┌──────────────┐
 │ UI thread    │  (egui settings, wizard, preview; on demand)          │ Overlay thread│
 │ Tray/hotkeys │                                                       │ (Win32 msg   │
 └──────────────┘                                                       │  loop, D3D)  │
                                                                        └──────────────┘
```

- **Camera thread** owns the device. It writes into a single-slot "latest frame"
  buffer. The vision worker always takes the newest frame, so latency never builds
  up behind slow inference.
- **Vision worker** runs the stage pipeline (§3.3) and emits one `VisionReport` per
  processed frame.
- **Policy engine** is a pure state machine fed by reports and clock ticks. It emits
  `ThreatChanged` events. It is deterministic, which makes replay testing possible.
- **Action engine** turns threat changes into action-chain execution and restore.
- `ThreatChanged` carries the frame sequence number that caused it; the vision worker
  keeps the last ~2 s of frames in a small ring buffer so the `snapshot` action can fetch that frame.
- **Overlay thread** owns all overlay HWNDs and runs its own message loop, so the UI
  thread can be closed without affecting protection.
- **Supervisor** wires channels, watches health (heartbeats from each thread) and
  applies fail-safe rules if a thread stalls.

Channels: `crossbeam-channel` (or `std::sync::mpsc`). Async runtime is not needed
for the core; WinRT async calls are driven with `.get()` on worker threads.

### 3.3 Vision pipeline

```
Frame (BGR/NV12) ─▶ preprocess (resize, colour convert)
   ─▶ [every frame]     Detect (YuNet): boxes + 5 landmarks + score
   ─▶ [every frame]     Track (SORT): assign track IDs
   ─▶ [per track]       Align (5-point similarity transform → 112×112)
   ─▶ [scheduled]       Recognize (SFace → 128-d) → gallery match → identity + score
   ─▶ [scheduled]       Liveness (MiniFASNet ×2 crops) → spoof score; temporal cues
   ─▶ [every frame]     Attention (head pose yaw/pitch/roll, face size → distance)
   ─▶ VisionReport { ts, tracks: [TrackObs], camera_health, liveness_tier }
```

Scheduling (to keep CPU low):
- Recognition runs for a track when it is new, then every `recog_interval`
  (default 1 s), and immediately when the track's pose changes a lot or its box jumps.
- Liveness runs when a track is new, then every `liveness_interval` (default 2 s),
  and **always** before a track is accepted as owner/authorised for the first time.
- Identity per track is a running vote over the last *N* recognitions (default 5)
  weighted by score, so a single bad frame cannot flip the decision.
- When no faces are present, detection drops to `idle_fps` (default 5 FPS).

Model input sizes (defaults): YuNet dynamic (frame scaled to 320 px on the long side),
SFace 112×112, MiniFASNet 80×80 at two crop scales, head pose 224×224 (6DRepNet) or
192×192 (Face Landmarker).

### 3.4 Data flow: core types (sketch)

```rust
pub struct TrackObs {
    pub id: TrackId,
    pub bbox: RectF,              // normalised 0..1
    pub landmarks5: [PointF; 5],
    pub det_score: f32,
    pub identity: IdentityState,  // Unknown | Pending | Known{person, score} | Stranger{best}
    pub liveness: LivenessState,  // Unknown | Live{score, tier} | Spoof{score, tier}
    pub head_pose: Option<HeadPose>, // yaw, pitch, roll in degrees
    pub face_width_px: f32,
    pub age_ms: u64,
}

pub enum ThreatLevel { None, Notice, Onlooker, Spoof, OwnerAbsent, CameraLost, Manual }
```

### 3.5 Technology choices

| Concern | Choice | Reason |
|---|---|---|
| Inference | `ort` + DirectML EP, CPU fallback | Mature, fast, runs every model we need |
| Image ops | `image`, `fast_image_resize`, `ndarray` | Pure Rust, SIMD |
| Geometry | `nalgebra` | Similarity transform, PnP, Kalman |
| Camera | `nokhwa` (MSMF backend) + `windows` crate | Enumeration/capture; WinRT for IR & watcher |
| Win32/WinRT | `windows` crate | Official bindings |
| UI | `egui`/`eframe` (wgpu) | Pure Rust, easy live preview with debug drawing |
| Tray / hotkeys | `tray-icon`, `global-hotkey` | Maintained, Windows-native |
| Config | `serde` + `toml` | Human-editable |
| Storage | `rusqlite` (bundled) | Gallery, event log |
| Crypto | DPAPI (`CryptProtectData`), `argon2`, `zeroize` | OS-bound keys, no key file on disk |
| Logging | `tracing` + `tracing-appender` | Structured, rotated |
| Errors | `thiserror` (libs), `anyhow` (bin) | Convention |
| Tests | `cargo test`, `insta` snapshots for policy timelines, `criterion` benches | |

---

## 4. Functional requirements

### 4.1 Camera (`fasec-camera`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-CAM-001 | P0 | The system MUST list all video capture devices with friendly name, stable device ID (symbolic link) and supported formats. | T/M |
| FR-CAM-002 | P0 | The user MUST be able to choose the active camera in settings and in the first-run wizard. | M |
| FR-CAM-003 | P0 | The selection MUST be persisted by stable device ID. On start, if that ID is present it MUST be opened, regardless of enumeration order. | T |
| FR-CAM-004 | P0 | The system MUST detect camera arrival/removal at runtime (DeviceWatcher) and update the list within 2 s. | M |
| FR-CAM-005 | P0 | When the active camera disappears, fails, or delivers no frame for `camera_timeout` (default 1500 ms), the system MUST raise `CameraLost` and apply the configured loss policy: `protect` (default), `fallback_to_next`, or `notify_only`. | M/R |
| FR-CAM-005b | P0 | On `CameraLost` the system MUST show a top-most popup (above overlays) stating the cause: unplugged, busy in another app (e.g. a video call), covered/black, permission denied, or driver error. Buttons: **Pause monitoring** (5/15/60 min or until camera returns; requires owner verification when protect-settings is on), **Switch camera** (list of available cameras), **Keep protecting**. Until the user picks, the loss policy applies. The popup MUST close by itself when the camera recovers. | M |
| FR-CAM-006 | P0 | When the preferred camera returns, the system MUST switch back to it automatically. | M |
| FR-CAM-007 | P0 | Per-camera settings MUST include resolution, FPS, rotation (0/90/180/270), horizontal mirror. | M |
| FR-CAM-008 | P0 | The system MUST detect "camera access denied by Windows privacy settings" and show how to enable it. | M |
| FR-CAM-009 | P0 | The system MUST detect a covered/black camera (mean luminance below threshold for > 2 s) and treat it as `CameraLost`. | R |
| FR-CAM-010 | P0 | Per-camera placement calibration: the user looks at the four screen corners and the centre; the system stores the yaw/pitch attention cone for that camera. Uncalibrated cameras use a default cone (yaw ±25°, pitch ±20°). | M |
| FR-CAM-011 | P1 | The system SHOULD enumerate `MediaFrameSourceGroup`s and offer an IR source; RGB and IR sources MAY come from different devices. | M |
| FR-CAM-012 | P1 | The system SHOULD pause capture when the session locks, the display turns off, or the user pauses, and resume afterwards. | M |
| FR-CAM-013 | P2 | A depth-camera plug-in interface MUST provide aligned depth frames to liveness. | T |

### 4.2 Identity and enrolment (`fasec-identity`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-ID-001 | P0 | First run MUST create the owner: name, owner secret (app PIN, Argon2id-hashed; Windows Hello MAY replace it), guided enrolment. | M |
| FR-ID-002 | P0 | Guided enrolment MUST capture at least 10 accepted samples across 5 poses (front, left, right, up, down). Each sample MUST pass detection quality (size ≥ 112 px, blur score, single face) and liveness. | M/T |
| FR-ID-003 | P0 | The gallery MUST store per person: ID, display name, role (owner/authorised), templates (128-d f32), creation time, model version. | I |
| FR-ID-004 | P0 | Templates MUST be stored encrypted with DPAPI (current-user scope). No key material may be written to disk in plain form. | I |
| FR-ID-005 | P0 | Raw enrolment images MUST NOT be stored unless the user opts in; if stored, they MUST be encrypted the same way. | I |
| FR-ID-006 | P0 | Adding/removing an authorised user, changing settings that weaken protection, and deleting data MUST require owner verification (PIN or Windows Hello). | M |
| FR-ID-007 | P0 | "Delete all biometric data" MUST erase gallery, optional images, intruder snapshots and calibration, then confirm. | T |
| FR-ID-008 | P0 | Templates MUST be tagged with the recogniser model ID; on model change the system MUST require re-enrolment rather than compare across models. | T |
| FR-ID-009 | P0 | Matching: cosine similarity against all templates of each person; person score = max over templates (P1: top-k mean). Accept if score ≥ `match_threshold` (default from model card; SFace 0.363), with a margin over the second-best person ≥ `match_margin` (default 0.05). | T/R |
| FR-ID-010 | P1 | Adaptive templates: high-confidence live owner observations MAY be added as templates (bounded set, FIFO, opt-in). | T |

### 4.3 Liveness (`fasec-vision::liveness`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-LV-001 | P0 | Each track MUST receive a passive RGB liveness score from MiniFASNet (two crop scales, fused). | R |
| FR-LV-002 | P0 | A track MUST NOT be treated as owner/authorised until it has at least one live verdict at or above `liveness_threshold`. | R |
| FR-LV-003 | P0 | A track with a spoof verdict MUST be reported as `Spoof` and MUST NOT count as owner presence. | R |
| FR-LV-004 | P0 | The system MUST expose the current liveness tier in UI and logs. | M |
| FR-LV-005 | P1 | Temporal cues: blink detection (eye aspect ratio from dense landmarks) and non-rigid motion SHOULD be fused into the score; a face with no blink and no non-rigid motion for `static_timeout` (default 20 s) SHOULD be downgraded. | R |
| FR-LV-006 | P1 | IR liveness: when an IR source is configured, the face region in the IR frame MUST show expected skin reflectance and detectable face structure; failure means spoof. | R/M |
| FR-LV-007 | P1 | Active challenge (random head-turn sequence) SHOULD be available for enrolment and unlock. | M |
| FR-LV-008 | P2 | Depth liveness: face depth relief ≥ `min_relief_mm` across nose–cheek–ear region. | T |

### 4.4 Attention (`fasec-vision::attention`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-AT-001 | P0 | Each track MUST have a head pose estimate (yaw, pitch, roll) with a confidence. | R |
| FR-AT-002 | P0 | A track is *facing* when its pose is inside the camera's attention cone (FR-CAM-010) with hysteresis of 5°. | R |
| FR-AT-003 | P0 | A track is *in range* when face width ≥ `min_face_px` (default derived from "max distance" setting, ~2 m) — faces smaller than this are ignored for triggering. | R |
| FR-AT-004 | P0 | A stranger/spoof track becomes an **onlooker** after being facing + in range for ≥ `dwell_ms` (default 300 ms Balanced, 150 Paranoid, 800 Relaxed). | R |
| FR-AT-005 | P1 | Gaze SHOULD refine the facing test when the face is large enough (iris visible, face width ≥ 120 px). | R |

### 4.5 Policy engine (`fasec-policy`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-PO-001 | P0 | The engine MUST be a deterministic function of (config, sequence of VisionReports, clock ticks). Same input → same output. | T |
| FR-PO-002 | P0 | It MUST compute threat level with this precedence: `Manual` > `CameraLost` > `Spoof` > `Onlooker` > `OwnerAbsent` > `Notice` > `None` (§5.4). | T |
| FR-PO-003 | P0 | Activation: a level is raised only after its condition holds for `activation_delay_ms` (per level; default 0 for Onlooker beyond dwell, 0 for CameraLost, 500 for Spoof, 10 000 for OwnerAbsent). | R |
| FR-PO-004 | P0 | Recovery: a level is cleared only after its condition is absent for `recovery_delay_ms` (default 2000) and after `min_protect_ms` (default 1500) has elapsed since activation. | R |
| FR-PO-005 | P0 | Unknown identity (track not yet recognised): behaviour depends on preset: Relaxed = ignore until resolved; Balanced = treat as stranger if facing for > 1 s unresolved; Paranoid = treat as stranger immediately. | R |
| FR-PO-006 | P0 | Owner-alone and owner-with-authorised-users MUST yield `None`. | R |
| FR-PO-007 | P0 | A stranger not facing or not in range MUST yield at most `Notice`. | R |
| FR-PO-008 | P0 | Manual "panic" hotkey MUST yield `Manual` immediately until the user releases it (owner verification if configured). | M |
| FR-PO-009 | P0 | "Pause for N minutes" MUST suppress all levels except `Manual`, and show a clear paused status in the tray. | M |
| FR-PO-010 | P0 | Owner-absent setting, independent of strangers: `action = off | blur | lock` (default `off`), `delay_s` (default 30). `blur` raises `OwnerAbsent` and runs that level's chain; `lock` calls `lock_workstation` once. A spoof of the owner does not count as owner present. | R |

### 4.6 Action engine (`fasec-actions`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-AE-001 | P0 | Actions MUST implement a common trait with `activate`, `restore`, and `is_active`; `restore` MUST be idempotent. | T |
| FR-AE-002 | P0 | A profile MUST map each threat level to an ordered chain of steps `{action, params, delay_ms}`. | T |
| FR-AE-003 | P0 | On level rise the engine MUST run the chain in order; on level clear it MUST restore active actions in reverse order. | T |
| FR-AE-004 | P0 | If the level changes during a chain, the engine MUST reconcile: run steps for the new chain that are not already active, restore steps that are no longer wanted. | T |
| FR-AE-005 | P0 | The first visual protection step SHOULD complete within 100 ms of `ThreatChanged`. | T/M |
| FR-AE-006 | P0 | Built-in actions: `blur` (all monitors), `solid` (colour, opacity), `image` (path, fit mode), `message` (text, style: fullscreen/banner), `minimize_all`, `lock_workstation`, `snapshot`. | M |
| FR-AE-007 | P0 | Non-reversible actions (lock) MUST be marked so; restore for them is a no-op. | T |
| FR-AE-008 | P0 | Profiles: create, rename, duplicate, delete, set active; switching from tray. At least one profile always exists. | M |
| FR-AE-009 | P1 | Additional actions: `blur`/`solid`/`image` scoped to one monitor or to a window/app; `minimize_apps(list)`; `media_pause`; `mute`; `sound_alert`; `toast`. | M |
| FR-AE-010 | P1 | Automatic profile switching rules (foreground app, schedule, network SSID). | M |
| FR-AE-011 | P0 | `snapshot` action: save the camera frame that caused the trigger (full frame + boxes/roles/scores) as JPEG, DPAPI-encrypted, at most 1 per `snapshot_min_interval_s` (default 30) per incident. Retention `snapshot_retention_days` (default 30), purged at startup and daily. Global switch `snapshots.enabled` (asked in wizard, default on). Viewer in Settings › Privacy & Data behind owner verification, with delete one / delete all / export (decrypted copy to a user-chosen folder). Not reversible (restore is a no-op). | T/M |

### 4.7 Overlay (`fasec-overlay`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-OV-001 | P0 | One overlay window per monitor, covering the full monitor rect, per-monitor-DPI aware (V2). | M |
| FR-OV-002 | P0 | Overlays MUST be top-most, tool windows (no taskbar button), non-activating (`WS_EX_NOACTIVATE`), and excluded from capture (`WDA_EXCLUDEFROMCAPTURE`). | I/M |
| FR-OV-003 | P0 | Overlays MUST be pre-created hidden at startup so showing them costs one `ShowWindow`/`SetWindowPos` call. | I |
| FR-OV-004 | P0 | Blur backends in order of preference: (1) DWM system backdrop/acrylic; (2) capture-based blur (Windows Graphics Capture of the monitor → GPU blur → present); (3) opaque solid. The selected backend MUST be detected at runtime and shown in diagnostics. | M |
| FR-OV-005 | P0 | Any blur backend failure MUST fall back to opaque (fail safe), never to transparent. | T/M |
| FR-OV-006 | P0 | Overlays MUST react to monitor add/remove/resolution change (`WM_DISPLAYCHANGE`) within 1 s. | M |
| FR-OV-007 | P0 | While protected, overlays MUST re-assert top-most z-order at least every 250 ms and on `EVENT_SYSTEM_FOREGROUND`. | M |
| FR-OV-008 | P0 | Overlay input mode is configurable: click-through (default for blur/message) or input-blocking (default for solid/image). | M |
| FR-OV-009 | P1 | Window-scoped overlays MUST follow the target window via `SetWinEventHook(EVENT_OBJECT_LOCATIONCHANGE)` and hide when the window minimises. | M |
| FR-OV-010 | P2 | A `uiAccess=true` signed build SHOULD be offered so overlays can sit above Start/Alt-Tab bands. | M |

### 4.8 Selective content protection (`fasec-content`, P2)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-SC-001 | P2 | For allow-listed apps, the system SHOULD find text elements via UI Automation and place mask rectangles over elements matching rules (control type, automation ID, class, name pattern, region of window). | M |
| FR-SC-002 | P2 | OCR masking: capture the target window (WGC), run OCR (Windows.Media.Ocr by default), classify lines, mask matching boxes. Target ≤ 300 ms end-to-end at 1080p. | M |
| FR-SC-003 | P2 | Classifiers: regex packs (email, phone, card numbers with Luhn, IBAN, API-key shapes), keyword lists, optional NER model. | T |
| FR-SC-004 | P2 | Masks MUST be our own top-most, capture-excluded windows; the system MUST NOT modify target apps. | I |
| FR-SC-005 | P2 | On scroll/resize of the target, masks SHOULD expand to cover the whole affected region until re-analysis finishes (prefer over-masking to leaks). | M |
| FR-SC-006 | P2 | A browser extension (Chromium MV3) MAY mask DOM elements by rules and talk to the app via native messaging. | M |

### 4.9 Application shell (`fasec-app`, `fasec-ui`, `fasec-platform`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-APP-001 | P0 | Tray icon MUST show state: Monitoring / Protected / Paused / Camera problem / Not set up. | M |
| FR-APP-002 | P0 | Tray menu: Open settings, Pause 5/15/60 min, Resume, Profile ▸, Panic, Quit (Quit requires owner verification if enabled). | M |
| FR-APP-003 | P0 | Global hotkeys (configurable): Panic (default `Ctrl+Alt+P`), Unlock/Pause (default `Ctrl+Alt+O`). | M |
| FR-APP-004 | P0 | Unlock while protected: owner verification (Windows Hello via `UserConsentVerifier` if available, else PIN) → restore and pause for `unlock_pause_min` (default 5). | M |
| FR-APP-005 | P0 | Settings window: camera, people, sensitivity, profiles/actions editor, diagnostics; live preview with boxes, names, liveness, pose and threat level. | M |
| FR-APP-006 | P0 | First-run wizard: welcome & privacy notice → camera pick + test → model download → owner PIN → enrolment → calibration → owner-absent action → intruder snapshots on/off (with notice) → test protection. | M |
| FR-APP-007 | P0 | Start with Windows (per-user Run key or Startup task), on by default after wizard. | M |
| FR-APP-008 | P0 | Single instance; a second launch focuses the running instance. | T |
| FR-APP-009 | P0 | Local event log: timestamp, level change, reason, track summary (no images, no embeddings). Viewer in settings, 30-day rotation. | I |
| FR-APP-010 | P0 | Model manager: download from manifest, verify SHA-256, show licence, allow manual import for offline installs. | T/M |
| FR-APP-011 | P1 | UI languages: English, Vietnamese. | M |
| FR-APP-012 | P1 | Performance mode: adaptive FPS on battery / battery saver. | M |

### 4.10 Tooling (`fasec-replay`)

| ID | Pri | Requirement | Verify |
|---|---|---|---|
| FR-RP-001 | P0 | CLI runs the full vision + policy pipeline on a video file or image sequence with a given gallery and config, using a simulated clock. | T |
| FR-RP-002 | P0 | Output: per-frame JSON lines (tracks, scores) and a threat timeline; optional annotated video. | T |
| FR-RP-003 | P0 | Given a label file (expected threat intervals), output latency, false-trigger count, missed-trigger count. | T |
| FR-RP-004 | P0 | A record mode in the app MUST capture raw camera clips for building local test sets (explicit, opt-in, stored locally). | M |

---

## 5. Behavioural specification

### 5.1 Per-track identity state

```
        new track
            │
            ▼
       ┌─────────┐ recog ok & live ok  ┌───────────┐
       │ Pending │ ───────────────────▶│ Known(p)  │
       └────┬────┘                     └─────┬─────┘
            │ N recogs, no match, live ok    │ vote flips (N consecutive)
            ▼                                ▼
       ┌──────────┐                   ┌─────────────┐
       │ Stranger │◀──────────────────│ re-evaluate │
       └──────────┘                   └─────────────┘
   any state ── spoof verdict ──▶ Spoof (sticky for spoof_hold_ms, default 5 s)
   track lost for > track_ttl_ms (default 1 s) ──▶ removed
```

### 5.2 Presence aggregates (computed each tick)

- `owner_present` = any track Known(owner) with live verdict
- `authorised_present` = any Known(authorised) live
- `strangers` = tracks in Stranger or Spoof
- `onlookers` = strangers satisfying FR-AT-004
- `spoof_present` = any Spoof track facing the camera

### 5.3 Scenario truth table (Balanced preset)

| Owner | Others | Others facing & in range | Result |
|---|---|---|---|
| yes | none | — | None |
| yes | authorised only | any | None |
| yes | stranger | no | Notice |
| yes | stranger | yes (≥ dwell) | **Onlooker** |
| no | none | — | None → OwnerAbsent after delay (if enabled) |
| no | stranger | yes (≥ dwell) | **Onlooker** |
| any | spoof | — | **Spoof** |
| camera lost | — | — | **CameraLost** |

### 5.4 Threat level semantics

| Level | Meaning | Default chain (profile "Default") |
|---|---|---|
| None | Safe | — |
| Notice | Stranger present but not looking | (none; tray icon turns amber) |
| Onlooker | Stranger looking at screen | blur(all) → snapshot |
| Spoof | Presentation attack seen | solid(black) → message("Protection active") → snapshot |
| OwnerAbsent | Owner gone for > delay (only when owner-absent action = `blur`) | blur(all) |
| CameraLost | No trustworthy camera data | blur(all) + camera-loss popup (FR-CAM-005b) |
| Manual | Panic hotkey | solid(black) |

### 5.5 Timing parameters (defaults)

| Parameter | Relaxed | Balanced | Paranoid |
|---|---|---|---|
| `dwell_ms` | 800 | 300 | 150 |
| Unknown-as-stranger after | never | 1000 ms | 0 ms |
| `recovery_delay_ms` | 1500 | 2000 | 4000 |
| `min_protect_ms` | 1000 | 1500 | 3000 |
| `match_threshold` offset | −0.02 | 0 | +0.04 |
| `liveness_threshold` | 0.6 | 0.75 | 0.9 |
| Detection FPS (faces present) | 10 | 15 | 20 |

---

## 6. External interfaces

### 6.1 User interface
- Tray icon + menu (always).
- Settings window (egui) with tabs: **Status**, **Camera**, **People**, **Protection**
  (profiles & chains), **Sensitivity**, **Privacy & Data**, **Diagnostics**, **About**.
- Overlays: blur / solid / image / message / banner.
- Toasts via Windows notifications (P1).

### 6.2 Configuration file
Location: `%APPDATA%\fasec\config.toml` (settings, no secrets).
Data: `%LOCALAPPDATA%\fasec\` (`gallery.db`, `events.db`, `snapshots\`, `models\`, `logs\`).

```toml
version = 1

[camera]
device_id = "\\\\?\\usb#vid_046d&pid_085e#..."
loss_policy = "protect"            # protect | fallback_to_next | notify_only
resolution = [1280, 720]
fps = 30
rotation = 0
mirror = false

[sensitivity]
preset = "balanced"                # relaxed | balanced | paranoid | custom
max_distance_m = 2.0

[owner_absent]
action = "off"                     # off | blur | lock
delay_s = 30

[snapshots]
enabled = true
retention_days = 30
min_interval_s = 30

[runtime]
execution_provider = "auto"        # auto | directml | cpu
idle_fps = 5

[profiles]
active = "default"

[[profiles.list]]
name = "default"
  [[profiles.list.chains]]
  level = "onlooker"
  steps = [ { action = "blur", scope = "all", strength = 30 }, { action = "snapshot" } ]
  [[profiles.list.chains]]
  level = "spoof"
  steps = [
    { action = "solid", color = "#000000", opacity = 1.0 },
    { action = "message", text = "Privacy protection active", delay_ms = 0 },
  ]
```

Config MUST carry a `version` and the loader MUST migrate older versions.

### 6.3 Model manifest

```toml
[[model]]
id = "yunet-2023mar"
task = "face_detection"
url = "https://.../face_detection_yunet_2023mar.onnx"
sha256 = "<hash>"
license = "MIT"
default = true
```

### 6.4 OS interfaces (summary)

| Purpose | API |
|---|---|
| Camera capture | Media Foundation (via `nokhwa`), `MediaCapture`/`MediaFrameReader` (IR) |
| Device hot-plug | `DeviceWatcher` (video capture class) |
| Overlays | `CreateWindowEx` (layered, topmost, tool, noactivate), `SetWindowDisplayAffinity`, DWM backdrop attributes |
| Capture-based blur | Windows Graphics Capture + D3D11 |
| Monitors | `EnumDisplayMonitors`, `GetDpiForMonitor`, `WM_DISPLAYCHANGE` |
| Window tracking | `SetWinEventHook`, `EnumWindows`, `GetWindowRect` |
| Minimise | `IShellDispatch::MinimizeAll` / `ShowWindow` |
| Lock | `LockWorkStation` |
| Session / power | `WTSRegisterSessionNotification`, `RegisterPowerSettingNotification` |
| Owner verify | `UserConsentVerifier::RequestVerificationAsync` |
| Secrets | `CryptProtectData` / `CryptUnprotectData` |
| UI Automation (P2) | `IUIAutomation` via `uiautomation` crate |
| OCR (P2) | `Windows.Media.Ocr.OcrEngine` |

---

## 7. Non-functional requirements

### 7.1 Performance
| ID | Requirement |
|---|---|
| NFR-PERF-001 | Onlooker latency (stranger starts facing → overlay visible) p95 ≤ 500 ms, p50 ≤ 300 ms at Balanced on reference hardware. |
| NFR-PERF-002 | Steady-state CPU ≤ 10 % average on reference laptop (4-core, DirectML available), ≤ 20 % CPU-only. |
| NFR-PERF-003 | Idle (no face) CPU ≤ 3 %. |
| NFR-PERF-004 | RAM ≤ 400 MB with default models loaded. |
| NFR-PERF-005 | Cold start to monitoring ≤ 3 s (models cached). |
| NFR-PERF-006 | Overlay show time from command ≤ 50 ms (pre-created windows). |

Reference hardware: Intel Core i5-1235U (or AMD equivalent), Iris Xe, 16 GB RAM, 720p webcam.

### 7.2 Reliability and fail-safety
| ID | Requirement |
|---|---|
| NFR-REL-001 | Every component failure that removes trustworthy vision data MUST end in a protecting state, not an unprotected one (camera loss, vision thread panic, model load failure while monitoring). |
| NFR-REL-002 | Thread panics MUST be caught by the supervisor; it restarts the component up to 3 times per minute, then stays in `CameraLost`-equivalent protection and notifies. |
| NFR-REL-003 | Overlay restore MUST happen even if the vision side is dead (restore is driven by unlock or by recovery). |
| NFR-REL-004 | 8-hour soak test without memory growth > 10 %. |

### 7.3 Security
| ID | Requirement |
|---|---|
| NFR-SEC-001 | Templates and optional images encrypted with DPAPI (user scope). |
| NFR-SEC-002 | Owner PIN hashed with Argon2id (memory ≥ 19 MiB, t ≥ 2); secrets zeroised after use. |
| NFR-SEC-003 | Settings that weaken protection (disable, lower sensitivity, remove profile, delete data, quit) require owner verification when "protect settings" is on (default on). |
| NFR-SEC-004 | Model files verified by SHA-256 before load; mismatches refuse to load. |
| NFR-SEC-005 | No listening sockets. Native-messaging host (P2) only via stdio, launched by the browser. |
| NFR-SEC-006 | `cargo deny` (licences, advisories) and `cargo audit` in CI. |
| NFR-SEC-007 | Threat model documented (`docs/THREAT_MODEL.md`, P1): attacker types, what is and is not defended. |

### 7.4 Privacy
| ID | Requirement |
|---|---|
| NFR-PRIV-001 | No frame, template or image leaves the device. |
| NFR-PRIV-002 | No telemetry. |
| NFR-PRIV-003 | Event log stores no images and no embeddings; it only references snapshot IDs. |
| NFR-PRIV-006 | Intruder snapshots: encrypted at rest, auto-purged after retention, viewable only after owner verification, never sent off-device; the wizard explains that they capture other people's faces. |
| NFR-PRIV-004 | Privacy notice in wizard explains what is processed and stored. |
| NFR-PRIV-005 | Uninstall offers to delete all data. |

### 7.5 Usability and accessibility
| ID | Requirement |
|---|---|
| NFR-USE-001 | First-run wizard completable in ≤ 3 minutes. |
| NFR-USE-002 | Every protection state is reversible by the owner within 5 s (hotkey + verification). |
| NFR-USE-003 | Overlays respect high-contrast mode for message text. |
| NFR-USE-004 | Settings UI keyboard navigable. |

### 7.6 Maintainability and portability
| ID | Requirement |
|---|---|
| NFR-MNT-001 | Platform-neutral crates (`core`, `config`, `vision`, `identity`, `policy`) build and test on Linux CI. |
| NFR-MNT-002 | Policy engine coverage ≥ 90 % lines; snapshot tests for scenario timelines. |
| NFR-MNT-003 | `clippy -D warnings`, `rustfmt` in CI. |
| NFR-MNT-004 | Models are swappable via a `FaceDetector` / `FaceEmbedder` / `LivenessModel` / `PoseEstimator` trait. |

---

## 8. Data model

### 8.1 `gallery.db` (SQLite)
```
person(id TEXT PK, name TEXT, role TEXT CHECK(role IN ('owner','authorised')),
       created_at INTEGER, updated_at INTEGER)
template(id INTEGER PK, person_id TEXT FK, model_id TEXT, blob BLOB /*DPAPI*/,
         pose TEXT, quality REAL, created_at INTEGER)
calibration(camera_id TEXT PK, yaw_min REAL, yaw_max REAL, pitch_min REAL,
            pitch_max REAL, created_at INTEGER)
secret(kind TEXT PK /*'pin'*/, hash TEXT /*argon2 PHC string*/)
```
Exactly one `owner` row (enforced by a partial unique index).

### 8.2 `events.db`
```
event(id INTEGER PK, ts INTEGER, kind TEXT, level TEXT, reason TEXT,
      tracks_json TEXT /* ids, roles, scores only */, snapshot_id TEXT NULL)
snapshot(id TEXT PK, ts INTEGER, level TEXT, file TEXT /* snapshots\<id>.bin, DPAPI */,
         meta_json TEXT /* boxes, roles, scores */, expires_at INTEGER)
```

---

## 9. Known limitations (must be documented to users)

1. RGB liveness is not proof of liveness. High-quality replays and masks can pass.
2. Overlays do not cover the secure desktop (UAC, Ctrl+Alt+Del) and may sit below
   Start menu, Alt-Tab, Task View and some toasts unless the `uiAccess` build is used.
3. Overlays cannot protect against someone photographing the screen *before* the
   trigger fires (latency window).
4. Head-pose attention is approximate; a stranger looking from a steep angle at the
   edge of the camera's view may be missed.
5. The camera only sees its own field of view: people outside it are invisible.
6. Remote Desktop / some VMs disable DWM effects, so blur falls back to opaque.
7. Granular masking (P2) lags behind fast scrolling and only works where UIA/OCR/extension can see the content.

---

## 10. Acceptance

The MVP acceptance test list lives in [MVP.md §6](MVP.md#6-acceptance-tests).
