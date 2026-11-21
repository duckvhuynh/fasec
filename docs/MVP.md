# fasec: MVP scope (v0.1)

| | |
|---|---|
| Status | Draft v0.1 |
| Date | 2026-11-21 |
| Related | [PRD](PRD.md) · [SRS](SRS.md) · [Roadmap](ROADMAP.md) |

## 1. MVP goal

**One sentence:** on a normal Windows laptop with any camera the user picks, fasec
recognises the owner and authorised users, ignores passers-by, and blurs every display
within half a second when a stranger looks at the screen, then restores it when the
stranger leaves. It resists simple photo and phone-screen tricks.

The MVP has to show that the core loop is reliable enough to keep running all day:
**camera → recognise → liveness → attention → decide → protect → restore**.

## 2. In scope

### 2.1 Camera
- Choose camera from a list (built-in, USB, virtual), stored by stable device ID.
- Hot-plug: list refresh, auto-reconnect to preferred camera.
- Loss policy: `protect` (default) / `fallback_to_next` / `notify_only`.
- Camera-loss popup with cause (unplugged, busy in a call/other app, covered, permission) and Pause / Switch camera / Keep protecting.
- Covered-lens / black-frame detection → treated as camera lost.
- Per-camera resolution, FPS, rotation, mirror.
- Per-camera placement calibration (look at 4 corners + centre).
- Windows camera-privacy permission detection with guidance.

### 2.2 Identity
- Owner with app PIN (Argon2id) and guided 5-pose enrolment (≥ 10 samples).
- Up to 10 authorised users, owner verification to add/remove.
- SFace templates, DPAPI-encrypted SQLite gallery, no images kept by default.
- Re-enrol and "add more samples".
- "Delete all biometric data".

### 2.3 Vision
- YuNet detection, SORT-style tracking, 5-point alignment.
- SFace recognition with per-track voting.
- MiniFASNet passive liveness (RGB tier), required before a track counts as owner.
- Head pose (6DRepNet *or* Face Landmarker, picked in the M0 spike) + distance gate.
- Adaptive scheduling (recognition/liveness intervals, idle FPS).
- DirectML when available, CPU fallback.

### 2.4 Policy
- Threat levels: None, Notice, Onlooker, Spoof, OwnerAbsent, CameraLost, Manual.
- Owner-absent setting: off (default) / blur / lock, with its own delay.
- Dwell, activation delay, recovery delay, minimum protect time.
- Presets: Relaxed / Balanced / Paranoid (+ custom values in config file).
- Pause 5/15/60 min; panic hotkey.

### 2.5 Actions and overlays
- Actions: `blur` (all displays), `solid`, `image`, `message`, `minimize_all`, `lock_workstation`, `snapshot`.
- Intruder snapshots: encrypted, 30-day retention, viewer behind owner verification, on/off switch.
- Ordered chains with per-step delay, one chain per threat level.
- Reverse-order restore; reconciliation when the level changes mid-chain.
- Profiles: create / duplicate / rename / delete / switch from tray.
- Overlays: per monitor, per-monitor DPI, pre-created, top-most, no-activate,
  capture-excluded, re-assert z-order while protected.
- Blur backends: DWM backdrop → capture-based blur → opaque (fail safe).

### 2.6 App shell
- Tray with status colours and menu.
- Settings window (egui): Status, Camera, People, Protection, Sensitivity, Privacy & Data, Diagnostics.
- Live preview with debug overlay (boxes, names, scores, liveness, pose, threat level).
- First-run wizard.
- Unlock: hotkey → Windows Hello (if available) or PIN → restore + pause 5 min.
- Start with Windows; single instance.
- Local event log with viewer.
- Model manager (manifest download, SHA-256 check, licence display, manual import).

### 2.7 Tooling
- `fasec-replay` CLI with labels → latency / false trigger / miss metrics.
- In-app opt-in clip recorder for building a personal test set.
- CI: fmt, clippy, tests (Windows + Linux for neutral crates), `cargo deny`.

## 3. Explicitly out of MVP

| Item | Why deferred | Target |
|---|---|---|
| IR liveness | Needs hardware-specific testing | v0.2 |
| Blink / motion temporal liveness | Needs dense landmarks + tuning | v0.2 |
| Active challenge liveness | UX work | v0.2 |
| Eye gaze | Head pose is enough for v0.1 | v0.3 |
| Per-monitor / per-window actions, minimise selected apps | Window tracking complexity | v0.3 |
| Media pause, mute, sound, toast actions | Nice to have | v0.3 |
| Automatic profile switching | Needs rules UI | v0.3 |
| Selective content masking (UIA/OCR) | Research-heavy | v0.4 |
| Browser extension | Separate codebase | v0.5 |
| Depth cameras | Niche hardware | v0.6+ |
| `uiAccess` signed build, installer signing, auto-update | Needs code-signing cert | v1.0 |
| Vietnamese UI | Do after strings are stable | v0.3 |
| ARM64 build | Test hardware | v0.3 |

## 4. MVP user journeys

### J1: First run
1. Install / unzip → launch `fasec.exe`.
2. Wizard: privacy notice → pick camera (live preview) → download models (progress, licences shown).
3. Set PIN → enrol owner (on-screen pose guide, 10+ samples, liveness on each).
4. Calibrate (look at 4 corners + centre).
5. Choose owner-absent action (off / blur / lock) and whether to keep intruder snapshots.
6. Test: "Ask someone to look at the screen" or use the "simulate onlooker" button → blur appears → restore.
7. Finish → tray icon green, monitoring.

### J2: Daily use
- Owner works → nothing happens.
- Colleague (authorised) sits next to them → nothing happens.
- Stranger stops behind → screen blurs in ≤ 0.5 s → stranger leaves → restore after ~2 s.
- Owner presses `Ctrl+Alt+O` → Windows Hello → restored + paused 5 min.

### J3: Add authorised user
Settings → People → Add → owner verifies → new person sits in front → guided enrolment → saved.

### J4: Change response
Settings → Protection → profile "Café" → Onlooker chain: `blur` → after 3 s `minimize_all` → Save → switch from tray.

### J5: Camera swap
User plugs in USB camera → Settings → Camera → select → calibrate → done. Unplugging it → screen protected (default) → plugging back → resumes.

## 5. Definition of done (MVP)

- All P0 requirements in the [SRS](SRS.md) implemented or explicitly waived in an issue.
- All acceptance tests in §6 pass on **two** machines: one Windows 10 22H2 laptop and one
  Windows 11 laptop (one with an external USB camera).
- Metrics on the replay set and in an 8-hour dogfooding day meet §7.
- README with install, privacy statement, limitations (SRS §9), model licences.
- Tagged release `v0.1.0` with a zip (portable) build.

## 6. Acceptance tests

| ID | Test | Pass criteria |
|---|---|---|
| AT-01 | Fresh install, run wizard end-to-end | Done ≤ 3 min, monitoring starts |
| AT-02 | Owner alone, 30 min normal work | 0 triggers |
| AT-03 | Owner + authorised user side by side, 10 min | 0 triggers |
| AT-04 | Stranger walks past behind, not looking, 10 passes | ≤ 1 trigger (Notice allowed) |
| AT-05 | Stranger stops and looks over shoulder, 20 trials | ≥ 19 triggers, p95 latency ≤ 500 ms |
| AT-06 | Stranger leaves | Restore within recovery delay + 0.5 s, 20/20 |
| AT-07 | Printed photo of owner held to camera, owner absent | Not accepted as owner in ≥ 9/10 trials |
| AT-08 | Owner photo on phone screen, owner absent | Not accepted as owner in ≥ 9/10 trials |
| AT-09 | Unplug active camera | Protected ≤ 1 s (loss policy = protect) |
| AT-10 | Replug camera | Monitoring resumes ≤ 3 s |
| AT-11 | Cover lens with finger | Protected ≤ 2.5 s |
| AT-12 | Two USB cameras, switch in settings | New camera used, old released, choice survives restart |
| AT-13 | Dual monitor, different DPI | Both monitors fully covered, no gaps |
| AT-14 | Unplug second monitor while protected | Remaining monitor still covered, no crash |
| AT-15 | Chain: blur → 2 s → minimize_all; then restore | Order correct; windows restored where restorable |
| AT-16 | Level changes Onlooker → Spoof mid-chain | Final state matches Spoof chain, no leftover overlays |
| AT-17 | Unlock with Windows Hello and with PIN | Restored + paused; wrong PIN rejected |
| AT-18 | Panic hotkey | Solid overlay ≤ 100 ms |
| AT-19 | Kill vision thread (debug command) | Protected, supervisor restarts, notification shown |
| AT-20 | Transparency effects off in Windows settings | Blur falls back to capture-based or opaque, never transparent |
| AT-21 | Remote Desktop session | App runs; blur falls back; no crash |
| AT-22 | Delete all biometric data | Gallery empty, app returns to "Not set up" |
| AT-23 | Tamper: edit gallery.db blob | Load fails safely, user told to re-enrol |
| AT-24 | Corrupted model file | Hash mismatch reported, monitoring refuses to start without protection notice |
| AT-25 | Lock / unlock Windows session | Capture paused while locked, resumes after |
| AT-26 | 8-hour soak | No crash, RAM growth ≤ 10 %, CPU within target |
| AT-27 | Screen recording (OBS / Snipping Tool) while protected | Overlay not captured (capture-excluded); documented behaviour |
| AT-28 | Camera taken by Teams/Zoom during a call | Either both work, or the camera-loss popup says "busy in another app" and Pause works |
| AT-29 | Camera-loss popup: Switch camera, Keep protecting, camera returns | Each button does what it says; popup closes itself when the camera recovers |
| AT-30 | Owner-absent = blur, then = lock, then = off; owner leaves | Blur / lock / nothing after the configured delay |
| AT-31 | Onlooker trigger with snapshots on | One encrypted snapshot per incident, visible only after owner verification, linked in event log |
| AT-32 | Snapshot older than retention; snapshots switched off | Purged at next start; no new snapshots when off |
| AT-33 | Launch on Windows 10 build < 19041 | Clear "unsupported Windows version" message, no crash |

## 7. MVP metrics targets

| Metric | Target |
|---|---|
| Onlooker latency p50 / p95 | ≤ 300 ms / ≤ 500 ms |
| False triggers (owner alone) | ≤ 1 per 8 h |
| Missed onlooker (AT-05) | ≤ 5 % |
| Photo/phone spoof acceptance (AT-07/08) | ≤ 10 % |
| CPU avg steady / idle | ≤ 10 % / ≤ 3 % |
| RAM | ≤ 400 MB |
| Cold start | ≤ 3 s |

## 8. Test data plan

- **Personal set** (never committed): recorded with the in-app recorder: owner alone
  (various light), owner + 1–2 authorised, 3–5 strangers passing/looking, printed photo,
  phone replay, tablet replay, covered lens. Each clip labelled with expected threat intervals.
- **Public sets** for offline threshold tuning only (respect their licences, do not
  redistribute): LFW for verification thresholds; CelebA-Spoof / CASIA-FASD-style sets
  for spoof sanity checks.
- `fasec-replay` runs the personal set in CI on a self-hosted runner (optional), or
  locally before each release.

## 9. MVP risks to retire early (M0 spikes)

| Spike | Question | Fallback if it fails |
|---|---|---|
| SP-1 Overlay blur | Does a top-most, non-activating window with DWM backdrop blur live content on Win10 & Win11? What happens when inactive / transparency off? | Capture-based blur via WGC + D3D11; else opaque |
| SP-2 Z-order | What covers our overlay: Start, Alt-Tab, Task View, toasts, Task Manager "always on top"? | Pair blur with `minimize_all`; document; `uiAccess` later |
| SP-3 Inference cost | YuNet + SFace + MiniFASNet + pose on reference laptop, DirectML vs CPU | Lower FPS, int8 models, skip pose for far faces |
| SP-4 Head pose model | 6DRepNet vs Face Landmarker + PnP: accuracy vs cost | Landmark-based coarse pose from YuNet 5 points |
| SP-5 Camera sharing | Can fasec and Teams use the same camera at once (driver-dependent)? | Detect busy camera, show status, optional pause-in-call |
| SP-6 nokhwa coverage | Stable IDs, format negotiation, hot-plug on several cameras | Direct Media Foundation via `windows` crate |
| SP-7 IR access (for v0.2) | Can we read IR frames from Windows Hello cameras on test laptops? | Mark IR tier as unsupported on that device |
