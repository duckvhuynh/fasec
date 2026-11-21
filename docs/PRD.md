# fasec — Product Requirements Document (PRD)

| | |
|---|---|
| Status | Draft v0.1 |
| Date | 2026-11-21 |
| Platform | Windows 10 (2004 / build 19041+) and Windows 11 |
| Language | Rust |
| License | Open source (MIT OR Apache-2.0 proposed) |
| Related | [SRS](SRS.md) · [MVP](MVP.md) · [Roadmap](ROADMAP.md) |

---

## 1. Summary

fasec is a local-first desktop privacy guard. It watches the camera, recognises the
owner and trusted users, and detects when an **unauthorised person is actually looking
at the screen**. When that happens it runs a configurable chain of protection actions
(live blur, solid/image overlay, warning, minimise, lock) and restores the screen once
only authorised people remain.

Two capabilities set it apart from plain "presence detection":

1. **Spoof-resistant face verification**: a photo, a printed face or a face on a
   phone screen must not count as the owner. RGB-only liveness by default,
   infrared liveness when the hardware has it, depth cameras as a plug-in.
2. **Granular content protection** (post-MVP): mask only sensitive regions inside
   third-party apps (message bodies, previews, e-mail content) instead of hiding the
   whole screen.

All biometric processing stays on the device. No cloud, no telemetry by default.

## 2. Problem

- Shoulder surfing in offices, cafés, trains and co-working spaces leaks chats,
  e-mail, credentials, customer data and source code.
- Existing answers are either hardware (privacy filters, HP Sure View), OEM-locked
  (Lenovo Glance, Dell/Windows onlooker detection, which needs specific presence
  sensors) or crude (lock the screen whenever any face appears).
- Plain 2D face recognition can be fooled by holding up a photo of the owner, so
  "owner is present, everything is fine" is not a safe signal on its own.

## 3. Goals and non-goals

### Goals
- G1. Detect an unauthorised **onlooker** (not just a passer-by) fast enough to
  hide content before it can be read: p95 under 500 ms from "starts looking" to
  "protected".
- G2. Stay quiet while the owner works alone or with authorised people: under 1
  false trigger per 8-hour working day in normal conditions.
- G3. Resist casual spoofing (photo, phone screen) with a standard webcam, and
  strong spoofing when IR hardware is present.
- G4. Let users compose their own response: several actions, ordered, with delays,
  per scenario, grouped in profiles.
- G5. Run on an ordinary laptop: under 10 % CPU average on a 4-core laptop CPU at
  steady state, under 400 MB RAM.
- G6. Work with any camera the user picks (built-in webcam, USB camera, IR camera),
  including hot-plugging.
- G7. Be trustworthy as open source: auditable, offline, biometric data encrypted,
  easy to delete.

### Non-goals
- Not an authentication system. fasec does not replace Windows sign-in or Windows
  Hello, and does not claim to stop a determined attacker with physical access.
- Not anti-screen-capture DRM. Stopping other software from screenshotting is out of scope.
- No cloud processing, accounts, or remote management (v1).
- No macOS/Linux in v1. Keep the core crates platform-neutral so ports stay possible.
- No hooking or injecting into other processes' rendering (see §8).

## 4. Target users and personas

| Persona | Context | Main need |
|---|---|---|
| **Remote/hybrid worker** | Laptop in cafés, trains, shared desks | Hide chats/e-mail when someone looks over the shoulder |
| **Developer / admin** | Terminals, secrets, customer data | Fast blur when someone walks up; quick unlock |
| **Privacy-aware home user** | Family members / roommates nearby | Allow partner as authorised user, hide from guests |
| **Small-office worker** | Open-plan office | Owner + colleague side by side is OK, strangers behind are not |
| **Contributor** | Open-source dev | Clear architecture, replay harness, pluggable models |

## 5. Key scenarios

| # | Scenario | Expected behaviour |
|---|---|---|
| S1 | Owner works alone | No action. Lowest-cost processing mode. |
| S2 | Authorised user sits next to owner | No action. |
| S3 | Stranger walks past in background, not looking | No action (optional "notice" indicator). |
| S4 | Stranger stops behind owner and looks at screen | Protection chain within the latency target. |
| S5 | Several people, at least one unauthorised and looking | Protection chain. |
| S6 | Owner leaves, nobody present | Separate "owner absent" setting: off / blur / lock, after a configurable delay. |
| S7 | Someone holds a photo or phone showing the owner | Treated as spoof: owner is not considered present; optional alert. |
| S8 | Camera unplugged / blocked / in use elsewhere (e.g. taken by a video call) | Popup explains the cause and offers Pause / Switch camera / Keep protecting; the loss policy (default: protect) applies until the user picks. |
| S9 | Stranger leaves, owner remains | Restore after the recovery delay. |
| S10 | Owner needs to override immediately | Hotkey + Windows Hello or PIN unlocks and pauses for N minutes. |

## 6. Feature list

Priority: **P0** = MVP, **P1** = next release, **P2** = later, **R** = research.

### 6.1 Camera
| ID | Feature | Pri |
|---|---|---|
| CAM-1 | List all cameras and pick one (built-in, USB, virtual) | P0 |
| CAM-2 | Remember choice by stable device ID, not index | P0 |
| CAM-3 | Hot-plug: refresh list, reconnect when the chosen camera returns | P0 |
| CAM-4 | Camera-loss policy: fail-safe protect / fall back to another camera / notify only | P0 |
| CAM-4b | Camera-loss popup with cause (unplugged, busy in another app, covered, permission) and actions (Pause, Switch camera, Keep protecting) | P0 |
| CAM-5 | Per-camera resolution, FPS, rotation, mirror | P0 |
| CAM-6 | Per-camera placement calibration ("look at the 4 corners") for external cameras | P0 |
| CAM-7 | Pick a separate IR source for liveness | P1 |
| CAM-8 | Depth camera plug-in (RealSense / Orbbec) | P2 |

### 6.2 Identity
| ID | Feature | Pri |
|---|---|---|
| ID-1 | Owner registration with guided multi-pose enrolment | P0 |
| ID-2 | Add/remove authorised users (owner approval required) | P0 |
| ID-3 | Encrypted local face template store; "delete all biometric data" | P0 |
| ID-4 | Re-enrol / add samples (glasses, lighting) | P0 |
| ID-5 | Owner verification for settings changes (Windows Hello or app PIN) | P0 |
| ID-6 | Optional stronger recogniser models (user-downloaded, own licence) | P1 |

### 6.3 Liveness / anti-spoofing
| ID | Feature | Pri |
|---|---|---|
| LV-1 | Passive RGB anti-spoof model per tracked face | P0 |
| LV-2 | Temporal cues (blink, micro-motion) to reject static photos | P1 |
| LV-3 | IR liveness when an IR stream is available | P1 |
| LV-4 | Active challenge on demand ("turn head left") for enrolment/unlock | P1 |
| LV-5 | Depth-based liveness via depth camera plug-in | P2 |

### 6.4 Attention and triggering
| ID | Feature | Pri |
|---|---|---|
| AT-1 | Multi-face detection and tracking with stable track IDs | P0 |
| AT-2 | Head-pose based "facing the screen" test, calibrated per camera | P0 |
| AT-3 | Distance gate (ignore faces too far away) | P0 |
| AT-4 | Dwell time before a stranger counts as an onlooker | P0 |
| AT-5 | Activation delay, recovery delay, minimum protected time | P0 |
| AT-6 | Owner-absent setting: off / blur / lock, with its own delay | P0 |
| AT-7 | Eye-gaze refinement | P1 |
| AT-8 | Sensitivity presets (Relaxed / Balanced / Paranoid) | P0 |

### 6.5 Protection actions
| ID | Feature | Pri |
|---|---|---|
| AC-1 | Live blur of all displays | P0 |
| AC-2 | Solid colour overlay (configurable colour/opacity) | P0 |
| AC-3 | Custom image overlay (user image, fit/fill/tile) | P0 |
| AC-4 | Full-screen or banner warning message (custom text) | P0 |
| AC-5 | Minimise all windows / show desktop | P0 |
| AC-6 | Lock workstation | P0 |
| AC-7 | Pause media / mute audio | P1 |
| AC-8 | Blur/overlay one display only | P1 |
| AC-9 | Blur/overlay a selected window or app (follows the window) | P1 |
| AC-10 | Minimise selected apps only | P1 |
| AC-11 | Sound alert / toast to owner | P1 |
| AC-12 | Intruder snapshot: save the camera frame when an onlooker/spoof triggers (encrypted, auto-expire, viewer behind owner verification) | P0 |

### 6.6 Action engine and profiles
| ID | Feature | Pri |
|---|---|---|
| AE-1 | Chain several actions with explicit order | P0 |
| AE-2 | Per-step delay | P0 |
| AE-3 | Different chains per threat level (notice / onlooker / spoof / camera lost / owner absent) | P0 |
| AE-4 | Auto-restore when the threat clears, in reverse order | P0 |
| AE-5 | Profiles (create, duplicate, switch from tray) | P0 |
| AE-6 | Automatic profile switching (by foreground app, schedule, network) | P1 |

### 6.7 Selective content protection (post-MVP)
| ID | Feature | Pri |
|---|---|---|
| SC-1 | Mask text regions in allow-listed apps using UI Automation | P2 |
| SC-2 | OCR-based masking for any window | P2 |
| SC-3 | Sensitive-content classifiers (regex + NER) | P2 |
| SC-4 | Browser extension that masks inside web pages (Gmail, web chat) | P2 |
| SC-5 | Per-app masking rules ("hide message bodies, keep sidebar") | P2 |

### 6.8 App shell
| ID | Feature | Pri |
|---|---|---|
| APP-1 | Tray icon with status, pause/resume, profile switch | P0 |
| APP-2 | Settings window with live camera preview and debug overlay | P0 |
| APP-3 | Global hotkeys (pause, panic-protect, unlock) | P0 |
| APP-4 | Start with Windows | P0 |
| APP-5 | First-run wizard (camera, model download, owner enrolment, test) | P0 |
| APP-6 | Local event log (no images by default) | P0 |
| APP-7 | English + Vietnamese UI | P1 |
| APP-8 | Signed installer, auto-update | P2 |

## 7. Approach per hard problem (summary)

Full detail in the [SRS](SRS.md). This is the product-level stance.

### 7.1 "3D" face verification: what is honest
- A standard RGB webcam gives **no real depth**. Passive anti-spoof models are trained
  to spot print/replay artefacts. They stop casual attacks but can be beaten by
  high-quality replays and masks.
- IR cameras (Windows Hello) give a strong signal: phone/monitor screens barely show
  up under active IR, and paper and skin reflect differently.
- Real 3D needs a depth camera (structured light / ToF / stereo).
- Monocular depth estimation is **not** a liveness signal: it "hallucinates" depth
  for flat photos too.
- Windows Hello itself is the strongest check available, but it is only exposed as a
  yes/no consent prompt for the signed-in user, so it is used for **unlock**, not
  for continuous monitoring.

fasec therefore reports a **liveness tier** for every session (RGB / RGB+temporal /
IR / Depth) and shows it in the UI, so users know how strong their setup is.

### 7.2 Granular protection: what is honest
- fasec cannot change what another app renders without injecting code into it.
  It **can** draw its own windows on top of precisely located regions.
- Region location comes from UI Automation (cheap, exact, app-dependent), OCR
  (universal, slower) or a browser extension (exact for web apps).
- During scrolling or animation, masks trail the content by one or more frames.
  This leak window cannot be avoided, so granular masking is a convenience layer and
  full-screen protection stays the safe default.

## 8. Feasibility matrix

| Capability | Classification | Mechanism / note |
|---|---|---|
| Camera selection, hot-plug | Fully feasible | Media Foundation enumeration, DeviceWatcher |
| Real-time face detection + recognition | Fully feasible | YuNet + SFace via ONNX Runtime (DirectML/CPU) |
| Multi-person tracking | Fully feasible | IoU/Kalman tracker (SORT-style) |
| Head-pose "facing screen" | Feasible with limitations | Accuracy drops beyond ~60° yaw and in low light; external cameras need calibration |
| Eye gaze "reading the screen" | Feasible with limitations | Unreliable at > ~1 m and at low resolution; a refinement only |
| Distinguish photo / phone screen with RGB webcam | Feasible with limitations | Passive models + temporal cues; not robust to high-quality attacks |
| Robust liveness | Requires specialised hardware | IR camera (common on business laptops) or depth camera |
| True 3D face geometry | Requires specialised hardware | Depth camera |
| Continuous Windows Hello verification | Not practically feasible | API only offers a consent prompt; no frames, no continuous result |
| Live blur of whole display(s) | Feasible with limitations | DWM acrylic backdrop; falls back to capture-based blur or opaque when transparency is off |
| Solid/image/warning overlay | Fully feasible | Top-most layered windows per monitor |
| Minimise / lock | Fully feasible | Win32 Shell / `LockWorkStation` |
| Overlay above Start menu, Alt-Tab, toasts, secure desktop | Feasible with limitations | Those surfaces sit in higher z-order bands; `uiAccess` (signed, installed) raises us, the secure desktop is unreachable |
| Blur a specific window, follow it | Feasible with limitations | Window tracking via WinEvent hooks; slight lag on move |
| Mask text inside native apps | Requires app-specific integration | UI Automation: good for Win32/WPF/UWP/Chromium/Electron, poor for Qt/custom/GPU UIs |
| Mask text inside any app | Feasible with limitations | OCR on captured frames; latency and CPU cost, lag on scroll |
| Mask content inside web apps | Requires app-specific integration | Browser extension + native messaging |
| Hide notification toasts | Feasible with limitations | Can toggle what fasec controls; Windows Focus/DND has no stable public API |
| Modify GPU-rendered content of other apps | Not practically feasible | Needs injection/Present hooking: fragile, flagged by AV/anti-cheat |
| Intercept DWM composition | Not practically feasible | Undocumented, breaks on updates, needs elevated tampering |

## 9. Reuse instead of build

| Need | Reuse |
|---|---|
| Inference runtime | `ort` (ONNX Runtime, DirectML EP) |
| Camera | `nokhwa` + `windows` crate (Media Foundation, `MediaFrameReader` for IR) |
| Face detection | YuNet (OpenCV Zoo, MIT) |
| Face recognition | SFace (OpenCV Zoo, Apache-2.0); optional user-supplied ArcFace |
| Anti-spoofing | Silent-Face-Anti-Spoofing MiniFASNet (Apache-2.0) |
| Head pose / landmarks | 6DRepNet (MIT) or MediaPipe Face Landmarker (Apache-2.0), ONNX conversions |
| Gaze | MediaPipe iris landmarks, L2CS-Net (MIT) |
| Tracking | SORT-style tracker (own small impl) or `similari` |
| Screen capture | `windows-capture` (Windows Graphics Capture) |
| UI Automation | `uiautomation` crate |
| OCR | `Windows.Media.Ocr`; PaddleOCR via `oar-ocr`; `ocrs` |
| PII detection | Regex sets; GLiNER via `gline-rs`; Presidio rules as reference |
| Blur backdrop | DWM backdrop / `window-vibrancy` |
| Tray / hotkeys | `tray-icon`, `global-hotkey` |
| UI | `egui` / `eframe` |
| Secrets | DPAPI via `windows` crate; `argon2` for PIN |

Model weights are **never committed** to the repo. A manifest pins URL + SHA-256 and
the app downloads on first run (or the user drops files in manually for offline install).

## 10. Success metrics

| Metric | Target (MVP) | Measured by |
|---|---|---|
| Onlooker reaction latency (p95) | ≤ 500 ms | Replay harness + manual stopwatch test |
| False triggers, owner alone | ≤ 1 per 8 h | Dogfooding log |
| Owner false reject → protection (FRR at decision level) | ≤ 1 per 8 h | Dogfooding log |
| Stranger accepted as owner (FAR, single frame) | ≤ 0.1 % at chosen threshold | Offline eval on own + public set |
| Print / phone-replay attack rejection (RGB tier) | ≥ 90 % of attempts in test set | Recorded attack set |
| CPU average, steady state | ≤ 10 % (4-core laptop) | Perf counters |
| RAM | ≤ 400 MB | Perf counters |
| Time-to-protect after camera loss | ≤ 1 s | Unplug test |

## 11. Privacy and trust principles

1. Local-only processing. The only network access is downloading models (hash-pinned)
   and, later, an opt-in update check.
2. Store templates (embeddings), not face images. Enrolment images are discarded
   unless the user opts in to keeping them for re-training. The one exception is
   intruder snapshots (AC-12): encrypted, auto-deleted after the retention period
   (default 30 days), viewable only after owner verification, and switchable off.
3. Templates are encrypted at rest with DPAPI (bound to the Windows user).
4. One-click "delete all biometric data".
5. No telemetry. Logs are local, contain no images, and are rotated.
6. Authorised users must consent during their own enrolment (they are in front of the camera).
7. The camera LED is on while monitoring. The UI says so plainly; the app never tries to hide it.

## 12. Risks (product level)

| Risk | Impact | Mitigation |
|---|---|---|
| False triggers annoy users and they uninstall | High | Dwell + hysteresis, sensitivity presets, replay-based tuning |
| Users over-trust RGB liveness | High | Show the liveness tier; clear docs |
| Overlay does not cover Start/Alt-Tab/toasts | Medium | Pair with minimise; `uiAccess` build later |
| Camera conflicts with Teams/Zoom on some drivers | Medium | Detect and surface; pause-while-in-call option |
| Model licence confusion | Medium | Default to permissive models; optional models downloaded by the user |
| Battery drain | Medium | Adaptive FPS, pause on lock / display off / battery saver |
| Acrylic blur unavailable (transparency off, RDP) | Low | Capture-based blur fallback, then opaque (fails safe) |
| Intruder snapshots capture third parties without consent (privacy law, e.g. GDPR, Vietnam Decree 13/2023) | Medium | Clear notice in wizard, can be switched off, short retention, encrypted, never uploaded |

## 13. Decisions

| # | Question | Decision (2026-11-21) |
|---|---|---|
| D1 | Owner absent: lock or blur? | Its own setting: off / blur / lock, with its own delay. Default: off. |
| D2 | Camera lost during a video call? | Show a popup with the cause and Pause / Switch camera / Keep protecting. Loss policy (default protect) applies until the user picks. |
| D3 | Keep intruder snapshots? | Yes. Encrypted, 30-day default retention, owner verification to view, can be switched off. |
| D4 | Minimum Windows version? | Windows 10 2004 (build 19041), needed for `WDA_EXCLUDEFROMCAPTURE`. Tested on Windows 10 22H2 and Windows 11. |
