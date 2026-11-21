# Spikes (M0)

Each spike has a program and a doc with a results table. "Measured by agent" means the parts a
script can check were run on the dev laptop (Windows 11 26300, i7-14650HX, RTX 5060, ASUS FHD
webcam + IR camera) with nobody watching the screen. The rows that need eyes on the screen or
other hardware are still empty, and no spike decision has been signed off.

| Spike | Doc | Program | Status | Headline |
|---|---|---|---|---|
| SP-1 Overlay blur backends | [SP-1](SP-1-overlay-blur.md) | `fasec-overlay` example `sp1_overlay` | needs a human run | Solid, acrylic and Win11 backdrop all apply without API errors and the overlay is capture-excluded; whether the blur is actually visible (Win10/Win11, transparency on/off) has not been looked at. |
| SP-1b Capture-based blur | [SP-1b](SP-1b-capture-blur.md) | `fasec-overlay` example `sp1b_capture_blur` | measured by agent; 1080p run and GPU decision need a human | Naive scalar CPU blur costs ~0.7 logical core per 1920×1200 monitor at ~47 FPS (readback 2.3 + blur 11–13 + present 2.2 ms); no hall of mirrors (marker in 0 captured frames). |
| SP-2 Z-order | [SP-2](SP-2-z-order.md) | `sp1_overlay --mode solid` | needs a human run | Top-most re-assert every 250 ms runs with 0 errors; which system surfaces (Start, Alt-Tab, toasts, UAC…) show above the overlay is not yet tested. |
| SP-3 `ort` + DirectML | (benches) | `cargo bench -p fasec-vision` | measured by agent | 720p, 1 face, YuNet@320: ~15 ms/frame on DirectML, ~30 ms on CPU when recognition + liveness run; DirectML is the default. Numbers in [ROADMAP M0](../ROADMAP.md). |
| SP-4 Head pose comparison | none | none | not started | Not done. M1 uses the 5-point coarse pose (uncalibrated, pitch biased +4..13°) behind `PoseEstimator`. |
| SP-5 Camera sharing | [SP-5](SP-5-camera-sharing.md) | `fasec-camera` example `camera_probe` | measured by agent (second probe as "other app"); Teams/Zoom need a human run | The RGB camera is not shared across processes: the second opener gets an explicit `Busy`, the first keeps streaming. IR can still be read with `SharedReadOnly`. |
| SP-6 nokhwa | [SP-6](SP-6-nokhwa.md) | `camera_probe list / watch / run` | measured by agent; replug, other port and a second camera need a human run | nokhwa (MF) works: the id is the MF symbolic link (lowercased to match WinRT), hot-plug ids match `list()`; IR cameras are not visible to it. |
| SP-7 Windows Hello IR | [SP-7](SP-7-ir.md) | `fasec-camera` example `ir_probe` | measured by agent; Hello interaction needs a human run | IR frames read without admin via WinRT `SharedReadOnly`: L8 640×360 @15, illuminated and blank frames alternate (`IsIlluminated` is correct). |

How to run the programs: see the [README](../../README.md#spike-programs) and each doc.
`sp1_overlay` and `sp1b_capture_blur` cover the whole screen; Esc closes them.
