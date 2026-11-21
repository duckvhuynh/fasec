# SP-1b: Capture-based blur fallback

**Question:** when DWM blur (SP-1) is unavailable, what does it cost on the CPU to capture the
monitor with Windows Graphics Capture (WGC), blur it ourselves and show it in our own
capture-excluded overlay? Do we need a GPU blur?

**Program:** `crates/fasec-overlay/examples/sp1b_capture_blur.rs`

```
cargo run --release -p fasec-overlay --example sp1b_capture_blur -- [--monitor 0] [--seconds 15] [--downscale 8] [--no-exclude]
```

- Overlay: one `WS_POPUP` covering the monitor, `WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
  | WS_EX_NOACTIVATE`, `WDA_EXCLUDEFROMCAPTURE` (the program refuses to run if that fails),
  top-most re-asserted every 250 ms. It stays invisible until the first blurred frame is presented.
- Capture: `windows-capture` 2.0.1 (`start_free_threaded`, its own thread), `Bgra8`, cursor
  off, capture border off where the OS supports turning it off. The default
  `MinUpdateInterval` is used, so there is no frame cap.
- Blur, CPU, single-threaded, on the capture thread: box-average `d`×`d` blocks
  (`--downscale d`), then bilinear upscale back to full resolution.
- Present, on the UI thread: copy into a top-down 32-bit DIB section, then
  `UpdateLayeredWindow(ULW_OPAQUE)`.
- Printed every second: capture FPS, readback ms (`Frame::buffer`: staging texture,
  `CopyResource`, `Map`), blur ms, present ms (DIB copy + `UpdateLayeredWindow`), process CPU %
  (`GetProcessTimes` delta / wall time / logical cores).
- Self-capture check: a 48×48 magenta square is drawn at (64,64) on top of the blurred image.
  Every captured frame is checked for it. If the overlay were in its own capture (a "hall of
  mirrors"), the square would show up in the captured frame. `--no-exclude` skips
  `WDA_EXCLUDEFROMCAPTURE` and serves as the positive control for this check.
- Esc (low-level keyboard hook, not swallowed, removed by a Drop guard) or `--seconds` closes
  the program. It also exits if the capture thread dies.

Use `--release`. The numbers below are release builds.

## Automated run on the dev machine (2026-11-21, by the implementer, screen not observed)

Machine: Intel Core i7-14650HX (24 logical cores), NVIDIA RTX 5060 Laptop + Intel UHD,
Windows 11 build 26300. Monitor 0: `\\.\DISPLAY1`, 1920×1200 native, 165 Hz, 100 % DPI.
Each run: `--seconds 10`, exit code 0, `keyboard hook removed`. The desktop behind the overlay
was whatever was open at the time (a terminal session). Nobody interacted with it.

Native resolution, 1920×1200. Means of the per-second lines:

| `--downscale` | capture FPS | readback ms | blur ms | present ms | process CPU % (of 24 logical) | ≈ logical cores | marker in capture |
|---|---|---|---|---|---|---|---|
| 4  | 45.1 | 2.4 | 12.1 | 2.1 | 2.8 | 0.67 | 0 / 456 |
| 8  | 47.4 | 2.3 | 11.6 | 2.2 | 2.9 | 0.70 | 0 / 479 |
| 16 | 47.3 | 2.5 | 11.5 | 2.2 | 2.8 | 0.67 | 0 / 478 |

Positive control, `--downscale 8 --no-exclude --seconds 3`: the marker was found in **140 / 142**
captured frames. The two misses were in the first second, presumably frames captured before
the overlay was first presented. So the check works. With `WDA_EXCLUDEFROMCAPTURE` the overlay is **not** in its own
capture: there is no hall of mirrors (0 marker frames in every excluded run). This was checked
programmatically. Nobody looked at the screen.

The first series of runs (before readback timing was added) matched these numbers. The one
exception was a noisy `--downscale 16` run (32–40 fps, blur 15–19 ms, present 4.3–5.6 ms). Present
cost does not depend on `--downscale`, so that run reflects load elsewhere on the machine, not
the setting.

**1080p:** not measured. This machine's only monitor is 1920×1200, and changing the display mode
was out of scope for an unattended run. Every stage is a full-resolution pass, so the cost scales
with pixel count. 1920×1080 is 0.90× the pixels, which gives an *estimate* of blur ≈ 10.4 ms,
readback ≈ 2.1 ms and present ≈ 2.0 ms per frame. To measure it, run the same command on a 1080p
monitor and add a row.

### Observations

- **`--downscale` barely matters** (blur 11.5–12.1 ms for 4/8/16). The cost is set by the two
  full-resolution passes: reading 9.2 MB in the box pass and writing 9.2 MB in the bilinear
  pass. The small intermediate image doesn't move the total.
- **Per frame, the CPU does about 16 ms of work** (readback + blur on the capture thread,
  present on the UI thread). At about 47 fps that comes to about 0.7 of a logical core,
  continuously, for one 1920×1200 monitor. On a 4-core / 8-thread laptop that would show as
  about 9 % total CPU, more if its cores are slower than the i7-14650HX's P-cores.
- **Capture FPS was 45–50, not 165.** Capture thread work per frame is about 14 ms, which would
  allow about 70 fps, so processing is not the only limit. WGC delivers frames when the
  composed desktop changes. Whether our own (excluded) overlay updates trigger new frames was
  not determined. If they do, the pipeline self-clocks and never idles, even on a static
  desktop.
- **Latency:** the blurred image is at least one readback+blur+present (about 16 ms) behind the
  real screen, plus WGC delivery. Fast-moving content behind the overlay will visibly lag.
- Visual quality (how blurred d=4 looks, whether box + bilinear shows blockiness at d=16) was
  **not observed**. Check by eye.

## Decision: GPU blur needed?

_To be decided by a human._ Data for the decision:

- CPU fallback as built: about 0.7 logical core per 1920×1200 monitor at about 47 fps, roughly
  linear in pixel count and monitor count. A 4K monitor would be about 3.6× that.
- The cheap CPU-side levers don't reduce the full-resolution passes. A bigger downscale doesn't
  help, as measured. What would help: a frame cap (`MinUpdateInterval`, for example 15–30 fps),
  multi-threading (lowers latency, not CPU), or downscaling on the GPU before readback, so that
  only the small image crosses to the CPU and gets blurred. The last option is already GPU work.
- Implementer's suggestion: yes, do downscale and blur on the GPU (a D3D11 compute or pixel
  shader on the WGC texture, presented through DirectComposition or a swap chain). Keep the CPU
  path only as a last-resort fallback with a frame cap. Not decided here.
