# SP-1b: Capture-based blur fallback

**Question:** when DWM blur (SP-1) is unavailable, what does it cost on the CPU to capture the
monitor with Windows Graphics Capture (WGC), blur it ourselves and show it in our own
capture-excluded overlay? Do we need a GPU blur?

**Program:** `crates/fasec-overlay/examples/sp1b_capture_blur.rs`

```
cargo run --release -p fasec-overlay --example sp1b_capture_blur -- [--monitor 0] [--seconds 15] [--downscale 8] [--no-exclude]
```

> **Warning:** while it runs, the whole monitor is covered by the blurred overlay and clicks do
> not reach the windows behind it. Press **Esc** to close it early. Otherwise it closes by
> itself after `--seconds`.

- Overlay: one `WS_POPUP` covering the monitor, `WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
  | WS_EX_NOACTIVATE`, `WDA_EXCLUDEFROMCAPTURE` (the program refuses to run if that fails),
  top-most re-asserted every 250 ms. It stays invisible until the first blurred frame is presented.
- Capture: `windows-capture` 2.0.1 (`start_free_threaded`, its own thread), `Bgra8`, cursor
  off, capture border off where the OS supports turning it off. The default
  `MinUpdateInterval` is used, so there is no frame cap.
- Blur, CPU, single-threaded, on the capture thread: box-average `d`×`d` blocks
  (`--downscale d`), then bilinear upscale back to full resolution. This is **naive scalar
  code** (no SIMD, per-channel bounds-checked indexing, per-pixel bilinear in the inner loop),
  written to get a number, not to be fast.
- Present, on the UI thread: copy into a top-down 32-bit DIB section, then
  `UpdateLayeredWindow(ULW_OPAQUE)`.
- Printed every second: capture FPS, readback ms (`Frame::buffer`: staging texture,
  `CopyResource`, `Map`), blur ms, present ms (DIB copy + `UpdateLayeredWindow`), process CPU %
  (`GetProcessTimes` delta / wall time / logical cores).
- Self-capture check: a 48×48 magenta square is drawn at (64,64) on top of the blurred image.
  Every captured frame is checked for it. If the overlay were in its own capture (a "hall of
  mirrors"), the square would show up in the captured frame. `--no-exclude` skips
  `WDA_EXCLUDEFROMCAPTURE` and serves as the positive control for this check.
- Esc (low-level keyboard hook, not swallowed) or `--seconds` closes the program. Both are
  checked on every frame as well as on the 250 ms timer, because `WM_TIMER`/`WM_QUIT` can be
  starved by a flood of frame messages. On exit the overlay is hidden and the hook removed
  *before* the capture is stopped, so a slow `stop()` cannot leave the screen covered. It also
  exits if the capture thread dies.

Use `--release`. The numbers below are release builds.

## Automated run on the dev machine (2026-11-21, by the implementer, screen not observed)

Machine: Intel Core i7-14650HX (24 logical cores), NVIDIA RTX 5060 Laptop + Intel UHD,
Windows 11 build 26300. Monitor 0: `\\.\DISPLAY1`, 1920×1200 native, 165 Hz, 100 % DPI.
Each run: `--seconds 10`, exit code 0, `keyboard hook removed`. All numbers are for the
naive scalar blur described above. The desktop behind the overlay
was whatever was open at the time (a terminal session). Nobody interacted with it.

Native resolution, 1920×1200. Means of the per-second lines (original box pass, with a per-pixel `x / d`):

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

After review, the box pass was rewritten row-wise (one accumulator per `d`-pixel chunk, which
removes the per-pixel `x / d` division) and measured again with the same 10 s runs:

| `--downscale` | capture FPS | readback ms | blur ms | present ms | process CPU % | marker in capture |
|---|---|---|---|---|---|---|
| 4  | 44.3 | 2.4 | 13.1 | 2.5 | 2.8 | 0 / 443 |
| 8  | 49.4 | 2.3 | 10.9 | 2.2 | 2.8 | 0 / 494 |
| 16 | 46.2 | 2.3 | 12.1 | 2.6 | 2.9 | 0 / 462 |

There was no measurable change: the 10.9–13.1 ms spread is within run-to-run noise (single
seconds ranged from 9.1 to 17.0 ms). The division was not the bottleneck, so the bilinear pass
probably dominates. This was not profiled.

**1080p: to be measured by a human.** This machine's only monitor is 1920×1200, and changing the
display mode is a system setting, which was out of scope for an unattended run. *Estimate
only*: every stage runs at full resolution and 1920×1080 is 0.90× the pixels, which gives blur
≈ 10.4 ms, readback ≈ 2.1 ms and present ≈ 2.0 ms per frame. To measure it, run the same command
on a 1080p monitor and fill in:

| `--downscale` | capture FPS | readback ms | blur ms | present ms | process CPU % | marker in capture |
|---|---|---|---|---|---|---|
| 4  | _to be measured by a human_ | | | | | |
| 8  | _to be measured by a human_ | | | | | |
| 16 | _to be measured by a human_ | | | | | |

### Observations

- **`--downscale` barely matters** (blur 11–13 ms for 4/8/16) because both passes, the box
  read and the bilinear write, loop over every full-resolution pixel no matter what `d` is.
  The cost is **compute-bound, not memory-bound**: 9.2 MB in about 12 ms is under 1 GB/s, far
  below memory bandwidth. It is the cost of this naive scalar code. Removing the per-pixel
  division gave no measurable gain, which suggests the bilinear inner loop dominates. That is
  a hypothesis: nothing was profiled.
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

- CPU fallback **as built (naive scalar, single-threaded)**: about 0.7 logical core per
  1920×1200 monitor at about 47 fps, roughly linear in pixel count and monitor count. A 4K
  monitor would be about 3.6× that. This is **not a ceiling for CPU blur**.
- Measured: a bigger downscale does not help, and neither does removing the `x / d` division.
  Untested CPU options that could plausibly cut the cost several times over: SIMD (or
  auto-vectorisable code without per-channel bounds checks), a separable bilinear upscale with
  precomputed row blends, or presenting the small image and letting a stretched blit or the
  compositor do the upscale. Options that cut CPU without speeding up the code: a frame cap
  (`MinUpdateInterval`, for example 15–30 fps), or downscaling on the GPU before readback, so
  that only the small image crosses to the CPU. Multi-threading lowers latency, not CPU.
- Implementer's suggestion: GPU downscale and blur (a D3D11 shader on the WGC texture) is the
  cleanest route, because the frame is already on the GPU. But the case for it is weaker than
  "CPU cannot do it": an optimised CPU path with a frame cap has not been tried. Not decided
  here.
