# SP-1: Overlay blur backends

**Question:** does a top-most, non-activating (`WS_EX_NOACTIVATE`), tool-window, capture-excluded
(`WDA_EXCLUDEFROMCAPTURE`) popup covering each monitor show a live blur of what is behind it on
Windows 10 and Windows 11, and what does it fall back to? (SRS FR-OV-001, -002, -004, -005, -008)

**Program:** `crates/fasec-overlay/examples/sp1_overlay.rs`

```
cargo run -p fasec-overlay --example sp1_overlay -- --mode <acrylic|backdrop|solid> [--seconds 15] [--click-through]
```

- `acrylic`: undocumented `user32!SetWindowCompositionAttribute(WCA_ACCENT_POLICY,
  ACCENT_ENABLE_ACRYLICBLURBEHIND)`, tint `0x99101010` (AABBGGRR), loaded dynamically.
- `backdrop`: `DwmSetWindowAttribute(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW)` then
  `DwmExtendFrameIntoClientArea(-1)` (Win11 22H2+, build 22621+). The frame is only extended if
  the backdrop call succeeds, so a failure stays opaque black instead of going transparent.
- `solid`: opaque black (class background brush).
- `--click-through` adds `WS_EX_LAYERED | WS_EX_TRANSPARENT` (+ `SetLayeredWindowAttributes(alpha 255)`).
- Re-asserts `HWND_TOPMOST` every 250 ms. Esc (low-level keyboard hook, not swallowed) or
  `--seconds` closes it.

Transparency effects = Settings > Personalization > Colors > Transparency effects
(`HKCU\Software\Microsoft\Windows\CurrentVersion\Themes\Personalize\EnableTransparency`; the
program prints it).

**Another app has focus** (table axis). The overlay never takes focus itself (`WS_EX_NOACTIVATE`),
so the axis is whether you have moved focus to a window behind it:
- **no**: launch the overlay and touch nothing. The terminal that launched it keeps focus.
- **yes**: after launch, give focus to a window behind the overlay. You can click into it, which
  only works with `--click-through` because without it the overlay eats the click. Or you can
  Alt-Tab to it, which works with or without `--click-through`.

Record whether `--click-through` was used in its own column. It makes the window layered
(`WS_EX_LAYERED`), which can change how DWM renders the blur, so the two cases must not be mixed.
If you test a row both with and without `--click-through`, copy the row and fill in one copy for
each. DWM may drop to a flat fallback colour once focus moves to another app.

## Automated run on the dev machine (2026-11-21, by the implementer, screen not observed)

Each mode was run with `--seconds 5`. Every run exited by itself after 5 s with exit code 0 and
printed `keyboard hook removed`.

| Item | Printed |
|---|---|
| Windows build | `26300.9974 (Windows 11)` |
| Transparency effects | `on` |
| Monitors | 1: `(0,0)-(1920,1200)`, 96 DPI (100%), primary |
| `WDA_EXCLUDEFROMCAPTURE` | `ok` in every mode |
| `--mode solid` | `Solid backend: applied`, 20 top-most re-asserts, 0 errors |
| `--mode acrylic` | `Acrylic backend: applied` (SetWindowCompositionAttribute returned TRUE), 20 re-asserts, 0 errors |
| `--mode backdrop` | `Backdrop backend: applied` (both DWM calls returned S_OK), 20 re-asserts, 0 errors |
| `--mode acrylic --click-through` | `Acrylic backend: applied`, 20 re-asserts, 0 errors |

No API returned an error. "Applied" only means the call succeeded. Whether a blur is actually
visible has to be checked by eye in the table below.

## Results (fill in by hand)

Mark each cell `y` / `n`. **live blur?** = content behind is visibly blurred and updates live.
**solid fallback?** = flat opaque/tinted colour, nothing behind readable. **transparent (bad)?** =
content behind is readable (fails open, violates FR-OV-005).

| OS (build) | Mode | Transparency | Another app has focus | `--click-through` (y/n) | live blur? | solid fallback? | transparent (bad)? | Notes |
|---|---|---|---|---|---|---|---|---|
| Win10 | acrylic | on | no | | | | | |
| Win10 | acrylic | on | yes | | | | | |
| Win10 | acrylic | off | no | | | | | |
| Win10 | acrylic | off | yes | | | | | |
| Win10 | backdrop | on | no | | | | | expected: unsupported, prints failure, stays black |
| Win10 | backdrop | on | yes | | | | | |
| Win10 | backdrop | off | no | | | | | |
| Win10 | backdrop | off | yes | | | | | |
| Win11 (26300) | acrylic | on | no | n | y | n | n | 2026-11-22, owner watched: content behind blurred and updating |
| Win11 (26300) | acrylic | on | yes | y | y | n | n | 2026-11-22: clicked into a window behind; blur stayed live |
| Win11 (26300) | acrylic | off | no | | | | | |
| Win11 (26300) | acrylic | off | yes | | | | | |
| Win11 (26300) | backdrop | on | no | n | n | y | n | 2026-11-22: solid colour only — system backdrop is not drawn for an inactive (NOACTIVATE) window |
| Win11 (26300) | backdrop | on | yes | | | | | |
| Win11 (26300) | backdrop | off | no | | | | | |
| Win11 (26300) | backdrop | off | yes | | | | | |

Also worth noting per row: does a screenshot (Win+Shift+S, OBS, Teams share) show the overlay?
It should not, because of `WDA_EXCLUDEFROMCAPTURE`. Is there any lag when you drag a window
behind the overlay?

## Conclusion

Win11 (2026-11-22, one machine, transparency on): **acrylic** (`SetWindowCompositionAttribute`,
`ACCENT_ENABLE_ACRYLICBLURBEHIND`) gives a live blur on a top-most, non-activating,
capture-excluded window, also while another app has focus and with click-through. The documented
Win11 **backdrop** API falls back to a solid colour for such a window, so it is not usable for the
overlay. Decision for M4: acrylic is backend 1, capture-based blur (SP-1b) backend 2, solid backend 3
(FR-OV-004). Still open: Win10, transparency effects off, and multi-monitor/mixed DPI.
