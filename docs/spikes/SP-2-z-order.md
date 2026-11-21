# SP-2: Overlay z-order

**Question:** which system surfaces and other "top-most" windows can appear above the fasec
overlay, given a `WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE` popup that re-asserts
`HWND_TOPMOST` every 250 ms (SRS FR-OV-002, FR-OV-007)? Anything that shows above the overlay can
leak screen content to an onlooker.

**Program:** same as SP-1, `crates/fasec-overlay/examples/sp1_overlay.rs`. Use `solid` so leaks
are obvious, with a long timeout and `--click-through` so you can interact with the desktop
behind it:

```
cargo run -p fasec-overlay --example sp1_overlay -- --mode solid --seconds 120 --click-through
```

Press Esc to close early (the key is passed through, so it also closes the Start menu etc.).
Note that the program does not yet hook `EVENT_SYSTEM_FOREGROUND` (FR-OV-007's second trigger);
only the 250 ms timer is tested here.

## Automated run on the dev machine (2026-11-21, by the implementer, screen not observed)

Windows build `26300.9974` (Windows 11), transparency effects on, 1 monitor 1920x1200 at 96 DPI.
`--mode solid --seconds 5` exited on its own after 5 s: 20 `HWND_TOPMOST` re-asserts, 0
`SetWindowPos` errors, `WDA_EXCLUDEFROMCAPTURE` ok, keyboard hook removed. None of the surfaces
below were opened during that run.

## Results (fill in by hand)

| Surface | Win10: above overlay? (y/n) | Win11: above overlay? (y/n) | Notes (flicker, only for <250 ms, etc.) |
|---|---|---|---|
| Start menu | | | |
| Alt-Tab | | | |
| Win+Tab (Task View) | | | |
| Action Center / Quick Settings / Notification Center | | | |
| Toast notification | | | |
| Task Manager with "Always on top" | | | |
| Full-screen video (browser F11 / Movies & TV) | | | |
| UAC prompt (secure desktop) | | | expected: y, the secure desktop is a different desktop |

## Conclusion

_To be written after the table is filled: which surfaces leak, whether any need extra handling
(e.g. `EVENT_SYSTEM_FOREGROUND` re-assert, band/UIAccess windows, or documenting as a known
limitation)._
