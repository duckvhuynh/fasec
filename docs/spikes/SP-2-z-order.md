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
| Start menu | | y | 2026-11-22, Win11 26300, acrylic overlay |
| Alt-Tab | | y | 2026-11-22; the switcher shows live window thumbnails, so content can leak |
| Win+Tab (Task View) | | y | 2026-11-22; shows window thumbnails |
| Action Center / Quick Settings / Notification Center | | | not tried |
| Toast notification | | | not tried |
| Task Manager | | n | 2026-11-22; stayed under the blur ("Always on top" state not recorded) |
| Full-screen video (browser F11 / Movies & TV) | | | |
| UAC prompt (secure desktop) | | | expected: y, the secure desktop is a different desktop |

## Conclusion

Win11 (2026-11-22): Start, Alt-Tab and Task View draw above a normal top-most overlay, as
expected from their higher z-order bands; Alt-Tab and Task View show live thumbnails, so they can
reveal content. Ordinary windows (Task Manager here) stay under it. For M4: pair blur with
`minimize_all` in stricter profiles, document the limitation (SRS §9 item 2), and keep the
`uiAccess` build (FR-OV-010, M12) as the real fix. Notification panel, toasts, full-screen video
and UAC are still untested.
