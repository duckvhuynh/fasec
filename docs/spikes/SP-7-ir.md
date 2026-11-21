# SP-7: Reading Windows Hello IR frames

**Question:** can a normal desktop app read frames from a Windows Hello IR camera, without
admin rights and without breaking Windows Hello, so a later version can use IR for stronger
liveness? Which sharing mode works, what format do the frames have, and can we tell
IR-illuminated frames from ambient ones?

**Program:** `crates/fasec-camera/examples/ir_probe.rs` (WinRT `Windows.Media.Capture.Frames`:
`MediaFrameSourceGroup::FindAllAsync`, then `MediaCapture` with `SourceGroup`, `SharingMode`,
`MemoryPreference = Cpu` and `StreamingCaptureMode = Video`, then `MediaFrameReader`).
It saves nothing to disk. Frames are only reduced to a mean value in memory.

## How to run

```powershell
cargo run -p fasec-camera --example ir_probe                    # list groups and sources
cargo run -p fasec-camera --example ir_probe -- read <group-index>
```

`list` prints every source group with each source's kind (`Color` / `Infrared` / `Depth` / ...),
stream type, source id and device id. `read` picks the group's first `Infrared` source. It
tries `SharedReadOnly` first and, only if that fails, `ExclusiveControl`. It reads 30 frames and
prints the resolution, pixel format, mean intensity and `InfraredMediaFrame.IsIlluminated` of
each frame, then `RESULT: ... with <mode>` or the error for each mode.

Checks, one table row each:

1. `read` on the IR-only group, and on any Color+IR group.
2. `read` while Windows Hello is in use (Settings > Accounts > Sign-in options > Facial
   recognition > "Improve recognition" or similar). Do both keep working?
3. Lock the screen (Win+L) while `read` runs, then unlock with your face. Does Hello still work?
4. `read` while a call app uses the RGB camera (see SP-5).

## Results (fill in by hand)

| Machine | Windows build | Camera / IR device | Check | Mode that worked | Format | Illuminated/ambient alternate? | Hello still OK? | Notes |
|---|---|---|---|---|---|---|---|---|
| | | | 1 IR group / Color+IR group | | | | | |
| | | | 2 during Hello enrolment | | | | | |
| | | | 3 Hello unlock while reading | | | | | |
| | | | 4 call app holds RGB | | | | | |

## Decision

_(empty, to be filled in after the results)_

## Observed on dev machine (2026-11-21, by the implementer)

Machine: ASUS laptop, Windows 11 `10.0.26300.9974`, built-in `ASUS FHD webcam` (`MI_00`) and
`ASUS IR camera` (`MI_02`) on one USB device `VID_3277&PID_0096`. Not elevated. Debug build.

`ir_probe` (list):

```
3 source group(s)
[0] ASUS FHD webcam
    group id: \\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\GLOBAL
    - Color / VideoRecord
      source id: Source#0@\\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\GLOBAL
      device id: \\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\GLOBAL
[1] Rts-DMFT-Group
    group id: \\?\SWD#SGDEVAPI#57AF77F27689C565E4AC351FCCE6E78DA6987E7EB1F4ACAD18FD3825DA62731B#{669c7214-0a88-4311-a7f3-4e79820e33bd}\57AF77F27689C565E4AC351FCCE6E78DA6987E7EB1F4ACAD18FD3825DA62731B
    - Color / VideoRecord
      source id: Source#0@\\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\GLOBAL
      device id: \\?\USB#VID_3277&PID_0096&MI_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\GLOBAL
    - Infrared / VideoRecord
      source id: Source#1@\\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
      device id: \\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
[2] ASUS IR camera
    group id: \\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
    - Infrared / VideoRecord
      source id: Source#0@\\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
      device id: \\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
```

`ir_probe read 2` (IR-only group), complete output:

```
group: ASUS IR camera
Infrared source: Source#0@\\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
-- sharing mode SharedReadOnly
current format: L8 640x360 @ 15/1
frame  1 + 248 ms: 640x360 Gray8 mean 71.4 illuminated true
frame  2 + 309 ms: 640x360 Gray8 mean 0.0 illuminated false
frame  3 + 375 ms: 640x360 Gray8 mean 70.1 illuminated true
frame  4 + 441 ms: 640x360 Gray8 mean 0.0 illuminated false
frame  5 + 500 ms: 640x360 Gray8 mean 60.4 illuminated true
frame  6 + 581 ms: 640x360 Gray8 mean 0.0 illuminated false
frame  7 + 646 ms: 640x360 Gray8 mean 50.3 illuminated true
frame  8 + 712 ms: 640x360 Gray8 mean 0.0 illuminated false
frame  9 + 773 ms: 640x360 Gray8 mean 47.6 illuminated true
frame 10 + 854 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 11 + 919 ms: 640x360 Gray8 mean 44.8 illuminated true
frame 12 + 984 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 13 +1045 ms: 640x360 Gray8 mean 44.0 illuminated true
frame 14 +1126 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 15 +1191 ms: 640x360 Gray8 mean 43.2 illuminated true
frame 16 +1256 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 17 +1320 ms: 640x360 Gray8 mean 42.4 illuminated true
frame 18 +1397 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 19 +1462 ms: 640x360 Gray8 mean 42.1 illuminated true
frame 20 +1528 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 21 +1593 ms: 640x360 Gray8 mean 41.9 illuminated true
frame 22 +1652 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 23 +1733 ms: 640x360 Gray8 mean 41.7 illuminated true
frame 24 +1797 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 25 +1863 ms: 640x360 Gray8 mean 41.7 illuminated true
frame 26 +1929 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 27 +2009 ms: 640x360 Gray8 mean 41.5 illuminated true
frame 28 +2073 ms: 640x360 Gray8 mean 0.0 illuminated false
frame 29 +2134 ms: 640x360 Gray8 mean 41.4 illuminated true
frame 30 +2197 ms: 640x360 Gray8 mean 0.0 illuminated false
30 frames in 2199 ms, 15 illuminated
RESULT: 30 Infrared frames read with SharedReadOnly
```

`ir_probe read 1` (Color+IR group `Rts-DMFT-Group`), first and last lines:

```
group: Rts-DMFT-Group
Infrared source: Source#1@\\?\USB#VID_3277&PID_0096&MI_02#6&ead33bb&0&0002#{24e552d7-6523-47f7-a647-d3465bf1f5ca}\GLOBAL
-- sharing mode SharedReadOnly
current format: L8 640x360 @ 15/1
frame  1 + 244 ms: 640x360 Gray8 mean 44.5 illuminated true
frame  2 + 310 ms: 640x360 Gray8 mean 0.0 illuminated false
...
frame 29 +2134 ms: 640x360 Gray8 mean 41.1 illuminated true
frame 30 +2201 ms: 640x360 Gray8 mean 0.0 illuminated false
30 frames in 2202 ms, 15 illuminated
RESULT: 30 Infrared frames read with SharedReadOnly
```

What this shows:

- **IR is readable without admin rights with `SharedReadOnly`.** `ExclusiveControl` was never
  needed, so it was not exercised.
- **Format:** `L8` (Gray8) 640x360 at 15 fps nominal. Frames arrive every 60-80 ms.
- **Frames alternate:** IR-illuminated (mean about 41-71, falling over the first ~1 s as exposure
  settles) and non-illuminated (mean 0.0 to one decimal, so the driver seems to blank or subtract
  ambient frames). `IsIlluminated` reports this correctly, so a liveness check can keep only
  the illuminated frames, which gives about 7.5 usable fps.
- **Pairing:** the `Rts-DMFT-Group` group pairs the RGB source (`MI_00`) and the IR source
  (`MI_02`) of the same USB device, and their instance ids share the parent `6&ead33bb&0`. This
  group is the natural RGB/IR pairing key. The IR-only group works too.
- **Independent of RGB use:** reading IR worked while another process held the RGB camera
  through nokhwa/MF (see SP-5).
- Not tested here: interaction with Windows Hello itself (checks 2 and 3), and whether the
  0.0 frames are truly all zeros (only the mean was computed).
