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
| ASUS laptop | 10.0.26300.9974 | ASUS IR camera | 1 IR group / Color+IR group | SharedReadOnly | L8 640x360 @ 15 | yes | not checked | both groups work (below) |
| ASUS laptop | 10.0.26300 | ASUS IR camera | 2 during Hello enrolment | | | | pending owner check | |
| ASUS laptop | 10.0.26300 | ASUS IR camera | 3 Hello unlock while reading | | | | pending owner check | |
| ASUS laptop | 10.0.26300.9974 | ASUS IR camera | 4 call app holds RGB | SharedReadOnly | L8 640x360 @ 15 | yes | not checked | another process held RGB through nokhwa/MF |

## Decision

Decided 2026-11-22 (Task 7) from recordings of the owner on the dev machine (below). One
subject, one device, one room. Only numbers are kept here: the recordings stay on that PC.

**IR is read with `SharedReadOnly`, and IR liveness is on by default (`[liveness] ir =
"prefer"`).** The thresholds in `IrLivenessConfig::default()` are set from the table below. Each
constant's comment names the measurement it came from.

**Owner live check (2026-11-22).** `live_debug --gallery` on the dev machine, debug build, face
about 240–276 px wide in the RGB frame (normal laptop distance). The owner was shown `Known` as
owner, match score 0.85. Liveness tier `Ir`, IR verdict `Live` 0.80, no usable ambient frame. IR
source opened `SharedReadOnly`, health `Ok`, about 3 lit frames/s.

**Windows Hello coexistence (checks 2 and 3): pending owner check.** It is not verified that
Windows Hello sign-in keeps working while fasec holds the IR camera.

### Recordings and method

`ir_debug --record <label>` on the ASUS laptop (RGB 1280×720, IR 640×360 lit frames, about 2.2
recorded steps/s in a debug build). Labels: `real-30cm`, `real-50cm`, `real-80cm`, `real-turning`
(2 sessions), `phone-photo` (2 sessions), `phone-video` (1 session). No monitor and no printed
photo were recorded: the owner has no printer, and the monitor case was not recorded.

Each session was replayed with `fasec-replay run --ir-input <dir> --ep cpu`. The cues are those
of the trusted frames, that is from the fifth lit frame on, once the IR↔RGB mapping is learned.
"(rgb, ir)" counts RGB tracks and IR faces per frame.

Cues before tuning (placeholder thresholds: relief ≥ 1.15 for Live):

| label | IR verdicts | RGB MiniFASNet | face_vs_bg min-med-max | relief | lit_face | lit_bg | (rgb, ir) |
|---|---|---|---|---|---|---|---|
| real-30cm | 17 FlatRefl, 5 Unknown | 15 Spoof, 7 Unknown | 2.31-2.35-2.61 | 0.87-0.90-0.91 | 109-117-126 | 46-49-52 | (1,1)×22 |
| real-50cm | 17 FlatRefl, 4 Unknown | 17 Spoof, 4 Live | 2.07-2.10-2.13 | 0.83-0.84-0.85 | 126-128-130 | 60-61-62 | (1,1)×21 |
| real-80cm | 13 FlatRefl, 4 Unknown | 17 Spoof | 1.72-1.74-1.76 | 0.81-0.82-0.84 | 128-129-133 | 74-74-76 | (1,1)×17 |
| real-turning a | 11 FlatRefl, 4 Unknown | 9 Spoof, 6 Unknown | 2.01-2.08-2.34 | 0.83-0.86-0.91 | 129-133-137 | 57-62-68 | (1,1)×15 |
| real-turning b | 15 Unknown (never trusted) | 12 Live | n/a | n/a | | | (1,1)×15 |
| phone-photo b | 26 Unknown (untrusted) | 26 Spoof | n/a | | | | (1,0)×24 |
| phone-video | 22 Unknown (untrusted) | 22 Spoof | n/a | | | | (1,0)×22 |

- The unlit (ambient) frame was blank on every recorded frame, so `lit_minus_ambient` and the
  self-emission cue were never measured.
- Glints were 0–2 per frame on real faces, so they do not separate anything here.
- A phone shows a face to RGB and none to IR. Replayed alone, a phone session never trusts the
  mapping, since that needs a face on both cameras. So the phone case is measured after a real
  session through one pipeline (below), as it happens live: trust is earned on the owner and
  then the phone is held up.
- On this camera the nose reads darker than the cheeks (relief 0.81–0.91). The assumption that
  the nose is brighter (relief ≥ 1.15) was wrong here, so every real face was called flat.

### Owner decisions

- **D1, IR overrides RGB.** MiniFASNet called the real owner `Spoof` on most frames at laptop
  distance. So IR `Live` now beats an RGB `Spoof` (result `Live`, tier `Ir`). This is the bar
  Windows Hello holds since CVE-2025-26644: an IR face that looks live plus an RGB camera that
  sees the same face. IR alone still never gives `Live`: fusion runs per RGB track. With IR
  unknown, absent, untrusted, stale or unhealthy, the RGB verdict still decides, `Spoof`
  included. An IR `Spoof` (3 lit frames in a row) still beats everything.
  - *Residual risk:* a screen that emits the owner's NIR image as well as a visible face. The
    self-emission cue that should catch it needs a usable ambient frame, and this laptop blanks
    every one.
- **D2, relief from real faces only.** Live needs relief in `[0.70, 0.97]`.
  - Relief above 0.97 is `Spoof(FlatReflectance)`. This rule is meant for a flat print, which
    should read about 1.0. **It is not verified against a printed photo, since none was
    recorded.**
  - Relief below 0.70 is `Unknown`.
  - The other thresholds keep their values, with margin:
    - `face_vs_background ≥ 1.5`: real minimum 1.72, at 80 cm.
    - `max_background 100`: real maximum 76. Replaced in fix round 1 (ruling R9) by
      `max_frame_median 100` on the median of the whole lit frame, real maximum 36; see the
      flood check below.
    - `min_lit_face 20`: real minimum 109.
  - `min_emitter_fraction` and `self_emit_contrast` stay unmeasured placeholders.

### Fusion

`fuse` in `crates/fasec-vision/src/pipeline.rs` fuses each RGB track's verdict with its IR
evidence:

| IR evidence | RGB | result |
|---|---|---|
| Spoof (3 lit frames in a row) | any | Spoof, tier Ir |
| Suspect (Spoof, streak not met; or a track the latest lit frame has not judged) | any | Unknown, tier Ir |
| Live | any, Spoof included | Live, tier Ir |
| Absent (IR off, unhealthy, untrusted, stale; IR `Unknown`) | any | the RGB verdict |

- **Fix round 2 (ruling R10): a pending IR spoof blocks Live.** A moving phone gets a new RGB
  track id often, and each id starts its streak over. Without this row a fooled MiniFASNet could
  make it `Live` on those frames. Cost: a real face with a single-frame IR miss reads `Unknown`
  for that frame.
- **Fix round 3: a track new since the latest lit frame is `Suspect`.** RGB runs faster than the
  lit IR frames. A track created on a call without a new lit frame has no IR verdict yet. While
  IR is healthy, trusted and fresh, it reads `Unknown` until the next lit frame judges it. It
  used to take the RGB verdict for that gap.
- An IR face-detector error on a lit frame resets IR, as an unhealthy IR camera does. The
  previous lit frame's evidence no longer counts.

### Result after tuning

From the ignored test `exit_criteria_on_the_owner_recordings` (`fasec-replay`, CPU EP, debug
build). Its output is copied below.

**Real face.** Each session was replayed alone. Counts are tracks on trusted frames, that is
after the 4-frame warm-up.

| label | trusted tracks | Live, tier Ir |
|---|---|---|
| real-30cm | 17 | 17 (100 %) |
| real-50cm | 17 | 17 (100 %) |
| real-80cm | 13 | 13 (100 %) |
| real-turning (2 sessions) | 11 + 0 | 11 (100 %) |

- Replayed alone, real-turning session b never trusts the IR mapping, so it stays on the RGB
  tier (12 RGB Live). The cause was not measured: the IR face box is not logged when it is not
  matched.
- Replayed after real-50cm through one pipeline, the same session is Live on tier Ir on 13 of 15
  frames. One frame is RGB Spoof: IR gave a single FlatReflectance frame, under the 3-frame
  streak. One frame is RGB Live.

**Phone.** Each phone session was replayed after each real session, through one pipeline. Trust
is earned on the owner first, as it would be live. Counts are tracks of the phone session from
the 3rd trusted frame with a face, so that the spoof streak can be met.

| label | tracks (4 predecessors) | Spoof (fused) | of which IR evidence Spoof |
|---|---|---|---|
| phone-photo (session b) | 92 (23 distinct frames × 4) | 92 (100 %) | 22 (23.9 %) |
| phone-video | 80 (20 × 4) | 80 (100 %) | 27 (33.8 %) |

After fix round 1 (frame-wide flood check, below). The last column counts each track from its
own 3rd trusted frame, since a new RGB track id starts a new spoof streak:

| label | tracks | Spoof (fused) | IR evidence Spoof | IR Spoof past each track's streak |
|---|---|---|---|---|
| phone-photo (session b) | 92 | 92 (100 %) | 64 (69.6 %) | 64 of 64 (100 %) |
| phone-video | 80 | 80 (100 %) | 44 (55.0 %) | 44 of 44 (100 %) |

Real faces are unchanged: Live on tier Ir on 17, 17, 13 and 11 of 17, 17, 13 and 11 trusted
tracks.

After fix round 2 (ruling R10, a pending IR spoof blocks Live). Same tracks as above:

| label | tracks | Live | Spoof (fused) | of which IR evidence Spoof | Unknown |
|---|---|---|---|---|---|
| phone-photo (session b) | 92 | 0 (0 %) | 68 (73.9 %) | 64 (69.6 %) | 24 (26.1 %) |
| phone-video | 80 | 0 (0 %) | 44 (55.0 %) | 44 (55.0 %) | 36 (45.0 %) |

- The Spoof share fell from 100 %: the first two lit frames of each new track id, which
  MiniFASNet's Spoof used to cover, now read `Unknown` on tier Ir. No phone frame is `Live`
  whatever MiniFASNet says on those frames.
- Real faces are unchanged: 17, 17, 13 and 11 of 17, 17, 13 and 11 trusted tracks Live on
  tier Ir.

After fix round 3 (a track new since the latest lit frame is `Suspect`): every number above is
unchanged. Real faces: 17, 17, 13 and 11 of 17, 17, 13 and 11 Live on tier Ir. Phone photo: 0 of
92 Live, 68 Spoof, 24 Unknown. Phone video: 0 of 80 Live, 44 Spoof, 36 Unknown. The recordings
pair a lit frame with every RGB frame, so replay never makes a call without a new lit frame. The
change is covered by unit tests only.

- When the predecessor is real-turning session b, the phone sessions add nothing: no trust was
  earned.
- Phone-photo session a (8 steps) is left out of the criterion, because the owner's real face is
  in view during it. Its first 5 steps show an IR face with live cues at the real-50cm box, and a
  second large face with live IR cues appears later. Replayed after real-50cm, it reads Live on
  tier Ir for the owner's track, and the small phone face is RGB Spoof.
- *Superseded by fix round 1:* before it, IR evidence alone called the phone a spoof on only
  24 % (photo) and 34 % (video) of these frames. The ring-based flood rule fired on a phone held
  close (ring 100–255, real maximum 76) and broke the streak.

**Flood check, fix round 1 (ruling R9).** The flood check now reads `lit_frame_median`, the
median grey level of the whole lit frame (a 256-bin histogram, lower median), instead of the
ring. Sunlight raises the whole scene; a phone held close lights only its own surroundings. The
ring stays as the denominator of `face_vs_background`.

| label | `lit_frame_median` min-med-max | ring `lit_background` min-max |
|---|---|---|
| real-30cm | 13-15-15 | 46-52 |
| real-50cm | 25-26-26 | 60-62 |
| real-80cm | 35-36-36 | 74-76 |
| real-turning a | 24-24-25 | 57-68 |
| phone-photo b (after real-30cm / real-50cm) | 3-7-122 | 10-255 |
| phone-video (after real-30cm / real-50cm) | 7-12-73 | 11-254 |

- Trusted frames with cues. The real-face median rises with distance (exposure), up to 36.
- `max_frame_median` is 100, margin 64 over the real maximum. 2 of the 48 phone frames read
  above it (122 and 116, the first two trusted frames of phone-photo b) and stay `Unknown`.
- With it, every phone frame past its track's 3-frame streak is IR Spoof. The IR share counted
  from the session's 3rd trusted frame is 70 % (photo) and 55 % (video): the RGB tracker gives
  the phone a new track id as it moves (6 ids over the trusted frames of phone-photo b, 8 over
  those of phone-video, replayed after real-50cm), and each new
  id must build its own streak. Since fix round 2 those frames read `Unknown`, never `Live`.

### In ISO/IEC 30107-3 terms

Frame-level rates, one subject and one device. The counts above come from overlapping replays, so
the distinct frame counts are given here.

Sample sizes, in distinct trusted frames: real-30cm 17, real-50cm 17, real-80cm 13, real-turning
a 11 (58 bona fide in all); phone photo 23; phone video 20. Frames of one session are not
independent, so the true sample is smaller still. Zero errors in n frames bounds the error rate
at about 3/n (95 % confidence): 5 % for 58, 13 % for 23, 15 % for 20. **These counts do not
support a 99 % claim.** That needs about 300 attack presentations per species with no error.

- **BPCER** (bona fide presentations classified as attacks), on trusted frames of the real
  sessions replayed alone: 0 of 58.
  - real-turning session b replayed after real-50cm: 1 of 15. That frame was RGB Spoof while IR
    was below its streak. Under R10 it would read `Unknown` instead (inferred, not re-run).
- **APCER, phone photo (session b):** 0 of 23 distinct frames accepted as Live. IR evidence
  alone called a spoof on 64 of 92 frame-replays (70 %; 24 % before fix round 1), and on all
  64 that were past their track's spoof streak.
- **APCER, phone video:** 0 of 20 distinct frames accepted as Live. IR alone: 44 of 80 (55 %;
  34 % before), and all 44 past the streak.
- **Open item:** the frames IR alone misses are a new track's first two frames. The phone gets
  a new RGB track id as it moves, which restarts the spoof streak. Since fix round 2 those
  frames read `Unknown`, never `Live`, so they no longer rest on MiniFASNet; they are still
  not called a spoof.
- The frame-wide flood threshold is not verified in sunlight: none was recorded.

Not measured, so no claim is made:
- printed photo, monitor, NIR-emitting screen, mask;
- other subjects, other cameras, strong sunlight;
- attempt-level (per presentation) rates. The rates above count frames.

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
  ambient frames). The probe reports mean 0.0 both for an all-black frame and for an empty
  read (`count == 0`), so the "driver blanks alternate frames" reading is unconfirmed.
  `IsIlluminated` reports this correctly, so a liveness check can keep only
  the illuminated frames, which gives about 7.5 usable fps.
- **Pairing:** the `Rts-DMFT-Group` group pairs the RGB source (`MI_00`) and the IR source
  (`MI_02`) of the same USB device, and their instance ids share the parent `6&ead33bb&0`. This
  group is the natural RGB/IR pairing key. The IR-only group works too.
- **Independent of RGB use:** reading IR worked while another process held the RGB camera
  through nokhwa/MF (see SP-5).
- Not tested here: interaction with Windows Hello itself (checks 2 and 3), and whether the
  0.0 frames are truly all zeros (only the mean was computed).
- **Correction (Tasks 1 and 7):** they are not all zeros. The unlit frames are near-black, with
  mean ≈ 0.00 and a maximum pixel of 1–4 (Task 1). "Mean 0.0" was rounded to one decimal.
  `fasec_vision::ir::ambient_is_blank` (mean < 1 and 99th percentile ≤ 8) detects them;
  `IrFrame::is_blank` (every pixel 0) does not. In the Task 7 recordings every unlit frame was
  blank in this sense.
