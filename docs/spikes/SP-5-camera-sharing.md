# SP-5: Sharing the camera with Teams / Zoom

**Question:** can fasec keep reading the RGB camera while a video-call app (Teams, Zoom, the
Windows Camera app) uses it, in either start order? If not, what does fasec see (an explicit
`Busy`, a stall, or black frames), and does the other app keep working? (SRS FR-CAM-005/009,
fail safe §7.2)

**Program:** `crates/fasec-camera/examples/camera_probe.rs` (uses `fasec_camera::Capture`, i.e.
nokhwa over Media Foundation, exactly like fasec itself).

## How to run

Use **PowerShell** with the id in single quotes. Git Bash rewrites the leading `\\?\` and the
probe then reports `camera ... not found`.

```powershell
cargo build -p fasec-camera --examples
$exe = '.\target\debug\examples\camera_probe.exe'
& $exe list                      # copy the id of the RGB camera
& $exe run '<id>'                # Ctrl+C to stop; or: & $exe run '<id>' 30  (stop after 30 s)
```

`run` prints one line per second: `fps`, frame size and mean luma, `health` and `last_error`.
`cargo run -p fasec-camera --example camera_probe -- run '<id>'` works too. Debug builds reach
only about 8 fps at 1280x720 because MJPEG decoding is slow in debug; add `--release` for real
frame rates (about 30 fps on the dev machine).

Scenarios, one table row each:

1. **App first:** start a Teams/Zoom call (or test call) with video on, then `run`.
2. **fasec first:** `run`, then turn video on in Teams/Zoom. Watch both the probe output and the
   app's self-view.
3. **App leaves:** with both running (if 1 or 2 worked), turn the app's video off. Does the
   probe keep `health Ok`?
4. **IR while the app holds RGB** (needs an IR camera, see SP-7): with the app's video on, run
   `cargo run -p fasec-camera --example ir_probe -- read <group-index>`.

Outcome values: `shared` (both see live video), `fasec Busy` (probe start fails with
`camera is in use by another application`), `fasec stalls` (starts, then `health Lost`),
`app loses video`, or describe what you saw.

## Results (fill in by hand)

| Machine | Windows build | Camera | App + version | Scenario | Outcome | fps in probe | App self-view OK? | Notes |
|---|---|---|---|---|---|---|---|---|
| dev laptop | 26300 | ASUS FHD webcam | Windows Camera | 1 app first | fasec `start` fails: "camera is in use by another application" (Busy) | — | yes, unaffected | 2026-11-22, verified by screenshot + probe output |
| dev laptop | 26300 | ASUS FHD webcam | Windows Camera | 2 fasec first | Camera app gets no video | 6–9 (debug build) | no | 2026-11-22, owner observed; probe stayed `health Ok` |
| | | | | 3 app leaves | | | | |
| | | | | 4 IR while app holds RGB | | | | |

## Decision

On this camera/driver the RGB stream is exclusive across processes in both directions. So fasec
cannot keep watching during a video call: when a call app holds the camera, fasec gets `Busy` and
must show the camera-loss popup with the "busy in another app" cause and Pause / Switch camera /
Keep protecting (decision D2, FR-CAM-005b). When fasec holds it, call apps get no video, so the
pause-in-call option matters. Not yet tried: Teams/Zoom specifically, WinRT `SharedReadOnly` for
the RGB source.

## Observed on dev machine (2026-11-21, by the implementer)

Machine: ASUS laptop, Windows 11 `10.0.26300.9974`, built-in `ASUS FHD webcam`
(USB `VID_3277&PID_0096`, `MI_00`) plus `ASUS IR camera` (`MI_02`). Debug build. Teams/Zoom
were **not** tested. Instead a second `camera_probe` process played the role of "the other app".

`camera_probe list`:

```
desktop camera access: Allowed
1 camera(s)
[0] name: ASUS FHD webcam
    id:   \\?\usb#vid_3277&pid_0096&mi_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global
    formats: 160x120@15 160x120@30 176x144@15 176x144@30 320x240@15 320x240@30 352x288@15 352x288@30 640x360@15 640x360@30 640x480@15 640x480@30 1280x720@10 1280x720@15 1280x720@30 1280x960@15 1280x960@30 1920x1080@5 1920x1080@15 1920x1080@30
```

`camera_probe run '<id>' 3` (alone):

```
started in 525 ms
fps   7.9  1280x720 luma 65.9  health Ok  last_error None
fps   7.9  1280x720 luma 64.4  health Ok  last_error None
stopped in 372 ms
```

Two processes. Process A ran `run '<id>' 7`. Two seconds later process B ran `run '<id>' 2`,
and then `ir_probe read 2` ran while A was still streaming:

```
== B: second camera_probe run while A streams
Error: start capture on \\?\usb#vid_3277&pid_0096&mi_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global

Caused by:
    camera is in use by another application
B exit 1
== IR read while A streams
...
RESULT: 30 Infrared frames read with SharedReadOnly
== A output (exit 0)
started in 757 ms
fps   7.8  1280x720 luma 58.3  health Ok  last_error None
fps   3.9  1280x720 luma 52.2  health Ok  last_error None
fps   3.0  1280x720 luma 51.0  health Ok  last_error None
fps   3.7  1280x720 luma 51.2  health Ok  last_error None
fps   4.6  1280x720 luma 51.1  health Ok  last_error None
stopped in 508 ms
```

What this shows (Task 10/11 findings included):

- **Across processes the RGB camera is not shared** through Media Foundation's source reader.
  The second process opens the device and the stream without error. The failure,
  `MF_E_HW_MFT_FAILED_START_STREAMING` `0xC00D3704` ("Hardware MFT failed to start streaming due
  to lack of hardware resources"), comes on the **first frame read**. `Capture::start` waits
  for that first frame, so it returns an explicit `Busy` instead of a silent stall. The process
  that was first (A) kept streaming with `health Ok`.
- **Inside one process** a second `Capture::start` on the same camera succeeds and shares the
  stream (Task 10).
- **IR is independent of RGB:** the IR source was read with WinRT `SharedReadOnly` while
  another process held the RGB camera.
- A's frame rate fell from about 8 to 3-5 fps while B and `ir_probe` were running. The cause is
  unclear: it could be CPU load from three debug processes or the shared USB device. Re-check
  with `--release`.
- Expectation for Teams/Zoom (not verified): if the app opens the camera first, fasec gets
  `Busy`; if fasec is first, the app may fail to start video. Scenario 2 is the one that
  matters most, because fasec runs all the time.
- Not tried here: reading the RGB source through WinRT `MediaCapture` with
  `SharedReadOnly`. That would attach to a stream another app has already started, and it is
  the likely way to coexist with call apps. `ir_probe` shows the pattern works for the IR
  source on this machine.
