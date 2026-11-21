# SP-6: nokhwa enumeration, ids and hot-plug

**Question:** is nokhwa 0.10 (Media Foundation backend, `input-msmf`) good enough for fasec's
camera layer? Specifically: does it list every camera with usable formats, is the id we persist
stable (across runs, reboots, replug and other ports), and does the WinRT `DeviceWatcher` used for
hot-plug report plug and unplug with ids that match `list()`? (SRS FR-CAM-001..004)

**Program:** `crates/fasec-camera/examples/camera_probe.rs`

## How to run

Use **PowerShell** and put ids in single quotes (Git Bash rewrites `\\?\`).

```powershell
cargo build -p fasec-camera --examples
$exe = '.\target\debug\examples\camera_probe.exe'
& $exe list                 # id / name / RGB-decodable formats, plus desktop camera access
& $exe watch                # hot-plug events until Ctrl+C (or: & $exe watch 60)
& $exe run '<id>'           # opens by id; prints fps, luma, health each second
```

Checks, one table row each:

1. **Stable across runs:** run `list` twice, and once more after a reboot. Is the id identical?
2. **Replug, same port:** while `watch` runs, unplug the USB camera and plug it back into the same
   port. Expect `Removed(id)` and then `Added(id)` with the same id. `list` afterwards should
   show the same id.
3. **Other port / hub:** plug the camera into a different port. Does the id change? (Expected:
   yes for cameras without a USB serial number, see below.)
4. **Unplug while running:** start `run '<id>'`, then unplug. Expect `health Lost` within
   about 1.5 s, with `last_error` set, and no hang. Ctrl+C should exit within about 2 s.
5. **Second camera:** with two cameras attached, does `list` show both, with formats?
6. **Formats:** are the formats `list` shows the ones you expect from the camera's spec?

## Results (fill in by hand)

| Machine | Windows build | Camera (USB serial y/n) | Check | Outcome | Notes |
|---|---|---|---|---|---|
| | | | 1 stable across runs/reboot | | |
| | | | 2 replug same port | | |
| | | | 3 other port / hub | | |
| | | | 4 unplug while running | | |
| | | | 5 second camera | | |
| | | | 6 formats | | |

## Decision

_(empty, to be filled in after the results)_

## Observed on dev machine (2026-11-21, by the implementer)

Machine: ASUS laptop, Windows 11 `10.0.26300.9974`, built-in `ASUS FHD webcam` and
`ASUS IR camera`. Both are built in, so no physical replug or port change was possible.

`camera_probe list`:

```
desktop camera access: Allowed
1 camera(s)
[0] name: ASUS FHD webcam
    id:   \\?\usb#vid_3277&pid_0096&mi_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global
    formats: 160x120@15 160x120@30 176x144@15 176x144@30 320x240@15 320x240@30 352x288@15 352x288@30 640x360@15 640x360@30 640x480@15 640x480@30 1280x720@10 1280x720@15 1280x720@30 1280x960@15 1280x960@30 1920x1080@5 1920x1080@15 1920x1080@30
```

`camera_probe watch 3`:

```
watching (cameras already present are reported as Added first); Ctrl+C to stop
Added("\\\\?\\usb#vid_3277&pid_0096&mi_00#6&ead33bb&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\\global")
stopped
```

(The event is printed with `{:?}`, so backslashes are doubled. The value is the same id as in
`list`.)

`camera_probe run '<id>' 3`:

```
started in 525 ms
fps   7.9  1280x720 luma 65.9  health Ok  last_error None
fps   7.9  1280x720 luma 64.4  health Ok  last_error None
stopped in 372 ms
```

A mangled id (Git Bash turned `\\?\` into `\?\`) fails cleanly:
`camera \?\usb#...\global not found` (exit 1), not a hang.

Findings from Tasks 10 and 11:

- **The id is the MF symbolic link** (`MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK`,
  nokhwa's `CameraInfo::misc()`), not an enumeration index. nokhwa resolves it to an index
  again on every open. The same id was printed by separate processes and by this probe.
- **Letter case differs by API:** MF returns the link in lowercase. WinRT
  `DeviceInformation` / `DeviceWatcher` returns `\\?\USB#VID_3277&...\GLOBAL` in mixed case.
  fasec lowercases every id (`normalize_id`) and opens case-insensitively, so watcher ids equal
  `list()` ids (confirmed above).
- **Port dependence:** the PnP instance segment `6&ead33bb&0&0000` comes from the parent and
  port, because this camera reports no USB serial number. Expected: the id is stable across
  reboots and replug into the **same** port, and changes on another port or hub. Cameras with
  a serial number should keep their id. Not verified physically (check 3).
- **The IR camera is not in the MF list** and not in the VideoCapture `DeviceWatcher`, although
  PnP and WinRT `MediaFrameSourceGroup` see it (see SP-7). nokhwa cannot be used for IR.
- **Busy shows up late:** another process holding the camera makes the first `frame()` fail
  with `0xC00D3704`. Open and `open_stream` succeed. `Capture::start` therefore waits for the
  first frame (see SP-5).
- **Formats:** `list` shows only RGB-decodable formats (MJPEG/YUYV/NV12/RAW), deduped across
  fourcc. A device whose formats cannot be read is still listed, with an empty format list.
- **Speed:** MJPEG→RGB decoding in nokhwa runs at about 8 fps at 720p in debug builds and about
  30 fps (53-61 frames in 2 s) in release.
- **nokhwa quirks we work around:** a process-global "MF initialized" flag around per-thread COM
  init (fasec runs nokhwa on its own threads with its own apartment), and a non-atomic camera
  refcount around `MFShutdown` (a process-wide `mf_lock` serialises open and drop).
  PermissionDenied (`0x80070005`) is mapped but has not been reproduced, because the privacy
  toggle was not touched.
