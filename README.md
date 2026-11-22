# fasec

Local-first privacy guard for Windows, written in Rust. fasec uses your camera to
recognise you and the people you trust, notices when someone else is **actually
looking** at your screen, and hides your content (blur, overlay, warning, minimise,
lock) until they are gone.

> Status: M0–M2 (spikes, camera, vision core, replay, identity and enrolment). No end-user app yet. See the documents below.

## Developer quick start

Needs Windows 10/11, Rust stable with the MSVC toolchain (`rustup` default on Windows,
MSRV 1.92) and the Visual Studio C++ build tools ("Desktop development with C++"). Video input
for replay also needs `ffmpeg`/`ffprobe` on `PATH`.

```sh
# 0. Export the MiniFASNet liveness models from the upstream Apache-2.0 weights (into models/),
#    in a venv with the pinned tools/requirements-export.txt (Python 3.12).
python tools/export_minifasnet.py

# 1. Download and SHA-256-verify the default models (into models/) and the public-domain
#    test portraits (into .assets/). Neither is ever committed.
cargo run -p fasec-models --example fetch

# 2. Checks (CI runs the same; ignored tests are skipped).
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Ignored tests: models and test images (after step 1).
cargo test -p fasec-vision -- --ignored
cargo test -p fasec-replay -- --ignored
cargo test -p fasec-identity -- --ignored
# Ignored tests: camera. They share one webcam, so run them one at a time.
cargo test -p fasec-camera -- --ignored --test-threads=1

# Per-stage benches (CPU and DirectML).
cargo bench -p fasec-vision
```

Run the vision pipeline offline on a folder of `*.png`/`*.jpg` frames or a video file. It writes
one JSON line per frame to stdout or `--out` (see `--help` for `--fps`, `--ep`,
`--with-embeddings`, `--with-timings`):

```sh
cargo run -p fasec-replay --release -- run --input <dir|video>
```

Live debug window (camera, boxes, track ids, liveness, pose, per-stage ms):

```sh
cargo run -p fasec-ui --example live_debug --release [-- --seconds 30]
```

### Enrol and see your name

Needs the models from steps 0 and 1 (the MiniFASNet export too: enrolment rejects samples
that are not judged live). Enrol the owner first: the window asks for a new owner PIN
(4–12 digits), then guides you through five head poses. Authorised people are added the same
way and need the owner PIN. Templates are DPAPI-encrypted for the current Windows user; no
face image is stored.

```sh
cargo run -p fasec-ui --example enroll --release -- --gallery gallery.db --name "<your name>"
cargo run -p fasec-ui --example enroll --release -- --gallery gallery.db --name "<name>" --role authorised
cargo run -p fasec-ui --example live_debug --release -- --gallery gallery.db
```

`live_debug --gallery` labels each face `name (owner|authorised) score`, `stranger <best>` or
`…` while the vote is still pending. A gallery that cannot be read is shown in the side panel
and nobody is recognised. `*.db` is git-ignored; never commit a gallery.

### IR liveness (Windows Hello IR camera)

An IR camera is optional. fasec uses the Windows Hello IR camera paired with the RGB webcam (same
frame source group, else same USB device) when it finds one. It opens it read-only and shared
(`SharedReadOnly`), so no admin rights are needed. Set it in `%APPDATA%\fasec\config.toml`:

```toml
[liveness]
ir = "prefer"   # default: use the paired IR camera when present and healthy
# ir = "off"    # never open the IR camera
```

Each face carries a liveness tier. `Rgb` means the verdict comes from MiniFASNet alone. `Ir`
means the IR camera decided it. The tier is `Ir` only while the IR camera is healthy, its latest
lit frame is at most 500 ms old and the IR↔RGB box mapping is trusted. The mapping is trusted
after 5 lit frames in a row that each show exactly one face on both cameras, in agreement. In
every other case fasec falls back to the `Rgb` tier.

How the verdict is fused:
- IR alone never makes a face live. The RGB camera must see the same face.
- A face that looks live in IR is `Live` on tier `Ir`, even when MiniFASNet calls it a spoof.
  MiniFASNet calls a real face at laptop distance a spoof on most frames.
- An RGB face with no IR face, or a flat IR face, is a spoof after 3 lit IR frames in a row. Until
  then it is `Unknown`, never `Live`.
- A new face is `Unknown` until the next lit IR frame judges it.

What IR catches: phone photo and phone video replays of the owner. A phone screen shows a face to
the RGB camera and none to the IR camera. On the owner's recordings no phone frame was `Live`.
What is not verified: printed photos, monitors, screens that emit near-infrared, masks, strong
sunlight, other people and other cameras. Whether Windows Hello sign-in keeps working while fasec
holds the IR camera is not verified yet. The numbers are in
[SP-7](docs/spikes/SP-7-ir.md#decision).

`ir_debug` shows the RGB and IR frames side by side with every IR cue per face. It records only
when you switch Record on or pass `--record <label>`. Recordings are the face images of whoever is
in view. They are written only under `%LOCALAPPDATA%\fasec\recordings\` and never leave the PC.
Replay one with `fasec-replay run --ir-input <dir>`.

```sh
cargo run -p fasec-ui --example ir_debug --release [-- --record <label>]
```

DirectML is used when available (`DirectML.dll` from `System32`; ort copies it next to the
built examples). A packaged release has to ship it beside the exe.

### Spike programs

See [docs/spikes/](docs/spikes/README.md) for what each measures and the results tables.

```sh
cargo run -p fasec-overlay --example sp1_overlay -- --mode <acrylic|backdrop|solid> [--seconds 15] [--click-through]
cargo run -p fasec-overlay --release --example sp1b_capture_blur -- [--monitor 0] [--seconds 15] [--downscale 8]
cargo run -p fasec-camera --example camera_probe -- list    # then: run '<id>' / watch
cargo run -p fasec-camera --example ir_probe                # then: -- read <group-index>
```

> **Warning:** `sp1_overlay` and `sp1b_capture_blur` cover the whole screen. Press **Esc** to
> close them early; otherwise they close after `--seconds`.

Camera ids start with `\\?\`. Pass them from **PowerShell** in single quotes; Git Bash rewrites
them and the probe reports `not found`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the rules.

## Documents

| Doc | What it covers |
|---|---|
| [PRD](docs/PRD.md) | Problem, goals, personas, features, feasibility matrix, what we reuse |
| [SRS](docs/SRS.md) | Architecture, functional and non-functional requirements, data model, interfaces |
| [MVP](docs/MVP.md) | v0.1 scope, user journeys, acceptance tests, metrics, early spikes |
| [Roadmap](docs/ROADMAP.md) | Milestones M0–M12 with tasks, estimates and exit criteria |

## Principles

- Everything runs on your device. No cloud, no telemetry.
- Face templates are encrypted with Windows DPAPI. No face images are stored by default.
- Model weights are downloaded from pinned URLs (or exported locally from upstream weights)
  with SHA-256 checks and are never committed to this repo.
- Honest about limits: a normal webcam cannot prove a face is 3D. fasec shows which
  liveness tier your hardware supports (RGB, IR, depth).

## License

MIT OR Apache-2.0 (proposed).
