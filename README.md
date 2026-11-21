# fasec

Local-first privacy guard for Windows, written in Rust. fasec uses your camera to
recognise you and the people you trust, notices when someone else is **actually
looking** at your screen, and hides your content (blur, overlay, warning, minimise,
lock) until they are gone.

> Status: M0, workspace skeleton. Nothing works yet. See the documents below.

## Build

Needs Rust stable with the MSVC toolchain.

```sh
cargo run -p fasec-app
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and rules.

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
- Model weights are downloaded from pinned URLs with SHA-256 checks and are never
  committed to this repo.
- Honest about limits: a normal webcam cannot prove a face is 3D. fasec shows which
  liveness tier your hardware supports (RGB, IR, depth).

## License

MIT OR Apache-2.0 (proposed).
