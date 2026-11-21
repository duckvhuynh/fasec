# Contributing to fasec

Read [docs/SRS.md](docs/SRS.md) (architecture) and [docs/ROADMAP.md](docs/ROADMAP.md)
(what is being worked on) first.

## Build and check

Needs Rust stable (MSVC toolchain on Windows).

```sh
cargo build
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI runs the same checks, plus `cargo deny check`.

## Rules

- **No model weights in git.** Add new models to `models/manifest.toml` (URL,
  SHA-256, licence). Default models must be MIT / Apache-2.0 / BSD.
- **No personal recordings, face images or databases in git**, not even in tests.
  Tests use synthetic data; replay sets stay on your machine.
- **Keep `fasec-core`, `fasec-config`, `fasec-vision`, `fasec-identity`,
  `fasec-policy`, `fasec-models` and `fasec-replay` platform-neutral**: no `windows`
  crate there. They are tested on Linux in CI.
- Requirements have IDs (e.g. `FR-CAM-005`). Mention them in PRs and tests.
- Fail safe: if a change can leave the screen unprotected when something breaks, it
  needs a test that shows it doesn't.

## License

By contributing you agree your work is dual-licensed under MIT OR Apache-2.0.
