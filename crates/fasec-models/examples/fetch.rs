//! Dev tool: fetch and verify default models and test images.
//!
//! `cargo run -p fasec-models --example fetch [-- --dir models]`

use anyhow::{Context, bail};
use fasec_models::{Manifest, ensure};
use std::path::{Path, PathBuf};

fn main() -> anyhow::Result<()> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .context("workspace root")?;
    let mut models_dir = root.join("models");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dir" => models_dir = PathBuf::from(args.next().context("--dir needs a value")?),
            other => bail!("unknown argument `{other}` (usage: fetch [--dir <models dir>])"),
        }
    }

    let models = Manifest::load(root.join("models/manifest.toml"))?;
    let assets = Manifest::load(root.join("assets/test-assets.toml"))?;
    let assets_dir = root.join(".assets");
    let jobs = models
        .defaults()
        .map(|e| (e, &models_dir))
        .chain(assets.model.iter().map(|e| (e, &assets_dir)));

    let mut failed = 0;
    for (entry, dir) in jobs {
        match ensure(dir, entry) {
            Ok(path) => println!("ok    {:<16} {}", entry.id, path.display()),
            Err(e) => {
                failed += 1;
                eprintln!("FAIL  {:<16} {e}", entry.id);
            }
        }
    }
    if failed > 0 {
        bail!("{failed} file(s) failed");
    }
    Ok(())
}
