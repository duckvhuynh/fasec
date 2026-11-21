//! SP-5 / SP-6 probe.
//!
//! `cargo run -p fasec-camera --example camera_probe -- [list | run <id> [secs] | watch [secs]]`
//!
//! `list` prints id/name/formats, `run` captures and prints FPS, mean luma and health every
//! second, `watch` prints hot-plug events. `run` and `watch` stop on Ctrl+C, or after `secs`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use crossbeam_channel::RecvTimeoutError;
use fasec_camera::{Capture, LatestFrame};
use fasec_config::CameraConfig;

const USAGE: &str = "usage: camera_probe [list | run <id> [secs] | watch [secs]]";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let stop = Arc::new(AtomicBool::new(false));
    let on_ctrl_c = Arc::clone(&stop);
    ctrlc::set_handler(move || on_ctrl_c.store(true, Ordering::Relaxed))
        .context("install Ctrl+C handler")?;
    let secs = |i: usize| -> Result<Option<Duration>> {
        args.get(i)
            .map(|s| s.parse().map(Duration::from_secs))
            .transpose()
            .context(USAGE)
    };
    let started = Instant::now();
    let done = |limit: Option<Duration>| {
        stop.load(Ordering::Relaxed) || limit.is_some_and(|l| started.elapsed() >= l)
    };
    match args.first().map(String::as_str) {
        None | Some("list") => list(),
        Some("run") => run(args.get(1).context(USAGE)?, secs(2)?, done),
        Some("watch") => watch(secs(1)?, done),
        _ => bail!(USAGE),
    }
}

fn list() -> Result<()> {
    println!(
        "desktop camera access: {:?}",
        fasec_camera::desktop_camera_access()
    );
    let cams = fasec_camera::list()?;
    println!("{} camera(s)", cams.len());
    for (i, c) in cams.iter().enumerate() {
        println!("[{i}] name: {}\n    id:   {}", c.name, c.id);
        let formats: Vec<String> = c
            .formats
            .iter()
            .map(|f| format!("{}x{}@{}", f.width, f.height, f.fps))
            .collect();
        println!("    formats: {}", formats.join(" "));
    }
    Ok(())
}

fn run(id: &str, limit: Option<Duration>, done: impl Fn(Option<Duration>) -> bool) -> Result<()> {
    let slot = Arc::new(LatestFrame::new());
    let t = Instant::now();
    let cap = Capture::start(id, &CameraConfig::default(), Arc::clone(&slot))
        .with_context(|| format!("start capture on {id}"))?;
    println!("started in {} ms", t.elapsed().as_millis());
    let (mut seq, mut frames, mut latest, mut tick) = (0, 0u32, None, Instant::now());
    while !done(limit) {
        if let Some(f) = slot.take_newer(seq, Duration::from_millis(100)) {
            seq = f.seq;
            frames += 1;
            latest = Some(f);
        }
        if tick.elapsed() >= Duration::from_secs(1) {
            let fps = f64::from(frames) / tick.elapsed().as_secs_f64();
            let frame = latest
                .take()
                .map(|f| format!("{}x{} luma {:.1}", f.width, f.height, f.mean_luma()))
                .unwrap_or_else(|| "no frame".into());
            println!(
                "fps {fps:5.1}  {frame}  health {:?}  last_error {:?}",
                cap.health(),
                cap.last_error()
            );
            (frames, tick) = (0, Instant::now());
        }
    }
    let t = Instant::now();
    cap.stop();
    println!("stopped in {} ms", t.elapsed().as_millis());
    Ok(())
}

fn watch(limit: Option<Duration>, done: impl Fn(Option<Duration>) -> bool) -> Result<()> {
    let (tx, rx) = crossbeam_channel::unbounded();
    let guard = fasec_camera::watch(tx)?;
    println!("watching (cameras already present are reported as Added first); Ctrl+C to stop");
    while !done(limit) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(ev) => println!("{ev:?}"),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => bail!("watcher disconnected"),
        }
    }
    drop(guard);
    println!("stopped");
    Ok(())
}
