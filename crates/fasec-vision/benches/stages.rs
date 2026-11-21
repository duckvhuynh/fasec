//! Per-stage latency on `.assets/face-a-1.jpg` (full 2000x3000, and scaled to 720 px high like a
//! webcam frame), CPU and DirectML: YuNet at 320 and 640 px long side, SFace, MiniFASNet (both
//! crops), coarse pose. Skips with a message if models or the image are missing.

use std::hint::black_box;
use std::path::{Path, PathBuf};

use criterion::{Criterion, criterion_group, criterion_main};
use fasec_config::ExecutionProvider;
use fasec_core::Frame;
use fasec_vision::attention::CoarsePose;
use fasec_vision::liveness::MiniFasNet;
use fasec_vision::recognize::SFace;
use fasec_vision::{
    FaceDetector, FaceEmbedder, LivenessModel, PoseEstimator, Runtime, YuNet, YuNetConfig,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Verified paths of yunet, sface, minifasnet-v2, minifasnet-v1se.
fn models() -> Result<[PathBuf; 4], String> {
    let dir = root().join("models");
    let manifest =
        fasec_models::Manifest::load(dir.join("manifest.toml")).map_err(|e| e.to_string())?;
    let ids = [
        "yunet-2023mar",
        "sface-2021dec",
        "minifasnet-v2",
        "minifasnet-v1se",
    ];
    let mut out = Vec::new();
    for id in ids {
        let entry = manifest.get(id).ok_or(format!("{id} not in manifest"))?;
        let path = fasec_models::model_path(&dir, entry).map_err(|e| format!("{id}: {e}"))?;
        fasec_models::verify_file(&path, &entry.sha256).map_err(|e| format!("{id}: {e}"))?;
        out.push(path);
    }
    out.try_into()
        .map_err(|_| "expected four model paths".into())
}

/// face-a-1 at full size and at 720 px high.
fn face_a1() -> Result<[(&'static str, Frame); 2], String> {
    let img = image::open(root().join(".assets/face-a-1.jpg")).map_err(|e| e.to_string())?;
    let full = img.to_rgb8();
    let h720 = image::imageops::resize(
        &full,
        full.width() * 720 / full.height(),
        720,
        image::imageops::FilterType::Triangle,
    );
    let frame = |img: image::RgbImage| {
        Frame::new(0, 0, img.width(), img.height(), img.into_raw()).map_err(|e| e.to_string())
    };
    Ok([("full", frame(full)?), ("720p", frame(h720)?)])
}

fn stages(c: &mut Criterion) {
    let (paths, frames) = match (models(), face_a1()) {
        (Ok(p), Ok(f)) => (p, f),
        (Err(e), _) | (_, Err(e)) => {
            eprintln!(
                "skipping stage benches: {e} (run `cargo run -p fasec-models --example fetch`)"
            );
            return;
        }
    };
    let [yunet, sface, v2, v1se] = &paths;
    for (name, ep) in [
        ("cpu", ExecutionProvider::Cpu),
        ("directml", ExecutionProvider::DirectMl),
    ] {
        let rt = match Runtime::new(ep) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("skipping {name}: {e}");
                continue;
            }
        };
        let det_cfg = |side| YuNetConfig {
            input_long_side: side,
            ..YuNetConfig::default()
        };
        let mut det320 = YuNet::new(&rt, yunet, det_cfg(320)).expect("yunet");
        let mut det640 = YuNet::new(&rt, yunet, det_cfg(640)).expect("yunet");
        let mut emb = SFace::new(&rt, sface).expect("sface");
        let mut live = MiniFasNet::new(&rt, v2, v1se).expect("minifasnet");
        let mut pose = CoarsePose;
        for (label, frame) in &frames {
            let dets = det320.detect(frame).expect("detect");
            let Some(face) = dets.first().copied() else {
                eprintln!("skipping {name}/{label}: no face detected in face-a-1");
                continue;
            };
            let mut g = c.benchmark_group(format!("{name}/{label}"));
            g.bench_function("detect@320", |b| b.iter(|| det320.detect(black_box(frame))));
            g.bench_function("detect@640", |b| b.iter(|| det640.detect(black_box(frame))));
            g.bench_function("embed", |b| b.iter(|| emb.embed(black_box(frame), &face)));
            g.bench_function("liveness", |b| {
                b.iter(|| live.score(black_box(frame), &face))
            });
            g.bench_function("pose", |b| {
                b.iter(|| pose.estimate(black_box(frame), &face))
            });
            g.finish();
        }
        eprintln!(
            "{name}: sessions on yunet={}/{}, sface={}, minifasnet={:?}",
            det320.ep(),
            det640.ep(),
            emb.ep(),
            live.ep()
        );
    }
}

criterion_group!(benches, stages);
criterion_main!(benches);
