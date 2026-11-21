# M0 + M1: skeleton, spikes, camera + vision core + replay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Frames from any chosen camera go in, a `VisionReport` (tracks, boxes, landmarks, embeddings-based recognition hooks, liveness, coarse head pose) comes out, testable offline through `fasec-replay`; plus runnable spike programs for the M0 questions that need a human at the machine.

**Architecture:** Platform-neutral data types (`fasec-core`), config (`fasec-config`), model manifest (`fasec-models`) and the vision pipeline (`fasec-vision`, ONNX Runtime via `ort`) are pure Rust and tested on any OS. Windows-only code lives in `fasec-camera` (capture, health, hot-plug) and `fasec-overlay` (spike programs only in this plan). `fasec-replay` and a `fasec-ui` example tie them together.

**Tech Stack:** Rust 2024 (MSVC), `ort` (+DirectML), `image`, `ndarray`, `nalgebra`, `pathfinding`, `serde`/`toml`/`serde_json`, `thiserror`/`anyhow`, `tracing`, `sha2`/`hex`, `ureq`, `nokhwa`, `windows`, `eframe`, `criterion`.

**Spec:** `docs/SRS.md` (authority), `docs/MVP.md`, `docs/ROADMAP.md` §M0–M1.

## Global Constraints

- Rust edition 2024, `rust-version = "1.85"`, target `x86_64-pc-windows-msvc`. Crate metadata inherits from `[workspace.package]`; every crate keeps `[lints] workspace = true`.
- Add dependencies with `cargo add -p <crate> <dep>` (latest compatible version) in the crate's own `Cargo.toml`. Do **not** add `[workspace.dependencies]`.
- No `windows` crate (and no `nokhwa`) in `fasec-core`, `fasec-config`, `fasec-vision`, `fasec-identity`, `fasec-policy`, `fasec-models`, `fasec-replay` (NFR-MNT-001). They must build and test on Linux.
- Never commit model weights, face images, videos or databases. Weights go to `models/` (gitignored by extension); test images go to `.assets/` (add `/.assets/` to `.gitignore`).
- Model files MUST be SHA-256 verified before load; a manifest entry with empty `url` or `sha256` MUST NOT be downloaded or loaded (NFR-SEC-004, `models/manifest.toml` header).
- Default models: YuNet `yunet-2023mar`, SFace `sface-2021dec` (ids exactly as in `models/manifest.toml`).
- Errors: `thiserror` in libraries, `anyhow` only in binaries/examples. Logging: `tracing`. No `unwrap()`/`expect()` in library code paths except on invariants with a one-line why.
- Coordinates in `fasec-core` public types are **normalised 0..1** relative to frame width/height.
- Tests that need model weights or `.assets/` images are `#[ignore = "needs models: run `cargo run -p fasec-models --example fetch`"]`; CI runs only non-ignored tests. Implementers MUST run their ignored tests locally (`cargo test -p <crate> -- --ignored`) and report the result.
- Fail safe (SRS §7.2): any camera/vision failure surfaces as an explicit error/health state, never as "no faces".
- Before each commit: `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` all clean.
- Commit messages: conventional commits (`feat(vision): …`), no attribution lines.

## Review Focus

1. **Frame size not a multiple of 32 / odd aspect ratios (e.g. 1280×720, 640×360, rotated 720×1280)** → YuNet still returns boxes in correct normalised coordinates. Test in Task 5.
2. **Faces touching or leaving the frame edge** → alignment/crops must clamp or pad, never panic or index out of bounds. Tests in Tasks 7 and 8.
3. **Camera delivering frames slower than inference, or stalling** → latest-frame slot never blocks the producer and never hands out stale frames twice; stall becomes `CameraHealth::Lost`. Tests in Tasks 11 and 12.
4. **Model file present but wrong/corrupted** → hash mismatch error naming the model id, no load. Test in Task 2.
5. **Zero faces for a long time, then a face** → scheduler drops to idle FPS and recovers to full FPS on the first detection; new track gets recognition and liveness immediately. Test in Task 9.

---

## Lane plan (for the controller)

Tasks 1–3 run first, sequentially, on branch `feat/m0-m1`. Then three lanes run in parallel, each in its own git worktree and branch cut from the Task 3 commit, each lane sequential inside:

| Lane | Branch | Tasks | Crates touched |
|---|---|---|---|
| V (vision) | `lane/vision` | 4 → 5 → 6 → 7 → 8 → 9 | `fasec-vision` |
| C (camera) | `lane/camera` | 10 → 11 → 12 | `fasec-camera` |
| O (overlay spikes) | `lane/overlay` | 13 → 14 | `fasec-overlay`, `docs/spikes/` |

After all lanes merge back into `feat/m0-m1` (resolve `Cargo.lock` by regenerating), Tasks 15 → 16 → 17 run sequentially.

---

### Task 1: `fasec-core` shared types

**Files:**
- Create: `crates/fasec-core/src/{lib.rs,geom.rs,frame.rs,vision.rs,threat.rs}`

**Interfaces:**
- Produces:
  - `geom`: `#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)] pub struct PointF { pub x: f32, pub y: f32 }`; `pub struct RectF { pub x: f32, pub y: f32, pub w: f32, pub h: f32 }` with `fn iou(&self, other: &RectF) -> f32`, `fn area(&self) -> f32`, `fn center(&self) -> PointF`, `fn clamp_unit(&self) -> RectF`.
  - `frame`: `pub struct Frame { pub seq: u64, pub ts_ms: u64, pub width: u32, pub height: u32, pub data: Arc<[u8]> }` (RGB8, row-major, `data.len() == width*height*3`), `Frame::new(seq, ts_ms, width, height, data: Vec<u8>) -> Result<Frame, CoreError>` validating length; `fn mean_luma(&self) -> f32` (Rec.601, 0..255).
  - `vision`: `pub struct TrackId(pub u64)`; `pub enum IdentityState { Unknown, Pending, Known { person: String, score: f32 }, Stranger { best: f32 } }`; `pub enum LivenessTier { Rgb, RgbTemporal, Ir, Depth }`; `pub enum LivenessState { Unknown, Live { score: f32, tier: LivenessTier }, Spoof { score: f32, tier: LivenessTier } }`; `pub struct HeadPose { pub yaw: f32, pub pitch: f32, pub roll: f32, pub confidence: f32 }` (degrees); `pub struct TrackObs { id, bbox: RectF, landmarks5: [PointF; 5], det_score: f32, identity: IdentityState, liveness: LivenessState, head_pose: Option<HeadPose>, face_width_px: f32, age_ms: u64, embedding: Option<Vec<f32>> }`; `pub enum CameraHealth { Ok, Lost, Covered, PermissionDenied, Busy, Error(String) }`; `pub struct StageTimings { pub detect_ms: f32, pub recognize_ms: f32, pub liveness_ms: f32, pub pose_ms: f32 }`; `pub struct VisionReport { pub seq: u64, pub ts_ms: u64, pub tracks: Vec<TrackObs>, pub camera_health: CameraHealth, pub liveness_tier: LivenessTier, pub timings: StageTimings }`.
  - `threat`: `pub enum ThreatLevel { None, Notice, Onlooker, Spoof, OwnerAbsent, CameraLost, Manual }` with `Ord` matching SRS FR-PO-002 precedence (`Manual` highest, `None` lowest).
  - All types derive `Clone, Debug, PartialEq, Serialize, Deserialize` (`Eq/Hash/Ord` where sensible). `landmarks5` order: right eye, left eye, nose tip, right mouth corner, left mouth corner (YuNet order).
  - `pub enum CoreError` (thiserror).

- [ ] **Step 1: Tests** in each module: `iou_identical_is_one`, `iou_disjoint_is_zero`, `iou_half_overlap` (two 1×1 rects offset by 0.5 → 1/3), `clamp_unit_clips_to_0_1`, `frame_new_rejects_bad_len`, `mean_luma_of_white_is_255` / `of_black_is_0`, `threat_order_matches_precedence` (sorted ascending == `[None, Notice, OwnerAbsent, Onlooker, Spoof, CameraLost, Manual]`), `vision_report_serde_roundtrip` (serde_json).
- [ ] **Step 2:** `cargo test -p fasec-core` → FAIL (not defined).
- [ ] **Step 3:** Implement. Deps: `serde` (derive), `thiserror`; dev-dep `serde_json`.
- [ ] **Step 4:** `cargo test -p fasec-core` → PASS.
- [ ] **Step 5:** Commit `feat(core): shared geometry, frame, vision and threat types`.

---

### Task 2: `fasec-models` manifest, verification, fetch

**Files:**
- Create: `crates/fasec-models/src/{lib.rs,manifest.rs,verify.rs,fetch.rs}`, `crates/fasec-models/examples/fetch.rs`, `assets/test-assets.toml`
- Modify: `.gitignore` (add `/.assets/`)

**Interfaces:**
- Produces:
  - `pub struct ModelEntry { pub id: String, pub task: String, pub url: String, pub sha256: String, pub file: Option<String>, pub size: Option<u64>, pub license: String, pub source: String, pub default: bool }` (`file` = local file name; needed when `url` is empty, e.g. manually imported exports); `pub struct Manifest { pub model: Vec<ModelEntry> }`; `Manifest::parse(&str) -> Result<Manifest, ModelError>`; `Manifest::load(path) -> Result<Manifest, ModelError>`; `fn get(&self, id: &str) -> Option<&ModelEntry>`; `fn defaults(&self) -> impl Iterator<Item=&ModelEntry>`.
  - `ModelEntry::is_fetchable(&self) -> bool` (url and sha256 non-empty, sha256 is 64 hex chars).
  - `pub fn verify_file(path: &Path, sha256_hex: &str) -> Result<(), ModelError>`.
  - `pub fn model_path(dir: &Path, entry: &ModelEntry) -> Result<PathBuf, ModelError>` → `dir/<entry.file>` if set, else `dir/<last url path segment>`; error if neither gives a name.
  - `pub fn ensure(dir: &Path, entry: &ModelEntry) -> Result<PathBuf, ModelError>`: empty/invalid `sha256` → `NotFetchable`; if the file exists and verifies → return it (even when `url` is empty: manual import); if it exists but mismatches → `HashMismatch` (do not overwrite silently); if missing and `url` empty → `NotFetchable`; else download (`ureq`) to `<file>.part`, verify, rename. Never leaves an unverified file at the final path.
  - `pub enum ModelError { Io(..), Parse(..), NotFetchable { id }, HashMismatch { id_or_path, expected, actual }, Download(..) }`.
  - Example `fetch`: `cargo run -p fasec-models --example fetch [-- --dir models]` fetches all `default = true` entries from `models/manifest.toml` into `models/`, and all entries of `assets/test-assets.toml` (same schema) into `.assets/`.
  - `assets/test-assets.toml`: at least **three public-domain portrait photos** (e.g. official NASA astronaut portraits on Wikimedia Commons / nasa.gov, which are US-government public domain): two different photos of the **same** person and one of a **different** person, each with pinned `url`, `sha256`, `license = "Public domain"`, `source` page. Ids: `face-a-1`, `face-a-2`, `face-b-1`. Use `task = "test_image"`. Compute hashes by downloading once.

- [ ] **Step 1: Tests**: `parse_repo_manifest` (parses `../../models/manifest.toml` via `include_str!`, finds `yunet-2023mar` and `sface-2021dec` as defaults), `empty_url_not_fetchable`, `bad_sha_len_not_fetchable`, `verify_ok` and `verify_mismatch_reports_expected_and_actual` (temp file with known content; use `std::env::temp_dir()` + unique name, no extra dep), `ensure_refuses_missing_file_with_empty_url`, `ensure_refuses_empty_sha`, `ensure_accepts_manually_imported_file` (empty url, `file` set, matching hash → Ok), `ensure_reports_mismatch_on_existing_bad_file`, `ensure_keeps_valid_existing_file_without_network` (pre-write file whose hash matches a synthetic entry with an unreachable url `http://127.0.0.1:9/x` → Ok).
- [ ] **Step 2:** `cargo test -p fasec-models` → FAIL.
- [ ] **Step 3:** Implement. Deps: `serde`, `toml`, `sha2`, `hex`, `ureq`, `thiserror`, `tracing`; example dep `anyhow`.
- [ ] **Step 4:** `cargo test -p fasec-models` → PASS; then `cargo run -p fasec-models --example fetch` → downloads YuNet, SFace, 3 test images, all verified.
- [ ] **Step 5:** Commit `feat(models): manifest parsing, sha256 verification and fetch tool`.

---

### Task 3: `fasec-config` schema (M1 subset)

**Files:**
- Create: `crates/fasec-config/src/{lib.rs,schema.rs}`

**Interfaces:**
- Produces: `pub struct Config { pub version: u32, pub camera: CameraConfig, pub sensitivity: SensitivityConfig, pub runtime: RuntimeConfig, pub owner_absent: OwnerAbsentConfig, pub snapshots: SnapshotsConfig }` with `Default` = SRS §6.2 values. `CameraConfig { device_id: Option<String>, loss_policy: LossPolicy /*Protect|FallbackToNext|NotifyOnly*/, resolution: [u32; 2] /*1280,720*/, fps: u32 /*30*/, rotation: u16 /*0*/, mirror: bool }`; `SensitivityConfig { preset: Preset /*Relaxed|Balanced|Paranoid|Custom*/, max_distance_m: f32 /*2.0*/ }`; `RuntimeConfig { execution_provider: ExecutionProvider /*Auto|DirectMl|Cpu*/, idle_fps: u32 /*5*/ }`; `OwnerAbsentConfig { action: OwnerAbsentAction /*Off|Blur|Lock*/, delay_s: u32 /*30*/ }`; `SnapshotsConfig { enabled: bool /*true*/, retention_days: u32 /*30*/, min_interval_s: u32 /*30*/ }`. Enums serialise `snake_case` exactly as in SRS §6.2 (`"protect"`, `"fallback_to_next"`, `"directml"`, …). Unknown fields: rejected (`deny_unknown_fields`). Missing sections → defaults (`#[serde(default)]`).
- `pub const CURRENT_VERSION: u32 = 1;` `Config::from_toml(&str) -> Result<Config, ConfigError>` (rejects `version > CURRENT_VERSION` and `version == 0`; validates `rotation ∈ {0,90,180,270}`, `fps ∈ 1..=120`, `idle_fps ∈ 1..=30`, `max_distance_m ∈ (0, 10]`); `fn to_toml(&self) -> String`; `Config::load_or_default(path) -> Result<Config, ConfigError>` (missing file → default; invalid file → error, never silently default); `fn save(&self, path)` writes atomically (temp file + rename).
- Profiles/chains are out of scope for this plan (M4).

- [ ] **Step 1: Tests**: `default_roundtrip`, `srs_example_parses` (the `[camera]`, `[sensitivity]`, `[owner_absent]`, `[snapshots]`, `[runtime]` parts of SRS §6.2 as a literal), `missing_sections_use_defaults`, `rejects_future_version`, `rejects_rotation_45`, `rejects_unknown_field`, `load_missing_file_gives_default`, `load_invalid_file_is_error`, `save_then_load_equal`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (deps `serde`, `toml`, `thiserror`). **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(config): versioned config schema with validation`.

---

### Task 4 (lane V): ONNX runtime wrapper and stage traits

**Files:**
- Create: `crates/fasec-vision/src/{lib.rs,error.rs,runtime.rs,traits.rs,image_ops.rs}`

**Interfaces:**
- Consumes: `fasec_core::{Frame, RectF, PointF, HeadPose, LivenessTier}`, `fasec_config::ExecutionProvider`.
- Produces:
  - `pub struct Runtime` with `Runtime::new(ep: ExecutionProvider) -> Result<Runtime, VisionError>`, `fn session(&self, model_path: &Path) -> Result<ort::session::Session, VisionError>`, `fn active_ep(&self) -> &'static str` (`"directml"` or `"cpu"`). `Auto` tries DirectML (Windows only, behind cargo feature `directml`, on by default on Windows via target-specific dependency features) and falls back to CPU, logging the reason.
  - `traits.rs`: `pub struct Detection { pub bbox: RectF, pub landmarks5: [PointF; 5], pub score: f32 }`; `pub trait FaceDetector: Send { fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>, VisionError>; }`; `pub trait FaceEmbedder: Send { fn embed(&mut self, frame: &Frame, det: &Detection) -> Result<Vec<f32>, VisionError>; fn model_id(&self) -> &str; }`; `pub trait LivenessModel: Send { fn score(&mut self, frame: &Frame, det: &Detection) -> Result<f32 /*0..1, 1 = live*/, VisionError>; fn tier(&self) -> LivenessTier; }`; `pub trait PoseEstimator: Send { fn estimate(&mut self, frame: &Frame, det: &Detection) -> Result<HeadPose, VisionError>; }`.
  - `image_ops.rs`: `fn resize_rgb(frame: &Frame, w: u32, h: u32) -> Vec<u8>`; `fn crop_scaled(frame: &Frame, bbox_px: [f32;4], scale: f32, out: u32) -> Vec<u8>` (expands box around centre by `scale`, pads with black outside the frame, resizes to `out×out`); `fn to_nchw_f32(rgb: &[u8], w: u32, h: u32, bgr: bool) -> ndarray::Array4<f32>` (values 0..255, no normalisation).
  - `pub enum VisionError` (thiserror): `Model`, `Ort(String)`, `Input`, `Io`.

- [ ] **Step 1: Tests**: `to_nchw_shape_and_channel_order` (2×1 image with known pixels, bgr=true swaps), `crop_scaled_pads_outside_frame` (box at corner → padded pixels are 0, no panic), `crop_scaled_centre_matches_source`, `runtime_cpu_loads_yunet` (`#[ignore]`, loads `models/face_detection_yunet_2023mar.onnx` after `verify_file`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (deps `ort`, `ndarray`, `image`, `thiserror`, `tracing`, path deps `fasec-core`, `fasec-config`). If the latest `ort` is a pre-release only, pin the newest `2.0.0-rc.*` explicitly. **Step 4:** PASS incl. `--ignored`; report `active_ep()` seen on this machine.
- [ ] **Step 5:** Commit `feat(vision): ort runtime, stage traits and image ops`.

---

### Task 5 (lane V): YuNet face detector

**Files:**
- Create: `crates/fasec-vision/src/detect/{mod.rs,yunet.rs,nms.rs}`

**Interfaces:**
- Produces: `pub struct YuNet` implementing `FaceDetector`; `YuNet::new(rt: &Runtime, model: &Path, cfg: YuNetConfig) -> Result<YuNet, VisionError>`; `pub struct YuNetConfig { pub input_long_side: u32 /*320*/, pub score_threshold: f32 /*0.6*/, pub nms_iou: f32 /*0.3*/, pub top_k: usize /*50*/ }` with `Default`. `nms::nms(dets: Vec<Detection>, iou: f32, top_k: usize) -> Vec<Detection>`.
- Decoding follows OpenCV `FaceDetectorYN` for the 2023mar model (read `modules/objdetect/src/face_detect.cpp` in OpenCV ≥ 4.8 as the reference): input BGR float 0..255, NCHW, padded so H and W are multiples of 32; outputs `cls_{8,16,32}`, `obj_{8,16,32}`, `bbox_{8,16,32}`, `kps_{8,16,32}`; score = `sqrt(clamp(cls)·clamp(obj))`; centre = `(col + dx)·stride`, `(row + dy)·stride`; size = `exp(dw)·stride`, `exp(dh)·stride`; landmarks = `(col + kx)·stride`, `(row + ky)·stride`. Output coordinates are mapped back through scale + padding to normalised 0..1 of the original frame.

- [ ] **Step 1: Tests**: `nms_suppresses_overlaps`, `nms_keeps_top_k`, `detects_face_a1` (`#[ignore]`: `.assets` image `face-a-1` → exactly 1 detection, score ≥ 0.8, bbox inside 0..1, eyes above mouth: `landmarks5[0].y < landmarks5[3].y`), `detects_on_non_multiple_of_32_sizes` (`#[ignore]`: same image resized to 1280×720 letterboxed, 640×360 and rotated 90° → 1 detection each, bbox centre consistent with the original within 0.05), `empty_on_black_frame` (`#[ignore]`: 640×480 black → 0 detections).
- [ ] **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS incl. `--ignored`; report detect ms on CPU and DirectML.
- [ ] **Step 5:** Commit `feat(vision): YuNet face detector with NMS`.

---

### Task 6 (lane V): SORT-style tracker

**Files:**
- Create: `crates/fasec-vision/src/track.rs`

**Interfaces:**
- Produces: `pub struct Tracker` with `Tracker::new(cfg: TrackerConfig)`, `fn update(&mut self, ts_ms: u64, dets: &[Detection]) -> Vec<TrackedDet>`; `pub struct TrackedDet { pub id: TrackId, pub det: Detection, pub age_ms: u64, pub is_new: bool }`; `pub struct TrackerConfig { pub iou_match: f32 /*0.3*/, pub track_ttl_ms: u64 /*1000, SRS §5.1*/ }`.
- Constant-velocity prediction on box centre/size (simple per-track velocity; a full Kalman is not required), assignment by maximising IoU with `pathfinding::kuhn_munkres` on an integer-scaled IoU matrix, pairs below `iou_match` rejected. Track IDs monotonically increase, never reused. Unmatched tracks survive until `track_ttl_ms` without a match, but are only returned when matched in this update.

- [ ] **Step 1: Tests**: `single_box_keeps_id_while_moving`, `two_crossing_boxes_keep_ids` (boxes swap sides over 10 steps without overlap swap), `new_box_gets_new_id_and_is_new`, `track_expires_after_ttl`, `track_survives_short_gap` (missing 1 frame within ttl → same id), `ids_never_reused`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (dep `pathfinding`). **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(vision): IoU tracker with Hungarian assignment`.

---

### Task 7 (lane V): Alignment and SFace embedder

**Files:**
- Create: `crates/fasec-vision/src/recognize/{mod.rs,align.rs,sface.rs}`

**Interfaces:**
- Produces: `align::similarity_transform(src: &[PointF;5], dst: &[PointF;5]) -> nalgebra::Matrix2x3<f32>` (Umeyama, no reflection); `align::warp_affine_rgb(frame: &Frame, m: &Matrix2x3<f32>, out_w: u32, out_h: u32) -> Vec<u8>` (bilinear, black outside frame); `align::ARCFACE_112: [PointF; 5] = [(38.2946,51.6963),(73.5318,51.5014),(56.0252,71.7366),(41.5493,92.3655),(70.7299,92.2041)]`; `pub struct SFace` implementing `FaceEmbedder` (`model_id() == "sface-2021dec"`), output L2-normalised 128-d; `pub fn cosine(a: &[f32], b: &[f32]) -> f32`; `pub const SFACE_COSINE_THRESHOLD: f32 = 0.363;`.
- SFace input: aligned 112×112, RGB (OpenCV uses `swapRB=true` from BGR), float 0..255, NCHW.

- [ ] **Step 1: Tests**: `umeyama_recovers_known_similarity` (rotate 10°, scale 1.5, translate → recovered within 1e-3), `warp_identity_preserves_pixels`, `warp_outside_frame_is_black_no_panic` (face landmarks near the edge), `cosine_basic` (identical = 1, orthogonal = 0), `same_person_above_threshold` (`#[ignore]`: face-a-1 vs face-a-2 > 0.363), `different_person_below_threshold` (`#[ignore]`: face-a-1 vs face-b-1 < 0.363), `embedding_is_unit_norm` (`#[ignore]`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (dep `nalgebra`). **Step 4:** PASS incl. `--ignored`; report the three cosine values.
- [ ] **Step 5:** Commit `feat(vision): 5-point alignment and SFace embeddings`.

---

### Task 8 (lane V): Liveness (MiniFASNet) and coarse head pose

**Files:**
- Create: `crates/fasec-vision/src/liveness/{mod.rs,minifasnet.rs}`, `crates/fasec-vision/src/attention/{mod.rs,coarse_pose.rs}`, `tools/export_minifasnet.py` (only if needed, see Step 3)
- Modify: `models/manifest.toml` (fill `minifasnet-v2` / `minifasnet-v1se` `sha256` and `file` (`minifasnet_v2.onnx`, `minifasnet_v1se.onnx`); `url` filled only if a permissively licensed hosted ONNX export exists)

**Interfaces:**
- Produces:
  - `pub struct MiniFasNet` implementing `LivenessModel` (`tier() == LivenessTier::Rgb`): runs `minifasnet-v2` on `crop_scaled(.., scale 2.7, 80)` and `minifasnet-v1se` on `crop_scaled(.., scale 4.0, 80)`, softmax each 3-class output, sums them, returns `live = summed[1] / 2.0`. Input preprocessing must match the upstream `Silent-Face-Anti-Spoofing` test code (check its `src/data_io/transform.py` for whether values are scaled; BGR order as upstream). `MiniFasNet::new(rt, v2_path, v1se_path)`.
  - `pub struct CoarsePose` implementing `PoseEstimator` from the 5 landmarks only (no model): `yaw = atan2(nose.x − eye_mid.x, inter_eye_dist) · K_YAW`, `pitch = atan2(nose.y − eye_mouth_mid.y, eye_mouth_dist) · K_PITCH` with constants chosen so a synthetic frontal face gives 0 and the sign convention is: **positive yaw = face turned to the camera's right (nose moves to image right), positive pitch = looking down**; `roll = atan2(dy, dx)` of the eye line; `confidence = det.score`. `pub fn face_width_px(det: &Detection, frame: &Frame) -> f32`.
  - Ruling recorded in plan: coarse pose is the M1 default; SP-4 may replace it with 6DRepNet/Face Landmarker later behind the same trait.
- Getting MiniFASNet ONNX: first look for an existing ONNX export of the upstream Apache-2.0 weights (Hugging Face / GitHub) whose provenance is clear; otherwise write `tools/export_minifasnet.py` (clones upstream at a pinned commit, `torch.onnx.export` both models with fixed 1×3×80×80 input, prints SHA-256) and run it once. Weights stay out of git either way.

- [ ] **Step 1: Tests**: `pose_frontal_synthetic_is_near_zero` (|yaw|,|pitch|,|roll| < 3°), `pose_nose_right_gives_positive_yaw`, `pose_nose_low_gives_positive_pitch`, `pose_tilted_eyes_gives_roll`, `liveness_score_in_unit_range` (`#[ignore]`: face-a-1), `liveness_deterministic` (`#[ignore]`: same input twice → identical score), `liveness_edge_face_no_panic` (`#[ignore]`: detection box partly outside frame).
- [ ] **Step 2:** FAIL. **Step 3:** Implement + obtain models, update manifest. **Step 4:** PASS incl. `--ignored`; report live score of face-a-1 (a printed photo of a photo is not available in tests — note this limit in the report).
- [ ] **Step 5:** Commit `feat(vision): MiniFASNet liveness and coarse 5-point head pose`.

---

### Task 9 (lane V): Pipeline and scheduler

**Files:**
- Create: `crates/fasec-vision/src/pipeline.rs`, `crates/fasec-vision/benches/stages.rs`

**Interfaces:**
- Consumes: all traits (Task 4), `Tracker` (Task 6), `fasec_core::VisionReport`.
- Produces: `pub struct Pipeline` with `Pipeline::new(det: Box<dyn FaceDetector>, emb: Box<dyn FaceEmbedder>, live: Box<dyn LivenessModel>, pose: Box<dyn PoseEstimator>, cfg: PipelineConfig)`; `fn process(&mut self, frame: &Frame, health: CameraHealth) -> Result<VisionReport, VisionError>`; `fn desired_interval_ms(&self) -> u64` (= `1000 / idle_fps` when the last report had no tracks, else `1000 / active_fps`); `pub struct PipelineConfig { pub recog_interval_ms: u64 /*1000*/, pub liveness_interval_ms: u64 /*2000*/, pub idle_fps: u32 /*5*/, pub active_fps: u32 /*15*/, pub rerecog_iou: f32 /*0.5*/, pub tracker: TrackerConfig }`; `Pipeline::from_models(rt: &Runtime, models_dir: &Path, cfg) -> Result<Pipeline, VisionError>` wiring YuNet+SFace+MiniFasNet+CoarsePose with `verify_file` against the manifest first.
- Scheduling (SRS §3.3): embedding + liveness run for a track when `is_new`, when their interval elapsed (by `frame.ts_ms`, never wall clock), or when IoU between current box and box at last recognition < `rerecog_iou`. In between, the track carries its last embedding/liveness. Identity in the report stays `IdentityState::Pending` (gallery matching is M2); `TrackObs.embedding` carries the latest embedding. Liveness → `Live`/`Spoof` with threshold 0.75 (Balanced, SRS §5.5) exposed as `PipelineConfig::liveness_threshold`.
- If `health != Ok`, `process` returns a report with no tracks and that health, without running models.

- [ ] **Step 1: Tests** (with mock stages counting calls, synthetic frames, simulated `ts_ms`): `new_track_runs_embed_and_liveness_once`, `embed_rate_limited_by_interval`, `big_box_jump_forces_rerecognition`, `health_not_ok_skips_models`, `idle_then_face_switches_interval` (Review Focus 5: 20 empty frames → `desired_interval_ms == 200`; then a face → 66, new track recognised on that frame), `report_timings_populated`, `pipeline_on_face_a1` (`#[ignore]`, real models: 1 track, embedding len 128, liveness not `Unknown`, head_pose Some).
- [ ] **Step 2:** FAIL. **Step 3:** Implement; criterion bench `stages` timing detect/embed/liveness/pose on face-a-1 (skips with a message if models missing). **Step 4:** PASS incl. `--ignored`; run `cargo bench -p fasec-vision` and put the numbers (CPU and DirectML) in the report.
- [ ] **Step 5:** Commit `feat(vision): scheduled pipeline producing VisionReport, stage benches`.

---

### Task 10 (lane C): Camera enumeration, capture thread, latest-frame slot

**Files:**
- Create: `crates/fasec-camera/src/{lib.rs,error.rs,slot.rs,device.rs,capture.rs,transform.rs}`

**Interfaces:**
- Consumes: `fasec_core::Frame`, `fasec_config::CameraConfig`.
- Produces:
  - `slot.rs` (platform-neutral, unit-tested): `pub struct LatestFrame` (`Mutex<Option<Frame>>` + `Condvar`): `fn put(&self, f: Frame)` (never blocks beyond the mutex, replaces older), `fn take_newer(&self, after_seq: u64, timeout: Duration) -> Option<Frame>` (returns only a frame with `seq > after_seq`).
  - `transform.rs`: `fn apply(rgb: &[u8], w: u32, h: u32, rotation: u16, mirror: bool) -> (Vec<u8>, u32, u32)`.
  - `device.rs`: `pub struct CameraInfo { pub id: String /*stable symbolic link / device path*/, pub name: String, pub formats: Vec<CameraFormat> }`, `pub struct CameraFormat { pub width: u32, pub height: u32, pub fps: u32 }`, `pub fn list() -> Result<Vec<CameraInfo>, CameraError>`.
  - `capture.rs`: `pub struct Capture` with `Capture::start(info_id: &str, cfg: &CameraConfig, slot: Arc<LatestFrame>) -> Result<Capture, CameraError>` (own thread, closest format to `cfg.resolution/fps`, converts to RGB8, applies transform, stamps `seq` (monotonic) and `ts_ms` (monotonic ms since capture start)), `fn stop(self)`, `fn last_frame_at(&self) -> Option<Instant>`.
  - Use `nokhwa` (Media Foundation backend). It MUST be verified that `id` is the MF symbolic link, stable across replug and independent of enumeration order; if `nokhwa` does not expose it, get it via `windows` crate `Windows.Devices.Enumeration.DeviceInformation::FindAllAsync(DeviceClass::VideoCapture)` and map by name/index — record the decision in the report.
  - `pub enum CameraError` (thiserror): `NotFound { id }`, `Busy`, `PermissionDenied`, `Backend(String)`.

- [ ] **Step 1: Tests**: `slot_take_newer_returns_only_newer`, `slot_put_replaces_old` (put seq 1,2,3 → take_newer(0) = 3), `slot_timeout_returns_none`, `slot_producer_never_blocks_on_slow_consumer` (Review Focus 3: producer thread puts 1000 frames while consumer sleeps; producer finishes in < 200 ms), `transform_rotate_90_dims_and_pixel`, `transform_mirror`, `list_cameras_smoke` (`#[ignore = "needs a camera"]`), `capture_delivers_frames` (`#[ignore = "needs a camera"]`: ≥ 10 frames in 2 s with increasing seq).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (deps `nokhwa` with MSMF + output-format features, `windows` if needed, `thiserror`, `tracing`). **Step 4:** PASS; run ignored tests if this machine has a camera and report what `list()` printed (names + ids).
- [ ] **Step 5:** Commit `feat(camera): enumeration, capture thread and latest-frame slot`.

---

### Task 11 (lane C): Camera health, hot-plug, permission detection

**Files:**
- Create: `crates/fasec-camera/src/{health.rs,watch.rs,permission.rs}`

**Interfaces:**
- Produces:
  - `health.rs` (platform-neutral): `pub struct HealthMonitor` with `HealthMonitor::new(cfg: HealthConfig)`, `fn on_frame(&mut self, now_ms: u64, mean_luma: f32)`, `fn evaluate(&self, now_ms: u64) -> CameraHealth`; `pub struct HealthConfig { pub frame_timeout_ms: u64 /*1500, FR-CAM-005*/, pub black_luma: f32 /*10.0*/, pub black_ms: u64 /*2000, FR-CAM-009*/ }`. No frame yet for > timeout after start → `Lost`.
  - `watch.rs`: `pub enum CameraEvent { Added(String), Removed(String) }`; `pub fn watch(tx: crossbeam_channel::Sender<CameraEvent>) -> Result<WatchGuard, CameraError>` using `Windows.Devices.Enumeration.DeviceWatcher` for `DeviceClass::VideoCapture`; ids match `CameraInfo::id`. Dropping `WatchGuard` stops the watcher.
  - `permission.rs`: `pub fn desktop_camera_access() -> Access /*Allowed|DeniedByUser|DeniedBySystem|Unknown*/` reading `HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged` `Value` and the HKLM equivalent (FR-CAM-008).
  - `Capture::start` maps backend "access denied" errors to `CameraError::PermissionDenied` and "in use" to `CameraError::Busy` where distinguishable.

- [ ] **Step 1: Tests**: `health_ok_with_regular_frames`, `health_lost_after_timeout`, `health_lost_if_never_started`, `health_covered_after_2s_black`, `health_recovers_after_light`, `health_black_shorter_than_2s_is_ok`, `permission_reads_without_panic` (runs on Windows, any value accepted), `watch_starts_and_stops` (`#[ignore = "needs Windows device stack"]`).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (deps `crossbeam-channel`, `windows` with the needed features, `winreg` only if `windows` registry APIs are clumsy). **Step 4:** PASS.
- [ ] **Step 5:** Commit `feat(camera): health monitor, hot-plug watcher, permission check`.

---

### Task 12 (lane C): Camera probe spikes (SP-5, SP-6, SP-7)

**Files:**
- Create: `crates/fasec-camera/examples/camera_probe.rs`, `crates/fasec-camera/examples/ir_probe.rs`, `docs/spikes/SP-5-camera-sharing.md`, `docs/spikes/SP-6-nokhwa.md`, `docs/spikes/SP-7-ir.md`

**Interfaces:**
- Consumes: Tasks 10–11.
- `camera_probe`: `cargo run -p fasec-camera --example camera_probe -- [list|run <id>|watch]`. `list` prints id/name/formats; `run` captures, prints FPS, mean luma, health every second; `watch` prints hot-plug events. Exit cleanly on Ctrl+C.
- `ir_probe`: lists `MediaFrameSourceGroup`s with each source's `SourceKind` (Color/Infrared/Depth) and device id; `ir_probe read <group-index>` opens the Infrared source with `MediaCapture` (`SharingMode::SharedReadOnly` first, then `ExclusiveControl`), reads 30 frames via `MediaFrameReader`, prints resolution, pixel format, mean intensity, and saves nothing to disk.
- Each `SP-*.md` uses: **Question**, **How to run** (exact commands), **Results table** (machine, Windows build, camera, outcome — rows left empty for the human), **Decision** (empty), and anything the implementer could already observe on this machine.

- [ ] **Step 1:** Implement both examples (`anyhow` allowed). **Step 2:** `cargo build -p fasec-camera --examples` clean; run `camera_probe list` and `ir_probe` here and paste outputs into the SP docs' "observed on dev machine" section. **Step 3:** Commit `feat(camera): probe examples and spike docs SP-5/6/7`.

---

### Task 13 (lane O): Overlay blur and z-order spike (SP-1, SP-2)

**Files:**
- Create: `crates/fasec-overlay/examples/sp1_overlay.rs`, `docs/spikes/SP-1-overlay-blur.md`, `docs/spikes/SP-2-z-order.md`

**Interfaces:**
- `cargo run -p fasec-overlay --example sp1_overlay -- --mode <acrylic|backdrop|solid> [--seconds 15] [--click-through]` creates one window per monitor (`EnumDisplayMonitors`, per-monitor-DPI-aware v2 via manifest-free `SetProcessDpiAwarenessContext`), styles `WS_POPUP`, ex-styles `WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE` (+ `WS_EX_LAYERED | WS_EX_TRANSPARENT` with `--click-through`), `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`, then:
  - `acrylic`: undocumented `SetWindowCompositionAttribute` with `ACCENT_ENABLE_ACRYLICBLURBEHIND` (load from `user32.dll` dynamically).
  - `backdrop`: `DwmSetWindowAttribute(DWMWA_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW)` + `DwmExtendFrameIntoClientArea(-1)` (Windows 11 22H2+; print a clear message if unsupported).
  - `solid`: opaque black.
- Re-asserts `HWND_TOPMOST` every 250 ms (FR-OV-007). Prints Windows build number, whether "transparency effects" is on (`HKCU\...\Themes\Personalize\EnableTransparency`), and per-monitor rects/DPI. Closes after `--seconds` or `Esc` via a low-level keyboard hook.
- SP docs: SP-1 table rows = {Win10, Win11} × {acrylic, backdrop} × {transparency on/off} × {overlay inactive while another app is focused} → "live blur? / solid fallback? / transparent (bad)?". SP-2 table rows = Start menu, Alt-Tab, Win+Tab, Action Center, toast, Task Manager "always on top", full-screen video, UAC prompt → "above overlay? y/n".

- [ ] **Step 1:** Implement (deps `windows` with Win32 UI/DWM/Gdi features, `anyhow`). **Step 2:** `cargo build -p fasec-overlay --examples` clean; run each mode for 5 s here and record what was observed (build number, transparency setting, any error) in the docs. **Step 3:** Commit `feat(overlay): SP-1/SP-2 overlay spike program and docs`.

---

### Task 14 (lane O): Capture-based blur spike (SP-1b)

**Files:**
- Create: `crates/fasec-overlay/examples/sp1b_capture_blur.rs`, `docs/spikes/SP-1b-capture-blur.md`

**Interfaces:**
- `cargo run -p fasec-overlay --example sp1b_capture_blur -- [--monitor 0] [--seconds 15] [--downscale 8]`: overlay window as in Task 13 (excluded from capture), captures the monitor with Windows Graphics Capture (`windows-capture` crate), CPU blur by downscaling by `--downscale` and upscaling (bilinear), pushes the image to the overlay with `UpdateLayeredWindow`. Prints every second: capture FPS, blur ms, present ms, process CPU %.
- Purpose is measurement, not production quality. The overlay MUST NOT appear in its own capture (verify by the absence of a "hall of mirrors" effect and say so in the doc).
- SP-1b doc: numbers on this machine at 1080p (and the monitor's native resolution) for downscale 4/8/16, plus the decision prompt: "GPU blur needed?".

- [ ] **Step 1:** Implement (deps `windows-capture`, `windows`, `anyhow`). **Step 2:** Build clean; run 10 s here; record numbers. **Step 3:** Commit `feat(overlay): SP-1b capture-based blur spike`.

---

### Task 15: `fasec-replay` CLI

**Files:**
- Create: `crates/fasec-replay/src/{main.rs,input.rs,output.rs}`

**Interfaces:**
- Consumes: `Pipeline::from_models`, `Runtime`, `fasec_core::{Frame, VisionReport, CameraHealth}`.
- Produces CLI (use `clap` derive): `fasec-replay run --input <dir|video> [--fps 15] [--models models] [--ep auto|cpu|directml] [--out report.jsonl]`.
  - `<dir>`: sorted `*.png|*.jpg` frames. `<video>`: decoded with the `ffmpeg` CLI (`ffmpeg -i <video> -f rawvideo -pix_fmt rgb24 -`, dimensions via `ffprobe`), error message if ffmpeg is missing.
  - Simulated clock: `ts_ms = index * 1000 / fps`; `seq = index`. Every frame is processed (no skipping; `desired_interval_ms` is reported, not applied, so output is deterministic).
  - Output: one JSON `VisionReport` per line (stdout if no `--out`), embeddings omitted from JSON unless `--with-embeddings`.

- [ ] **Step 1: Tests**: `dir_input_sorted_and_timestamped` (temp dir with 3 generated PNGs named out of order), `jsonl_one_line_per_frame` (with a mock pipeline via an internal `trait Process` seam), `deterministic_output` (`#[ignore]`: run twice on `.assets` images → byte-identical files), `missing_ffmpeg_is_clear_error`.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (deps `clap`, `serde_json`, `image`, `anyhow`). **Step 4:** PASS incl. `--ignored`.
- [ ] **Step 5:** Commit `feat(replay): offline pipeline runner with JSONL output`.

---

### Task 16: Live debug viewer (M1 exit check)

**Files:**
- Create: `crates/fasec-ui/examples/live_debug.rs`

**Interfaces:**
- Consumes: `fasec_camera::{list, Capture, LatestFrame, HealthMonitor}`, `fasec_vision::Pipeline`.
- `cargo run -p fasec-ui --example live_debug --release`: eframe window; camera combo box (from `list()`); a worker thread takes frames via `take_newer`, runs `Pipeline::process`, publishes the latest `(Frame, VisionReport)`; UI draws the frame as a texture with boxes, track id, `det_score`, liveness state+score, yaw/pitch/roll, face width px; side panel shows camera health, active EP, FPS processed, per-stage ms (`StageTimings`). Switching camera stops the old `Capture` before starting the new one.

- [ ] **Step 1:** Implement (deps `eframe`, `anyhow`). **Step 2:** `cargo build -p fasec-ui --examples --release` clean; if a camera is present, run 10 s and report processed FPS and stage ms. **Step 3:** Commit `feat(ui): live debug viewer example`.

---

### Task 17: CI and docs wrap-up

**Files:**
- Modify: `.github/workflows/ci.yml` (Linux job: ensure neutral crates still build without Windows deps; add `cargo build --workspace --examples` to the Windows job), `README.md` (dev quick start: fetch models, run replay, run live_debug, run spikes), `docs/ROADMAP.md` (tick completed M0/M1 boxes; leave human-only spike items unticked), `docs/spikes/README.md` (index + which spikes still need a human run).

- [ ] **Step 1:** Update files. **Step 2:** Run locally: `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace`, `cargo build --workspace --examples`; and verify Linux-neutral crates have no `windows`/`nokhwa` in their dependency tree: `cargo tree -p fasec-vision -p fasec-replay -p fasec-core -p fasec-config -p fasec-models -e normal | findstr /i "windows nokhwa"` → only `windows-sys`/`windows-targets` pulled transitively by std-adjacent crates are acceptable; record the output. **Step 3:** Commit `chore: CI examples build, dev docs, roadmap ticks`.
