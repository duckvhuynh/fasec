//! Local paired RGB + IR recordings, the input for tuning IR liveness (Task 7).
//!
//! Biometric-adjacent: `ir_debug` writes one only while its Record toggle is on, and only under
//! [`recordings_root`] (`%LOCALAPPDATA%\fasec\recordings`). A session is one new directory
//! `<label>-<yyyymmdd-hhmmss>` (UTC) holding, for step `n` (a lit IR frame and the RGB frame
//! processed with it): `rgb-<n>.png` (RGB8), `ir-lit-<n>.png`, `ir-amb-<n>.png` when the lit
//! frame had an ambient partner (Gray8), and line `n` of `frames.jsonl` ([`Step`]). No
//! embeddings, no names.

use std::ffi::OsString;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use fasec_core::{Frame, IrFrame};
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{ExtendedColorType, ImageEncoder};
use serde::{Deserialize, Serialize};

pub const FRAMES_FILE: &str = "frames.jsonl";
/// A session stops recording after this many steps (~4 min at the ASUS lit rate in debug).
pub const MAX_STEPS: u64 = 600;
/// Longest label kept by [`sanitize_label`].
const MAX_LABEL: usize = 40;
/// Labels offered in the UI; free text is allowed too.
pub const PRESETS: [&str; 8] = [
    "real-30cm",
    "real-50cm",
    "real-80cm",
    "real-turning",
    "phone-photo",
    "phone-video",
    "monitor",
    "printed-photo",
];

/// `%LOCALAPPDATA%\fasec\recordings`, the only place recordings are written.
pub fn recordings_root() -> Option<PathBuf> {
    root_under(std::env::var_os("LOCALAPPDATA"))
}

/// `None` for an unset, empty or relative `LOCALAPPDATA`, which would resolve against the
/// current directory.
fn root_under(local_app_data: Option<OsString>) -> Option<PathBuf> {
    let dir = PathBuf::from(local_app_data?);
    dir.is_absolute()
        .then(|| dir.join("fasec").join("recordings"))
}

/// `label` lower-cased with every run of characters outside `[a-z0-9]` turned into one `-`,
/// trimmed of `-` and cut to 40 characters; `None` when nothing is left. The result is a
/// single path component, so a label can never leave the recordings directory.
pub fn sanitize_label(label: &str) -> Option<String> {
    let mut out = String::new();
    for c in label.chars().flat_map(char::to_lowercase) {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    // only ASCII is left, so any byte index is a char boundary
    out.truncate(MAX_LABEL);
    let out = out.trim_end_matches('-');
    (!out.is_empty()).then(|| out.to_owned())
}

/// `t` as `yyyymmdd-hhmmss` in UTC.
pub fn stamp(t: SystemTime) -> String {
    let secs = t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let (days, rem) = (secs / 86_400, secs % 86_400);
    // civil-from-days (H. Hinnant), for days since 1970-01-01
    let z = days + 719_468;
    let (era, doe) = (z / 146_097, z % 146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    let (h, m, s) = (rem / 3600, rem % 3600 / 60, rem % 60);
    format!("{year:04}{month:02}{day:02}-{h:02}{m:02}{s:02}")
}

/// One line of `frames.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Step {
    /// 0, 1, 2, ... : the step's file names and line number.
    pub n: u64,
    pub rgb_seq: u64,
    /// RGB camera clock; replay feeds it to the pipeline as `Frame::ts_ms`.
    pub rgb_ts_ms: u64,
    pub rgb_w: u32,
    pub rgb_h: u32,
    pub ir_seq: u64,
    /// IR camera clock, never compared with `rgb_ts_ms`.
    pub ir_ts_ms: u64,
    pub ir_w: u32,
    pub ir_h: u32,
    pub ir_illuminated: bool,
    /// The lit frame's ambient (unlit) partner, if it had one.
    pub ambient: Option<AmbientMeta>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AmbientMeta {
    pub seq: u64,
    pub ts_ms: u64,
    pub illuminated: bool,
}

/// One recording session being written.
pub struct Recorder {
    dir: PathBuf,
    jsonl: BufWriter<File>,
    steps: u64,
    limit: u64,
}

impl Recorder {
    /// Creates the new session directory `<root>/<label>-<stamp(now)>`; an existing one is an
    /// error, never reused.
    pub fn create(root: &Path, label: &str, now: SystemTime) -> Result<Recorder> {
        let label = sanitize_label(label).context("the label needs a letter or digit")?;
        std::fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let dir = root.join(format!("{label}-{}", stamp(now)));
        std::fs::create_dir(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(FRAMES_FILE);
        let jsonl = File::create_new(&path)
            .map(BufWriter::new)
            .with_context(|| format!("creating {}", path.display()))?;
        Ok(Recorder {
            dir,
            jsonl,
            steps: 0,
            limit: MAX_STEPS,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Steps written so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Writes one step: its PNGs, then its `frames.jsonl` line (flushed), so every line has its
    /// files. `Ok(false)` and nothing written once `MAX_STEPS` are in.
    pub fn write(&mut self, rgb: &Frame, lit: &IrFrame, ambient: Option<&IrFrame>) -> Result<bool> {
        if self.steps >= self.limit {
            return Ok(false);
        }
        let n = self.steps;
        let at = |name: &str| self.dir.join(format!("{name}-{n}.png"));
        let rgb8 = ExtendedColorType::Rgb8;
        write_png(&at("rgb"), &rgb.data, rgb.width, rgb.height, rgb8)?;
        let grey = |f: &IrFrame, name| {
            write_png(&at(name), &f.data, f.width, f.height, ExtendedColorType::L8)
        };
        grey(lit, "ir-lit")?;
        if let Some(a) = ambient {
            grey(a, "ir-amb")?;
        }
        let step = Step {
            n,
            rgb_seq: rgb.seq,
            rgb_ts_ms: rgb.ts_ms,
            rgb_w: rgb.width,
            rgb_h: rgb.height,
            ir_seq: lit.seq,
            ir_ts_ms: lit.ts_ms,
            ir_w: lit.width,
            ir_h: lit.height,
            ir_illuminated: lit.illuminated,
            ambient: ambient.map(|a| AmbientMeta {
                seq: a.seq,
                ts_ms: a.ts_ms,
                illuminated: a.illuminated,
            }),
        };
        serde_json::to_writer(&mut self.jsonl, &step)?;
        self.jsonl.write_all(b"\n")?;
        self.jsonl.flush()?;
        self.steps += 1;
        Ok(true)
    }
}

/// Fast compression: the writer must keep up with the lit frames in a debug build.
fn write_png(path: &Path, data: &[u8], w: u32, h: u32, ty: ExtendedColorType) -> Result<()> {
    let file = File::create_new(path).with_context(|| format!("creating {}", path.display()))?;
    PngEncoder::new_with_quality(
        BufWriter::new(file),
        CompressionType::Fast,
        FilterType::Adaptive,
    )
    .write_image(data, w, h, ty)
    .with_context(|| format!("writing {}", path.display()))
}

/// A recorded step as read back: its metadata and IR pair; the RGB8 pixels come beside it.
pub struct Recorded {
    pub step: Step,
    pub lit: IrFrame,
    pub ambient: Option<IrFrame>,
}

/// The steps of the session in `dir`, in `frames.jsonl` order. The jsonl is parsed up front;
/// each step's PNGs are decoded when the iterator reaches it. An unparseable last line (the
/// recorder killed mid-write) is skipped with a warning; any earlier one is an error.
pub fn read(dir: &Path) -> Result<impl Iterator<Item = Result<(Vec<u8>, Recorded)>> + use<>> {
    let path = dir.join(FRAMES_FILE);
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .collect();
    let mut steps = Vec::with_capacity(lines.len());
    for (k, &(i, l)) in lines.iter().enumerate() {
        match serde_json::from_str::<Step>(l) {
            Ok(s) => steps.push(s),
            Err(e) if k + 1 == lines.len() => {
                eprintln!(
                    "warning: skipped {} line {} (cut short?): {e}",
                    path.display(),
                    i + 1
                );
            }
            Err(e) => {
                return Err(e).with_context(|| format!("{} line {}", path.display(), i + 1));
            }
        }
    }
    ensure!(!steps.is_empty(), "no steps in {}", path.display());
    let dir = dir.to_owned();
    Ok(steps.into_iter().map(move |s| load(&dir, s)))
}

/// Decodes step `s`'s PNGs and checks each against its line.
fn load(dir: &Path, s: Step) -> Result<(Vec<u8>, Recorded)> {
    let open = |name: &str| {
        let path = dir.join(format!("{name}-{}.png", s.n));
        image::open(&path).with_context(|| format!("decoding {}", path.display()))
    };
    let size = |name: &str, got: (u32, u32), want: (u32, u32)| {
        ensure!(
            got == want,
            "{name}-{}.png is {got:?}, its line says {want:?}",
            s.n
        );
        Ok(())
    };
    let rgb = open("rgb")?.into_rgb8();
    size("rgb", rgb.dimensions(), (s.rgb_w, s.rgb_h))?;
    let lit = open("ir-lit")?.into_luma8();
    size("ir-lit", lit.dimensions(), (s.ir_w, s.ir_h))?;
    let lit = IrFrame::new(
        s.ir_seq,
        s.ir_ts_ms,
        s.ir_w,
        s.ir_h,
        s.ir_illuminated,
        lit.into_raw(),
    )?;
    let ambient = match s.ambient {
        None => None,
        Some(a) => {
            let img = open("ir-amb")?.into_luma8();
            let (w, h) = img.dimensions();
            Some(IrFrame::new(
                a.seq,
                a.ts_ms,
                w,
                h,
                a.illuminated,
                img.into_raw(),
            )?)
        }
    };
    Ok((
        rgb.into_raw(),
        Recorded {
            step: s,
            lit,
            ambient,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An RGB frame and a Gray8 IR frame whose pixels and times identify them.
    fn rgb(seq: u64, w: u32, h: u32) -> Frame {
        let data = (0..w * h * 3).map(|i| (i as u64 + seq) as u8).collect();
        Frame::new(seq, 1000 + seq * 66, w, h, data).unwrap()
    }

    fn ir(seq: u64, w: u32, h: u32, lit: bool) -> IrFrame {
        let data = (0..w * h).map(|i| (i as u64 * 3 + seq) as u8).collect();
        IrFrame::new(seq, 50_000 + seq * 64, w, h, lit, data).unwrap()
    }

    #[test]
    fn labels_are_one_safe_path_component() {
        for (raw, want) in [
            ("real-30cm", Some("real-30cm")),
            ("Real 30cm!", Some("real-30cm")),
            ("phone_video", Some("phone-video")),
            ("..\\..\\Windows", Some("windows")),
            ("../x/./y", Some("x-y")),
            ("C:\\a", Some("c-a")),
            ("--a--b--", Some("a-b")),
            ("ảnh 1", Some("nh-1")),
            ("", None),
            ("!!! ...", None),
        ] {
            assert_eq!(sanitize_label(raw).as_deref(), want, "{raw:?}");
        }
        let long = sanitize_label(&"ab-".repeat(30)).unwrap();
        assert!(long.len() <= 40 && !long.ends_with('-'), "{long}");
    }

    #[test]
    fn stamp_is_utc_civil_time() {
        let at = |s: u64| stamp(UNIX_EPOCH + std::time::Duration::from_secs(s));
        assert_eq!(at(0), "19700101-000000");
        // leap day
        assert_eq!(at(951_782_400), "20000229-000000");
        assert_eq!(at(1_700_000_000), "20231114-221320");
    }

    #[test]
    fn session_round_trips_through_disk() {
        let root = tempfile::tempdir().unwrap();
        let now = UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let mut rec = Recorder::create(root.path(), "Real 30cm", now).unwrap();
        assert_eq!(rec.dir(), root.path().join("real-30cm-20231114-221320"));
        // the same session name again is refused, not reused
        assert!(Recorder::create(root.path(), "real-30cm", now).is_err());
        assert!(Recorder::create(root.path(), "!!", now).is_err());

        let (f0, l0, a0) = (rgb(7, 4, 3), ir(11, 5, 2, true), ir(10, 5, 2, false));
        let (f1, l1) = (rgb(9, 4, 3), ir(13, 5, 2, true));
        assert!(rec.write(&f0, &l0, Some(&a0)).unwrap());
        assert!(rec.write(&f1, &l1, None).unwrap());
        assert_eq!(rec.steps(), 2);
        let dir = rec.dir().to_owned();
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        let want = [
            "frames.jsonl",
            "ir-amb-0.png",
            "ir-lit-0.png",
            "ir-lit-1.png",
            "rgb-0.png",
            "rgb-1.png",
        ];
        assert_eq!(names, want);

        // the schema, as Task 7 reads it
        let text = std::fs::read_to_string(dir.join(FRAMES_FILE)).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(
            lines[0],
            serde_json::json!({
                "n": 0, "rgb_seq": 7, "rgb_ts_ms": 1462, "rgb_w": 4, "rgb_h": 3,
                "ir_seq": 11, "ir_ts_ms": 50704, "ir_w": 5, "ir_h": 2, "ir_illuminated": true,
                "ambient": {"seq": 10, "ts_ms": 50640, "illuminated": false}
            })
        );
        assert_eq!(lines[1]["n"], 1);
        assert_eq!(lines[1]["ambient"], serde_json::Value::Null);

        let back: Vec<(Vec<u8>, Recorded)> = read(&dir).unwrap().map(Result::unwrap).collect();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, &f0.data[..]);
        assert_eq!((back[0].1.step.rgb_w, back[0].1.step.rgb_h), (4, 3));
        assert_eq!(back[0].1.lit, l0);
        assert_eq!(back[0].1.ambient.as_ref(), Some(&a0));
        assert_eq!(back[1].0, &f1.data[..]);
        assert_eq!(back[1].1.lit, l1);
        assert!(back[1].1.ambient.is_none());
        assert_eq!(back[1].1.step.rgb_ts_ms, f1.ts_ms);
    }

    #[test]
    fn a_session_stops_at_its_limit() {
        let root = tempfile::tempdir().unwrap();
        let mut rec = Recorder::create(root.path(), "cap", UNIX_EPOCH).unwrap();
        rec.limit = 2;
        let (f, l) = (rgb(0, 2, 2), ir(1, 2, 2, true));
        assert!(rec.write(&f, &l, None).unwrap());
        assert!(rec.write(&f, &l, None).unwrap());
        assert!(!rec.write(&f, &l, None).unwrap());
        assert_eq!(read(rec.dir()).unwrap().count(), 2);
        assert!(!rec.dir().join("rgb-2.png").exists());
    }

    #[test]
    fn the_root_is_under_an_absolute_local_app_data_only() {
        let root = |v: &str| root_under(Some(v.into()));
        assert_eq!(
            root(r"C:\Users\a\AppData\Local"),
            Some(PathBuf::from(r"C:\Users\a\AppData\Local\fasec\recordings"))
        );
        // empty or relative would put the recordings under the current directory
        for bad in ["", ".", "AppData", r"..\x"] {
            assert_eq!(root(bad), None, "{bad:?}");
        }
        assert_eq!(root_under(None), None);
    }

    #[test]
    fn a_truncated_last_line_is_skipped() {
        let root = tempfile::tempdir().unwrap();
        let mut rec = Recorder::create(root.path(), "cut", UNIX_EPOCH).unwrap();
        for _ in 0..2 {
            rec.write(&rgb(0, 2, 2), &ir(1, 2, 2, true), None).unwrap();
        }
        let dir = rec.dir().to_owned();
        drop(rec);
        let path = dir.join(FRAMES_FILE);
        let good = std::fs::read_to_string(&path).unwrap();
        // killed mid-write: the last line is cut short, its PNGs may be missing
        std::fs::write(&path, format!("{good}{{\"n\":2,\"rgb_se")).unwrap();
        assert_eq!(read(&dir).unwrap().count(), 2);
        // a bad line before the last is damage, not a cut write
        let (first, second) = good.split_once('\n').unwrap();
        std::fs::write(&path, format!("{first}\n{{\"n\":1,\"rgb_se\n{second}")).unwrap();
        assert!(read(&dir).is_err());
    }

    #[test]
    fn a_damaged_session_is_an_error() {
        let root = tempfile::tempdir().unwrap();
        let mut rec = Recorder::create(root.path(), "bad", UNIX_EPOCH).unwrap();
        rec.write(&rgb(0, 2, 2), &ir(1, 3, 2, true), None).unwrap();
        let dir = rec.dir().to_owned();
        drop(rec);
        // a PNG whose size disagrees with its line
        std::fs::copy(dir.join("ir-lit-0.png"), dir.join("rgb-0.png")).unwrap();
        assert!(read(&dir).unwrap().next().unwrap().is_err());
        std::fs::write(dir.join(FRAMES_FILE), "{not json}\n").unwrap();
        assert!(read(&dir).is_err());
        assert!(read(&root.path().join("missing")).is_err());
    }
}
