//! Frame sources: a directory of still images, or a video decoded by the `ffmpeg` CLI.

use std::ffi::{OsStr, OsString};
use std::io::{ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, anyhow, ensure};
use fasec_replay::recording::{self, Recorded};

/// One decoded RGB8 frame, row-major, `data.len() == width * height * 3`.
pub struct RawFrame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    /// From an `ir_debug` recording: its index among the recordings replayed, and the step's
    /// RGB time and lit IR pair.
    pub recorded: Option<(usize, Recorded)>,
}

/// RGB time from one recording's last step to the next one's first: about one recorded step.
const JOIN_GAP_MS: u64 = 500;
/// IR seqs from one recording's last lit frame to the next one's: one unlit frame between.
const JOIN_GAP_SEQ: u64 = 2;

/// The steps of the `ir_debug` recordings in `dirs`, one after another, each RGB frame with its
/// lit IR pair. Each recording is a separate capture whose RGB time and IR seqs start over, so
/// every one after the first is shifted to continue right after the one before, as one capture
/// would: the pipeline then keeps its IR trust and never sees time or seqs go back.
pub fn open_recordings(dirs: &[PathBuf]) -> Result<Box<dyn Iterator<Item = Result<RawFrame>>>> {
    let sessions = dirs
        .iter()
        .map(|d| recording::read(d))
        .collect::<Result<Vec<_>>>()?;
    let steps = (0..)
        .zip(sessions)
        .flat_map(|(k, s)| s.map(move |r| (k, r)));
    // (RGB time, IR seq): the current recording's first, where it now starts, the last given
    let (mut current, mut first, mut start, mut last) = (None, (0, 0), (0, 0), (0, 0));
    Ok(Box::new(steps.map(move |(k, r)| {
        let (data, mut r) = r?;
        if current != Some(k) {
            first = (r.step.rgb_ts_ms, r.step.ir_seq);
            start = match current {
                None => first,
                Some(_) => (last.0 + JOIN_GAP_MS, last.1 + JOIN_GAP_SEQ),
            };
            current = Some(k);
        }
        let seq = |s: u64| (s + start.1).saturating_sub(first.1);
        r.step.rgb_ts_ms = (r.step.rgb_ts_ms + start.0).saturating_sub(first.0);
        r.step.ir_seq = seq(r.step.ir_seq);
        r.lit.seq = seq(r.lit.seq);
        if let Some(a) = &mut r.ambient {
            a.seq = seq(a.seq);
        }
        if let Some(a) = &mut r.step.ambient {
            a.seq = seq(a.seq);
        }
        last = (r.step.rgb_ts_ms, r.step.ir_seq);
        Ok(RawFrame {
            width: r.step.rgb_w,
            height: r.step.rgb_h,
            data,
            recorded: Some((k, r)),
        })
    })))
}

/// Programs used to decode video; tests point these at missing binaries.
pub struct Ffmpeg {
    pub ffmpeg: OsString,
    pub ffprobe: OsString,
}

impl Default for Ffmpeg {
    fn default() -> Self {
        Ffmpeg {
            ffmpeg: "ffmpeg".into(),
            ffprobe: "ffprobe".into(),
        }
    }
}

/// Frames from `input`: a directory of `*.png`/`*.jpg` sorted by file name, else a video.
pub fn open(input: &Path, ff: &Ffmpeg) -> Result<Box<dyn Iterator<Item = Result<RawFrame>>>> {
    if input.is_dir() {
        let files = image_files(input)?;
        ensure!(
            !files.is_empty(),
            "no *.png/*.jpg frames in {}",
            input.display()
        );
        return Ok(Box::new(files.into_iter().map(|p| {
            let img = image::open(&p)
                .with_context(|| format!("decoding {}", p.display()))?
                .into_rgb8();
            Ok(RawFrame {
                width: img.width(),
                height: img.height(),
                data: img.into_raw(),
                recorded: None,
            })
        })));
    }
    ensure!(input.is_file(), "input not found: {}", input.display());
    Ok(Box::new(VideoFrames::spawn(input, ff)?))
}

/// `*.png`/`*.jpg`/`*.jpeg` (any case) in `dir`, sorted by file name.
fn image_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        if path.is_file() && matches!(ext.as_deref(), Some("png" | "jpg" | "jpeg")) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn spawn_error(program: &OsStr, e: std::io::Error) -> anyhow::Error {
    let name = program.to_string_lossy();
    if e.kind() == ErrorKind::NotFound {
        anyhow!(
            "`{name}` not found: video input needs ffmpeg and ffprobe on PATH \
             (or pass a directory of *.png/*.jpg frames)"
        )
    } else {
        anyhow!(e).context(format!("running `{name}`"))
    }
}

/// Output frame size of the first video stream, after ffmpeg's default auto-rotation.
fn probe(input: &Path, ff: &Ffmpeg) -> Result<(u32, u32)> {
    let out = Command::new(&ff.ffprobe)
        .args(["-v", "error", "-select_streams", "v:0"])
        .args([
            "-show_entries",
            "stream=width,height:stream_side_data=rotation",
        ])
        .args(["-of", "json"])
        .arg(input)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| spawn_error(&ff.ffprobe, e))?;
    ensure!(
        out.status.success(),
        "ffprobe failed on {} ({}): {}",
        input.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr).trim()
    );
    let v: serde_json::Value =
        serde_json::from_slice(&out.stdout).context("parsing ffprobe output")?;
    let s = &v["streams"][0];
    let dim = |k: &str| {
        s[k].as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|&n| n > 0)
            .with_context(|| format!("no video stream {k} in {}", input.display()))
    };
    let (w, h) = (dim("width")?, dim("height")?);
    let rotation = s["side_data_list"]
        .as_array()
        .into_iter()
        .flatten()
        .find_map(|d| d["rotation"].as_f64())
        .unwrap_or(0.0);
    // ffmpeg auto-rotates, so a 90/270 degree display matrix swaps the decoded size.
    let quarter_turns = (rotation / 90.0).round() as i64;
    Ok(if quarter_turns % 2 != 0 {
        (h, w)
    } else {
        (w, h)
    })
}

/// Raw RGB24 frames read from an `ffmpeg` child's stdout.
struct VideoFrames {
    child: Child,
    stdout: ChildStdout,
    frame_len: usize,
    width: u32,
    height: u32,
    done: bool,
}

impl VideoFrames {
    fn spawn(input: &Path, ff: &Ffmpeg) -> Result<VideoFrames> {
        let (width, height) = probe(input, ff)?;
        let frame_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(3))
            .context("video frame size overflows")?;
        let mut child = Command::new(&ff.ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(input)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .spawn()
            .map_err(|e| spawn_error(&ff.ffmpeg, e))?;
        let stdout = child.stdout.take().context("ffmpeg stdout not captured")?;
        Ok(VideoFrames {
            child,
            stdout,
            frame_len,
            width,
            height,
            done: false,
        })
    }

    fn read_frame(&mut self) -> Result<Option<RawFrame>> {
        let mut data = vec![0; self.frame_len];
        let mut filled = 0;
        while filled < data.len() {
            match self.stdout.read(&mut data[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return Err(anyhow!(e).context("reading ffmpeg output")),
            }
        }
        if filled == data.len() {
            return Ok(Some(RawFrame {
                width: self.width,
                height: self.height,
                data,
                recorded: None,
            }));
        }
        let status = self.child.wait().context("waiting for ffmpeg")?;
        ensure!(status.success(), "ffmpeg failed ({status})");
        ensure!(
            filled == 0,
            "ffmpeg output ended mid-frame ({filled} of {} bytes)",
            data.len()
        );
        Ok(None)
    }
}

impl Iterator for VideoFrames {
    type Item = Result<RawFrame>;

    fn next(&mut self) -> Option<Result<RawFrame>> {
        if self.done {
            return None;
        }
        let r = self.read_frame().transpose();
        self.done = !matches!(r, Some(Ok(_)));
        r
    }
}

impl Drop for VideoFrames {
    /// Stopped early (e.g. a pipeline error): don't leave ffmpeg running.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_ffmpeg_is_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        let video = dir.path().join("clip.mp4");
        std::fs::write(&video, b"not really a video").unwrap();
        let ff = Ffmpeg {
            ffmpeg: "fasec-test-no-such-ffmpeg".into(),
            ffprobe: "fasec-test-no-such-ffprobe".into(),
        };
        let err = match open(&video, &ff) {
            Ok(mut frames) => frames.next().unwrap().err().unwrap(),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("not found"), "{msg}");
        assert!(msg.contains("ffmpeg"), "{msg}");
    }

    #[test]
    fn missing_input_is_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(open(&dir.path().join("nope.mp4"), &Ffmpeg::default()).is_err());
    }

    #[test]
    #[ignore = "needs ffmpeg and ffprobe on PATH"]
    fn video_decodes_every_frame_with_rotation() {
        let dir = tempfile::tempdir().unwrap();
        let (plain, rotated) = (dir.path().join("a.mp4"), dir.path().join("r.mp4"));
        let ffmpeg = |args: &[&str]| {
            let mut cmd = Command::new("ffmpeg");
            assert!(
                cmd.args(["-v", "error", "-y"])
                    .args(args)
                    .status()
                    .unwrap()
                    .success()
            );
        };
        let (p, r) = (plain.to_str().unwrap(), rotated.to_str().unwrap());
        ffmpeg(&[
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=5",
            "-frames:v",
            "4",
            "-c:v",
            "mpeg4",
            p,
        ]);
        ffmpeg(&["-display_rotation", "90", "-i", p, "-c", "copy", r]);
        for (path, dims) in [(&plain, (64, 48)), (&rotated, (48, 64))] {
            let frames: Vec<_> = open(path, &Ffmpeg::default())
                .unwrap()
                .collect::<Result<_>>()
                .unwrap();
            assert_eq!(frames.len(), 4);
            for f in frames {
                assert_eq!((f.width, f.height), dims);
                assert_eq!(f.data.len(), 64 * 48 * 3);
            }
        }
    }
}
