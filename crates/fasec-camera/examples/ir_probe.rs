//! SP-7 probe: can we read Windows Hello IR frames?
//!
//! `cargo run -p fasec-camera --example ir_probe -- [list | read <group-index>]`
//!
//! `list` (default) prints every WinRT `MediaFrameSourceGroup` with each source's kind and device
//! id. `read` opens the group's Infrared source with `MediaCapture` (`SharedReadOnly` first, then
//! `ExclusiveControl`), reads 30 frames and prints their format and mean intensity. Frames stay
//! in memory: nothing is saved to disk.

#[cfg(not(windows))]
fn main() {
    eprintln!("ir_probe needs Windows (WinRT Windows.Media.Capture.Frames)");
}

#[cfg(windows)]
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // WinRT async ops are joined (blocking) here, so run off the main thread like a real caller would.
    std::thread::spawn(move || win::run(&args))
        .join()
        .map_err(|_| anyhow::anyhow!("probe thread panicked"))?
}

#[cfg(windows)]
mod win {
    use std::time::{Duration, Instant};

    use anyhow::{Context, Result, bail};
    use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
    use windows::Media::Capture::Frames::{
        MediaFrameReader, MediaFrameReaderStartStatus, MediaFrameSourceGroup, MediaFrameSourceKind,
    };
    use windows::Media::Capture::{
        MediaCapture, MediaCaptureInitializationSettings, MediaCaptureMemoryPreference,
        MediaCaptureSharingMode, MediaStreamType, StreamingCaptureMode,
    };
    use windows::Storage::Streams::{Buffer, DataReader};
    use windows::core::HSTRING;

    const USAGE: &str = "usage: ir_probe [list | read <group-index>]";
    const FRAMES: u32 = 30;
    const READ_TIMEOUT: Duration = Duration::from_secs(10);

    pub fn run(args: &[String]) -> Result<()> {
        let view = MediaFrameSourceGroup::FindAllAsync()?.join()?;
        let groups = (0..view.Size()?)
            .map(|i| view.GetAt(i))
            .collect::<windows::core::Result<Vec<_>>>()?;
        match args.first().map(String::as_str) {
            None | Some("list") => list(&groups),
            Some("read") => {
                let i: usize = args.get(1).context(USAGE)?.parse().context(USAGE)?;
                read(groups.get(i).with_context(|| format!("no group {i}"))?)
            }
            _ => bail!(USAGE),
        }
    }

    fn list(groups: &[MediaFrameSourceGroup]) -> Result<()> {
        println!("{} source group(s)", groups.len());
        for (i, g) in groups.iter().enumerate() {
            println!("[{i}] {}\n    group id: {}", g.DisplayName()?, g.Id()?);
            let infos = g.SourceInfos()?;
            for j in 0..infos.Size()? {
                let s = infos.GetAt(j)?;
                let device = s
                    .DeviceInformation()
                    .and_then(|d| d.Id())
                    .map_or_else(|e| format!("<{e}>"), |id| id.to_string());
                println!(
                    "    - {} / {}\n      source id: {}\n      device id: {device}",
                    kind_name(s.SourceKind()?),
                    stream_name(s.MediaStreamType()?),
                    s.Id()?
                );
            }
        }
        Ok(())
    }

    fn read(group: &MediaFrameSourceGroup) -> Result<()> {
        let infos = group.SourceInfos()?;
        let mut ir = None;
        for j in 0..infos.Size()? {
            let s = infos.GetAt(j)?;
            if s.SourceKind()? == MediaFrameSourceKind::Infrared {
                ir = Some(s.Id()?);
                break;
            }
        }
        let ir = ir.with_context(|| {
            format!(
                "group {} has no Infrared source",
                group.DisplayName().unwrap_or_default()
            )
        })?;
        println!("group: {}\nInfrared source: {ir}", group.DisplayName()?);
        for (mode, name) in [
            (MediaCaptureSharingMode::SharedReadOnly, "SharedReadOnly"),
            (
                MediaCaptureSharingMode::ExclusiveControl,
                "ExclusiveControl",
            ),
        ] {
            println!("-- sharing mode {name}");
            match read_with(group, &ir, mode) {
                Ok(()) => {
                    println!("RESULT: {FRAMES} Infrared frames read with {name}");
                    return Ok(());
                }
                Err(e) => println!("{name} failed: {e:#}"),
            }
        }
        bail!("could not read Infrared frames with either sharing mode")
    }

    fn read_with(
        group: &MediaFrameSourceGroup,
        ir: &HSTRING,
        mode: MediaCaptureSharingMode,
    ) -> Result<()> {
        let settings = MediaCaptureInitializationSettings::new()?;
        settings.SetSourceGroup(group)?;
        settings.SetSharingMode(mode)?;
        settings.SetMemoryPreference(MediaCaptureMemoryPreference::Cpu)?;
        settings.SetStreamingCaptureMode(StreamingCaptureMode::Video)?;
        let capture = MediaCapture::new()?;
        let result = (|| {
            capture
                .InitializeWithSettingsAsync(&settings)?
                .join()
                .context("MediaCapture initialize")?;
            let source = capture.FrameSources()?.Lookup(ir).context("find source")?;
            let f = source.CurrentFormat()?;
            let rate = f.FrameRate()?;
            println!(
                "current format: {} {}x{} @ {}/{}",
                f.Subtype()?,
                f.VideoFormat()?.Width()?,
                f.VideoFormat()?.Height()?,
                rate.Numerator()?,
                rate.Denominator()?
            );
            let reader = capture.CreateFrameReaderAsync(&source)?.join()?;
            let status = reader.StartAsync()?.join()?;
            if status != MediaFrameReaderStartStatus::Success {
                bail!("frame reader start status {}", status.0);
            }
            let frames = read_frames(&reader);
            reader.StopAsync()?.join()?;
            frames
        })();
        let _ = capture.Close(); // releases the camera; nothing useful to do if it fails
        result
    }

    fn read_frames(reader: &MediaFrameReader) -> Result<()> {
        let deadline = Instant::now() + READ_TIMEOUT;
        let (mut n, mut lit) = (0, 0);
        let t0 = Instant::now();
        while n < FRAMES {
            if Instant::now() > deadline {
                bail!("only {n} frames in {} s", READ_TIMEOUT.as_secs());
            }
            // Fails (null) when no new frame has arrived since the last call.
            let Ok(frame) = reader.TryAcquireLatestFrame() else {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            };
            let video = frame.VideoMediaFrame()?;
            let bmp = video
                .SoftwareBitmap()
                .context("no CPU SoftwareBitmap in frame")?;
            let illuminated = video.InfraredMediaFrame().and_then(|i| i.IsIlluminated());
            if matches!(illuminated, Ok(true)) {
                lit += 1;
            }
            n += 1;
            println!(
                "frame {n:2} +{:4} ms: {}x{} {} mean {:.1} illuminated {}",
                t0.elapsed().as_millis(),
                bmp.PixelWidth()?,
                bmp.PixelHeight()?,
                pixel_name(bmp.BitmapPixelFormat()?),
                mean_intensity(&bmp)?,
                illuminated.map_or_else(|e| format!("<{e}>"), |b| b.to_string()),
            );
            let _ = bmp.Close();
            let _ = frame.Close();
        }
        println!(
            "{n} frames in {} ms, {lit} illuminated",
            t0.elapsed().as_millis()
        );
        Ok(())
    }

    /// Mean of the raw values (0..255 for Gray8, 0..65535 for Gray16); other formats are
    /// converted to Gray8 first. Includes row padding if the bitmap has any.
    fn mean_intensity(bmp: &SoftwareBitmap) -> Result<f64> {
        let format = bmp.BitmapPixelFormat()?;
        let bmp = if format == BitmapPixelFormat::Gray8 || format == BitmapPixelFormat::Gray16 {
            bmp.clone()
        } else {
            SoftwareBitmap::Convert(bmp, BitmapPixelFormat::Gray8)?
        };
        let (w, h) = (bmp.PixelWidth()?, bmp.PixelHeight()?);
        let buf = Buffer::Create(u32::try_from(w * h * 4).context("bitmap size")?)?;
        bmp.CopyToBuffer(&buf)?;
        let mut bytes = vec![0u8; buf.Length()? as usize];
        DataReader::FromBuffer(&buf)?.ReadBytes(&mut bytes)?;
        let (sum, count) = if bmp.BitmapPixelFormat()? == BitmapPixelFormat::Gray16 {
            let px = bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]));
            (px.map(f64::from).sum::<f64>(), bytes.len() / 2)
        } else {
            (bytes.iter().map(|&b| f64::from(b)).sum(), bytes.len())
        };
        Ok(if count == 0 { 0.0 } else { sum / count as f64 })
    }

    fn kind_name(k: MediaFrameSourceKind) -> String {
        match k {
            MediaFrameSourceKind::Color => "Color".into(),
            MediaFrameSourceKind::Infrared => "Infrared".into(),
            MediaFrameSourceKind::Depth => "Depth".into(),
            MediaFrameSourceKind::Custom => "Custom".into(),
            MediaFrameSourceKind::Image => "Image".into(),
            MediaFrameSourceKind::Metadata => "Metadata".into(),
            other => format!("kind {}", other.0),
        }
    }

    fn stream_name(t: MediaStreamType) -> String {
        match t {
            MediaStreamType::VideoPreview => "VideoPreview".into(),
            MediaStreamType::VideoRecord => "VideoRecord".into(),
            MediaStreamType::Photo => "Photo".into(),
            MediaStreamType::Metadata => "Metadata".into(),
            other => format!("stream {}", other.0),
        }
    }

    fn pixel_name(p: BitmapPixelFormat) -> String {
        match p {
            BitmapPixelFormat::Gray8 => "Gray8".into(),
            BitmapPixelFormat::Gray16 => "Gray16".into(),
            BitmapPixelFormat::Nv12 => "NV12".into(),
            BitmapPixelFormat::Yuy2 => "YUY2".into(),
            BitmapPixelFormat::Bgra8 => "BGRA8".into(),
            other => format!("pixel format {}", other.0),
        }
    }
}
