//! SP-1b spike: capture-based blur fallback. Captures one monitor with Windows Graphics
//! Capture, blurs it on the CPU (box downscale by `--downscale`, bilinear upscale) and shows it
//! in a top-most, capture-excluded layered overlay via `UpdateLayeredWindow`.
//! Throwaway code; results go in `docs/spikes/SP-1b-capture-blur.md`.
//!
//! `cargo run --release -p fasec-overlay --example sp1b_capture_blur -- [--monitor 0] [--seconds 15] [--downscale 8] [--no-exclude]`
//!
//! Prints every second: capture FPS, readback ms (GPU frame -> CPU, `Frame::buffer`), blur ms,
//! present ms, process CPU %, and how many captured frames contained the overlay's magenta
//! marker (must be 0: the overlay is excluded from capture, so there is no hall of mirrors). `--no-exclude` skips `WDA_EXCLUDEFROMCAPTURE` as a
//! positive control for that check. Closes after `--seconds` or on Esc.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::{COLORREF, FILETIME, HWND, LPARAM, LRESULT, POINT, SIZE, WPARAM};
use windows::Win32::Graphics::Gdi::{
    AC_SRC_OVER, BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BLENDFUNCTION, CreateCompatibleDC,
    CreateDIBSection, DIB_RGB_COLORS, GetMonitorInfoW, HMONITOR, MONITORINFO, SelectObject,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::Input::KeyboardAndMouse::VK_ESCAPE;
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, HHOOK,
    HWND_TOPMOST, KBDLLHOOKSTRUCT, MSG, PostMessageW, PostQuitMessage, RegisterClassW, SW_HIDE,
    SW_SHOWNOACTIVATE, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetTimer, SetWindowDisplayAffinity,
    SetWindowPos, SetWindowsHookExW, ShowWindow, ULW_OPAQUE, UnhookWindowsHookEx,
    UpdateLayeredWindow, WDA_EXCLUDEFROMCAPTURE, WH_KEYBOARD_LL, WM_APP, WM_KEYDOWN, WM_SYSKEYDOWN,
    WM_TIMER, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    WS_POPUP,
};
use windows::core::w;
use windows_capture::capture::{Context as CaptureContext, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::monitor::Monitor;
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};

/// Magenta square drawn on top of the blurred image; finding it in a captured frame means the
/// overlay captured itself.
const MARKER_X: usize = 64;
const MARKER_Y: usize = 64;
const MARKER_SIZE: usize = 48;
const MARKER_BGRA: [u8; 4] = [255, 0, 255, 255];
const WM_FRAME: u32 = WM_APP + 1;
/// Set by the Esc hook; checked on every frame because WM_QUIT/WM_TIMER can starve behind WM_FRAME.
static ESC: AtomicBool = AtomicBool::new(false);

struct Args {
    monitor: usize,
    seconds: u64,
    downscale: usize,
    exclude: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        monitor: 0,
        seconds: 15,
        downscale: 8,
        exclude: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut value = || it.next().with_context(|| format!("{a} needs a value"));
        match a.as_str() {
            "--monitor" => args.monitor = value()?.parse()?,
            "--seconds" => args.seconds = value()?.parse()?,
            "--downscale" => args.downscale = value()?.parse()?,
            "--no-exclude" => args.exclude = false,
            other => bail!(
                "unknown argument {other:?}; usage: [--monitor 0] [--seconds 15] [--downscale 8] [--no-exclude]"
            ),
        }
    }
    if args.downscale < 1 {
        bail!("--downscale must be >= 1");
    }
    Ok(args)
}

/// Latest blurred frame plus per-second counters, shared between capture and UI threads.
#[derive(Default)]
struct Shared {
    pixels: Vec<u8>,
    fresh: bool,
    frames: u32,
    readback: Duration,
    blur: Duration,
    marker_frames: u32,
    total_frames: u64,
    total_marker_frames: u64,
}

struct Job {
    shared: Arc<Mutex<Shared>>,
    hwnd: isize,
    width: usize,
    height: usize,
    downscale: usize,
}

struct Blur {
    job: Job,
    small: Vec<[u8; 4]>,
    out: Vec<u8>,
}

impl GraphicsCaptureApiHandler for Blur {
    type Flags = Job;
    type Error = anyhow::Error;

    fn new(ctx: CaptureContext<Job>) -> Result<Self> {
        Ok(Self {
            job: ctx.flags,
            small: Vec::new(),
            out: Vec::new(),
        })
    }

    fn on_frame_arrived(&mut self, frame: &mut Frame, _: InternalCaptureControl) -> Result<()> {
        let t = Instant::now();
        let mut buf = frame.buffer()?;
        let readback = t.elapsed();
        let (w, h, pitch) = (
            buf.width() as usize,
            buf.height() as usize,
            buf.row_pitch() as usize,
        );
        if (w, h) != (self.job.width, self.job.height) {
            bail!(
                "frame {w}x{h} != monitor {}x{} (resolution changed?)",
                self.job.width,
                self.job.height
            );
        }
        let raw = buf.as_raw_buffer();
        let marker = marker_seen(raw, pitch);
        let t = Instant::now();
        blur(
            raw,
            w,
            h,
            pitch,
            self.job.downscale,
            &mut self.small,
            &mut self.out,
        );
        draw_marker(&mut self.out, w);
        let blur_time = t.elapsed();
        {
            let mut s = self.job.shared.lock().expect("lock poisoned");
            std::mem::swap(&mut s.pixels, &mut self.out);
            s.fresh = true;
            s.frames += 1;
            s.readback += readback;
            s.blur += blur_time;
            s.total_frames += 1;
            if marker {
                s.marker_frames += 1;
                s.total_marker_frames += 1;
            }
        }
        // SAFETY: hwnd is our overlay; it outlives the capture (stopped before main returns).
        unsafe {
            PostMessageW(
                Some(HWND(self.job.hwnd as *mut c_void)),
                WM_FRAME,
                WPARAM(0),
                LPARAM(0),
            )
        }?;
        Ok(())
    }
}

/// CPU blur: average `d`x`d` blocks into `small`, then bilinear-upscale back to `w`x`h` BGRA.
fn blur(
    src: &[u8],
    w: usize,
    h: usize,
    pitch: usize,
    d: usize,
    small: &mut Vec<[u8; 4]>,
    out: &mut Vec<u8>,
) {
    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
    small.resize(sw * sh, [0; 4]);
    let mut acc = vec![[0u32; 4]; sw];
    for sy in 0..sh {
        acc.fill([0; 4]);
        let rows = (sy * d)..((sy + 1) * d).min(h);
        let nrows = rows.len();
        for y in rows {
            let row = &src[y * pitch..y * pitch + w * 4];
            for (a, block) in acc.iter_mut().zip(row.chunks(d * 4)) {
                for px in block.as_chunks::<4>().0 {
                    a[0] += u32::from(px[0]);
                    a[1] += u32::from(px[1]);
                    a[2] += u32::from(px[2]);
                }
            }
        }
        for (sx, a) in acc.iter().enumerate() {
            let n = (nrows * (((sx + 1) * d).min(w) - sx * d)) as u32;
            small[sy * sw + sx] = [(a[0] / n) as u8, (a[1] / n) as u8, (a[2] / n) as u8, 255];
        }
    }

    // Per-axis (lo index, hi index, weight of hi in 1/256) for sampling at pixel centres.
    let taps = |n: usize, sn: usize| -> Vec<(usize, usize, u32)> {
        (0..n)
            .map(|i| {
                let f = ((i as f32 + 0.5) / d as f32 - 0.5).clamp(0.0, (sn - 1) as f32);
                let lo = f as usize;
                (lo, (lo + 1).min(sn - 1), ((f - lo as f32) * 256.0) as u32)
            })
            .collect()
    };
    let (xt, yt) = (taps(w, sw), taps(h, sh));
    out.resize(w * h * 4, 0);
    for (y, &(y0, y1, wy)) in yt.iter().enumerate() {
        let (r0, r1) = (&small[y0 * sw..][..sw], &small[y1 * sw..][..sw]);
        let dst = &mut out[y * w * 4..][..w * 4];
        for (px, &(x0, x1, wx)) in dst.as_chunks_mut::<4>().0.iter_mut().zip(&xt) {
            for c in 0..3 {
                let top = u32::from(r0[x0][c]) * (256 - wx) + u32::from(r0[x1][c]) * wx;
                let bot = u32::from(r1[x0][c]) * (256 - wx) + u32::from(r1[x1][c]) * wx;
                px[c] = ((top * (256 - wy) + bot * wy) >> 16) as u8;
            }
            px[3] = 255;
        }
    }
}

fn draw_marker(img: &mut [u8], w: usize) {
    for y in MARKER_Y..MARKER_Y + MARKER_SIZE {
        let row = &mut img[(y * w + MARKER_X) * 4..][..MARKER_SIZE * 4];
        for px in row.as_chunks_mut::<4>().0 {
            px.copy_from_slice(&MARKER_BGRA);
        }
    }
}

/// True if >= 90% of the marker square in the captured frame is (near) magenta.
fn marker_seen(src: &[u8], pitch: usize) -> bool {
    let mut hits = 0;
    for y in MARKER_Y..MARKER_Y + MARKER_SIZE {
        let row = &src[y * pitch + MARKER_X * 4..][..MARKER_SIZE * 4];
        hits += row
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] >= 240 && p[1] <= 15 && p[2] >= 240)
            .count();
    }
    hits * 10 >= MARKER_SIZE * MARKER_SIZE * 9
}

fn process_cpu_time() -> Duration {
    let (mut c, mut e, mut k, mut u) = Default::default();
    // SAFETY: valid pseudo-handle and out pointers.
    if unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }.is_err() {
        return Duration::ZERO;
    }
    let ticks = |t: FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    Duration::from_nanos((ticks(k) + ticks(u)) * 100)
}

unsafe extern "system" fn wnd_proc(h: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    // SAFETY: forwarding the arguments we were given.
    unsafe { DefWindowProcW(h, msg, wp, lp) }
}

unsafe extern "system" fn kb_hook(code: i32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if code >= 0 && (wp.0 as u32 == WM_KEYDOWN || wp.0 as u32 == WM_SYSKEYDOWN) {
        // SAFETY: for WH_KEYBOARD_LL with code >= 0, lParam points to a KBDLLHOOKSTRUCT.
        let kb = unsafe { &*(lp.0 as *const KBDLLHOOKSTRUCT) };
        if kb.vkCode == u32::from(VK_ESCAPE.0) {
            println!("Esc pressed, closing");
            ESC.store(true, Ordering::Relaxed);
            // SAFETY: LL hooks run on the installing thread, so this ends our message loop.
            unsafe { PostQuitMessage(0) };
        }
    }
    // SAFETY: always pass the event on; we never swallow keys.
    unsafe { CallNextHookEx(None, code, wp, lp) }
}

/// Removes the keyboard hook on every exit path (including panics).
struct HookGuard(HHOOK);
impl Drop for HookGuard {
    fn drop(&mut self) {
        // SAFETY: self.0 is the hook returned by SetWindowsHookExW.
        match unsafe { UnhookWindowsHookEx(self.0) } {
            Ok(()) => println!("keyboard hook removed"),
            Err(e) => println!("UnhookWindowsHookEx failed: {e}"),
        }
    }
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // SAFETY: plain Win32 call, before any window exists. PMv2 so the monitor rect is in
    // physical pixels, matching the capture size.
    if let Err(e) =
        unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) }
    {
        println!("SetProcessDpiAwarenessContext(PMv2) failed: {e}");
    }

    let monitors = Monitor::enumerate().context("Monitor::enumerate")?;
    let monitor = *monitors.get(args.monitor).with_context(|| {
        format!(
            "--monitor {} out of range ({} monitors)",
            args.monitor,
            monitors.len()
        )
    })?;
    let mut mi = MONITORINFO {
        cbSize: size_of::<MONITORINFO>() as u32,
        ..Default::default()
    };
    // SAFETY: mi is size-tagged; the handle came from EnumDisplayMonitors (via windows-capture).
    if !unsafe { GetMonitorInfoW(HMONITOR(monitor.as_raw_hmonitor()), &mut mi) }.as_bool() {
        bail!("GetMonitorInfoW failed");
    }
    let r = mi.rcMonitor;
    let (width, height) = ((r.right - r.left) as usize, (r.bottom - r.top) as usize);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!(
        "monitor {} ({}): rect=({},{})-({},{}) {width}x{height}, refresh {} Hz, {cores} logical cores",
        args.monitor,
        monitor.device_name().unwrap_or_default(),
        r.left,
        r.top,
        r.right,
        r.bottom,
        monitor.refresh_rate().unwrap_or(0),
    );
    println!(
        "downscale={} seconds={} exclude_from_capture={}",
        args.downscale, args.seconds, args.exclude
    );
    if width < MARKER_X + MARKER_SIZE || height < MARKER_Y + MARKER_SIZE {
        bail!("monitor too small for the marker");
    }

    // SAFETY: standard class registration with our module handle.
    let hinstance = unsafe { GetModuleHandleW(None)? };
    let class = w!("fasec_sp1b_overlay");
    let wc = WNDCLASSW {
        lpfnWndProc: Some(wnd_proc),
        hInstance: hinstance.into(),
        lpszClassName: class,
        ..Default::default()
    };
    if unsafe { RegisterClassW(&wc) } == 0 {
        bail!(
            "RegisterClassW failed: {}",
            windows::core::Error::from_thread()
        );
    }
    // Layered for UpdateLayeredWindow; invisible until the first UpdateLayeredWindow call, so the
    // screen is never covered by an empty window.
    // SAFETY: registered class, valid instance, no parent/menu/param.
    let hwnd = unsafe {
        CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            class,
            w!("fasec SP-1b overlay"),
            WS_POPUP,
            r.left,
            r.top,
            width as i32,
            height as i32,
            None,
            None,
            Some(hinstance.into()),
            None,
        )
    }
    .context("CreateWindowExW")?;
    if args.exclude {
        // SAFETY: valid top-level hwnd owned by this process.
        unsafe { SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) }
            .context("WDA_EXCLUDEFROMCAPTURE failed; refusing to run (hall of mirrors)")?;
        println!("WDA_EXCLUDEFROMCAPTURE: ok");
    } else {
        println!("WDA_EXCLUDEFROMCAPTURE: skipped (--no-exclude positive control)");
    }

    // Top-down 32-bit DIB selected into a memory DC: the source for UpdateLayeredWindow.
    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: width as i32,
            biHeight: -(height as i32),
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut bits: *mut c_void = std::ptr::null_mut();
    // SAFETY: bmi is a valid header; bits receives the DIB's pixel pointer. GDI objects are
    // reclaimed at process exit.
    let mem_dc = unsafe { CreateCompatibleDC(None) };
    let dib = unsafe { CreateDIBSection(Some(mem_dc), &bmi, DIB_RGB_COLORS, &mut bits, None, 0) }
        .context("CreateDIBSection")?;
    unsafe { SelectObject(mem_dc, dib.into()) };
    // SAFETY: CreateDIBSection allocated width*height*4 bytes at `bits`, alive until exit.
    let dib_pixels = unsafe { std::slice::from_raw_parts_mut(bits as *mut u8, width * height * 4) };

    let shared = Arc::new(Mutex::new(Shared::default()));
    let border = if GraphicsCaptureApi::is_border_settings_supported().unwrap_or(false) {
        DrawBorderSettings::WithoutBorder
    } else {
        DrawBorderSettings::Default
    };
    let settings = Settings::new(
        monitor,
        CursorCaptureSettings::WithoutCursor,
        border,
        SecondaryWindowSettings::Default,
        MinimumUpdateIntervalSettings::Default,
        DirtyRegionSettings::Default,
        ColorFormat::Bgra8,
        Job {
            shared: shared.clone(),
            hwnd: hwnd.0 as isize,
            width,
            height,
            downscale: args.downscale,
        },
    );
    let capture =
        Blur::start_free_threaded(settings).map_err(|e| anyhow::anyhow!("start capture: {e}"))?;

    // SAFETY: LL hook with a proc in this module; removed by HookGuard.
    let _hook = HookGuard(
        unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(kb_hook), Some(hinstance.into()), 0) }
            .context("SetWindowsHookExW(WH_KEYBOARD_LL)")?,
    );
    // SAFETY: thread timer (no hwnd); WM_TIMER every 250 ms for top-most + stats + deadline.
    if unsafe { SetTimer(None, 0, 250, None) } == 0 {
        bail!("SetTimer failed: {}", windows::core::Error::from_thread());
    }

    let start = Instant::now();
    let deadline = start + Duration::from_secs(args.seconds);
    let (mut last, mut last_cpu) = (start, process_cpu_time());
    let (mut presents, mut present_time) = (0u32, Duration::ZERO);
    let (mut shown, mut cpu_sum, mut cpu_samples) = (false, 0.0, 0u32);
    let mut msg = MSG::default();
    // SAFETY: standard message loop on the thread that owns the window and the hook.
    while unsafe { GetMessageW(&mut msg, None, 0, 0) }.as_bool() {
        match msg.message {
            WM_FRAME => {
                // WM_TIMER is lowest priority: a frame flood must not remove the exits.
                if ESC.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    println!("Esc or {} s deadline reached, closing", args.seconds);
                    break;
                }
                let t = Instant::now();
                {
                    let mut s = shared.lock().expect("lock poisoned");
                    if !s.fresh {
                        continue;
                    }
                    dib_pixels.copy_from_slice(&s.pixels);
                    s.fresh = false;
                }
                let (pos, size, src) = (
                    POINT {
                        x: r.left,
                        y: r.top,
                    },
                    SIZE {
                        cx: width as i32,
                        cy: height as i32,
                    },
                    POINT::default(),
                );
                let blend = BLENDFUNCTION {
                    BlendOp: AC_SRC_OVER as u8,
                    SourceConstantAlpha: 255,
                    ..Default::default()
                };
                // SAFETY: mem_dc holds the DIB; all pointers reference locals alive for the call.
                if let Err(e) = unsafe {
                    UpdateLayeredWindow(
                        hwnd,
                        None,
                        Some(&pos),
                        Some(&size),
                        Some(mem_dc),
                        Some(&src),
                        COLORREF(0),
                        Some(&blend),
                        ULW_OPAQUE,
                    )
                } {
                    println!("UpdateLayeredWindow failed: {e}");
                    break;
                }
                if !shown {
                    // SAFETY: valid hwnd; SW_SHOWNOACTIVATE keeps focus where it is.
                    let _ = unsafe { ShowWindow(hwnd, SW_SHOWNOACTIVATE) };
                    shown = true;
                }
                presents += 1;
                present_time += t.elapsed();
            }
            WM_TIMER if msg.hwnd.is_invalid() => {
                let flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE;
                // SAFETY: valid hwnd.
                let _ = unsafe { SetWindowPos(hwnd, Some(HWND_TOPMOST), 0, 0, 0, 0, flags) };
                let now = Instant::now();
                let dt = now - last;
                if dt >= Duration::from_secs(1) {
                    let cpu = process_cpu_time();
                    let cpu_pct = cpu.saturating_sub(last_cpu).as_secs_f64()
                        / dt.as_secs_f64()
                        / cores as f64
                        * 100.0;
                    let mut s = shared.lock().expect("lock poisoned");
                    let ms = |d: Duration, n: u32| d.as_secs_f64() * 1000.0 / f64::from(n.max(1));
                    println!(
                        "capture {:5.1} fps | readback {:5.2} ms | blur {:6.2} ms | present {:5.2} ms ({} presents) | cpu {:5.1} % | marker in capture {}/{}",
                        f64::from(s.frames) / dt.as_secs_f64(),
                        ms(s.readback, s.frames),
                        ms(s.blur, s.frames),
                        ms(present_time, presents),
                        presents,
                        cpu_pct,
                        s.marker_frames,
                        s.frames
                    );
                    if s.frames > 0 {
                        cpu_sum += cpu_pct;
                        cpu_samples += 1;
                    }
                    (s.frames, s.readback, s.blur, s.marker_frames) =
                        (0, Duration::ZERO, Duration::ZERO, 0);
                    (presents, present_time, last, last_cpu) = (0, Duration::ZERO, now, cpu);
                }
                if capture.is_finished() {
                    println!("capture thread ended early, closing");
                    break;
                }
                if now >= deadline {
                    println!("{} s elapsed, closing", args.seconds);
                    break;
                }
            }
            _ => {
                // SAFETY: msg came from GetMessageW.
                unsafe { DispatchMessageW(&msg) };
            }
        }
    }

    // Uncover the screen and release the hook before anything that might block.
    // SAFETY: valid hwnd.
    let _ = unsafe { ShowWindow(hwnd, SW_HIDE) };
    drop(_hook);
    let result = capture.stop();
    let s = shared.lock().expect("lock poisoned");
    println!(
        "total: {} frames in {:.1} s, overlay marker found in {} captured frames ({}), mean cpu {:.1} %",
        s.total_frames,
        start.elapsed().as_secs_f64(),
        s.total_marker_frames,
        if s.total_marker_frames == 0 {
            "excluded from capture, no hall of mirrors"
        } else {
            "OVERLAY CAPTURED ITSELF"
        },
        cpu_sum / f64::from(cpu_samples.max(1)),
    );
    result.map_err(|e| anyhow::anyhow!("capture: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blur_keeps_flat_colour_and_marker_round_trips() {
        let (w, h, pitch) = (130, 121, 130 * 4 + 8); // padded rows, sizes not a multiple of d
        let src: Vec<u8> = (0..pitch * h)
            .map(|i| {
                if i % pitch < w * 4 {
                    [10, 20, 30, 255][i % 4]
                } else {
                    0
                }
            })
            .collect();
        let (mut small, mut out) = (Vec::new(), Vec::new());
        for d in [1, 4, 8, 16] {
            blur(&src, w, h, pitch, d, &mut small, &mut out);
            assert_eq!(out.len(), w * h * 4);
            assert!(out.chunks_exact(4).all(|p| p == [10, 20, 30, 255]), "d={d}");
        }
        assert!(!marker_seen(&out, w * 4));
        draw_marker(&mut out, w);
        assert!(marker_seen(&out, w * 4));
    }
}
