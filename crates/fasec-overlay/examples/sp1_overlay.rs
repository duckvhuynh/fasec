//! SP-1 / SP-2 spike: one top-most, non-activating, capture-excluded overlay per monitor
//! with a selectable blur backend. Throwaway code; results go in `docs/spikes/`.
//!
//! `cargo run -p fasec-overlay --example sp1_overlay -- --mode <acrylic|backdrop|solid> [--seconds 15] [--click-through]`
//!
//! Closes after `--seconds` or on Esc (low-level keyboard hook, Esc is not swallowed).

use std::ffi::c_void;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use windows::Wdk::System::SystemServices::RtlGetVersion;
use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Dwm::{
    DWM_SYSTEMBACKDROP_TYPE, DWMSBT_TRANSIENTWINDOW, DWMWA_SYSTEMBACKDROP_TYPE,
    DwmExtendFrameIntoClientArea, DwmSetWindowAttribute,
};
use windows::Win32::Graphics::Gdi::{
    BLACK_BRUSH, EnumDisplayMonitors, GetMonitorInfoW, GetStockObject, HBRUSH, HDC, HMONITOR,
    MONITORINFO,
};
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RegGetValueW,
};
use windows::Win32::System::SystemInformation::OSVERSIONINFOW;
use windows::Win32::UI::Controls::MARGINS;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForMonitor, MDT_EFFECTIVE_DPI,
    SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::Input::KeyboardAndMouse::VK_ESCAPE;
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, HHOOK,
    HWND_TOPMOST, KBDLLHOOKSTRUCT, LWA_ALPHA, MSG, PostQuitMessage, RegisterClassW,
    SW_SHOWNOACTIVATE, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SetLayeredWindowAttributes,
    SetTimer, SetWindowDisplayAffinity, SetWindowPos, SetWindowsHookExW, ShowWindow,
    UnhookWindowsHookEx, WDA_EXCLUDEFROMCAPTURE, WH_KEYBOARD_LL, WM_KEYDOWN, WM_SYSKEYDOWN,
    WM_TIMER, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST,
    WS_EX_TRANSPARENT, WS_POPUP,
};
use windows::core::{BOOL, PCWSTR, s, w};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Acrylic,
    Backdrop,
    Solid,
}

struct Args {
    mode: Mode,
    seconds: u64,
    click_through: bool,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        mode: Mode::Solid,
        seconds: 15,
        click_through: false,
    };
    let mut mode = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => {
                mode = Some(match it.next().as_deref() {
                    Some("acrylic") => Mode::Acrylic,
                    Some("backdrop") => Mode::Backdrop,
                    Some("solid") => Mode::Solid,
                    other => bail!("--mode must be acrylic|backdrop|solid, got {other:?}"),
                })
            }
            "--seconds" => args.seconds = it.next().context("--seconds needs a value")?.parse()?,
            "--click-through" => args.click_through = true,
            other => bail!(
                "unknown argument {other:?}; usage: --mode <acrylic|backdrop|solid> [--seconds 15] [--click-through]"
            ),
        }
    }
    args.mode = mode.context("--mode <acrylic|backdrop|solid> is required")?;
    Ok(args)
}

fn reg_dword(root: HKEY, key: PCWSTR, value: PCWSTR) -> Option<u32> {
    let mut data = 0u32;
    let mut size = size_of::<u32>() as u32;
    // SAFETY: data/size point to a valid u32 and its byte size.
    let err = unsafe {
        RegGetValueW(
            root,
            key,
            value,
            RRF_RT_REG_DWORD,
            None,
            Some(&mut data as *mut u32 as *mut c_void),
            Some(&mut size),
        )
    };
    err.is_ok().then_some(data)
}

fn windows_build() -> (u32, u32) {
    let mut info = OSVERSIONINFOW {
        dwOSVersionInfoSize: size_of::<OSVERSIONINFOW>() as u32,
        ..Default::default()
    };
    // SAFETY: info is a valid, size-tagged OSVERSIONINFOW. RtlGetVersion is used because
    // GetVersionExW lies to manifest-less processes.
    let _ = unsafe { RtlGetVersion(&mut info) };
    let ubr = reg_dword(
        HKEY_LOCAL_MACHINE,
        w!(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"),
        w!("UBR"),
    )
    .unwrap_or(0);
    (info.dwBuildNumber, ubr)
}

unsafe extern "system" fn enum_monitor(m: HMONITOR, _: HDC, _: *mut RECT, out: LPARAM) -> BOOL {
    // SAFETY: `out` is the &mut Vec passed to EnumDisplayMonitors in main, alive for the call.
    let v = unsafe { &mut *(out.0 as *mut Vec<HMONITOR>) };
    v.push(m);
    true.into()
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

#[repr(C)]
struct AccentPolicy {
    accent_state: u32,
    accent_flags: u32,
    gradient_color: u32,
    animation_id: u32,
}

#[repr(C)]
struct WindowCompositionAttribData {
    attrib: u32,
    pv_data: *mut c_void,
    cb_data: usize,
}

type SetWindowCompositionAttributeFn =
    unsafe extern "system" fn(HWND, *mut WindowCompositionAttribData) -> BOOL;

const WCA_ACCENT_POLICY: u32 = 19;
const ACCENT_ENABLE_ACRYLICBLURBEHIND: u32 = 4;

fn apply_acrylic(hwnd: HWND) -> Result<()> {
    // SAFETY: user32 is always loaded in a GUI process; the export is looked up by name.
    let proc = unsafe {
        let user32 = GetModuleHandleW(w!("user32.dll"))?;
        GetProcAddress(user32, s!("SetWindowCompositionAttribute"))
    }
    .context("user32!SetWindowCompositionAttribute not found")?;
    // SAFETY: undocumented but stable signature BOOL(HWND, WINDOWCOMPOSITIONATTRIBDATA*).
    let f: SetWindowCompositionAttributeFn = unsafe { std::mem::transmute(proc) };
    let mut policy = AccentPolicy {
        accent_state: ACCENT_ENABLE_ACRYLICBLURBEHIND,
        accent_flags: 2,
        gradient_color: 0x99_10_10_10, // AABBGGRR: dark tint, ~60% opaque
        animation_id: 0,
    };
    let mut data = WindowCompositionAttribData {
        attrib: WCA_ACCENT_POLICY,
        pv_data: &mut policy as *mut AccentPolicy as *mut c_void,
        cb_data: size_of::<AccentPolicy>(),
    };
    // SAFETY: data and policy outlive the call.
    if !unsafe { f(hwnd, &mut data) }.as_bool() {
        bail!(
            "SetWindowCompositionAttribute returned FALSE: {}",
            windows::core::Error::from_thread()
        );
    }
    Ok(())
}

fn apply_backdrop(hwnd: HWND, build: u32) -> Result<()> {
    if build < 22621 {
        println!(
            "  note: DWMSBT_TRANSIENTWINDOW needs Windows 11 22H2 (build 22621+); this is build {build}, expect failure/no effect"
        );
    }
    let kind: DWM_SYSTEMBACKDROP_TYPE = DWMSBT_TRANSIENTWINDOW;
    // SAFETY: pointer/size describe `kind`, which outlives the call.
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_SYSTEMBACKDROP_TYPE,
            &kind as *const _ as *const c_void,
            size_of::<DWM_SYSTEMBACKDROP_TYPE>() as u32,
        )
    }
    .context(
        "DwmSetWindowAttribute(DWMWA_SYSTEMBACKDROP_TYPE) unsupported/failed (needs Win11 22H2+)",
    )?;
    // Only extend the frame once the backdrop is accepted: an extended frame over a black
    // client with no backdrop renders transparent, which would fail open (FR-OV-005).
    let m = MARGINS {
        cxLeftWidth: -1,
        cxRightWidth: -1,
        cyTopHeight: -1,
        cyBottomHeight: -1,
    };
    // SAFETY: valid hwnd and margins.
    unsafe { DwmExtendFrameIntoClientArea(hwnd, &m) }
        .context("DwmExtendFrameIntoClientArea(-1) failed")?;
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;

    // SAFETY: plain Win32 call, before any window exists.
    if let Err(e) =
        unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) }
    {
        println!("SetProcessDpiAwarenessContext(PMv2) failed: {e}");
    }

    let (build, ubr) = windows_build();
    let transparency = reg_dword(
        HKEY_CURRENT_USER,
        w!(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"),
        w!("EnableTransparency"),
    );
    println!(
        "Windows build {build}.{ubr} ({})",
        if build >= 22000 {
            "Windows 11"
        } else {
            "Windows 10"
        }
    );
    println!(
        "Transparency effects (EnableTransparency): {}",
        match transparency {
            Some(1) => "on".to_string(),
            Some(v) => format!("off ({v})"),
            None => "unknown (value missing)".to_string(),
        }
    );
    println!(
        "mode={:?} seconds={} click_through={}",
        args.mode, args.seconds, args.click_through
    );

    let mut monitors: Vec<HMONITOR> = Vec::new();
    // SAFETY: the callback only pushes into `monitors`, which outlives the call.
    unsafe {
        EnumDisplayMonitors(
            None,
            None,
            Some(enum_monitor),
            LPARAM(&mut monitors as *mut _ as isize),
        )
    }
    .ok()
    .context("EnumDisplayMonitors failed")?;

    // SAFETY: standard class registration with a stock brush and our module handle.
    let hinstance = unsafe { GetModuleHandleW(None)? };
    let class = w!("fasec_sp1_overlay");
    let wc = WNDCLASSW {
        lpfnWndProc: Some(wnd_proc),
        hInstance: hinstance.into(),
        lpszClassName: class,
        // Black client: opaque for `solid`, and the "glass" colour for an extended DWM frame.
        hbrBackground: HBRUSH(unsafe { GetStockObject(BLACK_BRUSH) }.0),
        ..Default::default()
    };
    if unsafe { RegisterClassW(&wc) } == 0 {
        bail!(
            "RegisterClassW failed: {}",
            windows::core::Error::from_thread()
        );
    }

    let mut ex = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE;
    if args.click_through {
        ex |= WS_EX_LAYERED | WS_EX_TRANSPARENT;
    }

    let mut hwnds = Vec::new();
    for (i, &mon) in monitors.iter().enumerate() {
        let mut mi = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        // SAFETY: mi is size-tagged; mon came from EnumDisplayMonitors.
        if !unsafe { GetMonitorInfoW(mon, &mut mi) }.as_bool() {
            println!("monitor {i}: GetMonitorInfoW failed, skipped");
            continue;
        }
        let (mut dx, mut dy) = (0, 0);
        // SAFETY: valid monitor handle and out pointers.
        let dpi = match unsafe { GetDpiForMonitor(mon, MDT_EFFECTIVE_DPI, &mut dx, &mut dy) } {
            Ok(()) => format!("{dx}x{dy} ({}%)", dx * 100 / 96),
            Err(e) => format!("error {e}"),
        };
        let r = mi.rcMonitor;
        println!(
            "monitor {i}: rect=({},{})-({},{}) {}x{} dpi={dpi}{}",
            r.left,
            r.top,
            r.right,
            r.bottom,
            r.right - r.left,
            r.bottom - r.top,
            if mi.dwFlags & 1 != 0 { " primary" } else { "" }
        );

        // SAFETY: registered class, valid instance, no parent/menu/param.
        let hwnd = unsafe {
            CreateWindowExW(
                ex,
                class,
                w!("fasec SP-1 overlay"),
                WS_POPUP,
                r.left,
                r.top,
                r.right - r.left,
                r.bottom - r.top,
                None,
                None,
                Some(hinstance.into()),
                None,
            )
        }
        .with_context(|| format!("CreateWindowExW for monitor {i}"))?;

        if args.click_through {
            // SAFETY: valid layered hwnd. A layered window is invisible until this is set.
            if let Err(e) = unsafe { SetLayeredWindowAttributes(hwnd, COLORREF(0), 255, LWA_ALPHA) }
            {
                println!("  SetLayeredWindowAttributes failed: {e}");
            }
        }
        // SAFETY: valid top-level hwnd owned by this process.
        match unsafe { SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE) } {
            Ok(()) => println!("  WDA_EXCLUDEFROMCAPTURE: ok"),
            Err(e) => println!("  WDA_EXCLUDEFROMCAPTURE failed: {e}"),
        }
        let applied = match args.mode {
            Mode::Acrylic => apply_acrylic(hwnd),
            Mode::Backdrop => apply_backdrop(hwnd, build),
            Mode::Solid => Ok(()),
        };
        match applied {
            Ok(()) => println!("  {:?} backend: applied", args.mode),
            Err(e) => println!(
                "  {:?} backend FAILED, window stays opaque black: {e:#}",
                args.mode
            ),
        }
        // SAFETY: valid hwnd; SW_SHOWNOACTIVATE keeps focus where it is.
        let _ = unsafe { ShowWindow(hwnd, SW_SHOWNOACTIVATE) };
        hwnds.push(hwnd);
    }
    if hwnds.is_empty() {
        bail!("no overlay windows created");
    }

    // SAFETY: LL hook with a proc in this module; removed by HookGuard.
    let _hook = HookGuard(
        unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(kb_hook), Some(hinstance.into()), 0) }
            .context("SetWindowsHookExW(WH_KEYBOARD_LL)")?,
    );
    // SAFETY: thread timer (no hwnd); WM_TIMER arrives in our queue every 250 ms (FR-OV-007).
    if unsafe { SetTimer(None, 0, 250, None) } == 0 {
        bail!("SetTimer failed: {}", windows::core::Error::from_thread());
    }

    let deadline = Instant::now() + Duration::from_secs(args.seconds);
    let (mut reasserts, mut reassert_errors) = (0u32, 0u32);
    let mut msg = MSG::default();
    // SAFETY: standard message loop on the thread that owns the windows and the hook.
    while unsafe { GetMessageW(&mut msg, None, 0, 0) }.as_bool() {
        if msg.message == WM_TIMER && msg.hwnd.is_invalid() {
            for &h in &hwnds {
                let flags = SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE;
                // SAFETY: valid hwnd.
                match unsafe { SetWindowPos(h, Some(HWND_TOPMOST), 0, 0, 0, 0, flags) } {
                    Ok(()) => reasserts += 1,
                    Err(e) => {
                        if reassert_errors == 0 {
                            println!("SetWindowPos(HWND_TOPMOST) failed: {e}");
                        }
                        reassert_errors += 1;
                    }
                }
            }
            if Instant::now() >= deadline {
                println!("{} s elapsed, closing", args.seconds);
                break;
            }
            continue;
        }
        // SAFETY: msg came from GetMessageW.
        unsafe { DispatchMessageW(&msg) };
    }
    println!("top-most re-asserted {reasserts} times, {reassert_errors} errors");
    Ok(())
}
