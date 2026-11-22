//! Win32/WinRT wrappers: hotkeys, session events, Windows Hello, DPAPI, power.

mod dpapi;

pub use dpapi::{Dpapi, DpapiError};
