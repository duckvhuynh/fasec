//! Win32/WinRT wrappers: hotkeys, session events, Windows Hello, DPAPI, power.

mod dpapi;
#[cfg(feature = "identity")]
mod identity;

pub use dpapi::{Dpapi, DpapiError};
