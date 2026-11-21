//! Windows camera privacy setting for desktop (non-packaged) apps (FR-CAM-008). Read-only.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    Allowed,
    /// "Let desktop apps access your camera" is off for this user.
    DeniedByUser,
    /// Camera access is off for the whole device (HKLM, admin setting).
    DeniedBySystem,
    /// Not set or not readable (also on non-Windows).
    Unknown,
}

const CONSENT_WEBCAM: &str =
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam";

/// Reads HKCU `...\ConsentStore\webcam\NonPackaged` `Value`, plus HKLM `...\webcam` and
/// `...\webcam\NonPackaged` `Value` (HKLM has only `webcam` on Windows 11 here).
pub fn desktop_camera_access() -> Access {
    let np = format!(r"{CONSENT_WEBCAM}\NonPackaged");
    classify(
        reg::read(reg::Hive::User, &np).as_deref(),
        [
            reg::read(reg::Hive::Machine, CONSENT_WEBCAM).as_deref(),
            reg::read(reg::Hive::Machine, &np).as_deref(),
        ],
    )
}

/// `user`: HKCU NonPackaged value; `system`: HKLM values. "Deny" anywhere in HKLM wins.
fn classify(user: Option<&str>, system: [Option<&str>; 2]) -> Access {
    let is = |v: Option<&str>, want: &str| v.is_some_and(|v| v.eq_ignore_ascii_case(want));
    if system.iter().any(|&v| is(v, "Deny")) {
        Access::DeniedBySystem
    } else if is(user, "Deny") {
        Access::DeniedByUser
    } else if is(user, "Allow") {
        Access::Allowed
    } else {
        Access::Unknown
    }
}

#[cfg(windows)]
mod reg {
    use windows::Win32::Foundation::ERROR_SUCCESS;
    use windows::Win32::System::Registry::{
        HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW,
    };
    use windows::core::{HSTRING, w};

    pub(super) enum Hive {
        User,
        Machine,
    }

    /// The REG_SZ `Value` under `subkey`, or `None` if missing or not a string.
    pub(super) fn read(hive: Hive, subkey: &str) -> Option<String> {
        let hkey = match hive {
            Hive::User => HKEY_CURRENT_USER,
            Hive::Machine => HKEY_LOCAL_MACHINE,
        };
        let subkey = HSTRING::from(subkey);
        let mut buf = [0u16; 64]; // "Allow"/"Deny"; longer values are not ours to interpret
        let mut len = u32::try_from(std::mem::size_of_val(&buf)).ok()?;
        // SAFETY: buf/len describe a valid writable buffer; RRF_RT_REG_SZ NUL-terminates it.
        let rc = unsafe {
            RegGetValueW(
                hkey,
                &subkey,
                w!("Value"),
                RRF_RT_REG_SZ,
                None,
                Some(buf.as_mut_ptr().cast()),
                Some(&mut len),
            )
        };
        if rc != ERROR_SUCCESS {
            return None;
        }
        let chars = (len as usize / 2).min(buf.len());
        let s = String::from_utf16_lossy(&buf[..chars]);
        Some(s.trim_end_matches('\0').to_owned())
    }
}

#[cfg(not(windows))]
mod reg {
    pub(super) enum Hive {
        User,
        Machine,
    }
    pub(super) fn read(_: Hive, _: &str) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_registry_values() {
        let allow = Some("Allow");
        let deny = Some("Deny");
        assert_eq!(classify(allow, [allow, None]), Access::Allowed);
        assert_eq!(classify(allow, [None, None]), Access::Allowed);
        assert_eq!(classify(deny, [allow, None]), Access::DeniedByUser);
        assert_eq!(classify(allow, [deny, None]), Access::DeniedBySystem);
        assert_eq!(classify(deny, [None, deny]), Access::DeniedBySystem);
        assert_eq!(classify(None, [allow, None]), Access::Unknown);
        assert_eq!(classify(Some("Prompt"), [allow, None]), Access::Unknown);
    }

    #[test]
    fn permission_reads_without_panic() {
        let access = desktop_camera_access();
        println!("desktop camera access: {access:?}");
        #[cfg(not(windows))]
        assert_eq!(access, Access::Unknown);
    }
}
