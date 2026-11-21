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

/// Reads the `Value` of `...\ConsentStore\webcam` ("Let apps access your camera") and
/// `...\webcam\NonPackaged` ("Let desktop apps access your camera") in HKCU, and the same two
/// keys in HKLM (HKLM has only `webcam` on Windows 11 here).
pub fn desktop_camera_access() -> Access {
    let np = format!(r"{CONSENT_WEBCAM}\NonPackaged");
    classify(
        [
            reg::read(reg::Hive::User, CONSENT_WEBCAM).as_deref(),
            reg::read(reg::Hive::User, &np).as_deref(),
        ],
        [
            reg::read(reg::Hive::Machine, CONSENT_WEBCAM).as_deref(),
            reg::read(reg::Hive::Machine, &np).as_deref(),
        ],
    )
}

/// `user`/`system`: `[webcam, webcam\NonPackaged]` values from HKCU/HKLM. "Deny" anywhere in
/// HKLM wins, then "Deny" anywhere in HKCU; only an explicit HKCU NonPackaged "Allow" is Allowed.
fn classify(user: [Option<&str>; 2], system: [Option<&str>; 2]) -> Access {
    let is = |v: Option<&str>, want: &str| v.is_some_and(|v| v.eq_ignore_ascii_case(want));
    if system.iter().any(|&v| is(v, "Deny")) {
        Access::DeniedBySystem
    } else if user.iter().any(|&v| is(v, "Deny")) {
        Access::DeniedByUser
    } else if is(user[1], "Allow") {
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
        let (none, prompt) = (None, Some("Prompt"));
        assert_eq!(classify([allow, allow], [allow, none]), Access::Allowed);
        assert_eq!(classify([none, allow], [none, none]), Access::Allowed);
        assert_eq!(classify([allow, deny], [allow, none]), Access::DeniedByUser);
        // Master "Let apps access your camera" off while the desktop-apps switch still says Allow.
        assert_eq!(classify([deny, allow], [allow, none]), Access::DeniedByUser);
        assert_eq!(
            classify([allow, allow], [deny, none]),
            Access::DeniedBySystem
        );
        assert_eq!(classify([deny, deny], [none, deny]), Access::DeniedBySystem);
        assert_eq!(classify([allow, none], [allow, none]), Access::Unknown);
        assert_eq!(classify([allow, prompt], [allow, none]), Access::Unknown);
    }

    #[test]
    fn permission_reads_without_panic() {
        let access = desktop_camera_access();
        println!("desktop camera access: {access:?}");
        #[cfg(not(windows))]
        assert_eq!(access, Access::Unknown);
    }

    #[test]
    fn classify_is_case_insensitive_and_unset_means_unknown() {
        assert_eq!(
            classify([Some("allow"), Some("DENY")], [None, None]),
            Access::DeniedByUser
        );
        assert_eq!(
            classify([Some("ALLOW"), Some("aLLoW")], [None, None]),
            Access::Allowed
        );
        assert_eq!(classify([None, None], [None, None]), Access::Unknown);
        assert_eq!(
            classify([Some(""), Some("")], [None, None]),
            Access::Unknown
        );
        // A device-wide Allow never grants on its own; only the user's desktop-apps switch does.
        assert_eq!(
            classify([None, None], [Some("Allow"), Some("Allow")]),
            Access::Unknown
        );
    }
}
