//! DPAPI (current-user scope) protection for secrets stored at rest.

use zeroize::Zeroizing;

#[derive(Debug, thiserror::Error)]
pub enum DpapiError {
    #[error("DPAPI unsupported on this platform")]
    Unsupported,
    #[error("protect failed: {0}")]
    Protect(String),
    #[error("unprotect failed: {0}")]
    Unprotect(String),
}

/// Encrypts bytes so only the current Windows user can decrypt them.
#[derive(Debug, Default, Clone, Copy)]
pub struct Dpapi;

impl Dpapi {
    pub fn new() -> Self {
        Self
    }

    pub fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, DpapiError> {
        imp::protect(plaintext, ENTROPY)
    }

    pub fn unprotect(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>, DpapiError> {
        imp::unprotect(ciphertext, ENTROPY)
    }
}

/// Application-specific entropy: blobs from other DPAPI users of this account do not decrypt here.
const ENTROPY: &[u8] = b"fasec.gallery.v1";

#[cfg(windows)]
mod imp {
    use super::DpapiError;
    use windows::Win32::Foundation::{HLOCAL, LocalFree};
    use windows::Win32::Security::Cryptography::{
        CRYPT_INTEGER_BLOB, CRYPTPROTECT_UI_FORBIDDEN, CryptProtectData, CryptUnprotectData,
    };
    use windows::core::PCWSTR;
    use zeroize::{Zeroize, Zeroizing};

    fn blob(data: &[u8]) -> Option<CRYPT_INTEGER_BLOB> {
        Some(CRYPT_INTEGER_BLOB {
            cbData: u32::try_from(data.len()).ok()?,
            // DPAPI only reads input blobs; the API is just not const-correct.
            pbData: data.as_ptr().cast_mut(),
        })
    }

    /// Copies a DPAPI-allocated output blob, optionally wipes it, and frees it with `LocalFree`.
    ///
    /// # Safety
    /// `out` must be an output blob filled by a successful `CryptProtectData`/`CryptUnprotectData`.
    unsafe fn take(out: CRYPT_INTEGER_BLOB, wipe: bool) -> Vec<u8> {
        if out.pbData.is_null() {
            return Vec::new();
        }
        // SAFETY: DPAPI allocated `cbData` bytes at `pbData`; we own them until LocalFree.
        let buf = unsafe { std::slice::from_raw_parts_mut(out.pbData, out.cbData as usize) };
        let copy = buf.to_vec();
        if wipe {
            buf.zeroize();
        }
        // SAFETY: `pbData` was allocated by DPAPI with LocalAlloc and is freed exactly once.
        unsafe { LocalFree(Some(HLOCAL(out.pbData.cast()))) };
        copy
    }

    pub(super) fn protect(plaintext: &[u8], entropy: &[u8]) -> Result<Vec<u8>, DpapiError> {
        let too_big = || DpapiError::Protect("input too large".into());
        let input = blob(plaintext).ok_or_else(too_big)?;
        let entropy = blob(entropy).ok_or_else(too_big)?;
        let mut out = CRYPT_INTEGER_BLOB::default();
        // SAFETY: all blobs point to live memory for the duration of the call.
        unsafe {
            CryptProtectData(
                &input,
                PCWSTR::null(),
                Some(&entropy),
                None,
                None,
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut out,
            )
        }
        .map_err(|e| DpapiError::Protect(e.to_string()))?;
        // SAFETY: `out` was filled by the successful call above.
        Ok(unsafe { take(out, false) })
    }

    pub(super) fn unprotect(
        ciphertext: &[u8],
        entropy: &[u8],
    ) -> Result<Zeroizing<Vec<u8>>, DpapiError> {
        let too_big = || DpapiError::Unprotect("input too large".into());
        let input = blob(ciphertext).ok_or_else(too_big)?;
        let entropy = blob(entropy).ok_or_else(too_big)?;
        let mut out = CRYPT_INTEGER_BLOB::default();
        // SAFETY: all blobs point to live memory for the duration of the call.
        unsafe {
            CryptUnprotectData(
                &input,
                None,
                Some(&entropy),
                None,
                None,
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut out,
            )
        }
        .map_err(|e| DpapiError::Unprotect(e.to_string()))?;
        // SAFETY: `out` was filled by the successful call above; plaintext is wiped before free.
        Ok(Zeroizing::new(unsafe { take(out, true) }))
    }
}

#[cfg(not(windows))]
mod imp {
    use super::DpapiError;
    use zeroize::Zeroizing;

    pub(super) fn protect(_: &[u8], _: &[u8]) -> Result<Vec<u8>, DpapiError> {
        Err(DpapiError::Unsupported)
    }

    pub(super) fn unprotect(_: &[u8], _: &[u8]) -> Result<Zeroizing<Vec<u8>>, DpapiError> {
        Err(DpapiError::Unsupported)
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    fn dpapi_roundtrip() {
        let d = Dpapi::new();
        let ct = d.protect(b"face template bytes").unwrap();
        assert_eq!(d.unprotect(&ct).unwrap().as_slice(), b"face template bytes");
    }

    #[test]
    fn dpapi_ciphertext_differs_from_plaintext() {
        let pt = b"face template bytes";
        let ct = Dpapi::new().protect(pt).unwrap();
        assert!(ct.len() > pt.len());
        assert!(!ct.windows(pt.len()).any(|w| w == pt));
    }

    #[test]
    fn dpapi_rejects_tampered_ciphertext() {
        let d = Dpapi::new();
        let mut ct = d.protect(b"face template bytes").unwrap();
        let last = ct.len() - 1;
        ct[last] ^= 0x01;
        assert!(matches!(d.unprotect(&ct), Err(DpapiError::Unprotect(_))));
    }

    #[test]
    fn dpapi_rejects_wrong_entropy() {
        let ct = imp::protect(b"face template bytes", b"other.entropy").unwrap();
        assert!(matches!(
            Dpapi::new().unprotect(&ct),
            Err(DpapiError::Unprotect(_))
        ));
    }

    #[test]
    fn empty_plaintext_roundtrip() {
        let d = Dpapi::new();
        let ct = d.protect(b"").unwrap();
        assert!(d.unprotect(&ct).unwrap().is_empty());
    }
}

#[cfg(all(test, not(windows)))]
mod tests {
    use super::*;

    #[test]
    fn non_windows_is_unsupported() {
        assert!(matches!(
            Dpapi::new().protect(b"x"),
            Err(DpapiError::Unsupported)
        ));
        assert!(matches!(
            Dpapi::new().unprotect(b"x"),
            Err(DpapiError::Unsupported)
        ));
    }
}
