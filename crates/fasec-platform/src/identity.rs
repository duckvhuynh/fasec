//! `Dpapi` as the gallery's `SecretProtector` (feature `identity`).

use fasec_identity::{ProtectError, SecretProtector};
use zeroize::Zeroizing;

use crate::dpapi::{Dpapi, DpapiError};

impl From<DpapiError> for ProtectError {
    fn from(e: DpapiError) -> Self {
        match e {
            DpapiError::Unsupported => ProtectError::Unsupported,
            DpapiError::Protect(m) => ProtectError::Protect(m),
            DpapiError::Unprotect(m) => ProtectError::Unprotect(m),
        }
    }
}

impl SecretProtector for Dpapi {
    fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, ProtectError> {
        Ok(Dpapi::protect(self, plaintext)?)
    }

    fn unprotect(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>, ProtectError> {
        Ok(Dpapi::unprotect(self, ciphertext)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dpapi_errors_map_to_protect_errors() {
        let m = |e| ProtectError::from(e).to_string();
        assert_eq!(
            m(DpapiError::Unsupported),
            ProtectError::Unsupported.to_string()
        );
        assert_eq!(m(DpapiError::Protect("p".into())), "protect failed: p");
        assert_eq!(m(DpapiError::Unprotect("u".into())), "unprotect failed: u");
    }

    #[test]
    fn dpapi_works_as_secret_protector() {
        let p: &dyn SecretProtector = &Dpapi::new();
        let res = p.protect(b"template").and_then(|ct| p.unprotect(&ct));
        if cfg!(windows) {
            assert_eq!(res.unwrap().as_slice(), b"template");
        } else {
            assert!(matches!(res, Err(ProtectError::Unsupported)));
        }
    }
}
