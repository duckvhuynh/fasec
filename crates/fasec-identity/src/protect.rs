//! Pluggable at-rest protection for biometric bytes (DPAPI in production).

/// Encrypts and decrypts secrets at rest. Implementations must authenticate the
/// ciphertext: tampered or foreign blobs must fail `unprotect`, never decode to garbage.
pub trait SecretProtector: Send + Sync {
    fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, ProtectError>;
    fn unprotect(&self, ciphertext: &[u8]) -> Result<zeroize::Zeroizing<Vec<u8>>, ProtectError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ProtectError {
    #[error("protection backend unsupported on this platform")]
    Unsupported,
    #[error("protect failed: {0}")]
    Protect(String),
    #[error("unprotect failed: {0}")]
    Unprotect(String),
}

/// Test-only protectors. Enabled for this crate's tests and via the `testing` feature.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    use super::{ProtectError, SecretProtector};
    use zeroize::Zeroizing;

    /// **INSECURE. For tests only.** XORs every byte with a one-byte key and appends a
    /// 4-byte FNV-1a tag so tampering, truncation and a different key are detected.
    /// It provides no confidentiality against anyone; never use it for real data.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct XorProtector(pub u8);

    impl Default for XorProtector {
        fn default() -> Self {
            Self(0x5A)
        }
    }

    impl XorProtector {
        fn tag(&self, plaintext: &[u8]) -> [u8; 4] {
            let mut h: u32 = 0x811c_9dc5;
            for &b in std::iter::once(&self.0).chain(plaintext) {
                h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
            }
            h.to_le_bytes()
        }
    }

    impl SecretProtector for XorProtector {
        fn protect(&self, plaintext: &[u8]) -> Result<Vec<u8>, ProtectError> {
            let tag = self.tag(plaintext);
            Ok(plaintext.iter().chain(&tag).map(|b| b ^ self.0).collect())
        }

        fn unprotect(&self, ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>, ProtectError> {
            let Some(body_len) = ciphertext.len().checked_sub(4) else {
                return Err(ProtectError::Unprotect("blob too short".into()));
            };
            let mut plain = Zeroizing::new(Vec::with_capacity(ciphertext.len()));
            plain.extend(ciphertext.iter().map(|b| b ^ self.0));
            let tag: [u8; 4] = plain[body_len..].try_into().expect("4-byte tail");
            plain.truncate(body_len);
            if tag != self.tag(&plain) {
                return Err(ProtectError::Unprotect("integrity check failed".into()));
            }
            Ok(plain)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SecretProtector;
    use super::testing::XorProtector;

    #[test]
    fn xor_roundtrip() {
        let p = XorProtector::default();
        let ct = p.protect(b"hello").unwrap();
        assert_ne!(&ct[..5], b"hello");
        assert_eq!(&p.unprotect(&ct).unwrap()[..], b"hello");
    }

    #[test]
    fn xor_detects_tamper_truncation_and_other_key() {
        let p = XorProtector(1);
        let mut ct = p.protect(b"hello").unwrap();
        assert!(XorProtector(2).unprotect(&ct).is_err());
        assert!(p.unprotect(&ct[..ct.len() - 1]).is_err());
        assert!(p.unprotect(&ct[..2]).is_err());
        ct[0] ^= 0x80;
        assert!(p.unprotect(&ct).is_err());
    }
}
