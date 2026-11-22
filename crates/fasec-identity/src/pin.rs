//! Owner PIN hashing: Argon2id, PHC string (NFR-SEC-002).

use argon2::password_hash::Error as HashError;
use argon2::{Algorithm, Argon2, Params, PasswordHash, PasswordHasher, PasswordVerifier, Version};
use zeroize::Zeroizing;

use crate::error::IdentityError;

/// Argon2id cost: 19 MiB, 2 passes, 1 lane (OWASP minimum).
const M_COST_KIB: u32 = 19 * 1024;
const T_COST: u32 = 2;
const P_COST: u32 = 1;
/// Largest stored memory cost accepted on verify: 1 GiB.
const M_COST_MAX_KIB: u32 = 1024 * 1024;

fn argon2() -> Argon2<'static> {
    let params = Params::new(M_COST_KIB, T_COST, P_COST, None).expect("valid Argon2 params");
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
}

/// 4–12 ASCII digits.
pub(crate) fn is_valid(pin: &str) -> bool {
    (4..=12).contains(&pin.len()) && pin.bytes().all(|b| b.is_ascii_digit())
}

/// Hashes a PIN with a fresh 16-byte random salt and returns the PHC string.
pub(crate) fn hash(pin: &str) -> Result<String, IdentityError> {
    if !is_valid(pin) {
        return Err(IdentityError::Pin(
            "PIN must be 4 to 12 ASCII digits".into(),
        ));
    }
    let pin = Zeroizing::new(pin.to_owned());
    let phc = argon2()
        .hash_password(pin.as_bytes())
        .map_err(|e| IdentityError::Pin(format!("hashing failed: {e}")))?;
    Ok(phc.to_string())
}

/// Constant-time check of `pin` against a stored PHC string. `Ok(false)` on mismatch;
/// an unparsable or non-Argon2id stored hash is an error.
pub(crate) fn verify(pin: &str, phc: &str) -> Result<bool, IdentityError> {
    let invalid = || IdentityError::Pin("stored PIN hash is invalid".into());
    let parsed = PasswordHash::new(phc).map_err(|_| invalid())?;
    if parsed.algorithm != argon2::ARGON2ID_IDENT {
        return Err(invalid());
    }
    // Bound the stored cost before running Argon2: a planted hash must not force a huge
    // allocation, nor be accepted with weaker-than-policy params.
    let params = Params::try_from(&parsed).map_err(|_| invalid())?;
    if !(M_COST_KIB..=M_COST_MAX_KIB).contains(&params.m_cost())
        || params.t_cost() < T_COST
        || params.p_cost() != P_COST
    {
        return Err(IdentityError::Pin(
            "stored PIN hash parameters are out of bounds".into(),
        ));
    }
    if !is_valid(pin) {
        return Ok(false);
    }
    let pin = Zeroizing::new(pin.to_owned());
    match argon2().verify_password(pin.as_bytes(), &parsed) {
        Ok(()) => Ok(true),
        Err(HashError::PasswordInvalid) => Ok(false),
        Err(_) => Err(invalid()),
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    const SALT_HASH: &str = "c29tZXNhbHRzb21lc2FsdA$RdescudvJCsgt3ub+b+dWRWJTmaaJObG";

    #[test]
    fn non_argon2id_or_degenerate_stored_hash_is_error() {
        for head in [
            "$argon2i$v=19$m=19456,t=2,p=1", // parsable, wrong variant
            "$argon2d$v=19$m=19456,t=2,p=1",
            "$argon2id$v=19$m=19456,t=0,p=1", // zero passes
            "$argon2id$v=19$m=19456,t=2,p=0", // zero lanes
            "$argon2id$v=19$m=1,t=2,p=1",     // memory below 8*p
        ] {
            let phc = format!("{head}${SALT_HASH}");
            assert!(PasswordHash::new(&phc).is_ok(), "fixture must parse: {phc}");
            assert!(
                matches!(verify("1234", &phc), Err(IdentityError::Pin(_))),
                "{phc} must be an error, never Ok"
            );
        }
        let scrypt = "$scrypt$ln=4,r=8,p=1$c29tZXNhbHQ$RdescudvJCsgt3ub+b+dWRWJTmaaJObG";
        assert!(verify("1234", scrypt).is_err());
        assert!(verify("1234", "").is_err());
    }

    #[test]
    fn out_of_bounds_stored_params_are_refused_before_argon2_runs() {
        let out_of_bounds = |head: &str| {
            let phc = format!("{head}${SALT_HASH}");
            assert!(PasswordHash::new(&phc).is_ok(), "fixture must parse: {phc}");
            // this message is only produced before argon2 is invoked
            matches!(verify("1234", &phc), Err(IdentityError::Pin(m)) if m.contains("out of bounds"))
        };
        for head in [
            "$argon2id$v=19$m=2000000,t=2,p=1", // ~2 GiB: must not be allocated
            "$argon2id$v=19$m=1048577,t=2,p=1",
            "$argon2id$v=19$m=19456,t=1,p=1", // weaker than policy
            "$argon2id$v=19$m=19455,t=2,p=1",
            "$argon2id$v=19$m=19456,t=2,p=2",
        ] {
            assert!(out_of_bounds(head), "{head}");
        }
        // policy params pass the bounds check (the 1 GiB edge is not run: it would allocate)
        assert!(!out_of_bounds("$argon2id$v=19$m=19456,t=2,p=1"));
        let phc = hash("1234").unwrap();
        assert!(verify("1234", &phc).unwrap());
        assert!(!verify("4321", &phc).unwrap());
    }
}
