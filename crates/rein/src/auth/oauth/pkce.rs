use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

fn valid_verifier(verifier: &str) -> bool {
    (43..=128).contains(&verifier.len())
        && verifier
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~'))
}

pub fn s256_challenge(verifier: &str) -> Result<String, String> {
    if !valid_verifier(verifier) {
        return Err("invalid PKCE code_verifier".to_string());
    }
    let digest = Sha256::digest(verifier.as_bytes());
    Ok(URL_SAFE_NO_PAD.encode(digest))
}

pub fn verify_s256(verifier: &str, expected_challenge: &str) -> bool {
    let Ok(actual) = s256_challenge(verifier) else {
        return false;
    };
    actual
        .as_bytes()
        .ct_eq(expected_challenge.as_bytes())
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s256_challenge_matches_rfc7636_vector() {
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(
            s256_challenge(verifier).expect("valid verifier"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
        assert!(verify_s256(
            verifier,
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        ));
    }

    #[test]
    fn verifier_rejects_invalid_length_and_characters() {
        assert!(s256_challenge("short").is_err());
        assert!(s256_challenge("abcd+abcd+abcd+abcd+abcd+abcd+abcd+abcd+abcd+abcd").is_err());
    }

    #[test]
    fn verifier_comparison_rejects_wrong_challenge() {
        let verifier = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
        assert!(!verify_s256(verifier, "wrong"));
    }
}
