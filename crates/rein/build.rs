//! v0.32.1 — build-time source fingerprint for eval-gate run-freshness.
//!
//! Emits `REIN_BUILD_FINGERPRINT`: a hash of `src/` AND the eval-gate fixture
//! corpus, stamped on every scorecard.  A *run* scorecard whose value differs
//! from the running binary's is rejected as stale (codex v0.32.0 R9 P2-#2:
//! same-version source drift that `CARGO_PKG_VERSION` alone misses).  The
//! committed baseline is exempt — its `src/` may legitimately predate current
//! scoring logic; that drift is the experiment the gate measures.
//!
//! The *corpus-identity* fingerprint (codex v0.32.1 R2/R3 P2 — catches a
//! fixture edited in place with an unchanged id) is NOT computed here.  It is
//! computed at RUNTIME from the fixtures a gate actually loads
//! (`eval::gates::fixture_corpus_fingerprint`), because a build-time value is
//! blind to fixtures edited after the binary was built and then run via a
//! direct/installed `rein-eval` with no `cargo` rebuild.
//!
//! The hash is FNV-1a/128 — non-cryptographic, but collision risk for
//! change-detection is negligible and it needs no build-dependency.

use std::path::{Path, PathBuf};

/// FNV-1a 128-bit offset basis.
const FNV_OFFSET_128: u128 = 0x6c62272e07bb014262b821756295c58d;
/// FNV-1a 128-bit prime.
const FNV_PRIME_128: u128 = 0x0000000001000000000000000000013B;

fn fnv1a_128(state: &mut u128, bytes: &[u8]) {
    for &b in bytes {
        *state ^= b as u128;
        *state = state.wrapping_mul(FNV_PRIME_128);
    }
}

/// Recursively collect files under `root` with extension `ext`, returned as
/// (relative-path-string, absolute-path) — relative so the fingerprint is
/// independent of the builder's absolute checkout (also a privacy property:
/// no `$HOME` leaks into the embedded string).
fn collect(root: &Path, manifest: &Path, ext: &str, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, manifest, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            let rel = path
                .strip_prefix(manifest)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/"); // normalize across OSes
            out.push((rel, path));
        }
    }
}

/// Fold a sorted file list into the running FNV state, emitting a
/// `rerun-if-changed` for each file so any content edit re-runs the script.
fn hash_files(state: &mut u128, files: &[(String, PathBuf)]) {
    for (rel, abs) in files {
        fnv1a_128(state, rel.as_bytes()); // path → renames change the hash
        fnv1a_128(state, &[0]); // path/content separator
        if let Ok(bytes) = std::fs::read(abs) {
            fnv1a_128(state, &bytes);
        }
        fnv1a_128(state, &[0]); // file separator
        println!("cargo:rerun-if-changed={}", abs.display());
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    let src_root = manifest.join("src");
    let fixture_root = manifest.join("tests/fixtures/eval_gates");
    // Watch the dir roots so a *newly added* file (not yet in the per-file
    // watch list emitted by hash_files) still triggers a re-run.
    println!("cargo:rerun-if-changed={}", src_root.display());
    println!("cargo:rerun-if-changed={}", fixture_root.display());

    let mut src_files: Vec<(String, PathBuf)> = Vec::new();
    collect(&src_root, &manifest, "rs", &mut src_files);
    src_files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut fixture_files: Vec<(String, PathBuf)> = Vec::new();
    collect(&fixture_root, &manifest, "json", &mut fixture_files);
    fixture_files.sort_by(|a, b| a.0.cmp(&b.0));

    // Whole-build fingerprint (run freshness): src/ then fixtures.
    let mut build_state = FNV_OFFSET_128;
    hash_files(&mut build_state, &src_files);
    hash_files(&mut build_state, &fixture_files);

    println!("cargo:rustc-env=REIN_BUILD_FINGERPRINT={build_state:032x}");
    println!("cargo:rerun-if-changed=build.rs");
}
