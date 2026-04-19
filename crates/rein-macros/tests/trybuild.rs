//! trybuild harness for `#[op]` macro compile-time tests.
//!
//! Pass tests (`pass_*.rs`) must compile cleanly.
//! Fail tests (`fail_*.rs`) must fail with the expected diagnostic in `*.stderr`.
//!
//! To regenerate stderr files after intentional macro error message changes:
//!     TRYBUILD=overwrite cargo test -p rein-macros --test trybuild

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    // Pass tests live in the rein crate's integration tests (see
    // `crates/rein/tests/inventory_registration.rs`) since the macro emits
    // `::rein::ops::*` paths that can't resolve in trybuild scratch crates.
    t.compile_fail("tests/ui/fail_*.rs");
}
