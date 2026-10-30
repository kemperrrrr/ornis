//! Compile-fail tests for the safe API's method sets (trybuild).
//!
//! Each `.rs` file in `tests/ui/` must fail with exactly the diagnostics
//! frozen in the matching `.stderr` snapshot. Run with `TRYBUILD=overwrite`
//! to regenerate snapshots after intentional changes.

#[test]
fn uniform_tokens_have_no_write() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
