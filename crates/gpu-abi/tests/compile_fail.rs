//! Compile-fail tests for the safe API's method sets (trybuild).
//!
//! Each `.rs` file in `tests/ui/` must fail with exactly the diagnostics
//! frozen in the matching `.stderr` snapshot. Run with `TRYBUILD=overwrite`
//! to regenerate snapshots after intentional changes.
//!
//! The snapshot mirrors the pinned toolchain's rendering down to the
//! gutter width and note lines — a `rust-toolchain.toml` bump reflows
//! them, so rerun the command above when the toolchain moves.

#[test]
fn uniform_tokens_have_no_write() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
