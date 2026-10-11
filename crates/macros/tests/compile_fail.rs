//! Compile-fail tests for proc-macro error paths (trybuild).
//!
//! Each `.rs` file in ui/ must fail to compile with exactly the diagnostics
//! frozen in the matching `.stderr` snapshot. Run with:
//! `cargo test -p ornis-macros --test compile_fail`; `tests/ui/pass/*.rs` are
//! positive cases that must compile and run (set
//! TRYBUILD=overwrite to regenerate snapshots after intentional changes).

#[test]
fn macro_error_paths() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
    t.pass("tests/ui/pass/*.rs");
}
