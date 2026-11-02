//! Compile-fail tests for the `#[shader::*]` entry-point attributes (trybuild).
//!
//! Each `.rs` file in `ui/` must fail to compile with exactly the diagnostics
//! frozen in the matching `.stderr` snapshot. Run with
//! `cargo test -p ornis-gpu-abi-macros --test compile_fail` (set
//! `TRYBUILD=overwrite` to regenerate snapshots after intentional changes).

#[test]
fn shader_entry_error_paths() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
