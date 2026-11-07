//! Compile-fail tests for the typed SDK's guarantees (trybuild).
//!
//! Each `.rs` file in `tests/ui/` must fail with exactly the diagnostics frozen
//! in the matching `.stderr` snapshot. Run with `TRYBUILD=overwrite` to
//! regenerate snapshots after an intentional change.
//!
//! These are the claims the typed layer makes instead of the raw imports: a
//! read-only binding has no `store`, a uniform field read is checked against
//! the struct's declared tags, and a binding's declared kind must match its
//! type. Without them "typed" would be spelling only.
//!
//! `tests/ui-pass/` holds one accepted kernel: trybuild switches from
//! `cargo check` to `cargo build` when a passing case exists, and the tag check
//! is a `const` evaluation that `cargo check` does not run.

#[test]
fn typed_bindings_reject_mismatched_use() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui-pass/*.rs");
    tests.compile_fail("tests/ui/*.rs");
}
