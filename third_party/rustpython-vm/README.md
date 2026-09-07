# rustpython-vm 0.5.0 (vendored, one upstream fix)

This is an unmodified copy of the `rustpython-vm` crate as published on
crates.io as **0.5.0** (source: RustPython tag `0.5.0`, commit
`39a6486a4cc32231943e14be8a435a68ae35aab0`), with **a single upstream fix**
applied. It is wired in via `[patch.crates-io]` in the workspace root
`Cargo.toml`.

## Why it exists

The quality gate pins every dependency to the latest release
(`cargo outdated --workspace --exit-code 1` is a hard stage). Since
**libc 0.2.187** the `POSIX_SPAWN_SETSID` constant on linux-gnu is
`c_short` (i16, matching glibc), while `nix` 0.30's `PosixSpawnFlags` is a
`c_int` bitflags type. `rustpython-vm 0.5.0` — the newest published
release (2026-03-31) — passes the constant straight into
`PosixSpawnFlags::from_bits_retain`, so the crate **no longer compiles**
against current libc/nix:

```text
error[E0308]: mismatched types
  --> rustpython-vm-0.5.0/src/stdlib/posix.rs:1812:84
   | expected `i32`, found `i16`
```

There is no patched release to upgrade to (0.5.0 is `max_version`), and
the in-repo fix on `main` rides on a large unpublished refactor
(`rustpython-host-env`), so it cannot be consumed as a version-matched
dependency. Vendoring with the one-line upstream fix is the minimal way
to keep both the gate's "everything latest" policy and a green build.

## The fix

Exactly the change from upstream PR
[RustPython#8343](https://github.com/RustPython/RustPython/pull/8343)
(commit `ea798265`, "Fix building against new libc"), applied to
`src/stdlib/posix.rs`:

```rust
#[allow(clippy::useless_conversion)]
flags.insert(nix::spawn::PosixSpawnFlags::from_bits_retain(
    libc::POSIX_SPAWN_SETSID.into(),
));
```

`diff -r` against the tag shows this hunk as the only source change.
`build.rs` and the frozen `Lib/` python modules are byte-identical
(symlinks from the repo checkout are dereferenced into real files, as in
the published `.crate` tarball).

## Manifest

`Cargo.toml` is the standalone (de-workspaced) form of the upstream
manifest at that tag: `workspace = true` keys are resolved to the values
from the RustPython workspace manifest, sibling `rustpython-*` crates
point at their published registry versions, and requirements, features
and target tables mirror the published crate 1:1 (checked against the
crates.io dependency listing for `rustpython-vm` 0.5.0). The `[lints]`
table is dropped — it referenced the upstream workspace and has no effect
on a standalone crate.

## Maintenance

When a new `rustpython-vm` is published that compiles against current
libc (or the project moves to `rustpython-stdlib`/`host_env` layout),
delete this directory and the `[patch.crates-io]` entry from the root
`Cargo.toml`, then let cargo re-resolve `rustpython-vm` from the registry.
This directory is excluded from the rustqual ratchet (`rustqual.toml`)
because it is third-party code, not Ornis source.
