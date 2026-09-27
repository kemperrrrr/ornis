# rustqual — Ornis fork (vendored)

Upstream: https://github.com/SaschaOnTour/rustqual @ v1.8.2 (MIT, see
`LICENSE`). Unmodified except the fixes listed below; no 1.8.3 changes
ported (its `use`-as-exposure rework false-positives on facade-routed
items — assessed 2026-09-24, see project report).

Why vendored: two architecture-rule bugs block the Ornis quality gate
(`rustqual.toml`). Both fixed here with regression tests; fixes will be
offered upstream.

1. Same-crate `crate::x` layer resolution: upstream probes only
   `src/<x>.rs` (single-crate assumption), misclassifying every
   workspace-internal import. Fixed by ancestor-walk + existence gate
   (`layer_for_known_crate_import`), legacy probes kept as fallback.
2. `allowed_in` + `forbidden_in` on one pattern silently disabled the
   whole rule (documented as combinable). Fixed: `forbidden_in` is the
   scope, `allowed_in`/`except` are exemptions.

Build: `cargo build --manifest-path third_party/rustqual/Cargo.toml`
(standalone crate, own lockfile, excluded from the workspace). The gate
uses `./third_party/rustqual/target/debug/rustqual` (release for CI).
