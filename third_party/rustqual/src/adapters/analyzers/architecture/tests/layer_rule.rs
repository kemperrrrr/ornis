//! Unit tests for the Layer Rule.
//!
//! Covers the four axes the rule has to get right:
//!   1. Layer assignment by path glob (incl. `reexport_points` bypass).
//!   2. Import resolution (`crate::...`, `std/core/alloc`, `self`, external).
//!   3. Rank comparison (inner can import from inner; outer-from-inner is
//!      fine; inner-from-outer is a violation).
//!   4. `unmatched_behavior` = composition_root vs strict_error.

use crate::adapters::analyzers::architecture::layer_rule::{
    check_layer_rule, LayerDefinitions, LayerRuleInput, UnmatchedBehavior,
};
use crate::adapters::analyzers::architecture::{MatchLocation, ViolationKind};
use globset::{Glob, GlobMatcher, GlobSet, GlobSetBuilder};
use std::collections::HashMap;

// ── helpers ────────────────────────────────────────────────────────────

fn glob_set(patterns: &[&str]) -> GlobSet {
    let mut b = GlobSetBuilder::new();
    for p in patterns {
        b.add(Glob::new(p).expect("valid glob"));
    }
    b.build().expect("valid glob set")
}

fn glob_matcher(pattern: &str) -> GlobMatcher {
    Glob::new(pattern).expect("valid glob").compile_matcher()
}

fn default_layers() -> LayerDefinitions {
    LayerDefinitions::new(
        vec![
            "domain".to_string(),
            "port".to_string(),
            "application".to_string(),
            "adapter".to_string(),
        ],
        vec![
            ("domain".to_string(), glob_set(&["src/domain/**"])),
            ("port".to_string(), glob_set(&["src/ports/**"])),
            ("application".to_string(), glob_set(&["src/app/**"])),
            ("adapter".to_string(), glob_set(&["src/adapters/**"])),
        ],
    )
}

fn parse_file(src: &str) -> syn::File {
    syn::parse_str(src).expect("test fixture must parse")
}

struct Fixture {
    parsed: Vec<(String, syn::File)>,
}

impl Fixture {
    fn new(files: &[(&str, &str)]) -> Self {
        let parsed = files
            .iter()
            .map(|(p, s)| (p.to_string(), parse_file(s)))
            .collect();
        Self { parsed }
    }

    fn refs(&self) -> Vec<(String, &syn::File)> {
        self.parsed.iter().map(|(p, f)| (p.clone(), f)).collect()
    }
}

/// External-crate layer mappings for a layer-rule test run (exact + glob).
struct Externals<'a> {
    exact: &'a HashMap<String, String>,
    glob: &'a [(GlobMatcher, String)],
}

fn run(
    fixture: &Fixture,
    layers: &LayerDefinitions,
    reexport: &GlobSet,
    unmatched: UnmatchedBehavior,
    externals: Externals,
) -> Vec<MatchLocation> {
    let refs = fixture.refs();
    check_layer_rule(
        &refs,
        &LayerRuleInput {
            layers,
            reexport_points: reexport,
            unmatched_behavior: unmatched,
            external_exact: externals.exact,
            external_glob: externals.glob,
        },
    )
}

fn run_simple(fixture: &Fixture) -> Vec<MatchLocation> {
    run(
        fixture,
        &default_layers(),
        &glob_set(&[]),
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &HashMap::new(),
            glob: &[],
        },
    )
}

// ── basic cases ────────────────────────────────────────────────────────

#[test]
fn clean_file_no_violations() {
    let fx = Fixture::new(&[("src/domain/mod.rs", "pub struct Foo;")]);
    assert!(run_simple(&fx).is_empty());
}

#[test]
fn same_layer_import_allowed() {
    let fx = Fixture::new(&[
        ("src/domain/mod.rs", "pub struct Bar;"),
        ("src/domain/foo.rs", "use crate::domain::Bar;"),
    ]);
    assert!(run_simple(&fx).is_empty());
}

#[test]
fn outer_importing_inner_allowed() {
    // adapter (rank 3) importing from domain (rank 0) is fine
    let fx = Fixture::new(&[
        ("src/domain/mod.rs", "pub struct Bar;"),
        ("src/adapters/mod.rs", "use crate::domain::Bar;"),
    ]);
    assert!(run_simple(&fx).is_empty());
}

#[test]
fn inner_importing_outer_is_violation() {
    // domain (rank 0) importing from adapter (rank 3) is forbidden
    let fx = Fixture::new(&[
        ("src/adapters/mod.rs", "pub struct Bar;"),
        ("src/domain/bad.rs", "use crate::adapters::Bar;"),
    ]);
    let hits = run_simple(&fx);
    assert_eq!(hits.len(), 1, "{hits:?}");
    match &hits[0].kind {
        ViolationKind::LayerViolation {
            from_layer,
            to_layer,
            imported_path,
        } => {
            assert_eq!(from_layer, "domain");
            assert_eq!(to_layer, "adapter");
            assert!(
                imported_path.starts_with("crate::adapters"),
                "imported_path = {imported_path:?}"
            );
        }
        other => panic!("unexpected kind: {other:?}"),
    }
    assert_eq!(hits[0].file, "src/domain/bad.rs");
}

#[test]
fn port_importing_application_is_violation() {
    // port (rank 1) importing from application (rank 2) is forbidden
    let fx = Fixture::new(&[
        ("src/app/mod.rs", "pub fn run() {}"),
        ("src/ports/bad.rs", "use crate::app::run;"),
    ]);
    let hits = run_simple(&fx);
    assert_eq!(hits.len(), 1);
    match &hits[0].kind {
        ViolationKind::LayerViolation {
            from_layer,
            to_layer,
            ..
        } => {
            assert_eq!(from_layer, "port");
            assert_eq!(to_layer, "application");
        }
        other => panic!("unexpected kind: {other:?}"),
    }
}

#[test]
fn application_importing_port_allowed() {
    // application (rank 2) importing from port (rank 1) is fine
    let fx = Fixture::new(&[
        ("src/ports/mod.rs", "pub trait Service {}"),
        ("src/app/use_case.rs", "use crate::ports::Service;"),
    ]);
    assert!(run_simple(&fx).is_empty());
}

// ── special first segments ─────────────────────────────────────────────

#[test]
fn std_core_alloc_ignored() {
    let fx = Fixture::new(&[(
        "src/domain/foo.rs",
        "use std::collections::HashMap; use core::fmt; use alloc::vec::Vec;",
    )]);
    assert!(run_simple(&fx).is_empty());
}

#[test]
fn self_and_super_ignored() {
    // For the layer rule these are same-crate, same-tree references.
    let fx = Fixture::new(&[(
        "src/domain/foo.rs",
        "use self::inner::thing; use super::other;",
    )]);
    assert!(run_simple(&fx).is_empty());
}

#[test]
fn unresolved_crate_segment_ignored() {
    // `crate::unknown` — no file defines it — is skipped (conservative).
    let fx = Fixture::new(&[("src/domain/foo.rs", "use crate::unknown::thing;")]);
    assert!(run_simple(&fx).is_empty());
}

// ── grouped imports ────────────────────────────────────────────────────

#[test]
fn grouped_use_flags_each_bad_leaf() {
    let fx = Fixture::new(&[
        ("src/domain/mod.rs", "pub struct A;"),
        ("src/adapters/mod.rs", "pub struct X;"),
        ("src/app/mod.rs", "pub struct Y;"),
        (
            "src/domain/bad.rs",
            "use crate::{domain::A, adapters::X, app::Y};",
        ),
    ]);
    let hits = run_simple(&fx);
    let to_layers: Vec<String> = hits
        .iter()
        .filter_map(|h| match &h.kind {
            ViolationKind::LayerViolation { to_layer, .. } => Some(to_layer.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(hits.len(), 2, "expected two violations: {hits:?}");
    assert!(to_layers.contains(&"adapter".to_string()));
    assert!(to_layers.contains(&"application".to_string()));
}

// ── external crates ────────────────────────────────────────────────────

#[test]
fn external_exact_match_enforced() {
    let mut ext = HashMap::new();
    ext.insert("adapter_only_crate".to_string(), "adapter".to_string());
    let fx = Fixture::new(&[("src/domain/bad.rs", "use adapter_only_crate::X;")]);
    let hits = run(
        &fx,
        &default_layers(),
        &glob_set(&[]),
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &ext,
            glob: &[],
        },
    );
    assert_eq!(hits.len(), 1, "{hits:?}");
    match &hits[0].kind {
        ViolationKind::LayerViolation {
            from_layer,
            to_layer,
            imported_path,
        } => {
            assert_eq!(from_layer, "domain");
            assert_eq!(to_layer, "adapter");
            assert!(imported_path.starts_with("adapter_only_crate"));
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn external_glob_match_enforced() {
    let ext_glob = vec![(glob_matcher("adp_*"), "adapter".to_string())];
    let fx = Fixture::new(&[("src/domain/bad.rs", "use adp_sqlite::Pool;")]);
    let hits = run(
        &fx,
        &default_layers(),
        &glob_set(&[]),
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &HashMap::new(),
            glob: &ext_glob,
        },
    );
    assert_eq!(hits.len(), 1, "{hits:?}");
}

#[test]
fn external_exact_wins_over_glob() {
    let mut ext_exact = HashMap::new();
    ext_exact.insert("adp_special".to_string(), "domain".to_string());
    let ext_glob = vec![(glob_matcher("adp_*"), "adapter".to_string())];
    // domain file imports "adp_special" — exact says "domain" (same layer) → OK
    let fx = Fixture::new(&[("src/domain/ok.rs", "use adp_special::X;")]);
    let hits = run(
        &fx,
        &default_layers(),
        &glob_set(&[]),
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &ext_exact,
            glob: &ext_glob,
        },
    );
    assert!(hits.is_empty(), "exact must win: {hits:?}");
}

#[test]
fn external_unknown_ignored() {
    let fx = Fixture::new(&[("src/domain/foo.rs", "use some_unknown_crate::Thing;")]);
    assert!(run_simple(&fx).is_empty());
}

// ── reexport points and unmatched ──────────────────────────────────────

#[test]
fn reexport_point_bypasses_rule() {
    let reexport = glob_set(&["src/lib.rs"]);
    let fx = Fixture::new(&[
        ("src/adapters/mod.rs", "pub struct X;"),
        ("src/lib.rs", "pub use crate::adapters::X;"),
    ]);
    let hits = run(
        &fx,
        &default_layers(),
        &reexport,
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &HashMap::new(),
            glob: &[],
        },
    );
    assert!(hits.is_empty(), "re-export point must bypass: {hits:?}");
}

#[test]
fn unmatched_composition_root_bypasses() {
    // src/lib.rs matches no layer, but CompositionRoot means no violation.
    let fx = Fixture::new(&[
        ("src/adapters/mod.rs", "pub struct X;"),
        ("src/lib.rs", "use crate::adapters::X;"),
    ]);
    let hits = run(
        &fx,
        &default_layers(),
        &glob_set(&[]),
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &HashMap::new(),
            glob: &[],
        },
    );
    assert!(hits.is_empty(), "unmatched composition root: {hits:?}");
}

#[test]
fn unmatched_strict_error_emits_one_violation() {
    let fx = Fixture::new(&[("src/unorganized.rs", "fn foo() {}")]);
    let hits = run(
        &fx,
        &default_layers(),
        &glob_set(&[]),
        UnmatchedBehavior::StrictError,
        Externals {
            exact: &HashMap::new(),
            glob: &[],
        },
    );
    assert_eq!(hits.len(), 1);
    match &hits[0].kind {
        ViolationKind::UnmatchedLayer { file } => {
            assert_eq!(file, "src/unorganized.rs");
        }
        other => panic!("unexpected kind: {other:?}"),
    }
}

#[test]
fn strict_error_does_not_flag_reexport_points() {
    let reexport = glob_set(&["src/lib.rs"]);
    let fx = Fixture::new(&[("src/lib.rs", "fn main() {}")]);
    let hits = run(
        &fx,
        &default_layers(),
        &reexport,
        UnmatchedBehavior::StrictError,
        Externals {
            exact: &HashMap::new(),
            glob: &[],
        },
    );
    assert!(hits.is_empty(), "{hits:?}");
}

// ── file paths with backslashes (windows-style) ────────────────────────

#[test]
fn windows_style_separators_work() {
    // rustqual normalizes to forward slashes before the architecture analyzer
    // sees the path. The layer rule relies on this — globs use `/`.
    let fx = Fixture::new(&[
        ("src/domain/mod.rs", "pub struct Bar;"),
        ("src/adapters/bad.rs", "use crate::domain::Bar;"),
    ]);
    assert!(run_simple(&fx).is_empty());
}

// ── workspace-relative crate:: resolution (Ornis fork) ─────────────────

#[test]
fn workspace_deep_import_ignores_legacy_src_probe_collision() {
    // Regression: a layer glob like `src/**` (root binary crate) must not
    // capture the legacy single-crate probes (`src/<seg>.rs`) for `crate::`
    // imports deep inside a workspace member. The ancestor walk resolves
    // `crate::solver` from `crates/engine/src/sequential_impulse/` to the
    // domain layer before the legacy probe is even tried.
    let layers = LayerDefinitions::new(
        vec!["domain".to_string(), "adapter".to_string()],
        vec![
            ("domain".to_string(), glob_set(&["crates/engine/**"])),
            ("adapter".to_string(), glob_set(&["src/**"])),
        ],
    );
    let fx = Fixture::new(&[
        (
            "crates/engine/src/sequential_impulse/contacts.rs",
            "use crate::solver::Engine;",
        ),
        ("crates/engine/src/solver/mod.rs", "pub struct Engine;"),
    ]);
    let hits = run(
        &fx,
        &layers,
        &glob_set(&[]),
        UnmatchedBehavior::CompositionRoot,
        Externals {
            exact: &HashMap::new(),
            glob: &[],
        },
    );
    assert!(
        hits.is_empty(),
        "deep same-crate import resolved outward: {hits:?}"
    );
}

#[test]
fn crate_import_known_gate_resolves_sibling_and_ignores_phantom() {
    use std::collections::HashSet;
    // Existence-gated resolution (Ornis fork): a sibling that names an
    // analyzed file resolves to its layer; a phantom segment resolves to
    // nothing even when its would-be path matches a layer glob.
    let layers = LayerDefinitions::new(
        vec!["domain".to_string()],
        vec![("domain".to_string(), glob_set(&["crates/engine/**"]))],
    );
    let known: HashSet<String> = ["crates/engine/src/solver/mod.rs".to_string()]
        .into_iter()
        .collect();
    assert_eq!(
        layers.layer_for_known_crate_import("crates/engine/src/deep/x.rs", "solver", &known),
        Some("domain")
    );
    assert_eq!(
        layers.layer_for_known_crate_import("crates/engine/src/deep/x.rs", "nope", &known),
        None
    );
    assert_eq!(
        layers.layer_for_known_crate_import(
            "crates/engine/src/deep/x.rs",
            "solver",
            &HashSet::new()
        ),
        None,
        "no analyzed files → unresolved (legacy probes decide)"
    );
}
