//! Layer Rule — forbid inner layers from importing outer layers.
//!
//! The rule ranks layers from innermost (rank 0) to outermost and asserts
//! that every import resolves to a layer with rank `≤` the importing file's
//! rank. The file `src/domain/foo.rs` (layer `domain`, rank 0) may not
//! `use crate::adapters::…`. The reverse direction is always allowed.
//!
//! Imports are matched against one of four buckets:
//!   - `crate::<seg>::…` → resolve `<seg>` to a layer by probing candidate
//!     module files upward from the importing file's directory
//!     (`<dir>/<seg>.rs`, `<dir>/<seg>/mod.rs`, then ancestors), falling
//!     back to the legacy single-crate `src/<seg>.rs` probes, and
//!     consulting the layer globs for each candidate.
//!   - `std::` / `core::` / `alloc::` / `self::` / `super::` → ignored.
//!   - any other first segment → external crate; resolve via the provided
//!     exact map first, then the glob list. Unknown externals are ignored.
//!
//! Files matching `reexport_points` bypass the rule entirely (they are
//! typically at the composition root and may wire any layer to any other).
//! Files matching no layer are governed by `unmatched_behavior`:
//!   - `CompositionRoot` — treated as the outermost rank (can import
//!     anything).
//!   - `StrictError` — one `UnmatchedLayer` violation per such file.
//!
//! Phase-3 scope. `super::`-relative resolution and more refined
//! `crate::<seg>::<sub>` sub-segment layering are deferred.

#![cfg_attr(test, allow(dead_code))]

use crate::adapters::analyzers::architecture::{MatchLocation, ViolationKind};
use crate::adapters::shared::use_tree::gather_imports;
use globset::{GlobMatcher, GlobSet};
use std::collections::{HashMap, HashSet};

/// Behaviour when a file matches no layer glob.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnmatchedBehavior {
    /// Treat the file as the composition root (no rule applied).
    CompositionRoot,
    /// Emit one `UnmatchedLayer` violation for the file.
    StrictError,
}

/// Ordered layer definitions with rank lookup.
#[derive(Debug)]
pub struct LayerDefinitions {
    ranks: HashMap<String, usize>,
    definitions: Vec<(String, GlobSet)>,
}

impl LayerDefinitions {
    pub fn new(order: Vec<String>, definitions: Vec<(String, GlobSet)>) -> Self {
        let ranks = order
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        Self { ranks, definitions }
    }

    /// Rank assigned to `layer`. `None` for layers not in `order`.
    pub fn rank_of(&self, layer: &str) -> Option<usize> {
        self.ranks.get(layer).copied()
    }

    /// Layer assigned to the file at `path`, if any glob matches.
    pub fn layer_for_file(&self, path: &str) -> Option<&str> {
        self.definitions
            .iter()
            .find(|(_, gs)| gs.is_match(path))
            .map(|(name, _)| name.as_str())
    }

    /// Layer name + rank for the file at `path`. Returns `None` if no glob
    /// matches; the layer is guaranteed to be in the rank map because both
    /// lookups consult the same definitions.
    pub fn layer_and_rank_for_file(&self, path: &str) -> Option<(&str, usize)> {
        let layer = self.layer_for_file(path)?;
        let rank = self.rank_of(layer)?;
        Some((layer, rank))
    }

    /// Layer of `crate::<seg>` by probing `src/<seg>.rs` and
    /// `src/<seg>/mod.rs` against the layer globs.
    ///
    /// Legacy single-crate heuristic, kept for single-file diagnostics and
    /// single-crate projects. The workspace checker prefers
    /// [`LayerDefinitions::layer_for_known_crate_import`].
    pub fn layer_for_crate_segment(&self, seg: &str) -> Option<&str> {
        [format!("src/{seg}.rs"), format!("src/{seg}/mod.rs")]
            .iter()
            .find_map(|c| self.layer_for_file(c))
    }

    /// Probe one directory for `<seg>.rs` / `<seg>/mod.rs` present in the
    /// analyzed workspace. Operation: candidate filter + glob probe.
    fn probe_known_dir(
        &self,
        dir: &str,
        seg: &str,
        known_files: &std::collections::HashSet<String>,
    ) -> Option<&str> {
        if dir.is_empty() {
            return None;
        }
        [format!("{dir}/{seg}.rs"), format!("{dir}/{seg}/mod.rs")]
            .iter()
            .filter(|candidate| known_files.contains(*candidate))
            .find_map(|candidate| self.layer_for_file(candidate))
    }

    /// Existence-gated workspace resolution for `crate::<seg>` imported
    /// from the file at `importing_file` (Ornis fork): a candidate counts
    /// only when it names a file actually present in the analyzed workspace
    /// (`known_files`). Without the gate, glob-syntactic probes match phantom
    /// paths inside the importing file's own subtree (e.g.
    /// `src/domain/adapters.rs` matching a `src/domain/**` layer for a
    /// `crate::adapters` import that really targets `src/adapters/mod.rs`),
    /// hiding true inner→outer violations. Operation: ancestor walk of
    /// [`LayerDefinitions::probe_known_dir`], then existence-gated legacy
    /// probes.
    pub fn layer_for_known_crate_import(
        &self,
        importing_file: &str,
        seg: &str,
        known_files: &std::collections::HashSet<String>,
    ) -> Option<&str> {
        let dir = importing_file.rsplit_once('/').map_or("", |(d, _)| d);
        let mut cursor = Some(dir);
        while let Some(current) = cursor {
            if let Some(layer) = self.probe_known_dir(current, seg, known_files) {
                return Some(layer);
            }
            cursor = current.rsplit_once('/').map(|(parent, _)| parent);
        }
        self.probe_known_dir("src", seg, known_files)
    }

    /// Resolve a canonical call-target string (`crate::a::b::c`) to its
    /// layer. Non-crate prefixes (`<method>:`, `<bare>:`, `std::`, empty,
    /// etc.) yield `None` by design — only workspace-local targets carry
    /// layer meaning.
    ///
    /// For a path `crate::a::b::c`, every ancestor of the leaf is tried
    /// as a candidate file (`src/a/b/c.rs`, `src/a/b/c/mod.rs`,
    /// `src/a/b.rs`, …) in decreasing specificity. The first candidate
    /// that matches a layer's glob set wins — same greedy semantics
    /// as `layer_for_file`.
    /// Integration: delegates path splitting + per-candidate probe.
    pub fn layer_of_crate_path(&self, canonical: &str) -> Option<&str> {
        let inner = canonical.strip_prefix("crate::").or_else(|| {
            if canonical == "crate" {
                Some("")
            } else {
                None
            }
        })?;
        let segments: Vec<&str> = if inner.is_empty() {
            Vec::new()
        } else {
            inner.split("::").collect()
        };
        if segments.is_empty() {
            // Bare `crate` (module root) → treat as `src/lib.rs` / `src/main.rs`.
            return ["src/lib.rs", "src/main.rs"]
                .iter()
                .find_map(|c| self.layer_for_file(c));
        }
        for len in (1..=segments.len()).rev() {
            let head = &segments[..len];
            let joined = head.join("/");
            for candidate in [format!("src/{joined}.rs"), format!("src/{joined}/mod.rs")] {
                if let Some(layer) = self.layer_for_file(&candidate) {
                    return Some(layer);
                }
            }
        }
        // Single-segment path (`crate::run`) may target an item declared
        // directly in the crate root — probe `src/lib.rs` / `src/main.rs`
        // last, after the per-segment candidates. Ordering keeps
        // `crate::foo` preferring `src/foo.rs` when both exist.
        if segments.len() == 1 {
            return ["src/lib.rs", "src/main.rs"]
                .iter()
                .find_map(|c| self.layer_for_file(c));
        }
        None
    }
}

/// Input bundle for `check_layer_rule`.
pub struct LayerRuleInput<'a> {
    pub layers: &'a LayerDefinitions,
    pub reexport_points: &'a GlobSet,
    pub unmatched_behavior: UnmatchedBehavior,
    pub external_exact: &'a HashMap<String, String>,
    pub external_glob: &'a [(GlobMatcher, String)],
}

/// Classification of a file relative to the layer scheme.
enum FileClass<'a> {
    /// Re-export point or composition-root-treated unmatched file: rule skipped.
    Skip,
    /// Unmatched file under `StrictError`: emit one `UnmatchedLayer` violation.
    Unmatched,
    /// File is assigned to a layer; proceed to per-import check.
    Matched { layer: &'a str, rank: usize },
}

/// Context for per-file import checking.
struct FileInfo<'a> {
    path: &'a str,
    ast: &'a syn::File,
    layer: &'a str,
    rank: usize,
}

/// Layer to which an import path resolves.
enum ImportTarget<'a> {
    /// Resolved to a layer; `display_path` is the rendered import for reporting.
    Layer {
        layer: &'a str,
        display_path: String,
    },
    /// Resolution intentionally skipped (stdlib, self/super, unknown external).
    Ignore,
}

/// Check every file's imports against the layer ordering.
/// Integration: per-file iteration + flat-map of per-file hits.
pub fn check_layer_rule(
    files: &[(String, &syn::File)],
    input: &LayerRuleInput<'_>,
) -> Vec<MatchLocation> {
    let known: HashSet<String> = files.iter().map(|(p, _)| p.clone()).collect();
    files
        .iter()
        .flat_map(|(path, ast)| file_violations(path, ast, &known, input))
        .collect()
}

/// Collect every hit for one file by classifying it then walking imports.
/// Integration: match-dispatch delegation over classification.
fn file_violations(
    path: &str,
    ast: &syn::File,
    known: &HashSet<String>,
    input: &LayerRuleInput<'_>,
) -> Vec<MatchLocation> {
    match classify_file(path, input) {
        FileClass::Skip => Vec::new(),
        FileClass::Unmatched => vec![make_unmatched(path)],
        FileClass::Matched { layer, rank } => collect_file_violations(
            &FileInfo {
                path,
                ast,
                layer,
                rank,
            },
            known,
            input,
        ),
    }
}

/// Classify a file as Skip, Unmatched, or Matched(layer, rank).
/// Operation: classification logic over input data (no own calls).
fn classify_file<'a>(path: &str, input: &'a LayerRuleInput<'_>) -> FileClass<'a> {
    if input.reexport_points.is_match(path) {
        return FileClass::Skip;
    }
    let Some((layer, rank)) = input.layers.layer_and_rank_for_file(path) else {
        return match input.unmatched_behavior {
            UnmatchedBehavior::CompositionRoot => FileClass::Skip,
            UnmatchedBehavior::StrictError => FileClass::Unmatched,
        };
    };
    FileClass::Matched { layer, rank }
}

/// Build an `UnmatchedLayer` MatchLocation for `path`.
/// Operation: construction logic (no own calls).
fn make_unmatched(path: &str) -> MatchLocation {
    MatchLocation {
        file: path.to_string(),
        line: 1,
        column: 0,
        kind: ViolationKind::UnmatchedLayer {
            file: path.to_string(),
        },
    }
}

/// Walk a file's `use` items and flag imports that resolve to a higher-ranked
/// layer than the file's own.
/// Integration: orchestrates import gathering + evaluation through iterator chains.
fn collect_file_violations(
    file: &FileInfo<'_>,
    known: &HashSet<String>,
    input: &LayerRuleInput<'_>,
) -> Vec<MatchLocation> {
    gather_imports(file.ast)
        .into_iter()
        .filter_map(|(segments, span)| evaluate_import(file, &segments, span, known, input))
        .collect()
}

/// Resolve one import and return a violation hit if its target layer is outer.
/// Operation: resolution + rank comparison, no own calls.
fn evaluate_import(
    file: &FileInfo<'_>,
    segments: &[String],
    span: proc_macro2::Span,
    known: &HashSet<String>,
    input: &LayerRuleInput<'_>,
) -> Option<MatchLocation> {
    let ImportTarget::Layer {
        layer,
        display_path,
    } = resolve_target(file, segments, known, input)
    else {
        return None;
    };
    let to_rank = input.layers.rank_of(layer)?;
    if to_rank <= file.rank {
        return None;
    }
    let start = span.start();
    Some(MatchLocation {
        file: file.path.to_string(),
        line: start.line,
        column: start.column,
        kind: ViolationKind::LayerViolation {
            from_layer: file.layer.to_string(),
            to_layer: layer.to_string(),
            imported_path: display_path,
        },
    })
}

/// Decide which layer (if any) an import resolves to.
/// Integration: match-dispatch delegation over first segment.
fn resolve_target<'a>(
    file: &FileInfo<'a>,
    segments: &[String],
    known: &HashSet<String>,
    input: &'a LayerRuleInput<'_>,
) -> ImportTarget<'a> {
    let Some(first) = segments.first() else {
        return ImportTarget::Ignore;
    };
    match first.as_str() {
        "self" | "super" | "std" | "core" | "alloc" => ImportTarget::Ignore,
        "crate" => resolve_crate_target(file.path, segments, known, input),
        ext => resolve_external_target(ext, segments, input),
    }
}

/// Resolve a `crate::<seg>::…` import.
/// Operation: existence-gated workspace-aware lookup anchored at the
/// importing file, with a syntactic fallback when no candidate names an
/// analyzed file (single-file `--explain` runs, fixtures). The fallback
/// preserves upstream's guess semantics; the gate only sharpens hits when
/// the workspace is fully visible.
fn resolve_crate_target<'a>(
    importing_file: &str,
    segments: &[String],
    known: &HashSet<String>,
    input: &'a LayerRuleInput<'_>,
) -> ImportTarget<'a> {
    let Some(seg) = segments.get(1) else {
        return ImportTarget::Ignore;
    };
    let layer = input
        .layers
        .layer_for_known_crate_import(importing_file, seg, known)
        .or_else(|| input.layers.layer_for_crate_segment(seg));
    match layer {
        Some(layer) => ImportTarget::Layer {
            layer,
            display_path: segments.join("::"),
        },
        None => ImportTarget::Ignore,
    }
}

/// Resolve an external-crate import (exact map first, then glob list).
/// Operation: table lookup + glob probe.
fn resolve_external_target<'a>(
    crate_name: &str,
    segments: &[String],
    input: &'a LayerRuleInput<'_>,
) -> ImportTarget<'a> {
    if let Some(layer) = input.external_exact.get(crate_name) {
        return ImportTarget::Layer {
            layer: layer.as_str(),
            display_path: segments.join("::"),
        };
    }
    input
        .external_glob
        .iter()
        .find(|(m, _)| m.is_match(crate_name))
        .map(|(_, layer)| ImportTarget::Layer {
            layer: layer.as_str(),
            display_path: segments.join("::"),
        })
        .unwrap_or(ImportTarget::Ignore)
}
