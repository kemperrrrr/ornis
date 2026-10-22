//! `Entity` gather planning for `#[kernel]` (issue #110, feature 1).
//!
//! Attribute macros cannot see component struct definitions, so the
//! type of a foreign field cannot be derived from the field itself —
//! the use site annotates it (`let p: Vec3 = t.position;`) and this
//! module family turns each such read into a plan entry: one SoA
//! storage lane per (entity binding, field), one WGSL gather helper
//! per lane, and a CPU-body rewrite that reads the same lane as a
//! `&[T]` slice.
//!
//! Three consumers share one plan:
//! - the scanner ([`scan`]) validates read positions, rejects writes
//!   and entity equality, and records lanes;
//! - WGSL lowering consults [`helper_for`] inside [`with_reads`];
//! - [`rewrite_eval_body`] rewrites the CPU `eval` body to slice
//!   gathers with exactly the bounds check the WGSL helper spells.
//!
//! `t.index` / `t.generation` are direct struct fields on both sides
//! (`Entity` exists in WGSL and in the emitted CPU mirror) and are
//! never gathers. Everything else on an entity binding is a gather and
//! must sit in an allowed, annotated position — anywhere else is a
//! spanned compile error, never a silent translation.
//!
//! The implementation is split by responsibility (file length):
//! this file holds the plan types, [`crate::wgsl_gather_scan`] walks
//! the kernel, [`crate::wgsl_gather_rejects`] reports the spanned
//! errors, and [`crate::wgsl_gather_emit`] emits WGSL/CPU surface.

use std::collections::{BTreeMap, BTreeSet};

use ornis_shader_lang::ir::IrType;
use syn::Type;

pub(crate) use crate::wgsl_gather_emit::{
    cpu_gather_tokens, decls_prefix, entity_struct_tokens, helper_for, rewrite_eval_body,
    with_reads,
};
pub(crate) use crate::wgsl_gather_scan::scan;

/// One gather lane: a foreign field read from one entity binding.
pub(crate) struct Lane {
    pub(crate) binding: String,
    /// Lane element IR type — the annotation's lowered form.
    pub(crate) elem: IrType,
    /// Lane element Rust spelling, for the `eval` slice parameter.
    pub(crate) elem_ty: Type,
    pub(crate) lane_ident: String,
    pub(crate) helper_ident: String,
}

/// Everything lowering and macro emission need to know about `Entity`
/// in one kernel.
#[derive(Default)]
pub(crate) struct GatherPlan {
    /// Entity-typed bindings: params, annotated lets, and `Some`
    /// payloads of entity-typed scrutinees — over-approximated, which
    /// is sound because every recorded read is separately typed by its
    /// own annotation.
    pub(crate) entity_names: BTreeSet<String>,
    /// (binding, field) → lane, keyed in sorted order so lane binding
    /// numbers are deterministic.
    pub(crate) reads: BTreeMap<(String, String), Lane>,
    /// Whether the kernel mentions `Entity` at all — gates the
    /// per-module CPU/WGSL struct emission.
    pub(crate) mentions_entity: bool,
    /// Spanned scan errors; non-empty aborts before lowering.
    pub(crate) errors: Vec<syn::Error>,
}

impl GatherPlan {
    /// Lanes in deterministic (binding, field) order.
    pub(crate) fn lanes(&self) -> impl Iterator<Item = &Lane> {
        self.reads.values()
    }

    /// Whether the generated module must declare the `Entity` struct
    /// (CPU mirror and WGSL decl).
    pub(crate) fn has_entity(&self) -> bool {
        // Lanes always imply `mentions_entity` (`record_read` sets it);
        // reading both fields here keeps them in one LCOM cluster.
        self.mentions_entity || self.has_lanes()
    }

    /// Whether the kernel has any gather at all (CPU helper + lanes).
    pub(crate) fn has_lanes(&self) -> bool {
        !self.reads.is_empty()
    }

    pub(crate) fn helper_map(&self) -> BTreeMap<(String, String), String> {
        self.reads
            .iter()
            .map(|(key, lane)| (key.clone(), lane.helper_ident.clone()))
            .collect()
    }
}
