//! Emission surface of the entity-gather plan (issue #110): WGSL
//! module-scope declarations, the CPU `Entity` mirror and gather
//! helper, the eval-body rewrite, and the thread-local bridge WGSL
//! lowering consults through [`helper_for`].

use std::cell::RefCell;
use std::collections::BTreeMap;

use ornis_shader_lang::writer::{entity_gather_decl, entity_struct_decl, lane_var_decl};
use syn::Expr;
use syn::visit_mut::{self, VisitMut};

use crate::wgsl_entity_shape::entity_field_shape;
use crate::wgsl_gather::GatherPlan;

/// CPU-side gather helper name, shared by the eval-body rewrite and
/// the helper the macro module emits.
pub(crate) const CPU_GATHER: &str = "_ornis_gather";

thread_local! {
    /// (binding, field) → WGSL helper name for the kernel currently
    /// being lowered; empty outside [`with_reads`].
    static READS: RefCell<BTreeMap<(String, String), String>> =
        const { RefCell::new(BTreeMap::new()) };
}

/// WGSL: bind the plan's gather sites for the duration of `f` so
/// [`helper_for`] can resolve `t.position` during lowering.
pub(crate) fn with_reads<T>(plan: &GatherPlan, f: impl FnOnce() -> T) -> T {
    let next = plan.helper_map();
    let prev = READS.with(|cell| std::mem::replace(&mut *cell.borrow_mut(), next));
    let out = f();
    READS.with(|cell| *cell.borrow_mut() = prev);
    out
}

/// WGSL helper name for a recorded gather site, `None` when the site
/// is not a gather (`t.index`, non-entity fields, any other kernel).
pub(crate) fn helper_for(binding: &str, member: &str) -> Option<String> {
    READS.with(|cell| {
        cell.borrow()
            .get(&(binding.to_string(), member.to_string()))
            .cloned()
    })
}

/// Module-scope WGSL declarations for one kernel: the `Entity` struct
/// (when mentioned), the generated `Option` structs, then one storage
/// lane and one gather helper per lane. `include_lanes` is false for
/// the legacy compute form, which has no lane binding slots (gathers
/// there fail loudly at naga, as documented in the design note).
pub(crate) fn decls_prefix(plan: &GatherPlan, option_decls: &str, include_lanes: bool) -> String {
    let mut parts: Vec<String> = Vec::new();
    if plan.has_entity() {
        parts.push(entity_struct_decl());
    }
    if !option_decls.is_empty() {
        parts.push(option_decls.to_string());
    }
    if include_lanes {
        for (i, lane) in plan.lanes().enumerate() {
            parts.push(lane_var_decl(i as u32, &lane.lane_ident, &lane.elem));
        }
        for lane in plan.lanes() {
            parts.push(entity_gather_decl(
                &lane.helper_ident,
                &lane.lane_ident,
                &lane.elem,
            ));
        }
    }
    parts.join("\n")
}

/// CPU mirror of the WGSL `Entity` struct, emitted into the kernel's
/// helper module when the kernel mentions `Entity`.
pub(crate) fn entity_struct_tokens() -> proc_macro2::TokenStream {
    quote::quote! {
        /// A shader-side entity handle: a lane slot index plus a
        /// generation stamp. The WGSL mirror (`struct Entity`) is
        /// declared in `wgsl_source()`; `generation` is identity data
        /// for CPU-side staleness checks — GPU gather indexes lanes by
        /// `index` only.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
        pub struct Entity {
            /// Slot in the caller's SoA lanes.
            pub index: u32,
            /// Generation stamp for CPU-side staleness checks.
            pub generation: u32,
        }
    }
}

/// CPU gather helper: slice lookup with the same bounds condition the
/// WGSL helper spells (`index >= len` yields `T::default()`, the
/// element zero WGSL returns).
pub(crate) fn cpu_gather_tokens() -> proc_macro2::TokenStream {
    let helper = syn::Ident::new(CPU_GATHER, proc_macro2::Span::call_site());
    quote::quote! {
        #[allow(dead_code)]
        fn #helper<T: Copy + Default>(lane: &[T], e: Entity) -> T {
            let i = e.index as usize;
            if i >= lane.len() {
                T::default()
            } else {
                lane[i]
            }
        }
    }
}

/// Rewrite the CPU `eval` body: every recorded gather site
/// `t.position` becomes `_ornis_gather(_ornis_lane_t__position, t)`,
/// reading the lane slice parameter the macro appends to `eval`.
pub(crate) fn rewrite_eval_body(block: &syn::Block, plan: &GatherPlan) -> syn::Block {
    let mut block = block.clone();
    Rewriter { plan }.visit_block_mut(&mut block);
    block
}

struct Rewriter<'a> {
    plan: &'a GatherPlan,
}

impl VisitMut for Rewriter<'_> {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if let Expr::Field(field) = expr
            && let Some((root, member)) = entity_field_shape(field)
            && let Some(lane) = self.plan.reads.get(&(root.clone(), member.clone()))
        {
            let helper = syn::Ident::new(CPU_GATHER, proc_macro2::Span::call_site());
            let lane_ident = syn::Ident::new(&lane.lane_ident, proc_macro2::Span::call_site());
            let binding = syn::Ident::new(&lane.binding, proc_macro2::Span::call_site());
            *expr = syn::parse_quote! { #helper(#lane_ident, #binding) };
            return;
        }
        visit_mut::visit_expr_mut(self, expr);
    }
}
