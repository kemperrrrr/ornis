//! Lane set and per-run loop of `#[smart_system]` (IDEAS §32, v2b).
//!
//! Every field the body touches becomes one [`LaneSpec`]. The generated
//! `System::run` takes each lane guard once, in canonical (field-name)
//! order, then walks the selected entities and per entity:
//!
//! 1. resolves each own lane's dense index (hinted, see
//!    [`ornis_core::ComponentStore::index_hinted`]);
//! 2. gathers the touched fields by value ([`ornis_core::PackLane::gather`]);
//! 3. calls the body with `&T` / `&mut T` per field, plus the foreign lane
//!    stores for `t.field` reads;
//! 4. scatters written fields back.
//!
//! Entities are those owning every touched required lane. They come from the
//! aligned fast path when every own lane lists the same entities in ascending
//! id order, otherwise from [`ornis_core::PackMembership`]. Both visit
//! entities in entity-id order.

use std::collections::BTreeSet;

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{Ident, Type};

use crate::smart_system_fields::{FieldUsage, lane_type};
use crate::smart_system_step::{StepCtx, step_stmts};

/// One lane of the system: a field of the bundle touched by the body.
pub(crate) struct LaneSpec {
    pub(crate) field: String,
    pub(crate) lane_ty: Type,
    /// The body reads or writes `u.field`.
    pub(crate) own: bool,
    /// The body writes `u.field`.
    pub(crate) written: bool,
    /// The body reads `t.field` of a linked entity.
    pub(crate) foreign: bool,
}

impl LaneSpec {
    pub(crate) fn guard(&self) -> Ident {
        format_ident!("__smart_lane_{}", self.field, span = Span::mixed_site())
    }

    pub(crate) fn value(&self) -> Ident {
        format_ident!("__smart_v_{}", self.field, span = Span::mixed_site())
    }

    pub(crate) fn index(&self) -> Ident {
        format_ident!("__smart_ix_{}", self.field, span = Span::mixed_site())
    }

    /// Body parameter carrying the own field (`&T` or `&mut T`).
    pub(crate) fn u_param(&self) -> Ident {
        format_ident!("__smart_u_{}", self.field, span = Span::mixed_site())
    }
}

/// Lanes of the system in canonical (field-name) order.
pub(crate) fn build_lane_specs(usage: &FieldUsage, pack_path: &syn::Path) -> Vec<LaneSpec> {
    let fields: BTreeSet<&String> = usage.own.iter().chain(usage.foreign.iter()).collect();
    fields
        .into_iter()
        .map(|field| LaneSpec {
            field: field.clone(),
            lane_ty: lane_type(pack_path, field),
            own: usage.own.contains(field),
            written: usage.written.contains(field),
            foreign: usage.foreign.contains(field),
        })
        .collect()
}

/// Declares every lane: `writes_lane` for written fields, `reads_lane`
/// otherwise (foreign-only reads included).
pub(crate) fn lane_access_stmts(lanes: &[LaneSpec]) -> Vec<TokenStream2> {
    lanes
        .iter()
        .map(|lane| {
            let ty = &lane.lane_ty;
            if lane.written {
                quote! { __access = __access.writes_lane::<#ty>(); }
            } else {
                quote! { __access = __access.reads_lane::<#ty>(); }
            }
        })
        .collect()
}

/// Takes one guard per lane, in canonical order. A missing lane means no
/// entity can own every field, so the run is a no-op.
pub(crate) fn guard_stmts(lanes: &[LaneSpec], store: &Ident) -> Vec<TokenStream2> {
    lanes
        .iter()
        .map(|lane| {
            let ty = &lane.lane_ty;
            let guard = lane.guard();
            if lane.written {
                quote! {
                    let Some(mut #guard) = #store.write_lane::<#ty>() else { return; };
                }
            } else {
                quote! {
                    let Some(#guard) = #store.read_lane::<#ty>() else { return; };
                }
            }
        })
        .collect()
}

/// Body parameters for the lanes: `u_param` per own field, then the foreign
/// lane store per `t.field` lane. Order matches [`body_arg_stmts`].
pub(crate) fn body_lane_params(lanes: &[LaneSpec]) -> Vec<TokenStream2> {
    let mut params = Vec::new();
    for lane in lanes {
        let ty = &lane.lane_ty;
        if lane.own {
            let name = lane.u_param();
            if lane.written {
                params.push(quote! { #name: &mut <#ty as ornis_core::PackLane>::Field });
            } else {
                params.push(quote! { #name: &<#ty as ornis_core::PackLane>::Field });
            }
        }
        if lane.foreign {
            let name = lane.guard();
            params.push(quote! { #name: &ornis_core::ComponentStore<#ty> });
        }
    }
    params
}

/// Compile-time check that some touched own field is required (not
/// `Option`). Only required lanes select entities; sparse lanes list
/// entries that exist, so a body with only optional fields would silently
/// skip entities that have no value.
fn driver_assert(lanes: &[LaneSpec]) -> TokenStream2 {
    let required = lanes.iter().filter(|lane| lane.own).map(|lane| {
        let ty = &lane.lane_ty;
        quote! { || !<#ty as ornis_core::PackLane>::SPARSE }
    });
    quote! {
        const _: () = assert!(
            false #(#required)*,
            "smart_system: touch at least one non-`Option` field of the bundle — optional fields alone do not select entities"
        );
    }
}

/// Loop variables of a whole-run loop.
pub(crate) struct LoopIdents<'a> {
    pub(crate) entity: &'a Ident,
    pub(crate) hint: &'a Ident,
    /// Call of the const-generic aligned dispatch (`Zip` vs `Gather`).
    pub(crate) aligned_call: TokenStream2,
}

/// Whole-run loop: the aligned fast path (dense lane walk) or the general
/// membership walk. Both visit the same entities in the same order.
pub(crate) fn run_loop(
    lanes: &[LaneSpec],
    body_fn: &Ident,
    user_args: &[TokenStream2],
    ids: &LoopIdents,
) -> TokenStream2 {
    let LoopIdents {
        entity,
        hint,
        aligned_call,
        ..
    } = ids;
    let own: Vec<&LaneSpec> = lanes.iter().filter(|lane| lane.own).collect();
    let first = own[0];
    let first_guard = first.guard();
    let aligned_terms = own[1..].iter().map(|lane| {
        let guard = lane.guard();
        quote! { && #guard.entities == #first_guard.entities }
    });
    let mutable_sparse_terms = own.iter().filter(|lane| lane.written).map(|lane| {
        let ty = &lane.lane_ty;
        quote! { && !<#ty as ornis_core::PackLane>::SPARSE }
    });
    let membership_terms = own.iter().map(|lane| {
        let guard = lane.guard();
        quote! { __smart_membership.narrow(&*#guard); }
    });
    let step_ctx = StepCtx {
        body_fn,
        user_args,
        entity,
        hint,
    };
    let general_step = step_stmts(lanes, &step_ctx, false);
    let assert = driver_assert(lanes);
    let general_branch = quote! {
        let mut __smart_membership = ornis_core::PackMembership::new();
        #(#membership_terms)*
        let __smart_entities = __smart_membership.into_entities();
        for (__smart_i, #entity) in __smart_entities.into_iter().enumerate() {
            let #hint = __smart_i;
            #general_step
        }
    };
    let layout_reads = own.iter().map(|lane| {
        let guard = lane.guard();
        quote! { #guard.layout() }
    });
    quote! {
        #assert
        let __smart_layouts = [#(#layout_reads),*];
        let __smart_cached = {
            let __smart_cache = self.__smart_alignment.lock().unwrap_or_else(|e| e.into_inner());
            match &*__smart_cache {
                Some((layouts, aligned)) if layouts.as_slice() == __smart_layouts.as_slice() => {
                    Some(*aligned)
                }
                _ => None,
            }
        };
        let __smart_aligned = match __smart_cached {
            Some(aligned) => aligned,
            None => {
                let aligned = #first_guard.is_id_ordered()
                    #(#aligned_terms)*
                    #(#mutable_sparse_terms)*;
                *self.__smart_alignment.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((__smart_layouts.to_vec(), aligned));
                aligned
            }
        };
        if __smart_aligned {
            #aligned_call
        } else {
            #general_branch
        }
    }
}
