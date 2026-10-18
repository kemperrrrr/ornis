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
    fn guard(&self) -> Ident {
        format_ident!("__smart_lane_{}", self.field, span = Span::mixed_site())
    }

    fn value(&self) -> Ident {
        format_ident!("__smart_v_{}", self.field, span = Span::mixed_site())
    }

    fn index(&self) -> Ident {
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

/// Arguments matching [`body_lane_params`]: gathered values (by reference)
/// and the guarded foreign stores.
fn body_arg_stmts(lanes: &[LaneSpec]) -> Vec<TokenStream2> {
    let mut args = Vec::new();
    for lane in lanes {
        if lane.own {
            let value = lane.value();
            if lane.written {
                args.push(quote! { &mut #value });
            } else {
                args.push(quote! { &#value });
            }
        }
        if lane.foreign {
            let guard = lane.guard();
            args.push(quote! { &*#guard });
        }
    }
    args
}

/// Per-entity step: index resolution, gather, body call, scatter. Uses
/// `continue` to skip an entity, so it must sit directly in the loop.
fn step_stmts(
    lanes: &[LaneSpec],
    body_fn: &Ident,
    user_args: &[TokenStream2],
    entity: &Ident,
    hint: &Ident,
    aligned: bool,
) -> TokenStream2 {
    let own: Vec<&LaneSpec> = lanes.iter().filter(|lane| lane.own).collect();
    let indices = own.iter().map(|lane| {
        let index = lane.index();
        if aligned {
            quote! { let #index = Some(#hint); }
        } else {
            let guard = lane.guard();
            quote! { let #index = #guard.index_hinted(#entity, #hint); }
        }
    });
    let gathers = own.iter().map(|lane| {
        let ty = &lane.lane_ty;
        let guard = lane.guard();
        let index = lane.index();
        let value = lane.value();
        let mutability = if lane.written {
            quote! { mut }
        } else {
            quote! {}
        };
        quote! {
            let Some(#mutability #value) =
                <#ty as ornis_core::PackLane>::gather(&*#guard, #index) else { continue; };
        }
    });
    let body_args = body_arg_stmts(lanes);
    let scatters = own.iter().filter(|lane| lane.written).map(|lane| {
        let ty = &lane.lane_ty;
        let guard = lane.guard();
        let index = lane.index();
        let value = lane.value();
        quote! {
            <#ty as ornis_core::PackLane>::scatter(&mut *#guard, #entity, #index, #value);
        }
    });
    quote! {
        #(#indices)*
        #(#gathers)*
        #body_fn(#(#body_args,)* #(#user_args),*);
        #(#scatters)*
    }
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

/// Whole-run loop: the aligned fast path (dense lane walk) or the general
/// membership walk. Both visit the same entities in the same order.
pub(crate) fn run_loop(
    lanes: &[LaneSpec],
    body_fn: &Ident,
    user_args: &[TokenStream2],
    entity: &Ident,
    hint: &Ident,
) -> TokenStream2 {
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
    let fast_step = step_stmts(lanes, body_fn, user_args, entity, hint, true);
    let general_step = step_stmts(lanes, body_fn, user_args, entity, hint, false);
    let assert = driver_assert(lanes);
    quote! {
        #assert
        let __smart_aligned = #first_guard.is_id_ordered()
            #(#aligned_terms)*
            #(#mutable_sparse_terms)*;
        if __smart_aligned {
            for __smart_i in 0..#first_guard.entities.len() {
                let #entity = #first_guard.entities[__smart_i];
                let #hint = __smart_i;
                #fast_step
            }
        } else {
            let mut __smart_membership = ornis_core::PackMembership::new();
            #(#membership_terms)*
            let __smart_entities = __smart_membership.into_entities();
            for (__smart_i, #entity) in __smart_entities.into_iter().enumerate() {
                let #hint = __smart_i;
                #general_step
            }
        }
    }
}
