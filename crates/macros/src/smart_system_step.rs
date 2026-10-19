//! Per-entity gather/scatter step of `#[smart_system]`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Ident;

use crate::smart_system_lanes::LaneSpec;

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

/// Names used by one per-entity step.
pub(crate) struct StepCtx<'a> {
    pub(crate) body_fn: &'a Ident,
    pub(crate) user_args: &'a [TokenStream2],
    pub(crate) entity: &'a Ident,
    pub(crate) hint: &'a Ident,
}

/// Per-entity step: index, gather, body, scatter. The aligned variant
/// indexes by the loop counter; the general path uses hinted lookup and
/// `continue` to skip a missing required lane.
pub(crate) fn step_stmts(lanes: &[LaneSpec], ctx: &StepCtx<'_>, aligned: bool) -> TokenStream2 {
    let own: Vec<&LaneSpec> = lanes.iter().filter(|lane| lane.own).collect();
    let (indices, gathers, scatters) = if aligned {
        aligned_lane_ops(&own, ctx.hint)
    } else {
        general_lane_ops(&own, ctx.entity, ctx.hint)
    };
    let body_args = body_arg_stmts(lanes);
    let body_fn = ctx.body_fn;
    let user_args = ctx.user_args;
    quote! {
        #(#indices)*
        #(#gathers)*
        #body_fn(#(#body_args,)* #(#user_args),*);
        #(#scatters)*
    }
}

fn aligned_lane_ops(own: &[&LaneSpec], hint: &Ident) -> LaneOps {
    let gathers = own.iter().map(|lane| {
        let ty = &lane.lane_ty;
        let guard = lane.guard();
        let value = lane.value();
        let mutability = if lane.written {
            quote! { mut }
        } else {
            quote! {}
        };
        quote! {
            let #mutability #value = <#ty as ornis_core::PackLane>::gather_at(&*#guard, #hint);
        }
    });
    let scatters = own.iter().filter(|lane| lane.written).map(|lane| {
        let ty = &lane.lane_ty;
        let guard = lane.guard();
        let value = lane.value();
        quote! {
            <#ty as ornis_core::PackLane>::scatter_at(&mut *#guard, #hint, #value);
        }
    });
    (Vec::new(), gathers.collect(), scatters.collect())
}

type LaneOps = (Vec<TokenStream2>, Vec<TokenStream2>, Vec<TokenStream2>);

fn general_lane_ops(own: &[&LaneSpec], entity: &Ident, hint: &Ident) -> LaneOps {
    let indices = own.iter().map(|lane| {
        let index = lane.index();
        let guard = lane.guard();
        quote! { let #index = #guard.index_hinted(#entity, #hint); }
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
    let scatters = own.iter().filter(|lane| lane.written).map(|lane| {
        let ty = &lane.lane_ty;
        let guard = lane.guard();
        let index = lane.index();
        let value = lane.value();
        quote! {
            <#ty as ornis_core::PackLane>::scatter(&mut *#guard, #entity, #index, #value);
        }
    });
    (indices.collect(), gathers.collect(), scatters.collect())
}
/// Index-aligned gather loop used when a foreign lane holds a borrow of a
/// written store, which a zip over `data` cannot.
pub(crate) fn aligned_gather_loop(
    lanes: &[LaneSpec],
    ctx: &StepCtx<'_>,
    binds_entity: bool,
) -> TokenStream2 {
    let own: Vec<&LaneSpec> = lanes.iter().filter(|lane| lane.own).collect();
    let first_guard = own[0].guard();
    let hint = ctx.hint;
    let entity = ctx.entity;
    let bind = if binds_entity {
        quote! { let #entity = #first_guard.entities[#hint]; }
    } else {
        quote! {}
    };
    let step = step_stmts(lanes, ctx, true);
    quote! {
        for #hint in 0..#first_guard.entities.len() {
            #bind
            #step
        }
    }
}
