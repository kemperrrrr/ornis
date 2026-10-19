//! Const-generic aligned loop for `#[smart_system]` without foreign lanes.
//!
//! Systems that read `t.field` keep the gather loop: a zip would hold
//! `&mut data` and `&store` at once. `Dispatch<false, …>` zips dense
//! field columns through [`ornis_core::DensePackLane::as_slice`];
//! `Dispatch<true, …>` gathers. The kernel is a closure so the zip impl
//! never names the concrete body (`L::Field` vs the body's `f32`).

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::Ident;

use crate::smart_system::{ParamKind, TypedParam};
use crate::smart_system_lanes::LaneSpec;

/// Inputs for the per-system aligned dispatch trait.
pub(crate) struct DispatchSpec<'a> {
    /// Unique trait name (`{body}__Dispatch`).
    pub trait_name: &'a Ident,
    /// Per-entity body kernel.
    pub body_fn: &'a Ident,
    /// All lanes of the system (own only in this path).
    pub lanes: &'a [LaneSpec],
    /// User parameters of the original function.
    pub params: &'a [TypedParam],
    /// Body-call argument tokens, same order as `params`.
    pub user_args: &'a [TokenStream2],
    /// Binding for `#[entity]` inside the loop.
    pub entity: &'a Ident,
    /// Loop index binding of the aligned gather fallback.
    pub hint: &'a Ident,
    /// Whether the body takes an `Entity` parameter.
    pub binds_entity: bool,
}

fn own_lanes(lanes: &[LaneSpec]) -> Vec<&LaneSpec> {
    lanes.iter().filter(|lane| lane.own).collect()
}

fn generic_idents(own: &[&LaneSpec]) -> Vec<Ident> {
    own.iter()
        .enumerate()
        .map(|(i, _)| format_ident!("__L{i}"))
        .collect()
}

fn store_params(own: &[&LaneSpec], generics: &[Ident]) -> Vec<TokenStream2> {
    own.iter()
        .zip(generics)
        .map(|(lane, gty)| {
            let guard = lane.guard();
            if lane.written {
                quote! { #guard: &mut ornis_core::ComponentStore<#gty> }
            } else {
                quote! { #guard: &ornis_core::ComponentStore<#gty> }
            }
        })
        .collect()
}

fn store_call_args(own: &[&LaneSpec]) -> Vec<TokenStream2> {
    own.iter()
        .map(|lane| {
            let guard = lane.guard();
            if lane.written {
                quote! { &mut *#guard }
            } else {
                quote! { &*#guard }
            }
        })
        .collect()
}

fn kernel_params(
    own: &[&LaneSpec],
    generics: &[Ident],
    spec: &DispatchSpec<'_>,
) -> Vec<TokenStream2> {
    let mut params = Vec::new();
    for (lane, gty) in own.iter().zip(generics) {
        if lane.written {
            params.push(quote! { &mut <#gty as ornis_core::PackLane>::Field });
        } else {
            params.push(quote! { &<#gty as ornis_core::PackLane>::Field });
        }
    }
    if spec.binds_entity {
        params.push(quote! { ornis_core::Entity });
    }
    params
}

fn zip_loop(spec: &DispatchSpec<'_>, own: &[&LaneSpec]) -> TokenStream2 {
    let slices: Vec<TokenStream2> = own
        .iter()
        .enumerate()
        .map(|(i, lane)| {
            let guard = lane.guard();
            let slice = format_ident!("__s{i}");
            if lane.written {
                quote! {
                    let #slice = ornis_core::DensePackLane::as_mut_slice(&mut #guard.data);
                }
            } else {
                quote! {
                    let #slice = ornis_core::DensePackLane::as_slice(&#guard.data);
                }
            }
        })
        .collect();
    let first_slice = format_ident!("__s0");
    let mut iter = if own[0].written {
        quote! { #first_slice.iter_mut() }
    } else {
        quote! { #first_slice.iter() }
    };
    let id0 = format_ident!("__z0");
    let mut pat = quote! { #id0 };
    let mut call = vec![quote! { #id0 }];
    for (i, lane) in own.iter().enumerate().skip(1) {
        let slice = format_ident!("__s{i}");
        let id = format_ident!("__z{i}");
        let next = if lane.written {
            quote! { #slice.iter_mut() }
        } else {
            quote! { #slice.iter() }
        };
        iter = quote! { #iter.zip(#next) };
        pat = quote! { (#pat, #id) };
        call.push(quote! { #id });
    }
    let first = own[0].guard();
    let entity = spec.entity;
    if spec.binds_entity {
        iter = quote! { #iter.zip(#first.entities.iter().copied()) };
        pat = quote! { (#pat, #entity) };
        call.push(quote! { #entity });
    }
    quote! {
        #(#slices)*
        for #pat in #iter {
            __smart_kernel(#(#call),*);
        }
    }
}

fn gather_loop(spec: &DispatchSpec<'_>, own: &[&LaneSpec], generics: &[Ident]) -> TokenStream2 {
    let first = own[0].guard();
    let hint = format_ident!("__smart_hint");
    let entity = spec.entity;
    let bind = if spec.binds_entity {
        quote! { let #entity = #first.entities[#hint]; }
    } else {
        quote! {}
    };
    let gathers = own.iter().zip(generics).map(|(lane, gty)| {
        let guard = lane.guard();
        let value = lane.value();
        let mutability = if lane.written {
            quote! { mut }
        } else {
            quote! {}
        };
        quote! {
            let #mutability #value =
                <#gty as ornis_core::PackLane>::gather_at(#guard, #hint);
        }
    });
    let scatters = own
        .iter()
        .zip(generics)
        .filter(|(lane, _)| lane.written)
        .map(|(lane, gty)| {
            let guard = lane.guard();
            let value = lane.value();
            quote! {
                <#gty as ornis_core::PackLane>::scatter_at(#guard, #hint, #value);
            }
        });
    let mut call = Vec::new();
    for lane in own {
        let value = lane.value();
        if lane.written {
            call.push(quote! { &mut #value });
        } else {
            call.push(quote! { &#value });
        }
    }
    if spec.binds_entity {
        call.push(quote! { #entity });
    }
    quote! {
        for #hint in 0..#first.entities.len() {
            #bind
            #(#gathers)*
            __smart_kernel(#(#call),*);
            #(#scatters)*
        }
    }
}

fn closure_pats(own: &[&LaneSpec], spec: &DispatchSpec<'_>) -> Vec<Ident> {
    let mut pats: Vec<Ident> = own
        .iter()
        .enumerate()
        .map(|(i, _)| format_ident!("__c{i}"))
        .collect();
    if spec.binds_entity {
        pats.push(spec.entity.clone());
    }
    pats
}

fn body_call(spec: &DispatchSpec<'_>, pats: &[Ident]) -> TokenStream2 {
    let body_fn = spec.body_fn;
    let mut args: Vec<TokenStream2> = pats
        .iter()
        .take(own_lanes(spec.lanes).len())
        .map(|pat| quote! { #pat })
        .collect();
    for (param, user_arg) in spec.params.iter().zip(spec.user_args.iter()) {
        if matches!(param.kind, ParamKind::EntityHandle) {
            let entity = spec.entity;
            args.push(quote! { #entity });
        } else {
            args.push(user_arg.clone());
        }
    }
    quote! { #body_fn(#(#args),*); }
}

/// Trait and both impls, emitted next to the generated `System` impl.
pub(crate) fn emit_defs(spec: &DispatchSpec<'_>) -> TokenStream2 {
    let own = own_lanes(spec.lanes);
    let generics = generic_idents(&own);
    let trait_name = spec.trait_name;
    let store_params = store_params(&own, &generics);
    let kernel_params = kernel_params(&own, &generics, spec);
    let dense_bounds = generics
        .iter()
        .map(|g| quote! { #g: ornis_core::DensePackLane });
    let pack_bounds = generics.iter().map(|g| quote! { #g: ornis_core::PackLane });
    let zip_body = zip_loop(spec, &own);
    let gather_body = gather_loop(spec, &own, &generics);
    quote! {
        trait #trait_name<const __SMART_SPARSE: bool, #(#generics: ornis_core::PackLane),*> {
            fn run<F: FnMut(#(#kernel_params),*)>(
                #(#store_params,)*
                __smart_kernel: F,
            );
        }

        impl<#(#dense_bounds),*> #trait_name<false, #(#generics),*> for () {
            #[inline(always)]
            fn run<F: FnMut(#(#kernel_params),*)>(
                #(#store_params,)*
                mut __smart_kernel: F,
            ) {
                #zip_body
            }
        }

        impl<#(#pack_bounds),*> #trait_name<true, #(#generics),*> for () {
            #[inline(always)]
            fn run<F: FnMut(#(#kernel_params),*)>(
                #(#store_params,)*
                mut __smart_kernel: F,
            ) {
                #gather_body
            }
        }
    }
}

/// Zip dispatch, or the gather loop when a foreign lane would alias a write.
pub(crate) fn plan(spec: &DispatchSpec<'_>) -> (TokenStream2, TokenStream2) {
    if spec.lanes.iter().any(|lane| lane.foreign) {
        use crate::smart_system_step::{StepCtx, aligned_gather_loop};
        return (
            quote! {},
            aligned_gather_loop(
                spec.lanes,
                &StepCtx {
                    body_fn: spec.body_fn,
                    user_args: spec.user_args,
                    entity: spec.entity,
                    hint: spec.hint,
                },
                spec.binds_entity,
            ),
        );
    }
    (emit_defs(spec), emit_call(spec))
}

/// Call of [`emit_defs`] from the aligned branch, with the SPARSE-or of own lanes.
pub(crate) fn emit_call(spec: &DispatchSpec<'_>) -> TokenStream2 {
    let own = own_lanes(spec.lanes);
    let trait_name = spec.trait_name;
    let own_tys = own.iter().map(|lane| &lane.lane_ty);
    let sparse_or = own.iter().map(|lane| {
        let ty = &lane.lane_ty;
        quote! { | <#ty as ornis_core::PackLane>::SPARSE }
    });
    let stores = store_call_args(&own);
    let pats = closure_pats(&own, spec);
    let body = body_call(spec, &pats);
    quote! {
        <() as #trait_name<{ false #(#sparse_or)* }, #(#own_tys),*>>::run(
            #(#stores,)*
            |#(#pats),*| { #body }
        );
    }
}
