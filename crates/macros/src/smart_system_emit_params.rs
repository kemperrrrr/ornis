//! Parameter-side code generation for `#[smart_system]` (IDEAS §32).
//!
//! Builds the per-parameter pieces of the generated system: the user
//! parameters of the body function, the access contributions
//! (`Res`/`ResMut`/event roles), and the run-time fetches with their call
//! arguments and `#[config]` bindings. The lane side (field access, guards,
//! the per-entity loop) lives in [`crate::smart_system_lanes`], and
//! [`crate::smart_system_emit`] owns the surrounding emission.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::Ident;

use crate::smart_system::{ParamKind, TypedParam, with_elided_lifetime};

/// Body-function parameters for the user's own parameters (after the lane
/// parameters): `#[config]` by value, `#[entity]` as `Entity`, resource and
/// event roles by their wrapper type.
pub(crate) fn build_user_params(params: &[TypedParam]) -> Vec<TokenStream2> {
    params
        .iter()
        .map(|param| {
            let name = &param.name;
            match &param.kind {
                ParamKind::Config => {
                    let ty = &param.inner;
                    quote! { #name: #ty }
                }
                ParamKind::EntityHandle => quote! { #name: ornis_core::Entity },
                ParamKind::Res
                | ParamKind::ResMut
                | ParamKind::EventWriter
                | ParamKind::EventReader => {
                    let ty = with_elided_lifetime(param.wrapper.clone());
                    quote! { #name: #ty }
                }
            }
        })
        .collect()
}

/// Per-parameter access contributions (`Res`/`ResMut`/event roles only;
/// configs and entity handles contribute nothing). Event access follows
/// the wrapper type, never the syntax of the function body.
pub(crate) fn build_access_stmts(params: &[TypedParam]) -> Vec<TokenStream2> {
    let mut access_stmts = Vec::new();
    for param in params {
        if matches!(param.kind, ParamKind::Config | ParamKind::EntityHandle) {
            continue;
        }
        let inner = &param.inner;
        match &param.kind {
            ParamKind::Res => {
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::Res::<#inner>::access());
                });
            }
            ParamKind::ResMut => {
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::ResMut::<#inner>::access());
                });
            }
            ParamKind::EventWriter => {
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::EventWriter::<#inner>::access());
                });
            }
            ParamKind::EventReader => {
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::EventReader::<#inner>::access());
                });
            }
            ParamKind::Config | ParamKind::EntityHandle => {}
        }
    }

    access_stmts
}

/// Run-time fetches, body call arguments, and config bindings cloned
/// from the system struct.
pub(crate) fn build_fetch_stmts(
    params: &[TypedParam],
    resources_ident: &Ident,
    entity_ident: &Ident,
) -> (Vec<TokenStream2>, Vec<TokenStream2>, Vec<TokenStream2>) {
    let mut fetch_stmts = Vec::new();
    let mut call_args = Vec::new();
    for param in params {
        let name = &param.name;
        match &param.kind {
            ParamKind::Config => {
                call_args.push(quote! { #name.clone() });
            }
            ParamKind::EntityHandle => {
                call_args.push(quote! { #entity_ident });
            }
            ParamKind::Res => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::Res::<#inner>::fetch(#resources_ident);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::ResMut => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::ResMut::<#inner>::fetch(#resources_ident);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::EventWriter => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::EventWriter::<#inner>::fetch(#resources_ident);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::EventReader => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::EventReader::<#inner>::fetch(#resources_ident);
                });
                call_args.push(quote! { #name });
            }
        }
    }

    let mut config_bindings = Vec::new();
    for param in params {
        if matches!(param.kind, ParamKind::Config) {
            let name = &param.name;
            config_bindings.push(quote! {
                let #name = self.#name.clone();
            });
        }
    }
    (fetch_stmts, call_args, config_bindings)
}
