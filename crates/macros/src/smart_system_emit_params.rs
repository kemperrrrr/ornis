//! Parameter-side code generation for `#[smart_system]` (IDEAS §32).
//!
//! Builds the per-parameter pieces of the generated system: the body
//! function's parameter list, the access contributions
//! (`Res`/`ResMut`/event roles), and the run-time fetches with their call
//! arguments and `#[config]` bindings. [`crate::smart_system_emit`] owns
//! the surrounding system-struct and per-entity-loop emission.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Ident, Type};

use crate::smart_system::{ParamKind, TypedParam, with_elided_lifetime};

/// Body-function signature pieces: parameter inputs plus the entity
/// binding, its reference form, and the scatter-back statement.
pub(crate) fn build_body_sig(
    u_ident: &Ident,
    pack_ty: &Type,
    is_mut: bool,
    store_ident: &Ident,
    entity_ident: &Ident,
    local_u_ident: &Ident,
    params: &[TypedParam],
) -> (Vec<TokenStream2>, TokenStream2, TokenStream2, TokenStream2) {
    let mut body_inputs = Vec::new();
    body_inputs.push({
        let u = u_ident;
        if is_mut {
            quote! { #u: &mut #pack_ty }
        } else {
            quote! { #u: &#pack_ty }
        }
    });
    body_inputs.push(quote! { #store_ident: &ornis_core::SmartStore });
    for param in params {
        let name = &param.name;
        match &param.kind {
            ParamKind::Config => {
                let ty = &param.inner;
                body_inputs.push(quote! { #name: #ty });
            }
            ParamKind::EntityHandle => {
                body_inputs.push(quote! { #name: ornis_core::Entity });
            }
            ParamKind::Res
            | ParamKind::ResMut
            | ParamKind::EventWriter
            | ParamKind::EventReader => {
                let ty = with_elided_lifetime(param.wrapper.clone());
                body_inputs.push(quote! { #name: #ty });
            }
        }
    }
    let u_binding = if is_mut {
        quote! { mut #local_u_ident }
    } else {
        quote! { #local_u_ident }
    };
    let u_ref = if is_mut {
        quote! { &mut #local_u_ident }
    } else {
        quote! { &#local_u_ident }
    };
    let scatter = if is_mut {
        quote! {
            ornis_core::Pack::pack_put(&#local_u_ident, #store_ident, #entity_ident);
        }
    } else {
        quote! {}
    };
    (body_inputs, u_binding, u_ref, scatter)
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

