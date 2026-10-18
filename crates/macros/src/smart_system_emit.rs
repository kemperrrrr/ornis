//! Code generation for `#[smart_system]` (IDEAS §32, v2b).
//!
//! Expands one validated system into the body kernel (a function per
//! entity that takes the touched fields by reference), the system struct
//! with its constructor, and the `System` impl. Field analysis lives in
//! [`crate::smart_system_fields`], lane guards and the run loop in
//! [`crate::smart_system_lanes`], and the per-parameter pieces in
//! [`crate::smart_system_emit_params`].

use std::collections::HashSet;

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Ident, ItemFn, Stmt, Type};

use crate::smart_system::{TypedParam, camel_system_name};
use crate::smart_system_body_visit::BodyInfo;
use crate::smart_system_emit_params::{build_access_stmts, build_fetch_stmts, build_user_params};
use crate::smart_system_fields::{FieldRewrite, FieldUsage, scan_fields};
use crate::smart_system_lanes::{
    LaneSpec, body_lane_params, build_lane_specs, guard_stmts, lane_access_stmts, run_loop,
};
use crate::smart_system_parse::{
    check_signature, collect_params, combine_errors, config_struct_parts,
};

/// Expands one validated system function into the body kernel plus the
/// generated `System` impl. Thin orchestration over the parse / analyze /
/// build helpers.
pub(crate) fn expand(func: ItemFn) -> Result<TokenStream2, syn::Error> {
    check_signature(&func)?;
    let (u_ident, pack_ty, is_mut, params) = collect_params(func.sig.inputs.iter())?;
    let Type::Path(pack_type) = &pack_ty else {
        return Err(syn::Error::new_spanned(
            &pack_ty,
            "smart_system: the first parameter must be `u: &T` with `T: Pack` — a named bundle type",
        ));
    };
    let pack_path = pack_type.path.clone();

    // Pre-pass: link bindings and ban diagnostics. Access is never inferred
    // from body syntax; every resource role comes from its parameter type.
    let mut info = BodyInfo::collect(&u_ident);
    info.visit_block(&func.block);
    if !info.errors.is_empty() {
        return Err(combine_errors(info.errors));
    }

    let u_name = u_ident.to_string();
    let usage = scan_fields(&func.block, &u_name, &info.links);
    check_usage(&usage, &u_ident, is_mut)?;

    let lanes = build_lane_specs(&usage, &pack_path);
    let body_stmts = rewrite_body(&func.block.stmts, &u_name, &pack_path, &info.links);

    let body_fn_name = format_ident!(
        "__smart_system_body_{}",
        func.sig.ident,
        span = Span::mixed_site()
    );
    let struct_name = camel_system_name(&func.sig.ident);
    let fn_name_str = func.sig.ident.to_string();

    let body_fn = emit_body_fn(&body_fn_name, &lanes, &params, &body_stmts);
    let system_struct = emit_system_struct(&func, &struct_name, &params);
    let system_impl = emit_system_impl(&struct_name, &fn_name_str, &lanes, &params, &body_fn_name);

    Ok(quote! {
        #body_fn
        #system_struct
        #system_impl
    })
}

/// Rejects bodies whose `u` usage cannot be lowered to per-field lanes:
/// whole-entity use, no touched field, or a write through a shared `u`.
fn check_usage(usage: &FieldUsage, u_ident: &Ident, is_mut: bool) -> Result<(), syn::Error> {
    if let Some(span) = usage.bare {
        return Err(syn::Error::new(
            span,
            "smart_system: `u` may only appear as `u.field` — the body sees fields, not the whole entity",
        ));
    }
    if usage.own.is_empty() {
        return Err(syn::Error::new_spanned(
            u_ident,
            "smart_system: the body must read or write at least one `u.field`; the touched fields define the entity set",
        ));
    }
    if !is_mut && let Some(field) = usage.written.iter().next() {
        return Err(syn::Error::new_spanned(
            u_ident,
            format!("smart_system: `u.{field}` is written, so `u` must be `&mut`"),
        ));
    }
    Ok(())
}

/// Copies the body statements with `u.field` and linked `t.field` reads
/// rewritten to lane locals.
fn rewrite_body(
    stmts: &[Stmt],
    u_name: &str,
    pack_path: &syn::Path,
    links: &HashSet<String>,
) -> Vec<Stmt> {
    let mut rewrite = FieldRewrite {
        u: u_name,
        pack_path,
        links,
    };
    let mut body_stmts = stmts.to_vec();
    for stmt in &mut body_stmts {
        rewrite.visit_stmt_mut(stmt);
    }
    body_stmts
}

/// The per-entity body kernel: lane references first, then user parameters.
fn emit_body_fn(
    body_fn_name: &Ident,
    lanes: &[LaneSpec],
    params: &[TypedParam],
    body_stmts: &[Stmt],
) -> TokenStream2 {
    let mut body_inputs = body_lane_params(lanes);
    body_inputs.extend(build_user_params(params));
    quote! {
        #[allow(unused_variables)]
        fn #body_fn_name(#(#body_inputs),*) {
            #(#body_stmts)*
        }
    }
}

/// The system struct, its `#[config]` constructor, and the original attributes.
fn emit_system_struct(func: &ItemFn, struct_name: &Ident, params: &[TypedParam]) -> TokenStream2 {
    let attrs = &func.attrs;
    let (struct_fields, ctor_params, ctor_inits) = config_struct_parts(params);
    quote! {
        #(#attrs)*
        #[allow(non_camel_case_types)]
        pub struct #struct_name {
            #(#struct_fields)*
        }

        impl #struct_name {
            /// Creates the system; `#[config]` parameters become constructor arguments.
            pub fn new(#(#ctor_params),*) -> Self {
                Self {
                    #(#ctor_inits)*
                }
            }
        }
    }
}

/// The `System` impl: access declaration, then one guarded run loop per tick.
fn emit_system_impl(
    struct_name: &Ident,
    fn_name_str: &str,
    lanes: &[LaneSpec],
    params: &[TypedParam],
    body_fn_name: &Ident,
) -> TokenStream2 {
    let store_ident = format_ident!("__smart_store", span = Span::mixed_site());
    let entity_ident = format_ident!("__entity", span = Span::mixed_site());
    let hint_ident = format_ident!("__smart_hint", span = Span::mixed_site());
    let resources_ident = format_ident!("__resources", span = Span::mixed_site());

    let (fetch_stmts, call_args, config_bindings) =
        build_fetch_stmts(params, &resources_ident, &entity_ident);
    let lane_access = lane_access_stmts(lanes);
    let access_stmts = build_access_stmts(params);
    let guards = guard_stmts(lanes, &store_ident);
    let loop_tokens = run_loop(lanes, body_fn_name, &call_args, &entity_ident, &hint_ident);

    quote! {
        impl ornis_core::System for #struct_name {
            fn name(&self) -> &'static str {
                #fn_name_str
            }

            fn access(&self) -> ornis_core::SystemAccess {
                let mut __access =
                    ornis_core::SystemAccess::new().reads::<ornis_core::SmartStore>();
                #(#lane_access)*
                #(#access_stmts)*
                __access
            }

            fn run(&self, #resources_ident: &ornis_core::Resources) {
                let #store_ident: &ornis_core::SmartStore = #resources_ident
                    .get::<ornis_core::SmartStore>()
                    .expect(concat!("smart_system `", #fn_name_str, "`: missing SmartStore resource"));
                #(#config_bindings)*
                // Parameter fetches and lane guards are entity-independent:
                // taken once per run, not per entity.
                #(#fetch_stmts)*
                #(#guards)*
                #loop_tokens
            }
        }
    }
}
