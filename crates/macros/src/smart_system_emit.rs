//! Code generation for `#[smart_system]` (IDEAS §32).
//!
//! Expands one validated signature plus its analyzed body into the
//! per-entity loop (`pack_entities` → `pack_get` → body → `pack_put`)
//! and the `System::access` set derived from parameter types. Fetch
//! statements are hoisted out of the loop (entity-independent borrows).

use std::collections::HashMap;

use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Ident, ItemFn, Type};

use crate::smart_system::{ParamKind, TypedParam, camel_system_name, with_elided_lifetime};
use crate::smart_system_body_links::ForeignRewrite;
use crate::smart_system_body_visit::BodyInfo;
use crate::smart_system_parse::{
    check_signature, collect_params, combine_errors, config_struct_parts,
};

/// Body-function signature pieces: parameter inputs plus the entity
/// binding, its reference form, and the scatter-back statement.
fn build_body_sig(
    u_ident: &Ident,
    pack_ty: &Type,
    is_mut: bool,
    store_ident: &Ident,
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
            ParamKind::Res | ParamKind::ResMut | ParamKind::Events => {
                let ty = with_elided_lifetime(param.wrapper.clone());
                body_inputs.push(quote! { #name: #ty });
            }
        }
    }
    let u_binding = if is_mut {
        quote! { mut __u }
    } else {
        quote! { __u }
    };
    let u_ref = if is_mut {
        quote! { &mut __u }
    } else {
        quote! { &__u }
    };
    let scatter = if is_mut {
        quote! {
            ornis_core::Pack::pack_put(&__u, #store_ident, __entity);
        }
    } else {
        quote! {}
    };
    (body_inputs, u_binding, u_ref, scatter)
}

/// Per-parameter access contributions, run-time fetches, body call
/// arguments, and config bindings cloned from the system struct.
/// Access contributions per parameter (`Res`/`ResMut`/`Events` only;
/// configs and entity handles contribute nothing).
fn build_access_stmts(params: &[TypedParam], sends: &HashMap<String, bool>) -> Vec<TokenStream2> {
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
            ParamKind::Events => {
                let name = &param.name;
                let sends = sends.get(&name.to_string()).copied().unwrap_or(false);
                access_stmts.push(quote! {
                    __access = __access.combine(ornis_core::Events::<#inner>::access(#sends));
                });
            }
            // Config/EntityHandle skip above; the empty arm keeps the
            // match exhaustive without a panic path.
            ParamKind::Config | ParamKind::EntityHandle => {}
        }
    }

    access_stmts
}

/// Run-time fetches, body call arguments, and config bindings cloned
/// from the system struct.
fn build_fetch_stmts(
    params: &[TypedParam],
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
                call_args.push(quote! { __entity });
            }
            ParamKind::Res => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::Res::<#inner>::fetch(__resources);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::ResMut => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::ResMut::<#inner>::fetch(__resources);
                });
                call_args.push(quote! { #name });
            }
            ParamKind::Events => {
                let inner = &param.inner;
                fetch_stmts.push(quote! {
                    let #name = ornis_core::Events::<#inner>::fetch(__resources);
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

/// Whole-bundle lane access plus the `SmartStore` read every system
/// declares. `is_mut` selects write vs read declarations.
fn build_lane_access(pack_ty: &Type, is_mut: bool) -> TokenStream2 {
    quote! {
        let mut __access =
            ornis_core::SystemAccess::new().reads::<ornis_core::SmartStore>();
        for __lane in <#pack_ty as ornis_core::Pack>::pack_lane_ids() {
            __access = if #is_mut {
                __access.writes_lane_id(__lane)
            } else {
                __access.reads_lane_id(__lane)
            };
        }
    }
}

/// The per-entity body function (rewritten user body, store handle first).
fn emit_body_fn(
    body_fn_name: &Ident,
    body_inputs: Vec<TokenStream2>,
    rewritten_stmts: Vec<syn::Stmt>,
) -> TokenStream2 {
    quote! {
        #[allow(unused_variables)]
        fn #body_fn_name(#(#body_inputs),*) {
            #(#rewritten_stmts)*
        }
    }
}

/// Assembles the generated items: the per-entity body function, the
/// system struct (+ constructor), and the `System` impl.
#[allow(clippy::too_many_arguments)]
fn emit_system(
    struct_name: &Ident,
    fn_name_str: &str,
    attrs: &[syn::Attribute],
    struct_fields: Vec<TokenStream2>,
    ctor_params: Vec<TokenStream2>,
    ctor_inits: Vec<TokenStream2>,
    pack_ty: &Type,
    is_mut: bool,
    access_stmts: Vec<TokenStream2>,
    fetch_stmts: Vec<TokenStream2>,
    config_bindings: Vec<TokenStream2>,
    store_ident: &Ident,
    body_fn_name: &Ident,
    body_inputs: Vec<TokenStream2>,
    rewritten_stmts: Vec<syn::Stmt>,
    call_args: Vec<TokenStream2>,
    u_binding: TokenStream2,
    u_ref: TokenStream2,
    scatter: TokenStream2,
) -> TokenStream2 {
    let lane_access = build_lane_access(pack_ty, is_mut);
    let body_fn = emit_body_fn(body_fn_name, body_inputs, rewritten_stmts);
    quote! {
        #body_fn

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

        impl ornis_core::System for #struct_name {
            fn name(&self) -> &'static str {
                #fn_name_str
            }

            fn access(&self) -> ornis_core::SystemAccess {
                let mut __access = #lane_access;
                #(#access_stmts)*
                __access
            }

            fn run(&self, __resources: &ornis_core::Resources) {
                let #store_ident: &ornis_core::SmartStore = __resources
                    .get::<ornis_core::SmartStore>()
                    .expect(concat!("smart_system `", #fn_name_str, "`: missing SmartStore resource"));
                #(#config_bindings)*
                // Parameter fetches are entity-independent: hoist them out
                // of the per-entity loop (they are pure borrows, so this is
                // semantics-preserving and saves a map lookup per entity).
                #(#fetch_stmts)*
                for __entity in <#pack_ty as ornis_core::Pack>::pack_entities(#store_ident) {
                    let Some(#u_binding) =
                        <#pack_ty as ornis_core::Pack>::pack_get(#store_ident, __entity)
                    else {
                        continue;
                    };
                    #body_fn_name(#u_ref, #store_ident, #(#call_args),*);
                    #scatter
                }
            }
        }
    }
}

/// Expands one validated system function into the body kernel plus the
/// generated `System` impl. Thin orchestration over the parse / analyze
/// / build helpers above (each kept branch- or call-pure on its own).
pub(crate) fn expand(func: ItemFn) -> Result<TokenStream2, syn::Error> {
    check_signature(&func)?;
    let (u_ident, pack_ty, is_mut, params) = collect_params(func.sig.inputs.iter())?;

    // Pre-pass: links, sends, ban diagnostics.
    let mut info = BodyInfo::collect(&u_ident, &params);
    info.visit_block(&func.block);
    if !info.errors.is_empty() {
        return Err(combine_errors(info.errors));
    }

    // Rewrite foreign reads (everywhere incl. closures). Bare `return`
    // needs no rewrite: the body is a separate function called once per
    // entity, so returning from it naturally skips to the next entity.
    let store_ident = format_ident!("__smart_store");
    let mut foreign = ForeignRewrite {
        pack_ty: &pack_ty,
        links: &info.links,
        store_ident: store_ident.clone(),
    };
    let mut rewritten_stmts = func.block.stmts.clone();
    for stmt in &mut rewritten_stmts {
        foreign.visit_stmt_mut(stmt);
    }

    let body_fn_name = format_ident!("__smart_system_body_{}", func.sig.ident);
    let (body_inputs, u_binding, u_ref, scatter) =
        build_body_sig(&u_ident, &pack_ty, is_mut, &store_ident, &params);
    let access_stmts = build_access_stmts(&params, &info.sends);
    let (fetch_stmts, call_args, config_bindings) = build_fetch_stmts(&params);
    let struct_name = camel_system_name(&func.sig.ident);
    let fn_name_str = func.sig.ident.to_string();
    let attrs = func.attrs.clone();
    let (struct_fields, ctor_params, ctor_inits) = config_struct_parts(&params);
    Ok(emit_system(
        &struct_name,
        &fn_name_str,
        &attrs,
        struct_fields,
        ctor_params,
        ctor_inits,
        &pack_ty,
        is_mut,
        access_stmts,
        fetch_stmts,
        config_bindings,
        &store_ident,
        &body_fn_name,
        body_inputs,
        rewritten_stmts,
        call_args,
        u_binding,
        u_ref,
        scatter,
    ))
}
