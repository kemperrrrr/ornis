//! Code generation for `#[smart_system]` (IDEAS §32).
//!
//! Expands one validated signature plus its analyzed body into the
//! per-entity loop (`pack_entities` → `pack_get` → body → `pack_put`)
//! and the `System::access` set derived from parameter types. Fetch
//! statements are hoisted out of the loop (entity-independent borrows).
//! The per-parameter pieces (body signature, access statements, fetches)
//! live in [`crate::smart_system_emit_params`].

use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{Ident, ItemFn, Type};

use crate::smart_system::camel_system_name;
use crate::smart_system_body_links::ForeignRewrite;
use crate::smart_system_body_visit::BodyInfo;
use crate::smart_system_emit_params::{build_access_stmts, build_body_sig, build_fetch_stmts};
use crate::smart_system_parse::{
    check_signature, collect_params, combine_errors, config_struct_parts,
};

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
    entity_ident: &Ident,
    resources_ident: &Ident,
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
                #lane_access
                #(#access_stmts)*
                __access
            }

            fn run(&self, #resources_ident: &ornis_core::Resources) {
                let #store_ident: &ornis_core::SmartStore = #resources_ident
                    .get::<ornis_core::SmartStore>()
                    .expect(concat!("smart_system `", #fn_name_str, "`: missing SmartStore resource"));
                #(#config_bindings)*
                // Parameter fetches are entity-independent: hoist them out
                // of the per-entity loop (they are pure borrows, so this is
                // semantics-preserving and saves a map lookup per entity).
                #(#fetch_stmts)*
                for #entity_ident in <#pack_ty as ornis_core::Pack>::pack_entities(#store_ident) {
                    let Some(#u_binding) =
                        <#pack_ty as ornis_core::Pack>::pack_get(#store_ident, #entity_ident)
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

    // Pre-pass: link bindings and ban diagnostics. Access is never inferred
    // from body syntax; every resource role comes from its parameter type.
    let mut info = BodyInfo::collect(&u_ident);
    info.visit_block(&func.block);
    if !info.errors.is_empty() {
        return Err(combine_errors(info.errors));
    }

    // Rewrite foreign reads (everywhere incl. closures). Bare `return`
    // needs no rewrite: the body is a separate function called once per
    // entity, so returning from it naturally skips to the next entity.
    let store_ident = format_ident!("__smart_store", span = Span::mixed_site());
    let entity_ident = format_ident!("__entity", span = Span::mixed_site());
    let local_u_ident = format_ident!("__u", span = Span::mixed_site());
    let resources_ident = format_ident!("__resources", span = Span::mixed_site());
    let mut foreign = ForeignRewrite {
        pack_ty: &pack_ty,
        links: &info.links,
        store_ident: store_ident.clone(),
    };
    let mut rewritten_stmts = func.block.stmts.clone();
    for stmt in &mut rewritten_stmts {
        foreign.visit_stmt_mut(stmt);
    }

    let body_fn_name = format_ident!(
        "__smart_system_body_{}",
        func.sig.ident,
        span = Span::mixed_site()
    );
    let (body_inputs, u_binding, u_ref, scatter) = build_body_sig(
        &u_ident,
        &pack_ty,
        is_mut,
        &store_ident,
        &entity_ident,
        &local_u_ident,
        &params,
    );
    let access_stmts = build_access_stmts(&params);
    let (fetch_stmts, call_args, config_bindings) =
        build_fetch_stmts(&params, &resources_ident, &entity_ident);
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
        &entity_ident,
        &resources_ident,
        &body_fn_name,
        body_inputs,
        rewritten_stmts,
        call_args,
        u_binding,
        u_ref,
        scatter,
    ))
}
