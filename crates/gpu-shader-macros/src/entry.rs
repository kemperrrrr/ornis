//! Expansion of `#[gpu_entry(compute(threads(N)))]`.
//!
//! The expansion is a rewrite of the annotated function plus one metadata
//! global:
//!
//! 1. every `#[gpu(…)]` parameter leaves the exported signature (the
//!    translator lowers a plain `extern "C" fn cs_main(u32)`), and its name is
//!    bound at the head of the body to the zero-sized resource value — with the
//!    group, binding, and access mode the attribute names written into the
//!    type;
//! 2. the attribute's own arguments are validated — `compute(threads(N))` is
//!    the only stage the translator implements today;
//! 3. a `#[used]` record is emitted whose uniform tags are read from the
//!    parameter types, not from the attribute.

use quote::quote;
use syn::Expr;
use syn::Lit;
use syn::parse::Parser;
use syn::punctuated::Punctuated;

use crate::param;
use crate::tags::{self, UniformParam};

/// Expand one annotated function.
pub(super) fn expand(
    attr: proc_macro2::TokenStream,
    item: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let mut function: syn::ItemFn = syn::parse2(item)?;
    let name = function.sig.ident.clone();
    let workgroup = parse_attr(attr, &name)?;
    if let Some(asyncness) = &function.sig.asyncness {
        return Err(syn::Error::new_spanned(
            asyncness,
            "a GPU entry cannot be async",
        ));
    }
    if !function.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &function.sig.generics,
            "a GPU entry cannot be generic",
        ));
    }
    let resources = split_params(&mut function)?;
    let metadata = tags::emit(&name, workgroup, &resources.uniforms);
    let body = function.block.clone();
    let bindings = &resources.bindings;
    function.block = syn::parse2(quote! {
        {
            #(#bindings)*
            #body
        }
    })?;
    Ok(quote! {
        #function
        #metadata
    })
}

/// Parse `compute(threads(N))` into an `@workgroup_size` triple.
fn parse_attr(attr: proc_macro2::TokenStream, name: &syn::Ident) -> syn::Result<[u32; 3]> {
    let parser = Punctuated::<Expr, syn::Token![,]>::parse_terminated;
    let exprs = parser.parse2(attr)?;
    let mut threads = None;
    for expr in exprs {
        let Expr::Call(call) = expr else {
            return Err(syn::Error::new_spanned(
                expr,
                "expected `compute(threads(N))`",
            ));
        };
        if !path_ident_is(&call.func, "compute") {
            return Err(syn::Error::new_spanned(
                call.func,
                "only `compute(threads(N))` is supported; vertex and fragment entries follow",
            ));
        }
        let Some(Expr::Call(inner)) = call.args.first() else {
            return Err(syn::Error::new_spanned(call.args, "expected `threads(N)`"));
        };
        if !path_ident_is(&inner.func, "threads") {
            return Err(syn::Error::new_spanned(
                &inner.func,
                "expected `threads(N)`",
            ));
        }
        let Some(Expr::Lit(lit)) = inner.args.first() else {
            return Err(syn::Error::new_spanned(
                &inner.args,
                "threads size must be an integer literal",
            ));
        };
        let Lit::Int(value) = &lit.lit else {
            return Err(syn::Error::new_spanned(
                lit,
                "threads size must be an integer literal",
            ));
        };
        threads = Some(value.base10_parse::<u32>()?);
    }
    let threads =
        threads.ok_or_else(|| syn::Error::new_spanned(name, "missing `compute(threads(N))`"))?;
    if threads == 0 {
        return Err(syn::Error::new_spanned(
            name,
            "threads size must be positive",
        ));
    }
    Ok([threads, 1, 1])
}

fn path_ident_is(expr: &Expr, expected: &str) -> bool {
    match expr {
        Expr::Path(path) => path.path.get_ident().is_some_and(|id| id == expected),
        _ => false,
    }
}

/// Resources the signature declared, after they leave the parameter list.
struct Resources {
    /// `let name = Type::__bind();` lines for the body head.
    bindings: Vec<proc_macro2::TokenStream>,
    /// Uniform parameters, in signature order.
    uniforms: Vec<UniformParam>,
}

/// Move every `#[gpu(…)]` parameter out of the signature.
fn split_params(function: &mut syn::ItemFn) -> syn::Result<Resources> {
    let mut plain = Punctuated::new();
    let mut bindings = Vec::new();
    let mut uniforms = Vec::new();
    for arg in core::mem::take(&mut function.sig.inputs) {
        let syn::FnArg::Typed(mut typed) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "a GPU entry takes no `self` receiver",
            ));
        };
        let mut attributes = Vec::with_capacity(typed.attrs.len());
        let mut resource = None;
        for attribute in core::mem::take(&mut typed.attrs) {
            match param::parse(&attribute, &typed.ty)? {
                Some(parsed) => {
                    if resource.is_some() {
                        return Err(syn::Error::new_spanned(
                            attribute,
                            "a parameter has at most one gpu binding",
                        ));
                    }
                    resource = Some(parsed);
                }
                None => attributes.push(attribute),
            }
        }
        typed.attrs = attributes;
        let Some(resource) = resource else {
            plain.push(syn::FnArg::Typed(typed));
            continue;
        };
        let syn::Pat::Ident(pattern) = &*typed.pat else {
            return Err(syn::Error::new_spanned(
                &typed.pat,
                "a resource parameter must be a plain identifier",
            ));
        };
        let ident = pattern.ident.clone();
        let ty = &resource.ty;
        bindings.push(quote! {
            let #ident: #ty = <#ty>::__bind();
        });
        if resource.is_uniform() {
            uniforms.push(UniformParam {
                ty: resource.element.clone(),
                ident,
            });
        }
    }
    function.sig.inputs = plain;
    Ok(Resources { bindings, uniforms })
}
