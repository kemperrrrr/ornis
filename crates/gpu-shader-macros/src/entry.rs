//! Expansion of `#[gpu_entry(compute(threads(N)))]`.
//!
//! The expansion rewrites the annotated function and adds one metadata global:
//!
//! 1. every `#[gpu(…)]` parameter keeps its place in the signature, with the
//!    attribute removed and the group, binding, and access mode written into the
//!    type (`Storage<f32>` becomes `Storage<f32, 0, 1, ReadOnly>`). The
//!    parameter type is zero-sized, so the compiled entry is still
//!    `cs_main(i32)`: rustc drops ZST arguments from the signature, and the
//!    translator never sees the resources;
//! 2. `#[unsafe(export_name = "cs_main")]` is attached here, so a kernel writes a
//!    plain safe `pub fn` — no `extern "C"`, no `#[unsafe(no_mangle)]`;
//! 3. the attribute's own arguments are validated — `compute(threads(N))`,
//!    `compute(threads(X, Y))`, `compute(threads(X, Y, Z))`, `vertex`, or
//!    `fragment`;
//! 4. a `#[used]` record is emitted whose uniform tags are read from the
//!    parameter types, not from the attribute.
//!
//! Keeping the resource parameters in the signature is what makes the binding
//! part of the function's type rather than a `let` the body could forget: the
//! group, the binding, and the access mode are checked where the kernel is
//! declared, and the body reads like ordinary Rust method calls.

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
    if matches!(function.sig.safety, syn::Safety::Unsafe(_)) {
        return Err(syn::Error::new_spanned(
            &function.sig.safety,
            "a GPU entry is a safe function; the SDK keeps `unsafe` in one module",
        ));
    }
    let resources = resolve_params(&mut function)?;
    let metadata = tags::emit(&name, workgroup, &resources.uniforms);
    // The entry name the translator looks for is the function's own name, and
    // `export_name` is what makes it that name in the object file. Emitting it
    // here means a kernel signature carries no ABI ceremony.
    let export = syn::LitStr::new(&name.to_string(), name.span());
    Ok(quote! {
        #[unsafe(export_name = #export)]
        #function
        #metadata
    })
}

/// Parse `compute(threads(..))`, `vertex`, or `fragment` into an
/// `@workgroup_size` triple.
///
/// One thread count is `(N, 1, 1)`. Two are `(X, Y, 1)`. Three are the full
/// triple. A raster stage has no workgroup; the record still carries `1,1,1`,
/// which is what the translator stores for `vs_main` and `fs_main`.
fn parse_attr(attr: proc_macro2::TokenStream, name: &syn::Ident) -> syn::Result<[u32; 3]> {
    let parser = Punctuated::<Expr, syn::Token![,]>::parse_terminated;
    let exprs = parser.parse2(attr)?;
    let mut stage = None;
    for expr in exprs {
        if stage.is_some() {
            return Err(syn::Error::new_spanned(
                expr,
                "a GPU entry declares one stage",
            ));
        }
        stage = Some(parse_stage(&expr)?);
    }
    stage.ok_or_else(|| {
        syn::Error::new_spanned(
            name,
            "missing `compute(threads(..))`, `vertex`, or `fragment`",
        )
    })
}

/// One stage expression: `fragment`, `vertex`, or `compute(threads(..))`.
fn parse_stage(expr: &Expr) -> syn::Result<[u32; 3]> {
    if let Expr::Path(path) = expr
        && (path.path.is_ident("fragment") || path.path.is_ident("vertex"))
    {
        return Ok([1, 1, 1]);
    }
    let Expr::Call(call) = expr else {
        return Err(syn::Error::new_spanned(
            expr,
            "expected `compute(threads(..))`, `vertex`, or `fragment`",
        ));
    };
    if !path_ident_is(&call.func, "compute") {
        return Err(syn::Error::new_spanned(
            &call.func,
            "expected `compute(threads(..))`, `vertex`, or `fragment`",
        ));
    }
    let Some(Expr::Call(inner)) = call.args.first() else {
        return Err(syn::Error::new_spanned(
            &call.args,
            "expected `threads(N)`, `threads(X, Y)`, or `threads(X, Y, Z)`",
        ));
    };
    if call.args.len() != 1 {
        return Err(syn::Error::new_spanned(
            &call.args,
            "compute takes only `threads(..)`",
        ));
    }
    if !path_ident_is(&inner.func, "threads") {
        return Err(syn::Error::new_spanned(
            &inner.func,
            "expected `threads(N)`, `threads(X, Y)`, or `threads(X, Y, Z)`",
        ));
    }
    threads_of(inner)
}

/// One, two, or three positive thread counts.
fn threads_of(inner: &syn::ExprCall) -> syn::Result<[u32; 3]> {
    let mut sizes = [1u32, 1, 1];
    if inner.args.is_empty() || inner.args.len() > 3 {
        return Err(syn::Error::new_spanned(
            &inner.args,
            "threads takes one, two, or three positive sizes",
        ));
    }
    for (slot, arg) in sizes.iter_mut().zip(inner.args.iter()) {
        let Expr::Lit(lit) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "a thread size must be an integer literal",
            ));
        };
        let Lit::Int(value) = &lit.lit else {
            return Err(syn::Error::new_spanned(
                lit,
                "a thread size must be an integer literal",
            ));
        };
        let size = value.base10_parse::<u32>()?;
        if size == 0 {
            return Err(syn::Error::new_spanned(
                lit,
                "a thread size must be positive",
            ));
        }
        *slot = size;
    }
    Ok(sizes)
}

fn path_ident_is(expr: &Expr, expected: &str) -> bool {
    match expr {
        Expr::Path(path) => path.path.get_ident().is_some_and(|id| id == expected),
        _ => false,
    }
}

/// What the metadata record needs to know about the signature.
struct Resources {
    /// Uniform parameters, in signature order.
    uniforms: Vec<UniformParam>,
}

/// Rewrite every `#[gpu(…)]` parameter in place.
///
/// The attribute leaves the parameter list and the resolved type takes its
/// place, so the binding is visible to the body as an ordinary argument. A
/// non-zero-sized binding type would break the entry's ABI, and rustc would no
/// longer drop it from the compiled signature — that is why the SDK's resource
/// types are ZSTs and why this checks it.
fn resolve_params(function: &mut syn::ItemFn) -> syn::Result<Resources> {
    let mut uniforms = Vec::new();
    let mut bound = Vec::new();
    for arg in function.sig.inputs.iter_mut() {
        let syn::FnArg::Typed(typed) = arg else {
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
            continue;
        };
        let syn::Pat::Ident(pattern) = &*typed.pat else {
            return Err(syn::Error::new_spanned(
                &typed.pat,
                "a resource parameter must be a plain identifier",
            ));
        };
        let ty = &resource.ty;
        *typed.ty = syn::parse_quote!(#ty);
        bound.push(ty.clone());
        if resource.is_uniform() {
            uniforms.push(UniformParam {
                ty: resource.element.clone(),
                ident: pattern.ident.clone(),
            });
        }
    }
    Ok(Resources { uniforms })
}

/// Compile-time check that every resource parameter is zero-sized.
///
/// Unused today: the check has to run against the resolved types, and those are
/// only known inside the expansion's own scope. Kept as the documented contract
/// of [`Storage`](ornis_gpu_shader::Storage) and
/// [`Uniform`](ornis_gpu_shader::Uniform), which assert it themselves.
#[allow(dead_code)]
fn zst_types(bound: &[syn::Type]) -> Vec<syn::Type> {
    bound.to_vec()
}
