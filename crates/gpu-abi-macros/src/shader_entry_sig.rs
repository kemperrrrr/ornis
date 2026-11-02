//! Attribute arguments and signature checks for the `#[shader::*]` macros.
//!
//! These run at expansion time, before type checking: a compute entry takes
//! one `u32` and returns `()`, vertex/fragment entries use wasm scalars, and
//! only compute declares a workgroup size. Diagnostics point at the offending
//! parameter, return type, or literal.

use proc_macro::TokenStream;
use syn::parse::Parser;
use syn::{FnArg, ItemFn, Lit, Meta, PatType, ReturnType, Token, Type, punctuated::Punctuated};

use super::shader_entry::Stage;

pub(super) fn parse_args(stage: Stage, args: TokenStream) -> syn::Result<[u32; 3]> {
    if args.is_empty() {
        return if stage.requires_workgroup() {
            Err(syn::Error::new(
                proc_macro2::Span::call_site(),
                "#[shader::compute] needs `workgroup_size = N`",
            ))
        } else {
            Ok([1, 1, 1])
        };
    }
    if !stage.requires_workgroup() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[shader::vertex] and #[shader::fragment] take no arguments",
        ));
    }
    let metas = Punctuated::<Meta, Token![,]>::parse_terminated.parse(args)?;
    let mut workgroup = None;
    for meta in metas {
        match &meta {
            Meta::NameValue(nv) if nv.path.is_ident("workgroup_size") => {
                let syn::Expr::Lit(expr) = &nv.value else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`workgroup_size` expects an integer literal",
                    ));
                };
                let Lit::Int(lit) = &expr.lit else {
                    return Err(syn::Error::new_spanned(
                        &expr.lit,
                        "`workgroup_size` expects an integer literal",
                    ));
                };
                if workgroup.is_some() {
                    return Err(syn::Error::new_spanned(
                        &meta,
                        "duplicate `workgroup_size` argument",
                    ));
                }
                workgroup = Some(workgroup_size(lit)?);
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unsupported shader argument; expected `workgroup_size = N`",
                ));
            }
        }
    }
    workgroup.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "#[shader::compute] needs `workgroup_size = N`",
        )
    })
}

/// Upper bound on the invocations in one workgroup.
///
/// Matches wgpu's default `max_compute_invocations_per_workgroup` and
/// `max_compute_workgroup_size_x`, so a declared size always fits a default
/// device. Only the x axis is declared, so the product is the value itself.
const MAX_WORKGROUP_INVOCATIONS: u32 = 256;

fn workgroup_size(lit: &syn::LitInt) -> syn::Result<[u32; 3]> {
    let value: u32 = lit.base10_parse()?;
    if value == 0 {
        return Err(syn::Error::new_spanned(
            lit,
            "`workgroup_size` must be greater than zero",
        ));
    }
    let shape = [value, 1, 1];
    let invocations = shape
        .iter()
        .try_fold(1u32, |acc, axis| acc.checked_mul(*axis));
    if invocations.is_none_or(|total| total > MAX_WORKGROUP_INVOCATIONS) {
        return Err(syn::Error::new_spanned(
            lit,
            format!("`workgroup_size` must be at most {MAX_WORKGROUP_INVOCATIONS} invocations"),
        ));
    }
    Ok(shape)
}

pub(super) fn check_signature(stage: Stage, func: &ItemFn) -> syn::Result<()> {
    if func.sig.constness.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.constness,
            "a shader entry point cannot be `const`",
        ));
    }
    if func.sig.asyncness.is_some() {
        return Err(syn::Error::new_spanned(
            func.sig.asyncness,
            "a shader entry point cannot be `async`",
        ));
    }
    if let syn::Safety::Unsafe(token) = &func.sig.safety {
        return Err(syn::Error::new_spanned(
            token,
            "a shader entry point cannot be `unsafe`",
        ));
    }
    if func.sig.abi.is_some() {
        return Err(syn::Error::new_spanned(
            &func.sig.abi,
            "drop the manual ABI; the shader attribute adds `extern \"C\"`",
        ));
    }
    if !func.sig.generics.params.is_empty() || func.sig.generics.where_clause.is_some() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "a shader entry point cannot be generic",
        ));
    }
    let mut params = Vec::new();
    for input in &func.sig.inputs {
        let FnArg::Typed(param) = input else {
            return Err(syn::Error::new_spanned(
                input,
                "a shader entry point cannot take `self`",
            ));
        };
        check_type(param)?;
        params.push(param);
    }
    let result = match &func.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some(ty.as_ref()),
    };
    if let Some(ty) = result {
        check_return_type(ty)?;
    }
    if stage.requires_workgroup() {
        if params.len() != 1 {
            return Err(syn::Error::new_spanned(
                &func.sig.inputs,
                "#[shader::compute] takes exactly one `u32` parameter",
            ));
        }
        let name = scalar_name(&params[0].ty).ok_or_else(|| {
            syn::Error::new_spanned(&params[0].ty, "#[shader::compute] parameter must be `u32`")
        })?;
        if name != "u32" {
            return Err(syn::Error::new_spanned(
                &params[0].ty,
                "#[shader::compute] parameter must be `u32`",
            ));
        }
        if result.is_some_and(|ty| !is_unit(ty)) {
            return Err(syn::Error::new_spanned(
                &func.sig.output,
                "#[shader::compute] must return `()`",
            ));
        }
    }
    Ok(())
}

fn check_type(param: &PatType) -> syn::Result<()> {
    if scalar_name(&param.ty).is_some() {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            &param.ty,
            "supported parameter types are `f32`, `i32`, and `u32`",
        ))
    }
}

fn check_return_type(ty: &Type) -> syn::Result<()> {
    if is_unit(ty) || scalar_name(ty).is_some() {
        Ok(())
    } else {
        Err(syn::Error::new_spanned(
            ty,
            "supported return types are `()`, `f32`, `i32`, and `u32`",
        ))
    }
}

fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(tuple) if tuple.elems.is_empty())
}

fn scalar_name(ty: &Type) -> Option<&'static str> {
    let Type::Path(path) = ty else {
        return None;
    };
    // Only a bare one-segment path such as `f32`: `foo::f32` or `::f32` may
    // name a different type, and the macro runs before name resolution.
    if path.qself.is_some() || path.path.leading_colon.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = path.path.segments.first()?;
    if !segment.arguments.is_none() {
        return None;
    }
    if segment.ident == "f32" {
        Some("f32")
    } else if segment.ident == "i32" {
        Some("i32")
    } else if segment.ident == "u32" {
        Some("u32")
    } else {
        None
    }
}
