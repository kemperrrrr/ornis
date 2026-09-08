//! `#[derive(WgslInterface)]` — generate a WGSL shader-interface struct
//! declaration from a Rust mirror.
//!
//! Where [`WgslStruct`](crate::wgsl_struct) covers GPU *buffer* layouts
//! (uniform/storage, verified with `offset_of!`), this derive covers shader
//! *interface* structs: vertex inputs, varyings, fragment inputs/outputs.
//! These have no CPU-side memory layout to check — their contract is the
//! location/builtin assignment — so the derive emits only the declaration,
//! and `naga` validation in tests rejects duplicates or bad attributes.
//!
//! # Attributes
//!
//! - `#[wgsl(name = "Foo")]` (container) — emit `struct Foo` instead of
//!   `struct RustName`.
//! - `#[wgsl(location = N)]` (field) — `@location(N)`.
//! - `#[wgsl(builtin = "...")]` (field) — `@builtin(...)` (`position`,
//!   `vertex_index`, …). Exactly one of `location`/`builtin` is required.
//! - `#[wgsl(interpolate = "...")]` (field) — `@interpolate(...)` (`flat`,
//!   `linear`, …); only together with `location`.
//!
//! Field types are limited to scalars and `[T; 2..=4]` (WGSL scalars and
//! vectors) — interfaces never carry matrices or nested structs.
//!
//! # Generated API
//!
//! - `impl X { pub const WGSL_SOURCE: &'static str }` — the WGSL struct
//!   declaration, spliced into assembled shader sources.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input, spanned::Spanned};

/// Per-field interface assignment parsed from `#[wgsl(...)]`.
struct FieldAttr {
    location: Option<u32>,
    builtin: Option<String>,
    interpolate: Option<String>,
}

fn parse_field_attr(attrs: &[syn::Attribute]) -> syn::Result<FieldAttr> {
    let mut out = FieldAttr {
        location: None,
        builtin: None,
        interpolate: None,
    };
    for attr in attrs {
        if !attr.path().is_ident("wgsl") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("location") {
                let lit: syn::LitInt = meta.value()?.parse()?;
                out.location = Some(lit.base10_parse()?);
                Ok(())
            } else if meta.path.is_ident("builtin") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.builtin = Some(lit.value());
                Ok(())
            } else if meta.path.is_ident("interpolate") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                out.interpolate = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error(
                    "WgslInterface: unknown field option; expected `location = N`, `builtin = \"...\"` or `interpolate = \"...\"`",
                ))
            }
        })?;
    }
    Ok(out)
}

/// Read the optional `#[wgsl(name = "...")]` container attribute.
fn interface_name(input: &DeriveInput) -> syn::Result<String> {
    for attr in &input.attrs {
        if !attr.path().is_ident("wgsl") {
            continue;
        }
        let mut found = None;
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                found = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error("WgslInterface: unknown option; expected `name = \"...\"`"))
            }
        })?;
        if let Some(name) = found {
            return Ok(name);
        }
    }
    Ok(input.ident.to_string())
}

/// Map a mirror field type to WGSL: scalars and `[T; 2..=4]` only.
fn wgsl_field_type(ty: &syn::Type) -> syn::Result<String> {
    if let syn::Type::Path(tp) = ty {
        match tp
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .as_deref()
        {
            Some("f32") => return Ok("f32".to_string()),
            Some("u32") => return Ok("u32".to_string()),
            Some("i32") => return Ok("i32".to_string()),
            Some("bool") => return Ok("bool".to_string()),
            _ => {}
        }
    }
    if let syn::Type::Array(arr) = ty {
        let len: usize = match &arr.len {
            syn::Expr::Lit(l) => match &l.lit {
                syn::Lit::Int(i) => i.base10_parse()?,
                _ => {
                    return Err(syn::Error::new(
                        arr.len.span(),
                        "WgslInterface: array length must be an integer literal",
                    ));
                }
            },
            _ => {
                return Err(syn::Error::new(
                    arr.len.span(),
                    "WgslInterface: array length must be an integer literal",
                ));
            }
        };
        if !(2..=4).contains(&len) {
            return Err(syn::Error::new(
                arr.len.span(),
                format!("WgslInterface: array length {len} is not supported; use 2..=4"),
            ));
        }
        if let syn::Type::Path(tp) = arr.elem.as_ref() {
            match tp
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .as_deref()
            {
                Some("f32") => return Ok(format!("vec{len}<f32>")),
                Some("u32") => return Ok(format!("vec{len}<u32>")),
                Some("i32") => return Ok(format!("vec{len}<i32>")),
                _ => {}
            }
        }
    }
    Err(syn::Error::new(
        ty.span(),
        "WgslInterface: unsupported field type; use f32/u32/i32 or fixed-size arrays of them",
    ))
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let wgsl_name = match interface_name(&input) {
        Ok(n) => n,
        Err(e) => return e.to_compile_error().into(),
    };

    if !input.generics.params.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "WgslInterface: generic structs are not supported",
        )
        .to_compile_error()
        .into();
    }

    let fields = match &input.data {
        Data::Struct(s) => &s.fields,
        _ => {
            return syn::Error::new(
                input.ident.span(),
                "WgslInterface: only supported on structs",
            )
            .to_compile_error()
            .into();
        }
    };
    let Fields::Named(named) = fields else {
        return syn::Error::new(
            fields.span(),
            "WgslInterface: only structs with named fields are supported",
        )
        .to_compile_error()
        .into();
    };

    let mut decl_lines: Vec<String> = Vec::new();
    for field in &named.named {
        let ident = field.ident.as_ref().expect("named field");
        let attr = match parse_field_attr(&field.attrs) {
            Ok(a) => a,
            Err(e) => return e.to_compile_error().into(),
        };
        let wgsl_ty = match wgsl_field_type(&field.ty) {
            Ok(t) => t,
            Err(e) => return e.to_compile_error().into(),
        };
        let prefix = match (&attr.location, &attr.builtin, &attr.interpolate) {
            (Some(_), Some(_), _) => {
                return syn::Error::new(
                    field.ident.span(),
                    "WgslInterface: `location` and `builtin` are mutually exclusive",
                )
                .to_compile_error()
                .into();
            }
            (None, None, _) => {
                return syn::Error::new(
                    field.ident.span(),
                    "WgslInterface: every field needs `location = N` or `builtin = \"...\"`",
                )
                .to_compile_error()
                .into();
            }
            (Some(loc), None, interp) => match interp {
                Some(i) => format!("@location({loc}) @interpolate({i})"),
                None => format!("@location({loc})"),
            },
            (None, Some(_), Some(_)) => {
                return syn::Error::new(
                    field.ident.span(),
                    "WgslInterface: `interpolate` requires `location`",
                )
                .to_compile_error()
                .into();
            }
            (None, Some(b), None) => format!("@builtin({b})"),
        };
        decl_lines.push(format!("    {prefix} {ident}: {wgsl_ty},"));
    }

    let decl_text = format!("struct {wgsl_name} {{\n{}\n}}\n", decl_lines.join("\n"));
    let decl_lit = proc_macro2::Literal::string(&decl_text);

    let expanded = quote! {
        impl #name {
            /// The WGSL interface declaration generated from this Rust mirror.
            ///
            /// Field attributes are the single source of truth for the
            /// location/builtin assignment; see the `WgslInterface` derive
            /// documentation.
            pub const WGSL_SOURCE: &'static str = #decl_lit;
        }
    };

    TokenStream::from(expanded)
}
