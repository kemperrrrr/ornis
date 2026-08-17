//! `#[derive(WgslStruct)]` — generate a WGSL struct declaration from a Rust
//! struct and statically verify that the Rust memory layout matches WGSL
//! layout rules.
//!
//! This makes the Rust struct the single source of truth for a GPU buffer
//! layout: the WGSL declaration is produced from the field list instead of
//! being hand-written in parallel, and any divergence between the two layouts
//! becomes a compile error via `offset_of!`/`size_of` assertions.
//!
//! # Layout contract
//!
//! The struct must be `#[repr(C)]` (align(16) recommended) and every field
//! must be a scalar (`f32`, `u32`, `i32`, `bool`) or a fixed-size array of
//! one (`[f32; 2..=4]` etc.). Arrays map to WGSL vectors. WGSL aligns
//! `vec3<T>` to 16 bytes, so a field following a `[f32; 3]` must be padded
//! explicitly (e.g. `_pad: f32`) — the generated assertions reject any other
//! layout at compile time.
//!
//! Field names starting with `_` are stripped of the leading underscore in
//! the generated WGSL (WGSL identifiers may not start with `_`).
//!
//! # Generated API
//!
//! - `impl X { pub const WGSL_SOURCE: &'static str }` — the WGSL struct
//!   declaration (compile-time constant, usable in `format!`/string
//!   assembly).
//! - `const _: [(); 1]` items asserting `size_of::<X>()` and every field's
//!   `offset_of!` against the WGSL layout.

use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input, spanned::Spanned};

/// (WGSL type text, size in bytes, alignment in bytes) for a field type.
fn wgsl_field_type(ty: &syn::Type) -> syn::Result<(String, usize, usize)> {
    match ty {
        syn::Type::Path(tp) => {
            let last = tp.path.segments.last().map(|s| s.ident.to_string());
            match last.as_deref() {
                Some("f32") => Ok(("f32".to_string(), 4, 4)),
                Some("u32") => Ok(("u32".to_string(), 4, 4)),
                Some("i32") => Ok(("i32".to_string(), 4, 4)),
                Some("bool") => Ok(("bool".to_string(), 4, 4)),
                _ => Err(syn::Error::new(
                    ty.span(),
                    "WgslStruct: unsupported field type; use f32/u32/i32/bool or fixed-size arrays of them",
                )),
            }
        }
        syn::Type::Array(arr) => {
            let len: usize = match &arr.len {
                syn::Expr::Lit(l) => match &l.lit {
                    syn::Lit::Int(i) => i.base10_parse()?,
                    _ => {
                        return Err(syn::Error::new(
                            arr.len.span(),
                            "WgslStruct: array length must be an integer literal",
                        ));
                    }
                },
                _ => {
                    return Err(syn::Error::new(
                        arr.len.span(),
                        "WgslStruct: array length must be an integer literal",
                    ));
                }
            };
            if !(2..=4).contains(&len) {
                return Err(syn::Error::new(
                    arr.len.span(),
                    format!(
                        "WgslStruct: array length {len} is not supported; use 2..=4 (maps to a WGSL vector)"
                    ),
                ));
            }
            let elem = match arr.elem.as_ref() {
                syn::Type::Path(tp) => tp.path.segments.last().map(|s| s.ident.to_string()),
                _ => None,
            };
            let (elem_wgsl, elem_size) = match elem.as_deref() {
                Some("f32") => ("f32", 4),
                Some("u32") => ("u32", 4),
                Some("i32") => ("i32", 4),
                Some("bool") => ("bool", 4),
                _ => {
                    return Err(syn::Error::new(
                        arr.elem.span(),
                        "WgslStruct: unsupported array element type; use f32/u32/i32/bool",
                    ));
                }
            };
            // WGSL: vec2<T> aligns to 2×, vec3/vec4<T> to 4× the scalar.
            let align = if len == 2 { 2 * elem_size } else { 4 * elem_size };
            Ok((format!("vec{len}<{elem_wgsl}>"), len * elem_size, align))
        }
        _ => Err(syn::Error::new(
            ty.span(),
            "WgslStruct: unsupported field type; use f32/u32/i32/bool or fixed-size arrays of them",
        )),
    }
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    if !input.generics.params.is_empty() {
        return syn::Error::new(
            proc_macro2::Span::call_site(),
            "WgslStruct: generic structs are not supported",
        )
        .to_compile_error()
        .into();
    }

    let fields = match &input.data {
        Data::Struct(s) => &s.fields,
        _ => {
            return syn::Error::new(
                input.ident.span(),
                "WgslStruct: only supported on structs",
            )
            .to_compile_error()
            .into();
        }
    };
    let Fields::Named(named) = fields else {
        return syn::Error::new(
            fields.span(),
            "WgslStruct: only structs with named fields are supported",
        )
        .to_compile_error()
        .into();
    };

    // Layout walk: WGSL member offsets per WGSL alignment rules.
    let mut wgsl_fields: Vec<(String, String, usize)> = Vec::new(); // (wgsl name, wgsl ty, offset)
    let mut rust_field_idents: Vec<syn::Ident> = Vec::new();
    let mut offset = 0usize;
    let mut max_align = 1usize;

    for field in &named.named {
        let ident = field.ident.as_ref().expect("named field");
        let (wgsl_ty, size, align) = match wgsl_field_type(&field.ty) {
            Ok(t) => t,
            Err(e) => return e.to_compile_error().into(),
        };
        let wgsl_name = {
            let s = ident.to_string();
            s.trim_start_matches('_').to_string()
        };
        offset = offset.div_ceil(align) * align;
        wgsl_fields.push((wgsl_name, wgsl_ty, offset));
        rust_field_idents.push(ident.clone());
        offset += size;
        max_align = max_align.max(align);
    }
    let stride = offset.div_ceil(max_align) * max_align;

    let decl_lines = wgsl_fields
        .iter()
        .map(|(wgsl_name, wgsl_ty, _)| format!("    {wgsl_name}: {wgsl_ty},"));
    let decl_text = format!(
        "struct {name} {{\n{}\n}}\n",
        decl_lines.collect::<Vec<_>>().join("\n")
    );
    let decl_lit = proc_macro2::Literal::string(&decl_text);
    let stride_lit = proc_macro2::Literal::usize_suffixed(stride);

    let offset_asserts = wgsl_fields
        .iter()
        .zip(&rust_field_idents)
        .map(|((_, _, wgsl_offset), ident)| {
            let off = proc_macro2::Literal::usize_suffixed(*wgsl_offset);
            quote! {
                const _: [(); 1] = [(); (::core::mem::offset_of!(#name, #ident) == #off) as usize];
            }
        });

    let expanded = quote! {
        impl #name {
            /// The WGSL struct declaration generated from this Rust layout.
            ///
            /// The field list is the single source of truth for the GPU
            /// buffer layout; see the `WgslStruct` derive documentation.
            pub const WGSL_SOURCE: &'static str = #decl_lit;
        }

        // Compile-time verification that the Rust layout (repr(C)) matches
        // the WGSL layout computed above. Any mismatch is a compile error.
        const _: [(); 1] = [(); (::core::mem::size_of::<#name>() == #stride_lit) as usize];
        #(#offset_asserts)*
    };

    TokenStream::from(expanded)
}
