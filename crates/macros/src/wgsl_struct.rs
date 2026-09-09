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
//! must be a scalar (`f32`, `u32`, `i32`), a fixed-size array of one
//! (`[f32; 2..=4]` etc., mapping to WGSL vectors), a 4×4 float matrix
//! (`[[f32; 4]; 4]`, mapping to `mat4x4<f32>`), or another (possibly arrayed)
//! struct that itself has a WGSL layout.
//!
//! WGSL aligns `vec3<T>`/`vec4<T>`/`mat4x4<T>` to 16 bytes, so a scalar field
//! following one must be padded explicitly (e.g. `_pad: f32`) — the generated
//! assertions reject any other layout at compile time.
//!
//! Nested structs must be 16-byte aligned with a 16-multiple size (true for
//! any struct containing a `vec4`/`mat4x4` member); the derive emits a
//! compile-time check for this. Their WGSL reference is by name, so the inner
//! declaration must be visible wherever the outer `WGSL_SOURCE` is used.
//!
//! Field names starting with `_` are stripped of the leading underscore in
//! the generated WGSL (WGSL identifiers may not start with `_`).
//!
//! # Attributes
//!
//! - `#[wgsl(name = "Foo")]` (container) — emit the WGSL declaration as
//!   `struct Foo` instead of `struct RustName`.
//! - `#[wgsl(skip)]` (field) — exclude the field from the WGSL declaration
//!   while keeping it in the layout walk. For padding fields (`_pad`, …):
//!   they occupy real Rust bytes (so offsets/sizes still check out) but are
//!   not shader-visible data. Without `skip`, a `[u32; 3]` pad would surface
//!   as a `vec3<u32>` member and grow the WGSL struct span past the Rust
//!   size — the size assertion rejects that. Skips must be trailing.
//! - `#[wgsl(to = "Light")]` (field, nested structs only) — spell the
//!   reference with another WGSL type name (e.g. a field `lights:
//!   [GpuLight; 4]` referencing WGSL `Light`, whose `name` attribute on the
//!   inner struct this macro cannot see).
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

/// WGSL layout of one field: declaration type plus const-evaluable size and
/// alignment expressions (literals for scalars, `size_of`/`align` queries for
/// nested structs, whose layouts the macro cannot see).
struct FieldLayout {
    wgsl_ty: String,
    size: proc_macro2::TokenStream,
    align: proc_macro2::TokenStream,
}

/// Scalar element mapping: (WGSL scalar name, byte size).
fn scalar_elem(ty: &syn::Type) -> Option<(&'static str, usize)> {
    if let syn::Type::Path(tp) = ty {
        match tp
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .as_deref()
        {
            Some("f32") => Some(("f32", 4)),
            Some("u32") => Some(("u32", 4)),
            Some("i32") => Some(("i32", 4)),
            _ => None,
        }
    } else {
        None
    }
}

/// Resolve a field type to its [`FieldLayout`]; `is_nested` reports whether
/// the field references another struct by name (the caller emits the
/// 16-alignment precondition check for it). `as_name` (`#[wgsl(to = ...)]`)
/// overrides the WGSL type name for nested references, whose `name`
/// attribute the macro cannot see; it is rejected on scalar/vector/matrix
/// fields.
fn wgsl_field_layout(
    ty: &syn::Type,
    as_name: Option<&str>,
) -> syn::Result<(FieldLayout, Option<proc_macro2::TokenStream>)> {
    // Plain scalar: `f32` etc.
    if let Some((name, size)) = scalar_elem(ty) {
        if as_name.is_some() {
            return Err(syn::Error::new(
                ty.span(),
                "WgslStruct: `to` only applies to nested struct fields",
            ));
        }
        let size_lit = proc_macro2::Literal::usize_suffixed(size);
        return Ok((
            FieldLayout {
                wgsl_ty: name.to_string(),
                size: quote! { #size_lit },
                align: quote! { #size_lit },
            },
            None,
        ));
    }
    if let syn::Type::Array(arr) = ty {
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
        // 4×4 float matrix: `[[f32; 4]; 4]` ↔ `mat4x4<f32>` (64 bytes,
        // 16-aligned — the same bytes `bytemuck` uploads for `[[f32; 4]; 4]`).
        if len == 4
            && let syn::Type::Array(inner) = arr.elem.as_ref()
            && let syn::Expr::Lit(l) = &inner.len
            && let syn::Lit::Int(i) = &l.lit
            && i.base10_parse::<usize>().is_ok_and(|n| n == 4)
            && let Some(("f32", _)) = scalar_elem(&inner.elem)
        {
            if as_name.is_some() {
                return Err(syn::Error::new(
                    ty.span(),
                    "WgslStruct: `to` only applies to nested struct fields",
                ));
            }
            return Ok((
                FieldLayout {
                    wgsl_ty: "mat4x4<f32>".to_string(),
                    size: quote! { 64usize },
                    align: quote! { 16usize },
                },
                None,
            ));
        }
        // Scalar vector: `[f32; 2..=4]` ↔ `vecN<f32>`.
        if let Some((elem_wgsl, elem_size)) = scalar_elem(&arr.elem) {
            if as_name.is_some() {
                return Err(syn::Error::new(
                    ty.span(),
                    "WgslStruct: `to` only applies to nested struct fields",
                ));
            }
            if !(2..=4).contains(&len) {
                return Err(syn::Error::new(
                    arr.len.span(),
                    format!(
                        "WgslStruct: array length {len} is not supported; use 2..=4 (maps to a WGSL vector)"
                    ),
                ));
            }
            // WGSL: vec2<T> aligns to 2×, vec3/vec4<T> to 4× the scalar.
            let align = if len == 2 {
                2 * elem_size
            } else {
                4 * elem_size
            };
            let size_lit = proc_macro2::Literal::usize_suffixed(len * elem_size);
            let align_lit = proc_macro2::Literal::usize_suffixed(align);
            return Ok((
                FieldLayout {
                    wgsl_ty: format!("vec{len}<{elem_wgsl}>"),
                    size: quote! { #size_lit },
                    align: quote! { #align_lit },
                },
                None,
            ));
        }
        // Array of nested structs: `[Inner; K]` ↔ `array<W, K>` where `W` is
        // `#[wgsl(to)]` if given, else the Rust type name.
        if let syn::Type::Path(tp) = arr.elem.as_ref() {
            let inner = tp
                .path
                .segments
                .last()
                .expect("path has segments")
                .ident
                .clone();
            let wgsl_inner = as_name
                .map(str::to_string)
                .unwrap_or_else(|| inner.to_string());
            let k = proc_macro2::Literal::usize_suffixed(len);
            return Ok((
                FieldLayout {
                    wgsl_ty: format!("array<{wgsl_inner}, {len}>"),
                    size: quote! { #k * (::core::mem::size_of::<#inner>().div_ceil(16usize) * 16usize) },
                    align: quote! { 16usize },
                },
                Some(quote! { #inner }),
            ));
        }
        return Err(syn::Error::new(
            arr.elem.span(),
            "WgslStruct: unsupported array element type; use f32/u32/i32 or a struct with a WGSL layout",
        ));
    }
    // Nested struct by value: `label: Inner` ↔ `label: W` (see above).
    if let syn::Type::Path(tp) = ty {
        let inner = tp
            .path
            .segments
            .last()
            .expect("path has segments")
            .ident
            .clone();
        let wgsl_inner = as_name
            .map(str::to_string)
            .unwrap_or_else(|| inner.to_string());
        return Ok((
            FieldLayout {
                wgsl_ty: wgsl_inner,
                size: quote! { ::core::mem::size_of::<#inner>() },
                align: quote! { 16usize },
            },
            Some(quote! { #inner }),
        ));
    }
    Err(syn::Error::new(
        ty.span(),
        "WgslStruct: unsupported field type; use f32/u32/i32, fixed-size arrays of them, [[f32; 4]; 4], or a struct with a WGSL layout",
    ))
}

/// Read the container `#[wgsl(name = "...")]` option.
fn wgsl_struct_name(input: &DeriveInput) -> syn::Result<Option<String>> {
    let mut name = None;
    for attr in &input.attrs {
        if !attr.path().is_ident("wgsl") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("name") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                name = Some(lit.value());
                Ok(())
            } else {
                Err(meta.error("WgslStruct: unknown container option; expected `name = \"...\"`"))
            }
        })?;
    }
    Ok(name)
}

/// Field options from `#[wgsl(...)]`: skip/to.
#[derive(Default)]
struct FieldOpts {
    skip: bool,
    as_name: Option<String>,
}

fn field_opts(attrs: &[syn::Attribute]) -> syn::Result<FieldOpts> {
    let mut opts = FieldOpts::default();
    for attr in attrs {
        if !attr.path().is_ident("wgsl") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                opts.skip = true;
                Ok(())
            } else if meta.path.is_ident("to") {
                let lit: syn::LitStr = meta.value()?.parse()?;
                opts.as_name = Some(lit.value());
                Ok(())
            } else {
                Err(meta
                    .error("WgslStruct: unknown field option; expected `skip` or `to = \"...\"`"))
            }
        })?;
    }
    Ok(opts)
}

pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let wgsl_name = match wgsl_struct_name(&input) {
        Ok(opt) => opt.unwrap_or_else(|| name.to_string()),
        Err(e) => return e.to_compile_error().into(),
    };

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
            return syn::Error::new(input.ident.span(), "WgslStruct: only supported on structs")
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

    // Layout walk with symbolic const expressions: `off`/`max_align` are
    // token streams so nested-struct sizes (only known via `size_of` in the
    // downstream crate) still produce exact compile-time assertions.
    // Skipped (padding) fields occupy Rust bytes but are not WGSL members:
    // their offset asserts against the unaligned cursor and they advance it
    // without alignment; they never raise `max_align`. Skips must be
    // trailing — a WGSL-visible member after a skip would sit at a different
    // offset than its Rust counterpart.
    let mut decl_lines: Vec<String> = Vec::new();
    let mut offset_asserts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut nested: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut off: proc_macro2::TokenStream = quote! { 0usize };
    let mut max_align: proc_macro2::TokenStream = quote! { 1usize };
    let mut saw_skip = false;

    for field in &named.named {
        let ident = field.ident.as_ref().expect("named field");
        let opts = match field_opts(&field.attrs) {
            Ok(opts) => opts,
            Err(e) => return e.to_compile_error().into(),
        };
        let (layout, nested_ty) = match wgsl_field_layout(&field.ty, opts.as_name.as_deref()) {
            Ok(t) => t,
            Err(e) => return e.to_compile_error().into(),
        };
        if let Some(inner) = nested_ty {
            nested.push(inner);
        }
        if opts.skip {
            saw_skip = true;
        } else if saw_skip {
            return syn::Error::new(
                field.ident.span(),
                "WgslStruct: #[wgsl(skip)] fields must be trailing — a visible member after a skip would diverge from the Rust layout",
            )
            .to_compile_error()
            .into();
        }
        let wgsl_field = {
            let s = ident.to_string();
            s.trim_start_matches('_').to_string()
        };
        if !opts.skip {
            decl_lines.push(format!("    {}: {},", wgsl_field, layout.wgsl_ty));
        }
        let size = &layout.size;
        let align = &layout.align;
        if opts.skip {
            offset_asserts.push(quote! {
                const _: [(); 1] = [(); (::core::mem::offset_of!(#name, #ident) == #off) as usize];
            });
            off = quote! { (#off + #size) };
        } else {
            off = quote! { (#off.div_ceil(#align) * #align) };
            offset_asserts.push(quote! {
                const _: [(); 1] = [(); (::core::mem::offset_of!(#name, #ident) == #off) as usize];
            });
            off = quote! { (#off + #size) };
            max_align = quote! { if #max_align > #align { #max_align } else { #align } };
        }
    }
    let stride = quote! { (#off.div_ceil(#max_align) * #max_align) };

    let decl_text = format!("struct {wgsl_name} {{\n{}\n}}\n", decl_lines.join("\n"));
    let decl_lit = proc_macro2::Literal::string(&decl_text);
    let name_lit = proc_macro2::Literal::string(&wgsl_name);

    // Nested structs must be 16-aligned for the hardcoded WGSL alignment to
    // hold; any struct containing a vec4/mat4 member satisfies this.
    let nested_asserts = nested.iter().map(|inner| {
        quote! {
            const _: [(); 1] = [(); (::core::mem::align_of::<#inner>() == 16usize) as usize];
        }
    });

    let expanded = quote! {
        impl #name {
            /// The WGSL struct declaration generated from this Rust layout.
            ///
            /// The field list is the single source of truth for the GPU
            /// buffer layout; see the `WgslStruct` derive documentation.
            pub const WGSL_SOURCE: &'static str = #decl_lit;

            /// The WGSL type name of this struct (the `#[wgsl(name)]`
            /// override when present, else the Rust type name).
            pub const WGSL_NAME: &'static str = #name_lit;
        }

        // Compile-time verification that the Rust layout (repr(C)) matches
        // the WGSL layout computed above. Any mismatch is a compile error.
        #(#nested_asserts)*
        const _: [(); 1] = [(); (::core::mem::size_of::<#name>() == #stride) as usize];
        #(#offset_asserts)*
    };

    TokenStream::from(expanded)
}
