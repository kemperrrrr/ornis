//! `#[derive(ShaderContext)]` — declare a stage-entry context bundle.
//!
//! A context bundle is a plain Rust struct whose fields name the WGSL
//! globals an entry reads (`ctx: LightingContext`, `ctx.depth_tex`):
//! field name === global name, no mapping. The `#[stage]` `context`
//! lowering strips the `ctx.` prefix, so the struct is the single source
//! of truth for the contract — and this derive re-exports it as
//! `GLOBALS` for tests (`stage_globals_declared` pins every name against
//! the assembled shader).
//!
//! Field types are documentary for GPU handles (see the `Texture2d` …
//! markers) and real for buffer layouts (`CameraUniform`, …). The struct
//! itself is ordinary compiled Rust; only its *uses* inside `#[stage]`
//! bodies are DSL-lowered.

use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

/// Derive the `GLOBALS` contract list on a context-bundle struct.
///
/// `# Errors`
///
/// Fails when the input is not a braced struct (only named fields can
/// name globals).
pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;
    let syn::Data::Struct(data) = &input.data else {
        return syn::Error::new_spanned(
            &name,
            "ShaderContext: only structs can be context bundles",
        )
        .to_compile_error()
        .into();
    };
    let syn::Fields::Named(fields) = &data.fields else {
        return syn::Error::new_spanned(
            &name,
            "ShaderContext: context bundles need named fields (`field: Type`)",
        )
        .to_compile_error()
        .into();
    };
    let names: Vec<String> = fields
        .named
        .iter()
        .map(|f| {
            f.ident
                .as_ref()
                .expect("named field has an ident")
                .to_string()
        })
        .collect();
    TokenStream::from(quote! {
        impl #name {
            /// WGSL global names this context bundle reads — one per field,
            /// in field order. Pinned against assembled shaders by
            /// `stage_globals_declared`, so adding a field without wiring
            /// the global fails loudly.
            #[allow(dead_code)]
            pub const GLOBALS: &'static [&'static str] = &[#(#names),*];
        }
    })
}
