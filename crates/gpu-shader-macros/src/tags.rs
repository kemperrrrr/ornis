//! Uniform field tags of a parameter's `GpuStruct` type.
//!
//! The attribute cannot read another trait's associated constants, so it does
//! not know the tags at expansion time. It therefore reads them from the type
//! in a `const` context while the shader crate is compiled, which keeps the
//! published table and the fields the kernel reads in step. A hand-written
//! `uniform_tags = […]` list could drift; the first form of this attribute
//! accepted one without checking it against anything.

use quote::quote;
use syn::Ident;

/// A uniform parameter the record must describe.
pub(super) struct UniformParam {
    /// The struct type the parameter is a `Uniform` of.
    pub(super) ty: syn::Type,
    /// The parameter's name, which names its metadata constants.
    pub(super) ident: Ident,
}

/// Emit the record global and the per-parameter metadata constants.
///
/// `workgroup` is already validated by the caller; `uniforms` is in signature
/// order, which is the order the record lists them in.
pub(super) fn emit(
    entry: &Ident,
    workgroup: [u32; 3],
    uniforms: &[UniformParam],
) -> proc_macro2::TokenStream {
    let prefix_text = format!(
        "entry={entry};workgroup={},{},{}",
        workgroup[0], workgroup[1], workgroup[2]
    );
    let prefix = syn::LitByteStr::new(prefix_text.as_bytes(), entry.span());
    let prefix_len = prefix_text.len();
    let names: Vec<(Ident, Ident)> = uniforms
        .iter()
        .map(|param| {
            let upper = param.ident.to_string().to_uppercase();
            (
                Ident::new(&format!("ORNIS_GPU_TAGS_{upper}"), param.ident.span()),
                Ident::new(&format!("ORNIS_GPU_ID_{upper}"), param.ident.span()),
            )
        })
        .collect();
    let definitions = uniforms.iter().zip(&names).map(|(param, (tags, id))| {
        let ty = &param.ty;
        quote! {
            const #tags: &'static [u8] = <#ty as ::ornis_gpu_shader::GpuStruct>::FIELD_TAGS;
            const #id: u32 = <#ty as ::ornis_gpu_shader::GpuStruct>::ID;
        }
    });
    let lengths: Vec<proc_macro2::TokenStream> = names
        .iter()
        .map(|(tags, _)| {
            quote! { #tags.len() }
        })
        .collect();
    let total = quote! { ::ornis_gpu_shader::meta::len(#prefix, &[#(#lengths),*]) };
    // Fold the record left to right: each step starts where the previous one
    // ended, and `meta::cursor` computes that offset from the same tag counts.
    let mut record = quote! {
        ::ornis_gpu_shader::meta::entry::<{ #total }>(#prefix)
    };
    for (index, (tags, id)) in names.iter().enumerate() {
        let first = index == 0;
        let index = syn::Index::from(index);
        record = quote! {
            ::ornis_gpu_shader::meta::with_uniforms(
                #record,
                ::ornis_gpu_shader::meta::cursor(#prefix_len, &[#(#lengths),*], #index),
                #first,
                #id,
                #tags,
            )
        };
    }
    quote! {
        #(#definitions)*

        #[used]
        #[cfg_attr(
            target_arch = "wasm32",
            unsafe(link_section = "ornis_gpu.shader_meta")
        )]
        #[unsafe(export_name = "ornis_gpu_shader_meta")]
        static ORNIS_GPU_SHADER_META: [u8; { #total }] = #record;
    }
}
