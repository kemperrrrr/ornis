//! Bind-group-layout emission for the explicit resources of a
//! `#[gpu_pipeline(pack(...))]` shader.
//!
//! One `wgpu::BindGroupLayoutEntry` per `storage` / `uniform` /
//! `texture` / `sampler` option, mirroring the resource table
//! `ornis_render::shaders::ResourceKind::bgl_ty` (compute pipelines:
//! `COMPUTE` visibility, no multisampling); `var<private>` takes no
//! bind-group slot. The pack lanes append their own entries in
//! `super::create_bind_group_layout_tokens`.

use ornis_shader_lang::ir::{IrGlobal, IrItem, IrTexture};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::LitStr;

use crate::gpu_pipeline::ShaderConfig;
use crate::gpu_pipeline_pack::literal;

/// Module-scope WGSL names (globals and struct declarations) the pack
/// lanes must not collide with, plus the bind-group entries of the
/// globals that occupy a binding.
pub(crate) fn explicit_resources(config: &ShaderConfig) -> (Vec<LitStr>, Vec<TokenStream2>) {
    // WGSL module-scope declarations share one identifier space.
    let mut explicit_names: Vec<LitStr> = Vec::new();
    let mut explicit_entries: Vec<TokenStream2> = Vec::new();
    for item in &config.items {
        match item {
            IrItem::Global(global) => {
                explicit_names.push(literal(&global_name(global)));
                if let Some(entry) = bgl_entry_tokens(global) {
                    explicit_entries.push(entry);
                }
            }
            IrItem::Struct { name, .. } => {
                explicit_names.push(literal(name));
            }
        }
    }
    (explicit_names, explicit_entries)
}

fn global_name(global: &IrGlobal) -> String {
    match global {
        IrGlobal::Storage { name, .. }
        | IrGlobal::Uniform { name, .. }
        | IrGlobal::Texture { name, .. }
        | IrGlobal::Sampler { name, .. }
        | IrGlobal::Private { name, .. } => name.clone(),
    }
}

/// `wgpu::BindGroupLayoutEntry` literal for one explicit resource,
/// mirroring `crates/render/src/shaders/mod.rs` `ResourceKind::bgl_ty`
/// (compute pipelines: COMPUTE visibility, no multisampling). `None`
/// for `var<private>` — it takes no bind-group slot.
fn bgl_entry_tokens(global: &IrGlobal) -> Option<TokenStream2> {
    Some(match global {
        IrGlobal::Storage {
            binding, read_only, ..
        } => quote! {
            wgpu::BindGroupLayoutEntry {
                binding: #binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: #read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        },
        IrGlobal::Uniform { binding, .. } => quote! {
            wgpu::BindGroupLayoutEntry {
                binding: #binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        },
        IrGlobal::Texture { binding, kind, .. } => {
            let (sample_type, view_dimension) = texture_sample_ty(kind);
            quote! {
                wgpu::BindGroupLayoutEntry {
                    binding: #binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: #sample_type,
                        view_dimension: #view_dimension,
                        multisampled: false,
                    },
                    count: None,
                }
            }
        }
        IrGlobal::Sampler {
            binding,
            comparison,
            ..
        } => {
            let ty = if *comparison {
                quote! { wgpu::SamplerBindingType::Comparison }
            } else {
                quote! { wgpu::SamplerBindingType::Filtering }
            };
            quote! {
                wgpu::BindGroupLayoutEntry {
                    binding: #binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(#ty),
                    count: None,
                }
            }
        }
        // `var<private>` takes no bind-group slot.
        IrGlobal::Private { .. } => return None,
    })
}

/// `(sample_type, view_dimension)` for one texture kind — the same
/// closed world `parse_texture_sampler_group` accepts.
fn texture_sample_ty(kind: &IrTexture) -> (TokenStream2, TokenStream2) {
    match kind {
        IrTexture::Tex2dF32 => (
            quote! { wgpu::TextureSampleType::Float { filterable: true } },
            quote! { wgpu::TextureViewDimension::D2 },
        ),
        IrTexture::Tex2dU32 => (
            quote! { wgpu::TextureSampleType::Uint },
            quote! { wgpu::TextureViewDimension::D2 },
        ),
        IrTexture::Tex2dI32 => (
            quote! { wgpu::TextureSampleType::Sint },
            quote! { wgpu::TextureViewDimension::D2 },
        ),
        IrTexture::Depth2d => (
            quote! { wgpu::TextureSampleType::Depth },
            quote! { wgpu::TextureViewDimension::D2 },
        ),
        IrTexture::Depth2dArray => (
            quote! { wgpu::TextureSampleType::Depth },
            quote! { wgpu::TextureViewDimension::D2Array },
        ),
        IrTexture::DepthCubeArray => (
            quote! { wgpu::TextureSampleType::Depth },
            quote! { wgpu::TextureViewDimension::CubeArray },
        ),
    }
}
