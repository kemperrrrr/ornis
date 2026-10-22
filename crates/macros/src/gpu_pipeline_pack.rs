//! `#[gpu_pipeline(pack(T))]`: SoA storage lanes from `#[derive(Pack)]`.
//!
//! The macro cannot see another item's derive expansion, so the lanes
//! are linked the same way `helpers(...)` links kernel sources — at
//! runtime, through the `ornis_core::PackGpu` trait the Pack derive
//! emits. `pack(T, …)` appends one read-only storage declaration per
//! lane of each listed bundle, after every explicit `storage` /
//! `uniform` / `texture` / `sampler` option (so explicit binding
//! numbers stay compile-time constants; the rule is enforced at macro
//! time). The generated module also grows the layout half the
//! full-shader compute path never had: `create_bind_group_layout`
//! covering explicit resources *and* pack lanes, `create_pipeline`
//! against it, and `create_pack_buffers` serializing a `SmartStore`
//! into the lane buffers the layout describes.
//!
//! Layout contract (docs/rendering/shader-lang-pack-lanes.md): lane `i`
//! of the concatenated pack list binds at `base + i`, `base` being the
//! number of explicit bindings; the WGSL name of a lane is its field
//! name; duplicates — across packs or against explicit bindings —
//! panic at `wgsl_source()` with the colliding name, before any GPU
//! object exists.

use ornis_shader_lang::writer::print_builtin_param;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{Ident, LitStr};

use crate::gpu_pipeline::ShaderConfig;
use crate::gpu_pipeline_pack_layout::explicit_resources;

/// Pack-augmented compute module: the static items and entry point are
/// emitted as two literals, `wgsl_source()` stitches helpers → items →
/// pack lanes → entry at runtime, and the layout/pipeline/buffer
/// helpers cover explicit resources plus the pack lanes.
///
/// Like `helpers(...)`, the pack type paths resolve in the *caller's*
/// module: the macro cannot evaluate another item's expansion, so the
/// `PackGpu` references live in loader functions emitted next to the
/// generated module (a bare path inside the module would resolve one
/// scope too deep).
pub(crate) fn expand(
    fn_name: &Ident,
    config: &ShaderConfig,
    items_wgsl: &str,
    body_wgsl: &str,
    helper_loader: Option<Ident>,
) -> TokenStream2 {
    let items_lit = literal(&format!("{items_wgsl}\n"));
    let entry_lit = entry_literal(config, body_wgsl);
    let (explicit_names, explicit_entries) = explicit_resources(config);
    let helper_push = helper_push_tokens(helper_loader);
    let label_str = literal(&fn_name.to_string());
    let base_binding = config.packs_base_binding;
    let lanes_loader = format_ident!("__ornis_{fn_name}_pack_lanes");
    let columns_loader = format_ident!("__ornis_{fn_name}_pack_columns");
    let loaders = loaders_tokens(&config.packs, &lanes_loader, &columns_loader);
    let label = pipeline_label_tokens(&label_str);
    let lanes = pack_gpu_lanes_tokens(&lanes_loader);
    let source = wgsl_source_tokens(
        &items_lit,
        &entry_lit,
        base_binding,
        &explicit_names,
        helper_push,
    );
    let shader = create_shader_module_tokens();
    let bgl = create_bind_group_layout_tokens(&explicit_entries, base_binding);
    let pipeline = create_pipeline_tokens();
    let buffers = create_pack_buffers_tokens(&columns_loader);
    quote! {
        #loaders

        pub mod #fn_name {
            #label
            #lanes
            #source
            #shader
            #bgl
            #pipeline
            #buffers
        }
    }
}

/// The two caller-scope loader functions: every `pack(T)` type resolves
/// here, in the caller's module, never inside the generated module.
fn loaders_tokens(
    packs: &[syn::Path],
    lanes_loader: &Ident,
    columns_loader: &Ident,
) -> TokenStream2 {
    quote! {
        fn #lanes_loader() -> Vec<ornis_core::PackGpuLane> {
            let mut __lanes = Vec::new();
            #(
                __lanes.extend_from_slice(<#packs as ornis_core::PackGpu>::pack_gpu_lanes());
            )*
            __lanes
        }

        fn #columns_loader(
            store: &ornis_core::SmartStore,
        ) -> Option<Vec<Vec<u8>>> {
            let mut __columns: Vec<Vec<u8>> = Vec::new();
            #(
                __columns.extend(
                    <#packs as ornis_core::PackGpu>::pack_gpu_upload(store)?.columns,
                );
            )*
            Some(__columns)
        }
    }
}

/// The entry-point literal: builtins are printed once, here.
fn entry_literal(config: &ShaderConfig, body_wgsl: &str) -> LitStr {
    let builtins = config
        .builtins
        .iter()
        .map(|(name, semantic)| print_builtin_param(name, semantic))
        .collect::<Vec<_>>()
        .join(", ");
    literal(&format!(
        "@compute @workgroup_size({})\nfn main({}) {{\n{}\n}}\n",
        config.workgroup_size, builtins, body_wgsl
    ))
}

/// Optional first stitch piece: `helpers(...)`'s loader, pushed before
/// the static items.
fn helper_push_tokens(helper_loader: Option<Ident>) -> Option<TokenStream2> {
    helper_loader.map(|loader| {
        quote! {
            source.push_str(&super::#loader());
            source.push('\n');
        }
    })
}

fn pipeline_label_tokens(label_str: &LitStr) -> TokenStream2 {
    quote! {
        #[allow(dead_code)]
        pub fn pipeline_label() -> &'static str {
            #label_str
        }
    }
}

fn pack_gpu_lanes_tokens(lanes_loader: &Ident) -> TokenStream2 {
    quote! {
        /// Concatenated pack lanes, in `pack(...)` order.
        #[allow(dead_code)]
        pub fn pack_gpu_lanes() -> Vec<ornis_core::PackGpuLane> {
            super::#lanes_loader()
        }
    }
}

/// The stitch function: helpers → items → pack lanes (each checked
/// against the seen-name list) → entry point.
fn wgsl_source_tokens(
    items_lit: &LitStr,
    entry_lit: &LitStr,
    base_binding: u32,
    explicit_names: &[LitStr],
    helper_push: Option<TokenStream2>,
) -> TokenStream2 {
    quote! {
        #[allow(dead_code)]
        pub fn wgsl_source() -> String {
            let mut source = String::new();
            #helper_push
            source.push_str(#items_lit);
            let mut __binding: u32 = #base_binding;
            let mut __seen: Vec<&'static str> = vec![#(#explicit_names),*];
            for __lane in &pack_gpu_lanes() {
                if __seen.contains(&__lane.field) {
                    panic!(
                        "pack pipeline `{}`: duplicate WGSL storage name `{}`; \
                         rename the pack field or the explicit binding",
                        pipeline_label(),
                        __lane.field,
                    );
                }
                __seen.push(__lane.field);
                source.push_str(&format!(
                    "@group(0) @binding({}) var<storage, read> {}: array<{}>;\n",
                    __binding, __lane.field, __lane.wgsl_elem,
                ));
                __binding += 1;
            }
            source.push_str(#entry_lit);
            source
        }
    }
}

fn create_shader_module_tokens() -> TokenStream2 {
    quote! {
        #[allow(dead_code)]
        pub fn create_shader_module(device: &wgpu::Device) -> wgpu::ShaderModule {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(pipeline_label()),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl_source())),
            })
        }
    }
}

/// `wgpu::BindGroupLayoutEntry`-shaped emission for the explicit
/// resources plus the pack-lane loop.
fn create_bind_group_layout_tokens(
    explicit_entries: &[TokenStream2],
    base_binding: u32,
) -> TokenStream2 {
    quote! {
        /// Bind-group layout covering the explicit `storage`/
        /// `uniform`/`texture`/`sampler` options *and* every pack
        /// lane — the layout `create_pipeline` builds against.
        #[allow(dead_code)]
        pub fn create_bind_group_layout(
            device: &wgpu::Device,
        ) -> wgpu::BindGroupLayout {
            let mut __entries: Vec<wgpu::BindGroupLayoutEntry> =
                vec![#(#explicit_entries),*];
            let mut __binding: u32 = #base_binding;
            for __lane in &pack_gpu_lanes() {
                __entries.push(wgpu::BindGroupLayoutEntry {
                    binding: __binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
                __binding += 1;
            }
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(pipeline_label()),
                entries: &__entries,
            })
        }
    }
}

fn create_pipeline_tokens() -> TokenStream2 {
    quote! {
        #[allow(dead_code)]
        pub fn create_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
            let __shader = create_shader_module(device);
            let __bgl = create_bind_group_layout(device);
            let __layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(pipeline_label()),
                bind_group_layouts: &[Some(&__bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(pipeline_label()),
                layout: Some(&__layout),
                module: &__shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }
    }
}

/// One storage buffer per lane, in `pack(...)` order. `None` when the
/// store does not carry the packs.
fn create_pack_buffers_tokens(columns_loader: &Ident) -> TokenStream2 {
    quote! {
        /// Serializes the store's pack lanes into one storage buffer
        /// per lane, in `pack(...)` order — the buffers the pack
        /// entries of [`create_bind_group_layout`] describe. `None`
        /// when the store does not carry the packs. An empty store is
        /// not an error: every upload pads its columns with one zeroed
        /// element (wgpu rejects zero-size bindings), so `len == 0`
        /// still produces bindable buffers.
        #[allow(dead_code)]
        pub fn create_pack_buffers(
            device: &wgpu::Device,
            store: &ornis_core::SmartStore,
        ) -> Option<Vec<wgpu::Buffer>> {
            use wgpu::util::DeviceExt;
            let __columns = super::#columns_loader(store)?;
            let __lanes = pack_gpu_lanes();
            if __lanes.len() != __columns.len() {
                return None;
            }
            Some(
                __columns
                    .into_iter()
                    .zip(__lanes.iter())
                    .map(|(__bytes, __lane)| {
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some(__lane.field),
                            contents: &__bytes,
                            usage: wgpu::BufferUsages::STORAGE,
                        })
                    })
                    .collect(),
            )
        }
    }
}

/// Shared with `gpu_pipeline_pack_layout` — one copy, two callers.
pub(crate) fn literal(s: &str) -> LitStr {
    LitStr::new(s, proc_macro2::Span::call_site())
}
