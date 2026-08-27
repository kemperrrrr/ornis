//! Ornis proc-macro crate: GPU pipeline and packing code generation.
//!
//! The macros form a compile-time bridge between Rust component structs and
//! the wgpu render/compute layer:
//!
//! - `#[derive(Pack)]` / `#[derive(AutoPipeline)]` — GPU packing: generate
//!   bytemuck-compatible flat layouts for uploading Rust data into uniform/
//!   storage buffers (component packing).
//! - `#[derive(WgslStruct)]` — mirror a Rust struct as a WGSL struct
//!   definition so shader-side types stay layout-compatible with the Rust
//!   side.
//! - `#[kernel]` — translate a small Rust function AST into WGSL compute
//!   code (used e.g. for BRDF math in `render/shaders/math.rs`).
//! - `#[gpu_pipeline]`, `#[derive(PipelineConfig)]`,
//!   `#[smart_pipeline]` — declare pipelines with static bind-group /
//!   layout metadata, letting the engine build them without hand-written
//!   boilerplate; `for_each_entity!` stamps per-component dispatch glue.
#![warn(missing_docs)]

mod auto_pipeline;
mod for_each_entity;
mod gpu_pipeline;
mod kernel;
mod pack;
mod pipeline_config;
mod smart_pipeline;
mod static_profile;
mod wgsl;
mod wgsl_struct;

use proc_macro::TokenStream;

/// Derive GPU buffer packing for a component struct.
///
/// Generates the flat, bytemuck-compatible representation used when the
/// engine uploads entity/component data into uniform or storage buffers;
/// `#[pack(...)]` attributes on fields control alignment/ordering of the
/// packed layout. Paired with `#[derive(WgslStruct)]` on the shader side.
#[proc_macro_derive(AutoPipeline, attributes(pack))]
pub fn derive_auto_pipeline(input: TokenStream) -> TokenStream {
    auto_pipeline::derive(input)
}

/// Derive the minimal packed-layout impl for a plain data struct.
///
/// Like [`derive_auto_pipeline`] but without pipeline wiring: just the
/// buffer packing derived from the `#[pack(...)]` field attributes.
#[proc_macro_derive(Pack, attributes(pack))]
pub fn derive_pack(input: TokenStream) -> TokenStream {
    pack::derive(input)
}

/// Mirror a Rust struct as a WGSL struct definition.
///
/// Emits the equivalent `struct { ... }` in WGSL source (field names,
/// vector/matrix types) so the shader-side type stays in sync with the CPU
/// definition; intended to be combined with packing derives to guarantee
/// identical memory layouts on both sides of a buffer upload.
#[proc_macro_derive(WgslStruct)]
pub fn derive_wgsl_struct(input: TokenStream) -> TokenStream {
    wgsl_struct::derive(input)
}

/// Stamp a code block once per registered entity component.
///
/// Expansion-time repetition over the engine's component list: each copy
/// gets the component type substituted, generating the per-component
/// dispatch/registration glue that would otherwise be maintained by hand.
#[proc_macro]
pub fn for_each_entity(input: TokenStream) -> TokenStream {
    for_each_entity::for_each_entity(input)
}

/// Declare a GPU pipeline whose configuration is chosen at runtime between
/// handwritten variants ("smart" dispatch): inspects the attributed item and
/// emits a dispatcher selecting the appropriate precompiled path based on
/// the attribute's criteria.
#[proc_macro_attribute]
pub fn smart_pipeline(attr: TokenStream, item: TokenStream) -> TokenStream {
    smart_pipeline::attribute(attr, item)
}

/// Derive static pipeline configuration from `#[gpu]` / `#[cpu]` / `#[auto]`
/// attributes: bind-group layout, shader entry points and backend selection
/// metadata consumed by the engine's pipeline builder.
#[proc_macro_derive(PipelineConfig, attributes(gpu, cpu, auto))]
pub fn derive_pipeline_config(input: TokenStream) -> TokenStream {
    pipeline_config::derive(input)
}

/// Attribute macro declaring a GPU pipeline on a function or struct.
///
/// Transforms the annotated item into a pipeline definition (shader source,
/// bind-group/layout descriptors, entry point) that the renderer can build
/// directly, replacing hand-written `wgpu` boilerplate.
#[proc_macro_attribute]
pub fn gpu_pipeline(attr: TokenStream, item: TokenStream) -> TokenStream {
    gpu_pipeline::gpu_pipeline(attr, item)
}

/// Translate a small Rust function into WGSL compute-shader code.
///
/// Parses the annotated function's AST (not runtime execution) and re-emits
/// it as WGSL — the mechanism behind the engine's Rust-authored BRDF/shader
/// math (`render/shaders/math.rs`) staying a single source of truth.
#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    kernel::kernel(attr, item)
}
