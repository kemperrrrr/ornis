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
//! - `#[derive(WgslInterface)]` — mirror a Rust struct as a WGSL
//!   shader-interface struct (vertex inputs, varyings): location/builtin
//!   assignments generated from field attributes.
//! - `#[derive(PipelineConfig)]` — declare pipeline metadata (bind-group
//!   layout, backend selection) that the engine builds without hand-written
//!   boilerplate.
//! - `for_each_entity!` stamps per-component dispatch glue; `#[smart_system]`
//!   turns a per-object function into a `System` whose accesses come from
//!   parameter types.
//!
//! Shader *bodies* are not generated here: they are ordinary Rust in the
//! `gpu-*` crates, translated by `ornis-llvm2gpu` (`crates/llvm2gpu`) from
//! LLVM IR, with the WGSL checked in. The old DSL entry points (`#[stage]`,
//! `#[wgsl_fn]`, `#[kernel]`, `#[gpu_pipeline]`) and the `ornis-shader-lang`
//! crate they lowered through were removed in #263.
#![warn(missing_docs)]

mod auto_pipeline;
mod for_each_entity;
mod pack;
mod pack_gpu;
mod pack_lanes;
mod pack_mut;
mod pipeline_config;
mod plain_resource;
mod register_component;
mod smart_system;
mod smart_system_body_links;
mod smart_system_body_visit;
mod smart_system_dispatch;
mod smart_system_emit;
mod smart_system_emit_params;
mod smart_system_fields;
mod smart_system_lanes;
mod smart_system_parse;
mod smart_system_parse_params;
mod smart_system_step;
mod static_profile;
mod wgsl_interface;
mod wgsl_struct;
mod words;

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
#[proc_macro_derive(WgslStruct, attributes(wgsl))]
pub fn derive_wgsl_struct(input: TokenStream) -> TokenStream {
    wgsl_struct::derive(input)
}

/// Mirror a Rust struct as a WGSL shader-interface struct definition.
///
/// Emits the `struct { ... }` with `@location`/`@builtin`/`@interpolate`
/// attributes generated from `#[wgsl(...)]` field attributes, so the
/// interface assignment lives in Rust instead of a handwritten literal.
#[proc_macro_derive(WgslInterface, attributes(wgsl))]
pub fn derive_wgsl_interface(input: TokenStream) -> TokenStream {
    wgsl_interface::derive(input)
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

/// Derive static pipeline configuration from `#[gpu]` / `#[cpu]` / `#[auto]`
/// attributes: bind-group layout, shader entry points and backend selection
/// metadata consumed by the engine's pipeline builder.
#[proc_macro_derive(PipelineConfig, attributes(gpu, cpu, auto))]
pub fn derive_pipeline_config(input: TokenStream) -> TokenStream {
    pipeline_config::derive(input)
}

/// Derive the structural "no interior mutability" marker for a
/// `#[smart_system]` resource.
///
/// Emits `unsafe impl ornis_core::PlainResource` and its `ResMutResource`
/// marker with one `Field: PlainResource` bound per field, so a struct holding
/// a `Mutex`/`RwLock`/atomic fails instead of sharing mutable state through
/// `Res` (see `ornis_core::Res`).
/// See [`smart_system`].
#[proc_macro_derive(PlainResource)]
#[allow(dead_code)] // proc-macro entry: invoked by the compiler on derive use, invisible to the call graph (8 sibling entries carry the same note via baseline)
pub fn derive_plain_resource(input: TokenStream) -> TokenStream {
    plain_resource::derive(input)
}

/// Per-object system kernel (IDEAS §32).
///
/// Turns a function for one object (`u: &mut T` / `&T`, `T: Pack`, plus
/// `Res`/`ResMut`/`EventWriter`/`EventReader`/`#[config]` parameters) into
/// a struct implementing `ornis_core::System`, generating the entity loop and the
/// access set from the parameter types.
#[proc_macro_attribute]
#[allow(dead_code)] // proc-macro entry: invoked by the compiler on attribute use, invisible to the call graph (8 sibling entries carry the same note via baseline)
pub fn smart_system(attr: TokenStream, item: TokenStream) -> TokenStream {
    smart_system::attribute(attr, item)
}

/// Derive the canonical component name for the F0 registry.
///
/// Implements `ornis_core::registry::RegisterComponent` with
/// `COMPONENT_NAME` coming from `#[component(name = \"...\")]` or, when
/// absent, from the type ident itself. Use with
/// `ComponentRegistry::register_component::<T>()`.
#[proc_macro_derive(RegisterComponent, attributes(component))]
pub fn derive_register_component(input: TokenStream) -> TokenStream {
    register_component::derive(input)
}
