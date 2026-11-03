//! Editor outline shader (selection/hover silhouettes), generated from Rust.
//!
//! The pass draws an inverted hull: every triangle of an `outline_flags`-marked
//! instance is expanded along its world normal, front faces are culled and the
//! back faces are depth-tested without writing depth. Outside the object's
//! silhouette the expanded back faces survive the depth test and produce a
//! constant-color ring; inside it they lose to the object's own depth.
//!
//! One `u32` flag per uploaded instance drives the effect:
//! `0` = no outline, `1` = hover, `2` = selected. The flags buffer is uploaded
//! by the host (the WASM replica maps `Selected`/`Hovered` onto instance order),
//! so this module stays entity-agnostic.

use super::interface::{
    GbufferVertexInput as VertexInput, OutlineFragmentInput, OutlineVertexOutput,
};
use super::{Resource, ResourceKind, ShaderModule, wgsl_decl};
use crate::renderer::{CameraUniform, PerObjectGpu};
use ornis_macros::stage;

/// Outline vertex resources as a context bundle (`ctx.per_objects`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct OutlineVertexContext {
    pub per_objects: Vec<PerObjectGpu>,
    pub camera: CameraUniform,
    pub outline_flags: Vec<u32>,
}

/// WGSL constants spliced into the vertex module.
///
/// The DSL emits identifiers verbatim, so the numeric literals live here
/// once instead of in the translated Rust body.
const OUTLINE_CONSTS: &str = "\
const OUTLINE_WIDTH: f32 = 0.05;
const OUTLINE_HOVER: vec3<f32> = vec3<f32>(0.55, 0.75, 1.0);
const OUTLINE_SELECTED: vec3<f32> = vec3<f32>(0.16, 0.55, 1.0);
";

/// Outline vertex entry: expand flagged instances along their world normal.
/// Unflagged instances collapse outside clip space, so the draw culls them
/// without a separate draw call per silhouette.
// The `#[stage]` macro consumes this Rust body: the generated WGSL is the
// only caller, so rustc/rustqual see the entry as uncalled.
#[allow(dead_code)]
#[stage(vertex)]
fn vs_main(
    input: VertexInput,
    instance_index: super::InstanceIndex,
    ctx: Context<OutlineVertexContext>,
) -> OutlineVertexOutput {
    let obj = ctx.per_objects[instance_index];
    let flag = ctx.outline_flags[instance_index];
    // Expand in object space so the ring keeps a constant fraction of the
    // mesh size: the replica bakes an entity's radius into the model scale.
    let expanded = input.position + input.normal * OUTLINE_WIDTH;
    let mut clip_position = ctx.camera.view_proj * (obj.model * Vec4::new(expanded, 1.0));
    if flag == 0u {
        // Outside NDC: every vertex of the instance is clipped.
        clip_position = Vec4::new(2.0, 2.0, 2.0, 1.0);
    }
    let mut color = OUTLINE_HOVER;
    if flag == 2u {
        color = OUTLINE_SELECTED;
    }
    let mut output: OutlineVertexOutput;
    output.clip_position = clip_position;
    output.color = color;
    return output;
}

/// Outline fragment entry: the vertex stage already chose hover/selected.
#[allow(dead_code)]
#[stage(fragment)]
fn fs_main(input: OutlineFragmentInput) -> super::Location<0, glam::Vec4> {
    return glam::Vec4::new(input.color, 1.0);
}

/// Outline bindings: camera + per-object records + the per-instance flags.
pub const OUTLINE_RESOURCES: [Resource; 3] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "camera",
        kind: ResourceKind::Uniform(CameraUniform::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 1,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "per_objects",
        kind: ResourceKind::StorageReadArray(PerObjectGpu::WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 2,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "outline_flags",
        kind: ResourceKind::StorageReadArray("u32"),
        min_size: None,
    },
];

/// Outline vertex module: camera/instance transforms plus the flag check.
pub fn wgsl_vertex_source() -> String {
    ShaderModule::new()
        .decl(CameraUniform::WGSL_SOURCE)
        .decl(PerObjectGpu::WGSL_SOURCE)
        .resources(&OUTLINE_RESOURCES, &[0, 1, 2])
        .decl(wgsl_decl(VertexInput::WGSL_SOURCE))
        .decl(wgsl_decl(OutlineVertexOutput::WGSL_SOURCE))
        .consts(OUTLINE_CONSTS)
        .entry(vs_main::wgsl_source())
        .emit()
}

/// Outline fragment module: constant color selected by the vertex stage.
pub fn wgsl_fragment_source() -> String {
    ShaderModule::new()
        .decl(wgsl_decl(OutlineFragmentInput::WGSL_SOURCE))
        .entry(fs_main::wgsl_source())
        .emit()
}
