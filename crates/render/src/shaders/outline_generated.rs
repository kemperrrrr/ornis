//! Editor outline pass: translated from Rust through LLVM IR.
//!
//! The shader lives in `crates/gpu-outline`. The pass draws an inverted hull:
//! every triangle of an `outline_flags`-marked instance is expanded along its
//! object-space normal, front faces are culled and the back faces are
//! depth-tested without writing depth, so the ring survives only outside the
//! object's own silhouette.
//!
//! The two stages are compiled separately, because the vertex module reads four
//! buffers the fragment module never touches:
//! `xtask build-shaders --package ornis-gpu-outline --entry vs_main` writes
//! `outline_vertex_generated.wgsl` and `--entry fs_main` writes
//! `outline_fragment_generated.wgsl`. `--check` in CI catches drift.

use super::{Resource, ResourceKind};
use crate::renderer::{CameraUniform, PerObjectGpu};

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main";

/// Fragment entry point of the translated module.
pub const FS_ENTRY: &str = "fs_main";

/// Outline bindings: camera + per-object records + the compact instance index
/// list and its flags.
///
/// The translated vertex module declares the two storage bindings as
/// `array<u32>` and reinterprets the float words bit for bit, so the table stays
/// the single source of truth for the layout while the shader reads words.
pub const OUTLINE_RESOURCES: [Resource; 4] = [
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
        name: "outline_indices",
        kind: ResourceKind::StorageReadArray("u32"),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "outline_flags",
        kind: ResourceKind::StorageReadArray("u32"),
        min_size: None,
    },
];

/// Outline vertex module: the hull expansion and the flag check.
pub fn wgsl_vertex_source() -> String {
    include_str!("outline_vertex_generated.wgsl").to_string()
}

/// Outline fragment module: the constant colour the vertex stage selected.
pub fn wgsl_fragment_source() -> String {
    include_str!("outline_fragment_generated.wgsl").to_string()
}
