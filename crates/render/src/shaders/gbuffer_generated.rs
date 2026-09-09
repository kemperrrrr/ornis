//! G-buffer shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the instance-transform
//! vertex boilerplate and the 5-MRT fragment skeleton live here as Rust
//! strings, and the `octahedral_encode` kernel is spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The former handwritten `shaders/wgsl/gbuffer_*.wgsl` sources were deleted
//! after the `#[stage]` translation; the `gbuffer_*_matches_legacy_shape`
//! tests pin the entry shapes.
//!
//! Note: the vertex boilerplate matches the forward-PBR vertex (same
//! transform. The PBR migration (`pbr_generated`) reuses
//! [`wgsl_vertex_source`] rather than duplicating it.

use super::interface::{
    GbufferFragmentInput as FragmentInput, GbufferOutput as GBufferOutput,
    GbufferVertexInput as VertexInput, GbufferVertexOutput as VertexOutput,
};
use super::{
    OPENPBR_WGSL_NAME, Resource, ResourceKind, openpbr_material_decl, resource_decls, wgsl_decl,
};
use crate::renderer::{CameraUniform, PerObjectGpu};
use crate::shaders::math::octahedral_encode;
use ornis_macros::stage;

/// G-buffer vertex resources as a context bundle (`ctx.per_objects`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct GbufferVertexContext {
    pub per_objects: Vec<PerObjectGpu>,
    pub camera: CameraUniform,
}

/// G-buffer vertex entry, translated by [`stage`](ornis_macros::stage):
/// instance transform + world-space varying. DSL-only — `per_objects` /
/// `camera` globals declared via `#[wgsl(global)]`.
#[stage(vertex, entry = "vs_main")]
fn gbuffer_vs_entry(
    input: VertexInput,
    #[wgsl(builtin = "instance_index")] instance: u32,
    #[wgsl(context)] ctx: GbufferVertexContext,
) -> VertexOutput {
    let obj = ctx.per_objects[instance];
    let world_pos = obj.model * Vec4::new(input.position, 1.0);
    let mut world_normal = normalize((obj.normal_matrix * Vec4::new(input.normal, 0.0)).xyz);
    let mut world_tangent = normalize((obj.normal_matrix * Vec4::new(input.tangent, 0.0)).xyz);
    let mut output: VertexOutput;
    output.clip_position = ctx.camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.world_tangent = world_tangent;
    output.material_index = obj.material_index;
    return output;
}

/// G-buffer vertex shader: instance transforms + world position.
///
/// Assembled from the derived `Camera`/`PerObject` layouts plus the vertex
/// body; entry point name `vs_main` is kept for compatibility with
/// `create_gbuffer_pass` / `create_pbr_pass`. Matches the former handwritten
/// vertex except the dropped `_padding` line
/// (`#[wgsl(skip)]` pads are not shader-visible).
///
/// Note: unlike the fragment skeletons, this stage terminates structs
/// with bare `}` (no `;`), so the layouts splice raw here instead of via
/// [`wgsl_decl`](super::wgsl_decl).
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{cam}\n{per}\n{bindings}\n{vin}\n{vout}\n{body}",
        cam = CameraUniform::WGSL_SOURCE,
        per = PerObjectGpu::WGSL_SOURCE,
        bindings = vertex_bindings(),
        vin = wgsl_decl(VertexInput::WGSL_SOURCE),
        vout = wgsl_decl(VertexOutput::WGSL_SOURCE),
        body = gbuffer_vs_entry::wgsl_source(),
    )
}

/// G-buffer fragment entry, translated by [`stage`](ornis_macros::stage):
/// 5-MRT packing. DSL-only — `materials` global declared via
/// `#[wgsl(global)]`.
#[stage(fragment, entry = "fs_main")]
fn gbuffer_fs_entry(
    input: FragmentInput,
    #[wgsl(global = "materials")] materials: [OpenPBRMaterial],
) -> GBufferOutput {
    let mat = materials[input.material_index];
    let n = normalize(input.world_normal);
    let world_pos = input.world_position;
    let base_color = mat.base_color.rgb;
    let opacity = mat.base_color.a;
    let normal_enc = octahedral_encode(n);
    let material_id = input.material_index;
    let world_pos_enc = Vec2::new(world_pos.x, world_pos.y);
    let roughness = mat.specular_params.y;
    let metalness = mat.base_params.z;
    let specular_weight = mat.specular_params.x;
    let specular_ior = mat.specular_params.z;
    let coat_weight = mat.coat_params.x;
    let coat_roughness = mat.coat_params.y;
    let subsurface_weight = mat.subsurface_params.x;
    let transmission_weight = mat.transmission_params.x;
    let fuzz_weight = mat.fuzz_params.x;
    let thin_film_weight = mat.thin_film_params.x;
    let emission_luminance = mat.emission_params.x;
    let mat_params = Vec4::new(roughness, metalness, specular_ior, coat_weight);
    let mut output: GBufferOutput;
    output.albedo = Vec4::new(base_color, opacity);
    output.normal = normal_enc;
    output.material_id = material_id;
    output.world_pos = world_pos_enc;
    output.mat_params = mat_params;
    return output;
}

/// G-buffer fragment shader: 5-MRT packing, splicing the octahedral
/// normal-encoding kernel via `wgsl_source()`.
///
/// Assembled from the shared [`openpbr_material_decl()`], the shared varyings,
/// the translated [`gbuffer_fs_entry`] body and kernel; entry point `fs_main`
/// is kept.
pub fn wgsl_source() -> String {
    format!(
        "{mat}\n{bindings}\n{fin}\n{gout}\n{body}\n{kernel}",
        mat = openpbr_material_decl(),
        bindings = fragment_bindings(),
        fin = wgsl_decl(FragmentInput::WGSL_SOURCE),
        gout = wgsl_decl(GBufferOutput::WGSL_SOURCE),
        body = gbuffer_fs_entry::wgsl_source(),
        kernel = octahedral_encode::wgsl_source()
    )
}

/// Static view for naga validation in tests.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

/// Resource layout of the g-buffer pass. Type names come from the Rust
/// side (`WGSL_NAME` / [`OPENPBR_WGSL_NAME`]) — never retyped.
pub const GBUFFER_RESOURCES: [Resource; 3] = [
    Resource {
        group: 0,
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
        visibility: wgpu::ShaderStages::FRAGMENT,
        name: "materials",
        kind: ResourceKind::StorageReadArray(OPENPBR_WGSL_NAME),
        min_size: None,
    },
];

/// Vertex resource bindings (0, 1) from the table.
fn vertex_bindings() -> String {
    resource_decls(&GBUFFER_RESOURCES, &[0, 1])
}

/// Fragment resource binding (2) from the table.
fn fragment_bindings() -> String {
    resource_decls(&GBUFFER_RESOURCES, &[2])
}

#[cfg(test)]
mod tests {
    use super::super::{bgl_entry, resource_decl};
    use super::*;

    fn assert_valid_wgsl(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} must parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    }

    #[test]
    fn gbuffer_generated_validates_with_naga() {
        assert_valid_wgsl("gbuffer_vertex", &wgsl_vertex_source());
        assert_valid_wgsl("gbuffer_fragment", &wgsl_source());
    }

    #[test]
    fn gbuffer_generated_contains_expected_bindings() {
        let vs = wgsl_vertex_source();
        assert!(vs.contains("@group(0) @binding(1) var<storage, read> per_objects"));
        assert!(vs.contains("fn vs_main("));
        let fs = wgsl_source();
        assert!(fs.contains("@group(0) @binding(2) var<storage, read> materials"));
        assert!(fs.contains("fn fs_main("));
        assert!(fs.contains("fn octahedral_encode"));
    }

    /// The translated vertex entry must keep the legacy shape: storage read,
    /// instance transform, var-out varying. (Byte-parity no longer applies —
    /// the generated entry is single-line.)
    #[test]
    fn gbuffer_vertex_entry_matches_legacy_shape() {
        let entry = gbuffer_vs_entry::wgsl_source();
        assert!(entry.starts_with(
            "@vertex\nfn vs_main(input: VertexInput, @builtin(instance_index) instance: u32)"
        ));
        assert!(entry.contains("-> VertexOutput"));
        assert!(entry.contains("let obj = per_objects[instance];"));
        assert!(entry.contains("output.clip_position = camera.view_proj * world_pos;"));
        assert!(entry.contains("output.material_index = obj.material_index;"));
        assert!(entry.contains("return output;"));
    }

    /// The translated fragment entry must keep the legacy shape: storage
    /// read, 5-MRT packing, same kernel call. (Byte-parity no longer
    /// applies — the generated entry is single-line.)
    #[test]
    fn gbuffer_fragment_entry_matches_legacy_shape() {
        let entry = gbuffer_fs_entry::wgsl_source();
        assert!(entry.starts_with("@fragment\nfn fs_main(input: FragmentInput)"));
        assert!(entry.contains("-> GBufferOutput"));
        assert!(entry.contains("let mat = materials[input.material_index];"));
        assert!(entry.contains("let normal_enc = octahedral_encode(n);"));
        assert!(entry.contains("output.mat_params = mat_params;"));
        assert!(entry.contains("return output;"));
    }

    /// Every table row's declaration appears in the assembled stages, and
    /// every row maps to a layout entry: shader and pipeline agree.
    #[test]
    fn gbuffer_resources_cover_stages_and_layout() {
        let src = wgsl_vertex_source() + &wgsl_source();
        assert_eq!(GBUFFER_RESOURCES.len(), 3);
        for r in GBUFFER_RESOURCES {
            assert!(src.contains(&resource_decl(&r)), "missing {}", r.name);
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
    }

    #[test]
    fn gbuffer_generated_parity_with_legacy_assembly() {
        // Layouts still splice the derived declarations.
        let src = wgsl_source();
        assert!(src.contains(openpbr_material_decl().as_str()));
        assert!(src.contains(&wgsl_decl(FragmentInput::WGSL_SOURCE)));
        assert!(src.contains(&wgsl_decl(GBufferOutput::WGSL_SOURCE)));
    }
}
