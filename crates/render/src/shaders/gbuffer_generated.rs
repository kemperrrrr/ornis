//! G-buffer shader generated from Rust (Render path 2).
//!
//! Canonical source is the Rust code in this module: the instance-transform
//! vertex boilerplate and the 5-MRT fragment skeleton live here as Rust
//! strings, and the `octahedral_encode` kernel is spliced in from
//! [`crate::shaders::math`] (single source of truth via `#[kernel]`).
//! The handwritten `shaders/wgsl/gbuffer_vertex.wgsl` and
//! `shaders/wgsl/gbuffer_fragment.wgsl` remain as references; the
//! `gbuffer_generated_parity_with_legacy_assembly` test pins this module
//! byte-identical to them.
//!
//! Note: the vertex boilerplate is byte-identical to the forward-PBR vertex
//! (`shaders/wgsl/pbr_vertex.wgsl`); both passes share the same instance
//! transform. The PBR migration (`pbr_generated`) reuses
//! [`wgsl_vertex_source`] rather than duplicating it.

use super::OPENPBR_MATERIAL_DECL;
use super::interface::{
    GbufferFragmentInput, GbufferOutput, GbufferVertexInput, GbufferVertexOutput,
};
use super::wgsl_decl;
use crate::renderer::{CameraUniform, PerObjectGpu};
use crate::shaders::math::octahedral_encode;

/// G-buffer vertex shader: instance transforms + world position.
///
/// Assembled from the derived `Camera`/`PerObject` layouts plus the vertex
/// body; entry point name `vs_main` is kept for compatibility with
/// `create_gbuffer_pass` / `create_pbr_pass`. Byte-identical to
/// `shaders/wgsl/gbuffer_vertex.wgsl` except the dropped `_padding` line
/// (`#[wgsl(skip)]` pads are not shader-visible).
///
/// Note: unlike the fragment skeletons, this legacy file terminates structs
/// with bare `}` (no `;`), so the layouts splice raw here instead of via
/// [`wgsl_decl`](super::wgsl_decl).
pub fn wgsl_vertex_source() -> String {
    format!(
        "\n{cam}\n{per}\n{bindings}\n{vin}\n{vout}\n{body}",
        cam = CameraUniform::WGSL_SOURCE,
        per = PerObjectGpu::WGSL_SOURCE,
        bindings = WGSL_VERTEX_BINDINGS,
        vin = wgsl_decl(GbufferVertexInput::WGSL_SOURCE),
        vout = wgsl_decl(GbufferVertexOutput::WGSL_SOURCE),
        body = WGSL_VERTEX_BODY,
    )
}

/// G-buffer fragment shader: 5-MRT packing, splicing the octahedral
/// normal-encoding kernel via `wgsl_source()`.
///
/// Assembled from the shared [`OPENPBR_MATERIAL_DECL`] plus the fragment body
/// and kernel; entry point `fs_main` is kept. Byte-identical to the legacy
/// `shaders::gbuffer_fragment()`.
pub fn wgsl_source() -> String {
    format!(
        "{mat}\n{bindings}\n{fin}\n{gout}\n{body}\n{kernel}",
        mat = OPENPBR_MATERIAL_DECL,
        bindings = WGSL_FRAGMENT_BINDINGS,
        fin = wgsl_decl(GbufferFragmentInput::WGSL_SOURCE),
        gout = wgsl_decl(GbufferOutput::WGSL_SOURCE),
        body = WGSL_FRAGMENT_BODY,
        kernel = octahedral_encode::wgsl_source()
    )
}

/// Static view for naga validation in tests.
pub fn wgsl_source_static() -> String {
    wgsl_source()
}

const WGSL_VERTEX_BINDINGS: &str = r#"@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> per_objects: array<PerObject>;
"#;

const WGSL_VERTEX_BODY: &str = r#"@vertex
fn vs_main(
    input: VertexInput,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    let obj = per_objects[instance];
    let world_pos = obj.model * vec4<f32>(input.position, 1.0);
    var world_normal = (obj.normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    world_normal = normalize(world_normal);
    var world_tangent = (obj.normal_matrix * vec4<f32>(input.tangent, 0.0)).xyz;
    world_tangent = normalize(world_tangent);

    var output: VertexOutput;
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.world_tangent = world_tangent;
    output.material_index = obj.material_index;
    return output;
}
"#;

const WGSL_FRAGMENT_BINDINGS: &str = r#"@group(0) @binding(2) var<storage, read> materials: array<OpenPBRMaterial>;
"#;

const WGSL_FRAGMENT_BODY: &str = r#"@fragment
fn fs_main(input: FragmentInput) -> GBufferOutput {
    let mat = materials[input.material_index];

    let N = normalize(input.world_normal);
    let world_pos = input.world_position;
    let base_color = mat.base_color.rgb;
    let opacity = mat.base_color.a;

    let normal_enc = octahedral_encode(N);

    let material_id = input.material_index;

    let world_pos_enc = vec2<f32>(world_pos.x, world_pos.y);

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

    let mat_params = vec4<f32>(roughness, metalness, specular_ior, coat_weight);

    var output: GBufferOutput;
    output.albedo = vec4<f32>(base_color, opacity);
    output.normal = normal_enc;
    output.material_id = material_id;
    output.world_pos = world_pos_enc;
    output.mat_params = mat_params;
    return output;
}
"#;

#[cfg(test)]
mod tests {
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

    #[test]
    fn gbuffer_generated_parity_with_legacy_assembly() {
        // The only admitted difference: the `_padding` line is gone — skipped
        // padding is not shader-visible (`#[wgsl(skip)]`).
        let legacy_vertex =
            include_str!("wgsl/gbuffer_vertex.wgsl").replace("    _padding: u32,\n", "");
        assert_eq!(wgsl_vertex_source(), legacy_vertex);
        let legacy_fragment = format!(
            "{}\n{}",
            include_str!("wgsl/gbuffer_fragment.wgsl"),
            octahedral_encode::wgsl_source()
        );
        assert_eq!(wgsl_source(), legacy_fragment);
    }

    #[test]
    fn gbuffer_vertex_shared_with_pbr_vertex() {
        let pbr_vertex = include_str!("wgsl/pbr_vertex.wgsl").replace("    _padding: u32,\n", "");
        assert_eq!(wgsl_vertex_source(), pbr_vertex);
    }
}
