//! Phase D GPU skinning: joint-palette layout and the skinned vertex stage.
//!
//! The CPU path blends bind vertices on the host and uploads world-space
//! rows ([`crate::extraction`] skin entries). This module stages the
//! same final joint matrices (`model * inverse_bind`) as a storage palette
//! ([`JointPalette`]) and blends them in the vertex stage
//! ([`vs_main_skinned`], DSL-only — no handwritten WGSL outside generated
//! sources). Bind-pose vertices ride [`SkinnedVertexInput`] (joints as
//! `vec4<u32>`, weights as `vec4<f32>`); the varying is the shared
//! [`GbufferVertexOutput`](crate::shaders::interface::GbufferVertexOutput),
//! so the fragment stage is untouched.
//!
//! CPU-vs-GPU parity is approximate, never bit-identical: the shader uses
//! the joint linear part for normals while the CPU path uses the
//! inverse-transpose 3x3 (see
//! [`blend_vertex_reference`](ornis_animation::blend_vertex_reference)),
//! exact for rigid/uniform-scale joints only. Callers assert
//! [`ornis_animation::CPU_GPU_TOLERANCE`], not equality.
//!
//! Slice note: extraction stages the palette bytes alongside the
//! CPU-skinned vertices (renderer-compatible); the draw path binds the
//! palette and switches to [`wgsl_vertex_source_skinned`] in a follow-up —
//! the shader here already validates with naga and pins its shape.

use glam::Mat4;
use ornis_animation::{JointLimit, SkinError};
use ornis_macros::{WgslInterface, WgslStruct, stage};

use crate::renderer::{CameraUniform, PerObjectGpu};
use crate::shaders::interface::GbufferVertexOutput as VertexOutput;
use crate::shaders::{Resource, ResourceKind, ShaderModule, wgsl_decl};

/// One GPU palette joint: the final skinning matrix (`model * inverse_bind`).
///
/// The WGSL `SkinJoint` declaration is generated from this layout
/// ([`SkinJoint::WGSL_SOURCE`]); the field list is the single source of
/// truth for the buffer layout. `align(16)` keeps the nested-layout check
/// of [`JointPalette`] honest (a bare `mat4` member would align to 4).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "SkinJoint")]
pub struct SkinJoint {
    /// Final joint matrix (`model * inverse_bind`), column-major.
    pub matrix: [[f32; 4]; 4],
}

impl SkinJoint {
    /// Wraps one final joint matrix for upload.
    pub const fn from_matrix(matrix: [[f32; 4]; 4]) -> Self {
        Self { matrix }
    }
}

/// Whole joint palette: one [`SkinJoint`] per joint, zero-padded to the
/// [`JointLimit::GPU`] capacity (8 KiB).
///
/// The WGSL `JointPalette` declaration is generated from this layout
/// ([`JointPalette::WGSL_SOURCE`]). The vertex stage binds it as
/// `array<SkinJoint>` (see [`GBUFFER_SKINNED_RESOURCES`]); the struct form
/// here exists so the byte size and stride are compile-time gated through
/// the derive (`size_of` / `offset_of` assertions).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "JointPalette")]
pub struct JointPalette {
    /// One entry per joint (only the first `joint_count` are indexed).
    pub joints: [SkinJoint; 128],
}

/// Byte size of one staged [`JointPalette`] (128 joints × 64 bytes).
pub const PALETTE_BYTE_SIZE: usize = 8 * 1024;

/// Skinned vertex input: bind-pose attributes plus joint influences.
///
/// Same locations 0–3 as
/// [`GbufferVertexInput`](crate::shaders::interface::GbufferVertexInput)
/// (position, normal, uv, tangent), then joints (`vec4<u32>`) and weights
/// (`vec4<f32>`). The shader reads lanes unrolled (`.x/.y/.z/.w`), so no
/// dynamic vector indexing is needed — only the palette index is dynamic.
///
/// Declaration-only mirror (like `shaders::interface`): the WGSL derive is
/// the use; Rust never reads the fields.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "SkinnedVertexInput")]
pub(crate) struct SkinnedVertexInput {
    /// Bind-pose position.
    #[wgsl(location = 0)]
    pub position: [f32; 3],
    /// Bind-pose normal.
    #[wgsl(location = 1)]
    pub normal: [f32; 3],
    /// Texture coordinates.
    #[wgsl(location = 2)]
    pub uv: [f32; 2],
    /// Bind-pose tangent.
    #[wgsl(location = 3)]
    pub tangent: [f32; 3],
    /// Influencing joints (top-4, validated `< joint_count` at staging).
    #[wgsl(location = 4)]
    pub joints: [u32; 4],
    /// Influence weights (canonicalized at staging).
    #[wgsl(location = 5)]
    pub weights: [f32; 4],
}

/// Skinned vertex resources as a context bundle (`ctx.per_objects`, …).
#[allow(dead_code)]
#[derive(ornis_macros::ShaderContext)]
pub(crate) struct SkinnedGbufferContext {
    pub per_objects: Vec<PerObjectGpu>,
    pub camera: CameraUniform,
    pub palette: Vec<SkinJoint>,
}

/// Skinned g-buffer vertex entry, translated by [`stage`](ornis_macros::stage):
/// linear blend skinning over the joint palette, then the instance
/// transform and the shared world-space varying. DSL-only — `per_objects` /
/// `camera` / `palette` globals declared via the context bundle.
///
/// Normals (and tangents) blend through the joint linear part, not the
/// inverse-transpose the CPU path uses: exact for rigid/uniform-scale
/// joints, approximate otherwise (the documented parity допуск).
#[stage(vertex)]
fn vs_main_skinned(
    input: SkinnedVertexInput,
    instance_index: crate::shaders::InstanceIndex,
    ctx: Context<SkinnedGbufferContext>,
) -> VertexOutput {
    let obj = ctx.per_objects[instance_index];
    let joint0 = ctx.palette[input.joints.x];
    let joint1 = ctx.palette[input.joints.y];
    let joint2 = ctx.palette[input.joints.z];
    let joint3 = ctx.palette[input.joints.w];
    let weight0 = input.weights.x;
    let weight1 = input.weights.y;
    let weight2 = input.weights.z;
    let weight3 = input.weights.w;
    let skinned_pos = (joint0.matrix * Vec4::new(input.position, 1.0)).xyz * weight0
        + (joint1.matrix * Vec4::new(input.position, 1.0)).xyz * weight1
        + (joint2.matrix * Vec4::new(input.position, 1.0)).xyz * weight2
        + (joint3.matrix * Vec4::new(input.position, 1.0)).xyz * weight3;
    let skinned_nrm = normalize(
        (joint0.matrix * Vec4::new(input.normal, 0.0)).xyz * weight0
            + (joint1.matrix * Vec4::new(input.normal, 0.0)).xyz * weight1
            + (joint2.matrix * Vec4::new(input.normal, 0.0)).xyz * weight2
            + (joint3.matrix * Vec4::new(input.normal, 0.0)).xyz * weight3,
    );
    let skinned_tan = normalize(
        (joint0.matrix * Vec4::new(input.tangent, 0.0)).xyz * weight0
            + (joint1.matrix * Vec4::new(input.tangent, 0.0)).xyz * weight1
            + (joint2.matrix * Vec4::new(input.tangent, 0.0)).xyz * weight2
            + (joint3.matrix * Vec4::new(input.tangent, 0.0)).xyz * weight3,
    );
    let world_pos = obj.model * Vec4::new(skinned_pos, 1.0);
    let mut world_normal = normalize((obj.normal_matrix * Vec4::new(skinned_nrm, 0.0)).xyz);
    let mut world_tangent = normalize((obj.normal_matrix * Vec4::new(skinned_tan, 0.0)).xyz);
    let mut output: VertexOutput;
    output.clip_position = ctx.camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.uv = input.uv;
    output.world_tangent = world_tangent;
    output.material_index = obj.material_index;
    return output;
}

/// Skinned g-buffer vertex shader: palette blend + instance transform.
///
/// Assembled from the derived `Camera`/`PerObject`/`SkinJoint` layouts plus
/// the skinned entry body; the varying matches the classic vertex output,
/// so the fragment stage is shared. Entry point `vs_main_skinned` is new
/// (no legacy shape to preserve).
pub fn wgsl_vertex_source_skinned() -> String {
    ShaderModule::new()
        .decl(CameraUniform::WGSL_SOURCE)
        .decl(PerObjectGpu::WGSL_SOURCE)
        .decl(SkinJoint::WGSL_SOURCE)
        .resources(&GBUFFER_SKINNED_RESOURCES, &[0, 1, 3])
        .decl(wgsl_decl(SkinnedVertexInput::WGSL_SOURCE))
        .decl(wgsl_decl(VertexOutput::WGSL_SOURCE))
        .entry(vs_main_skinned::wgsl_source())
        .emit()
}

/// Resource layout of the skinned g-buffer vertex stage: the classic rows
/// plus the joint palette at binding 3. Type names come from the Rust side
/// (`WGSL_NAME`) — never retyped.
pub const GBUFFER_SKINNED_RESOURCES: [Resource; 4] = [
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
        kind: ResourceKind::StorageReadArray(crate::shaders::OPENPBR_WGSL_NAME),
        min_size: None,
    },
    Resource {
        group: 0,
        binding: 3,
        visibility: wgpu::ShaderStages::VERTEX,
        name: "palette",
        kind: ResourceKind::StorageReadArray(SkinJoint::WGSL_NAME),
        min_size: None,
    },
];

/// Stages final joint matrices as palette upload bytes: the first `count`
/// slots carry `palette` (column-major), the rest are zero (never indexed —
/// joint indices are validated `< count` before staging).
///
/// # Errors
///
/// Returns [`SkinError::EmptyPalette`] on zero matrices and
/// [`SkinError::PaletteOverflow`] past [`JointLimit::GPU`] — the caller
/// falls back to the CPU path, never a truncated palette.
pub fn joint_palette_bytes(palette: &[Mat4]) -> Result<Vec<u8>, SkinError> {
    if palette.is_empty() {
        return Err(SkinError::EmptyPalette);
    }
    let limit = JointLimit::GPU;
    if palette.len() > limit.index() {
        return Err(SkinError::PaletteOverflow {
            count: palette.len().min(u32::MAX as usize) as u32,
            limit: limit.get(),
        });
    }
    let mut joints: [SkinJoint; 128] = [bytemuck::Zeroable::zeroed(); 128];
    for (slot, matrix) in joints.iter_mut().zip(palette.iter()) {
        slot.matrix = matrix.to_cols_array_2d();
    }
    Ok(bytemuck::cast_slice(&joints).to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shaders::interface::GbufferVertexInput as ClassicVertexInput;
    use ornis_animation::{CPU_GPU_TOLERANCE as TOL, SkinningMode as Mode};
    use ornis_animation::{SkinningResources, blend_vertex_reference, skin_vertices};

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
    fn palette_layout_matches_wgsl() {
        // 128 joints × 64 bytes, first slot at zero; the derive gates the
        // offsets/sizes at compile time, this pins the totals for upload.
        assert_eq!(std::mem::size_of::<SkinJoint>(), 64);
        assert_eq!(std::mem::offset_of!(SkinJoint, matrix), 0);
        assert_eq!(std::mem::size_of::<JointPalette>(), PALETTE_BYTE_SIZE);
        assert_eq!(PALETTE_BYTE_SIZE, 8 * 1024);
        assert_eq!(std::mem::offset_of!(JointPalette, joints), 0);
        assert!(SkinJoint::WGSL_SOURCE.contains("matrix: mat4x4<f32>"));
        assert!(JointPalette::WGSL_SOURCE.contains("array<SkinJoint, 128>"));
    }

    #[test]
    fn skinned_input_pins_shared_locations() {
        // Locations 0–3 are the classic attributes (one pipeline, one
        // vertex-buffer prefix); joints/weights extend at 4–5.
        let src = SkinnedVertexInput::WGSL_SOURCE;
        assert!(src.contains("@location(0) position: vec3<f32>"));
        assert!(src.contains("@location(1) normal: vec3<f32>"));
        assert!(src.contains("@location(2) uv: vec2<f32>"));
        assert!(src.contains("@location(3) tangent: vec3<f32>"));
        assert!(src.contains("@location(4) joints: vec4<u32>"));
        assert!(src.contains("@location(5) weights: vec4<f32>"));
        // Same prefix spelling as the classic input (shared buffers).
        for line in [
            "@location(0) position: vec3<f32>",
            "@location(1) normal: vec3<f32>",
            "@location(2) uv: vec2<f32>",
            "@location(3) tangent: vec3<f32>",
        ] {
            assert!(
                ClassicVertexInput::WGSL_SOURCE.contains(line),
                "classic input drifted from {line}"
            );
        }
    }

    #[test]
    fn skinned_vertex_validates_with_naga() {
        assert_valid_wgsl("skinned_vertex", &wgsl_vertex_source_skinned());
    }

    #[test]
    fn skinned_entry_keeps_gpu_shape() {
        let entry = vs_main_skinned::wgsl_source();
        assert!(entry.starts_with(
            "@vertex\nfn vs_main_skinned(input: SkinnedVertexInput, @builtin(instance_index) instance_index: u32)"
        ));
        assert!(entry.contains("-> VertexOutput"));
        assert!(entry.contains("let joint0 = palette[input.joints.x];"));
        assert!(entry.contains("let joint3 = palette[input.joints.w];"));
        assert!(entry.contains("joint0.matrix * "));
        assert!(entry.contains("input.weights.x"));
        assert!(entry.contains("output.clip_position = camera.view_proj * world_pos;"));
        assert!(entry.contains("output.material_index = obj.material_index;"));
        assert!(entry.contains("return output;"));
    }

    #[test]
    fn skinned_resources_cover_stages_and_layout() {
        use crate::shaders::{bgl_entry, resource_decl};
        let src = wgsl_vertex_source_skinned();
        assert_eq!(GBUFFER_SKINNED_RESOURCES.len(), 4);
        for r in GBUFFER_SKINNED_RESOURCES {
            // The palette row is vertex-only; the materials row belongs to
            // the shared fragment stage (declared by its own assembly).
            if r.binding != 2 {
                assert!(src.contains(&resource_decl(&r)), "missing {}", r.name);
            }
            let e = bgl_entry(&r, false);
            assert_eq!((e.binding, e.visibility), (r.binding, r.visibility));
        }
        assert!(src.contains("@group(0) @binding(3) var<storage, read> palette: array<SkinJoint>"));
    }

    #[test]
    fn palette_bytes_stage_and_gate() {
        // Two rigid joints stage into an 8 KiB zero-padded upload.
        let palette = [
            Mat4::IDENTITY,
            Mat4::from_rotation_translation(
                glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                glam::Vec3::X,
            ),
        ];
        let bytes = joint_palette_bytes(&palette).expect("fits");
        assert_eq!(bytes.len(), PALETTE_BYTE_SIZE);
        let first: [[f32; 4]; 4] = bytemuck::cast_slice::<u8, [[f32; 4]; 4]>(&bytes[..64])[0];
        assert_eq!(first, Mat4::IDENTITY.to_cols_array_2d());
        assert!(bytes[128..].iter().all(|byte| *byte == 0));
        // Empty and over-limit never stage (typed errors, not strings).
        assert_eq!(joint_palette_bytes(&[]), Err(SkinError::EmptyPalette));
        assert!(matches!(
            joint_palette_bytes(&[Mat4::IDENTITY; 129]),
            Err(SkinError::PaletteOverflow { .. })
        ));
    }

    #[test]
    fn cpu_gpu_parity_for_rigid_joints() {
        // Rigid palette (`M1 = T(1,0,0)·Rz(90°)`, identity binds): the CPU
        // path and the shader-mirror reference agree within the documented
        // допуск — the parity gate for the staged palette.
        let rotation = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let palette = vec![
            Mat4::IDENTITY,
            Mat4::from_rotation_translation(rotation, glam::Vec3::X),
        ];
        let staged = SkinningResources::build(&palette, Mode::Gpu).expect("two joints stage");
        assert_eq!(staged.mode(), Mode::Gpu);
        let joints = vec![[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]];
        let weights = vec![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ];
        let positions = vec![[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let normals = vec![[0.0, 0.0, 1.0]; 3];
        let (cpu_positions, cpu_normals) = skin_vertices(
            staged.palette_matrices(),
            &joints,
            &weights,
            &positions,
            &normals,
        );
        for (index, joint) in joints.iter().enumerate() {
            let (ref_position, ref_normal) = blend_vertex_reference(
                staged.palette_matrices(),
                *joint,
                weights[index],
                positions[index],
                normals[index],
            );
            let position_drift = (glam::Vec3::from_array(ref_position)
                - glam::Vec3::from_array(cpu_positions[index]))
            .length();
            let normal_drift = (glam::Vec3::from_array(ref_normal)
                - glam::Vec3::from_array(cpu_normals[index]))
            .length();
            assert!(
                position_drift < TOL,
                "vertex {index} drifts {position_drift}"
            );
            assert!(normal_drift < TOL, "vertex {index} drifts {normal_drift}");
        }
        // Hand-computed spot checks (same pose as the acceptance test).
        assert!(
            (glam::Vec3::from_array(cpu_positions[0]) - glam::Vec3::new(1.0, 1.0, 0.0)).length()
                < TOL
        );
        assert!(
            (glam::Vec3::from_array(cpu_positions[2]) - glam::Vec3::new(1.0, 0.5, 0.0)).length()
                < TOL
        );
    }
}
