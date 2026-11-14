//! Distance-fog pass on the stable Rust → LLVM IR → WGSL path.
//!
//! The pass reconstructs the world position of every covered pixel from the
//! hardware depth buffer and the camera's inverse view-projection, takes the
//! Euclidean distance to the eye, and blends the fog colour in by
//! `1 - exp(-density * distance)`. The result goes through the same ACES curve
//! the composite uses, because this pass replaces it on the swapchain.
//!
//! `xtask build-shaders --package ornis-gpu-fog --entry all` writes the
//! single-sample artifact; `--multisampled-textures 0:2` writes the one whose
//! g-buffer depth is a multisampled image (it has no resolve target, so it is
//! the only texture in the pass that changes).

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_quad::standard_quad;
use ornis_gpu_shader::DepthTexture2d;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::exp;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::length3;

/// Composite sampler binding (the HDR fetch).
const FOG_SAMPLER: u32 = 1;

/// `CameraUniform` field of the inverse view-projection's first row.
const INV_VIEW_PROJ: usize = 4;
/// `CameraUniform` field of the transposed eye position.
const CAMERA_POS: usize = 8;
/// Lane of the homogeneous `w` a projection matrix puts in the fourth row.
const W_LANE: usize = 3;
/// Lane of the `x` component of a clip-space position.
const X_LANE: usize = 0;
/// Lane of the `y` component of a clip-space position.
const Y_LANE: usize = 1;
/// Lane of the `z` component of a clip-space position.
const Z_LANE: usize = 2;
/// Offset of the last column in a `mat4x4` uniform: four `vec4` slots.
const LAST_COLUMN: usize = 3;
/// `FogUniform` field holding colour and density in one std140 slot.
const FOG_PARAMS: usize = 0;
/// Lane of that slot the density sits in.
const DENSITY_LANE: usize = 3;

/// ACES fit coefficients (`x * (a·x + b) / (x · (c·x + d) + e)`).
const ACES_FIT_A: f32 = 2.51;
/// See [`ACES_FIT_A`].
const ACES_FIT_B: f32 = 0.03;
/// See [`ACES_FIT_A`].
const ACES_FIT_C: f32 = 2.43;
/// See [`ACES_FIT_A`].
const ACES_FIT_D: f32 = 0.59;
/// See [`ACES_FIT_A`].
const ACES_FIT_E: f32 = 0.14;

/// Frame camera: two view-projection matrices, the eye position, the previous
/// frame's matrix.
///
/// Only the inverse view-projection and the eye position are read here; the
/// tags still describe the whole host struct so the layout matches byte for
/// byte (thirteen `vec4` slots).
pub struct Camera;

impl GpuStruct for Camera {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
    ];
}

/// Fog colour and density, packed into the first std140 slot.
pub struct FogParams;

impl GpuStruct for FogParams {
    const ID: u32 = 1;
    const FIELD_TAGS: &'static [u8] = &[ornis_gpu_shader::TYPE_TAG_VEC4];
}

/// One lane of the ACES fit.
#[inline(always)]
fn aces_lane(value: f32) -> f32 {
    value * (ACES_FIT_A * value + ACES_FIT_B)
        / (value * (ACES_FIT_C * value + ACES_FIT_D) + ACES_FIT_E)
}

/// One lane of `inv_view_proj * (x, y, z, 1)`.
///
/// The host matrix is column-major, and so is a WGSL `mat4x4`: uniform slots
/// 4..8 hold the four *columns*, so lane `LANE` of the product is
/// `c0[LANE]·x + c1[LANE]·y + c2[LANE]·z + c3[LANE]`. Reading the slots as rows
/// would silently transpose the matrix — the translation would land in the
/// wrong place and the fog would still look plausible.
#[inline(always)]
fn view_lane<const LANE: usize>(camera: &Uniform<Camera, 0, 3>, x: f32, y: f32, z: f32) -> f32 {
    camera.vec4_lane::<{ INV_VIEW_PROJ }, { LANE }>() * x
        + camera.vec4_lane::<{ INV_VIEW_PROJ + Y_LANE }, { LANE }>() * y
        + camera.vec4_lane::<{ INV_VIEW_PROJ + Z_LANE }, { LANE }>() * z
        + camera.vec4_lane::<{ INV_VIEW_PROJ + LAST_COLUMN }, { LANE }>()
}

/// Fog vertex stage: the quad, straight from `vertex_index`.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    standard_quad();
}

/// Fog fragment stage: HDR + depth → world position → exponential mix → ACES.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] hdr_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 1)] fog_sampler: Sampler,
    #[gpu(depth_texture_2d, group = 0, binding = 2)] depth_tex: DepthTexture2d,
    #[gpu(uniform, group = 0, binding = 3)] camera: Uniform<Camera>,
    #[gpu(uniform, group = 0, binding = 4)] fog_params: Uniform<FogParams>,
) {
    let u = fragment_input_lane::<0, 2, 0>();
    let v = fragment_input_lane::<0, 2, 1>();
    // Texel of the depth buffer under this pixel: `uv * dimensions`, truncated.
    let width = depth_tex.width() as f32;
    let height = depth_tex.height() as f32;
    let depth = depth_tex.load((u * width) as u32, (v * height) as u32);
    // NDC from the UV (y flips) and the depth, back to world space by `w`.
    let ndc_x = u * 2.0 - 1.0;
    let ndc_y = 1.0 - v * 2.0;
    let w = view_lane::<W_LANE>(&camera, ndc_x, ndc_y, depth);
    let world_x = view_lane::<X_LANE>(&camera, ndc_x, ndc_y, depth) / w;
    let world_y = view_lane::<Y_LANE>(&camera, ndc_x, ndc_y, depth) / w;
    let world_z = view_lane::<Z_LANE>(&camera, ndc_x, ndc_y, depth) / w;
    let dist = length3(
        camera.vec4_lane::<CAMERA_POS, X_LANE>() - world_x,
        camera.vec4_lane::<CAMERA_POS, Y_LANE>() - world_y,
        camera.vec4_lane::<CAMERA_POS, Z_LANE>() - world_z,
    );
    let density = fog_params.vec4_lane::<FOG_PARAMS, DENSITY_LANE>();
    let factor = 1.0 - exp(0.0 - density * dist);
    let fog_r = fog_params.vec4_lane::<FOG_PARAMS, X_LANE>();
    let fog_g = fog_params.vec4_lane::<FOG_PARAMS, Y_LANE>();
    let fog_b = fog_params.vec4_lane::<FOG_PARAMS, Z_LANE>();
    let hdr_r = hdr_tex.sample_lane::<0, { FOG_SAMPLER }, { X_LANE as u32 }>(fog_sampler, u, v);
    let hdr_g = hdr_tex.sample_lane::<0, { FOG_SAMPLER }, { Y_LANE as u32 }>(fog_sampler, u, v);
    let hdr_b = hdr_tex.sample_lane::<0, { FOG_SAMPLER }, { Z_LANE as u32 }>(fog_sampler, u, v);
    let r = aces_lane(hdr_r + (fog_r - hdr_r) * factor);
    let g = aces_lane(hdr_g + (fog_g - hdr_g) * factor);
    let b = aces_lane(hdr_b + (fog_b - hdr_b) * factor);
    fragment_store_at::<0>(r, g, b, 1.0);
}
