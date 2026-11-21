//! The deferred pass's binding table and the uniform lanes it reads.
//!
//! Bindings are the deferred table's numbers, not the forward pass's: the
//! lighting block sits at 1 (forward: 3), the shadow atlas at 10 and the
//! clustered light storage at 20. Everything the walk shares with the forward
//! pass is therefore parameterised over bindings rather than hard-coded, and
//! this file is the only place the deferred numbers appear.
//!
//! Only the bindings the *code* names are constants — the entry's own
//! `#[gpu(..., binding = N)]` attributes carry the full table, because the
//! attribute is what the translator reads.

use ornis_gpu_openpbr::math::v3;
use ornis_gpu_pbr::CLUSTERED_FIELD;
use ornis_gpu_pbr::Camera;
use ornis_gpu_pbr::IBL_MAX_MIP_FIELD;
use ornis_gpu_pbr::IBL_WEIGHT_FIELD;
use ornis_gpu_pbr::LIGHT_COUNT_FIELD;
use ornis_gpu_pbr::Lighting;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;

/// Group every binding of the pass lives in.
pub const GROUP: u32 = ornis_gpu_pbr::GROUP;
/// Binding of the frame camera block.
pub const CAMERA_BINDING: u32 = ornis_gpu_pbr::CAMERA_BINDING;
/// Binding of the lights, cascades and ambient block.
pub const LIGHTING_BINDING: u32 = 1;
/// Binding of the sampler every g-buffer float layer shares.
pub const LIGHTING_SAMPLER_BINDING: u32 = 9;
/// Binding of the spot/CSM shadow atlas.
pub const SHADOW_TEX_BINDING: u32 = 10;
/// Binding of the comparison sampler both shadow families share.
pub const SHADOW_SAMPLER_BINDING: u32 = 11;
/// Binding of the point-light shadow cube pool.
pub const SHADOW_CUBE_BINDING: u32 = 12;
/// Binding of the prefiltered environment map.
pub const PREFILTER_BINDING: u32 = 13;
/// Binding of the diffuse irradiance map.
pub const IRRADIANCE_BINDING: u32 = 14;
/// Binding of the split-sum BRDF lookup table.
pub const BRDF_LUT_BINDING: u32 = 15;
/// Binding of the IBL sampler.
pub const IBL_SAMPLER_BINDING: u32 = 16;
/// Binding of the cascaded-shadow-map atlas.
pub const CSM_TEX_BINDING: u32 = 18;
/// Binding of the cluster counters.
pub const CLUSTER_PARAMS_BINDING: u32 = 19;
/// Binding of the per-frame light records.
pub const CLUSTERED_LIGHTS_BINDING: u32 = 20;
/// Binding of the per-cluster light index list.
pub const CLUSTER_INDICES_BINDING: u32 = 21;

/// Lane of `x` as a `usize` field index.
pub const X_LANE_U: usize = 0;
/// Lane of `y` as a `usize` field index.
pub const Y_LANE_U: usize = 1;
/// Lane of `z` as a `usize` field index.
pub const Z_LANE_U: usize = 2;
/// Lane of `w` as a `usize` field index.
pub const W_LANE_U: usize = 3;

/// `Camera::camera_pos.xyz`.
#[inline(always)]
pub(crate) fn camera_position(camera: &Uniform<Camera, GROUP, CAMERA_BINDING>) -> Vec3 {
    v3(
        camera.vec4_lane::<{ ornis_gpu_pbr::CAMERA_POS }, X_LANE_U>(),
        camera.vec4_lane::<{ ornis_gpu_pbr::CAMERA_POS }, Y_LANE_U>(),
        camera.vec4_lane::<{ ornis_gpu_pbr::CAMERA_POS }, Z_LANE_U>(),
    )
}

/// `Lighting::ibl_weight`.
#[inline(always)]
pub(crate) fn ibl_weight(lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>) -> f32 {
    lighting.field_f32::<IBL_WEIGHT_FIELD>()
}

/// `Lighting::ibl_max_mip`.
#[inline(always)]
pub(crate) fn ibl_max_mip(lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>) -> f32 {
    lighting.field_f32::<IBL_MAX_MIP_FIELD>()
}

/// `Lighting::ambient_color.xyz`.
#[inline(always)]
pub(crate) fn ambient_color(lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>) -> Vec3 {
    v3(
        lighting.vec4_lane::<0, X_LANE_U>(),
        lighting.vec4_lane::<0, Y_LANE_U>(),
        lighting.vec4_lane::<0, Z_LANE_U>(),
    )
}

/// `Lighting::light_count`.
#[inline(always)]
pub(crate) fn light_count(lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>) -> u32 {
    lighting.field_u32::<LIGHT_COUNT_FIELD>()
}

/// `ClusterParams::clustered`: `1` while clustered lighting is on.
#[inline(always)]
pub(crate) fn clustered(
    cluster_params: &Uniform<ornis_gpu_pbr::ClusterParams, GROUP, CLUSTER_PARAMS_BINDING>,
) -> u32 {
    cluster_params.field_u32::<CLUSTERED_FIELD>()
}

/// Rebuild the world position of a deferred texel from its depth.
///
/// The g-buffer stores an octahedral normal rather than a position, so the
/// position comes back through the inverse view-projection: `uv` and depth
/// give an NDC point, and the matrix undoes the perspective divide.
///
/// The camera block is read at [`CAMERA_BINDING`], which the forward pass also
/// uses, so the lane indices are `gpu-pbr`'s.
// qual:allow(test_quality, untested) reason: "no host path: every uniform read panics off wasm32, so a unit test cannot build the camera block; the translated body is covered by the artifact and naga-validation tests in render"
#[inline(always)]
pub(crate) fn reconstruct_world_pos(
    camera: &Uniform<ornis_gpu_pbr::Camera, GROUP, CAMERA_BINDING>,
    u: f32,
    v: f32,
    depth: f32,
) -> Vec3 {
    let ndc_x = u * 2.0 - 1.0;
    let ndc_y = 1.0 - v * 2.0;
    const C0: usize = ornis_gpu_pbr::CAMERA_INV_VIEW_PROJ;
    const C1: usize = C0 + 1;
    const C2: usize = C0 + 2;
    const C3: usize = C0 + 3;
    let w = mat_lane::<C0, C1, C2, C3, W_LANE_U>(camera, ndc_x, ndc_y, depth);
    let x = mat_lane::<C0, C1, C2, C3, X_LANE_U>(camera, ndc_x, ndc_y, depth);
    let y = mat_lane::<C0, C1, C2, C3, Y_LANE_U>(camera, ndc_x, ndc_y, depth);
    let z = mat_lane::<C0, C1, C2, C3, Z_LANE_U>(camera, ndc_x, ndc_y, depth);
    v3(x / w, y / w, z / w)
}

/// Lane `LANE` of `m · (x, y, z, 1)` for the matrix whose first column is at
/// field `COL`.
///
/// The matrix arrives column-major, so each column is scaled by its own
/// coordinate — reading the columns as rows would transpose the inverse and
/// fold the whole g-buffer into the near plane.
// qual:allow(test_quality, untested) reason: "reads four camera lanes, which off wasm32 is the diverging SDK stub"
#[inline(always)]
fn mat_lane<
    const C0: usize,
    const C1: usize,
    const C2: usize,
    const C3: usize,
    const LANE: usize,
>(
    camera: &Uniform<Camera, GROUP, CAMERA_BINDING>,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    let c0 = camera.vec4_lane::<C0, LANE>();
    let c1 = camera.vec4_lane::<C1, LANE>();
    let c2 = camera.vec4_lane::<C2, LANE>();
    let c3 = camera.vec4_lane::<C3, LANE>();
    c0 * x + c1 * y + c2 * z + c3
}
