//! Fragment stage: one material record in, the five lighting targets and the
//! two temporal ones out.
//!
//! The encoded normal is the octahedral projection of the interpolated
//! world-space normal; the motion vector and the quantized previous depth come
//! from projecting the current and the previous world position through their
//! own matrices. A pixel without both projections on screen, or without a valid
//! previous position, keeps the `PREVIOUS_DEPTH_INVALID` marker — the temporal
//! resolve reads that as "no history".

use crate::layout::{
    ALBEDO_TARGET, BASE_COLOR_WORD, BASE_PARAMS_WORD, BLUE_LANE, COAT_PARAMS_WORD, Camera,
    DEPTH_ROUNDING_BIAS, DIRECTION, HOMOGENEOUS, MAT_PARAMS_TARGET, MATERIAL_ID_TARGET,
    MATERIAL_INDEX, MATERIAL_STRIDE, METALNESS_LANE, MOTION_TARGET, MOTION_VALID_THRESHOLD,
    NDC_TO_UV_HALF, NORMAL_TARGET, OPACITY_LANE, PREVIOUS_DEPTH_INVALID, PREVIOUS_DEPTH_MAX,
    PREVIOUS_DEPTH_TARGET, PREVIOUS_WORLD_POSITION, ROUGHNESS_LANE, SPECULAR_IOR_LANE,
    SPECULAR_PARAMS_WORD, VEC3_LANES, VEC4_LANES, W_LANE, WORLD_NORMAL, WORLD_POS_TARGET,
    WORLD_POSITION, X_LANE, Y_LANE, Z_LANE, previous_view_lane, view_lane,
};
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_input_u32;
use ornis_gpu_shader::fragment_store_2_at;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::fragment_store_u32_at;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::sign;

/// Lane `LANE` of the world-space position the vertex stage interpolated.
#[inline(always)]
fn world_lane<const LANE: u32>() -> f32 {
    fragment_input_lane::<{ WORLD_POSITION }, { VEC3_LANES }, LANE>()
}

/// Lane `LANE` of the previous-frame world position.
#[inline(always)]
fn previous_lane<const LANE: u32>() -> f32 {
    fragment_input_lane::<{ PREVIOUS_WORLD_POSITION }, { VEC4_LANES }, LANE>()
}

/// Fold one octahedral coordinate into `[-1, 1]` without a zero sign.
#[inline(always)]
fn fold(value: f32) -> f32 {
    let folded = sign(value);
    if folded == DIRECTION {
        HOMOGENEOUS
    } else {
        folded
    }
}

/// G-buffer fragment stage: materials → albedo, normal, motion, previous depth.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
    #[gpu(storage_buffer, group = 0, binding = 2, read_only)] materials: Storage<f32>,
) {
    let material = fragment_input_u32::<{ MATERIAL_INDEX }>();
    let base = material * MATERIAL_STRIDE;
    let albedo_r = materials.load(base + BASE_COLOR_WORD + X_LANE as u32);
    let albedo_g = materials.load(base + BASE_COLOR_WORD + Y_LANE as u32);
    let albedo_b = materials.load(base + BASE_COLOR_WORD + BLUE_LANE);
    let opacity = materials.load(base + BASE_COLOR_WORD + OPACITY_LANE);
    let roughness = materials.load(base + SPECULAR_PARAMS_WORD + ROUGHNESS_LANE);
    let specular_ior = materials.load(base + SPECULAR_PARAMS_WORD + SPECULAR_IOR_LANE);
    let metalness = materials.load(base + BASE_PARAMS_WORD + METALNESS_LANE);
    let coat_weight = materials.load(base + COAT_PARAMS_WORD + X_LANE as u32);
    let normal_x = fragment_input_lane::<{ WORLD_NORMAL }, { VEC3_LANES }, { X_LANE as u32 }>();
    let normal_y = fragment_input_lane::<{ WORLD_NORMAL }, { VEC3_LANES }, { Y_LANE as u32 }>();
    let normal_z = fragment_input_lane::<{ WORLD_NORMAL }, { VEC3_LANES }, { Z_LANE as u32 }>();
    let world_x = world_lane::<{ X_LANE as u32 }>();
    let world_y = world_lane::<{ Y_LANE as u32 }>();
    let world_z = world_lane::<{ Z_LANE as u32 }>();
    let previous_x = previous_lane::<{ X_LANE as u32 }>();
    let previous_y = previous_lane::<{ Y_LANE as u32 }>();
    let previous_z = previous_lane::<{ Z_LANE as u32 }>();
    let clip_x = view_lane::<{ X_LANE }>(&camera, world_x, world_y, world_z, HOMOGENEOUS);
    let clip_y = view_lane::<{ Y_LANE }>(&camera, world_x, world_y, world_z, HOMOGENEOUS);
    let clip_w = view_lane::<{ W_LANE }>(&camera, world_x, world_y, world_z, HOMOGENEOUS);
    let previous_clip_x =
        previous_view_lane::<{ X_LANE }>(&camera, previous_x, previous_y, previous_z, HOMOGENEOUS);
    let previous_clip_y =
        previous_view_lane::<{ Y_LANE }>(&camera, previous_x, previous_y, previous_z, HOMOGENEOUS);
    let previous_clip_z =
        previous_view_lane::<{ Z_LANE }>(&camera, previous_x, previous_y, previous_z, HOMOGENEOUS);
    let previous_clip_w =
        previous_view_lane::<{ W_LANE }>(&camera, previous_x, previous_y, previous_z, HOMOGENEOUS);
    // Octahedral normal encoding: project onto the unit octahedron, then fold
    // the lower hemisphere onto it so two floats cover the whole sphere.
    let normal_scale = HOMOGENEOUS / (abs(normal_x) + abs(normal_y) + abs(normal_z));
    let octa_x = normal_x * normal_scale;
    let octa_y = normal_y * normal_scale;
    let mut encoded_x = octa_x;
    let mut encoded_y = octa_y;
    if normal_z < DIRECTION {
        encoded_x = (HOMOGENEOUS - abs(octa_y)) * fold(octa_x);
        encoded_y = (HOMOGENEOUS - abs(octa_x)) * fold(octa_y);
    }
    let mut motion_x = DIRECTION;
    let mut motion_y = DIRECTION;
    let mut previous_depth = PREVIOUS_DEPTH_INVALID;
    if clip_w > DIRECTION
        && previous_clip_w > DIRECTION
        && previous_lane::<{ W_LANE as u32 }>() > MOTION_VALID_THRESHOLD
    {
        let current_u = clip_x / clip_w * NDC_TO_UV_HALF + NDC_TO_UV_HALF;
        let current_v = NDC_TO_UV_HALF - clip_y / clip_w * NDC_TO_UV_HALF;
        let previous_u = previous_clip_x / previous_clip_w * NDC_TO_UV_HALF + NDC_TO_UV_HALF;
        let previous_v = NDC_TO_UV_HALF - previous_clip_y / previous_clip_w * NDC_TO_UV_HALF;
        motion_x = current_u - previous_u;
        motion_y = current_v - previous_v;
        let previous_ndc_z = previous_clip_z / previous_clip_w;
        previous_depth = (clamp(previous_ndc_z, DIRECTION, HOMOGENEOUS) * PREVIOUS_DEPTH_MAX
            + DEPTH_ROUNDING_BIAS) as u32;
    }
    fragment_store_at::<{ ALBEDO_TARGET }>(albedo_r, albedo_g, albedo_b, opacity);
    fragment_store_2_at::<{ NORMAL_TARGET }>(encoded_x, encoded_y);
    fragment_store_u32_at::<{ MATERIAL_ID_TARGET }>(material);
    fragment_store_2_at::<{ WORLD_POS_TARGET }>(world_x, world_y);
    fragment_store_at::<{ MAT_PARAMS_TARGET }>(roughness, metalness, specular_ior, coat_weight);
    fragment_store_2_at::<{ MOTION_TARGET }>(motion_x, motion_y);
    fragment_store_u32_at::<{ PREVIOUS_DEPTH_TARGET }>(previous_depth);
}
