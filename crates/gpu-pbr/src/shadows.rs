//! Cascaded shadow map lookup for the direct term.
//!
//! The fragment picks the cascade from its view depth, samples that cascade and
//! blends toward the next one inside a band at the split so the seam does not
//! pop. Everything it needs — the four view-projections, the splits and the
//! camera forward axis — is one nested struct in the lighting block, which is
//! why the reads are two field indices deep.

use ornis_gpu_shader::ComparisonSampler;
use ornis_gpu_shader::DepthTexture2dArray;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::mix;

use ornis_gpu_openpbr::math::sub3;
use ornis_gpu_shader::DepthCubeArray;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::abs;

use crate::layout::CASCADE_ARRAY_LEN;
use crate::layout::CLUSTERED_LIGHTS_BINDING;
use crate::layout::CSM_FIELD;
use crate::layout::CSM_SPLIT_FAR_FIELD;
use crate::layout::CSM_SPLITS_LO_FIELD;
use crate::layout::CSM_TEX_BINDING;
use crate::layout::CSM_VIEW_FWD_FIELD;
use crate::layout::CSM_VPS_FIELD;
use crate::layout::FIRST_COLUMN;
use crate::layout::FOURTH_COLUMN_WORD;
use crate::layout::GROUP;
use crate::layout::LAST_COLUMN;
use crate::layout::LIGHT_POSITION_WORD;
use crate::layout::LIGHT_RANGE_WORD;
use crate::layout::LIGHT_SHADOW_LAYER_WORD;
use crate::layout::LIGHT_SHADOW_VP_WORD;
use crate::layout::LIGHTING_BINDING;
use crate::layout::Lighting;
use crate::layout::MIDDLE_COLUMN;
use crate::layout::NDC_TO_UV_HALF;
use crate::layout::SECOND_COLUMN_WORD;
use crate::layout::SHADOW_CUBE_BINDING;
use crate::layout::SHADOW_SAMPLER_BINDING;
use crate::layout::SHADOW_TEX_BINDING;
use crate::layout::THIRD_COLUMN_WORD;
use crate::layout::W_LANE_U;
use crate::layout::X_LANE_U;
use crate::layout::Y_LANE_U;
use crate::layout::Z_LANE_U;
use crate::lights::LightRef;

/// Blend fraction of a split: the band over which the next cascade fades in.
const CSM_BLEND_FRACTION: f32 = 0.08;
/// Depth bias subtracted from the fragment's reference depth.
const SHADOW_REF_BIAS: f32 = 0.001;
/// Guard against a zero-width blend band.
const EPS: f32 = 0.000001;
/// Near plane of the point-light shadow cube.
const SHADOW_CUBE_NEAR: f32 = 0.1;
/// The last cascade has no successor to blend with.
const CSM_LAST_CASCADE: u32 = CASCADE_ARRAY_LEN as u32 - 1;

/// `1.0` when `x >= edge`, `0.0` otherwise — the WGSL `step`.
///
/// The reference spells the cascade pick as three `step`s over the splits and
/// sums them; a comparison is what `step` lowers to, and a split is never NaN.
#[inline(always)]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}

/// WGSL `max(a, b)` for two scalars.
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Visibility of a world position in the cascaded shadow atlas.
///
/// `0` fully shadowed, `1` fully lit, blended across the split band.
#[inline(always)]
pub(crate) fn csm_visibility(
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    csm_tex: &DepthTexture2dArray<GROUP, CSM_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    world_pos: Vec3,
    cam_pos: Vec3,
) -> f32 {
    let view_depth = dot3(
        world_pos.x - cam_pos.x,
        world_pos.y - cam_pos.y,
        world_pos.z - cam_pos.z,
        lighting.nested_vec4_lane::<CSM_FIELD, CSM_VIEW_FWD_FIELD, X_LANE_U>(),
        lighting.nested_vec4_lane::<CSM_FIELD, CSM_VIEW_FWD_FIELD, Y_LANE_U>(),
        lighting.nested_vec4_lane::<CSM_FIELD, CSM_VIEW_FWD_FIELD, Z_LANE_U>(),
    );
    let b0 = lighting.nested_vec4_lane::<CSM_FIELD, CSM_SPLITS_LO_FIELD, X_LANE_U>();
    let b1 = lighting.nested_vec4_lane::<CSM_FIELD, CSM_SPLITS_LO_FIELD, Y_LANE_U>();
    let b2 = lighting.nested_vec4_lane::<CSM_FIELD, CSM_SPLITS_LO_FIELD, Z_LANE_U>();
    let b3 = lighting.nested_vec4_lane::<CSM_FIELD, CSM_SPLITS_LO_FIELD, W_LANE_U>();
    let b4 = lighting.nested_field_f32::<CSM_FIELD, CSM_SPLIT_FAR_FIELD>();
    let past_b1 = step(b1, view_depth);
    let past_b2 = step(b2, view_depth);
    let past_b3 = step(b3, view_depth);
    let cascade = (past_b1 + past_b2 + past_b3) as u32;
    let upper = mix(mix(mix(b1, b2, past_b1), b3, past_b2), b4, past_b3);
    let lower = mix(mix(mix(b0, b1, past_b1), b2, past_b2), b3, past_b3);
    let mut vis = cascade_sample(lighting, csm_tex, shadow_sampler, world_pos, cascade);
    if cascade < CSM_LAST_CASCADE {
        let band = max_f32((upper - lower) * CSM_BLEND_FRACTION, EPS);
        let t = ((upper - view_depth) / band).clamp(0.0, 1.0);
        let vis1 = cascade_sample(lighting, csm_tex, shadow_sampler, world_pos, cascade + 1);
        vis = mix(vis1, vis, t);
    }
    vis
}

/// One cascade lookup: project, map to UV and compare against the stored depth.
///
/// The projection is read column by column from the nested array, which is the
/// only way a uniform matrix can be indexed: WGSL allows a dynamic index into a
/// fixed-size array, not into a matrix.
#[inline(always)]
fn cascade_sample(
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    csm_tex: &DepthTexture2dArray<GROUP, CSM_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    world_pos: Vec3,
    cascade: u32,
) -> f32 {
    let x = clip_lane::<X_LANE_U>(lighting, cascade, world_pos);
    let y = clip_lane::<Y_LANE_U>(lighting, cascade, world_pos);
    let z = clip_lane::<Z_LANE_U>(lighting, cascade, world_pos);
    let w = clip_lane::<W_LANE_U>(lighting, cascade, world_pos);
    let u = (x / w) * NDC_TO_UV_HALF + NDC_TO_UV_HALF;
    let v = NDC_TO_UV_HALF - (y / w) * NDC_TO_UV_HALF;
    csm_tex.sample_compare(shadow_sampler, u, v, cascade, (z / w) - SHADOW_REF_BIAS)
}

/// Lane `LANE` of `csm.vps[cascade] · (x, y, z, 1)`.
///
/// Word `LANE` of each of the four columns scaled by that column's coordinate:
/// reading the columns as rows would silently transpose the projection.
#[inline(always)]
fn clip_lane<const LANE: usize>(
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    cascade: u32,
    world_pos: Vec3,
) -> f32 {
    let c0 = lighting.nested_array_vec4_lane::<CSM_FIELD, CSM_VPS_FIELD, 0, LANE>(cascade);
    let c1 =
        lighting.nested_array_vec4_lane::<CSM_FIELD, CSM_VPS_FIELD, FIRST_COLUMN, LANE>(cascade);
    let c2 =
        lighting.nested_array_vec4_lane::<CSM_FIELD, CSM_VPS_FIELD, MIDDLE_COLUMN, LANE>(cascade);
    let c3 =
        lighting.nested_array_vec4_lane::<CSM_FIELD, CSM_VPS_FIELD, LAST_COLUMN, LANE>(cascade);
    c0 * world_pos.x + c1 * world_pos.y + c2 * world_pos.z + c3
}

/// Visibility in the spotlight/directional shadow atlas.
///
/// The fragment is projected with the light's view-projection and compared
/// against the stored depth at the layer `params.w` names.
#[inline(always)]
pub(crate) fn spot_shadow(
    shadow_tex: &DepthTexture2dArray<GROUP, SHADOW_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    light: &LightRef,
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    world_pos: &Vec3,
) -> f32 {
    let x = shadow_lane::<X_LANE_U>(light, lights, world_pos);
    let y = shadow_lane::<Y_LANE_U>(light, lights, world_pos);
    let z = shadow_lane::<Z_LANE_U>(light, lights, world_pos);
    let w = shadow_lane::<W_LANE_U>(light, lights, world_pos);
    let u = (x / w) * NDC_TO_UV_HALF + NDC_TO_UV_HALF;
    let v = NDC_TO_UV_HALF - (y / w) * NDC_TO_UV_HALF;
    let layer = light.word(lights, LIGHT_SHADOW_LAYER_WORD) as u32;
    shadow_tex.sample_compare(shadow_sampler, u, v, layer, (z / w) - SHADOW_REF_BIAS)
}

/// Lane `LANE` of `shadow_vp · (x, y, z, 1)` of the current light.
pub(crate) fn shadow_lane<const LANE: usize>(
    light: &LightRef,
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    world_pos: &Vec3,
) -> f32 {
    let c0 = light.word(lights, LIGHT_SHADOW_VP_WORD + LANE as u32);
    let c1 = light.word(
        lights,
        LIGHT_SHADOW_VP_WORD + SECOND_COLUMN_WORD + LANE as u32,
    );
    let c2 = light.word(
        lights,
        LIGHT_SHADOW_VP_WORD + THIRD_COLUMN_WORD + LANE as u32,
    );
    let c3 = light.word(
        lights,
        LIGHT_SHADOW_VP_WORD + FOURTH_COLUMN_WORD + LANE as u32,
    );
    c0 * world_pos.x + c1 * world_pos.y + c2 * world_pos.z + c3
}

/// Visibility in the point-light shadow cube.
///
/// A cube face stores a linear depth along the major axis, so the reference
/// depth is reconstructed from the light range rather than from a projection.
#[inline(always)]
pub(crate) fn point_shadow(
    shadow_cube_tex: &DepthCubeArray<GROUP, SHADOW_CUBE_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    light: &LightRef,
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    world_pos: &Vec3,
) -> f32 {
    let to_frag = sub3(*world_pos, light.xyz(lights, LIGHT_POSITION_WORD));
    let major = max_f32(max_f32(abs(to_frag.x), abs(to_frag.y)), abs(to_frag.z));
    let far = max_f32(light.word(lights, LIGHT_RANGE_WORD), 1.0);
    let denom = far - SHADOW_CUBE_NEAR;
    let reference = (far / denom) - (SHADOW_CUBE_NEAR * far) / (denom * major) - SHADOW_REF_BIAS;
    let layer = light.word(lights, LIGHT_SHADOW_LAYER_WORD) as u32;
    shadow_cube_tex.sample_compare(shadow_sampler, to_frag, layer, reference)
}
