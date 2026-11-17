//! One punctual light: where its record lives, its direction and its radiance.
//!
//! The fragment's cell names `count` clustered lights; those come after the
//! `light_count` uniform lights, so the index list turns a loop counter into a
//! record offset and the shading body is the same for both families. Radiance
//! is where the three shadow selectors live: cascaded, spot/atlas and cube.

use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::math::scale3;
use ornis_gpu_openpbr::math::sub3;
use ornis_gpu_openpbr::math::v3;
use ornis_gpu_shader::ComparisonSampler;
use ornis_gpu_shader::DepthCubeArray;
use ornis_gpu_shader::DepthTexture2dArray;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::length3;
use ornis_gpu_shader::mix;
use ornis_gpu_shader::smoothstep;

use crate::direct::Frame;
use crate::layout::CLUSTER_INDICES_BINDING;
use crate::layout::CLUSTERED_LIGHTS_BINDING;
use crate::layout::CSM_TEX_BINDING;
use crate::layout::GROUP;
use crate::layout::LIGHT_COLOR_WORD;
use crate::layout::LIGHT_DIRECTION_WORD;
use crate::layout::LIGHT_INTENSITY_WORD;
use crate::layout::LIGHT_KIND_WORD;
use crate::layout::LIGHT_PARAMS_WORD;
use crate::layout::LIGHT_POSITION_WORD;
use crate::layout::LIGHT_SHADOW_LAYER_WORD;
use crate::layout::LIGHT_STRIDE;
use crate::layout::LIGHTING_BINDING;
use crate::layout::Lighting;
use crate::layout::SHADOW_CUBE_BINDING;
use crate::layout::SHADOW_SAMPLER_BINDING;
use crate::layout::SHADOW_TEX_BINDING;
use crate::layout::light_word;
use crate::shadows::csm_visibility;
use crate::shadows::point_shadow;
use crate::shadows::spot_shadow;

/// Smallest value a divisor or a cosine is allowed to take.
const EPS: f32 = 0.000001;
/// `kind` at or above which a light is point or spot.
const POINT_OR_SPOT_KIND_EDGE: f32 = 0.5;
/// `kind` at or above which a point light is a spot.
const SPOT_KIND_EDGE: f32 = 1.5;
/// Fraction of the range at which the cutoff starts falling off.
const RANGE_CUTOFF_INNER_FRACTION: f32 = 0.99;
/// `params.x` at or above which a directional light casts cascaded shadows.
const CSM_ACTIVE_EDGE: f32 = 0.5;
/// Selector threshold shared by the light-family switches.
const LIGHT_SELECTOR_SPLIT: f32 = 0.5;

/// `1.0` when `x >= edge`, `0.0` otherwise — the WGSL `step`.
#[inline(always)]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}

/// WGSL `max(a, b)` for two scalars.
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// `mix(a, b, t)` lane by lane.
#[inline(always)]
fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    v3(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t))
}

/// The light record the current iteration shades and where it lives.
pub(crate) struct LightRef {
    /// Word offset of the record inside `clustered_lights`.
    pub(crate) base: u32,
}

impl LightRef {
    /// `f32` word `word` of this record.
    #[inline(always)]
    pub(crate) fn word(
        &self,
        lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
        word: u32,
    ) -> f32 {
        light_word(lights, self.base, word)
    }

    /// Lanes `x`, `y`, `z` of the `vec4` at `word`.
    #[inline(always)]
    pub(crate) fn xyz(
        &self,
        lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
        word: u32,
    ) -> Vec3 {
        v3(
            self.word(lights, word),
            self.word(lights, word + 1),
            self.word(lights, word + 2),
        )
    }
}

/// Word offset of the record the `k`-th iteration shades.
///
/// The uniform array is indexed directly; the clustered tail goes through the
/// cell's list, which stores light indices rather than offsets.
#[inline(always)]
pub(crate) fn light_base(
    indices: &Storage<u32, GROUP, CLUSTER_INDICES_BINDING, ReadOnly>,
    k: u32,
    light_count: u32,
    span_offset: u32,
) -> u32 {
    if k >= light_count {
        indices.load(span_offset + (k - light_count)) * LIGHT_STRIDE
    } else {
        k * LIGHT_STRIDE
    }
}

/// Unit vector toward the light: the stored direction for a directional or spot
/// light, the direction to the light centre for a point light.
#[inline(always)]
pub(crate) fn light_dir(
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    light: &LightRef,
    frame: &Frame,
) -> Vec3 {
    let kind = light.word(lights, LIGHT_KIND_WORD);
    let to_light = sub3(light.xyz(lights, LIGHT_POSITION_WORD), frame.world_pos);
    let dist = length3(to_light.x, to_light.y, to_light.z);
    let to_point = scale3(to_light, 1.0 / max_f32(dist, EPS));
    let use_point = step(POINT_OR_SPOT_KIND_EDGE, kind);
    normalize3(mix3(
        normalize3(light.xyz(lights, LIGHT_DIRECTION_WORD)),
        to_point,
        use_point,
    ))
}

/// Radiance arriving from one light: colour × intensity, range cutoff, spot
/// cone, distance attenuation and shadow.
// quant:allow(srp, max_parameters=8) reason: "the two shadow families and the cascade block are three different bindings; bundling them would be a struct the translator has to keep in registers"
// qual:allow(srp, max_parameters=9) reason: "two shadow textures, the cascade block and the sampler they share, plus the light record and the frame position"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn shaded_radiance(
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    shadow_tex: &DepthTexture2dArray<GROUP, SHADOW_TEX_BINDING>,
    shadow_cube_tex: &DepthCubeArray<GROUP, SHADOW_CUBE_BINDING>,
    csm_tex: &DepthTexture2dArray<GROUP, CSM_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    light: &LightRef,
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    world_pos: &Vec3,
    cam_pos: &Vec3,
) -> Vec3 {
    let kind = light.word(lights, LIGHT_KIND_WORD);
    let to_light = sub3(light.xyz(lights, LIGHT_POSITION_WORD), *world_pos);
    let dist = length3(to_light.x, to_light.y, to_light.z);
    let use_point = step(POINT_OR_SPOT_KIND_EDGE, kind);
    let range = light.word(lights, LIGHT_PARAMS_WORD);
    let edge0 = range * RANGE_CUTOFF_INNER_FRACTION;
    let range_cut = mix(1.0, 1.0 - smoothstep(edge0, range, dist), use_point);
    let color = light.xyz(lights, LIGHT_COLOR_WORD);
    let intensity = light.word(lights, LIGHT_INTENSITY_WORD);
    let to_frag = normalize3(sub3(*world_pos, light.xyz(lights, LIGHT_POSITION_WORD)));
    let axis = normalize3(light.xyz(lights, LIGHT_DIRECTION_WORD));
    let cos_theta = dot3(axis.x, axis.y, axis.z, to_frag.x, to_frag.y, to_frag.z);
    let cone = mix(
        1.0,
        smoothstep(
            light.word(lights, LIGHT_PARAMS_WORD + 2),
            light.word(lights, LIGHT_PARAMS_WORD + 1),
            cos_theta,
        ),
        step(SPOT_KIND_EDGE, kind),
    );
    let attenuation = mix(1.0, 1.0 / max_f32(dist * dist, EPS), use_point);
    let mut radiance = scale3(
        scale3(scale3(color, intensity), attenuation * range_cut),
        cone,
    );
    let is_point = step(POINT_OR_SPOT_KIND_EDGE, kind) * (1.0 - step(SPOT_KIND_EDGE, kind));
    let is_csm = (1.0 - step(POINT_OR_SPOT_KIND_EDGE, kind)) * step(CSM_ACTIVE_EDGE, range);
    if is_csm > LIGHT_SELECTOR_SPLIT {
        radiance = scale3(
            radiance,
            csm_visibility(lighting, csm_tex, shadow_sampler, *world_pos, *cam_pos),
        );
    }
    if light.word(lights, LIGHT_SHADOW_LAYER_WORD) >= 0.0 {
        if is_point <= LIGHT_SELECTOR_SPLIT {
            radiance = scale3(
                radiance,
                spot_shadow(shadow_tex, shadow_sampler, light, lights, world_pos),
            );
        }
        if is_point > LIGHT_SELECTOR_SPLIT {
            radiance = scale3(
                radiance,
                point_shadow(shadow_cube_tex, shadow_sampler, light, lights, world_pos),
            );
        }
    }
    radiance
}
