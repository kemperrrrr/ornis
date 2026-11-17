//! The punctual light walk: attenuation, shadowing and the layer sum per light.
//!
//! The fragment's cell names `count` clustered lights; those come after the
//! `light_count` uniform lights in one loop, because the shader has to treat
//! them identically and a second copy of the body would be a second place for
//! the shadow selectors to drift. The first `light_count` iterations read the
//! uniform array straight, the rest indirect through the cell's index list.

use ornis_gpu_openpbr::layers::evaluate_base_layer;
use ornis_gpu_openpbr::layers::evaluate_coat_layer;
use ornis_gpu_openpbr::layers::evaluate_fuzz_layer;
use ornis_gpu_openpbr::layers::evaluate_subsurface_layer;
use ornis_gpu_openpbr::layers::evaluate_transmission_layer;
use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::math::add3;
use ornis_gpu_openpbr::math::mul3;
use ornis_gpu_openpbr::math::scale3;
use ornis_gpu_openpbr::math::splat3;
use ornis_gpu_openpbr::math::v3;
use ornis_gpu_shader::ComparisonSampler;
use ornis_gpu_shader::DepthCubeArray;
use ornis_gpu_shader::DepthTexture2dArray;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::dot3;

use crate::layout::CLUSTER_INDICES_BINDING;
use crate::layout::CLUSTERED_LIGHTS_BINDING;
use crate::layout::CSM_TEX_BINDING;
use crate::layout::GROUP;
use crate::layout::LIGHTING_BINDING;
use crate::layout::Lighting;
use crate::layout::SHADOW_CUBE_BINDING;
use crate::layout::SHADOW_SAMPLER_BINDING;
use crate::layout::SHADOW_TEX_BINDING;
use crate::lights::LightRef;
use crate::lights::light_base;
use crate::lights::light_dir;
use crate::lights::shaded_radiance;
use crate::material::Material;

/// Smallest value a divisor or a cosine is allowed to take.
const EPS: f32 = 0.000001;

/// WGSL `max(a, b)` for two scalars.
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Everything the walk needs that does not change between lights.
pub(crate) struct Frame {
    /// Fragment position in world space.
    pub(crate) world_pos: Vec3,
    /// Eye position.
    pub(crate) cam_pos: Vec3,
    /// Shading normal.
    pub(crate) n: Vec3,
    /// Eye direction.
    pub(crate) v: Vec3,
    /// Tangent frame's first axis.
    pub(crate) t: Vec3,
    /// Tangent frame's second axis.
    pub(crate) b: Vec3,
    /// `dot(n, v)`, floored at `EPS`.
    pub(crate) nov: f32,
    /// Thin-film modulation of the base layer.
    pub(crate) thin_film_mod: Vec3,
    /// How far the coat darkens the layers under it.
    pub(crate) coat_darken: Vec3,
}

/// Sum of the direct layers over every light the fragment's cell names.
///
/// `no_l` is floored at `EPS` and a light whose cosine stays at the floor is
/// skipped: the layers would all return zero and the branch keeps the divide in
/// the transmission lobe from seeing a grazing angle.
// qual:allow(complexity, max_function_lines=130) reason: "one loop over the lights; splitting the body would split the shadow selectors from the radiance they scale"
// qual:allow(srp, max_parameters=12) reason: "the three bindings the shadow families read, the light tables and the frame scalars; a bundle would be a struct the translator keeps in registers"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn direct(
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    shadow_tex: &DepthTexture2dArray<GROUP, SHADOW_TEX_BINDING>,
    shadow_cube_tex: &DepthCubeArray<GROUP, SHADOW_CUBE_BINDING>,
    csm_tex: &DepthTexture2dArray<GROUP, CSM_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    light_indices: &Storage<u32, GROUP, CLUSTER_INDICES_BINDING, ReadOnly>,
    mat: &Material,
    frame: &Frame,
    light_count: u32,
    span_offset: u32,
    span_count: u32,
) -> Vec3 {
    let total = light_count + span_count;
    let mut lo = splat3(0.0);
    let mut k = 0;
    while k < total {
        let light = LightRef {
            base: light_base(light_indices, k, light_count, span_offset),
        };
        let radiance = shaded_radiance(
            lighting,
            shadow_tex,
            shadow_cube_tex,
            csm_tex,
            shadow_sampler,
            &light,
            lights,
            &frame.world_pos,
            &frame.cam_pos,
        );
        let l = light_dir(lights, &light, frame);
        let h = normalize3(add3(add3(frame.v, l), v3(0.0, 0.0, EPS)));
        let nol = max_f32(dot3(frame.n.x, frame.n.y, frame.n.z, l.x, l.y, l.z), EPS);
        let noh = max_f32(dot3(frame.n.x, frame.n.y, frame.n.z, h.x, h.y, h.z), EPS);
        let voh = max_f32(dot3(frame.v.x, frame.v.y, frame.v.z, h.x, h.y, h.z), EPS);
        if nol > EPS {
            let base = mul3(
                evaluate_base_layer(
                    frame.n,
                    frame.v,
                    l,
                    h,
                    frame.nov,
                    nol,
                    noh,
                    voh,
                    mat.base_weight,
                    mat.base_color,
                    mat.metalness,
                    mat.diffuse_roughness,
                    mat.specular_weight,
                    mat.specular_roughness,
                    mat.specular_ior,
                    mat.specular_anisotropy,
                    mat.specular_edge_tint,
                    frame.t,
                    frame.b,
                    frame.thin_film_mod,
                ),
                frame.coat_darken,
            );
            let coat = evaluate_coat_layer(
                frame.n,
                frame.v,
                l,
                h,
                frame.nov,
                nol,
                noh,
                voh,
                mat.coat_weight,
                mat.coat_roughness,
                mat.coat_anisotropy,
                mat.coat_darkening,
                mat.coat_ior,
                mat.coat_color,
                mat.metalness,
                mat.base_color,
                mat.base_weight,
                mat.specular_weight,
                mat.subsurface_weight,
                mat.subsurface_color,
                frame.t,
                frame.b,
            );
            let fuzz = evaluate_fuzz_layer(
                frame.n,
                frame.v,
                l,
                h,
                frame.nov,
                nol,
                noh,
                voh,
                mat.fuzz_weight,
                mat.fuzz_roughness,
                mat.fuzz_color,
            );
            let transmission = evaluate_transmission_layer(
                frame.n,
                frame.v,
                l,
                h,
                frame.nov,
                nol,
                noh,
                voh,
                mat.transmission_weight,
                mat.transmission_depth,
                mat.transmission_dispersion_scale,
                mat.transmission_dispersion_abbe,
                mat.transmission_color,
                mat.transmission_scatter,
                mat.transmission_scatter_anisotropy,
                mat.specular_ior,
                mat.specular_roughness,
                mat.specular_anisotropy,
                mat.thin_walled,
            );
            let subsurface = evaluate_subsurface_layer(
                frame.n,
                frame.v,
                l,
                frame.nov,
                nol,
                mat.subsurface_weight,
                mat.subsurface_radius,
                mat.subsurface_radius_scale_r,
                mat.subsurface_radius_scale_g,
                mat.subsurface_radius_scale_b,
                mat.subsurface_scatter_anisotropy,
                mat.subsurface_color,
            );
            let layer = add3(add3(add3(base, coat), add3(fuzz, transmission)), subsurface);
            lo = add3(lo, scale3(layer, radiance.x * nol));
        }
        k += 1;
    }
    lo
}
