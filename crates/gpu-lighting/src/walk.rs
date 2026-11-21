//! The lit fragment: the light walk plus the ambient, emissive and IBL terms.
//!
//! The deferred pass differs from forward PBR in what it *records*. The debug
//! views need the direct specular lobe and the diffuse remainder per light, and
//! the shadow view needs the minimum visibility over the light walk, so the
//! accumulators live here rather than being summed away inside the walk. They
//! are plain locals: a `&mut` record carried across the loop is outside the
//! translator's subset.
// qual:allow(srp, file_length=520) reason: "the walk, its two shadow-selector helpers and the decode frame read as one shader source; splitting them would move the accumulators away from the loop that feeds them"

use ornis_gpu_openpbr::layers::debug_view;
use ornis_gpu_openpbr::layers::evaluate_base_layer;
use ornis_gpu_openpbr::layers::evaluate_base_specular;
use ornis_gpu_openpbr::layers::evaluate_coat_darkening;
use ornis_gpu_openpbr::layers::evaluate_coat_layer;
use ornis_gpu_openpbr::layers::evaluate_emission;
use ornis_gpu_openpbr::layers::evaluate_fuzz_layer;
use ornis_gpu_openpbr::layers::evaluate_subsurface_layer;
use ornis_gpu_openpbr::layers::evaluate_transmission_layer;
use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::layers::sanitize_hdr;
use ornis_gpu_openpbr::layers::select_shading_debug;
use ornis_gpu_openpbr::math::add3;
use ornis_gpu_openpbr::math::evaluate_ibl;
use ornis_gpu_openpbr::math::evaluate_ibl_specular;
use ornis_gpu_openpbr::math::mul3;
use ornis_gpu_openpbr::math::scale3;
use ornis_gpu_openpbr::math::splat3;
use ornis_gpu_openpbr::math::sub3;
use ornis_gpu_openpbr::math::thin_film_modulation;
use ornis_gpu_openpbr::math::thin_film_weight_mix;
use ornis_gpu_openpbr::math::v3;
use ornis_gpu_pbr::CSM_ACTIVE_EDGE;
use ornis_gpu_pbr::Frame;
use ornis_gpu_pbr::LIGHT_DIRECTION_WORD;
use ornis_gpu_pbr::LIGHT_KIND_WORD;
use ornis_gpu_pbr::LIGHT_POSITION_WORD;
use ornis_gpu_pbr::LIGHT_RANGE_WORD;
use ornis_gpu_pbr::LIGHT_SELECTOR_SPLIT;
use ornis_gpu_pbr::LIGHT_SHADOW_LAYER_WORD;
use ornis_gpu_pbr::LightRef;
use ornis_gpu_pbr::Material;
use ornis_gpu_pbr::POINT_OR_SPOT_KIND_EDGE;
use ornis_gpu_pbr::SPOT_KIND_EDGE;
use ornis_gpu_pbr::csm_visibility;
use ornis_gpu_pbr::light_base;
use ornis_gpu_pbr::point_shadow;
use ornis_gpu_pbr::shaded_radiance;
use ornis_gpu_pbr::spot_shadow;
use ornis_gpu_shader::ComparisonSampler;
use ornis_gpu_shader::DepthCubeArray;
use ornis_gpu_shader::DepthTexture2dArray;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::TextureCube;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::cross_lane;
use ornis_gpu_shader::dot3;

use crate::layout::BRDF_LUT_BINDING;
use crate::layout::CLUSTER_INDICES_BINDING;
use crate::layout::CLUSTERED_LIGHTS_BINDING;
use crate::layout::CSM_TEX_BINDING;
use crate::layout::GROUP;
use crate::layout::IBL_SAMPLER_BINDING;
use crate::layout::IRRADIANCE_BINDING;
use crate::layout::LIGHTING_BINDING;
use crate::layout::PREFILTER_BINDING;
use crate::layout::SHADOW_CUBE_BINDING;
use crate::layout::SHADOW_SAMPLER_BINDING;
use crate::layout::SHADOW_TEX_BINDING;
use ornis_gpu_pbr::Lighting;

/// Smallest value a divisor or a cosine is allowed to take.
pub(crate) const EPS: f32 = 0.000001;
/// `geometry_params.z` at or above which the surface is unlit.
const UNLIT_THRESHOLD: f32 = 0.5;
/// `params.w` below which a light casts no shadow.
const NO_SHADOW_LAYER: f32 = 0.0;

/// `mix(a, b, t)` lane by lane.
#[inline(always)]
fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    v3(
        ornis_gpu_shader::mix(a.x, b.x, t),
        ornis_gpu_shader::mix(a.y, b.y, t),
        ornis_gpu_shader::mix(a.z, b.z, t),
    )
}

/// WGSL `max(a, b)` for two scalars.
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// WGSL `min(a, b)` for two scalars.
#[inline(always)]
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Micrometers per nanometer: the record stores micrometers, the modulation
/// wants nanometers.
const NANOMETERS_PER_MICROMETER: f32 = 1000.0;

/// The per-fragment frame the walk reuses: the two precomputed layer factors
/// and the floored `dot(n, v)`.
// qual:allow(test_quality, untested) reason: "the layer factors it reads come out of the material storage buffer, so the body has no host path"
// qual:allow(srp, max_parameters=8) reason: "the material record and the seven frame values the layers take; a struct would be `Frame` itself"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn frame_of(
    mat: &Material,
    world_pos: Vec3,
    cam_pos: Vec3,
    n: Vec3,
    v: Vec3,
    t: Vec3,
    b: Vec3,
    nov: f32,
) -> Frame {
    Frame {
        world_pos,
        cam_pos,
        n,
        v,
        t,
        b,
        nov,
        thin_film_mod: thin_film_weight_mix(
            mat.thin_film_weight,
            thin_film_modulation(
                nov,
                mat.thin_film_ior,
                mat.thin_film_thickness_um * NANOMETERS_PER_MICROMETER,
                1.0,
            ),
        ),
        coat_darken: evaluate_coat_darkening(
            mat.coat_weight,
            mat.coat_darkening,
            mat.coat_ior,
            mat.metalness,
            mat.base_color,
            mat.base_weight,
            mat.specular_weight,
            mat.subsurface_weight,
            mat.subsurface_color,
        ),
    }
}

/// Everything the walk reads that does not change between lights, minus the
/// binding tables.
pub(crate) struct Sample {
    /// Material record of the fragment.
    pub(crate) mat: Material,
    /// Per-fragment frame scalars.
    pub(crate) frame: Frame,
    /// Albedo alpha; below [`ALPHA_CUTOFF`](crate::ALPHA_CUTOFF) the fragment
    /// is discarded before the walk.
    pub(crate) alpha: f32,
    /// Ambient-occlusion factor, `1` unoccluded.
    pub(crate) ao: f32,
    /// Which `SHADING_DEBUG_*` view to return.
    pub(crate) debug_view: u32,
    /// Uniform lights every pixel evaluates.
    pub(crate) light_count: u32,
    /// First word of this pixel's cluster span.
    pub(crate) span_offset: u32,
    /// Words in this pixel's cluster span.
    pub(crate) span_count: u32,
}

/// Visibility of one light, for the debug shadow view.
///
/// The beauty path multiplies these into the radiance inside
/// [`shaded_radiance`]; the shadow view needs the factor itself, so the three
/// selectors run a second time — only under that view, which is why the branch
/// is on the outside rather than folded into the product.
// qual:allow(srp, max_parameters=8) reason: "the three shadow bindings, the lighting block, the light record and the frame the selectors read"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn light_visibility(
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    shadow_tex: &DepthTexture2dArray<GROUP, SHADOW_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    shadow_cube_tex: &DepthCubeArray<GROUP, SHADOW_CUBE_BINDING>,
    csm_tex: &DepthTexture2dArray<GROUP, CSM_TEX_BINDING>,
    lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    light: &LightRef,
    frame: &Frame,
) -> f32 {
    let kind = light.word::<CLUSTERED_LIGHTS_BINDING>(lights, LIGHT_KIND_WORD);
    let range = light.word::<CLUSTERED_LIGHTS_BINDING>(lights, LIGHT_RANGE_WORD);
    let layer = light.word::<CLUSTERED_LIGHTS_BINDING>(lights, LIGHT_SHADOW_LAYER_WORD);
    let is_point = step(POINT_OR_SPOT_KIND_EDGE, kind) * (1.0 - step(SPOT_KIND_EDGE, kind));
    let is_csm = (1.0 - step(POINT_OR_SPOT_KIND_EDGE, kind)) * step(CSM_ACTIVE_EDGE, range);
    if is_csm > LIGHT_SELECTOR_SPLIT {
        return csm_visibility::<LIGHTING_BINDING, CSM_TEX_BINDING, SHADOW_SAMPLER_BINDING>(
            lighting,
            csm_tex,
            shadow_sampler,
            frame.world_pos,
            frame.cam_pos,
        );
    }
    if layer < NO_SHADOW_LAYER {
        return 1.0;
    }
    if is_point > LIGHT_SELECTOR_SPLIT {
        return point_shadow::<SHADOW_CUBE_BINDING, SHADOW_SAMPLER_BINDING, CLUSTERED_LIGHTS_BINDING>(
            shadow_cube_tex,
            shadow_sampler,
            light,
            lights,
            &frame.world_pos,
        );
    }
    spot_shadow::<SHADOW_TEX_BINDING, SHADOW_SAMPLER_BINDING, CLUSTERED_LIGHTS_BINDING>(
        shadow_tex,
        shadow_sampler,
        light,
        lights,
        &frame.world_pos,
    )
}

/// `1.0` when `x >= edge`, `0.0` otherwise — the WGSL `step`.
#[inline(always)]
fn step(edge: f32, x: f32) -> f32 {
    if x < edge { 0.0 } else { 1.0 }
}

/// Shaded colour of one deferred fragment; the entry writes the material
/// opacity as alpha.
///
/// `alpha` is the albedo's alpha: below [`ALPHA_CUTOFF`](crate::ALPHA_CUTOFF)
/// the caller has already discarded. `ao` scales the ambient and IBL terms
/// only — direct light, emission and unlit sprites stay untouched.
// qual:allow(complexity, max_function_lines=300) reason: "one light walk plus the five additive terms and the debug picker it feeds; splitting the walk from its accumulators is what the debug views measure"
// qual:allow(srp, max_parameters=12) reason: "the fragment sample and one parameter per binding the walk reads; the entry's attributes above are the table"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn shade(
    sample: Sample,
    lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>,
    shadow_tex: &DepthTexture2dArray<GROUP, SHADOW_TEX_BINDING>,
    shadow_sampler: ComparisonSampler<GROUP, SHADOW_SAMPLER_BINDING>,
    shadow_cube_tex: &DepthCubeArray<GROUP, SHADOW_CUBE_BINDING>,
    prefilter_cube: &TextureCube<GROUP, PREFILTER_BINDING>,
    irradiance_cube: &TextureCube<GROUP, IRRADIANCE_BINDING>,
    brdf_lut: &Texture2d<GROUP, BRDF_LUT_BINDING>,
    ibl_sampler: Sampler<GROUP, IBL_SAMPLER_BINDING>,
    csm_tex: &DepthTexture2dArray<GROUP, CSM_TEX_BINDING>,
    clustered_lights: &Storage<u32, GROUP, CLUSTERED_LIGHTS_BINDING, ReadOnly>,
    cluster_light_indices: &Storage<u32, GROUP, CLUSTER_INDICES_BINDING, ReadOnly>,
) -> Vec3 {
    let mat = sample.mat;
    let frame = sample.frame;
    let n = frame.n;
    let v = frame.v;
    let total = sample.light_count + sample.span_count;
    let mut lo = splat3(0.0);
    let mut spec_acc = splat3(0.0);
    let mut diff_acc = splat3(0.0);
    let mut shadow_vis = 1.0;
    let mut k = 0;
    while k < total {
        let light = LightRef::at(light_base::<CLUSTER_INDICES_BINDING>(
            cluster_light_indices,
            k,
            sample.light_count,
            sample.span_offset,
        ));
        let radiance = shaded_radiance::<
            LIGHTING_BINDING,
            SHADOW_TEX_BINDING,
            SHADOW_CUBE_BINDING,
            CSM_TEX_BINDING,
            SHADOW_SAMPLER_BINDING,
            CLUSTERED_LIGHTS_BINDING,
        >(
            lighting,
            shadow_tex,
            shadow_cube_tex,
            csm_tex,
            shadow_sampler,
            &light,
            clustered_lights,
            &frame,
        );
        if sample.debug_view == debug_view::SHADOW {
            shadow_vis = min_f32(
                shadow_vis,
                light_visibility(
                    lighting,
                    shadow_tex,
                    shadow_sampler,
                    shadow_cube_tex,
                    csm_tex,
                    clustered_lights,
                    &light,
                    &frame,
                ),
            );
        }
        // Unit vector toward the light: the stored direction for a
        // directional or spot light, the direction to the centre for a point.
        let to_light = sub3(
            light.xyz::<CLUSTERED_LIGHTS_BINDING>(clustered_lights, LIGHT_POSITION_WORD),
            frame.world_pos,
        );
        let use_point = step(
            POINT_OR_SPOT_KIND_EDGE,
            light.word::<CLUSTERED_LIGHTS_BINDING>(clustered_lights, LIGHT_KIND_WORD),
        );
        let l = normalize3(mix3(
            normalize3(
                light.xyz::<CLUSTERED_LIGHTS_BINDING>(clustered_lights, LIGHT_DIRECTION_WORD),
            ),
            normalize3(to_light),
            use_point,
        ));
        let h = normalize3(add3(add3(v, l), v3(0.0, 0.0, EPS)));
        let nol = max_f32(dot3(n.x, n.y, n.z, l.x, l.y, l.z), EPS);
        if nol > EPS {
            let noh = max_f32(dot3(n.x, n.y, n.z, h.x, h.y, h.z), EPS);
            let voh = max_f32(dot3(v.x, v.y, v.z, h.x, h.y, h.z), EPS);
            let base_bsdf = mul3(
                evaluate_base_layer(
                    n,
                    v,
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
            if sample.debug_view != debug_view::BEAUTY {
                let spec_lobe = mul3(
                    evaluate_base_specular(
                        v,
                        l,
                        h,
                        frame.nov,
                        nol,
                        noh,
                        voh,
                        mat.base_weight,
                        mat.base_color,
                        mat.metalness,
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
                spec_acc = add3(spec_acc, mul3(scale3(spec_lobe, nol), radiance));
                diff_acc = add3(
                    diff_acc,
                    mul3(scale3(sub3(base_bsdf, spec_lobe), nol), radiance),
                );
            }
            let coat_bsdf = evaluate_coat_layer(
                n,
                v,
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
            let fuzz_bsdf = evaluate_fuzz_layer(
                n,
                v,
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
            let trans_bsdf = evaluate_transmission_layer(
                n,
                v,
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
            let ss_bsdf = evaluate_subsurface_layer(
                n,
                v,
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
            let layer = add3(
                add3(add3(base_bsdf, coat_bsdf), add3(fuzz_bsdf, trans_bsdf)),
                ss_bsdf,
            );
            lo = add3(lo, mul3(scale3(layer, nol), radiance));
        }
        k += 1;
    }
    let ambient_tint = mix3(
        mat.base_color,
        mul3(mat.base_color, splat3(mat.specular_weight)),
        mat.metalness,
    );
    let ambient = scale3(
        mul3(crate::layout::ambient_color(lighting), ambient_tint),
        sample.ao,
    );
    let emission = evaluate_emission(
        mat.emission_luminance,
        mat.emission_color,
        mat.coat_weight,
        mat.coat_color,
        frame.nov,
    );
    let f0 = ornis_gpu_pbr::specular_f0(&mat);
    let reflect_dir = ornis_gpu_pbr::reflect_dir(n, v);
    let max_mip = crate::layout::ibl_max_mip(lighting) * mat.specular_roughness;
    let prefiltered = v3(
        prefilter_cube.sample_level_lane::<GROUP, IBL_SAMPLER_BINDING, 0>(
            ibl_sampler,
            reflect_dir,
            max_mip,
        ),
        prefilter_cube.sample_level_lane::<GROUP, IBL_SAMPLER_BINDING, 1>(
            ibl_sampler,
            reflect_dir,
            max_mip,
        ),
        prefilter_cube.sample_level_lane::<GROUP, IBL_SAMPLER_BINDING, 2>(
            ibl_sampler,
            reflect_dir,
            max_mip,
        ),
    );
    let irradiance = v3(
        irradiance_cube.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, 0>(ibl_sampler, n),
        irradiance_cube.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, 1>(ibl_sampler, n),
        irradiance_cube.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, 2>(ibl_sampler, n),
    );
    let lut_scale = brdf_lut.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, 0>(
        ibl_sampler,
        frame.nov,
        mat.specular_roughness,
    );
    let lut_bias = brdf_lut.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, 1>(
        ibl_sampler,
        frame.nov,
        mat.specular_roughness,
    );
    let ibl_weight = crate::layout::ibl_weight(lighting);
    let ibl = scale3(
        evaluate_ibl(
            prefiltered,
            irradiance,
            lut_scale,
            lut_bias,
            f0,
            scale3(mat.base_color, 1.0 - mat.metalness),
            ibl_weight,
        ),
        sample.ao,
    );
    let ibl_spec = evaluate_ibl_specular(prefiltered, lut_scale, lut_bias, f0, ibl_weight);
    let full = add3(add3(add3(ambient, lo), emission), ibl);
    // Unlit sprites: the emission alone, no BRDF, light, shadow or IBL term.
    let raw = if mat.unlit >= UNLIT_THRESHOLD {
        emission
    } else {
        full
    };
    let beauty = sanitize_hdr(raw);
    let diffuse = add3(add3(diff_acc, ambient), sub3(ibl, ibl_spec));
    select_shading_debug(
        sample.debug_view,
        beauty,
        spec_acc,
        ibl_spec,
        diffuse,
        mat.specular_roughness,
        mat.metalness,
        shadow_vis,
        raw,
    )
}

/// `cross(n, t)` of the tangent frame the layers take.
///
/// The deferred pass has no tangent varying: the g-buffer normal is the only
/// axis it carries, so the frame is built from it and world up.
pub(crate) fn tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let up = v3(0.0, 1.0, 0.0);
    let t = normalize3(add3(
        v3(
            cross_lane::<0>(n.x, n.y, n.z, up.x, up.y, up.z),
            cross_lane::<1>(n.x, n.y, n.z, up.x, up.y, up.z),
            cross_lane::<2>(n.x, n.y, n.z, up.x, up.y, up.z),
        ),
        v3(0.0, 0.0, EPS),
    ));
    let b = v3(
        cross_lane::<0>(n.x, n.y, n.z, t.x, t.y, t.z),
        cross_lane::<1>(n.x, n.y, n.z, t.x, t.y, t.z),
        cross_lane::<2>(n.x, n.y, n.z, t.x, t.y, t.z),
    );
    (t, b)
}

#[cfg(test)]
mod tests {
    use ornis_gpu_shader::dot3;

    use super::tangent_frame;
    use super::v3;

    /// The frame is a frame: `t` and `b` are unit length and both stand off the
    /// normal, which is what the anisotropic layers rotate by.
    #[test]
    fn the_tangent_frame_is_orthonormal_against_the_normal() {
        let n = ornis_gpu_openpbr::layers::normalize3(v3(0.3, 0.9, -0.2));
        let (t, b) = tangent_frame(n);
        let near = |a: f32, b: f32| (a - b).abs() < 1e-5;
        assert!(near(dot3(t.x, t.y, t.z, t.x, t.y, t.z), 1.0));
        assert!(near(dot3(n.x, n.y, n.z, t.x, t.y, t.z), 0.0));
        assert!(near(dot3(n.x, n.y, n.z, b.x, b.y, b.z), 0.0));
        // The bitangent is not normalised: it carries `|n|`, which is one here.
        assert!(b.length() > 0.9);
    }

    /// Up-facing normals make `n × up` vanish, so the `EPS` lift in `t` is the
    /// only thing keeping the frame from collapsing to zero.
    #[test]
    fn a_normal_along_up_still_yields_a_usable_tangent() {
        let (t, _b) = tangent_frame(v3(0.0, 1.0, 0.0));
        assert!(t.length() > 0.9);
    }
}
