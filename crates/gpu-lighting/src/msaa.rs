//! The 4x deferred-lighting resolve.
//!
//! The g-buffer keeps depth, the material index and the octahedral normal per
//! sample, so this entry shades every covered sample of one texel and then
//! decides how to combine them: a pixel whose samples landed on one surface
//! takes the coverage-weighted (Karis) average, and a pixel whose samples
//! straddle a depth, normal or material-index edge keeps sample 0.
//!
//! It lives in its own module because it is a separate *build* of the same
//! crate: the multisampled binding types belong to the module, so
//! `cargo xtask build-shaders --entry fs_main_lighting_msaa
//! --multisampled-textures 0:4,0:5,0:8` emits this entry alone, while the 1x
//! artifact is the parent module's.

#![warn(missing_docs)]

use crate::ALPHA_CUTOFF;
use crate::layout;
use crate::walk::EPS;
use crate::walk::Sample;
use crate::walk::frame_of;
use crate::walk::max_f32;
use crate::walk::min_f32;
use crate::walk::mix3;
use crate::walk::shade;
use crate::walk::tangent_frame;
use layout::GROUP;
use layout::LIGHTING_SAMPLER_BINDING;
use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::math::add3;
use ornis_gpu_openpbr::math::octahedral_decode;
use ornis_gpu_openpbr::math::scale3;
use ornis_gpu_openpbr::math::sub3;
use ornis_gpu_openpbr::math::v3;
use ornis_gpu_pbr::CLUSTER_SPAN_WORDS;
use ornis_gpu_pbr::Camera;
use ornis_gpu_pbr::ClusterParams;
use ornis_gpu_pbr::DEBUG_VIEW_FIELD;
use ornis_gpu_pbr::Lighting;
use ornis_gpu_pbr::Material;
use ornis_gpu_shader::ComparisonSampler;
use ornis_gpu_shader::DepthCubeArray;
use ornis_gpu_shader::DepthTexture2d;
use ornis_gpu_shader::DepthTexture2dArray;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Texture2dU32;
use ornis_gpu_shader::TextureCube;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::Vec3;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::fragment_discard;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::gpu_entry;

/// Location of the quad's `uv` varying.
const UV_LOC: u32 = 0;

/// Lane of the albedo's alpha.
const ALPHA_LANE: u32 = 3;

/// Number of samples the 4x pipeline takes.
const MSAA_SAMPLES: u32 = 4;

/// Samples the pipeline takes, as an `f32`, for the "all covered" test.
const MSAA_SAMPLES_F: f32 = 4.0;

/// Depth a cleared sample keeps — the far plane, `1.0`.
const CLEAR_DEPTH: f32 = 1.0;

/// Depth difference above which two samples are on different surfaces.
const DEPTH_EDGE: f32 = 0.002;

/// Cosine below which two samples face different ways.
const NORMAL_AGREE: f32 = 0.999;

/// Bias of the Karis coverage weight, in luminance units.
const KARIS_LUMA_BIAS: f32 = 1.0;

/// Red weight of the luminance the coverage weight divides by.
const LUMA_RED: f32 = 0.2126;

/// Green weight of that luminance.
const LUMA_GREEN: f32 = 0.7152;

/// Blue weight of that luminance.
const LUMA_BLUE: f32 = 0.0722;

/// Whether the samples of one pixel landed on surfaces that disagree.
///
/// A pixel whose samples straddle a depth, normal or material-index edge has no
/// single answer: averaging the four shadings would shade a colour that belongs
/// to neither surface. The resolve falls back to sample 0 for such a pixel, and
/// this is the test that decides it.
///
/// The reference scalars come in separately rather than as a record because
/// they are updated inside the sample loop — a record carried across the loop
/// body is outside the translator's subset, plain locals are not.
// qual:allow(srp, max_parameters=6) reason: "the three sample values and the three parts of the reference the first covered sample left behind"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn samples_disagree(
    depth: f32,
    normal: Vec3,
    id: u32,
    ref_depth: f32,
    ref_normal: Vec3,
    ref_id: u32,
) -> f32 {
    let depth_far = if abs_f32(depth - ref_depth) > DEPTH_EDGE {
        1.0
    } else {
        0.0
    };
    let normal_far = if dot3(
        normal.x,
        normal.y,
        normal.z,
        ref_normal.x,
        ref_normal.y,
        ref_normal.z,
    ) < NORMAL_AGREE
    {
        1.0
    } else {
        0.0
    };
    let id_far = if id == ref_id { 0.0 } else { 1.0 };
    max_f32(depth_far, max_f32(normal_far, id_far))
}

/// Absolute value of `x`.
#[inline(always)]
fn abs_f32(x: f32) -> f32 {
    if x < 0.0 { -x } else { x }
}

/// Linear interpolation of two scalars.
#[inline(always)]
fn mix_f32(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Rec. 709 luminance of `color`, floored at zero.
///
/// The coverage weight of the resolve divides by it, so a negative luminance —
/// a colour with out-of-gamut channels — would flip the sign of that weight.
#[inline(always)]
fn luminance(color: Vec3) -> f32 {
    max_f32(
        LUMA_RED * color.x + LUMA_GREEN * color.y + LUMA_BLUE * color.z,
        0.0,
    )
}

/// Deferred lighting at 4x, resolved per sample.
///
/// The g-buffer keeps depth, the material index and the octahedral normal
/// per sample (the renderer binds those three as multisampled textures); the
/// colour, AO and the two layers this pass no longer reads are resolved, so
/// they stay single-sample and are read at the pixel centre.
///
/// The samples are combined with a coverage-weighted (Karis) average rather
/// than a plain one, but only where they agree: a pixel whose samples straddle
/// a depth, normal or material-index edge falls back to sample 0, because an
/// average across such an edge is a colour no surface has.
#[gpu_entry(fragment)]
// qual:allow(complexity, max_function_lines=130) reason: "one linear pass: read the resolved layers once, walk the samples, resolve. The parts the loop body needs are helpers; splitting the entry itself would only move the accumulators into a struct the translator cannot carry across the loop"
// qual:allow(srp, max_parameters=23) reason: "one parameter per binding the deferred pass declares; the attributes above are the table"
#[allow(clippy::too_many_arguments)]
pub fn fs_main_lighting_msaa(
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
    #[gpu(uniform, group = 0, binding = 1)] lighting: Uniform<Lighting>,
    #[gpu(storage_buffer, group = 0, binding = 2, read_only)] materials: Storage<u32>,
    #[gpu(texture_2d, group = 0, binding = 3)] albedo_tex: Texture2d,
    #[gpu(texture_2d, group = 0, binding = 4)] normal_tex: Texture2d,
    #[gpu(texture_2d_u32, group = 0, binding = 5)] material_id_tex: Texture2dU32,
    #[gpu(texture_2d, group = 0, binding = 6)] world_pos_tex: Texture2d,
    #[gpu(texture_2d, group = 0, binding = 7)] mat_params_tex: Texture2d,
    #[gpu(depth_texture_2d, group = 0, binding = 8)] depth_tex: DepthTexture2d,
    #[gpu(sampler, group = 0, binding = 9)] lighting_sampler: Sampler,
    #[gpu(depth_texture_2d_array, group = 0, binding = 10)] shadow_tex: DepthTexture2dArray,
    #[gpu(comparison_sampler, group = 0, binding = 11)] shadow_sampler: ComparisonSampler,
    #[gpu(depth_texture_cube_array, group = 0, binding = 12)] shadow_cube_tex: DepthCubeArray,
    #[gpu(texture_cube, group = 0, binding = 13)] prefilter_cube: TextureCube,
    #[gpu(texture_cube, group = 0, binding = 14)] irradiance_cube: TextureCube,
    #[gpu(texture_2d, group = 0, binding = 15)] brdf_lut: Texture2d,
    #[gpu(sampler, group = 0, binding = 16)] ibl_sampler: Sampler,
    #[gpu(texture_2d, group = 0, binding = 17)] ao_tex: Texture2d,
    #[gpu(depth_texture_2d_array, group = 0, binding = 18)] csm_tex: DepthTexture2dArray,
    #[gpu(uniform, group = 0, binding = 19)] cluster_params: Uniform<ClusterParams>,
    #[gpu(storage_buffer, group = 0, binding = 20, read_only)] clustered_lights: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 21, read_only)] cluster_light_indices: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 22, read_only)] cluster_grid: Storage<u32>,
) {
    let u = fragment_input_lane::<UV_LOC, 2, 0>();
    let w = fragment_input_lane::<UV_LOC, 2, 1>();
    let width = depth_tex.width();
    let height = depth_tex.height();
    let texel_x = (u * width as f32) as u32;
    let texel_y = (w * height as f32) as u32;
    // The resolved layers, read once at the pixel centre.
    let alpha = albedo_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, ALPHA_LANE>(
        lighting_sampler,
        u,
        w,
    );
    let ao = ao_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    let _unused_world_pos =
        world_pos_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    let _unused_mat_params =
        mat_params_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    let cam_pos = layout::camera_position(&camera);
    let clustered = layout::clustered(&cluster_params);
    let debug_view = lighting.field_u32::<DEBUG_VIEW_FIELD>();
    let light_count = layout::light_count(&lighting);
    if alpha < ALPHA_CUTOFF {
        fragment_discard();
    }
    let mut acc = v3(0.0, 0.0, 0.0);
    let mut alpha_acc = 0.0;
    let mut weight = 0.0;
    let mut single = v3(0.0, 0.0, 0.0);
    let mut single_alpha = 0.0;
    let mut covered_count = 0.0;
    let mut have_ref = 0.0;
    let mut disagree_acc = 0.0;
    let mut ref_depth = CLEAR_DEPTH;
    let mut ref_id = 0;
    let mut ref_normal = v3(0.0, 0.0, 1.0);
    let mut s = 0;
    while s < MSAA_SAMPLES {
        let depth = depth_tex.load_sample(texel_x, texel_y, s);
        let id = material_id_tex.load_sample(texel_x, texel_y, s);
        let n = octahedral_decode((
            normal_tex.load_sample::<0>(texel_x, texel_y, s),
            normal_tex.load_sample::<1>(texel_x, texel_y, s),
        ));
        let covered = if depth < CLEAR_DEPTH { 1.0 } else { 0.0 };
        let is_ref = if have_ref == 0.0 && covered == 1.0 {
            1.0
        } else {
            0.0
        };
        let split = samples_disagree(depth, n, id, ref_depth, ref_normal, ref_id);
        disagree_acc += split * covered * have_ref;
        ref_normal = mix3(ref_normal, n, is_ref);
        ref_depth = mix_f32(ref_depth, depth, is_ref);
        ref_id = if is_ref == 1.0 { id } else { ref_id };
        have_ref = max_f32(have_ref, covered);
        covered_count += covered;
        // Each sample is a fragment of its own: its own depth, so its own
        // world position, normal and cluster span.
        let world_pos = layout::reconstruct_world_pos(&camera, u, w, depth);
        let v = normalize3(sub3(cam_pos, world_pos));
        let nov = {
            let d = dot3(n.x, n.y, n.z, v.x, v.y, v.z);
            if d > EPS { d } else { EPS }
        };
        let mat = Material::load(&materials, id, n);
        let opacity = mat.opacity;
        let (t, b) = tangent_frame(n);
        let span_base = ornis_gpu_pbr::cluster_index(&camera, world_pos) * CLUSTER_SPAN_WORDS;
        let frame = frame_of(&mat, world_pos, cam_pos, n, v, t, b, nov);
        let drawn = Sample {
            mat,
            frame,
            alpha,
            ao,
            debug_view,
            light_count,
            span_offset: cluster_grid.load(span_base) * clustered,
            span_count: cluster_grid.load(span_base + 1) * clustered,
        };
        let lit = shade(
            drawn,
            &lighting,
            &shadow_tex,
            shadow_sampler,
            &shadow_cube_tex,
            &prefilter_cube,
            &irradiance_cube,
            &brdf_lut,
            ibl_sampler,
            &csm_tex,
            &clustered_lights,
            &cluster_light_indices,
        );
        let karis = covered / (KARIS_LUMA_BIAS + luminance(lit));
        acc = add3(acc, scale3(lit, karis));
        alpha_acc += opacity * covered;
        weight += karis;
        let first = if s == 0 { 1.0 } else { 0.0 };
        single = mix3(single, lit, first);
        single_alpha = mix_f32(single_alpha, opacity, first);
        s += 1;
    }
    // A pixel whose samples all miss, or all hit and agree, is interior: its
    // average is the answer. Anything else is an edge and keeps sample 0.
    let all_clear = if covered_count == 0.0 { 1.0 } else { 0.0 };
    let all_hit = if covered_count == MSAA_SAMPLES_F {
        1.0
    } else {
        0.0
    };
    // The accumulator counts disagreeing samples; the resolve only asks whether
    // there were any, so it is capped back to the `max` the DSL kept. Keeping
    // the loop-carried value a plain sum is deliberate: LLVM sinks a `max`
    // chain into the loop body and then lists the exit block before the body,
    // and the translator lowers blocks in file order.
    let disagree = min_f32(disagree_acc, 1.0);
    let calm = if disagree == 0.0 { 1.0 } else { 0.0 };
    let edge = 1.0 - max_f32(all_clear, all_hit * calm);
    let divisor = max_f32(weight, EPS);
    let color = mix3(single, scale3(acc, 1.0 / divisor), edge);
    let out_alpha = mix_f32(single_alpha, alpha_acc / divisor, edge);
    fragment_store(color.x, color.y, color.z, out_alpha);
}

#[cfg(test)]
mod tests {
    use ornis_gpu_openpbr::math::v3;

    use super::DEPTH_EDGE;
    use super::luminance;
    use super::mix_f32;
    use super::samples_disagree;

    /// The reference is what the first covered sample left behind; every later
    /// sample is compared to it on all three axes.
    #[test]
    fn a_sample_matching_the_reference_does_not_disagree() {
        let n = v3(0.0, 0.0, 1.0);
        let split = samples_disagree(0.5, n, 7, 0.5 + DEPTH_EDGE * 0.5, n, 7);
        assert_eq!(split, 0.0);
    }

    /// Any one axis past its edge marks the pixel as an edge: a depth step, a
    /// normal bend or a different material all break the average.
    #[test]
    fn one_axis_past_its_edge_marks_a_disagreement() {
        let n = v3(0.0, 0.0, 1.0);
        let far_depth = samples_disagree(0.5 + DEPTH_EDGE * 8.0, n, 7, 0.5, n, 7);
        assert_eq!(far_depth, 1.0);
        let bent = samples_disagree(0.5, v3(1.0, 0.0, 0.0), 7, 0.5, n, 7);
        assert_eq!(bent, 1.0);
        let other_material = samples_disagree(0.5, n, 8, 0.5, n, 7);
        assert_eq!(other_material, 1.0);
    }

    /// The interpolation endpoints are the two inputs, which is what makes the
    /// reference update and the final mix read the same way.
    #[test]
    fn interpolation_hits_both_ends_exactly() {
        assert_eq!(mix_f32(2.0, 5.0, 0.0), 2.0);
        assert_eq!(mix_f32(2.0, 5.0, 1.0), 5.0);
        assert_eq!(mix_f32(2.0, 4.0, 0.5), 3.0);
    }

    /// Rec. 709 weights, floored: a coverage weight that divided by a negative
    /// luminance would flip sign and let one firefly dominate the pixel.
    #[test]
    fn luminance_weights_the_channels_and_floors_at_zero() {
        assert!((luminance(v3(1.0, 1.0, 1.0)) - 1.0).abs() < 1e-6);
        assert!((luminance(v3(1.0, 0.0, 0.0)) - 0.2126).abs() < 1e-6);
        assert_eq!(luminance(v3(-8.0, -8.0, -8.0)), 0.0);
    }
}
