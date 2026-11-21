//! Deferred lighting on the llvm2gpu SDK.
//!
//! The g-buffer is resolved into albedo, an octahedral normal, a packed
//! material index, world position, the material parameters and AO; this pass
//! decodes them, rebuilds the world position from depth and runs the same
//! OpenPBR light walk the forward pass uses. It differs from forward PBR in
//! three ways: the material is looked up from the g-buffer index, the shading
//! normal comes from the octahedral decode rather than a varying, and the debug
//! views (`SHADING_DEBUG_*`) are serviced here — which is why the walk keeps
//! the per-light specular/diffuse split and the minimum shadow visibility that
//! the forward pass sums away.
//!
//! Bindings follow the deferred pass's table exactly (0–22). The walk itself is
//! shared with the forward pass: `ornis-gpu-pbr` exposes the shadow selectors
//! and the radiance product parameterised over their binding numbers, so the
//! difference between the two passes is only which slots those numbers name.

#![warn(missing_docs)]

mod layout;
mod msaa;
mod walk;

use layout::GROUP;
use layout::LIGHTING_SAMPLER_BINDING;
use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::math::octahedral_decode;
use ornis_gpu_openpbr::math::sub3;
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
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::fragment_discard;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::gpu_entry;

use crate::walk::EPS;
use crate::walk::Sample;
use crate::walk::frame_of;
use crate::walk::shade;
use crate::walk::tangent_frame;

/// Albedo alpha below which the fragment is dropped.
///
/// The cut happens after the material record is read: the record's specular
/// roughness runs through a screen-space derivative filter, and a derivative
/// after a non-uniform discard is undefined.
pub const ALPHA_CUTOFF: f32 = 0.001;

/// Location of the quad's `uv` varying.
const UV_LOC: u32 = 0;

/// Lane of the albedo's alpha.
const ALPHA_LANE: u32 = 3;

/// Deferred lighting at 1x.
///
/// Every input is a g-buffer layer except the depth buffer, which is read at
/// the texel centre the uv names. The material index is a `u32` layer, so it is
/// loaded rather than sampled: a filtering sampler would blend two indices into
/// a record that belongs to neither.
#[gpu_entry(fragment)]
// qual:allow(srp, max_parameters=23) reason: "one parameter per binding the deferred pass declares; the attributes above are the table"
#[allow(clippy::too_many_arguments)]
pub fn fs_main_lighting(
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
    let depth = depth_tex.load((u * width as f32) as u32, (w * height as f32) as u32);
    // Only the alpha lane is read: the resolved albedo colour is not a shading
    // input, the material table carries the base colour.
    let alpha = albedo_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, ALPHA_LANE>(
        lighting_sampler,
        u,
        w,
    );
    let normal_x =
        normal_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    let normal_y =
        normal_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 1>(lighting_sampler, u, w);
    let world_pos = layout::reconstruct_world_pos(&camera, u, w, depth);
    let material_id = material_id_tex.load((u * width as f32) as u32, (w * height as f32) as u32);
    let n = octahedral_decode((normal_x, normal_y));
    let cam_pos = layout::camera_position(&camera);
    let v = normalize3(sub3(cam_pos, world_pos));
    let nov = {
        let d = dot3(n.x, n.y, n.z, v.x, v.y, v.z);
        if d > EPS { d } else { EPS }
    };
    // The record's specular roughness runs through a screen-space derivative
    // filter, so it is read here — before the discard below.
    let mat = Material::load(&materials, material_id, n);
    let (t, b) = tangent_frame(n);
    let span_base = ornis_gpu_pbr::cluster_index(&camera, world_pos) * CLUSTER_SPAN_WORDS;
    let clustered = layout::clustered(&cluster_params);
    let ao = ao_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    // The g-buffer still carries these two layers and the pass still declares
    // them: the world position now comes back from depth, and the material
    // parameters live in the material table, so neither texel is read.
    let _unused_world_pos =
        world_pos_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    let _unused_mat_params =
        mat_params_tex.sample_lane::<GROUP, LIGHTING_SAMPLER_BINDING, 0>(lighting_sampler, u, w);
    let opacity = mat.opacity;
    let frame = frame_of(&mat, world_pos, cam_pos, n, v, t, b, nov);
    let sample = Sample {
        mat,
        frame,
        alpha,
        ao,
        debug_view: lighting.field_u32::<DEBUG_VIEW_FIELD>(),
        light_count: layout::light_count(&lighting),
        span_offset: cluster_grid.load(span_base) * clustered,
        span_count: cluster_grid.load(span_base + 1) * clustered,
    };
    if sample.alpha < ALPHA_CUTOFF {
        fragment_discard();
    }
    let color = shade(
        sample,
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
    fragment_store(color.x, color.y, color.z, opacity);
}
