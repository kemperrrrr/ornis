//! The forward PBR fragment pass on the stable Rust → LLVM IR → WGSL path.
//!
//! One fragment entry shades a g-buffer sample: it loads the material record,
//! walks the punctual lights (the eight uniform lights followed by the cluster
//! list of the fragment's cell), attenuates and shadows each of them, sums the
//! OpenPBR layers, adds the ambient, emissive and image-based terms and writes
//! the linear colour.
//!
//! The layer maths itself lives in `ornis-gpu-openpbr` — the same functions the
//! reference shader calls, ported once and gated against it numerically. What
//! stays here is what the pass alone does: the binding table, the light walk,
//! the shadow selectors and the composite of the terms.
//!
//! The material table mixes `vec4` fields with counters, so it is read as
//! storage words and the float words are reinterpreted bit for bit. The uniform
//! blocks are declared by field tag: `Lighting` carries `array<Light, 8>`
//! between two scalars, and WGSL uniform offsets put the scalar after the array
//! at byte 144, not 20.
//!
//! `xtask build-shaders --package ornis-gpu-pbr --entry fs_main` writes the
//! artifact the renderer includes.

#![no_std]
#![warn(missing_docs)]

mod clusters;
mod direct;
mod layout;
mod lights;
mod material;
mod shadows;

use ornis_gpu_openpbr::layers::evaluate_coat_darkening;
use ornis_gpu_openpbr::layers::evaluate_emission;
use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::layers::sanitize_hdr;
use ornis_gpu_openpbr::math::add3;
use ornis_gpu_openpbr::math::evaluate_ibl;
use ornis_gpu_openpbr::math::fresnel0_from_ior;
use ornis_gpu_openpbr::math::mul3;
use ornis_gpu_openpbr::math::scale3;
use ornis_gpu_openpbr::math::splat3;
use ornis_gpu_openpbr::math::sub3;
use ornis_gpu_openpbr::math::thin_film_modulation;
use ornis_gpu_openpbr::math::thin_film_weight_mix;
use ornis_gpu_openpbr::math::v3;
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
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_input_u32;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::mix;

use crate::direct::Frame;
use crate::direct::direct;
use crate::layout::AMBIENT_FIELD;
use crate::layout::CAMERA_POS;
use crate::layout::CLUSTER_SPAN_WORDS;
use crate::layout::CLUSTERED_FIELD;
use crate::layout::Camera;
use crate::layout::ClusterParams;
use crate::layout::GROUP;
use crate::layout::IBL_MAX_MIP_FIELD;
use crate::layout::IBL_SAMPLER_BINDING;
use crate::layout::IBL_WEIGHT_FIELD;
use crate::layout::LIGHT_COUNT_FIELD;
use crate::layout::LIGHTING_BINDING;
use crate::layout::Lighting;
use crate::layout::X_LANE;
use crate::layout::X_LANE_U;
use crate::layout::Y_LANE;
use crate::layout::Y_LANE_U;
use crate::layout::Z_LANE;
use crate::layout::Z_LANE_U;
use crate::material::Material;

/// Fragment input location of the world position.
const WORLD_POS_LOC: u32 = 0;
/// Fragment input location of the world normal.
const WORLD_NORMAL_LOC: u32 = 1;
/// Fragment input location of the world tangent.
const WORLD_TANGENT_LOC: u32 = 3;
/// Fragment input location of the flat material index.
const MATERIAL_LOC: u32 = 4;

/// Lanes of the world-position interface value.
const VEC3_LANES: u32 = 3;

/// Smallest value a divisor or a cosine is allowed to take.
const EPS: f32 = 0.000001;
/// `geometry_params.z` at or above which the surface is drawn unlit.
const UNLIT_THRESHOLD: f32 = 0.5;
/// Factor of the reflection direction: `2 · dot(n, v)`.
const REFLECT_SCALE: f32 = 2.0;
/// The thin-film thickness arrives in micrometers and the layer wants
/// nanometers.
const NANOMETERS_PER_MICROMETER: f32 = 1000.0;

/// `mix(a, b, t)` lane by lane — one WGSL `mix` per channel.
#[inline(always)]
fn mix3(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    v3(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t))
}

/// WGSL `max(a, b)` for two scalars.
#[inline(always)]
fn max_f32(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Lanes `x`, `y`, `z` of the interface value at `LOC`.
#[inline(always)]
fn input_xyz<const LOC: u32>() -> Vec3 {
    v3(
        fragment_input_lane::<LOC, VEC3_LANES, X_LANE>(),
        fragment_input_lane::<LOC, VEC3_LANES, Y_LANE>(),
        fragment_input_lane::<LOC, VEC3_LANES, Z_LANE>(),
    )
}

/// The fragment pass: one g-buffer sample shaded to a linear colour.
///
/// Reads the material record once, walks the lights, then adds the ambient,
/// emissive and image-based terms. Alpha is the material opacity: the pass
/// writes straight alpha and the composite stage blends it.
#[gpu_entry(fragment)]
// qual:allow(complexity, max_function_lines=150) reason: "one fragment entry; the pass is the binding table plus the shading it drives"
// qual:allow(srp, max_parameters=15) reason: "one parameter per binding the pass declares; the attributes above are the table"
#[allow(clippy::too_many_arguments)]
pub fn fs_main(
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
    #[gpu(storage_buffer, group = 0, binding = 2, read_only)] materials: Storage<u32>,
    #[gpu(uniform, group = 0, binding = 3)] lighting: Uniform<Lighting>,
    #[gpu(depth_texture_2d_array, group = 0, binding = 4)] shadow_tex: DepthTexture2dArray,
    #[gpu(comparison_sampler, group = 0, binding = 5)] shadow_sampler: ComparisonSampler,
    #[gpu(depth_texture_cube_array, group = 0, binding = 6)] shadow_cube_tex: DepthCubeArray,
    #[gpu(texture_cube, group = 0, binding = 7)] prefilter_cube: TextureCube,
    #[gpu(texture_cube, group = 0, binding = 8)] irradiance_cube: TextureCube,
    #[gpu(texture_2d, group = 0, binding = 9)] brdf_lut: Texture2d,
    #[gpu(sampler, group = 0, binding = 10)] ibl_sampler: Sampler,
    #[gpu(depth_texture_2d_array, group = 0, binding = 11)] csm_tex: DepthTexture2dArray,
    #[gpu(uniform, group = 0, binding = 12)] cluster_params: Uniform<ClusterParams>,
    #[gpu(storage_buffer, group = 0, binding = 13, read_only)] clustered_lights: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 14, read_only)] cluster_light_indices: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 15, read_only)] cluster_grid: Storage<u32>,
) {
    let world_pos = input_xyz::<WORLD_POS_LOC>();
    let n = normalize3(input_xyz::<WORLD_NORMAL_LOC>());
    let cam_pos = v3(
        camera.vec4_lane::<CAMERA_POS, X_LANE_U>(),
        camera.vec4_lane::<CAMERA_POS, Y_LANE_U>(),
        camera.vec4_lane::<CAMERA_POS, Z_LANE_U>(),
    );
    let v = normalize3(sub3(cam_pos, world_pos));
    let nov = max_f32(dot3(n.x, n.y, n.z, v.x, v.y, v.z), EPS);
    let tangent = input_xyz::<WORLD_TANGENT_LOC>();
    let t = normalize3(tangent);
    let b = v3(
        cross_lane::<X_LANE>(n.x, n.y, n.z, t.x, t.y, t.z),
        cross_lane::<Y_LANE>(n.x, n.y, n.z, t.x, t.y, t.z),
        cross_lane::<Z_LANE>(n.x, n.y, n.z, t.x, t.y, t.z),
    );
    let mat = Material::load(&materials, fragment_input_u32::<MATERIAL_LOC>(), n);
    let thin_film_mod = thin_film_weight_mix(
        mat.thin_film_weight,
        thin_film_modulation(
            nov,
            mat.thin_film_ior,
            mat.thin_film_thickness_um * NANOMETERS_PER_MICROMETER,
            1.0,
        ),
    );
    let coat_darken = evaluate_coat_darkening(
        mat.coat_weight,
        mat.coat_darkening,
        mat.coat_ior,
        mat.metalness,
        mat.base_color,
        mat.base_weight,
        mat.specular_weight,
        mat.subsurface_weight,
        mat.subsurface_color,
    );
    let light_count = lighting.field_u32::<LIGHT_COUNT_FIELD>();
    let span_base = clusters::cluster_index(&camera, world_pos) * CLUSTER_SPAN_WORDS;
    let clustered = cluster_params.field_u32::<CLUSTERED_FIELD>();
    let span_offset = cluster_grid.load(span_base) * clustered;
    let span_count = cluster_grid.load(span_base + 1) * clustered;
    let frame = Frame {
        world_pos,
        cam_pos,
        n,
        v,
        t,
        b,
        nov,
        thin_film_mod,
        coat_darken,
    };
    let lo = direct(
        &lighting,
        &shadow_tex,
        &shadow_cube_tex,
        &csm_tex,
        shadow_sampler,
        &clustered_lights,
        &cluster_light_indices,
        &mat,
        &frame,
        light_count,
        span_offset,
        span_count,
    );
    let ambient_tint = mix3(
        mat.base_color,
        mul3(mat.base_color, splat3(mat.specular_weight)),
        mat.metalness,
    );
    let ambient = mul3(ambient_color(&lighting), ambient_tint);
    let emission = evaluate_emission(
        mat.emission_luminance,
        mat.emission_color,
        mat.coat_weight,
        mat.coat_color,
        nov,
    );
    let f0_dielectric = fresnel0_from_ior(mat.specular_ior);
    let f0 = mix3(
        splat3(f0_dielectric),
        scale3(mat.base_color, mat.base_weight),
        mat.metalness,
    );
    let reflect_dir = sub3(
        scale3(n, REFLECT_SCALE * dot3(n.x, n.y, n.z, v.x, v.y, v.z)),
        v,
    );
    let ibl_max_mip = lighting.field_f32::<IBL_MAX_MIP_FIELD>();
    let prefiltered = v3(
        prefilter_cube.sample_level_lane::<GROUP, IBL_SAMPLER_BINDING, X_LANE>(
            ibl_sampler,
            reflect_dir,
            mat.specular_roughness * ibl_max_mip,
        ),
        prefilter_cube.sample_level_lane::<GROUP, IBL_SAMPLER_BINDING, Y_LANE>(
            ibl_sampler,
            reflect_dir,
            mat.specular_roughness * ibl_max_mip,
        ),
        prefilter_cube.sample_level_lane::<GROUP, IBL_SAMPLER_BINDING, Z_LANE>(
            ibl_sampler,
            reflect_dir,
            mat.specular_roughness * ibl_max_mip,
        ),
    );
    let irradiance = v3(
        irradiance_cube.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, X_LANE>(ibl_sampler, n),
        irradiance_cube.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, Y_LANE>(ibl_sampler, n),
        irradiance_cube.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, Z_LANE>(ibl_sampler, n),
    );
    let lut_scale = brdf_lut.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, X_LANE>(
        ibl_sampler,
        nov,
        mat.specular_roughness,
    );
    let lut_bias = brdf_lut.sample_auto_lane::<GROUP, IBL_SAMPLER_BINDING, Y_LANE>(
        ibl_sampler,
        nov,
        mat.specular_roughness,
    );
    let ibl = evaluate_ibl(
        prefiltered,
        irradiance,
        lut_scale,
        lut_bias,
        f0,
        scale3(mat.base_color, 1.0 - mat.metalness),
        lighting.field_f32::<IBL_WEIGHT_FIELD>(),
    );
    let full = add3(add3(add3(ambient, lo), emission), ibl);
    let color = sanitize_hdr(if mat.unlit >= UNLIT_THRESHOLD {
        emission
    } else {
        full
    });
    fragment_store(color.x, color.y, color.z, mat.opacity);
}

/// `Lighting::ambient_color.xyz`.
#[inline(always)]
fn ambient_color(lighting: &Uniform<Lighting, GROUP, LIGHTING_BINDING>) -> Vec3 {
    v3(
        lighting.vec4_lane::<AMBIENT_FIELD, X_LANE_U>(),
        lighting.vec4_lane::<AMBIENT_FIELD, Y_LANE_U>(),
        lighting.vec4_lane::<AMBIENT_FIELD, Z_LANE_U>(),
    )
}
