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
pub mod layout;
mod lights;
mod material;
mod shadows;

pub use crate::clusters::cluster_index;
pub use crate::layout::BRDF_LUT_BINDING;
pub use crate::layout::CAMERA_BINDING;
pub use crate::layout::CAMERA_INV_VIEW_PROJ;
pub use crate::layout::CAMERA_POS;
pub use crate::layout::CLUSTER_INDICES_BINDING;
pub use crate::layout::CLUSTER_SPAN_WORDS;
pub use crate::layout::CLUSTERED_FIELD;
pub use crate::layout::CLUSTERED_LIGHTS_BINDING;
pub use crate::layout::CSM_TEX_BINDING;
pub use crate::layout::Camera;
pub use crate::layout::ClusterParams;
pub use crate::layout::DEBUG_VIEW_FIELD;
pub use crate::layout::GROUP;
pub use crate::layout::IBL_MAX_MIP_FIELD;
pub use crate::layout::IBL_SAMPLER_BINDING;
pub use crate::layout::IBL_WEIGHT_FIELD;
pub use crate::layout::IRRADIANCE_BINDING;
pub use crate::layout::LIGHT_COUNT_FIELD;
pub use crate::layout::LIGHT_DIRECTION_WORD;
pub use crate::layout::LIGHT_KIND_WORD;
pub use crate::layout::LIGHT_PARAMS_WORD;
pub use crate::layout::LIGHT_POSITION_WORD;
pub use crate::layout::LIGHT_RANGE_WORD;
pub use crate::layout::LIGHT_SHADOW_LAYER_WORD;
pub use crate::layout::LIGHTING_BINDING;
pub use crate::layout::Lighting;
pub use crate::layout::MATERIALS_BINDING;
pub use crate::layout::PREFILTER_BINDING;
pub use crate::layout::SHADOW_CUBE_BINDING;
pub use crate::layout::SHADOW_SAMPLER_BINDING;
pub use crate::layout::SHADOW_TEX_BINDING;
pub use crate::lights::CSM_ACTIVE_EDGE;
pub use crate::lights::Frame;
pub use crate::lights::LIGHT_SELECTOR_SPLIT;
pub use crate::lights::LightRef;
pub use crate::lights::POINT_OR_SPOT_KIND_EDGE;
pub use crate::lights::SPOT_KIND_EDGE;
pub use crate::lights::light_base;
pub use crate::lights::shaded_radiance;
pub use crate::material::Material;
pub use crate::shadows::csm_visibility;
pub use crate::shadows::point_shadow;
pub use crate::shadows::spot_shadow;

use ornis_gpu_openpbr::layers::evaluate_coat_darkening;
use ornis_gpu_openpbr::layers::evaluate_emission;
use ornis_gpu_openpbr::layers::normalize3;
use ornis_gpu_openpbr::layers::sanitize_hdr;
use ornis_gpu_openpbr::layers::specular_aa_roughness;
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

use crate::direct::direct;
use crate::layout::AMBIENT_FIELD;
use crate::layout::MATERIAL_SAMPLER_BINDING;
use crate::layout::X_LANE;
use crate::layout::X_LANE_U;
use crate::layout::Y_LANE;
use crate::layout::Y_LANE_U;
use crate::layout::Z_LANE;
use crate::layout::Z_LANE_U;

/// Fragment input location of the world position.
const WORLD_POS_LOC: u32 = 0;
/// Fragment input location of the world normal.
const WORLD_NORMAL_LOC: u32 = 1;
/// Fragment input location of the world tangent.
const WORLD_TANGENT_LOC: u32 = 3;
/// Fragment input location of the flat material index.
const MATERIAL_LOC: u32 = 4;
/// Fragment input location of the material texture coordinates.
const UV_LOC: u32 = 2;
/// Lanes of the UV interface value.
const VEC2_LANES: u32 = 2;

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
    let cam_pos = camera_position(&camera);
    let v = normalize3(sub3(cam_pos, world_pos));
    let nov = max_f32(dot3(n.x, n.y, n.z, v.x, v.y, v.z), EPS);
    let t = normalize3(input_xyz::<WORLD_TANGENT_LOC>());
    let b = tangent_bitangent(n, t);
    let mat = Material::load(&materials, fragment_input_u32::<MATERIAL_LOC>(), n);
    let span_base = clusters::cluster_index(&camera, world_pos) * CLUSTER_SPAN_WORDS;
    let clustered = cluster_params.field_u32::<CLUSTERED_FIELD>();
    shade(
        mat,
        world_pos,
        cam_pos,
        n,
        v,
        t,
        b,
        nov,
        lighting.field_u32::<LIGHT_COUNT_FIELD>(),
        cluster_grid.load(span_base) * clustered,
        cluster_grid.load(span_base + 1) * clustered,
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
}

/// The textured forward pass: the same shading with three material maps.
///
/// The gLTF rule is factor times texel: the sampled base colour and emission
/// scale their authored colours, the roughness texel scales the authored
/// roughness *before* the screen-space filter, and the metallic texel scales
/// the authored metallic factor. An unbound slot samples a 1x1 white fallback,
/// which is the exact no-op, so an untextured material renders identically to
/// [`fs_main`].
#[gpu_entry(fragment)]
// qual:allow(complexity, max_function_lines=85) reason: "one fragment entry: the interface decode, the map fold and the shading call"
// qual:allow(srp, max_parameters=19) reason: "one parameter per binding the pass declares; the attributes above are the table"
#[allow(clippy::too_many_arguments)]
pub fn fs_main_textured(
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
    #[gpu(texture_2d, group = 0, binding = 16)] base_color_tex: Texture2d,
    #[gpu(texture_2d, group = 0, binding = 17)] metallic_roughness_tex: Texture2d,
    #[gpu(texture_2d, group = 0, binding = 18)] emissive_tex: Texture2d,
    #[gpu(sampler, group = 0, binding = 19)] material_sampler: Sampler,
) {
    // qual:allow(dry, fragment) reason: "the entry decodes its own interface; the decode returns five values, which the translator cannot pass back from a helper"
    let world_pos = input_xyz::<WORLD_POS_LOC>();
    let n = normalize3(input_xyz::<WORLD_NORMAL_LOC>());
    let cam_pos = camera_position(&camera);
    let v = normalize3(sub3(cam_pos, world_pos));
    let nov = max_f32(dot3(n.x, n.y, n.z, v.x, v.y, v.z), EPS);
    let t = normalize3(input_xyz::<WORLD_TANGENT_LOC>());
    let b = tangent_bitangent(n, t);
    let mut mat = Material::load(&materials, fragment_input_u32::<MATERIAL_LOC>(), n);
    let u = fragment_input_lane::<UV_LOC, VEC2_LANES, X_LANE>();
    let w = fragment_input_lane::<UV_LOC, VEC2_LANES, Y_LANE>();
    let base_map = v3(
        base_color_tex.sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, X_LANE>(
            material_sampler,
            u,
            w,
        ),
        base_color_tex.sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, Y_LANE>(
            material_sampler,
            u,
            w,
        ),
        base_color_tex.sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, Z_LANE>(
            material_sampler,
            u,
            w,
        ),
    );
    let roughness_map = metallic_roughness_tex
        .sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, Y_LANE>(material_sampler, u, w);
    let metallic_map = metallic_roughness_tex
        .sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, Z_LANE>(material_sampler, u, w);
    let emissive_map = v3(
        emissive_tex.sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, X_LANE>(
            material_sampler,
            u,
            w,
        ),
        emissive_tex.sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, Y_LANE>(
            material_sampler,
            u,
            w,
        ),
        emissive_tex.sample_auto_lane::<GROUP, MATERIAL_SAMPLER_BINDING, Z_LANE>(
            material_sampler,
            u,
            w,
        ),
    );
    mat.base_color = mul3(mat.base_color, base_map);
    mat.metalness *= metallic_map;
    mat.specular_roughness = specular_aa_roughness(mat.specular_roughness_raw * roughness_map, n);
    mat.emission_color = mul3(mat.emission_color, emissive_map);
    // qual:allow(dry, fragment) reason: "the entry hands its frame to the shared shading; the call is the seam, not copyable logic"
    let span_base = clusters::cluster_index(&camera, world_pos) * CLUSTER_SPAN_WORDS;
    let clustered = cluster_params.field_u32::<CLUSTERED_FIELD>();
    shade(
        mat,
        world_pos,
        cam_pos,
        n,
        v,
        t,
        b,
        nov,
        lighting.field_u32::<LIGHT_COUNT_FIELD>(),
        cluster_grid.load(span_base) * clustered,
        cluster_grid.load(span_base + 1) * clustered,
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
}

/// Everything after the material record: the light walk and the ambient,
/// emissive and image-based terms.
///
/// Both forward entries end here — the plain one and the textured one, which
/// differs only in where its base colour, roughness, metalness and emission
/// come from. The geometry is passed as scalars rather than a struct because
/// the translator keeps them in registers.
// qual:allow(complexity, max_function_lines=120) reason: "the pass's document: one light walk plus the three additive terms it is summed with"
// qual:allow(srp, max_parameters=22) reason: "the binding table and the frame scalars; a bundle struct would be a second copy of the same list"
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn shade(
    mat: Material,
    world_pos: Vec3,
    cam_pos: Vec3,
    n: Vec3,
    v: Vec3,
    t: Vec3,
    b: Vec3,
    nov: f32,
    light_count: u32,
    span_offset: u32,
    span_count: u32,
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
) {
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
        lighting,
        shadow_tex,
        shadow_cube_tex,
        csm_tex,
        shadow_sampler,
        clustered_lights,
        cluster_light_indices,
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
    let ambient = mul3(ambient_color(lighting), ambient_tint);
    let emission = evaluate_emission(
        mat.emission_luminance,
        mat.emission_color,
        mat.coat_weight,
        mat.coat_color,
        nov,
    );
    let f0 = specular_f0(&mat);
    let reflect_dir = reflect_dir(n, v);
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

/// Specular `F0` of the base layer: the dielectric term, lerped to the metal
/// value by the metalness.
///
/// Both lit passes build it the same way before the image-based term, and the
/// two copies had drifted into a duplicate fragment.
#[inline(always)]
#[must_use]
pub fn specular_f0(mat: &Material) -> Vec3 {
    mix3(
        splat3(fresnel0_from_ior(mat.specular_ior)),
        scale3(mat.base_color, mat.base_weight),
        mat.metalness,
    )
}

/// Mirror reflection direction of `v` about `n`.
#[inline(always)]
#[must_use]
pub fn reflect_dir(n: Vec3, v: Vec3) -> Vec3 {
    sub3(
        scale3(n, REFLECT_SCALE * dot3(n.x, n.y, n.z, v.x, v.y, v.z)),
        v,
    )
}

/// `cross(n, t)`: the bitangent of the surface frame.
#[inline(always)]
fn tangent_bitangent(n: Vec3, t: Vec3) -> Vec3 {
    v3(
        cross_lane::<X_LANE>(n.x, n.y, n.z, t.x, t.y, t.z),
        cross_lane::<Y_LANE>(n.x, n.y, n.z, t.x, t.y, t.z),
        cross_lane::<Z_LANE>(n.x, n.y, n.z, t.x, t.y, t.z),
    )
}

/// `Camera::camera_pos.xyz`.
#[inline(always)]
fn camera_position(camera: &Uniform<Camera, GROUP, CAMERA_BINDING>) -> Vec3 {
    v3(
        camera.vec4_lane::<CAMERA_POS, X_LANE_U>(),
        camera.vec4_lane::<CAMERA_POS, Y_LANE_U>(),
        camera.vec4_lane::<CAMERA_POS, Z_LANE_U>(),
    )
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
