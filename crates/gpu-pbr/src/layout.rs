//! Binding numbers, record offsets and the uniform structs the pass declares.
//!
//! Every offset here is the byte layout of a host record, so the table doubles
//! as the place where the pass states what it expects the CPU to upload. The
//! per-material record mixes `vec4` fields with counters and is therefore read
//! as storage words; the uniform blocks are declared by field tag and let the
//! translator lay them out (WGSL uniform offsets are not the C offsets).
//!
//! Only the bindings the pass's *code* names appear as constants: the entry's
//! own `#[gpu(..., binding = N)]` attributes carry the full table, because the
//! attribute is what the translator reads.

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::TYPE_TAG_ARRAY;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::TYPE_TAG_STRUCT;
use ornis_gpu_shader::TYPE_TAG_U32;
use ornis_gpu_shader::TYPE_TAG_VEC4;

/// Group every binding of the pass lives in.
pub const GROUP: u32 = 0;
/// Binding of the frame camera block.
pub const CAMERA_BINDING: u32 = 0;
/// Binding of the per-material table.
pub const MATERIALS_BINDING: u32 = 2;
/// Binding of the lights, cascades and ambient block.
pub const LIGHTING_BINDING: u32 = 3;
/// Binding of the spotlight/CSM shadow atlas.
pub const SHADOW_TEX_BINDING: u32 = 4;
/// Binding of the comparison sampler both shadow families share.
pub const SHADOW_SAMPLER_BINDING: u32 = 5;
/// Binding of the point-light shadow cube pool.
pub const SHADOW_CUBE_BINDING: u32 = 6;
/// Binding of the prefiltered environment map.
pub const PREFILTER_BINDING: u32 = 7;
/// Binding of the diffuse irradiance map.
pub const IRRADIANCE_BINDING: u32 = 8;
/// Binding of the split-sum BRDF lookup table.
pub const BRDF_LUT_BINDING: u32 = 9;
/// Binding of the sampler every material texture shares (textured entry).
pub const MATERIAL_SAMPLER_BINDING: u32 = 19;
/// Binding of the IBL sampler.
pub const IBL_SAMPLER_BINDING: u32 = 10;
/// Binding of the cascaded-shadow-map atlas.
pub const CSM_TEX_BINDING: u32 = 11;
/// Binding of the per-frame light array.
pub const CLUSTERED_LIGHTS_BINDING: u32 = 13;
/// Binding of the per-cluster light index list.
pub const CLUSTER_INDICES_BINDING: u32 = 14;

/// Words per `OpenPBRMaterial` record (`20` `vec4` fields).
pub const MATERIAL_STRIDE: u32 = 80;
/// Offset of `base_params`.
pub const BASE_PARAMS_WORD: u32 = 0;
/// Offset of `base_color`.
pub const BASE_COLOR_WORD: u32 = 4;
/// Offset of `specular_params`.
pub const SPECULAR_PARAMS_WORD: u32 = 8;
/// Offset of `specular_color`.
pub const SPECULAR_COLOR_WORD: u32 = 12;
/// Offset of `transmission_params`.
pub const TRANSMISSION_PARAMS_WORD: u32 = 16;
/// Offset of `transmission_color`.
pub const TRANSMISSION_COLOR_WORD: u32 = 20;
/// Offset of `transmission_scatter`.
pub const TRANSMISSION_SCATTER_WORD: u32 = 24;
/// Offset of `subsurface_params`.
pub const SUBSURFACE_PARAMS_WORD: u32 = 28;
/// Offset of `subsurface_color`.
pub const SUBSURFACE_COLOR_WORD: u32 = 32;
/// Offset of `subsurface_radius_scale_gb`.
pub const RADIUS_SCALE_WORD: u32 = 36;
/// Offset of `fuzz_params`.
pub const FUZZ_PARAMS_WORD: u32 = 40;
/// Offset of `fuzz_color`.
pub const FUZZ_COLOR_WORD: u32 = 44;
/// Offset of `coat_params`.
pub const COAT_PARAMS_WORD: u32 = 48;
/// Offset of `coat_color`.
pub const COAT_COLOR_WORD: u32 = 52;
/// Offset of `coat_ior`.
pub const COAT_IOR_WORD: u32 = 56;
/// Offset of `thin_film_params`.
pub const THIN_FILM_PARAMS_WORD: u32 = 60;
/// Offset of `emission_params`.
pub const EMISSION_PARAMS_WORD: u32 = 64;
/// Offset of `emission_color`.
pub const EMISSION_COLOR_WORD: u32 = 68;
/// Offset of `geometry_params`.
pub const GEOMETRY_PARAMS_WORD: u32 = 72;

/// Words per `Light` record: five `vec4` fields and a `mat4x4` shadow matrix.
pub const LIGHT_STRIDE: u32 = 36;
/// Offset of `Light::kind`.
pub const LIGHT_KIND_WORD: u32 = 0;
/// Offset of `Light::direction`.
pub const LIGHT_DIRECTION_WORD: u32 = 4;
/// Offset of `Light::position`.
pub const LIGHT_POSITION_WORD: u32 = 8;
/// Offset of `Light::color` (rgb plus intensity in `w`).
pub const LIGHT_COLOR_WORD: u32 = 12;
/// Offset of `Light::color::w`, the intensity.
pub const LIGHT_INTENSITY_WORD: u32 = LIGHT_COLOR_WORD + 3;
/// Offset of `Light::params`.
pub const LIGHT_PARAMS_WORD: u32 = 16;
/// Offset of `Light::params::x`, the light range.
pub const LIGHT_RANGE_WORD: u32 = LIGHT_PARAMS_WORD;
/// Offset of `Light::params::w`, the shadow atlas layer (`-1` for none).
pub const LIGHT_SHADOW_LAYER_WORD: u32 = LIGHT_PARAMS_WORD + 3;
/// Offset of the shadow view-projection's first column.
pub const LIGHT_SHADOW_VP_WORD: u32 = 20;
/// Words per matrix column.
pub const COLUMN_WORDS: u32 = 4;
/// Index of a matrix's second column, as a field index.
pub const FIRST_COLUMN: usize = 1;
/// Index of a matrix's third column, as a field index.
pub const MIDDLE_COLUMN: usize = 2;
/// Index of a matrix's fourth column, as a field index.
pub const LAST_COLUMN: usize = 3;
/// Word offset of a matrix's second column.
pub const SECOND_COLUMN_WORD: u32 = COLUMN_WORDS;
/// Word offset of a matrix's third column.
pub const THIRD_COLUMN_WORD: u32 = COLUMN_WORDS + COLUMN_WORDS;
/// Word offset of a matrix's fourth column.
pub const FOURTH_COLUMN_WORD: u32 = COLUMN_WORDS + COLUMN_WORDS + COLUMN_WORDS;

/// Field index of `Camera::camera_pos`.
pub const CAMERA_POS: usize = 8;
/// Field index of the view-projection's first column.
pub const CAMERA_VIEW_PROJ: usize = 0;
/// Field index of the inverse view-projection's first column.
pub const CAMERA_INV_VIEW_PROJ: usize = 4;
/// Field index of `Lighting::ambient_color`.
pub const AMBIENT_FIELD: usize = 0;
/// Field index of `Lighting::light_count`.
pub const LIGHT_COUNT_FIELD: usize = 2;
/// Field index of `Lighting::ibl_weight`.
pub const IBL_WEIGHT_FIELD: usize = 3;
/// Field index of `Lighting::ibl_max_mip`.
pub const IBL_MAX_MIP_FIELD: usize = 4;
/// Field index of `Lighting::debug_view`.
pub const DEBUG_VIEW_FIELD: usize = 5;
/// Field index of `Lighting::csm`.
pub const CSM_FIELD: usize = 6;
/// Field index of `Csm::vps`.
pub const CSM_VPS_FIELD: usize = 0;
/// Field index of `Csm::view_fwd`.
pub const CSM_VIEW_FWD_FIELD: usize = 1;
/// Field index of `Csm::splits_lo`.
pub const CSM_SPLITS_LO_FIELD: usize = 2;
/// Field index of `Csm::split_far`.
pub const CSM_SPLIT_FAR_FIELD: usize = 3;
/// Field index of `ClusterParams::clustered`.
pub const CLUSTERED_FIELD: usize = 2;

/// Word offset of a `vec4`'s second lane, for records read word by word.
pub const LANE_Y: u32 = 1;
/// Word offset of the third lane.
pub const LANE_Z: u32 = 2;
/// Word offset of the fourth lane.
pub const LANE_W: u32 = 3;
/// Lane of `x`.
pub const X_LANE: u32 = 0;
/// Lane of `y`.
pub const Y_LANE: u32 = 1;
/// Lane of `z`.
pub const Z_LANE: u32 = 2;
/// Lane of `x` as a `usize` field index.
pub const X_LANE_U: usize = 0;
/// Lane of `y` as a `usize` field index.
pub const Y_LANE_U: usize = 1;
/// Lane of `z` as a `usize` field index.
pub const Z_LANE_U: usize = 2;
/// Lane of `w` as a `usize` field index.
pub const W_LANE_U: usize = 3;
/// Entries in the uniform light array of [`Lighting`].
pub const LIGHT_ARRAY_LEN: u8 = 8;

/// Clusters along x.
pub const CLUSTERS_X: u32 = 16;
/// Clusters along y.
pub const CLUSTERS_Y: u32 = 9;
/// Depth slices.
pub const CLUSTERS_Z: u32 = 24;
/// Near plane of the cluster depth mapping.
pub const CLUSTER_DEPTH_NEAR: f32 = 0.5;
/// Scale of `log(w / near)` that distributes the slices.
pub const CLUSTER_SLICE_SCALE: f32 = 3.440_738_4;
/// Bias that keeps a surface exactly on a slice boundary inside its own slice.
pub const CLUSTER_SLICE_LOOKUP_EPS: f32 = 0.0001;
/// Half of the NDC-to-UV range.
pub const NDC_TO_UV_HALF: f32 = 0.5;
/// Words per `ClusterSpan` (`offset`, `count`).
pub const CLUSTER_SPAN_WORDS: u32 = 2;

/// The frame camera.
///
/// Only `view_proj` and `camera_pos` are read: the cluster lookup projects the
/// fragment, and `camera_pos` turns the shading point into a view vector.
pub struct Camera;

impl GpuStruct for Camera {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
    ];
}

/// One punctual light.
///
/// `shadow_vp` is a `mat4x4` in the record and four `vec4` columns in the
/// uniform block: WGSL uniform offsets pad every column to 16 bytes, so the two
/// agree byte for byte.
pub struct Light;

impl GpuStruct for Light {
    const ID: u32 = 1;
    const FIELD_TAGS: &'static [u8] = &[
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
    ];
}

/// Cascades the shadow block carries.
pub const CASCADE_ARRAY_LEN: u8 = 4;

/// One cascade's view-projection, as four `vec4` columns.
pub struct CsmVp;

impl GpuStruct for CsmVp {
    const ID: u32 = 2;
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_VEC4, TYPE_TAG_VEC4, TYPE_TAG_VEC4, TYPE_TAG_VEC4];
    const NESTED: &'static [(u32, &'static [u8])] = &[];
}

/// The cascade block: one view-projection per cascade, the camera forward axis
/// and the split distances.
///
/// The trailing `pad` of the host struct is not declared: it is the alignment
/// tail WGSL inserts anyway.
pub struct Csm;

impl GpuStruct for Csm {
    const ID: u32 = 3;
    const FIELD_TAGS: &'static [u8] = &[
        TYPE_TAG_ARRAY,
        TYPE_TAG_STRUCT,
        CsmVp::ID as u8,
        CASCADE_ARRAY_LEN,
        TYPE_TAG_VEC4,
        TYPE_TAG_VEC4,
        TYPE_TAG_F32,
    ];
    const NESTED: &'static [(u32, &'static [u8])] = &[(CsmVp::ID, CsmVp::FIELD_TAGS)];
}

/// Ambient light, the punctual light array and the cascade block.
///
/// The array sits between two scalars, so `light_count` lands at byte 144
/// rather than 20 — the reason the struct is declared by tags instead of by
/// hand.
pub struct Lighting;

impl GpuStruct for Lighting {
    const ID: u32 = 4;
    const FIELD_TAGS: &'static [u8] = &[
        TYPE_TAG_VEC4,
        TYPE_TAG_ARRAY,
        TYPE_TAG_STRUCT,
        Light::ID as u8,
        LIGHT_ARRAY_LEN,
        TYPE_TAG_U32,
        TYPE_TAG_F32,
        TYPE_TAG_F32,
        TYPE_TAG_U32,
        TYPE_TAG_STRUCT,
        Csm::ID as u8,
    ];
    // `Csm` names `CsmVp`, so the table lists both: it is everything the block
    // reaches, not just its direct children.
    const NESTED: &'static [(u32, &'static [u8])] = &[
        (Light::ID, Light::FIELD_TAGS),
        (Csm::ID, Csm::FIELD_TAGS),
        (CsmVp::ID, CsmVp::FIELD_TAGS),
    ];
}

/// Cluster grid counters.
pub struct ClusterParams;

impl GpuStruct for ClusterParams {
    const ID: u32 = 5;
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_U32, TYPE_TAG_U32, TYPE_TAG_U32, TYPE_TAG_U32];
}

/// One float word of the material record.
///
/// The record mixes `vec4` fields with counters, so it is bound as words and
/// the float ones are reinterpreted bit for bit — a conversion would turn the
/// packed `material_index` of the g-buffer into a denormal.
// qual:allow(iosp) reason: "the load and the bitcast are the whole body; splitting them would move the storage read away from the only place the record layout is stated"
#[inline(always)]
pub(crate) fn material_word(
    materials: &Storage<u32, GROUP, MATERIALS_BINDING, ReadOnly>,
    base: u32,
    word: u32,
) -> f32 {
    f32::from_bits(materials.load(base + word))
}

/// One float word of a `Light` record in `clustered_lights`.
// qual:allow(iosp) reason: "the load and the bitcast are the whole body, the same shape as `material_word`"
#[inline(always)]
pub(crate) fn light_word<const LIGHTS_B: u32>(
    lights: &Storage<u32, GROUP, LIGHTS_B, ReadOnly>,
    base: u32,
    word: u32,
) -> f32 {
    f32::from_bits(lights.load(base + word))
}
