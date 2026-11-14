//! Record layouts the pass reads, and the camera uniform it projects with.
//!
//! The host binds `per_objects` and `materials` as arrays of records, but a
//! kernel sees words: `Storage<u32>` reads one 4-byte word at a time, so the
//! constants here spell the record layout the host (`PerObjectGpu`,
//! `OpenPBRMaterial`) already owns. Every matrix is stored column-major, which
//! is also how a WGSL `mat4x4` lays out its four `vec4` columns.

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;

/// `CameraUniform` slot of the first column of `view_proj`.
pub(crate) const VIEW_PROJ: usize = 0;
/// `CameraUniform` slot of the first column of `previous_view_proj`.
pub(crate) const PREVIOUS_VIEW_PROJ: usize = 9;
/// Offset of the fourth column of a `mat4x4`: three columns precede it.
pub(crate) const LAST_COLUMN: usize = 3;
/// Lane of a clip-space `x`.
pub(crate) const X_LANE: usize = 0;
/// Lane of a clip-space `y`.
pub(crate) const Y_LANE: usize = 1;
/// Lane of a clip-space `z`.
pub(crate) const Z_LANE: usize = 2;
/// Lane of the homogeneous `w`.
pub(crate) const W_LANE: usize = 3;

/// Words per `PerObjectGpu` record (208 bytes).
pub(crate) const OBJECT_STRIDE: u32 = 52;
/// Word of the record's `model` matrix.
pub(crate) const MODEL_WORD: u32 = 0;
/// Word of the record's `normal_matrix` (inverse transpose).
pub(crate) const NORMAL_MATRIX_WORD: u32 = 16;
/// Word of the record's `previous_model` matrix.
pub(crate) const PREVIOUS_MODEL_WORD: u32 = 32;
/// Word of the record's `material_index`.
pub(crate) const MATERIAL_INDEX_WORD: u32 = 48;
/// Word of the record's `motion_valid` flag.
pub(crate) const MOTION_VALID_WORD: u32 = 49;
/// Words per matrix column; one column is one homogeneous coordinate.
pub(crate) const COLUMN_WORDS: u32 = 4;
/// Word of the second matrix column.
pub(crate) const SECOND_COLUMN_WORD: u32 = 2 * COLUMN_WORDS;
/// Word of the third matrix column.
pub(crate) const THIRD_COLUMN_WORD: u32 = 3 * COLUMN_WORDS;

/// Words per `OpenPBRMaterial` record (20 `vec4` slots).
pub(crate) const MATERIAL_STRIDE: u32 = 80;
/// Word of `OpenPBRMaterial::base_params`.
pub(crate) const BASE_PARAMS_WORD: u32 = 0;
/// Word of `OpenPBRMaterial::base_color`.
pub(crate) const BASE_COLOR_WORD: u32 = 4;
/// Word of `OpenPBRMaterial::specular_params`.
pub(crate) const SPECULAR_PARAMS_WORD: u32 = 8;
/// Word of `OpenPBRMaterial::coat_params`.
pub(crate) const COAT_PARAMS_WORD: u32 = 48;
/// `base_params` lane of the metalness.
pub(crate) const METALNESS_LANE: u32 = 2;
/// `specular_params` lane of the roughness.
pub(crate) const ROUGHNESS_LANE: u32 = 1;
/// `specular_params` lane of the specular IOR.
pub(crate) const SPECULAR_IOR_LANE: u32 = 2;
/// `base_color` lane of the opacity.
pub(crate) const OPACITY_LANE: u32 = 3;
/// Word of a `vec4` slot's blue lane, which is also the `rgb` run length.
pub(crate) const BLUE_LANE: u32 = 2;

/// Vertex varying location of the world-space position.
pub(crate) const WORLD_POSITION: u32 = 0;
/// Vertex varying location of the world-space normal.
pub(crate) const WORLD_NORMAL: u32 = 1;
/// Vertex varying location of the texture coordinates.
pub(crate) const UV_LOCATION: u32 = 2;
/// Vertex varying location of the world-space tangent.
pub(crate) const WORLD_TANGENT: u32 = 3;
/// Vertex varying location of the flat material index.
pub(crate) const MATERIAL_INDEX: u32 = 4;
/// Vertex varying location of the previous-frame world position.
pub(crate) const PREVIOUS_WORLD_POSITION: u32 = 5;

/// G-buffer target of the albedo + opacity.
pub(crate) const ALBEDO_TARGET: u32 = 0;
/// G-buffer target of the octahedral-encoded normal.
pub(crate) const NORMAL_TARGET: u32 = 1;
/// G-buffer target of the material id.
pub(crate) const MATERIAL_ID_TARGET: u32 = 2;
/// G-buffer target of the world position's `xy`.
pub(crate) const WORLD_POS_TARGET: u32 = 3;
/// G-buffer target of the packed material parameters.
pub(crate) const MAT_PARAMS_TARGET: u32 = 4;
/// G-buffer target of the motion vector.
pub(crate) const MOTION_TARGET: u32 = 5;
/// G-buffer target of the quantized previous depth.
pub(crate) const PREVIOUS_DEPTH_TARGET: u32 = 6;

/// Lanes of the `vec2` varyings (uv, motion, encoded normal, world `xy`).
pub(crate) const VEC2_LANES: u32 = 2;
/// Lanes of the `vec3` varyings (position, normal, tangent).
pub(crate) const VEC3_LANES: u32 = 3;
/// Lanes of the `vec4` varyings (the previous world position).
pub(crate) const VEC4_LANES: u32 = 4;

/// Half-extent that maps NDC `[-1, 1]` onto a texture UV `[0, 1]`.
pub(crate) const NDC_TO_UV_HALF: f32 = 0.5;
/// Threshold separating a valid previous position from its `w = 0` marker.
pub(crate) const MOTION_VALID_THRESHOLD: f32 = 0.5;
/// Round-to-nearest bias for packing normalized depth into a 16-bit code.
pub(crate) const DEPTH_ROUNDING_BIAS: f32 = 0.5;
/// Largest depth code that still represents a valid motion sample.
pub(crate) const PREVIOUS_DEPTH_MAX: f32 = 65534.0;
/// Reserved 16-bit marker for pixels without a valid previous position.
pub(crate) const PREVIOUS_DEPTH_INVALID: u32 = 65535;
/// The zero bit pattern, as a word: `f32` `0.0` and the `motion_valid` flag
/// both spell invalid with it.
pub(crate) const ZERO_BITS: u32 = 0;
/// One, as a homogeneous coordinate.
pub(crate) const HOMOGENEOUS: f32 = 1.0;
/// Zero, as a direction's homogeneous coordinate.
pub(crate) const DIRECTION: f32 = 0.0;

/// Frame camera: two view-projection matrices, the eye position, the previous
/// frame's matrix.
///
/// Only the view-projection and its predecessor are read here; the tags still
/// describe the whole host struct so the layout matches byte for byte
/// (thirteen `vec4` slots).
pub struct Camera;

impl GpuStruct for Camera {
    const ID: u32 = 0;
    const FIELD_TAGS: &'static [u8] = &[
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
        ornis_gpu_shader::TYPE_TAG_VEC4,
    ];
}

/// Row `LANE` of `m · (x, y, z, w)` for the matrix at word `base`.
///
/// The record stores columns: word `base + 4·j + LANE` is row `LANE` of column
/// `j`, so a lane of the product is the sum of one word per column scaled by
/// that column's coordinate. Reading the words as rows would silently
/// transpose every transform.
// qual:allow(srp, max_parameters=6) reason: "one matrix-vector row: the record base plus the four homogeneous coordinates, all scalars the translator keeps in registers"
#[inline(always)]
pub(crate) fn row_lane<const LANE: u32>(
    source: &Storage<u32, 0, 1>,
    base: u32,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
) -> f32 {
    object_word(source, base + LANE) * x
        + object_word(source, base + COLUMN_WORDS + LANE) * y
        + object_word(source, base + SECOND_COLUMN_WORD + LANE) * z
        + object_word(source, base + THIRD_COLUMN_WORD + LANE) * w
}

/// One float word of the per-object record at `index`.
///
/// The record mixes matrices and counters, so the binding is read as words and
/// the float ones are reinterpreted bit for bit — a conversion would turn a
/// material index into a denormal.
#[inline(always)]
pub(crate) fn object_word(source: &Storage<u32, 0, 1>, index: u32) -> f32 {
    f32::from_bits(source.load(index))
}

/// Row `LANE` of `view_proj · (x, y, z, w)`.
///
/// A `mat4x4` uniform is four `vec4` slots, one per column; the tags of
/// [`Camera`] name them in that order, so the slot is a module constant and
/// the row is the const parameter.
#[inline(always)]
pub(crate) fn view_lane<const LANE: usize>(
    camera: &Uniform<Camera, 0, 0>,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
) -> f32 {
    camera.vec4_lane::<{ VIEW_PROJ }, { LANE }>() * x
        + camera.vec4_lane::<{ VIEW_PROJ + 1 }, { LANE }>() * y
        + camera.vec4_lane::<{ VIEW_PROJ + 2 }, { LANE }>() * z
        + camera.vec4_lane::<{ VIEW_PROJ + LAST_COLUMN }, { LANE }>() * w
}

/// Row `LANE` of `previous_view_proj · (x, y, z, w)`.
#[inline(always)]
pub(crate) fn previous_view_lane<const LANE: usize>(
    camera: &Uniform<Camera, 0, 0>,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
) -> f32 {
    camera.vec4_lane::<{ PREVIOUS_VIEW_PROJ }, { LANE }>() * x
        + camera.vec4_lane::<{ PREVIOUS_VIEW_PROJ + 1 }, { LANE }>() * y
        + camera.vec4_lane::<{ PREVIOUS_VIEW_PROJ + 2 }, { LANE }>() * z
        + camera.vec4_lane::<{ PREVIOUS_VIEW_PROJ + LAST_COLUMN }, { LANE }>() * w
}
