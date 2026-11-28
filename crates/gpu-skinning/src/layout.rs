//! Record layouts the skinned stage reads, and the camera uniform it projects
//! with.
//!
//! The host binds `per_objects` and the two joint palettes as arrays of
//! records, but a kernel sees words: `Storage<u32>` reads one 4-byte word at a
//! time, so the constants here spell the record layout the host
//! (`PerObjectGpu`, `SkinJoint`) already owns. Every matrix is stored
//! column-major, which is also how a WGSL `mat4x4` lays out its four `vec4`
//! columns.

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;

/// `CameraUniform` slot of the first column of `view_proj`.
pub(crate) const VIEW_PROJ: usize = 0;
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

/// Words per `SkinJoint` record: two `mat4x4`s (128 bytes).
pub(crate) const JOINT_STRIDE: u32 = 32;
/// Word of the joint's `matrix` (blend position and tangent).
pub(crate) const JOINT_MATRIX_WORD: u32 = 0;
/// Word of the joint's `normal_matrix` (blend the normal).
pub(crate) const JOINT_NORMAL_WORD: u32 = 16;

/// Words per matrix column; one column is one homogeneous coordinate.
pub(crate) const COLUMN_WORDS: u32 = 4;
/// Word of the second matrix column.
pub(crate) const SECOND_COLUMN_WORD: u32 = 2 * COLUMN_WORDS;
/// Word of the third matrix column.
pub(crate) const THIRD_COLUMN_WORD: u32 = 3 * COLUMN_WORDS;

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

/// Lanes of the `vec2` varyings (uv).
pub(crate) const VEC2_LANES: u32 = 2;
/// Lanes of the `vec3` varyings (position, normal, tangent).
pub(crate) const VEC3_LANES: u32 = 3;
/// Homogeneous coordinate of a point.
pub(crate) const HOMOGENEOUS: f32 = 1.0;
/// Homogeneous coordinate of a direction.
pub(crate) const DIRECTION: f32 = 0.0;
/// The zero bit pattern: `f32` `0.0` and the `motion_valid` flag both spell
/// "no history" with it.
pub(crate) const ZERO_BITS: u32 = 0;

/// Frame camera: the view-projection, the inverse, the eye position and the
/// previous frame's matrix.
///
/// Only the view-projection is read here; the tags still describe the whole
/// host struct so the layout matches byte for byte (thirteen `vec4` slots).
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

/// Row `LANE` of the matrix at word `base` times `(x, y, z, w)`.
///
/// The record stores columns: word `base + 4·j + LANE` is row `LANE` of column
/// `j`, so a lane of the product is the sum of one word per column scaled by
/// that column's coordinate. Reading the words as rows would silently
/// transpose every transform.
// qual:allow(srp, max_parameters=6) reason: "one matrix-vector row: the record base plus the four homogeneous coordinates, all scalars the translator keeps in registers"
#[inline(always)]
pub(crate) fn row_lane<const B: u32, const LANE: u32>(
    source: &Storage<u32, 0, B>,
    base: u32,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
) -> f32 {
    // The records mix matrices and counters, so the binding is read as words
    // and the float ones are reinterpreted bit for bit — a conversion would
    // turn a material index into a denormal.
    f32::from_bits(source.load(base + LANE)) * x
        + f32::from_bits(source.load(base + COLUMN_WORDS + LANE)) * y
        + f32::from_bits(source.load(base + SECOND_COLUMN_WORD + LANE)) * z
        + f32::from_bits(source.load(base + THIRD_COLUMN_WORD + LANE)) * w
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
