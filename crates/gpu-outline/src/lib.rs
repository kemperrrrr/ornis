//! The editor outline pass (selection and hover silhouettes) on the stable
//! Rust → LLVM IR → WGSL path.
//!
//! The pass draws an inverted hull: every triangle of an `outline_flags`-marked
//! instance is expanded along its object-space normal, front faces are culled
//! and the back faces are depth-tested without writing depth. Outside the
//! object's silhouette the expanded back faces survive the depth test and paint
//! a constant-colour ring; inside it they lose to the object's own depth.
//!
//! The draw carries only the flagged instances as a compact index list, so the
//! vertex stage starts by turning `instance_index` into a scene index. One `u32`
//! flag per compact entry picks the colour: `1` hover, `2` selected.
//!
//! `xtask build-shaders --package ornis-gpu-outline --entry vs_main` and
//! `--entry fs_main` write the two artifacts: the stages are compiled
//! separately because the vertex module reads four buffers the fragment module
//! never touches.
//!
//! The expansion happens in object space so the ring keeps a constant fraction
//! of the mesh size — the replica bakes an entity's radius into the model
//! scale, so a world-space offset would be thin on large objects and fat on
//! small ones.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::fragment_input_lane;
use ornis_gpu_shader::fragment_store_at;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::instance_index;
use ornis_gpu_shader::vertex_input_lane;
use ornis_gpu_shader::vertex_position;
use ornis_gpu_shader::vertex_store_3;

/// Vertex input location of the object-space position.
const POSITION_LOCATION: u32 = 0;
/// Vertex input location of the object-space normal.
const NORMAL_LOCATION: u32 = 1;
/// Vertex output location the flat colour travels in.
const COLOR_LOCATION: u32 = 0;
/// Fragment input location of that colour.
const COLOR_IN_LOCATION: u32 = 0;
/// Lanes of a `vec3` interface value.
const VEC3_LANES: u32 = 3;

/// Lane of the `x` component.
const X_LANE: u32 = 0;
/// Lane of the `y` component.
const Y_LANE: u32 = 1;
/// Lane of the `z` component.
const Z_LANE: u32 = 2;
/// Lane of the homogeneous coordinate.
const W_LANE: usize = 3;

/// `CameraUniform` field of the view-projection's first column.
const VIEW_PROJ: usize = 0;
/// Offset of the last column in a `mat4x4` uniform: four `vec4` slots.
const LAST_COLUMN: usize = 3;

/// Words per matrix column in the per-object record.
const COLUMN_WORDS: u32 = 4;
/// Word of the record's model matrix.
const MODEL_WORD: u32 = 0;
/// Words per `PerObjectGpu` record.
const OBJECT_STRIDE: u32 = 52;

/// Flag value that selects the "selected" colour.
const SELECTED_FLAG: u32 = 2;

/// How far the hull is pushed along the normal, in object-space units.
const OUTLINE_WIDTH: f32 = 0.05;
/// Colour of a hovered object.
const OUTLINE_HOVER_R: f32 = 0.55;
/// Green of a hovered object.
const OUTLINE_HOVER_G: f32 = 0.75;
/// Blue of a hovered object.
const OUTLINE_HOVER_B: f32 = 1.0;
/// Red of a selected object.
const OUTLINE_SELECTED_R: f32 = 0.16;
/// Green of a selected object.
const OUTLINE_SELECTED_G: f32 = 0.55;
/// Blue of a selected object.
const OUTLINE_SELECTED_B: f32 = 1.0;

/// Frame camera: two view-projection matrices, the eye position, the previous
/// frame's matrix.
///
/// Only the current view-projection is read here; the tags still describe the
/// whole host struct so the layout matches byte for byte.
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

/// Lane `LANE` of `model · (x, y, z, 1)`.
///
/// The record stores columns: word `base + 4·j + LANE` is row `LANE` of column
/// `j`, so a lane of the product is one word per column scaled by that column's
/// coordinate. Reading the words as rows would silently transpose every
/// transform.
#[inline(always)]
fn model_lane<const LANE: u32>(
    objects: &Storage<u32, 0, 1>,
    base: u32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    // qual:allow(magic_number, fragment) reason: "column offsets of the record; `SECOND_COLUMN_WORD`/`THIRD_COLUMN_WORD` in the g-buffer crate are the same two values"
    let c0 = object_word(objects, base + MODEL_WORD + LANE);
    let c1 = object_word(objects, base + MODEL_WORD + COLUMN_WORDS + LANE);
    let c2 = object_word(objects, base + MODEL_WORD + 2 * COLUMN_WORDS + LANE);
    let c3 = object_word(objects, base + MODEL_WORD + 3 * COLUMN_WORDS + LANE);
    c0 * x + c1 * y + c2 * z + c3
}

/// One float word of the per-object record.
///
/// The record mixes matrices and counters, so the binding is read as words and
/// the float ones are reinterpreted bit for bit — a conversion would turn a
/// material index into a denormal.
#[inline(always)]
fn object_word(objects: &Storage<u32, 0, 1>, index: u32) -> f32 {
    f32::from_bits(objects.load(index))
}

/// Lane `LANE` of `view_proj · (x, y, z, w)`.
///
/// A `mat4x4` uniform is four `vec4` slots, one per column; the row is the const
/// parameter.
#[inline(always)]
fn clip_lane<const LANE: usize>(
    camera: &Uniform<Camera, 0, 0>,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
) -> f32 {
    let c0 = camera.vec4_lane::<{ VIEW_PROJ }, { LANE }>() * x;
    let c1 = camera.vec4_lane::<{ VIEW_PROJ + 1 }, { LANE }>() * y;
    let c2 = camera.vec4_lane::<{ VIEW_PROJ + 2 }, { LANE }>() * z;
    let c3 = camera.vec4_lane::<{ VIEW_PROJ + LAST_COLUMN }, { LANE }>() * w;
    c0 + c1 + c2 + c3
}

/// Object-space position of the shaded vertex, lane `LANE`, after the hull
/// expansion.
#[inline(always)]
fn expanded_lane<const LANE: u32>() -> f32 {
    vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, { LANE }>()
        + vertex_input_lane::<{ NORMAL_LOCATION }, { VEC3_LANES }, { LANE }>() * OUTLINE_WIDTH
}

/// Outline vertex stage: expand a compacted instance along its normals and pick
/// the hover/selected colour.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main(
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] objects: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 2, read_only)] outline_indices: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 3, read_only)] outline_flags: Storage<u32>,
) {
    let compact = instance_index();
    let scene_index = outline_indices.load(compact);
    let base = scene_index * OBJECT_STRIDE;
    let world_x = model_lane::<X_LANE>(
        &objects,
        base,
        expanded_lane::<X_LANE>(),
        expanded_lane::<Y_LANE>(),
        expanded_lane::<Z_LANE>(),
    );
    let world_y = model_lane::<Y_LANE>(
        &objects,
        base,
        expanded_lane::<X_LANE>(),
        expanded_lane::<Y_LANE>(),
        expanded_lane::<Z_LANE>(),
    );
    let world_z = model_lane::<Z_LANE>(
        &objects,
        base,
        expanded_lane::<X_LANE>(),
        expanded_lane::<Y_LANE>(),
        expanded_lane::<Z_LANE>(),
    );
    vertex_position(
        clip_lane::<{ X_LANE as usize }>(&camera, world_x, world_y, world_z, 1.0),
        clip_lane::<{ Y_LANE as usize }>(&camera, world_x, world_y, world_z, 1.0),
        clip_lane::<{ Z_LANE as usize }>(&camera, world_x, world_y, world_z, 1.0),
        clip_lane::<W_LANE>(&camera, world_x, world_y, world_z, 1.0),
    );
    let flag = outline_flags.load(compact);
    // `select` keeps both sides identical when the flag is neither: hover.
    let selected = flag == SELECTED_FLAG;
    let r = if selected {
        OUTLINE_SELECTED_R
    } else {
        OUTLINE_HOVER_R
    };
    let g = if selected {
        OUTLINE_SELECTED_G
    } else {
        OUTLINE_HOVER_G
    };
    let b = if selected {
        OUTLINE_SELECTED_B
    } else {
        OUTLINE_HOVER_B
    };
    vertex_store_3::<COLOR_LOCATION>(r, g, b);
}

/// Outline fragment stage: the vertex stage already chose hover or selected.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main() {
    let r = fragment_input_lane::<COLOR_IN_LOCATION, VEC3_LANES, X_LANE>();
    let g = fragment_input_lane::<COLOR_IN_LOCATION, VEC3_LANES, Y_LANE>();
    let b = fragment_input_lane::<COLOR_IN_LOCATION, VEC3_LANES, Z_LANE>();
    fragment_store_at::<0>(r, g, b, 1.0);
}
