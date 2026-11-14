//! Vertex stage: the instance transform, world-space varyings and the motion
//! frame's previous position.
//!
//! `vs_main` indexes the per-object record with `instance_index`;
//! `vs_main_indirect` first reads the scene index out of the compacted
//! visibility list the culling pass wrote.

use crate::layout::{
    Camera, DIRECTION, HOMOGENEOUS, MATERIAL_INDEX, MATERIAL_INDEX_WORD, MODEL_WORD,
    MOTION_VALID_WORD, NORMAL_MATRIX_WORD, OBJECT_STRIDE, PREVIOUS_MODEL_WORD,
    PREVIOUS_WORLD_POSITION, UV_LOCATION, VEC2_LANES, VEC3_LANES, W_LANE, WORLD_NORMAL,
    WORLD_POSITION, WORLD_TANGENT, X_LANE, Y_LANE, Z_LANE, ZERO_BITS, row_lane, view_lane,
};
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::instance_index;
use ornis_gpu_shader::normalize_lane;
use ornis_gpu_shader::vertex_input_lane;
use ornis_gpu_shader::vertex_position;
use ornis_gpu_shader::vertex_store_2;
use ornis_gpu_shader::vertex_store_3;
use ornis_gpu_shader::vertex_store_4;
use ornis_gpu_shader::vertex_store_u32;

/// Vertex input location of the object-space position.
const POSITION_LOCATION: u32 = 0;
/// Vertex input location of the object-space normal.
const NORMAL_LOCATION: u32 = 1;
/// Vertex input location of the texture coordinates.
const UV_OUT_LOCATION: u32 = 2;
/// Vertex input location of the object-space tangent.
const TANGENT_LOCATION: u32 = 3;

/// World-space position of the shaded vertex, lane `LANE`.
#[inline(always)]
fn world_lane<const LANE: u32>(source: &Storage<u32, 0, 1>, base: u32) -> f32 {
    row_lane::<LANE>(
        source,
        base + MODEL_WORD,
        vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, 0>(),
        vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, 1>(),
        vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, 2>(),
        HOMOGENEOUS,
    )
}

/// Previous-frame world position, lane `LANE`.
#[inline(always)]
fn previous_lane<const LANE: u32>(source: &Storage<u32, 0, 1>, base: u32) -> f32 {
    row_lane::<LANE>(
        source,
        base + PREVIOUS_MODEL_WORD,
        vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, 0>(),
        vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, 1>(),
        vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, 2>(),
        HOMOGENEOUS,
    )
}

/// World-space normal of the shaded vertex, lane `LANE`.
///
/// The normal goes through the inverse transpose: under non-uniform scale the
/// model matrix itself shears a normal off the surface.
#[inline(always)]
fn normal_lane<const LANE: u32>(source: &Storage<u32, 0, 1>, base: u32) -> f32 {
    row_lane::<LANE>(
        source,
        base + NORMAL_MATRIX_WORD,
        vertex_input_lane::<{ NORMAL_LOCATION }, { VEC3_LANES }, 0>(),
        vertex_input_lane::<{ NORMAL_LOCATION }, { VEC3_LANES }, 1>(),
        vertex_input_lane::<{ NORMAL_LOCATION }, { VEC3_LANES }, 2>(),
        DIRECTION,
    )
}

/// World-space tangent of the shaded vertex, lane `LANE`.
///
/// A tangent is a direction, so it goes through the model matrix and not
/// through the inverse transpose the normal uses.
#[inline(always)]
fn tangent_lane<const LANE: u32>(source: &Storage<u32, 0, 1>, base: u32) -> f32 {
    row_lane::<LANE>(
        source,
        base + MODEL_WORD,
        vertex_input_lane::<{ TANGENT_LOCATION }, { VEC3_LANES }, 0>(),
        vertex_input_lane::<{ TANGENT_LOCATION }, { VEC3_LANES }, 1>(),
        vertex_input_lane::<{ TANGENT_LOCATION }, { VEC3_LANES }, 2>(),
        DIRECTION,
    )
}

/// Write the world-space position, the normalized normal, the uv, the
/// normalized tangent, the flat material index and the previous position.
#[inline(always)]
fn emit_varyings(index: u32, source: &Storage<u32, 0, 1>) {
    let base = index * OBJECT_STRIDE;
    vertex_store_3::<WORLD_POSITION>(
        world_lane::<{ X_LANE as u32 }>(source, base),
        world_lane::<{ Y_LANE as u32 }>(source, base),
        world_lane::<{ Z_LANE as u32 }>(source, base),
    );
    let normal_x = normal_lane::<{ X_LANE as u32 }>(source, base);
    let normal_y = normal_lane::<{ Y_LANE as u32 }>(source, base);
    let normal_z = normal_lane::<{ Z_LANE as u32 }>(source, base);
    vertex_store_3::<WORLD_NORMAL>(
        normalize_lane::<{ X_LANE as u32 }>(normal_x, normal_y, normal_z),
        normalize_lane::<{ Y_LANE as u32 }>(normal_x, normal_y, normal_z),
        normalize_lane::<{ Z_LANE as u32 }>(normal_x, normal_y, normal_z),
    );
    vertex_store_2::<UV_LOCATION>(
        vertex_input_lane::<{ UV_OUT_LOCATION }, { VEC2_LANES }, 0>(),
        vertex_input_lane::<{ UV_OUT_LOCATION }, { VEC2_LANES }, 1>(),
    );
    let tangent_x = tangent_lane::<{ X_LANE as u32 }>(source, base);
    let tangent_y = tangent_lane::<{ Y_LANE as u32 }>(source, base);
    let tangent_z = tangent_lane::<{ Z_LANE as u32 }>(source, base);
    vertex_store_3::<WORLD_TANGENT>(
        normalize_lane::<{ X_LANE as u32 }>(tangent_x, tangent_y, tangent_z),
        normalize_lane::<{ Y_LANE as u32 }>(tangent_x, tangent_y, tangent_z),
        normalize_lane::<{ Z_LANE as u32 }>(tangent_x, tangent_y, tangent_z),
    );
    vertex_store_u32::<MATERIAL_INDEX>(source.load(base + MATERIAL_INDEX_WORD));
    // The motion-valid flag is per object: when it is zero the previous
    // position is dropped to `w = 0`, which the fragment stage reads as "no
    // history".
    let previous_w = if source.load(base + MOTION_VALID_WORD) == ZERO_BITS {
        DIRECTION
    } else {
        previous_lane::<{ W_LANE as u32 }>(source, base)
    };
    vertex_store_4::<PREVIOUS_WORLD_POSITION>(
        previous_lane::<{ X_LANE as u32 }>(source, base),
        previous_lane::<{ Y_LANE as u32 }>(source, base),
        previous_lane::<{ Z_LANE as u32 }>(source, base),
        previous_w,
    );
}

/// Clip-space position of the shaded vertex.
#[inline(always)]
fn emit_clip_position(index: u32, camera: &Uniform<Camera, 0, 0>, source: &Storage<u32, 0, 1>) {
    let base = index * OBJECT_STRIDE;
    let world_x = world_lane::<{ X_LANE as u32 }>(source, base);
    let world_y = world_lane::<{ Y_LANE as u32 }>(source, base);
    let world_z = world_lane::<{ Z_LANE as u32 }>(source, base);
    vertex_position(
        view_lane::<{ X_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
        view_lane::<{ Y_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
        view_lane::<{ Z_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
        view_lane::<{ W_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
    );
}

/// Transform one vertex of object `index` and write the stage interface.
#[inline(always)]
fn shade_vertex(index: u32, camera: &Uniform<Camera, 0, 0>, source: &Storage<u32, 0, 1>) {
    emit_varyings(index, source);
    emit_clip_position(index, camera, source);
}

/// G-buffer vertex stage: the instance transform, indexed by `instance_index`.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main(
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] objects: Storage<u32>,
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
) {
    shade_vertex(instance_index(), &camera, &objects);
}

/// G-buffer indirect vertex stage: the compacted visibility list picks the
/// object record before the shared transform.
#[gpu_entry(vertex)]
pub fn vs_main_indirect(
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] objects: Storage<u32>,
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
    #[gpu(storage_buffer, group = 1, binding = 0, read_only)] indices: Storage<u32>,
) {
    shade_vertex(indices.load(instance_index()), &camera, &objects);
}
