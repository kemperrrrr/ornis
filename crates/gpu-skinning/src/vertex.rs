//! Vertex stage: linear blend skinning over the joint palette, then the
//! instance transform and the world-space varyings.
//!
//! The palette is two storage arrays — this frame's and the previous frame's —
//! of `SkinJoint` records (the joint matrix plus the inverse transpose for
//! normals). Four joint indices ride as a flat `vec4<u32>` vertex input and
//! four weights as a `vec4<f32>`; the only runtime index in the stage is the
//! palette one.
//!
//! Normals blend through each joint's inverse transpose, positions and
//! tangents through the joint matrix. A degenerate joint uploads an identity
//! normal matrix, matching the CPU path.

use crate::layout::{
    Camera, DIRECTION, HOMOGENEOUS, JOINT_MATRIX_WORD, JOINT_NORMAL_WORD, JOINT_STRIDE,
    MATERIAL_INDEX, MATERIAL_INDEX_WORD, MODEL_WORD, MOTION_VALID_WORD, NORMAL_MATRIX_WORD,
    OBJECT_STRIDE, PREVIOUS_MODEL_WORD, PREVIOUS_WORLD_POSITION, UV_LOCATION, VEC2_LANES,
    VEC3_LANES, W_LANE, WORLD_NORMAL, WORLD_POSITION, WORLD_TANGENT, X_LANE, Y_LANE, Z_LANE,
    ZERO_BITS, row_lane, view_lane,
};
use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::instance_index;
use ornis_gpu_shader::normalize_lane;
use ornis_gpu_shader::vertex_input_lane;
use ornis_gpu_shader::vertex_input_u32x4;
use ornis_gpu_shader::vertex_position;
use ornis_gpu_shader::vertex_store_2;
use ornis_gpu_shader::vertex_store_3;
use ornis_gpu_shader::vertex_store_4;
use ornis_gpu_shader::vertex_store_u32;

/// Vertex input location of the bind-pose position.
const POSITION_LOCATION: u32 = 0;
/// Vertex input location of the bind-pose normal.
const NORMAL_LOCATION: u32 = 1;
/// Vertex input location of the texture coordinates.
const UV_IN_LOCATION: u32 = 2;
/// Vertex input location of the bind-pose tangent.
const TANGENT_LOCATION: u32 = 3;
/// Vertex input location of the flat `vec4<u32>` joint indices.
const JOINTS_LOCATION: u32 = 4;
/// Vertex input location of the `vec4<f32>` influence weights.
const WEIGHTS_LOCATION: u32 = 5;

/// Influences per vertex: the four lanes of the joint and weight inputs.
const INFLUENCES: u32 = 4;

/// Joint index `LANE` of the vertex's influence list.
#[inline(always)]
fn joint_index<const LANE: u32>() -> u32 {
    vertex_input_u32x4::<JOINTS_LOCATION, LANE>()
}

/// Influence weight `LANE` of the vertex.
#[inline(always)]
fn weight<const LANE: u32>() -> f32 {
    vertex_input_lane::<{ WEIGHTS_LOCATION }, { INFLUENCES }, LANE>()
}

/// Lane `LANE` of `Σ_j w_j · M_j · (x, y, z, w)`.
///
/// `base_word` names the matrix inside the joint record (`JOINT_MATRIX_WORD`
/// to blend a point or a direction, `JOINT_NORMAL_WORD` for the inverse
/// transpose a normal needs), so the two palettes share one blend.
// qual:allow(srp, max_parameters=6) reason: "one blend: the palette, the matrix slot, and the four homogeneous coordinates it transforms"
// qual:allow(complexity, magic_numbers) reason: "the four influences are unrolled: their indices are the ABI, not tunables"
#[inline(always)]
fn blended_lane<const B: u32, const LANE: u32>(
    palette: &Storage<u32, 0, B>,
    base_word: u32,
    x: f32,
    y: f32,
    z: f32,
    w: f32,
) -> f32 {
    row_lane::<B, LANE>(
        palette,
        joint_index::<0>() * JOINT_STRIDE + base_word,
        x,
        y,
        z,
        w,
    ) * weight::<0>()
        + row_lane::<B, LANE>(
            palette,
            joint_index::<1>() * JOINT_STRIDE + base_word,
            x,
            y,
            z,
            w,
        ) * weight::<1>()
        + row_lane::<B, LANE>(
            palette,
            joint_index::<2>() * JOINT_STRIDE + base_word,
            x,
            y,
            z,
            w,
        ) * weight::<2>()
        + row_lane::<B, LANE>(
            palette,
            joint_index::<3>() * JOINT_STRIDE + base_word,
            x,
            y,
            z,
            w,
        ) * weight::<3>()
}

/// Bind-pose position lane `LANE` (the `vec3` at location 0).
#[inline(always)]
fn position_input<const LANE: u32>() -> f32 {
    vertex_input_lane::<{ POSITION_LOCATION }, { VEC3_LANES }, LANE>()
}

/// Bind-pose normal lane `LANE` (the `vec3` at location 1).
#[inline(always)]
fn normal_input<const LANE: u32>() -> f32 {
    vertex_input_lane::<{ NORMAL_LOCATION }, { VEC3_LANES }, LANE>()
}

/// Bind-pose tangent lane `LANE` (the `vec3` at location 3).
#[inline(always)]
fn tangent_input<const LANE: u32>() -> f32 {
    vertex_input_lane::<{ TANGENT_LOCATION }, { VEC3_LANES }, LANE>()
}

/// Blend, transform and write the whole stage interface.
///
/// The world position is computed once: the clip-space output is that same
/// point through the camera, and recomputing it would duplicate the blend.
// qual:allow(complexity, magic_numbers) reason: "palette and object binding numbers are the bind-group ABI"
// qual:allow(complexity, max_function_lines=200) reason: "one vertex shader: blend, three transforms and the varying writes are one sequence"
#[inline(always)]
fn shade(
    base: u32,
    camera: &Uniform<Camera, 0, 0>,
    palette: &Storage<u32, 0, 3>,
    previous_palette: &Storage<u32, 0, 4>,
    objects: &Storage<u32, 0, 1>,
) {
    let px = position_input::<{ X_LANE as u32 }>();
    let py = position_input::<{ Y_LANE as u32 }>();
    let pz = position_input::<{ Z_LANE as u32 }>();
    // Skinned point, then the instance transform on top of it.
    let skinned_x =
        blended_lane::<3, { X_LANE as u32 }>(palette, JOINT_MATRIX_WORD, px, py, pz, HOMOGENEOUS);
    let skinned_y =
        blended_lane::<3, { Y_LANE as u32 }>(palette, JOINT_MATRIX_WORD, px, py, pz, HOMOGENEOUS);
    let skinned_z =
        blended_lane::<3, { Z_LANE as u32 }>(palette, JOINT_MATRIX_WORD, px, py, pz, HOMOGENEOUS);
    let world_x = row_lane::<1, { X_LANE as u32 }>(
        objects,
        base + MODEL_WORD,
        skinned_x,
        skinned_y,
        skinned_z,
        HOMOGENEOUS,
    );
    let world_y = row_lane::<1, { Y_LANE as u32 }>(
        objects,
        base + MODEL_WORD,
        skinned_x,
        skinned_y,
        skinned_z,
        HOMOGENEOUS,
    );
    let world_z = row_lane::<1, { Z_LANE as u32 }>(
        objects,
        base + MODEL_WORD,
        skinned_x,
        skinned_y,
        skinned_z,
        HOMOGENEOUS,
    );
    vertex_store_3::<WORLD_POSITION>(world_x, world_y, world_z);
    let nx = normal_input::<{ X_LANE as u32 }>();
    let ny = normal_input::<{ Y_LANE as u32 }>();
    let nz = normal_input::<{ Z_LANE as u32 }>();
    let skinned_nx =
        blended_lane::<3, { X_LANE as u32 }>(palette, JOINT_NORMAL_WORD, nx, ny, nz, DIRECTION);
    let skinned_ny =
        blended_lane::<3, { Y_LANE as u32 }>(palette, JOINT_NORMAL_WORD, nx, ny, nz, DIRECTION);
    let skinned_nz =
        blended_lane::<3, { Z_LANE as u32 }>(palette, JOINT_NORMAL_WORD, nx, ny, nz, DIRECTION);
    let world_nx = row_lane::<1, { X_LANE as u32 }>(
        objects,
        base + NORMAL_MATRIX_WORD,
        skinned_nx,
        skinned_ny,
        skinned_nz,
        DIRECTION,
    );
    let world_ny = row_lane::<1, { Y_LANE as u32 }>(
        objects,
        base + NORMAL_MATRIX_WORD,
        skinned_nx,
        skinned_ny,
        skinned_nz,
        DIRECTION,
    );
    let world_nz = row_lane::<1, { Z_LANE as u32 }>(
        objects,
        base + NORMAL_MATRIX_WORD,
        skinned_nx,
        skinned_ny,
        skinned_nz,
        DIRECTION,
    );
    vertex_store_3::<WORLD_NORMAL>(
        normalize_lane::<{ X_LANE as u32 }>(world_nx, world_ny, world_nz),
        normalize_lane::<{ Y_LANE as u32 }>(world_nx, world_ny, world_nz),
        normalize_lane::<{ Z_LANE as u32 }>(world_nx, world_ny, world_nz),
    );
    vertex_store_2::<UV_LOCATION>(
        vertex_input_lane::<{ UV_IN_LOCATION }, { VEC2_LANES }, 0>(),
        vertex_input_lane::<{ UV_IN_LOCATION }, { VEC2_LANES }, 1>(),
    );
    let tx = tangent_input::<{ X_LANE as u32 }>();
    let ty = tangent_input::<{ Y_LANE as u32 }>();
    let tz = tangent_input::<{ Z_LANE as u32 }>();
    let skinned_tx =
        blended_lane::<3, { X_LANE as u32 }>(palette, JOINT_MATRIX_WORD, tx, ty, tz, DIRECTION);
    let skinned_ty =
        blended_lane::<3, { Y_LANE as u32 }>(palette, JOINT_MATRIX_WORD, tx, ty, tz, DIRECTION);
    let skinned_tz =
        blended_lane::<3, { Z_LANE as u32 }>(palette, JOINT_MATRIX_WORD, tx, ty, tz, DIRECTION);
    // A tangent is a direction: the model matrix, not the inverse transpose.
    let world_tx = row_lane::<1, { X_LANE as u32 }>(
        objects,
        base + MODEL_WORD,
        skinned_tx,
        skinned_ty,
        skinned_tz,
        DIRECTION,
    );
    let world_ty = row_lane::<1, { Y_LANE as u32 }>(
        objects,
        base + MODEL_WORD,
        skinned_tx,
        skinned_ty,
        skinned_tz,
        DIRECTION,
    );
    let world_tz = row_lane::<1, { Z_LANE as u32 }>(
        objects,
        base + MODEL_WORD,
        skinned_tx,
        skinned_ty,
        skinned_tz,
        DIRECTION,
    );
    vertex_store_3::<WORLD_TANGENT>(
        normalize_lane::<{ X_LANE as u32 }>(world_tx, world_ty, world_tz),
        normalize_lane::<{ Y_LANE as u32 }>(world_tx, world_ty, world_tz),
        normalize_lane::<{ Z_LANE as u32 }>(world_tx, world_ty, world_tz),
    );
    vertex_store_u32::<MATERIAL_INDEX>(objects.load(base + MATERIAL_INDEX_WORD));
    // The motion frame blends the same vertices through the previous palette;
    // a zero `motion_valid` flag drops the result to `w = 0`, which the
    // fragment stage reads as "no history".
    let previous_x = blended_lane::<4, { X_LANE as u32 }>(
        previous_palette,
        JOINT_MATRIX_WORD,
        px,
        py,
        pz,
        HOMOGENEOUS,
    );
    let previous_y = blended_lane::<4, { Y_LANE as u32 }>(
        previous_palette,
        JOINT_MATRIX_WORD,
        px,
        py,
        pz,
        HOMOGENEOUS,
    );
    let previous_z = blended_lane::<4, { Z_LANE as u32 }>(
        previous_palette,
        JOINT_MATRIX_WORD,
        px,
        py,
        pz,
        HOMOGENEOUS,
    );
    let previous_w = if objects.load(base + MOTION_VALID_WORD) == ZERO_BITS {
        DIRECTION
    } else {
        row_lane::<1, { W_LANE as u32 }>(
            objects,
            base + PREVIOUS_MODEL_WORD,
            previous_x,
            previous_y,
            previous_z,
            HOMOGENEOUS,
        )
    };
    vertex_store_4::<PREVIOUS_WORLD_POSITION>(
        row_lane::<1, { X_LANE as u32 }>(
            objects,
            base + PREVIOUS_MODEL_WORD,
            previous_x,
            previous_y,
            previous_z,
            HOMOGENEOUS,
        ),
        row_lane::<1, { Y_LANE as u32 }>(
            objects,
            base + PREVIOUS_MODEL_WORD,
            previous_x,
            previous_y,
            previous_z,
            HOMOGENEOUS,
        ),
        row_lane::<1, { Z_LANE as u32 }>(
            objects,
            base + PREVIOUS_MODEL_WORD,
            previous_x,
            previous_y,
            previous_z,
            HOMOGENEOUS,
        ),
        previous_w,
    );
    // Clip space of the same world position the varying carries.
    vertex_position(
        view_lane::<{ X_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
        view_lane::<{ Y_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
        view_lane::<{ Z_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
        view_lane::<{ W_LANE }>(camera, world_x, world_y, world_z, HOMOGENEOUS),
    );
}

/// Skinned g-buffer vertex stage: blend, transform, write the shared varying.
#[gpu_entry(vertex)]
pub fn vs_main_skinned(
    #[gpu(uniform, group = 0, binding = 0)] camera: Uniform<Camera>,
    #[gpu(storage_buffer, group = 0, binding = 1, read_only)] objects: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 3, read_only)] palette: Storage<u32>,
    #[gpu(storage_buffer, group = 0, binding = 4, read_only)] previous_palette: Storage<u32>,
) {
    shade(
        instance_index() * OBJECT_STRIDE,
        &camera,
        &palette,
        &previous_palette,
        &objects,
    );
}
