//! Scalarized linear-blend skinning guest (issue #228): one vertex per lane.
//!
//! The compute kernel mirrors the CPU reference
//! (`ornis_animation::skin_vertices`) lane for lane: canonical weights
//! (finite positive sums normalize, everything else falls back to joint 0),
//! positions blend through each joint matrix, normals through the packed
//! per-joint inverse-transpose and renormalize unless the blend collapses.
//! An out-of-range joint index contributes the identity transform — the
//! same fallback as the CPU path — never an out-of-bounds palette read.
//!
//! Storage ABI (group 0, all `array<f32>`/`array<u32>` scalar lanes):
//! palette joints pack 32 lanes (column-major skin matrix, then its
//! inverse-transpose) — byte-identical to the production `SkinJoint`
//! staging, so the host reuses the production palette bytes verbatim.
//! Bind and output vertices pack position then normal (6 lanes).
//!
//! Everything stays in scalar locals: a local array would spill to the
//! wasm shadow stack and re-enter the lifted graph through the
//! stack-pointer global, which the translator refuses (see `vs_main`).

use ornis_gpu_abi::shader;
use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_at_u32;
use ornis_gpu_abi::storage_store_f32;

#[cfg(test)]
mod tests;

/// Storage binding of the joint palette (32 `f32` lanes per joint).
const PALETTE_BINDING: u32 = 0;
/// Storage binding of the bind-pose vertices (position, normal per vertex).
const BIND_BINDING: u32 = 1;
/// Storage binding of the per-vertex joint indices (4 per vertex).
const JOINTS_BINDING: u32 = 2;
/// Storage binding of the per-vertex influence weights (4 per vertex).
const WEIGHTS_BINDING: u32 = 3;
/// Storage binding of the skinned output vertices (position, normal).
const OUT_BINDING: u32 = 4;
/// Storage binding of the frame words: vertex count, then joint count.
const PARAMS_BINDING: u32 = 5;

/// Palette lanes per joint: skin matrix plus inverse-transpose, 16 each.
const JOINT_LANES: u32 = 32;
/// Lanes of one palette matrix (column-major 4×4).
const MATRIX_LANES: u32 = 16;
/// Vertex lanes: position `xyz`, then normal `xyz`.
const VERTEX_LANES: u32 = 6;
/// Influence lanes per vertex (top-4 joints and weights).
const MAX_INFLUENCES: u32 = 4;
/// First normal lane inside one vertex record.
const NORMAL_LANE: u32 = 3;
/// Column stride of one column-major 4×4 matrix.
const COLUMN_LANES: u32 = 4;
/// Lane offset of the translation column inside one palette matrix.
const TRANSLATION_LANE: u32 = 12;
/// Lane of the joint-count word inside the params binding.
const JOINT_COUNT_LANE: u32 = 1;
/// Weight-sum floor before normalize (same rule as the CPU path).
const NEAR_ZERO: f32 = 1e-6;
/// Squared length floor for collapsed blended normals (CPU-path rule).
const DEGENERATE_LEN2: f32 = 1e-12;
/// Upper edge of the sqrt Newton range (also its growth factor).
const SQRT_RANGE_TOP: f32 = 4.0;
/// Shrink factor bringing large sqrt inputs into the Newton range.
const SQRT_SHRINK: f32 = 0.25;
/// Secant-seed divisor through the Newton range endpoints (1, 1), (4, 2).
const SQRT_SEED_DIVISOR: f32 = 3.0;
/// Newton iterations after the secant seed (relative error below 1e-6).
const SQRT_STEPS: u32 = 4;
/// Cap on range-reduction steps: every finite `f32` needs at most ~76
/// (subnormals up, `f32::MAX` down). The cap exists only so a stray
/// non-finite input terminates instead of spinning the kernel forever.
const SQRT_RANGE_STEPS: u32 = 128;
/// One half, for the Newton mean and the inverse range scale.
const HALF_F32: f32 = 0.5;
/// Finiteness bound for kernel guards: magnitudes at or above this are
/// treated as non-finite (weight sums, squared normal lengths).
const MAX_FINITE_SUM: f32 = 3.4e38;

/// Wasm compute entry: skins vertex `id` into the output binding.
///
/// One lane owns one vertex, so lanes past the vertex count (the final
/// partial workgroup) do nothing. Joint indices at or past the joint
/// count accumulate into an identity-transform weight, which is applied
/// to the bind vertex after the loop — arithmetically the CPU path's
/// per-slot identity fallback.
#[shader::compute(workgroup_size = 64)]
pub fn skin(id: u32) {
    let vertex_count = storage_at_u32::<0, PARAMS_BINDING>(0);
    if id < vertex_count {
        skin_vertex(id);
    }
}

#[inline(always)]
fn skin_vertex(id: u32) {
    let joint_count = storage_at_u32::<0, PARAMS_BINDING>(JOINT_COUNT_LANE);
    let bind = id * VERTEX_LANES;
    let pos_x = storage_at_f32::<0, BIND_BINDING>(bind);
    let pos_y = storage_at_f32::<0, BIND_BINDING>(bind + 1);
    let pos_z = storage_at_f32::<0, BIND_BINDING>(bind + 2);
    let nrm_x = storage_at_f32::<0, BIND_BINDING>(bind + NORMAL_LANE);
    let nrm_y = storage_at_f32::<0, BIND_BINDING>(bind + NORMAL_LANE + 1);
    let nrm_z = storage_at_f32::<0, BIND_BINDING>(bind + NORMAL_LANE + 2);
    let scale = weight_scale(id);
    let mut out_x = 0.0_f32;
    let mut out_y = 0.0_f32;
    let mut out_z = 0.0_f32;
    let mut out_nx = 0.0_f32;
    let mut out_ny = 0.0_f32;
    let mut out_nz = 0.0_f32;
    let mut identity = 0.0_f32;
    let mut slot = 0_u32;
    while slot < MAX_INFLUENCES {
        let joint = storage_at_u32::<0, JOINTS_BINDING>(id * MAX_INFLUENCES + slot);
        let weight = slot_weight(id, slot, scale);
        if joint < joint_count {
            let base = joint * JOINT_LANES;
            out_x += weight * position_row(base, pos_x, pos_y, pos_z, 0);
            out_y += weight * position_row(base, pos_x, pos_y, pos_z, 1);
            out_z += weight * position_row(base, pos_x, pos_y, pos_z, 2);
            out_nx += weight * normal_row(base, nrm_x, nrm_y, nrm_z, 0);
            out_ny += weight * normal_row(base, nrm_x, nrm_y, nrm_z, 1);
            out_nz += weight * normal_row(base, nrm_x, nrm_y, nrm_z, 2);
        } else {
            identity += weight;
        }
        slot += 1;
    }
    out_x += identity * pos_x;
    out_y += identity * pos_y;
    out_z += identity * pos_z;
    out_nx += identity * nrm_x;
    out_ny += identity * nrm_y;
    out_nz += identity * nrm_z;
    // Renormalize unless the blended normal collapsed (CPU-path rule) or
    // its squared length overflowed: a non-finite `len2` (huge joint
    // scale, corrupt palette) must skip the sqrt entirely, never spin in
    // its range reduction.
    let len2 = out_nx * out_nx + out_ny * out_ny + out_nz * out_nz;
    let normal_scale = if len2 > DEGENERATE_LEN2 && len2 < MAX_FINITE_SUM {
        1.0 / sqrt_approx(len2)
    } else {
        1.0
    };
    storage_store_f32::<0, OUT_BINDING>(bind, out_x);
    storage_store_f32::<0, OUT_BINDING>(bind + 1, out_y);
    storage_store_f32::<0, OUT_BINDING>(bind + 2, out_z);
    storage_store_f32::<0, OUT_BINDING>(bind + NORMAL_LANE, out_nx * normal_scale);
    storage_store_f32::<0, OUT_BINDING>(bind + NORMAL_LANE + 1, out_ny * normal_scale);
    storage_store_f32::<0, OUT_BINDING>(bind + NORMAL_LANE + 2, out_nz * normal_scale);
}

/// One row of `matrix * vec4(position, 1)` over the palette lanes.
#[inline(always)]
fn position_row(base: u32, x: f32, y: f32, z: f32, row: u32) -> f32 {
    dot_row(base, x, y, z, row)
        + storage_at_f32::<0, PALETTE_BINDING>(base + TRANSLATION_LANE + row)
}

/// One row of `normal_matrix * vec4(normal, 0)` over the palette lanes.
#[inline(always)]
fn normal_row(base: u32, x: f32, y: f32, z: f32, row: u32) -> f32 {
    dot_row(base + MATRIX_LANES, x, y, z, row)
}

/// Range-reduced square root on scalar operations the lifter supports.
///
/// The guest crate is `no_std` on wasm32, where `f32::sqrt` does not
/// exist and `core::arch::wasm32::f32_sqrt` is still unstable — same
/// constraint as `log2_approx`/`exp2_approx` in the auto-exposure math.
/// The value is reduced by powers of four into `[1, 4)`, Newton-iterated
/// from a secant seed, then scaled back by powers of two.
///
/// Total over all inputs: NaN yields NaN and `+inf` yields `+inf`, and
/// the reduction loops are step-capped ([`SQRT_RANGE_STEPS`]), so even a
/// stray non-finite value terminates. The kernel still guards finiteness
/// at the call site; accuracy is only promised for positive finite input.
#[inline(always)]
fn sqrt_approx(value: f32) -> f32 {
    let mut mantissa = value;
    let mut scale = 1.0_f32;
    let mut range_step = 0;
    while mantissa >= SQRT_RANGE_TOP && range_step < SQRT_RANGE_STEPS {
        mantissa *= SQRT_SHRINK;
        scale *= 2.0;
        range_step += 1;
    }
    range_step = 0;
    while mantissa < 1.0 && range_step < SQRT_RANGE_STEPS {
        mantissa *= SQRT_RANGE_TOP;
        scale *= HALF_F32;
        range_step += 1;
    }
    // Secant seed through (1, 1) and (4, 2); Newton refines it quadratically.
    let mut guess = (mantissa + 2.0) / SQRT_SEED_DIVISOR;
    let mut step = 0;
    while step < SQRT_STEPS {
        guess = HALF_F32 * (guess + mantissa / guess);
        step += 1;
    }
    guess * scale
}

/// The linear part of one palette-matrix row against a direction.
#[inline(always)]
fn dot_row(base: u32, x: f32, y: f32, z: f32, row: u32) -> f32 {
    storage_at_f32::<0, PALETTE_BINDING>(base + row) * x
        + storage_at_f32::<0, PALETTE_BINDING>(base + COLUMN_LANES + row) * y
        + storage_at_f32::<0, PALETTE_BINDING>(base + COLUMN_LANES * 2 + row) * z
}

/// Canonical-weight multiplier of one vertex: `1 / sum` for a finite
/// positive sum, `0.0` as the fallback marker [`slot_weight`] reads.
#[inline(always)]
fn weight_scale(id: u32) -> f32 {
    let base = id * MAX_INFLUENCES;
    let mut sum = 0.0_f32;
    let mut slot = 0_u32;
    while slot < MAX_INFLUENCES {
        sum += storage_at_f32::<0, WEIGHTS_BINDING>(base + slot);
        slot += 1;
    }
    if sum > NEAR_ZERO && sum < MAX_FINITE_SUM {
        1.0 / sum
    } else {
        0.0
    }
}

/// Canonical weight of one influence slot.
///
/// A zero `scale` is the fallback marker: full weight on slot 0, exactly
/// the CPU canonicalization of a non-finite or collapsed weight sum.
#[inline(always)]
fn slot_weight(id: u32, slot: u32, scale: f32) -> f32 {
    if scale == 0.0 {
        if slot == 0 { 1.0 } else { 0.0 }
    } else {
        storage_at_f32::<0, WEIGHTS_BINDING>(id * MAX_INFLUENCES + slot) * scale
    }
}
