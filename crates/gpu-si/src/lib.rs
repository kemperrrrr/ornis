//! One iteration of the wide sequential-impulse contact solver, translated from
//! stable Rust through LLVM IR.
//!
//! The kernel solves a batch of up to four single-point contacts per dispatch:
//! the projected Gauss-Seidel normal row, the circular Coulomb clamp on two
//! friction axes, and the one-shot restitution pass on the last iteration. Its
//! CPU references live in `ornis-physics` — `contact_normal_step`,
//! `contact_friction_clamp` and the wide/scalar islands that call them.
//!
//! Buffers are word-addressed — [`Storage`] of `f32` — because the host uploads
//! `#[repr(C)]` records and the same buffers are read back by the CPU side:
//!
//! | binding | words per record | meaning |
//! |---|---|---|
//! | 0 | 8 | body state: velocity, pad, angular, pad |
//! | 1 | 312 | one contact batch (see [`layout`]), read-write accumulators |
//! | 2 | 4 | uniforms: `iter`, `total`, `allow_rest`, pad |
//!
//! The host dispatches one four-thread workgroup per batch, so the global
//! invocation id is `batch * 4 + lane`; the batch's `count` word retires the
//! unused lanes of the last batch.
//!
//! Every local here is a scalar. The kernel reads 77 lane groups out of the
//! batch record, and keeping the walk scalar (a `+ lane` on a constant word
//! offset, never a dynamically indexed local array) is what keeps the generated
//! WGSL identical on every host: a local array small enough for LLVM to
//! scalarize on one machine and not on another changes the artifact.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::TYPE_TAG_U32;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::cross_lane;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::global_invocation_lane;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::local_invocation_index;
use ornis_gpu_shader::sqrt;

pub mod layout;

use layout::{
    ACC, ACC_F1, ACC_F2, APPLY_N_AX, APPLY_N_AY, APPLY_N_AZ, APPLY_N_BX, APPLY_N_BY, APPLY_N_BZ,
    APPLY_W_AX, APPLY_W_AY, APPLY_W_AZ, APPLY_W_BX, APPLY_W_BY, APPLY_W_BZ, BATCH_WORDS, BIAS,
    BODY_A, BODY_B, INV_K_N, INV_K_T1, INV_K_T2, INV_MA, INV_MB, MU, NX, NY, NZ, RAX, RAY, RAZ,
    RBX, RBY, RBZ, SPEC_TARGET, T1X, T1Y, T1Z, T2X, T2Y, T2Z, W_A00, W_A01, W_A02, W_A10, W_A11,
    W_A12, W_A20, W_A21, W_A22, W_B00, W_B01, W_B02, W_B10, W_B11, W_B12, W_B20, W_B21, W_B22,
};

/// Words per body-state record: velocity, pad, angular, pad.
const BODY_WORDS: u32 = 8;
/// First word of the angular half of a body-state record.
const ANGULAR_WORD: u32 = 4;
/// Impulse magnitudes below this are treated as no correction at all.
const IMPULSE_EPS: f32 = 1e-12;
/// Lengths below this are degenerate for the friction clamp (no division).
const LENGTH_EPS: f32 = 1e-12;
/// Squared friction impulses below this leave the angular terms untouched.
const FRICTION_EPS: f32 = 1e-24;
/// Lanes per contact batch: one contact per lane, disjoint bodies per batch.
const BATCH_LANES: u32 = 4;
/// The `count` word: the last word before the record's padding.
const COUNT_WORD: u32 = BATCH_WORDS - 4;

/// Solver uniforms: the iteration index, the total, and the restitution flag.
pub struct Params;

impl GpuStruct for Params {
    /// The only uniform struct in this crate.
    const ID: u32 = 0;
    /// Three `u32` values and the padding the 16-byte block needs.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_U32; 4];
}

/// Projected Gauss-Seidel normal update for one isotropic contact.
///
/// Returns the caller's accumulation plus `(spec_target - vn) * inv_k`,
/// projected onto the push-only half-line. Mirrors
/// `ornis_physics::contact_math::contact_normal_step` exactly.
#[inline(always)]
fn normal_step(vn: f32, spec_target: f32, inv_k: f32, acc: f32) -> f32 {
    let candidate = acc + (spec_target - vn) * inv_k;
    let mut new_acc = 0.0;
    if candidate > 0.0 {
        new_acc = candidate;
    }
    new_acc
}

/// Circular Coulomb-cone clamp for one friction axis.
///
/// `new_t` is the candidate on this axis and `other` the accumulation on the
/// perpendicular one; outside the cone of radius `max_friction` the candidate
/// is scaled onto the circle. A degenerate pair passes through untouched.
/// Mirrors `ornis_physics::contact_math::contact_friction_clamp` exactly.
#[inline(always)]
fn friction_clamp(new_t: f32, other: f32, max_friction: f32) -> f32 {
    let len = sqrt(new_t * new_t + other * other);
    let mut clamped = new_t;
    if len > max_friction && len > LENGTH_EPS {
        clamped = new_t * (max_friction / len);
    }
    clamped
}

/// One solver iteration for one contact lane.
///
/// Lane `id % 4` of batch `id / 4`: gather both bodies, apply the normal row,
/// clamp the two friction axes, run the restitution pass when the host allows
/// it on the final iteration, and write both bodies and the accumulators back.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "cs_main_contact")]`.
#[gpu_entry(compute(threads(4)))]
pub fn cs_main_contact(
    #[gpu(storage_buffer, group = 0, binding = 0)] body_buf: Storage<f32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 1)] batch_buf: Storage<f32, ReadWrite>,
    #[gpu(uniform, group = 0, binding = 2)] params: Uniform<Params>,
) {
    let id = global_invocation_lane::<0>();
    let lane = local_invocation_index();
    // The host dispatches one workgroup per batch, so the global invocation id
    // counts lanes: `id - lane` is the batch's first invocation, and dividing
    // by the batch width gives its index.
    let base = (id - lane) / BATCH_LANES * BATCH_WORDS;
    if lane >= batch_buf.load(base + COUNT_WORD).to_bits() {
        return;
    }
    let iter = params.field_u32::<0>();
    let total = params.field_u32::<1>();
    let allow_rest = params.field_u32::<2>();

    // Body state bases: the batch carries global body indices as `u32` words.
    let body_a = batch_buf.load(base + BODY_A + lane).to_bits() * BODY_WORDS;
    let body_b = batch_buf.load(base + BODY_B + lane).to_bits() * BODY_WORDS;

    // Gather velocities; the accumulated impulses stay in registers until the
    // end of the invocation, so the read-after-write stays inside one lane.
    let mut va_x = body_buf.load(body_a);
    let mut va_y = body_buf.load(body_a + 1);
    let mut va_z = body_buf.load(body_a + 2);
    let mut wa_x = body_buf.load(body_a + ANGULAR_WORD);
    let mut wa_y = body_buf.load(body_a + ANGULAR_WORD + 1);
    let mut wa_z = body_buf.load(body_a + ANGULAR_WORD + 2);
    let mut vb_x = body_buf.load(body_b);
    let mut vb_y = body_buf.load(body_b + 1);
    let mut vb_z = body_buf.load(body_b + 2);
    let mut wb_x = body_buf.load(body_b + ANGULAR_WORD);
    let mut wb_y = body_buf.load(body_b + ANGULAR_WORD + 1);
    let mut wb_z = body_buf.load(body_b + ANGULAR_WORD + 2);

    let n_x = batch_buf.load(base + NX + lane);
    let n_y = batch_buf.load(base + NY + lane);
    let n_z = batch_buf.load(base + NZ + lane);
    let ra_x = batch_buf.load(base + RAX + lane);
    let ra_y = batch_buf.load(base + RAY + lane);
    let ra_z = batch_buf.load(base + RAZ + lane);
    let rb_x = batch_buf.load(base + RBX + lane);
    let rb_y = batch_buf.load(base + RBY + lane);
    let rb_z = batch_buf.load(base + RBZ + lane);
    let inv_ma = batch_buf.load(base + INV_MA + lane);
    let inv_mb = batch_buf.load(base + INV_MB + lane);
    let inv_k = batch_buf.load(base + INV_K_N + lane);
    let spec_target = batch_buf.load(base + SPEC_TARGET + lane);
    let mut acc = batch_buf.load(base + ACC + lane);

    // Normal row: relative normal velocity, then the shared projection.
    let rel_x = (vb_x + cross_lane::<0>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
        - (va_x + cross_lane::<0>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
    let rel_y = (vb_y + cross_lane::<1>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
        - (va_y + cross_lane::<1>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
    let rel_z = (vb_z + cross_lane::<2>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
        - (va_z + cross_lane::<2>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
    let vn = dot3(rel_x, rel_y, rel_z, n_x, n_y, n_z);
    let new_acc = normal_step(vn, spec_target, inv_k, acc);
    let delta = new_acc - acc;
    acc = new_acc;
    if abs(delta) > IMPULSE_EPS {
        va_x -= delta * batch_buf.load(base + APPLY_N_AX + lane);
        va_y -= delta * batch_buf.load(base + APPLY_N_AY + lane);
        va_z -= delta * batch_buf.load(base + APPLY_N_AZ + lane);
        wa_x -= delta * batch_buf.load(base + APPLY_W_AX + lane);
        wa_y -= delta * batch_buf.load(base + APPLY_W_AY + lane);
        wa_z -= delta * batch_buf.load(base + APPLY_W_AZ + lane);
        vb_x += delta * batch_buf.load(base + APPLY_N_BX + lane);
        vb_y += delta * batch_buf.load(base + APPLY_N_BY + lane);
        vb_z += delta * batch_buf.load(base + APPLY_N_BZ + lane);
        wb_x += delta * batch_buf.load(base + APPLY_W_BX + lane);
        wb_y += delta * batch_buf.load(base + APPLY_W_BY + lane);
        wb_z += delta * batch_buf.load(base + APPLY_W_BZ + lane);
    }

    // Friction: remeasure the relative velocity after the normal impulse.
    let rel2_x = (vb_x + cross_lane::<0>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
        - (va_x + cross_lane::<0>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
    let rel2_y = (vb_y + cross_lane::<1>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
        - (va_y + cross_lane::<1>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
    let rel2_z = (vb_z + cross_lane::<2>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
        - (va_z + cross_lane::<2>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
    let max_f = batch_buf.load(base + MU + lane) * acc;
    let t1_x = batch_buf.load(base + T1X + lane);
    let t1_y = batch_buf.load(base + T1Y + lane);
    let t1_z = batch_buf.load(base + T1Z + lane);
    let t2_x = batch_buf.load(base + T2X + lane);
    let t2_y = batch_buf.load(base + T2Y + lane);
    let t2_z = batch_buf.load(base + T2Z + lane);
    let mut f1 = batch_buf.load(base + ACC_F1 + lane);
    let mut f2 = batch_buf.load(base + ACC_F2 + lane);
    let mut f_x = 0.0;
    let mut f_y = 0.0;
    let mut f_z = 0.0;

    // Axis 1: the candidate depends on axis 2's accumulation, so both are read
    // before either is replaced.
    if batch_buf.load(base + INV_K_T1 + lane) > 0.0 {
        let vt1 = dot3(rel2_x, rel2_y, rel2_z, t1_x, t1_y, t1_z);
        let raw = f1 - vt1 * batch_buf.load(base + INV_K_T1 + lane);
        let new_t = friction_clamp(raw, f2, max_f);
        f_x += t1_x * (new_t - f1);
        f_y += t1_y * (new_t - f1);
        f_z += t1_z * (new_t - f1);
        f1 = new_t;
    }
    // Axis 2: the clamp sees axis 1's new accumulation.
    if batch_buf.load(base + INV_K_T2 + lane) > 0.0 {
        let vt2 = dot3(rel2_x, rel2_y, rel2_z, t2_x, t2_y, t2_z);
        let raw = f2 - vt2 * batch_buf.load(base + INV_K_T2 + lane);
        let new_t = friction_clamp(raw, f1, max_f);
        f_x += t2_x * (new_t - f2);
        f_y += t2_y * (new_t - f2);
        f_z += t2_z * (new_t - f2);
        f2 = new_t;
    }
    if dot3(f_x, f_y, f_z, f_x, f_y, f_z) > FRICTION_EPS {
        // Angular uses `r x f` with the same sign convention as the scalar
        // solver: `wa -= W·(ra×f)`, `wb += W·(rb×f)`.
        let ca_x = cross_lane::<0>(ra_x, ra_y, ra_z, f_x, f_y, f_z);
        let ca_y = cross_lane::<1>(ra_x, ra_y, ra_z, f_x, f_y, f_z);
        let ca_z = cross_lane::<2>(ra_x, ra_y, ra_z, f_x, f_y, f_z);
        let w_a_x = dot3(
            batch_buf.load(base + W_A00 + lane),
            batch_buf.load(base + W_A01 + lane),
            batch_buf.load(base + W_A02 + lane),
            ca_x,
            ca_y,
            ca_z,
        );
        let w_a_y = dot3(
            batch_buf.load(base + W_A10 + lane),
            batch_buf.load(base + W_A11 + lane),
            batch_buf.load(base + W_A12 + lane),
            ca_x,
            ca_y,
            ca_z,
        );
        let w_a_z = dot3(
            batch_buf.load(base + W_A20 + lane),
            batch_buf.load(base + W_A21 + lane),
            batch_buf.load(base + W_A22 + lane),
            ca_x,
            ca_y,
            ca_z,
        );
        let cb_x = cross_lane::<0>(rb_x, rb_y, rb_z, f_x, f_y, f_z);
        let cb_y = cross_lane::<1>(rb_x, rb_y, rb_z, f_x, f_y, f_z);
        let cb_z = cross_lane::<2>(rb_x, rb_y, rb_z, f_x, f_y, f_z);
        let w_b_x = dot3(
            batch_buf.load(base + W_B00 + lane),
            batch_buf.load(base + W_B01 + lane),
            batch_buf.load(base + W_B02 + lane),
            cb_x,
            cb_y,
            cb_z,
        );
        let w_b_y = dot3(
            batch_buf.load(base + W_B10 + lane),
            batch_buf.load(base + W_B11 + lane),
            batch_buf.load(base + W_B12 + lane),
            cb_x,
            cb_y,
            cb_z,
        );
        let w_b_z = dot3(
            batch_buf.load(base + W_B20 + lane),
            batch_buf.load(base + W_B21 + lane),
            batch_buf.load(base + W_B22 + lane),
            cb_x,
            cb_y,
            cb_z,
        );
        va_x -= f_x * inv_ma;
        va_y -= f_y * inv_ma;
        va_z -= f_z * inv_ma;
        vb_x += f_x * inv_mb;
        vb_y += f_y * inv_mb;
        vb_z += f_z * inv_mb;
        wa_x -= w_a_x;
        wa_y -= w_a_y;
        wa_z -= w_a_z;
        wb_x += w_b_x;
        wb_y += w_b_y;
        wb_z += w_b_z;
    }

    // Restitution: one shot, on the last iteration, only where the packer
    // recorded a positive bias.
    if allow_rest > 0 && iter + 1 == total {
        let bias = batch_buf.load(base + BIAS + lane);
        if bias > 0.0 {
            let rel3_x = (vb_x + cross_lane::<0>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
                - (va_x + cross_lane::<0>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
            let rel3_y = (vb_y + cross_lane::<1>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
                - (va_y + cross_lane::<1>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
            let rel3_z = (vb_z + cross_lane::<2>(wb_x, wb_y, wb_z, rb_x, rb_y, rb_z))
                - (va_z + cross_lane::<2>(wa_x, wa_y, wa_z, ra_x, ra_y, ra_z));
            let lr = (bias - dot3(rel3_x, rel3_y, rel3_z, n_x, n_y, n_z)) * inv_k;
            if lr > 0.0 {
                va_x -= lr * batch_buf.load(base + APPLY_N_AX + lane);
                va_y -= lr * batch_buf.load(base + APPLY_N_AY + lane);
                va_z -= lr * batch_buf.load(base + APPLY_N_AZ + lane);
                wa_x -= lr * batch_buf.load(base + APPLY_W_AX + lane);
                wa_y -= lr * batch_buf.load(base + APPLY_W_AY + lane);
                wa_z -= lr * batch_buf.load(base + APPLY_W_AZ + lane);
                vb_x += lr * batch_buf.load(base + APPLY_N_BX + lane);
                vb_y += lr * batch_buf.load(base + APPLY_N_BY + lane);
                vb_z += lr * batch_buf.load(base + APPLY_N_BZ + lane);
                wb_x += lr * batch_buf.load(base + APPLY_W_BX + lane);
                wb_y += lr * batch_buf.load(base + APPLY_W_BY + lane);
                wb_z += lr * batch_buf.load(base + APPLY_W_BZ + lane);
            }
        }
    }

    // Write both bodies back and persist the accumulators for the next
    // dispatch (warm start).
    body_buf.store(body_a, va_x);
    body_buf.store(body_a + 1, va_y);
    body_buf.store(body_a + 2, va_z);
    body_buf.store(body_a + ANGULAR_WORD, wa_x);
    body_buf.store(body_a + ANGULAR_WORD + 1, wa_y);
    body_buf.store(body_a + ANGULAR_WORD + 2, wa_z);
    body_buf.store(body_b, vb_x);
    body_buf.store(body_b + 1, vb_y);
    body_buf.store(body_b + 2, vb_z);
    body_buf.store(body_b + ANGULAR_WORD, wb_x);
    body_buf.store(body_b + ANGULAR_WORD + 1, wb_y);
    body_buf.store(body_b + ANGULAR_WORD + 2, wb_z);
    batch_buf.store(base + ACC + lane, acc);
    batch_buf.store(base + ACC_F1 + lane, f1);
    batch_buf.store(base + ACC_F2 + lane, f2);
}

#[cfg(test)]
mod tests {
    use super::{friction_clamp, normal_step};

    fn assert_close(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }

    /// The normal row accumulates on approach and never pulls: the projection
    /// onto the push-only half-line is what makes the row stable.
    #[test]
    fn normal_step_accumulates_and_projects() {
        assert_close(normal_step(-2.0, 0.0, 0.5, 0.0), 1.0);
        assert_close(normal_step(1.0, 0.0, 0.5, 0.0), 0.0);
        assert_close(normal_step(-1.0, 0.0, 1.0, 2.0), 3.0);
        assert_close(normal_step(-2.0, 0.0, 0.0, 1.5), 1.5);
    }

    /// The cone clamp is the identity inside the disc, scales onto the circle
    /// outside it, and never divides by a zero length.
    #[test]
    fn friction_clamp_projects_the_cone() {
        assert_close(friction_clamp(1.0, 1.0, 5.0), 1.0);
        assert_close(friction_clamp(3.0, 4.0, 2.5), 1.5);
        assert_close(friction_clamp(0.0, 0.0, 1.0), 0.0);
        assert_close(friction_clamp(2.0, 0.0, 0.0), 0.0);
    }
}
