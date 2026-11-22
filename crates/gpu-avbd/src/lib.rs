//! AVBD inertial solver rung 1: the device-side inertial Hessian diagonal and
//! its diagonal LDL solve, translated from stable Rust through LLVM IR.
//!
//! Rung 1 is the part of the AVBD solver that needs no contacts: one
//! invocation per body reads its mass roster entry, assembles the 6-entry
//! lumped Hessian diagonal (`m/dt²` linear, body-frame `I/dt²` angular) and
//! solves that diagonal system against the rhs entry into the delta buffer.
//! The CPU references it must agree with live in `ornis-physics`
//! (`avbd_inertial_hessian_diag`, `avbd_diag_solve_cpu`).
//!
//! Buffers are word-addressed — [`Storage`] of `f32` — because the host uploads
//! `#[repr(C)]` records, exactly as the hand written kernel read them:
//!
//! | binding | words | meaning |
//! |---|---|---|
//! | 0 | `inertia[3]`, `inv_mass` | mass roster, one record per body |
//! | 1 | `velocity[3]`, pad, `angular[3]`, pad | base residual |
//! | 2 | same as 1 | solved delta (read-write) |
//! | 3 | `count` + pad | bodies to solve |
//! | 4 | `dt` + pad | timestep |
//!
//! Statics (`inv_mass <= 0`) and a degenerate timestep assemble a zero Hessian,
//! and the diagonal solve zeroes those lanes instead of dividing — the shader
//! has no `Option`, so "no solution" and "solves to zero" are the same answer on
//! both sides of the comparison.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::ReadWrite;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::TYPE_TAG_U32;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::global_invocation_lane;
use ornis_gpu_shader::gpu_entry;

/// Body count uniform: the count and three pad words.
pub struct BodyCount;

impl GpuStruct for BodyCount {
    /// The count uniform is struct 0.
    const ID: u32 = 0;
    /// One `u32` and the padding the 16-byte uniform block needs.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_U32; 4];
}

/// Timestep uniform: `dt` and three pad words.
pub struct Timestep;

impl GpuStruct for Timestep {
    /// The timestep uniform is struct 1.
    const ID: u32 = 1;
    /// One `f32` and the padding the 16-byte uniform block needs.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_F32; 4];
}

/// Words per mass roster record: `inertia` (`vec3<f32>`) then `inv_mass`.
const MASS_WORDS: u32 = 4;
/// Words per body-state record: velocity, pad, angular, pad.
const STATE_WORDS: u32 = 8;
/// Word of `inv_mass` inside a mass record.
const INV_MASS_WORD: u32 = 3;
/// First word of the angular half of a body-state record.
const ANGULAR_WORD: u32 = 4;
/// Timesteps at or below this magnitude are degenerate.
const DEGENERATE_DT2: f32 = 1e-12;

/// Linear Hessian entry `m/dt²`, or `0.0` for statics and a degenerate `dt`.
#[inline(always)]
fn hessian_lin(inv_mass: f32, dt: f32) -> f32 {
    let mut h = 0.0;
    if inv_mass > 0.0 {
        let dt2 = dt * dt;
        if dt2 > DEGENERATE_DT2 {
            h = 1.0 / (inv_mass * dt2);
        }
    }
    h
}

/// Angular Hessian entry `I_lane/dt²`, or `0.0` for a degenerate `dt`.
#[inline(always)]
fn hessian_ang(inertia: f32, dt: f32) -> f32 {
    let dt2 = dt * dt;
    let mut h = 0.0;
    if dt2 > DEGENERATE_DT2 {
        h = inertia / dt2;
    }
    h
}

/// Diagonal LDL solve of one lane: `r / h`, or `0.0` on a non-positive pivot.
#[inline(always)]
fn diag_solve(h: f32, r: f32) -> f32 {
    let mut x = 0.0;
    if h > DEGENERATE_DT2 {
        x = r / h;
    }
    x
}

/// Rung-1 kernel: inertial Hessian diagonal plus diagonal LDL, one body per
/// invocation.
///
/// The DSL version indexed by `workgroup_id`, which left 63 of every 64
/// threads idle and solved only the first `workgroups` bodies; this one uses
/// the global invocation index, so `ceil(count / 64)` workgroups cover the
/// roster.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "cs_main")]`.
#[gpu_entry(compute(threads(64)))]
pub fn cs_main(
    #[gpu(storage_buffer, group = 0, binding = 0)] mass: Storage<f32>,
    #[gpu(storage_buffer, group = 0, binding = 1)] rhs: Storage<f32>,
    #[gpu(storage_buffer, group = 0, binding = 2)] delta: Storage<f32, ReadWrite>,
    #[gpu(uniform, group = 0, binding = 3)] count: Uniform<BodyCount>,
    #[gpu(uniform, group = 0, binding = 4)] dt_uniform: Uniform<Timestep>,
) {
    let index = global_invocation_lane::<0>();
    if index >= count.field_u32::<0>() {
        return;
    }
    let mass_base = index * MASS_WORDS;
    let dt = dt_uniform.field_f32::<0>();
    let linear = hessian_lin(mass.load(mass_base + INV_MASS_WORD), dt);
    let state_base = index * STATE_WORDS;
    delta.store(state_base, diag_solve(linear, rhs.load(state_base)));
    delta.store(state_base + 1, diag_solve(linear, rhs.load(state_base + 1)));
    delta.store(state_base + 2, diag_solve(linear, rhs.load(state_base + 2)));
    let ang = hessian_ang(mass.load(mass_base), dt);
    let ang_y = hessian_ang(mass.load(mass_base + 1), dt);
    let ang_z = hessian_ang(mass.load(mass_base + 2), dt);
    let angular = state_base + ANGULAR_WORD;
    delta.store(angular, diag_solve(ang, rhs.load(angular)));
    delta.store(angular + 1, diag_solve(ang_y, rhs.load(angular + 1)));
    delta.store(angular + 2, diag_solve(ang_z, rhs.load(angular + 2)));
}

/// Words per staged system record: mass (4) + body state (8) + contact row (12).
const SYSTEM_WORDS: u32 = 24;
/// Word of `state.velocity.x` inside a system record.
const STATE_WORD: u32 = 4;
/// Word of `row.axis.x` inside a system record.
const ROW_WORD: u32 = 12;
/// Word of `row.pen` inside a system record.
const ROW_PEN_WORD: u32 = ROW_WORD + 3;
/// Word of `row.lever.x` inside a system record.
const ROW_LEVER_WORD: u32 = ROW_WORD + 4;
/// Word of `row.force` inside a system record.
const ROW_FORCE_WORD: u32 = ROW_WORD + 7;
/// Word of `row.sign` inside a system record.
const ROW_SIGN_WORD: u32 = ROW_WORD + 8;
/// Spatial degrees of freedom the system solves for.
const DOF: usize = 6;
/// Entries in the dense `6×6` system.
const MATRIX_ELEMS: usize = DOF * DOF;

/// `h[row * DOF + col]`, the dense row-major accessor both sides spell.
#[inline(always)]
fn at(matrix: &[f32; MATRIX_ELEMS], row: u32, col: u32) -> f32 {
    matrix[(row * DOF as u32 + col) as usize]
}

/// Write `h[row * DOF + col]`.
#[inline(always)]
fn put(matrix: &mut [f32; MATRIX_ELEMS], row: u32, col: u32, value: f32) {
    matrix[(row * DOF as u32 + col) as usize] = value;
}

/// One contact row stamped into the dense 6×6 system, exactly as
/// `avbd_stamp_row_cpu` does: `nn = sign·axis`, `t = lever × nn`, and the
/// penalty couples each (linear, angular) pair both ways.
#[inline(always)]
fn stamp(
    h: &mut [f32; MATRIX_ELEMS],
    b: &mut [f32; DOF],
    na: [f32; 3],
    ta: [f32; 3],
    pen: f32,
    force: f32,
) {
    let mut x = 0u32;
    while x < 3 {
        let mut y = 0u32;
        while y < 3 {
            let nx = na[x as usize];
            let ny = na[y as usize];
            let tx = ta[x as usize];
            let ty = ta[y as usize];
            put(h, x, y, at(h, x, y) + pen * nx * ny);
            put(h, 3 + x, 3 + y, at(h, 3 + x, 3 + y) + pen * tx * ty);
            put(h, x, 3 + y, at(h, x, 3 + y) + pen * nx * ty);
            put(h, 3 + x, y, at(h, 3 + x, y) + pen * tx * ny);
            y += 1;
        }
        b[x as usize] = b[x as usize] + force * na[x as usize];
        b[(3 + x) as usize] = b[(3 + x) as usize] + force * ta[x as usize];
        x += 1;
    }
}

/// Dense LDLᵀ solve of the stamped system, mirroring `solve_6x6`: no
/// pivoting, and a non-positive pivot reports breakdown instead of producing a
/// usable answer.
///
/// The caller resolves the flag: on breakdown the delta is zeroed, because the
/// partial factorization holds garbage. Division by a non-positive pivot is
/// well-defined `f32` arithmetic here (never a trap), and its result is
/// discarded.
#[inline(always)]
fn solve(h: &[f32; MATRIX_ELEMS], b: &[f32; DOF], out: &mut [f32; DOF]) -> f32 {
    let mut l = [0.0; MATRIX_ELEMS];
    let mut d = [0.0f32; DOF];
    let mut ok = 1.0;
    let mut i = 0u32;
    while i < DOF as u32 {
        let mut j = 0u32;
        while j < DOF as u32 {
            if j <= i {
                let mut s = at(h, i, j);
                let mut k = 0u32;
                while k < DOF as u32 {
                    if k < j {
                        s = s - at(&l, i, k) * d[k as usize] * at(&l, j, k);
                    }
                    k += 1;
                }
                if i == j {
                    if s <= DEGENERATE_DT2 {
                        ok = 0.0;
                    }
                    d[i as usize] = s;
                    put(&mut l, i, i, 1.0);
                }
                if i != j {
                    put(&mut l, i, j, s / d[j as usize]);
                }
            }
            j += 1;
        }
        i += 1;
    }
    // Forward substitution `L y = b`, then `D z = y`.
    let mut y = [0.0f32; DOF];
    let mut z = [0.0f32; DOF];
    let mut i = 0u32;
    while i < DOF as u32 {
        let mut s = b[i as usize];
        let mut k = 0u32;
        while k < DOF as u32 {
            if k < i {
                s = s - at(&l, i, k) * y[k as usize];
            }
            k += 1;
        }
        y[i as usize] = s;
        z[i as usize] = s / d[i as usize];
        i += 1;
    }
    // Back substitution runs top-down: the translator lowers increasing
    // ranges, so count down through index arithmetic.
    let mut r = 0u32;
    while r < DOF as u32 {
        let i = DOF as u32 - 1 - r;
        let mut s = z[i as usize];
        let mut k = 0u32;
        while k < DOF as u32 {
            if k > i {
                s = s - at(&l, k, i) * out[k as usize];
            }
            k += 1;
        }
        out[i as usize] = s;
        r += 1;
    }
    ok
}

/// Rung-2 kernel: inertial Hessian, one stamped contact row and a dense
/// `6×6` LDL per body, one invocation per staged system.
///
/// The flag buffer carries the breakdown signal (`1.0` solved, `0.0` with a
/// zeroed delta), so the host can mirror the CPU `Option` without reading a
/// sentinel out of the solution.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "cs_main_row")]`.
#[gpu_entry(compute(threads(64)))]
pub fn cs_main_row(
    #[gpu(storage_buffer, group = 0, binding = 0)] systems: Storage<f32>,
    #[gpu(storage_buffer, group = 0, binding = 1)] delta: Storage<f32, ReadWrite>,
    #[gpu(storage_buffer, group = 0, binding = 2)] ok_flags: Storage<f32, ReadWrite>,
    #[gpu(uniform, group = 0, binding = 3)] count: Uniform<BodyCount>,
    #[gpu(uniform, group = 0, binding = 4)] dt_uniform: Uniform<Timestep>,
) {
    let index = global_invocation_lane::<0>();
    if index >= count.field_u32::<0>() {
        return;
    }
    let base = index * SYSTEM_WORDS;
    let dt = dt_uniform.field_f32::<0>();
    let linear = hessian_lin(systems.load(base + INV_MASS_WORD), dt);
    let mut h = [0.0f32; MATRIX_ELEMS];
    put(&mut h, 0, 0, linear);
    put(&mut h, 1, 1, linear);
    put(&mut h, 2, 2, linear);
    put(&mut h, 3, 3, hessian_ang(systems.load(base), dt));
    put(&mut h, 4, 4, hessian_ang(systems.load(base + 1), dt));
    put(&mut h, 5, 5, hessian_ang(systems.load(base + 2), dt));
    let mut b = [0.0f32; DOF];
    b[0] = systems.load(base + STATE_WORD);
    b[1] = systems.load(base + STATE_WORD + 1);
    b[2] = systems.load(base + STATE_WORD + 2);
    b[3] = systems.load(base + STATE_WORD + ANGULAR_WORD);
    b[4] = systems.load(base + STATE_WORD + ANGULAR_WORD + 1);
    b[5] = systems.load(base + STATE_WORD + ANGULAR_WORD + 2);
    // `nn = sign·axis`, `t = lever × nn` — written lane by lane so no helper
    // returns an aggregate.
    let sign = systems.load(base + ROW_SIGN_WORD);
    let na = [
        systems.load(base + ROW_WORD) * sign,
        systems.load(base + ROW_WORD + 1) * sign,
        systems.load(base + ROW_WORD + 2) * sign,
    ];
    let lever = [
        systems.load(base + ROW_LEVER_WORD),
        systems.load(base + ROW_LEVER_WORD + 1),
        systems.load(base + ROW_LEVER_WORD + 2),
    ];
    let ta = [
        lever[1] * na[2] - lever[2] * na[1],
        lever[2] * na[0] - lever[0] * na[2],
        lever[0] * na[1] - lever[1] * na[0],
    ];
    stamp(
        &mut h,
        &mut b,
        na,
        ta,
        systems.load(base + ROW_PEN_WORD),
        systems.load(base + ROW_FORCE_WORD),
    );
    let mut x = [0.0f32; DOF];
    let ok = solve(&h, &b, &mut x);
    let state = index * STATE_WORDS;
    if ok > 0.5 {
        delta.store(state, x[0]);
        delta.store(state + 1, x[1]);
        delta.store(state + 2, x[2]);
        delta.store(state + ANGULAR_WORD, x[3]);
        delta.store(state + ANGULAR_WORD + 1, x[4]);
        delta.store(state + ANGULAR_WORD + 2, x[5]);
        ok_flags.store(index, 1.0);
    }
    if ok <= 0.5 {
        delta.store(state, 0.0);
        delta.store(state + 1, 0.0);
        delta.store(state + 2, 0.0);
        delta.store(state + ANGULAR_WORD, 0.0);
        delta.store(state + ANGULAR_WORD + 1, 0.0);
        delta.store(state + ANGULAR_WORD + 2, 0.0);
        ok_flags.store(index, 0.0);
    }
}
