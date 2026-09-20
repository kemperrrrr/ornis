//! GPU sequential-impulse accelerator for wide contact batches (G7) — `gpu` feature only.
//!
//! Offloads single-point contact constraint solving to the GPU through wgpu
//! compute shaders. Each workgroup (4 invocations) processes one wide batch
//! of up to 4 single-point contacts — the same batching strategy as the CPU
//! SIMD-wide path (`WideBatch`), but running on the GPU.
//!
//! # Rust-authored shader (no hand-written WGSL)
//!
//! Following the engine-wide CPU↔GPU idea (IDEAS.md #4), the compute shader
//! is written in Rust and translated to WGSL at compile time:
//!
//! - `GpuBodyState` / `GpuBatch` carry `#[derive(WgslStruct)]`: the WGSL
//!   struct declarations are generated from the Rust field lists, and the
//!   derive emits compile-time `offset_of!`/`size_of` assertions that the
//!   `repr(C)` layout matches WGSL alignment rules (vec3 fields must be
//!   padded explicitly — WGSL aligns `vec3<f32>` to 16 bytes). The Rust
//!   structs are the single source of truth for the buffer layout.
//! - `contact_solver` is a `#[gpu_pipeline(...)]` function whose body is the
//!   compute entry point, written in the kernel DSL; the macro generates
//!   the bindings, built-in parameters and `fn main` around it.
//!
//! The assembled source is validated by naga in tests (no device required)
//! and executed against a software adapter (`gpu_solver_*` tests).
//!
//! # Hybrid model
//!
//! Multi-point (block-LCP) manifolds stay on the CPU island path. The GPU
//! and CPU pass are NOT interleaved at Gauss-Seidel granularity — they run
//! sequentially per iteration. This is a Jacobi/GS hybrid that converges
//! slightly differently from the pure CPU path. Consequently the GPU path
//! is NOT bit-identical to the CPU solver. It is off by default and intended
//! for visual-scale scenes where the Strong-Confluence CPU path is adequate
//! for deterministic simulation and the GPU accelerates the visual bulk.
//!
//! # Bulk dispatch (v2)
//!
//! `solve` records all iterations as back-to-back compute passes in ONE
//! command encoder (per-pass params via dynamic uniform offsets) instead of
//! one submit + blocking wait per iteration. Passes on a single queue keep
//! dispatch-boundary memory visibility, so the Jacobi consistency model is
//! unchanged — only the N−1 CPU round-trips are gone. The shader row calls
//! the shared `contact_math` kernels (normal + friction clamp) stitched via
//! `helpers(...)` — one source of truth with the CPU wide/scalar paths;
//! anisotropic and rolling coefficients stay CPU-only.
//!
//! A true AVBD port (affine bodies, per-body Hessian assembly + LDL in the
//! shader) lands rung by rung: rung 1 (here) assembles the lumped inertial
//! Hessian diagonal and solves the diagonal LDL system per body in
//! `avbd_stub_kernel` (helpers stitched via `helpers(...)`); contact rows,
//! the dense 6x6 LDL and device execution stay future work. The DSL
//! preconditions are closed: `ShaderType::Mat3` (`mat3x3<f32>`,
//! `Mat3::from_cols` constructor, `Mat3::IDENTITY`/`ZERO`), local
//! fixed-size scratch arrays (incl. nested Hessian shapes; effectful
//! repeats rejected loudly), and `helpers(...)` inclusion in
//! `#[gpu_pipeline]` (stitches `#[wgsl_fn]`/`#[kernel]` sources ahead of
//! the entry). Pinned by `macros/tests/compute_dsl.rs` and
//! `helpers_stitch_ahead_of_main_and_validate`.

use glam::Vec3;
use ornis_macros::{WgslStruct, gpu_pipeline, wgsl_fn};
use std::sync::Arc;

use crate::avbd::AvbdEngine;
use crate::body::{BodyType, RigidBody};
use crate::contact_math::{contact_friction_clamp, contact_normal_step};
use crate::engine::{Manifold, ManifoldState, PhysicsEngine};
use bytemuck::Zeroable;

// ---------------------------------------------------------------------------
// Buffer strides (verified against the WGSL layout by WgslStruct)
// ---------------------------------------------------------------------------

/// Number of bytes per GPU body state. The value and the per-field offsets
/// are checked against the generated WGSL layout at compile time.
pub const GPU_BODY_STRIDE: u64 = std::mem::size_of::<GpuBodyState>() as u64;

/// Number of bytes per GPU batch (see `GpuBatch`; same compile-time check).
pub const GPU_BATCH_STRIDE: u64 = std::mem::size_of::<GpuBatch>() as u64;

/// Maximum solver passes per [`GpuSequentialImpulse::solve`] call (bulk
/// dispatch uploads one params entry per pass; 8 velocity iterations ×
/// substeps never approach this — it is a buffer-size bound, not a
/// physics bound).
const PARAMS_CAP: u64 = 64;

/// Per-pass solver params `(iter, total, allow_rest, 0)` for a bulk
/// dispatch: pass `k` reads entry `k`, so the shader sees the same
/// per-iteration values as the old one-dispatch-per-iteration loop.
/// Pure (no device) so unit tests pin the layout.
pub fn solve_params(iterations: u32, allow_restitution: bool) -> Vec<[u32; 4]> {
    (0..iterations)
        .map(|k| [k, iterations, u32::from(allow_restitution), 0])
        .collect()
}

// ---------------------------------------------------------------------------
// GPU body state (32 bytes; WGSL declaration generated by WgslStruct)
// ---------------------------------------------------------------------------

/// GPU copy of the solver-relevant body state: linear + angular velocity.
///
/// The explicit `_pad_*` fields mirror WGSL's 16-byte `vec3<f32>` alignment;
/// the WGSL struct declaration is generated from this layout.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
pub struct GpuBodyState {
    /// Linear velocity.
    pub velocity: [f32; 3],
    _pad_v: f32,
    /// Angular velocity.
    pub angular: [f32; 3],
    _pad_w: f32,
}

impl GpuBodyState {
    fn from_body(b: &RigidBody) -> Self {
        Self {
            velocity: b.velocity.to_array(),
            angular: b.angular_velocity.to_array(),
            _pad_v: 0.0,
            _pad_w: 0.0,
        }
    }

    fn write_to_body(&self, b: &mut RigidBody) {
        b.velocity = Vec3::from_array(self.velocity);
        b.angular_velocity = Vec3::from_array(self.angular);
    }
}

// ---------------------------------------------------------------------------
// AVBD inertial mass roster (rung 1 device input)
// ---------------------------------------------------------------------------

/// Per-body mass roster for the AVBD device rung: body-frame inertia
/// diagonal + inverse mass. Field order is layout-driven: `inertia`
/// (`vec3<f32>`, 16-aligned) first so no padding is needed — the struct is
/// exactly 16 bytes (see the layout test below).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
pub struct GpuAvbdMass {
    /// Body-frame inertia diagonal.
    pub inertia: [f32; 3],
    /// Inverse mass (0 for non-dynamics).
    pub inv_mass: f32,
}

impl GpuAvbdMass {
    fn from_body(b: &RigidBody) -> Self {
        if b.body_type == BodyType::Dynamic {
            Self {
                inertia: b.inertia.to_array(),
                inv_mass: b.inv_mass,
            }
        } else {
            Self {
                inertia: [0.0; 3],
                inv_mass: 0.0,
            }
        }
    }
}

/// Inertial Hessian diagonal for one AVBD body (rung-1 lumped/Kurtz-style
/// approximation): `[m/dt² × 3, I/dt² × 3]`.
///
/// This is the exact diagonal of the inertial block [`crate::avbd`]'s
/// `solve_body` assembles before contact rows for the linear part
/// (`m/dt²` with `m = 1/inv_mass`); the angular part keeps the body-frame
/// diagonal and drops the world off-diagonal coupling, so it is exact for
/// isotropic bodies (spheres/cubes) and a documented approximation
/// otherwise. Statics (`inv_mass <= 0`) and non-positive `dt` yield zeros.
/// Pure (no device) so unit tests pin it without an adapter.
pub fn avbd_inertial_hessian_diag(inv_mass: f32, inertia: [f32; 3], dt: f32) -> [f32; 6] {
    // Non-positive or non-finite inputs assemble nothing (matches the
    // shader guards, which test the positive form and zero otherwise —
    // NaN fails both spellings and lands on zeros either way).
    if !inv_mass.is_finite() || inv_mass <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return [0.0; 6];
    }
    let dt2 = dt * dt;
    if !dt2.is_finite() || dt2 <= 1e-12 {
        return [0.0; 6];
    }
    let lin = 1.0 / (inv_mass * dt2);
    [
        lin,
        lin,
        lin,
        inertia[0] / dt2,
        inertia[1] / dt2,
        inertia[2] / dt2,
    ]
}

/// Diagonal LDL solve (L = I, D = diag): `x[i] = rhs[i] / diag[i]`,
/// degenerate axes (`<= 1e-12`) solve to `0.0` — statics and sleepers
/// (zeroed mass model) carry no correction.
///
/// This mirrors the `avbd_diag_solve` shader helper exactly (WGSL has no
/// `Option`); on strictly positive systems it agrees with the dense AVBD
/// LDL within float tolerance (pinned by test, never bit-identical by
/// promise). Full 6x6 device LDL with breakdown signaling is a later rung.
pub fn avbd_diag_solve_cpu(diag: [f32; 6], rhs: [f32; 6]) -> [f32; 6] {
    let mut out = [0.0f32; 6];
    for i in 0..6 {
        if diag[i] > 1e-12 {
            out[i] = rhs[i] / diag[i];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// GPU batch (SoA, 4 lanes; WGSL declaration generated by WgslStruct)
// ---------------------------------------------------------------------------

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
/// One GPU contact batch: up to 4 single-point lanes over disjoint bodies.
pub struct GpuBatch {
    // Geometry: SoA layout, 4 lanes each → [f32; 4]
    nx: [f32; 4],
    ny: [f32; 4],
    nz: [f32; 4],
    rax: [f32; 4],
    ray: [f32; 4],
    raz: [f32; 4],
    rbx: [f32; 4],
    rby: [f32; 4],
    rbz: [f32; 4],
    ra_nx: [f32; 4],
    ra_ny: [f32; 4],
    ra_nz: [f32; 4],
    rb_nx: [f32; 4],
    rb_ny: [f32; 4],
    rb_nz: [f32; 4],
    t1x: [f32; 4],
    t1y: [f32; 4],
    t1z: [f32; 4],
    t2x: [f32; 4],
    t2y: [f32; 4],
    t2z: [f32; 4],
    ra_t1x: [f32; 4],
    ra_t1y: [f32; 4],
    ra_t1z: [f32; 4],
    rb_t1x: [f32; 4],
    rb_t1y: [f32; 4],
    rb_t1z: [f32; 4],
    ra_t2x: [f32; 4],
    ra_t2y: [f32; 4],
    ra_t2z: [f32; 4],
    rb_t2x: [f32; 4],
    rb_t2y: [f32; 4],
    rb_t2z: [f32; 4],
    // Scalar constants per lane
    inv_ma: [f32; 4],
    inv_mb: [f32; 4],
    total_inv: [f32; 4],
    inv_k_n: [f32; 4],
    inv_k_t1: [f32; 4],
    inv_k_t2: [f32; 4],
    // Linear application factors: n * inv_mass
    apply_n_ax: [f32; 4],
    apply_n_ay: [f32; 4],
    apply_n_az: [f32; 4],
    apply_n_bx: [f32; 4],
    apply_n_by: [f32; 4],
    apply_n_bz: [f32; 4],
    // Angular application factors: I⁻¹_world · (ra×n)  (precomputed)
    apply_w_ax: [f32; 4],
    apply_w_ay: [f32; 4],
    apply_w_az: [f32; 4],
    apply_w_bx: [f32; 4],
    apply_w_by: [f32; 4],
    apply_w_bz: [f32; 4],
    // World-space inverse inertia matrix rows (3 rows × 2 bodies)
    w_a00: [f32; 4],
    w_a01: [f32; 4],
    w_a02: [f32; 4],
    w_a10: [f32; 4],
    w_a11: [f32; 4],
    w_a12: [f32; 4],
    w_a20: [f32; 4],
    w_a21: [f32; 4],
    w_a22: [f32; 4],
    w_b00: [f32; 4],
    w_b01: [f32; 4],
    w_b02: [f32; 4],
    w_b10: [f32; 4],
    w_b11: [f32; 4],
    w_b12: [f32; 4],
    w_b20: [f32; 4],
    w_b21: [f32; 4],
    w_b22: [f32; 4],
    // Restitution bias, speculative target, friction coeff
    bias: [f32; 4],
    // `target` is a reserved WGSL keyword — the member is named `spec_target`.
    spec_target: [f32; 4],
    mu: [f32; 4],
    // Body indices
    body_a: [u32; 4],
    body_b: [u32; 4],
    // Accumulators (read-write on GPU)
    /// Accumulated normal impulse per lane.
    pub acc: [f32; 4],
    acc_f1: [f32; 4],
    acc_f2: [f32; 4],
    // Active lane count
    /// Active lane count.
    pub count: u32,
    // Explicit tail padding: keeps the struct a multiple of its 16-byte
    // alignment (WGSL stride) without implicit padding, which `Pod` rejects.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// One lane's worth of CPU-side contact data for `GpuBatch::fill_lane`
/// (mirrors the WGSL batch layout: one scalar per field).
pub struct LaneInput<'a> {
    /// Lane index within the batch (0..4).
    pub lane: usize,
    /// Contact normal (body A to body B).
    pub n: Vec3,
    /// Contact offset from body A's center.
    pub ra: Vec3,
    /// Contact offset from body B's center.
    pub rb: Vec3,
    /// Approach-speed target for the lane.
    pub target: f32,
    /// Coulomb friction coefficient.
    pub mu: f32,
    /// Restitution bias.
    pub bias: f32,
    /// Incoming accumulated normal impulse (warm start).
    pub acc_in: f32,
    /// Body A snapshot.
    pub a: &'a RigidBody,
    /// Body A index.
    pub ba_idx: u32,
    /// Body B snapshot.
    pub b: &'a RigidBody,
    /// Body B index.
    pub bb_idx: u32,
}

impl GpuBatch {
    /// Blank batch: zeroed lanes, zero active count.
    pub fn zero() -> Self {
        Self::zeroed()
    }

    /// Fill one lane from CPU-side contact data.
    /// The per-lane scalars are packed into `LaneInput` (which mirrors the
    /// WGSL batch layout: one scalar per field) to stay within the structural
    /// gate's argument-count limit.
    pub fn fill_lane(&mut self, input: LaneInput<'_>) {
        let LaneInput {
            lane,
            n,
            ra,
            rb,
            target,
            mu,
            bias,
            acc_in,
            a,
            ba_idx,
            b,
            bb_idx,
        } = input;
        let l3 = |x: &mut [f32; 4], y: &mut [f32; 4], z: &mut [f32; 4], v: Vec3| {
            x[lane] = v.x;
            y[lane] = v.y;
            z[lane] = v.z;
        };

        l3(&mut self.nx, &mut self.ny, &mut self.nz, n);
        l3(&mut self.rax, &mut self.ray, &mut self.raz, ra);
        l3(&mut self.rbx, &mut self.rby, &mut self.rbz, rb);
        self.spec_target[lane] = target;
        self.mu[lane] = mu;
        self.bias[lane] = bias;
        self.acc[lane] = acc_in;
        self.body_a[lane] = ba_idx;
        self.body_b[lane] = bb_idx;
        self.inv_ma[lane] = a.inv_mass;
        self.inv_mb[lane] = b.inv_mass;
        self.total_inv[lane] = a.inv_mass + b.inv_mass;

        // Precompute cross products.
        let ra_n = ra.cross(n);
        let rb_n = rb.cross(n);
        l3(&mut self.ra_nx, &mut self.ra_ny, &mut self.ra_nz, ra_n);
        l3(&mut self.rb_nx, &mut self.rb_ny, &mut self.rb_nz, rb_n);

        // Tangent basis.
        let t1 = crate::math::tangent_basis(n).0;
        let t2 = t1.cross(n);
        l3(&mut self.t1x, &mut self.t1y, &mut self.t1z, t1);
        l3(&mut self.t2x, &mut self.t2y, &mut self.t2z, t2);
        l3(
            &mut self.ra_t1x,
            &mut self.ra_t1y,
            &mut self.ra_t1z,
            ra.cross(t1),
        );
        l3(
            &mut self.rb_t1x,
            &mut self.rb_t1y,
            &mut self.rb_t1z,
            rb.cross(t1),
        );
        l3(
            &mut self.ra_t2x,
            &mut self.ra_t2y,
            &mut self.ra_t2z,
            ra.cross(t2),
        );
        l3(
            &mut self.rb_t2x,
            &mut self.rb_t2y,
            &mut self.rb_t2z,
            rb.cross(t2),
        );

        // World-space inverse inertia matrix rows + application factors.
        let wa = world_inertia_matrix(a.orientation, a.inertia);
        let wb = world_inertia_matrix(b.orientation, b.inertia);
        let set_mat = |m: &[Vec3; 3], dest: &mut [&mut [f32; 4]; 9]| {
            for r in 0..3 {
                dest[r * 3][lane] = m[r].x;
                dest[r * 3 + 1][lane] = m[r].y;
                dest[r * 3 + 2][lane] = m[r].z;
            }
        };
        set_mat(
            &wa,
            &mut [
                &mut self.w_a00,
                &mut self.w_a01,
                &mut self.w_a02,
                &mut self.w_a10,
                &mut self.w_a11,
                &mut self.w_a12,
                &mut self.w_a20,
                &mut self.w_a21,
                &mut self.w_a22,
            ],
        );
        set_mat(
            &wb,
            &mut [
                &mut self.w_b00,
                &mut self.w_b01,
                &mut self.w_b02,
                &mut self.w_b10,
                &mut self.w_b11,
                &mut self.w_b12,
                &mut self.w_b20,
                &mut self.w_b21,
                &mut self.w_b22,
            ],
        );

        // Application factors.
        l3(
            &mut self.apply_n_ax,
            &mut self.apply_n_ay,
            &mut self.apply_n_az,
            n * a.inv_mass,
        );
        l3(
            &mut self.apply_n_bx,
            &mut self.apply_n_by,
            &mut self.apply_n_bz,
            n * b.inv_mass,
        );
        l3(
            &mut self.apply_w_ax,
            &mut self.apply_w_ay,
            &mut self.apply_w_az,
            matvec(&wa, ra_n),
        );
        l3(
            &mut self.apply_w_bx,
            &mut self.apply_w_by,
            &mut self.apply_w_bz,
            matvec(&wb, rb_n),
        );

        // Effective stiffness inverses.
        let total = a.inv_mass + b.inv_mass;
        let k_n = total + ra_n.dot(matvec(&wa, ra_n)) + rb_n.dot(matvec(&wb, rb_n));
        let k_t1 = total
            + ra.cross(t1).dot(matvec(&wa, ra.cross(t1)))
            + rb.cross(t1).dot(matvec(&wb, rb.cross(t1)));
        let k_t2 = total
            + ra.cross(t2).dot(matvec(&wa, ra.cross(t2)))
            + rb.cross(t2).dot(matvec(&wb, rb.cross(t2)));
        self.inv_k_n[lane] = if k_n >= 1e-10 { 1.0 / k_n } else { 0.0 };
        self.inv_k_t1[lane] = if k_t1 >= 1e-10 { 1.0 / k_t1 } else { 0.0 };
        self.inv_k_t2[lane] = if k_t2 >= 1e-10 { 1.0 / k_t2 } else { 0.0 };
    }
}

/// World inverse inertia as `[row0, row1, row2]` (3 Vec3 columns → row-major).
fn world_inertia_matrix(rot: glam::Quat, inertia: Vec3) -> [Vec3; 3] {
    let inv = Vec3::new(
        if inertia.x > 0.0 {
            1.0 / inertia.x
        } else {
            0.0
        },
        if inertia.y > 0.0 {
            1.0 / inertia.y
        } else {
            0.0
        },
        if inertia.z > 0.0 {
            1.0 / inertia.z
        } else {
            0.0
        },
    );
    let m = glam::Mat3::from_quat(rot);
    let col_x = m.x_axis * inv.x;
    let col_y = m.y_axis * inv.y;
    let col_z = m.z_axis * inv.z;
    [
        m.x_axis * col_x.x + m.y_axis * col_y.x + m.z_axis * col_z.x,
        m.x_axis * col_x.y + m.y_axis * col_y.y + m.z_axis * col_z.y,
        m.x_axis * col_x.z + m.y_axis * col_y.z + m.z_axis * col_z.z,
    ]
}

#[inline]
fn matvec(m: &[Vec3; 3], v: Vec3) -> Vec3 {
    Vec3::new(m[0].dot(v), m[1].dot(v), m[2].dot(v))
}

// ---------------------------------------------------------------------------
// Compute shader — written in Rust, translated to WGSL by #[gpu_pipeline]
// ---------------------------------------------------------------------------

/// One iteration of the wide contact solver for a single batch.
///
/// The body of this function is the WGSL compute entry point (kernel DSL):
/// the bindings below are in scope as storage/uniform variables and `gid`/
/// `lid` are the built-in workgroup parameters. The isotropic contact row
/// (normal update, friction clamp) is NOT duplicated here: it calls the
/// shared `contact_math` kernels stitched via `helpers(...)` — one source
/// of truth with the CPU wide/scalar paths. Only buffer plumbing (gather,
/// accumulator read/write, body write-back) stays inline: bindings and
/// builtins are entry-scoped and cannot live in helpers. All per-lane
/// writes to `batch_buf` accumulators go through the full buffer path so
/// that they persist across dispatches (the `let b = ...` copy is
/// read-only).
//
// qual:allow(abc) — kernel-DSL body: every statement is translated
// verbatim into the WGSL compute shader by #[gpu_pipeline], which embeds
// ONLY this function's body into `fn main`. The isotropic row calls the
// shared helpers stitched via `helpers(...)` (see `contact_math`); the
// remaining size is entry-scoped buffer plumbing that cannot move.
#[gpu_pipeline(
    workgroup_size = 4,
    storage(body_buf: [GpuBodyState; 64], read_write),
    storage(batch_buf: [GpuBatch; 64], read_write),
    uniform(params: [u32; 4]),
    builtin(gid: workgroup_id, lid: local_invocation_id),
    helpers(contact_normal_step, contact_friction_clamp),
)]
fn contact_solver() {
    // qual:allow(abc) — kernel-DSL body (see the item docs above): buffer
    // plumbing stays inline, the contact row calls the stitched helpers.
    let b = batch_buf[gid.x];
    let l = lid.x;
    if l >= b.count {
        return;
    }
    let iter = params.x;
    let total = params.y;
    let allow_rest = params.z;

    // Gather body velocities.
    let mut ba = body_buf[b.body_a[l]];
    let mut bb = body_buf[b.body_b[l]];
    let va = ba.velocity;
    let wa = ba.angular;
    let vb = bb.velocity;
    let wb = bb.angular;

    let n = vec3(b.nx[l], b.ny[l], b.nz[l]);
    let ra = vec3(b.rax[l], b.ray[l], b.raz[l]);
    let rb = vec3(b.rbx[l], b.rby[l], b.rbz[l]);
    let inv_ma = b.inv_ma[l];
    let inv_mb = b.inv_mb[l];
    let inv_k = b.inv_k_n[l];
    let spec_target = b.spec_target[l];
    let mut acc = batch_buf[gid.x].acc[l];

    // Normal impulse: shared isotropic row (contact_math).
    let rel = (vb + cross(wb, rb)) - (va + cross(wa, ra));
    let vn = dot(rel, n);
    let new_acc = contact_normal_step(vn, spec_target, inv_k, acc);
    let delta = new_acc - acc;
    acc = new_acc;

    if abs(delta) > 1e-12 {
        ba.velocity -= delta * vec3(b.apply_n_ax[l], b.apply_n_ay[l], b.apply_n_az[l]);
        bb.velocity += delta * vec3(b.apply_n_bx[l], b.apply_n_by[l], b.apply_n_bz[l]);
        ba.angular -= delta * vec3(b.apply_w_ax[l], b.apply_w_ay[l], b.apply_w_az[l]);
        bb.angular += delta * vec3(b.apply_w_bx[l], b.apply_w_by[l], b.apply_w_bz[l]);
    }

    // Friction (remeasure rel after the normal impulse): shared circular
    // clamp per axis (contact_math).
    let rel2 = (bb.velocity + cross(bb.angular, rb)) - (ba.velocity + cross(ba.angular, ra));
    let max_f = b.mu[l] * acc;
    let t1 = vec3(b.t1x[l], b.t1y[l], b.t1z[l]);
    let t2 = vec3(b.t2x[l], b.t2y[l], b.t2z[l]);
    let mut f_imp = vec3(0.0);

    // Axis 1.
    if b.inv_k_t1[l] > 0.0 {
        let vt1 = dot(rel2, t1);
        let raw_t1 = batch_buf[gid.x].acc_f1[l] - vt1 * b.inv_k_t1[l];
        let new_t1 = contact_friction_clamp(raw_t1, batch_buf[gid.x].acc_f2[l], max_f);
        f_imp += t1 * (new_t1 - batch_buf[gid.x].acc_f1[l]);
        batch_buf[gid.x].acc_f1[l] = new_t1;
    }
    // Axis 2.
    if b.inv_k_t2[l] > 0.0 {
        let vt2 = dot(rel2, t2);
        let raw_t2 = batch_buf[gid.x].acc_f2[l] - vt2 * b.inv_k_t2[l];
        let new_t2 = contact_friction_clamp(raw_t2, batch_buf[gid.x].acc_f1[l], max_f);
        f_imp += t2 * (new_t2 - batch_buf[gid.x].acc_f2[l]);
        batch_buf[gid.x].acc_f2[l] = new_t2;
    }
    if dot(f_imp, f_imp) > 1e-24 {
        // Angular impulse uses r × f (same convention as the scalar solver:
        // `wa -= W·(ra×f)`, `wb += W·(rb×f)`).
        let ca = cross(ra, f_imp);
        let w_a = vec3(
            dot(vec3(b.w_a00[l], b.w_a01[l], b.w_a02[l]), ca),
            dot(vec3(b.w_a10[l], b.w_a11[l], b.w_a12[l]), ca),
            dot(vec3(b.w_a20[l], b.w_a21[l], b.w_a22[l]), ca),
        );
        let cb = cross(rb, f_imp);
        let w_b = vec3(
            dot(vec3(b.w_b00[l], b.w_b01[l], b.w_b02[l]), cb),
            dot(vec3(b.w_b10[l], b.w_b11[l], b.w_b12[l]), cb),
            dot(vec3(b.w_b20[l], b.w_b21[l], b.w_b22[l]), cb),
        );
        ba.velocity -= f_imp * inv_ma;
        bb.velocity += f_imp * inv_mb;
        ba.angular -= w_a;
        bb.angular += w_b;
    }

    // Restitution (one-shot on the last iteration).
    if allow_rest > 0 && iter == total - 1 {
        let bias = b.bias[l];
        if bias > 0.0 {
            let rel3 =
                (bb.velocity + cross(bb.angular, rb)) - (ba.velocity + cross(ba.angular, ra));
            let vn3 = dot(rel3, n);
            let lr = (bias - vn3) * inv_k;
            if lr > 0.0 {
                ba.velocity -= lr * vec3(b.apply_n_ax[l], b.apply_n_ay[l], b.apply_n_az[l]);
                bb.velocity += lr * vec3(b.apply_n_bx[l], b.apply_n_by[l], b.apply_n_bz[l]);
                ba.angular -= lr * vec3(b.apply_w_ax[l], b.apply_w_ay[l], b.apply_w_az[l]);
                bb.angular += lr * vec3(b.apply_w_bx[l], b.apply_w_by[l], b.apply_w_bz[l]);
            }
        }
    }

    // Write back.
    body_buf[b.body_a[l]].velocity = ba.velocity;
    body_buf[b.body_a[l]].angular = ba.angular;
    body_buf[b.body_b[l]].velocity = bb.velocity;
    body_buf[b.body_b[l]].angular = bb.angular;
    batch_buf[gid.x].acc[l] = acc;
}

/// The complete WGSL source: struct declarations (generated from the Rust
/// layouts) + the compute shader translated from `contact_solver`.
pub fn contact_solver_wgsl() -> String {
    format!(
        "{}\n{}\n{}",
        GpuBodyState::WGSL_SOURCE,
        GpuBatch::WGSL_SOURCE,
        contact_solver::wgsl_source()
    )
}

// ---------------------------------------------------------------------------
// GPU sequential-impulse solver
// ---------------------------------------------------------------------------

/// GPU sequential-impulse solver for single-point manifold batches: the
/// same SI velocity iterations as the CPU wide path, run per wide batch.
pub struct GpuSequentialImpulse {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    body_buf: wgpu::Buffer,     // read-write body state
    batch_buf: wgpu::Buffer,    // read-write batch data (acc accumulators)
    uniform_buf: wgpu::Buffer,  // params table: one (iter,total,rest,0) vec4 per pass
    readback_buf: wgpu::Buffer, // staging copy for body download
    max_bodies: usize,
    max_batches: usize,
    /// Byte stride between params-table entries (device uniform-offset
    /// alignment; entries are 16-byte vec4s padded up to it).
    param_stride: u64,
}

impl GpuSequentialImpulse {
    /// Create a new GPU solver attached to the given wgpu context.
    /// `max_bodies` and `max_batches` must be large enough for the scene.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        max_bodies: usize,
        max_batches: usize,
    ) -> Self {
        let source = contact_solver_wgsl();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics_contact"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics_contact_bgl"),
            entries: &[
                // body_buf: read-write storage
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // batch_buf: read-write storage (acc accumulators)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // uniform: params table (dynamic offset selects the pass entry)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physics_contact_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics_contact_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let body_size = max_bodies.next_power_of_two().max(64) as u64 * GPU_BODY_STRIDE;
        let batch_size = max_batches.next_power_of_two().max(64) as u64 * GPU_BATCH_STRIDE;

        let body_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics_body_state"),
            size: body_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let batch_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics_contact_batches"),
            size: batch_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1) as u64;
        let param_stride = 16u64.next_multiple_of(align);
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics_contact_params"),
            size: PARAMS_CAP * param_stride,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics_contact_readback"),
            size: body_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics_contact_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: body_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: batch_buf.as_entire_binding(),
                },
                // Bind a single params-table entry, not the whole buffer: the
                // binding size is what the dynamic offset may slide within, so
                // an entire-buffer binding left zero headroom and any pass
                // offset > 0 overran the buffer (wgpu validation error).
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buf,
                        offset: 0,
                        size: wgpu::BufferSize::new(param_stride),
                    }),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group,
            body_buf,
            batch_buf,
            uniform_buf,
            readback_buf,
            max_bodies,
            max_batches,
            param_stride,
        }
    }

    /// Upload body velocities to the GPU buffer.
    pub fn upload_bodies(&self, bodies: &[RigidBody]) {
        let n = bodies.len().min(self.max_bodies);
        let mut data = vec![GpuBodyState::zeroed(); self.max_bodies];
        for (i, b) in bodies.iter().enumerate().take(n) {
            data[i] = GpuBodyState::from_body(b);
        }
        self.queue
            .write_buffer(&self.body_buf, 0, bytemuck::cast_slice(&data));
    }

    /// Download body velocities from the GPU buffer (blocking).
    pub fn download_bodies(&self, bodies: &mut [RigidBody]) {
        let n = bodies.len().min(self.max_bodies);
        let copy_size = self.max_bodies as u64 * GPU_BODY_STRIDE;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics_download"),
            });
        encoder.copy_buffer_to_buffer(&self.body_buf, 0, &self.readback_buf, 0, copy_size);
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        let mapped = slice.get_mapped_range().unwrap();
        let states: &[GpuBodyState] = bytemuck::cast_slice(&mapped);
        for (i, b) in bodies.iter_mut().enumerate().take(n) {
            states[i].write_to_body(b);
        }
        drop(mapped);
        self.readback_buf.unmap();
    }

    /// Upload contact batches to the GPU buffer.
    pub fn upload_batches(&self, batches: &[GpuBatch]) {
        let n = batches.len().min(self.max_batches);
        let mut data = vec![GpuBatch::zeroed(); self.max_batches];
        for (i, b) in batches.iter().enumerate().take(n) {
            data[i] = *b;
        }
        self.queue
            .write_buffer(&self.batch_buf, 0, bytemuck::cast_slice(&data));
    }

    /// Download accumulated impulses back from the GPU batch buffer.
    pub fn download_acc(&self, batches: &mut [GpuBatch]) {
        let n = batches.len().min(self.max_batches);
        let copy_size = self.max_batches as u64 * GPU_BATCH_STRIDE;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics_acc_readback"),
            size: copy_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics_acc_dl"),
            });
        encoder.copy_buffer_to_buffer(&self.batch_buf, 0, &readback, 0, copy_size);
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let raw: &[u8] = &mapped;
        let gpu_entries: &[GpuBatch] = bytemuck::cast_slice(raw);
        for (i, b) in batches.iter_mut().enumerate().take(n) {
            b.acc = gpu_entries[i].acc;
            b.acc_f1 = gpu_entries[i].acc_f1;
            b.acc_f2 = gpu_entries[i].acc_f2;
        }
        drop(mapped);
        readback.unmap();
    }

    /// Run the GPU contact solver for `iterations` GS iterations plus one
    /// restitution pass if `allow_restitution` (folded into the last
    /// iteration by the shader, as before).
    ///
    /// Bulk dispatch (v2): all iterations go into ONE command encoder as
    /// back-to-back compute passes — passes on one queue execute in order
    /// with the same dispatch-boundary memory visibility as the old
    /// one-submit-per-iteration loop, so the Jacobi/across-batch
    /// consistency model is unchanged. Per-pass params come from the
    /// params table via dynamic uniform offsets (pass `k` reads entry
    /// `k`: identical `(iter, total, rest)` values to the old loop).
    /// One upload, one submit, one blocking wait per call instead of one
    /// CPU round-trip per iteration.
    pub fn solve(&self, num_batches: u32, iterations: u32, allow_restitution: bool) {
        assert!(
            u64::from(iterations) <= PARAMS_CAP,
            "solve iterations {iterations} exceed params-table cap {PARAMS_CAP}"
        );
        if iterations == 0 || num_batches == 0 {
            return;
        }
        // One upload for all passes (entries padded to the device stride).
        let mut blob = vec![0u8; iterations as usize * self.param_stride as usize];
        for (k, entry) in solve_params(iterations, allow_restitution)
            .iter()
            .enumerate()
        {
            let bytes: &[u8] = bytemuck::cast_slice(entry.as_slice());
            let base = k * self.param_stride as usize;
            blob[base..base + bytes.len()].copy_from_slice(bytes);
        }
        self.queue.write_buffer(&self.uniform_buf, 0, &blob);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics_contact_bulk"),
            });
        for k in 0..iterations {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("physics_contact_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(
                0,
                &self.bind_group,
                &[u32::try_from(k as u64 * self.param_stride).unwrap()],
            );
            cpass.dispatch_workgroups(num_batches, 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        // Single barrier for the whole bulk (was: one per iteration).
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
    }
}

// ---------------------------------------------------------------------------
// Public API: pack GPU batches from engine data
// ---------------------------------------------------------------------------

/// Pack single-point manifold states into GPU batches.
/// Returns the batches and their count.
pub fn pack_single_point_batches(
    bodies: &[RigidBody],
    states: &[ManifoldState],
    manifolds: &[Manifold],
    single_indices: &[usize], // indices into states
) -> (Vec<GpuBatch>, u32) {
    // Group into batches of ≤4 disjoint contacts (same strategy as
    // build_solver_steps, but using global body indices).
    let mut batches: Vec<GpuBatch> = Vec::new();
    let mut cur = GpuBatch::zero();
    let mut cur_len = 0usize;
    let mut cur_bodies: [usize; 8] = [usize::MAX; 8];
    let mut cur_n = 0usize;

    fn flush_batch(
        batches: &mut Vec<GpuBatch>,
        cur: &mut GpuBatch,
        cur_len: &mut usize,
        cur_bodies: &mut [usize; 8],
        cur_n: &mut usize,
    ) {
        if *cur_len == 0 {
            return;
        }
        cur.count = *cur_len as u32;
        batches.push(*cur);
        *cur = GpuBatch::zero();
        *cur_len = 0;
        *cur_n = 0;
        *cur_bodies = [usize::MAX; 8];
    }

    for &si in single_indices {
        let st = &states[si];
        let m = &manifolds[st.mi];
        let (i, j) = (st.i, st.j);
        let conflicts = cur_bodies[..cur_n].contains(&i) || cur_bodies[..cur_n].contains(&j);
        if cur_len >= 4 || conflicts {
            flush_batch(
                &mut batches,
                &mut cur,
                &mut cur_len,
                &mut cur_bodies,
                &mut cur_n,
            );
        }
        let p = m.points[0].world_point;
        let ra = p - bodies[i].position;
        let rb = p - bodies[j].position;
        cur.fill_lane(LaneInput {
            lane: cur_len,
            n: m.normal,
            ra,
            rb,
            target: st.target[0],
            mu: st.mu,
            bias: st.bias[0],
            acc_in: st.acc[0],
            a: &bodies[i],
            ba_idx: i as u32,
            b: &bodies[j],
            bb_idx: j as u32,
        });
        cur_bodies[cur_n] = i;
        cur_bodies[cur_n + 1] = j;
        cur_n += 2;
        cur_len += 1;
        cur.count = cur_len as u32;
    }
    flush_batch(
        &mut batches,
        &mut cur,
        &mut cur_len,
        &mut cur_bodies,
        &mut cur_n,
    );

    let count = batches.len() as u32;
    (batches, count)
}

/// Write GPU batch accumulated impulses back to the ManifoldState array
/// (for warm-cache persistence).
pub fn write_back_acc(
    states: &mut [ManifoldState],
    single_indices: &[usize],
    batches: &[GpuBatch],
) {
    let mut bi = 0usize;
    let mut lane = 0usize;
    for &si in single_indices {
        if lane >= batches[bi].count as usize {
            bi += 1;
            lane = 0;
        }
        states[si].acc[0] = batches[bi].acc[lane];
        lane += 1;
    }
}

// ---------------------------------------------------------------------------
// STUB spike-interface: GPU AVBD solver (no device path yet)
// ---------------------------------------------------------------------------

/// STUB dispatch seam for a future GPU AVBD solver.
///
/// Rung 1 pins the per-body inertial contract in the shader: the kernel
/// below assembles the lumped Hessian diagonal from the mass roster and
/// solves the diagonal LDL system (helpers stitched via `helpers(...)`).
/// Contact rows, the dense 6x6 LDL and device execution stay future work.
/// Stepping always falls back to the CPU [`AvbdEngine`]; the kernel moves
/// no engine state yet. Honest by construction: `runs_on_device` returns
/// `false` until a real dispatch lands.
///
/// Unwired spike surface (no engine holds one yet), hence `allow(dead_code)`.
#[allow(dead_code)]
pub trait GpuAvbdDispatch {
    /// Advance `engine` by `dt`. CPU fallback until the device path lands.
    fn step_avbd(&self, engine: &mut AvbdEngine, dt: f32);
    /// Whether stepping touches the device. `false` while the stub falls back.
    fn runs_on_device(&self) -> bool;
}

/// STUB GPU AVBD solver: config + CPU fallback, no device path.
///
/// Holds only the buffer-size bound a future dispatch will need. Pair with
/// [`avbd_stub_wgsl`] for the rung-1 kernel source (inertial Hessian
/// diagonal + diagonal LDL, no contacts yet).
///
/// Unwired spike surface (no engine holds one yet), hence `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct GpuAvbdStub {
    /// Maximum bodies one future dispatch covers (buffer-size bound only,
    /// not a physics bound — mirrors the `GpuSequentialImpulse` caps).
    pub max_bodies: usize,
}

impl GpuAvbdStub {
    /// Describe a stub dispatch for up to `max_bodies` bodies.
    #[allow(dead_code)]
    pub fn new(max_bodies: usize) -> Self {
        Self { max_bodies }
    }

    /// Whether stepping touches the device. Always `false` on the stub.
    pub fn runs_on_device(&self) -> bool {
        false
    }

    /// Honest CPU fallback: steps the CPU AVBD engine in place.
    pub fn step_cpu(&self, engine: &mut AvbdEngine, dt: f32) {
        engine.step(dt);
    }

    /// Build the device mass roster for `bodies`: one [`GpuAvbdMass`] per
    /// body (dynamics carry their mass model, everything else zeroes).
    /// Pure (no device); the future dispatch will upload this verbatim.
    /// Entries mirror [`avbd_inertial_hessian_diag`]'s inputs exactly.
    #[allow(dead_code)]
    pub fn mass_roster(&self, bodies: &[RigidBody]) -> Vec<GpuAvbdMass> {
        bodies.iter().map(GpuAvbdMass::from_body).collect()
    }

    /// CPU reference for the rung-1 device dispatch: assemble the inertial
    /// Hessian diagonal per body and solve the diagonal LDL system —
    /// exactly what `avbd_stub_kernel` computes per invocation. The future
    /// device path validates against this (tolerance, never bit-identical).
    /// Returns `None` only on a body/rhs length mismatch.
    #[allow(dead_code)]
    pub fn stage_diag_solve(
        &self,
        bodies: &[RigidBody],
        rhs: &[[f32; 6]],
        dt: f32,
    ) -> Option<Vec<[f32; 6]>> {
        if bodies.len() != rhs.len() {
            return None;
        }
        bodies
            .iter()
            .zip(rhs.iter())
            .map(|(b, r)| {
                let m = GpuAvbdMass::from_body(b);
                let h = avbd_inertial_hessian_diag(m.inv_mass, m.inertia, dt);
                Some(avbd_diag_solve_cpu(h, *r))
            })
            .collect()
    }
}

impl GpuAvbdDispatch for GpuAvbdStub {
    fn step_avbd(&self, engine: &mut AvbdEngine, dt: f32) {
        self.step_cpu(engine, dt);
    }

    fn runs_on_device(&self) -> bool {
        self.runs_on_device()
    }
}

/// Rung-1 AVBD helper: linear Hessian entry `m/dt²` from inverse mass,
/// or `0.0` for statics / degenerate `dt`. Mirrors the linear half of
/// [`avbd_inertial_hessian_diag`]; the shader-side spelling is pinned by
/// naga validation, the values by the CPU test.
#[wgsl_fn]
fn avbd_hessian_lin(inv_mass: f32, dt: f32) -> f32 {
    let mut h = 0.0;
    if inv_mass > 0.0 {
        let dt2 = dt * dt;
        if dt2 > 1e-12 {
            h = 1.0 / (inv_mass * dt2);
        }
    }
    return h;
}

/// Rung-1 AVBD helper: angular Hessian diagonal `I/dt²` (body-frame lumped
/// approximation — the world off-diagonal coupling stays CPU-side, exact
/// for isotropic bodies). Zeroes on degenerate `dt`. Mirrors the angular
/// half of [`avbd_inertial_hessian_diag`].
#[wgsl_fn]
fn avbd_hessian_ang(inertia: Vec3, dt: f32) -> Vec3 {
    let dt2 = dt * dt;
    let mut h0 = 0.0;
    let mut h1 = 0.0;
    let mut h2 = 0.0;
    if dt2 > 1e-12 {
        h0 = inertia[0] / dt2;
        h1 = inertia[1] / dt2;
        h2 = inertia[2] / dt2;
    }
    return Vec3::new(h0, h1, h2);
}

/// Rung-1 AVBD helper: diagonal LDL solve (`L = I`), one 3-block.
/// Degenerate axes solve to `0.0` (no `Option` in WGSL); the CPU
/// [`avbd_diag_solve_cpu`] mirrors this exactly — the two agree on
/// strictly positive systems (pinned by test).
#[wgsl_fn]
fn avbd_diag_solve(h: Vec3, r: Vec3) -> Vec3 {
    let mut x0 = 0.0;
    let mut x1 = 0.0;
    let mut x2 = 0.0;
    if h[0] > 1e-12 {
        x0 = r[0] / h[0];
    }
    if h[1] > 1e-12 {
        x1 = r[1] / h[1];
    }
    if h[2] > 1e-12 {
        x2 = r[2] / h[2];
    }
    return Vec3::new(x0, x1, x2);
}

/// Rung-1 AVBD device kernel: per-body inertial Hessian diagonal + diagonal
/// LDL solve (one invocation per body).
///
/// Each body reads its mass entry, assembles the 6-entry lumped Hessian
/// diagonal (`m/dt²` linear, body-frame `I/dt²` angular) and solves the
/// diagonal system against its rhs entry into the delta buffer. Contact
/// rows and the dense 6x6 LDL stay future work; the kernel moves no engine
/// state (no dispatch exists yet) — it pins the buffer/bindings/dispatch
/// shape plus the exact per-body math the CPU tests verify in tolerances.
//
// qual:allow(abc) — kernel-DSL body: every statement is translated
// verbatim into the WGSL compute shader by #[gpu_pipeline], which embeds
// ONLY this function's body into `fn main`. Extracting helpers would emit
// calls to functions that do not exist in the shader; splitting requires a
// macro-level helper-inclusion feature, not a local edit.
#[gpu_pipeline(
    workgroup_size = 64,
    storage(avbd_mass: [GpuAvbdMass; 64], read),
    storage(avbd_rhs: [GpuBodyState; 64], read),
    storage(avbd_delta: [GpuBodyState; 64], read_write),
    uniform(avbd_count: [u32; 4]),
    uniform(avbd_dt: [f32; 4]),
    builtin(gid: workgroup_id),
    helpers(avbd_hessian_lin, avbd_hessian_ang, avbd_diag_solve),
)]
fn avbd_stub_kernel() {
    // qual:allow(abc) — kernel-DSL body: every statement is translated
    // verbatim into the WGSL compute shader by #[gpu_pipeline], which embeds
    // ONLY this function's body into `fn main`. Extracting helpers would emit
    // calls to functions that do not exist in the shader; splitting requires a
    // macro-level helper-inclusion feature, not a local edit.
    if gid.x >= avbd_count.x {
        return;
    }
    let m = avbd_mass[gid.x];
    let r = avbd_rhs[gid.x];
    let dt = avbd_dt.x;
    let hl = avbd_hessian_lin(m.inv_mass, dt);
    let h_lin = vec3(hl, hl, hl);
    let h_ang = avbd_hessian_ang(m.inertia, dt);
    let x_lin = avbd_diag_solve(h_lin, r.velocity);
    let x_ang = avbd_diag_solve(h_ang, r.angular);
    avbd_delta[gid.x].velocity = x_lin;
    avbd_delta[gid.x].angular = x_ang;
}

/// Rung-1 kernel WGSL source: mass/body layouts + inertial-Hessian/diagonal-
/// LDL entry (no contacts, no engine state). Pure (no device) so tests pin
/// it without an adapter.
///
/// Unwired spike surface (no engine holds one yet), hence `allow(dead_code)`.
#[allow(dead_code)]
pub fn avbd_stub_wgsl() -> String {
    format!(
        "{}\n{}\n{}",
        GpuAvbdMass::WGSL_SOURCE,
        GpuBodyState::WGSL_SOURCE,
        avbd_stub_kernel::wgsl_source()
    )
}
