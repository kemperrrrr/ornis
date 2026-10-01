//! GPU AVBD solver: rung-1 inertial diagonal kernel plus the rung-2
//! contact-row + dense 6x6 LDL kernel with device execution.
//!
//! Owns [`GpuAvbdMass`], the [`avbd_inertial_hessian_diag`]/[`avbd_diag_solve_cpu`]
//! CPU references, the [`GpuAvbdStub`]/[`GpuAvbdDispatch`] seam with its
//! `avbd_stub_kernel` entry (inertial Hessian diagonal + diagonal LDL, helpers
//! stitched via `helpers(...)`), and the rung-2 [`WgpuAvbdSolver`] with its
//! `avbd_row_kernel` entry (one contact row stamped per body + dense 6x6 LDL
//! with a breakdown flag, executed on the device). Stepping an [`AvbdEngine`]
//! still falls back to the CPU [`crate::avbd::AvbdEngine`] until rung 3
//! (host discovery staging rows into the device solve). Moved verbatim from
//! `gpu.rs` (phase 3); rung 2 added in place.

use ornis_macros::{WgslStruct, gpu_pipeline, wgsl_fn};

use super::{GpuBodyState, GpuDispatchError};
use crate::avbd::{ANGULAR_OFFSET, AvbdEngine, SPATIAL_DOF, solve_6x6};
use crate::body::{BodyType, RigidBody};
use crate::engine::PhysicsEngine;
use bytemuck::Zeroable as _;

/// Numerical zero for host-side guards: time steps, diagonal entries and
/// Hessian components at or below this magnitude are treated as exactly
/// zero (degenerate axis, unsteppable dt). Shader bodies keep literals —
/// the DSL embeds `#[wgsl_fn]`/`#[gpu_pipeline]` bodies verbatim and has
/// no spelling for a Rust `const` (likewise `[T; N]` lowers literal N
/// only, see `avbd_row_kernel`): a named const inside a shader body
/// emits an unknown WGSL identifier (regression: `DEGENERATE_EPS`
/// broke the rung-1/rung-2 naga gates, fixed by restoring literals).
const DEGENERATE_EPS: f32 = 1e-12;
/// Minimum power-of-two body capacity for AVBD GPU buffers.
const MIN_BUFFER_CAP: usize = 64;
/// Bytes per `f32` / WGSL `f32` storage element.
const F32_BYTES: u64 = 4;
/// Bytes reserved for a single uniform block (`vec4`-sized params).
const UNIFORM_BLOCK_BYTES: u64 = 16;
/// Bind-group slot for the body-count uniform.
const BINDING_COUNT: u32 = 3;
/// Bind-group slot for the dt uniform.
const BINDING_DT: u32 = 4;

// ---------------------------------------------------------------------------
// AVBD inertial mass roster (rung 1 device input)
// ---------------------------------------------------------------------------

/// Per-body mass roster for the AVBD device rung: body-frame inertia
/// diagonal + inverse mass. Field order is layout-driven: `inertia`
/// (`vec3<f32>`, 16-aligned) first so no padding is needed — the struct is
/// exactly 16 bytes (see the layout test below). `align(16)`: the roster
/// also nests by value inside [`GpuAvbdSystem`], and nested structs must be
/// 16-byte aligned for the WGSL layout walk.
#[repr(C, align(16))]
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
                inertia: [0.0; ANGULAR_OFFSET],
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
pub fn avbd_inertial_hessian_diag(inv_mass: f32, inertia: [f32; 3], dt: f32) -> [f32; SPATIAL_DOF] {
    // Non-positive or non-finite inputs assemble nothing (matches the
    // shader guards, which test the positive form and zero otherwise —
    // NaN fails both spellings and lands on zeros either way).
    if !inv_mass.is_finite() || inv_mass <= 0.0 || !dt.is_finite() || dt <= 0.0 {
        return [0.0; SPATIAL_DOF];
    }
    let dt2 = dt * dt;
    if !dt2.is_finite() || dt2 <= DEGENERATE_EPS {
        return [0.0; SPATIAL_DOF];
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
/// degenerate axes (`<= DEGENERATE_EPS`) solve to `0.0` — statics and sleepers
/// (zeroed mass model) carry no correction.
///
/// This mirrors the `avbd_diag_solve` shader helper exactly (WGSL has no
/// `Option`); on strictly positive systems it agrees with the dense AVBD
/// LDL within float tolerance (pinned by test, never bit-identical by
/// promise). Full 6x6 device LDL with breakdown signaling is a later rung.
pub fn avbd_diag_solve_cpu(
    diag: [f32; SPATIAL_DOF],
    rhs: [f32; SPATIAL_DOF],
) -> [f32; SPATIAL_DOF] {
    let mut out = [0.0f32; SPATIAL_DOF];
    for i in 0..SPATIAL_DOF {
        if diag[i] > DEGENERATE_EPS {
            out[i] = rhs[i] / diag[i];
        }
    }
    out
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
        rhs: &[[f32; SPATIAL_DOF]],
        dt: f32,
    ) -> Option<Vec<[f32; SPATIAL_DOF]>> {
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
        // Literal: shader bodies cannot name `DEGENERATE_EPS` (see above).
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
    // Literal: shader bodies cannot name `DEGENERATE_EPS` (see above).
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
    // Literals: shader bodies cannot name `DEGENERATE_EPS` (see above).
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
    // macro-level helper-inclusion feature, not a local edit. Array dimensions
    // stay integer literals for the same reason (`[T; N]` lowers literal N
    // only — a Rust `const` has no WGSL spelling at macro time); host-side
    // mirrors use `SPATIAL_DOF`.
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

// ---------------------------------------------------------------------------
// Rung 2: one contact row + dense 6x6 LDL on the device
// ---------------------------------------------------------------------------

/// One contact-row input for the rung-2 device kernel: the exact arguments
/// [`AvbdEngine::solve_body`] stamps per body side (constraint direction,
/// world lever arm, penalty, row force, side sign).
///
/// Layout-driven: `axis` (`vec3<f32>`, 16-aligned) + `pen` fill 16 bytes,
/// `lever` + `force` fill 16 bytes, `sign` + explicit padding fill 16 bytes —
/// 48 bytes total (see the layout test). `align(16)`: rows nest by value
/// inside [`GpuAvbdSystem`], and nested structs must be 16-byte aligned
/// for the WGSL layout walk.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
pub struct GpuAvbdRow {
    /// Constraint direction, A-side (unit axis: contact normal/tangent or
    /// joint axis).
    pub axis: [f32; 3],
    /// Penalty stiffness of the row.
    pub pen: f32,
    /// World lever arm `r` of this body side.
    pub lever: [f32; 3],
    /// Row force `f` (penalty × violation + dual).
    pub force: f32,
    /// Side sign (`+1.0` for body A, `-1.0` for body B).
    pub sign: f32,
    /// Explicit tail padding (not shader-visible: a visible `vec3<f32>`
    /// member would grow the WGSL span past the Rust size — see
    /// `#[wgsl(skip)]`).
    #[wgsl(skip)]
    _pad: [f32; 3],
}

impl GpuAvbdRow {
    /// Build a row input from engine-side row arguments. Pure (no device).
    pub fn new(axis: glam::Vec3, pen: f32, lever: glam::Vec3, force: f32, sign: f32) -> Self {
        Self {
            axis: axis.to_array(),
            pen,
            lever: lever.to_array(),
            force,
            sign,
            _pad: [0.0; 3],
        }
    }

    /// Inert row: zero penalty and force stamp nothing, so the kernel solves
    /// the bare inertial system (statics then report breakdown, see
    /// [`avbd_ldl_6x6_cpu`]).
    pub fn inert() -> Self {
        Self {
            axis: [0.0; 3],
            pen: 0.0,
            lever: [0.0; 3],
            force: 0.0,
            sign: 1.0,
            _pad: [0.0; 3],
        }
    }
}

/// One staged per-body system for the rung-2 device kernel: mass roster
/// entry + base 6-residual + contact row.
///
/// A single input array keeps the kernel at 3 storage buffers (systems in,
/// deltas + flags out) — inside the 4-storage-buffer downlevel limit that
/// one buffer per input would exceed. Layout: 16 (mass) + 32 (state) + 48
/// (row) = 96 bytes, all members 16-aligned (see the layout test).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
pub struct GpuAvbdSystem {
    /// Body-frame inertia diagonal + inverse mass.
    pub mass: GpuAvbdMass,
    /// Base 6-residual `(linear, angular)` the row-corrected solve starts from.
    pub state: GpuBodyState,
    /// The one stamped contact row.
    pub row: GpuAvbdRow,
}

impl GpuAvbdSystem {
    /// Pack one staged system from its parts. Pure (no device).
    pub fn new(mass: GpuAvbdMass, residual: [f32; SPATIAL_DOF], row: GpuAvbdRow) -> Self {
        Self {
            mass,
            state: GpuBodyState::from_residual(residual),
            row,
        }
    }
}

/// Outer product of two 3-arrays (row-major): the same op
/// [`AvbdEngine::solve_body`] builds via `outer` before stamping.
fn outer3(a: [f32; 3], b: [f32; 3]) -> [[f32; 3]; 3] {
    [
        [a[0] * b[0], a[0] * b[1], a[0] * b[2]],
        [a[1] * b[0], a[1] * b[1], a[1] * b[2]],
        [a[2] * b[0], a[2] * b[1], a[2] * b[2]],
    ]
}

/// Stamp one AVBD constraint row into a 6x6 system: `nn = sign*axis`,
/// `t = r×nn`, outer-product Hessian plus force right-hand side.
///
/// Exact CPU mirror of the row math in [`AvbdEngine::solve_body`]'s
/// `stamp_row` (same ops in the same order — bit-identical on CPU); the
/// rung-2 kernel stamps the same row in the shader, pinned against this in
/// tolerance (never bit-identical by promise: device contraction may differ
/// ±1 ulp per op). Pure (no device) so unit tests pin it.
pub fn avbd_stamp_row_cpu(
    lhs: &mut [[f32; SPATIAL_DOF]; SPATIAL_DOF],
    rhs: &mut [f32; SPATIAL_DOF],
    axis: [f32; ANGULAR_OFFSET],
    pen: f32,
    force: f32,
    lever: [f32; ANGULAR_OFFSET],
    sign: f32,
) {
    let axis = glam::Vec3::from_array(axis);
    let r = glam::Vec3::from_array(lever);
    let nn = sign * axis;
    let t = r.cross(nn);
    let nn = [nn.x, nn.y, nn.z];
    let t = [t.x, t.y, t.z];
    let o_nn = outer3(nn, nn);
    let o_tt = outer3(t, t);
    let o_nt = outer3(nn, t);
    for x in 0..ANGULAR_OFFSET {
        for y in 0..ANGULAR_OFFSET {
            lhs[x][y] += pen * o_nn[x][y];
            lhs[ANGULAR_OFFSET + x][ANGULAR_OFFSET + y] += pen * o_tt[x][y];
            lhs[x][ANGULAR_OFFSET + y] += pen * o_nt[x][y];
            lhs[ANGULAR_OFFSET + x][y] += pen * o_nt[y][x];
        }
    }
    rhs[0] += force * nn[0];
    rhs[1] += force * nn[1];
    rhs[2] += force * nn[2];
    rhs[ANGULAR_OFFSET] += force * t[0];
    rhs[ANGULAR_OFFSET + 1] += force * t[1];
    rhs[ANGULAR_OFFSET + 2] += force * t[2];
}

/// Dense 6x6 LDL with breakdown signal.
///
/// Mirrors [`solve_6x6`] exactly (`Some(x)` ↔ `(x, true)`); WGSL has no
/// `Option`, so the device kernel reports the same outcome as an `ok` flag
/// into its `avbd_ok` buffer (`1.0` solved, `0.0` breakdown with a zeroed
/// delta — statics and sleepers carry no correction). Pure (no device).
pub fn avbd_ldl_6x6_cpu(
    lhs: [[f32; SPATIAL_DOF]; SPATIAL_DOF],
    rhs: [f32; SPATIAL_DOF],
) -> ([f32; SPATIAL_DOF], bool) {
    match solve_6x6(lhs, rhs) {
        Some(x) => (x, true),
        None => ([0.0; SPATIAL_DOF], false),
    }
}

/// CPU reference staging for the rung-2 device dispatch: per body, assemble
/// the lumped inertial Hessian diagonal ([`avbd_inertial_hessian_diag`],
/// same approximation as rung 1), stamp that body's [`GpuAvbdRow`] and solve
/// the dense system ([`avbd_ldl_6x6_cpu`]) — exactly what `avbd_row_kernel`
/// computes per invocation. The device path validates against this in
/// tolerance, per member plus the breakdown flag, never bit-identical.
/// Returns `None` only on a bodies/rhs/rows length mismatch.
pub fn avbd_stage_contact_solve(
    bodies: &[RigidBody],
    rhs: &[[f32; SPATIAL_DOF]],
    rows: &[GpuAvbdRow],
    dt: f32,
) -> Option<Vec<([f32; SPATIAL_DOF], bool)>> {
    if bodies.len() != rhs.len() || bodies.len() != rows.len() {
        return None;
    }
    bodies
        .iter()
        .zip(rhs.iter())
        .zip(rows.iter())
        .map(|((b, r), row)| {
            let m = GpuAvbdMass::from_body(b);
            let diag = avbd_inertial_hessian_diag(m.inv_mass, m.inertia, dt);
            let mut lhs = [[0.0f32; SPATIAL_DOF]; SPATIAL_DOF];
            for (i, row_lhs) in lhs.iter_mut().enumerate() {
                row_lhs[i] = diag[i];
            }
            let mut full_rhs = *r;
            avbd_stamp_row_cpu(
                &mut lhs,
                &mut full_rhs,
                row.axis,
                row.pen,
                row.force,
                row.lever,
                row.sign,
            );
            Some(avbd_ldl_6x6_cpu(lhs, full_rhs))
        })
        .collect()
}

// qual:allow(abc) — kernel-DSL body: every statement is translated
// verbatim into the WGSL compute shader by #[gpu_pipeline], which embeds
// ONLY this function's body into `fn main`. Extracting helpers would emit
// calls to functions that do not exist in the shader; splitting requires a
// macro-level helper-inclusion feature, not a local edit.
//
/// Rung-2 AVBD device kernel: per-body inertial Hessian + one stamped
/// contact row + dense 6x6 LDL with breakdown signal (one invocation per
/// body, `global_invocation_id` over `ceil(count/64)` workgroups of 64).
///
/// Each body reads its mass entry, assembles the 6x6 lumped inertial system
/// (same diagonal approximation as rung 1), stamps its contact row (same
/// math as [`avbd_stamp_row_cpu`]), solves the dense LDL (same math as
/// [`solve_6x6`](crate::avbd::solve_6x6)) and writes the delta plus the `ok`
/// flag (`1.0` solved, `0.0` breakdown with a zeroed delta). Multi-row
/// manifolds and engine-step integration stay rung-3 work; the kernel moves
/// no engine state — it pins the buffer/bindings/dispatch shape plus the
/// exact per-body math the CPU tests verify in tolerances.
#[gpu_pipeline(
    workgroup_size = 64,
    storage(avbd_sys: [GpuAvbdSystem; 64], read),
    storage(avbd_delta: [GpuBodyState; 64], read_write),
    storage(avbd_ok: [f32; 64], read_write),
    uniform(avbd_count: [u32; 4]),
    uniform(avbd_dt: [f32; 4]),
    builtin(gid: global_invocation_id),
    helpers(avbd_hessian_lin, avbd_hessian_ang),
)]
fn avbd_row_kernel() {
    // qual:allow(abc) — kernel-DSL body, see the item docs above.
    if gid.x >= avbd_count.x {
        return;
    }
    let s = avbd_sys[gid.x];
    let m = s.mass;
    let r = s.state;
    let row = s.row;
    let dt = avbd_dt.x;
    let hl = avbd_hessian_lin(m.inv_mass, dt);
    let h_ang = avbd_hessian_ang(m.inertia, dt);
    let mut h: [[f32; 6]; 6] = [[0.0; 6]; 6];
    h[0][0] = hl;
    h[1][1] = hl;
    h[2][2] = hl;
    h[3][3] = h_ang[0];
    h[4][4] = h_ang[1];
    h[5][5] = h_ang[2];
    let mut b: [f32; 6] = [0.0; 6];
    b[0] = r.velocity[0];
    b[1] = r.velocity[1];
    b[2] = r.velocity[2];
    b[3] = r.angular[0];
    b[4] = r.angular[1];
    b[5] = r.angular[2];
    // One contact row: nn = sign*axis, t = r×nn (mirrors
    // `avbd_stamp_row_cpu`; components bounce through scratch arrays so
    // the loop indices stay plain array indexing).
    let nn = row.axis * row.sign;
    let t = cross(row.lever, nn);
    let mut na: [f32; 3] = [0.0; 3];
    na[0] = nn[0];
    na[1] = nn[1];
    na[2] = nn[2];
    let mut ta: [f32; 3] = [0.0; 3];
    ta[0] = t[0];
    ta[1] = t[1];
    ta[2] = t[2];
    for x in 0u32..3u32 {
        for y in 0u32..3u32 {
            h[x][y] = h[x][y] + row.pen * na[x] * na[y];
            h[3u32 + x][3u32 + y] = h[3u32 + x][3u32 + y] + row.pen * ta[x] * ta[y];
            h[x][3u32 + y] = h[x][3u32 + y] + row.pen * na[x] * ta[y];
            h[3u32 + x][y] = h[3u32 + x][y] + row.pen * ta[x] * na[y];
        }
        b[x] = b[x] + row.force * na[x];
        b[3u32 + x] = b[3u32 + x] + row.force * ta[x];
    }
    // Dense LDL without pivoting (mirrors `solve_6x6`): breakdown sets the
    // flag and the write below zeroes the delta instead of the garbage the
    // scratch holds (division by a non-positive pivot is well-defined f32
    // arithmetic, never a trap — its result is discarded via `ok`).
    // No `else` arms: the DSL lowers plain `if`s only.
    let mut l: [[f32; 6]; 6] = [[0.0; 6]; 6];
    let mut d: [f32; 6] = [0.0; 6];
    let mut ok = 1.0;
    for i in 0u32..6u32 {
        for j in 0u32..6u32 {
            if j <= i {
                let mut s = h[i][j];
                for k in 0u32..6u32 {
                    if k < j {
                        s = s - l[i][k] * d[k] * l[j][k];
                    }
                }
                if i == j {
                    if s <= 1e-12 {
                        ok = 0.0;
                    }
                    d[i] = s;
                    l[i][i] = 1.0;
                }
                if i != j {
                    l[i][j] = s / d[j];
                }
            }
        }
    }
    let mut y: [f32; 6] = [0.0; 6];
    for i in 0u32..6u32 {
        let mut s = b[i];
        for k in 0u32..6u32 {
            if k < i {
                s = s - l[i][k] * y[k];
            }
        }
        y[i] = s;
    }
    let mut z: [f32; 6] = [0.0; 6];
    for i in 0u32..6u32 {
        z[i] = y[i] / d[i];
    }
    // Back substitution runs top-down (the DSL lowers increasing ranges
    // only: count down via index arithmetic).
    let mut x: [f32; 6] = [0.0; 6];
    for ri in 0u32..6u32 {
        let i: u32 = 5u32 - ri;
        let mut s = z[i];
        for k in 0u32..6u32 {
            if k > i {
                s = s - l[k][i] * x[k];
            }
        }
        x[i] = s;
    }
    if ok > 0.5 {
        avbd_delta[gid.x].velocity = vec3(x[0], x[1], x[2]);
        avbd_delta[gid.x].angular = vec3(x[3], x[4], x[5]);
        avbd_ok[gid.x] = 1.0;
    }
    if ok <= 0.5 {
        avbd_delta[gid.x].velocity = vec3(0.0, 0.0, 0.0);
        avbd_delta[gid.x].angular = vec3(0.0, 0.0, 0.0);
        avbd_ok[gid.x] = 0.0;
    }
}

/// Rung-2 kernel WGSL source: mass/body/row/system layouts +
/// contact-row/dense-LDL entry (no engine state). Pure (no device) so tests
/// pin it without an adapter.
pub fn avbd_row_wgsl() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}",
        GpuAvbdMass::WGSL_SOURCE,
        GpuBodyState::WGSL_SOURCE,
        GpuAvbdRow::WGSL_SOURCE,
        GpuAvbdSystem::WGSL_SOURCE,
        avbd_row_kernel::wgsl_source()
    )
}

// ---------------------------------------------------------------------------
// Rung-2 device solver: buffers + pipeline over `avbd_row_kernel`
// ---------------------------------------------------------------------------

/// Number of bytes per staged GPU AVBD system (see [`GpuAvbdSystem`]; same
/// compile-time layout check as the body/batch strides).
pub const GPU_AVBD_SYSTEM_STRIDE: u64 = std::mem::size_of::<GpuAvbdSystem>() as u64;

/// Rung-2 GPU AVBD solver: dispatches [`avbd_row_kernel`] over staged
/// per-body systems ([`GpuAvbdSystem`]: mass roster + base residual + one
/// contact row) and reads back the dense-LDL delta plus the breakdown flag
/// per body.
///
/// This is a linear-system device path, not an engine step: discovery,
/// multi-row manifolds and iteration stay host-side (rung 3), so no
/// [`AvbdEngine`] holds one yet and [`GpuAvbdDispatch::step_avbd`] is not
/// implemented here — stepping still falls back to the CPU
/// ([`GpuAvbdStub`], whose `runs_on_device` stays `false`). This solver's
/// own [`WgpuAvbdSolver::runs_on_device`] returns `true`: it really
/// dispatches, proven by the device round-trip test against
/// [`avbd_stage_contact_solve`] in tolerance per member plus the flag.
pub struct WgpuAvbdSolver {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    sys_buf: wgpu::Buffer,
    delta_buf: wgpu::Buffer,
    ok_buf: wgpu::Buffer,
    count_buf: wgpu::Buffer,
    dt_buf: wgpu::Buffer,
    readback_delta: wgpu::Buffer,
    readback_ok: wgpu::Buffer,
    /// Buffer-size bound for one dispatch (not a physics bound — mirrors
    /// the [`GpuAvbdStub`] / `GpuSequentialImpulse` caps).
    pub max_bodies: usize,
}

impl WgpuAvbdSolver {
    /// Create a solver attached to the given wgpu context. `max_bodies`
    /// must cover the staged systems.
    pub fn new(
        device: std::sync::Arc<wgpu::Device>,
        queue: std::sync::Arc<wgpu::Queue>,
        max_bodies: usize,
    ) -> Self {
        let source = avbd_row_wgsl();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics_avbd_row"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("physics_avbd_row_bgl"),
            entries: &[
                storage(0, true),
                storage(1, false),
                storage(2, false),
                uniform(BINDING_COUNT),
                uniform(BINDING_DT),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physics_avbd_row_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("physics_avbd_row_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let cap = max_bodies.next_power_of_two().max(MIN_BUFFER_CAP) as u64;
        let storage_buf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sys_buf = storage_buf("physics_avbd_systems", cap * GPU_AVBD_SYSTEM_STRIDE);
        let delta_buf = storage_buf("physics_avbd_delta", cap * super::GPU_BODY_STRIDE);
        let ok_buf = storage_buf("physics_avbd_ok", cap * F32_BYTES);
        let uniform_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: UNIFORM_BLOCK_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let count_buf = uniform_buf("physics_avbd_count");
        let dt_buf = uniform_buf("physics_avbd_dt");
        let readback = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let readback_delta = readback("physics_avbd_delta_readback", cap * super::GPU_BODY_STRIDE);
        let readback_ok = readback("physics_avbd_ok_readback", cap * F32_BYTES);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("physics_avbd_row_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sys_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: delta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ok_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_COUNT,
                    resource: count_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: BINDING_DT,
                    resource: dt_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group,
            sys_buf,
            delta_buf,
            ok_buf,
            count_buf,
            dt_buf,
            readback_delta,
            readback_ok,
            max_bodies,
        }
    }

    /// Whether dispatch touches the device. Always `true`: [`solve`](Self::solve)
    /// records and submits a real compute pass (proven by the device
    /// round-trip test, skipped only where no adapter exists).
    pub fn runs_on_device(&self) -> bool {
        true
    }

    /// Upload staged per-body systems ([`GpuAvbdSystem`]: mass roster entry,
    /// base residual, contact row). The slice must fit `max_bodies`; the
    /// buffer tail is zeroed so stale systems never leak into a shorter
    /// dispatch.
    pub fn upload(&self, systems: &[GpuAvbdSystem]) {
        assert!(
            systems.len() <= self.max_bodies,
            "staged {} systems exceed max_bodies {}",
            systems.len(),
            self.max_bodies
        );
        let mut data = vec![GpuAvbdSystem::zeroed(); self.max_bodies];
        data[..systems.len()].copy_from_slice(systems);
        self.queue
            .write_buffer(&self.sys_buf, 0, bytemuck::cast_slice(&data));
    }

    /// Dispatch the row kernel over the first `count` staged systems with
    /// timestep `dt`, then block until the pass completes. No-op for
    /// `count == 0`.
    pub fn solve(&self, count: u32, dt: f32) {
        if count == 0 {
            return;
        }
        assert!(
            usize::try_from(count).unwrap_or(usize::MAX) <= self.max_bodies,
            "solve count {count} exceeds max_bodies {}",
            self.max_bodies
        );
        self.queue
            .write_buffer(&self.count_buf, 0, bytemuck::cast_slice(&[count, 0, 0, 0]));
        self.queue
            .write_buffer(&self.dt_buf, 0, bytemuck::cast_slice(&[dt, 0.0, 0.0, 0.0]));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics_avbd_row"),
            });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("physics_avbd_row_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
    }

    /// Download the per-body deltas and breakdown flags (`1.0` solved,
    /// `0.0` breakdown). Returns `max_bodies` entries; only the first
    /// `count` dispatched ones are meaningful.
    ///
    /// Fallible path: both `map_async` outcomes are awaited through a
    /// channel (same pattern as
    /// [`GpuSequentialImpulse::try_download_bodies`](crate::gpu::GpuSequentialImpulse::try_download_bodies))
    /// instead of fire-and-forget closures. If the second mapping fails after the
    /// first succeeded, the first buffer is unmapped before returning, so
    /// the persistent readback buffers stay reusable for the next download.
    ///
    /// # Errors
    ///
    /// [`GpuDispatchError`] when either device mapping or either
    /// mapped-range view fails. The numeric path is unchanged on success.
    pub fn try_download(&self) -> Result<(Vec<GpuBodyState>, Vec<f32>), GpuDispatchError> {
        let delta_size = self.max_bodies as u64 * super::GPU_BODY_STRIDE;
        let ok_size = self.max_bodies as u64 * 4;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics_avbd_row_dl"),
            });
        encoder.copy_buffer_to_buffer(&self.delta_buf, 0, &self.readback_delta, 0, delta_size);
        encoder.copy_buffer_to_buffer(&self.ok_buf, 0, &self.readback_ok, 0, ok_size);
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        let delta_slice = self.readback_delta.slice(..);
        let ok_slice = self.readback_ok.slice(..);
        super::await_map(&self.device, &delta_slice)?;
        if let Err(e) = super::await_map(&self.device, &ok_slice) {
            self.readback_delta.unmap();
            return Err(e);
        }
        match (delta_slice.get_mapped_range(), ok_slice.get_mapped_range()) {
            (Ok(delta_mapped), Ok(ok_mapped)) => {
                let deltas: Vec<GpuBodyState> = bytemuck::cast_slice(&delta_mapped).to_vec();
                drop(delta_mapped);
                self.readback_delta.unmap();
                let oks: Vec<f32> = bytemuck::cast_slice(&ok_mapped).to_vec();
                drop(ok_mapped);
                self.readback_ok.unmap();
                Ok((deltas, oks))
            }
            (Ok(delta_mapped), Err(e)) => {
                drop(delta_mapped);
                self.readback_delta.unmap();
                Err(e.into())
            }
            (Err(e), Ok(ok_mapped)) => {
                drop(ok_mapped);
                self.readback_ok.unmap();
                Err(e.into())
            }
            (Err(e), Err(_)) => Err(e.into()),
        }
    }

    /// Download the per-body deltas and breakdown flags (`1.0` solved,
    /// `0.0` breakdown). Returns `max_bodies` entries; only the first
    /// `count` dispatched ones are meaningful.
    ///
    /// Legacy wrapper over [`try_download`](Self::try_download): panics on
    /// a device mapping failure — the same failure class the old code
    /// surfaced as a `get_mapped_range` panic (numeric path unchanged).
    pub fn download(&self) -> (Vec<GpuBodyState>, Vec<f32>) {
        self.try_download()
            .expect("GPU AVBD download: buffer mapping failed")
    }
}
