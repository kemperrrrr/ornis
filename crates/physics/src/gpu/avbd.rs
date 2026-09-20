//! GPU AVBD stub: inertial mass roster and the rung-1 diagonal kernel (no device yet).
//!
//! Owns [`GpuAvbdMass`], the [`avbd_inertial_hessian_diag`]/[`avbd_diag_solve_cpu`]
//! CPU references and the [`GpuAvbdStub`]/[`GpuAvbdDispatch`] seam with its
//! `avbd_stub_kernel` entry (inertial Hessian diagonal + diagonal LDL, helpers
//! stitched via `helpers(...)`). Stepping always falls back to the CPU
//! [`crate::avbd::AvbdEngine`]. Moved verbatim from `gpu.rs` (phase 3).

use ornis_macros::{WgslStruct, gpu_pipeline, wgsl_fn};

use super::GpuBodyState;
use crate::avbd::AvbdEngine;
use crate::body::{BodyType, RigidBody};
use crate::engine::PhysicsEngine;

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
