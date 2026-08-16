//! GPU-accelerated wide contact solver (G7) — `gpu` feature only.
//!
//! Offloads single-point contact constraint solving to the GPU through wgpu
//! compute shaders. Each workgroup (4 invocations) processes one wide batch
//! of up to 4 single-point contacts — the same batching strategy as the CPU
//! SIMD-wide path (`WideBatch`), but running on the GPU.
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
//! # WGSL compute shader
//!
//! A single workgroup (4 invocations) reads one `ContactBatch`, gathers the
//! body velocities, solves the normal + friction constraints per lane, and
//! writes back the updated velocities and accumulated impulses. The host
//! dispatches `N_iterations + restitution` times using ping-pong body buffers.

use glam::Vec3;
use std::sync::Arc;
use wgpu;

use bytemuck::Zeroable;
use crate::body::RigidBody;
use crate::engine::{Manifold, ManifoldState};

// ---------------------------------------------------------------------------
// Layout constants (match WGSL struct)
// ---------------------------------------------------------------------------

/// Number of bytes per GPU body state (6 f32 + 2 pad = 32).
const GPU_BODY_STRIDE: u64 = 32;

/// Number of bytes per GPU batch (see `GpuBatch` layout below).
const GPU_BATCH_STRIDE: u64 = 2144; // ~536 f32 = 2144 bytes

// ---------------------------------------------------------------------------
// GPU body state (32 bytes, matches BodyState in WGSL)
// ---------------------------------------------------------------------------

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBodyState {
    velocity: [f32; 3],
    angular: [f32; 3],
    _pad: [f32; 2],
}

impl GpuBodyState {
    fn from_body(b: &RigidBody) -> Self {
        Self {
            velocity: b.velocity.to_array(),
            angular: b.angular_velocity.to_array(),
            _pad: [0.0; 2],
        }
    }

    fn write_to_body(&self, b: &mut RigidBody) {
        b.velocity = Vec3::from_array(self.velocity);
        b.angular_velocity = Vec3::from_array(self.angular);
    }
}

// ---------------------------------------------------------------------------
// GPU batch (matches ContactBatch in the WGSL shader)
// ---------------------------------------------------------------------------

#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuBatch {
    // Geometry: SoA layout, 4 lanes each → [f32; 4]
    nx: [f32; 4], ny: [f32; 4], nz: [f32; 4],
    rax: [f32; 4], ray: [f32; 4], raz: [f32; 4],
    rbx: [f32; 4], rby: [f32; 4], rbz: [f32; 4],
    ra_nx: [f32; 4], ra_ny: [f32; 4], ra_nz: [f32; 4],
    rb_nx: [f32; 4], rb_ny: [f32; 4], rb_nz: [f32; 4],
    t1x: [f32; 4], t1y: [f32; 4], t1z: [f32; 4],
    t2x: [f32; 4], t2y: [f32; 4], t2z: [f32; 4],
    // Precomputed lever × tan
    ra_t1x: [f32; 4], ra_t1y: [f32; 4], ra_t1z: [f32; 4],
    rb_t1x: [f32; 4], rb_t1y: [f32; 4], rb_t1z: [f32; 4],
    ra_t2x: [f32; 4], ra_t2y: [f32; 4], ra_t2z: [f32; 4],
    rb_t2x: [f32; 4], rb_t2y: [f32; 4], rb_t2z: [f32; 4],
    // Scalar constants per lane
    inv_ma: [f32; 4], inv_mb: [f32; 4],
    total_inv: [f32; 4],
    inv_k_n: [f32; 4], inv_k_t1: [f32; 4], inv_k_t2: [f32; 4],
    // Linear application factors: n * inv_mass
    apply_n_ax: [f32; 4], apply_n_ay: [f32; 4], apply_n_az: [f32; 4],
    apply_n_bx: [f32; 4], apply_n_by: [f32; 4], apply_n_bz: [f32; 4],
    // Angular application factors: I⁻¹_world · (ra×n)  (precomputed)
    apply_w_ax: [f32; 4], apply_w_ay: [f32; 4], apply_w_az: [f32; 4],
    apply_w_bx: [f32; 4], apply_w_by: [f32; 4], apply_w_bz: [f32; 4],
    // World-space inverse inertia matrix rows (3 rows × 2 bodies)
    w_a00: [f32; 4], w_a01: [f32; 4], w_a02: [f32; 4],
    w_a10: [f32; 4], w_a11: [f32; 4], w_a12: [f32; 4],
    w_a20: [f32; 4], w_a21: [f32; 4], w_a22: [f32; 4],
    w_b00: [f32; 4], w_b01: [f32; 4], w_b02: [f32; 4],
    w_b10: [f32; 4], w_b11: [f32; 4], w_b12: [f32; 4],
    w_b20: [f32; 4], w_b21: [f32; 4], w_b22: [f32; 4],
    // Restitution bias, speculative target, friction coeff
    bias: [f32; 4], target: [f32; 4], mu: [f32; 4],
    // Body indices
    body_a: [u32; 4], body_b: [u32; 4],
    // Accumulators (read-write on GPU)
    acc: [f32; 4], acc_f1: [f32; 4], acc_f2: [f32; 4],
    // Active lane count
    count: u32,
    _pad: u32,
}

impl GpuBatch {
    fn zero() -> Self { Self::zeroed() }

    /// Fill one lane from CPU-side contact data.
    fn fill_lane(
        &mut self,
        lane: usize,
        n: Vec3, ra: Vec3, rb: Vec3,
        target: f32, mu: f32, bias: f32, acc_in: f32,
        a: &RigidBody, ba_idx: u32, b: &RigidBody, bb_idx: u32,
    ) {
        let l = |buf: &mut [f32; 4]| &mut buf[lane];
        let l3 = |x: &mut [f32; 4], y: &mut [f32; 4], z: &mut [f32; 4], v: Vec3| {
            x[lane] = v.x; y[lane] = v.y; z[lane] = v.z;
        };

        l3(&mut self.nx, &mut self.ny, &mut self.nz, n);
        l3(&mut self.rax, &mut self.ray, &mut self.raz, ra);
        l3(&mut self.rbx, &mut self.rby, &mut self.rbz, rb);
        *l(&mut self.target) = target;
        *l(&mut self.mu) = mu;
        *l(&mut self.bias) = bias;
        *l(&mut self.acc) = acc_in;
        self.body_a[lane] = ba_idx;
        self.body_b[lane] = bb_idx;
        *l(&mut self.inv_ma) = a.inv_mass;
        *l(&mut self.inv_mb) = b.inv_mass;
        *l(&mut self.total_inv) = a.inv_mass + b.inv_mass;

        // Precompute cross products.
        let ra_n = ra.cross(n);
        let rb_n = rb.cross(n);
        l3(&mut self.ra_nx, &mut self.ra_ny, &mut self.ra_nz, ra_n);
        l3(&mut self.rb_nx, &mut self.rb_ny, &mut self.rb_nz, rb_n);

        // Tangent basis.
        let t1 = tangent_basis(n);
        let t2 = t1.cross(n);
        l3(&mut self.t1x, &mut self.t1y, &mut self.t1z, t1);
        l3(&mut self.t2x, &mut self.t2y, &mut self.t2z, t2);
        l3(&mut self.ra_t1x, &mut self.ra_t1y, &mut self.ra_t1z, ra.cross(t1));
        l3(&mut self.rb_t1x, &mut self.rb_t1y, &mut self.rb_t1z, rb.cross(t1));
        l3(&mut self.ra_t2x, &mut self.ra_t2y, &mut self.ra_t2z, ra.cross(t2));
        l3(&mut self.rb_t2x, &mut self.rb_t2y, &mut self.rb_t2z, rb.cross(t2));

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
        set_mat(&wa, &mut [&mut self.w_a00, &mut self.w_a01, &mut self.w_a02,
                          &mut self.w_a10, &mut self.w_a11, &mut self.w_a12,
                          &mut self.w_a20, &mut self.w_a21, &mut self.w_a22]);
        set_mat(&wb, &mut [&mut self.w_b00, &mut self.w_b01, &mut self.w_b02,
                          &mut self.w_b10, &mut self.w_b11, &mut self.w_b12,
                          &mut self.w_b20, &mut self.w_b21, &mut self.w_b22]);

        // Application factors.
        l3(&mut self.apply_n_ax, &mut self.apply_n_ay, &mut self.apply_n_az, n * a.inv_mass);
        l3(&mut self.apply_n_bx, &mut self.apply_n_by, &mut self.apply_n_bz, n * b.inv_mass);
        l3(&mut self.apply_w_ax, &mut self.apply_w_ay, &mut self.apply_w_az, matvec(&wa, ra_n));
        l3(&mut self.apply_w_bx, &mut self.apply_w_by, &mut self.apply_w_bz, matvec(&wb, rb_n));

        // Effective stiffness inverses.
        let total = a.inv_mass + b.inv_mass;
        let k_n = total + ra_n.dot(matvec(&wa, ra_n)) + rb_n.dot(matvec(&wb, rb_n));
        let k_t1 = total + ra.cross(t1).dot(matvec(&wa, ra.cross(t1)))
                       + rb.cross(t1).dot(matvec(&wb, rb.cross(t1)));
        let k_t2 = total + ra.cross(t2).dot(matvec(&wa, ra.cross(t2)))
                       + rb.cross(t2).dot(matvec(&wb, rb.cross(t2)));
        *l(&mut self.inv_k_n) = if k_n >= 1e-10 { 1.0 / k_n } else { 0.0 };
        *l(&mut self.inv_k_t1) = if k_t1 >= 1e-10 { 1.0 / k_t1 } else { 0.0 };
        *l(&mut self.inv_k_t2) = if k_t2 >= 1e-10 { 1.0 / k_t2 } else { 0.0 };
    }
}

#[inline]
fn tangent_basis(n: Vec3) -> Vec3 {
    let axis = if n.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    n.cross(axis).normalize_or(Vec3::Z)
}

/// World inverse inertia as `[row0, row1, row2]` (3 Vec3 columns → row-major).
fn world_inertia_matrix(rot: glam::Quat, inertia: Vec3) -> [Vec3; 3] {
    let inv = Vec3::new(
        if inertia.x > 0.0 { 1.0 / inertia.x } else { 0.0 },
        if inertia.y > 0.0 { 1.0 / inertia.y } else { 0.0 },
        if inertia.z > 0.0 { 1.0 / inertia.z } else { 0.0 },
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
// WGSL shader source
// ---------------------------------------------------------------------------

const CONTACT_SOLVER_WGSL: &str = r#"
struct BodyState {
    velocity: vec3<f32>,
    angular: vec3<f32>,
};

struct ContactBatch {
    nx: vec4<f32>, ny: vec4<f32>, nz: vec4<f32>,
    rax: vec4<f32>, ray: vec4<f32>, raz: vec4<f32>,
    rbx: vec4<f32>, rby: vec4<f32>, rbz: vec4<f32>,
    ra_nx: vec4<f32>, ra_ny: vec4<f32>, ra_nz: vec4<f32>,
    rb_nx: vec4<f32>, rb_ny: vec4<f32>, rb_nz: vec4<f32>,
    t1x: vec4<f32>, t1y: vec4<f32>, t1z: vec4<f32>,
    t2x: vec4<f32>, t2y: vec4<f32>, t2z: vec4<f32>,
    ra_t1x: vec4<f32>, ra_t1y: vec4<f32>, ra_t1z: vec4<f32>,
    rb_t1x: vec4<f32>, rb_t1y: vec4<f32>, rb_t1z: vec4<f32>,
    ra_t2x: vec4<f32>, ra_t2y: vec4<f32>, ra_t2z: vec4<f32>,
    rb_t2x: vec4<f32>, rb_t2y: vec4<f32>, rb_t2z: vec4<f32>,
    inv_ma: vec4<f32>, inv_mb: vec4<f32>,
    total_inv: vec4<f32>,
    inv_k_n: vec4<f32>, inv_k_t1: vec4<f32>, inv_k_t2: vec4<f32>,
    apply_n_ax: vec4<f32>, apply_n_ay: vec4<f32>, apply_n_az: vec4<f32>,
    apply_n_bx: vec4<f32>, apply_n_by: vec4<f32>, apply_n_bz: vec4<f32>,
    apply_w_ax: vec4<f32>, apply_w_ay: vec4<f32>, apply_w_az: vec4<f32>,
    apply_w_bx: vec4<f32>, apply_w_by: vec4<f32>, apply_w_bz: vec4<f32>,
    w_a00: vec4<f32>, w_a01: vec4<f32>, w_a02: vec4<f32>,
    w_a10: vec4<f32>, w_a11: vec4<f32>, w_a12: vec4<f32>,
    w_a20: vec4<f32>, w_a21: vec4<f32>, w_a22: vec4<f32>,
    w_b00: vec4<f32>, w_b01: vec4<f32>, w_b02: vec4<f32>,
    w_b10: vec4<f32>, w_b11: vec4<f32>, w_b12: vec4<f32>,
    w_b20: vec4<f32>, w_b21: vec4<f32>, w_b22: vec4<f32>,
    bias: vec4<f32>, target: vec4<f32>, mu: vec4<f32>,
    body_a: vec4<u32>, body_b: vec4<u32>,
    acc: vec4<f32>, acc_f1: vec4<f32>, acc_f2: vec4<f32>,
    count: u32,
};

@group(0) @binding(0) var<storage, read_write> body_buf: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> batch_buf: array<ContactBatch>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

fn cross(a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

fn matvec_row(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3(dot(r0, v), dot(r1, v), dot(r2, v));
}

fn apply_angular(w_row0: vec3<f32>, w_row1: vec3<f32>, w_row2: vec3<f32>,
                 lever: vec3<f32>, imp: vec3<f32>) -> vec3<f32> {
    let crs = cross(imp, lever);
    return matvec_row(w_row0, w_row1, w_row2, crs);
}

@compute @workgroup_size(4)
fn main(@builtin(workgroup_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let b = &batch_buf[gid.x];
    let l = lid.x;
    if l >= b.count { return; }
    let iter = params.x;
    let total = params.y;
    let allow_rest = params.z;

    // Gather body velocities.
    var ba = body_buf[b.body_a[l]];
    var bb = body_buf[b.body_b[l]];
    let va = ba.velocity;
    let wa = ba.angular;
    let vb = bb.velocity;
    let wb = bb.angular;

    let n   = vec3<f32>(b.nx[l], b.ny[l], b.nz[l]);
    let ra  = vec3<f32>(b.rax[l], b.ray[l], b.raz[l]);
    let rb  = vec3<f32>(b.rbx[l], b.rby[l], b.rbz[l]);
    let ra_n = vec3<f32>(b.ra_nx[l], b.ra_ny[l], b.ra_nz[l]);
    let inv_ma = b.inv_ma[l];
    let inv_mb = b.inv_mb[l];
    let inv_k = b.inv_k_n[l];
    let target = b.target[l];
    var acc = b.acc[l];

    // Normal impulse.
    let rel = (vb + cross(wb, rb)) - (va + cross(wa, ra));
    let vn = dot(rel, n);
    let lambda = (target - vn) * inv_k;
    let new_acc = max(acc + lambda, 0.0);
    let delta = new_acc - acc;
    acc = new_acc;

    if abs(delta) > 1e-12 {
        ba.velocity -= delta * vec3<f32>(b.apply_n_ax[l], b.apply_n_ay[l], b.apply_n_az[l]);
        bb.velocity += delta * vec3<f32>(b.apply_n_bx[l], b.apply_n_by[l], b.apply_n_bz[l]);
        ba.angular -= delta * vec3<f32>(b.apply_w_ax[l], b.apply_w_ay[l], b.apply_w_az[l]);
        bb.angular += delta * vec3<f32>(b.apply_w_bx[l], b.apply_w_by[l], b.apply_w_bz[l]);
    }

    // Friction (remeasure rel after normal impulse).
    let rel2 = (bb.velocity + cross(bb.angular, rb)) - (ba.velocity + cross(ba.angular, ra));
    let max_f = b.mu[l] * acc;
    let t1 = vec3<f32>(b.t1x[l], b.t1y[l], b.t1z[l]);
    let t2 = vec3<f32>(b.t2x[l], b.t2y[l], b.t2z[l]);
    var f_imp = vec3<f32>(0.0);

    // Axis 1.
    if b.inv_k_t1[l] > 0.0 {
        let vt1 = dot(rel2, t1);
        let new_t1 = b.acc_f1[l] - vt1 * b.inv_k_t1[l];
        let len1 = sqrt(new_t1 * new_t1 + b.acc_f2[l] * b.acc_f2[l]);
        let new_t1c = select(new_t1, new_t1 * (max_f / len1), len1 > max_f && len1 > 1e-12);
        f_imp += t1 * (new_t1c - b.acc_f1[l]);
        b.acc_f1[l] = new_t1c;
    }
    // Axis 2.
    if b.inv_k_t2[l] > 0.0 {
        let vt2 = dot(rel2, t2);
        let new_t2 = b.acc_f2[l] - vt2 * b.inv_k_t2[l];
        let len2 = sqrt(new_t2 * new_t2 + b.acc_f1[l] * b.acc_f1[l]);
        let new_t2c = select(new_t2, new_t2 * (max_f / len2), len2 > max_f && len2 > 1e-12);
        f_imp += t2 * (new_t2c - b.acc_f2[l]);
        b.acc_f2[l] = new_t2c;
    }
    if length_sq(f_imp) > 1e-24 {
        let w_a = matvec_row(
            vec3<f32>(b.w_a00[l], b.w_a01[l], b.w_a02[l]),
            vec3<f32>(b.w_a10[l], b.w_a11[l], b.w_a12[l]),
            vec3<f32>(b.w_a20[l], b.w_a21[l], b.w_a22[l]),
            cross(f_imp, ra));
        let w_b = matvec_row(
            vec3<f32>(b.w_b00[l], b.w_b01[l], b.w_b02[l]),
            vec3<f32>(b.w_b10[l], b.w_b11[l], b.w_b12[l]),
            vec3<f32>(b.w_b20[l], b.w_b21[l], b.w_b22[l]),
            cross(f_imp, rb));
        ba.velocity -= f_imp * inv_ma;
        bb.velocity += f_imp * inv_mb;
        ba.angular -= w_a;
        bb.angular += w_b;
    }

    // Restitution (one-shot on last iteration).
    if allow_rest > 0u && iter == total - 1u {
        let bias = b.bias[l];
        if bias > 0.0 {
            let rel3 = (bb.velocity + cross(bb.angular, rb)) - (ba.velocity + cross(ba.angular, ra));
            let vn3 = dot(rel3, n);
            let lr = (bias - vn3) * inv_k;
            if lr > 0.0 {
                ba.velocity -= lr * vec3<f32>(b.apply_n_ax[l], b.apply_n_ay[l], b.apply_n_az[l]);
                bb.velocity += lr * vec3<f32>(b.apply_n_bx[l], b.apply_n_by[l], b.apply_n_bz[l]);
                ba.angular -= lr * vec3<f32>(b.apply_w_ax[l], b.apply_w_ay[l], b.apply_w_az[l]);
                bb.angular += lr * vec3<f32>(b.apply_w_bx[l], b.apply_w_by[l], b.apply_w_bz[l]);
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
"#;

// ---------------------------------------------------------------------------
// WGPU contact solver
// ---------------------------------------------------------------------------

/// GPU-accelerated contact solver for single-point manifold batches.
pub struct WgpuContactSolver {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    body_buf: wgpu::Buffer,       // read-write body state
    batch_buf: wgpu::Buffer,      // read-write batch data (acc accumulators)
    uniform_buf: wgpu::Buffer,    // params: (iter, total, allow_rest, 0)
    readback_buf: wgpu::Buffer,   // staging copy for body download
    max_bodies: usize,
    max_batches: usize,
}

impl WgpuContactSolver {
    /// Create a new GPU solver attached to the given wgpu context.
    /// `max_bodies` and `max_batches` must be large enough for the scene.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        max_bodies: usize,
        max_batches: usize,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("physics_contact"),
            source: wgpu::ShaderSource::Wgsl(CONTACT_SOLVER_WGSL.into()),
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
                    // uniform: uniform buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("physics_contact_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("physics_contact_uniform"),
            size: 16, // vec4<u32>
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
            layout: &pipeline_layout.bind_group_layouts[0],
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: body_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: batch_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: uniform_buf.as_entire_binding() },
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
        }
    }

    /// Upload body velocities to the GPU buffer.
    pub fn upload_bodies(&self, bodies: &[RigidBody]) {
        let n = bodies.len().min(self.max_bodies);
        let mut data = vec![GpuBodyState::zeroed(); self.max_bodies];
        for (i, b) in bodies.iter().enumerate().take(n) {
            data[i] = GpuBodyState::from_body(b);
        }
        self.queue.write_buffer(&self.body_buf, 0, bytemuck::cast_slice(&data));
    }

    /// Download body velocities from the GPU buffer (blocking).
    pub fn download_bodies(&self, bodies: &mut [RigidBody]) {
        let n = bodies.len().min(self.max_bodies);
        let copy_size = self.max_bodies as u64 * GPU_BODY_STRIDE;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("physics_download"),
        });
        encoder.copy_buffer_to_buffer(&self.body_buf, 0, &self.readback_buf, 0, copy_size);
        self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let slice = self.readback_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let mapped = slice.get_mapped_range();
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
        self.queue.write_buffer(&self.batch_buf, 0, bytemuck::cast_slice(&data));
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
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("physics_acc_dl"),
        });
        encoder.copy_buffer_to_buffer(&self.batch_buf, 0, &readback, 0, copy_size);
        self.queue.submit([encoder.finish()]);
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let mapped = readback.slice(..).get_mapped_range();
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
    /// restitution dispatch if `allow_restitution`.
    pub fn solve(
        &self,
        num_batches: u32,
        iterations: u32,
        allow_restitution: bool,
    ) {
        for i in 0..iterations {
            let params = [i, iterations, if allow_restitution { 1 } else { 0 }, 0];
            self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&params));
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("physics_contact_iter"),
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("physics_contact_pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.dispatch_workgroups(num_batches, 1, 1);
            }
            self.queue.submit([encoder.finish()]);
            // Barrier between iterations: wait for the previous dispatch.
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
        }
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
            flush_batch(&mut batches, &mut cur, &mut cur_len, &mut cur_bodies, &mut cur_n);
        }
        let p = m.points[0].world_point;
        let ra = p - bodies[i].position;
        let rb = p - bodies[j].position;
        cur.fill_lane(
            cur_len,
            m.normal, ra, rb,
            st.target[0], st.mu, st.bias[0], st.acc[0],
            &bodies[i], i as u32,
            &bodies[j], j as u32,
        );
        cur_bodies[cur_n] = i;
        cur_bodies[cur_n + 1] = j;
        cur_n += 2;
        cur_len += 1;
        cur.count = cur_len as u32;
    }
    flush_batch(&mut batches, &mut cur, &mut cur_len, &mut cur_bodies, &mut cur_n);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::RigidBody;
    use crate::engine::{Manifold, ManifoldPoint, ManifoldState};

    #[test]
    fn gpu_pack_produces_disjoint_batches() {
        let bodies = vec![
            RigidBody::new_sphere(Vec3::ZERO, 0.5, 0.0),    // 0: static
            RigidBody::new_sphere(Vec3::new(1.0, 0.0, 0.0), 0.5, 1.0), // 1
            RigidBody::new_sphere(Vec3::new(2.0, 0.0, 0.0), 0.5, 1.0), // 2
            RigidBody::new_sphere(Vec3::new(3.0, 0.0, 0.0), 0.5, 1.0), // 3
            RigidBody::new_sphere(Vec3::new(4.0, 0.0, 0.0), 0.5, 1.0), // 4
        ];
        // Single-point manifold helper
        fn mk_manifold(i: usize, j: usize, n: Vec3, p: Vec3) -> Manifold {
            Manifold {
                body_a: i, body_b: j, normal: n, point_count: 1,
                points: [ManifoldPoint { world_point: p, penetration: 0.01 }; 4],
            }
        }
        fn mk_state(i: usize, j: usize) -> ManifoldState {
            ManifoldState {
                mi: 0, i, j, count: 1, acc: [0.0; 4], acc_friction: [0.0; 4],
                acc_friction2: [0.0; 4], bias: [0.0; 4], target: [0.0; 4],
                mu: 0.3, t1: Vec3::X, t2: Vec3::Z,
                la: [Vec3::ZERO; 4], lb: [Vec3::ZERO; 4], pen0: [0.0; 4],
            }
        }

        // Create contacts (0,1), (0,2) — conflict on 0
        let manifolds = vec![
            mk_manifold(0, 1, Vec3::X, Vec3::ZERO),
            mk_manifold(0, 2, Vec3::X, Vec3::ZERO),
            mk_manifold(3, 4, Vec3::X, Vec3::new(3.5, 0.0, 0.0)),
        ];
        let states: Vec<ManifoldState> = (0..3).map(|i| {
            let m = &manifolds[i];
            mk_state(m.body_a, m.body_b)
        }).collect();
        let single_indices: Vec<usize> = (0..3).collect();

        let (batches, count) =
            pack_single_point_batches(&bodies, &states, &manifolds, &single_indices);
        assert_eq!(count, 2, "should form 2 batches: [1, 2] with conflict");
        assert_eq!(batches[0].count, 1, "first batch only has the non-conflicting contact... wait");
        // Actually with greedy pack: (0,1) starts batch; (0,2) conflicts → flush batch1(1), start batch2 with (0,2); (3,4) clears bodies → adds to batch2.
        // So batch[0].count = 1 (just 0,1), batch[1].count = 2 (0,2 + 3,4).
        assert_eq!(batches[1].count, 2, "second batch has 2 lanes");
    }
}