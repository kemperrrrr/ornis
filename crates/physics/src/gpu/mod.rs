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
//! shader) lands rung by rung: rung 1 assembles the lumped inertial
//! Hessian diagonal and solves the diagonal LDL system per body in
//! `avbd_stub_kernel` (helpers stitched via `helpers(...)`); rung 2
//! (`avbd_row_kernel` in `avbd.rs`) stamps one contact row per body and
//! solves the dense 6x6 LDL with a breakdown flag on the device through
//! [`WgpuAvbdSolver`]. Engine-step integration (host discovery staging rows
//! into the device solve) stays rung-3 work: stepping a
//! [`crate::avbd::AvbdEngine`] still falls back to the CPU. The DSL
//! preconditions are closed: `ShaderType::Mat3` (`mat3x3<f32>`,
//! `Mat3::from_cols` constructor, `Mat3::IDENTITY`/`ZERO`), local
//! fixed-size scratch arrays (incl. nested Hessian shapes; effectful
//! repeats rejected loudly), `u32` range `for` loops (increasing only —
//! back substitution counts down via index arithmetic), and `helpers(...)`
//! inclusion in `#[gpu_pipeline]` (stitches `#[wgsl_fn]`/`#[kernel]`
//! sources ahead of the entry). Pinned by `macros/tests/compute_dsl.rs`
//! and `helpers_stitch_ahead_of_main_and_validate`.

mod avbd;
mod si_batches;

pub use avbd::{
    GPU_AVBD_SYSTEM_STRIDE, GpuAvbdDispatch, GpuAvbdMass, GpuAvbdRow, GpuAvbdStub, GpuAvbdSystem,
    WgpuAvbdSolver, avbd_diag_solve_cpu, avbd_inertial_hessian_diag, avbd_ldl_6x6_cpu,
    avbd_row_wgsl, avbd_stage_contact_solve, avbd_stamp_row_cpu, avbd_stub_wgsl,
};
pub use si_batches::{
    GpuBatch, LaneInput, contact_solver_wgsl, pack_single_point_batches, write_back_acc,
};

use glam::Vec3;
use ornis_macros::WgslStruct;
use std::sync::Arc;

use crate::body::RigidBody;
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
/// Minimum power-of-two body/batch capacity for SI GPU buffers.
const MIN_BUFFER_CAP: usize = 64;
/// Bytes reserved for a single uniform params entry (`vec4`).
const UNIFORM_ENTRY_BYTES: u64 = 16;

/// Per-pass solver params `(iter, total, allow_rest, 0)` for a bulk
/// dispatch: pass `k` reads entry `k`, so the shader sees the same
/// per-iteration values as the old one-dispatch-per-iteration loop.
/// Pure (no device) so unit tests pin the layout.
pub fn solve_params(iterations: u32, gate: crate::flags::RestitutionGate) -> Vec<[u32; 4]> {
    (0..iterations)
        .map(|k| [k, iterations, u32::from(gate.is_enabled()), 0])
        .collect()
}

/// Boolean-compat wrapper for [`solve_params`] (kept for tests).
pub fn solve_params_bool(iterations: u32, allow_restitution: bool) -> Vec<[u32; 4]> {
    solve_params(
        iterations,
        crate::flags::RestitutionGate::from(allow_restitution),
    )
}

// ---------------------------------------------------------------------------
// Dispatch hygiene: reported truncation + observed buffer mapping
// ---------------------------------------------------------------------------

/// How many inputs one staging upload kept vs dropped at the capacity bound.
///
/// Uploads stage the first `max` entries and zero the buffer tail; inputs
/// past the bound keep their previous values and are counted here instead
/// of vanishing silently. The staged bytes are identical to the old
/// `take(n)` path — only the counts are new.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchReport {
    /// Entries staged into the GPU buffer.
    pub staged: usize,
    /// Inputs past the capacity bound (not staged, previous values kept).
    pub dropped: usize,
}

impl DispatchReport {
    fn split(len: usize, cap: usize) -> Self {
        let staged = len.min(cap);
        Self {
            staged,
            dropped: len.saturating_sub(staged),
        }
    }
}

/// Typed GPU dispatch failure: the device refused a buffer mapping the host
/// waits on. Returned by the `try_download_*` paths; the legacy infallible
/// wrappers keep the old panic-on-driver-failure behavior.
#[derive(Debug, Clone, thiserror::Error)]
pub enum GpuDispatchError {
    /// The `map_async` callback reported a device mapping failure.
    #[error("GPU buffer mapping failed")]
    MapFailed(#[from] wgpu::BufferAsyncError),
    /// The buffer mapped but no view could be taken for the slice.
    #[error("GPU mapped range unavailable")]
    RangeUnavailable(#[from] wgpu::MapRangeError),
    /// The mapping callback never fired (lost sender — never in practice).
    #[error("GPU mapping callback lost before it fired")]
    CallbackLost,
}

/// Block until a `map_async(Read)` callback fires and return its outcome.
///
/// The callback result travels through a channel: the `poll(Wait)` below
/// drives it and `recv` proves it fired, replacing the old fire-and-forget
/// `|_| {}` closure that dropped device errors silently. A device mapping
/// failure becomes [`GpuDispatchError::MapFailed`].
fn await_map(device: &wgpu::Device, slice: &wgpu::BufferSlice) -> Result<(), GpuDispatchError> {
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();
    match rx.recv() {
        Ok(outcome) => outcome.map_err(GpuDispatchError::from),
        Err(_) => Err(GpuDispatchError::CallbackLost),
    }
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

    /// Pack a 6-vector `(linear, angular)` residual into body-state layout
    /// (the rung-2 base residual / solved delta). Pure (no device).
    pub fn from_residual(v: [f32; 6]) -> Self {
        Self {
            velocity: [v[0], v[1], v[2]],
            angular: [v[3], v[4], v[5]],
            _pad_v: 0.0,
            _pad_w: 0.0,
        }
    }

    /// Unpack body-state layout back into a 6-vector `(linear, angular)`.
    /// Pure (no device).
    pub fn to_residual(&self) -> [f32; 6] {
        [
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
            self.angular[0],
            self.angular[1],
            self.angular[2],
        ]
    }

    fn write_to_body(&self, b: &mut RigidBody) {
        b.velocity = Vec3::from_array(self.velocity);
        b.angular_velocity = Vec3::from_array(self.angular);
    }
}
// ---------------------------------------------------------------------------
// GPU sequential-impulse solver
// ---------------------------------------------------------------------------

/// GPU sequential-impulse solver for single-point manifold batches: the
/// same SI velocity iterations as the CPU wide path, run per wide batch.
///
/// Buffer lifecycle: `body_buf`, `batch_buf`, `uniform_buf` and
/// `readback_buf` are allocated once in [`new`](Self::new), sized by the
/// `max_bodies`/`max_batches` caps, and never recreated per dispatch —
/// uploads only `write_buffer` into them. (The `download_acc` staging
/// buffer is the one exception: a transient per-call `MAP_READ` buffer,
/// never bound, created and dropped inside the call.)
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
                        min_binding_size: wgpu::BufferSize::new(UNIFORM_ENTRY_BYTES),
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

        let body_size = max_bodies.next_power_of_two().max(MIN_BUFFER_CAP) as u64 * GPU_BODY_STRIDE;
        let batch_size =
            max_batches.next_power_of_two().max(MIN_BUFFER_CAP) as u64 * GPU_BATCH_STRIDE;

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
        let param_stride = UNIFORM_ENTRY_BYTES.next_multiple_of(align);
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
    ///
    /// Stages the first `max_bodies` entries (zeroed tail — the same bytes
    /// as the old `take(n)` path) and reports kept vs dropped counts in
    /// [`DispatchReport`]. The engine path ignores the report (its caps are
    /// sized by construction); direct users should check `dropped`.
    pub fn upload_bodies(&self, bodies: &[RigidBody]) -> DispatchReport {
        let report = DispatchReport::split(bodies.len(), self.max_bodies);
        let mut data = vec![GpuBodyState::zeroed(); self.max_bodies];
        for (i, b) in bodies.iter().enumerate().take(report.staged) {
            data[i] = GpuBodyState::from_body(b);
        }
        self.queue
            .write_buffer(&self.body_buf, 0, bytemuck::cast_slice(&data));
        report
    }

    /// Download body velocities from the GPU buffer (blocking).
    ///
    /// Writes back the first `max_bodies` entries and reports kept vs
    /// dropped counts in [`DispatchReport`]. The `map_async` outcome is
    /// awaited through a channel ([`await_map`]) instead of a
    /// fire-and-forget closure.
    ///
    /// # Errors
    ///
    /// [`GpuDispatchError`] when the device mapping or the mapped-range
    /// view fails. The numeric path is unchanged on success.
    pub fn try_download_bodies(
        &self,
        bodies: &mut [RigidBody],
    ) -> Result<DispatchReport, GpuDispatchError> {
        let report = DispatchReport::split(bodies.len(), self.max_bodies);
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
        await_map(&self.device, &slice)?;
        let mapped = slice.get_mapped_range()?;
        let states: &[GpuBodyState] = bytemuck::cast_slice(&mapped);
        for (i, b) in bodies.iter_mut().enumerate().take(report.staged) {
            states[i].write_to_body(b);
        }
        drop(mapped);
        self.readback_buf.unmap();
        Ok(report)
    }

    /// Download body velocities from the GPU buffer (blocking).
    ///
    /// Legacy wrapper over [`try_download_bodies`](Self::try_download_bodies):
    /// kept so the engine path needs no signature change. Panics on a
    /// device mapping failure — the same failure class the old code
    /// surfaced as a `get_mapped_range` panic (numeric path unchanged).
    pub fn download_bodies(&self, bodies: &mut [RigidBody]) {
        let _ = self.try_download_bodies(bodies);
    }

    /// Upload contact batches to the GPU buffer.
    ///
    /// Stages the first `max_batches` entries (zeroed tail — the same bytes
    /// as the old `take(n)` path) and reports kept vs dropped counts in
    /// [`DispatchReport`]; see [`upload_bodies`](Self::upload_bodies).
    pub fn upload_batches(&self, batches: &[GpuBatch]) -> DispatchReport {
        let report = DispatchReport::split(batches.len(), self.max_batches);
        let mut data = vec![GpuBatch::zeroed(); self.max_batches];
        for (i, b) in batches.iter().enumerate().take(report.staged) {
            data[i] = *b;
        }
        self.queue
            .write_buffer(&self.batch_buf, 0, bytemuck::cast_slice(&data));
        report
    }

    /// Download accumulated impulses back from the GPU batch buffer.
    ///
    /// Fallible path behind [`download_acc`](Self::download_acc): the
    /// `map_async` outcome is awaited through a channel ([`await_map`])
    /// instead of a fire-and-forget closure.
    ///
    /// # Errors
    ///
    /// [`GpuDispatchError`] when the device mapping or the mapped-range
    /// view fails. The numeric path is unchanged on success.
    pub fn try_download_acc(
        &self,
        batches: &mut [GpuBatch],
    ) -> Result<DispatchReport, GpuDispatchError> {
        let report = DispatchReport::split(batches.len(), self.max_batches);
        let copy_size = self.max_batches as u64 * GPU_BATCH_STRIDE;
        // Transient per-call staging buffer (never bound): created and
        // dropped inside the call — pre-existing lifecycle, unchanged.
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
        let slice = readback.slice(..);
        await_map(&self.device, &slice)?;
        let mapped = slice.get_mapped_range()?;
        let raw: &[u8] = &mapped;
        let gpu_entries: &[GpuBatch] = bytemuck::cast_slice(raw);
        for (i, b) in batches.iter_mut().enumerate().take(report.staged) {
            b.acc = gpu_entries[i].acc;
            b.acc_f1 = gpu_entries[i].acc_f1;
            b.acc_f2 = gpu_entries[i].acc_f2;
        }
        drop(mapped);
        readback.unmap();
        Ok(report)
    }

    /// Download accumulated impulses back from the GPU batch buffer.
    ///
    /// Legacy wrapper over [`try_download_acc`](Self::try_download_acc):
    /// kept so the engine path needs no signature change. Panics on a
    /// device mapping failure — the same failure class the old code
    /// surfaced as a `get_mapped_range` panic (numeric path unchanged).
    pub fn download_acc(&self, batches: &mut [GpuBatch]) {
        let _ = self.try_download_acc(batches);
    }

    /// Run the GPU contact solver for `iterations` GS iterations plus one
    /// restitution pass if `gate` is enabled (folded into the last
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
    pub fn solve(&self, num_batches: u32, iterations: u32, gate: crate::flags::RestitutionGate) {
        if u64::from(iterations) > PARAMS_CAP || iterations == 0 || num_batches == 0 {
            return;
        }
        // One upload for all passes (entries padded to the device stride).
        let mut blob = vec![0u8; iterations as usize * self.param_stride as usize];
        for (k, entry) in solve_params(iterations, gate).iter().enumerate() {
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
            let Ok(offset) = u32::try_from(u64::from(k) * self.param_stride) else {
                break;
            };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("physics_contact_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[offset]);
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

#[cfg(test)]
mod dispatch_report_tests {
    use super::DispatchReport;

    #[test]
    fn split_counts_truncate_at_the_cap() {
        assert_eq!(
            DispatchReport::split(0, 8),
            DispatchReport {
                staged: 0,
                dropped: 0
            }
        );
        assert_eq!(
            DispatchReport::split(5, 8),
            DispatchReport {
                staged: 5,
                dropped: 0
            }
        );
        assert_eq!(
            DispatchReport::split(8, 8),
            DispatchReport {
                staged: 8,
                dropped: 0
            }
        );
        assert_eq!(
            DispatchReport::split(11, 8),
            DispatchReport {
                staged: 8,
                dropped: 3
            }
        );
    }
}
