//! Owned GPU buffers and the dispatch entry point of the culling pass
//! ([`Culler`]); the parent module holds the shared layout and the CPU
//! reference.

use std::sync::PoisonError;

use glam::Mat4;

use super::CULL_COMPUTE_WGSL;
use super::CULL_PARAMS_BYTE_SIZE;
use super::DirtyRange;
use super::DrawIndexedIndirectArgs;
use super::INDIRECT_ARGS_WORD_BYTES;
use super::INDIRECT_ARGS_WORD_COUNT;
use super::cull_params;
use super::cull_workgroup_count;
use super::diff_staged_records;
use super::stage_next_instances;
use crate::renderer::InstanceData;
use crate::renderer::PerObjectGpu;

/// Binding slot of the `CullParams` uniform in [`CULL_COMPUTE_WGSL`].
const PARAMS_BINDING: u32 = 0;
/// Binding slot of the source `PerObject` array.
const SRC_BINDING: u32 = 1;
/// Binding slot of the dense destination array.
const DST_BINDING: u32 = 2;
/// Binding slot of the indirect-args words.
const ARGS_BINDING: u32 = 3;

/// Errors from [`Culler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CullerError {
    /// A culler was requested with zero instance capacity.
    #[error("culler capacity must be at least one instance")]
    EmptyCapacity,
    /// One frame staged more instances than the construction capacity.
    #[error("instance count {count} exceeds culler capacity {capacity}")]
    CapacityExceeded {
        /// Instances staged for the frame.
        count: u32,
        /// Per-frame capacity fixed at construction.
        capacity: u32,
    },
}

/// Owned GPU resources for one culling dispatch: source staging, the
/// dense compacted destination, the indirect-args buffer and the
/// compute pipeline of [`CULL_COMPUTE_WGSL`].
///
/// [`Self::compact`] stages one frame of instances and pre-writes the
/// indirect args with a zero counter; [`Self::record_compact`] records
/// the compaction dispatch into the caller's encoder (no extra submit),
/// leaving the result in [`Self::visible_buffer`] (dense
/// [`PerObjectGpu`](crate::renderer::PerObjectGpu) records) and
/// [`Self::args_buffer`] (a
/// [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)-ready
/// [`DrawIndexedIndirectArgs`] carrying the caller's `index_count`).
/// The gbuffer pass renders through it when armed (see
/// `Renderer3D::prepare_indirect_batch`); shadows always draw directly.
pub struct Culler {
    /// Dense compacted destination: `PerObject` records of the visible
    /// instances, in dispatch order — the buffer the future indirect
    /// draw binds as `per_objects`.
    pub visible_buffer: wgpu::Buffer,
    /// The indirect-args buffer: after [`Self::compact`] it is ready
    /// for
    /// [`draw_indexed_indirect`](wgpu::RenderPass::draw_indexed_indirect)
    /// at offset `0`.
    pub args_buffer: wgpu::Buffer,
    /// Per-frame instance capacity fixed at construction.
    pub capacity: u32,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    src_buffer: wgpu::Buffer,
    /// Resident CPU-side staging: `prev` is the last uploaded frame,
    /// `next` is reused across frames (no per-frame allocation), so a
    /// static scene re-uploads nothing — only the dirty runs tracked
    /// by the fused convert-and-diff pass are written.
    staging: std::sync::Mutex<CullerStaging>,
    /// Instances staged by the last [`Culler::compact`] (drives the
    /// [`Culler::record_compact`] dispatch width).
    staged_count: std::sync::atomic::AtomicU32,
}

/// Double-buffered reusable staging of [`Culler`]: both vectors keep
/// their allocation across frames and swap roles every [`Culler::compact`].
#[derive(Debug, Default)]
struct CullerStaging {
    /// Last uploaded frame (diff baseline).
    prev: Vec<PerObjectGpu>,
    /// Frame under construction (reused, never reallocated per frame).
    next: Vec<PerObjectGpu>,
}

/// One buffer binding of [`CULL_COMPUTE_WGSL`].
fn cull_bind_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One ordinary (non-mappable) GPU buffer of the culler.
fn cull_storage_buffer(
    device: &wgpu::Device,
    label: &str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

/// Src/dst/args/params buffers for a [`Culler`] of `capacity` instances.
fn culler_buffers(
    device: &wgpu::Device,
    capacity: u32,
) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let record_bytes = std::mem::size_of::<PerObjectGpu>() as u64 * u64::from(capacity);
    let src_buffer = cull_storage_buffer(
        device,
        "culler src",
        record_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let dst_buffer = cull_storage_buffer(
        device,
        "culler dst",
        record_bytes,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let args_buffer = cull_storage_buffer(
        device,
        "culler args",
        INDIRECT_ARGS_WORD_COUNT as u64 * INDIRECT_ARGS_WORD_BYTES,
        wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    );
    let params_buffer = cull_storage_buffer(
        device,
        "culler params",
        CULL_PARAMS_BYTE_SIZE,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    (src_buffer, dst_buffer, args_buffer, params_buffer)
}

impl Culler {
    /// Builds a culler with fixed room for `capacity` instances per
    /// frame (the src/dst buffers are sized once, here).
    ///
    /// # Errors
    ///
    /// Returns [`CullerError::EmptyCapacity`] when `capacity` is zero.
    pub fn try_new(device: &wgpu::Device, capacity: u32) -> Result<Self, CullerError> {
        if capacity == 0 {
            return Err(CullerError::EmptyCapacity);
        }
        let (src_buffer, dst_buffer, args_buffer, params_buffer) = culler_buffers(device, capacity);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("culler shader"),
            source: wgpu::ShaderSource::Wgsl(CULL_COMPUTE_WGSL.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("culler bind group layout"),
            entries: &[
                cull_bind_entry(PARAMS_BINDING, wgpu::BufferBindingType::Uniform),
                cull_bind_entry(
                    SRC_BINDING,
                    wgpu::BufferBindingType::Storage { read_only: true },
                ),
                cull_bind_entry(
                    DST_BINDING,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
                cull_bind_entry(
                    ARGS_BINDING,
                    wgpu::BufferBindingType::Storage { read_only: false },
                ),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("culler pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("culler pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cull_instances"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("culler bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: PARAMS_BINDING,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: SRC_BINDING,
                    resource: src_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: DST_BINDING,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: ARGS_BINDING,
                    resource: args_buffer.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            visible_buffer: dst_buffer,
            args_buffer,
            capacity,
            pipeline,
            bind_group,
            params_buffer,
            src_buffer,
            staging: std::sync::Mutex::new(CullerStaging::default()),
            staged_count: std::sync::atomic::AtomicU32::new(0),
        })
    }

    /// One write of the resident staging's dirty runs into the source
    /// buffer — the single dirty-fill implementation
    /// [`Culler::compact`], [`Culler::compact_records`] and the profile
    /// hook all share (one copy of the offset/size/write loop).
    fn write_dirty(&self, queue: &wgpu::Queue, records: &[PerObjectGpu], ranges: &[DirtyRange]) {
        for range in ranges {
            let start = range.start as usize;
            queue.write_buffer(
                &self.src_buffer,
                start as u64 * std::mem::size_of::<PerObjectGpu>() as u64,
                bytemuck::cast_slice(&records[start..start + range.count as usize]),
            );
        }
    }

    /// One staging transition under the resident lock: `build` fills the
    /// frame's records and returns their dirty runs against the previous
    /// frame, those runs are written into the source buffer, and the
    /// staging roles swap. [`Culler::compact`] and
    /// [`Culler::compact_records`] differ only in `build`, so the diff
    /// and the dirty fill each exist exactly once.
    fn stage_frame(
        &self,
        queue: &wgpu::Queue,
        build: impl FnOnce(&[PerObjectGpu], &mut Vec<PerObjectGpu>) -> Vec<DirtyRange>,
    ) {
        let mut staging = self.staging.lock().unwrap_or_else(PoisonError::into_inner);
        let CullerStaging { prev, next } = &mut *staging;
        let ranges = build(prev, next);
        self.write_dirty(queue, next, &ranges);
        std::mem::swap(prev, next);
    }

    /// Stages one culling frame: converts `instances` into records,
    /// diffs them against the previously staged frame to track dirty
    /// runs (a static scene uploads nothing), writes only those runs
    /// into the resident source buffer, packs the
    /// [`CullParams`](super::CullParams) uniform of `view_proj`, and
    /// pre-writes the indirect args with `index_count` and a zero
    /// instance counter (consecutive frames never accumulate). Uploads
    /// only — the compaction dispatch is recorded separately by
    /// [`Culler::record_compact`] into the caller's encoder, so the
    /// frame needs no extra submit.
    ///
    /// # Errors
    ///
    /// Returns [`CullerError::CapacityExceeded`] when
    /// `instances.len()` exceeds [`Self::capacity`].
    pub fn compact(
        &self,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        index_count: u32,
        instances: &[InstanceData],
    ) -> Result<(), CullerError> {
        if instances.len() > self.capacity as usize {
            return Err(CullerError::CapacityExceeded {
                count: instances.len() as u32,
                capacity: self.capacity,
            });
        }
        self.stage_frame(queue, |prev, next| {
            stage_next_instances(prev, next, instances)
        });
        self.prewrite_frame(queue, view_proj, index_count, instances.len() as u32);
        Ok(())
    }

    // qual:allow(test_quality, untested) reason: "test-util profiling hook reached only by the criterion culling bench (a separate crate), never under libtest"
    /// Test-only profiling hook: the per-frame uniform and indirect-args
    /// pre-write of [`Culler::prewrite_frame`], timed on its own by the
    /// culling bench (the production entry points still call the private
    /// method inline).
    #[cfg(feature = "test-util")]
    #[doc(hidden)]
    pub fn profile_prewrite(
        &self,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        index_count: u32,
        count: u32,
    ) {
        self.prewrite_frame(queue, view_proj, index_count, count);
    }

    // qual:allow(test_quality, untested) reason: "test-util profiling hook reached only by the criterion culling bench (a separate crate), never under libtest"
    /// Test-only profiling hook: the dirty-fill writes of one frame over
    /// bench-owned records and `(start, count)` runs — the same
    /// [`Culler::write_dirty`] the staging entry points use, so the
    /// bench times production write code against its own diff history.
    #[cfg(feature = "test-util")]
    #[doc(hidden)]
    pub fn profile_write_dirty(
        &self,
        queue: &wgpu::Queue,
        records: &[PerObjectGpu],
        ranges: &[(u32, u32)],
    ) {
        let runs: Vec<DirtyRange> = ranges
            .iter()
            .map(|&(start, count)| DirtyRange { start, count })
            .collect();
        self.write_dirty(queue, records, &runs);
    }

    /// Pre-writes the per-frame uniform, the indirect args (with a zero
    /// instance counter, so consecutive frames never accumulate) and the
    /// staged count shared by [`Culler::compact`] and
    /// [`Culler::compact_records`].
    fn prewrite_frame(&self, queue: &wgpu::Queue, view_proj: &Mat4, index_count: u32, count: u32) {
        let params = cull_params(view_proj, count);
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
        let args = DrawIndexedIndirectArgs::compacted(index_count, 0);
        queue.write_buffer(&self.args_buffer, 0, bytemuck::bytes_of(&args));
        self.staged_count
            .store(count, std::sync::atomic::Ordering::Relaxed);
    }

    /// Stages one culling frame from caller-converted `records`: diffs
    /// them against the last uploaded frame and writes only the dirty
    /// runs — like [`Culler::compact`], but without converting. The
    /// submit path converts once and shares the records between the
    /// direct upload and this staging, so a frame converts exactly once.
    ///
    /// # Errors
    ///
    /// Returns [`CullerError::CapacityExceeded`] when `records.len()`
    /// exceeds [`Self::capacity`].
    pub fn compact_records(
        &self,
        queue: &wgpu::Queue,
        view_proj: &Mat4,
        index_count: u32,
        records: &[PerObjectGpu],
    ) -> Result<(), CullerError> {
        if records.len() > self.capacity as usize {
            let count = records.len() as u32;
            return Err(CullerError::CapacityExceeded {
                count,
                capacity: self.capacity,
            });
        }
        self.stage_frame(queue, |prev, next| {
            next.clear();
            next.extend_from_slice(records);
            diff_staged_records(prev, next)
        });
        self.prewrite_frame(queue, view_proj, index_count, records.len() as u32);
        Ok(())
    }

    /// Records the compaction dispatch into the caller's encoder (no
    /// submit — the frame submits once, downstream): one compute pass
    /// over the [`Culler::compact`] staging, compacting the visible
    /// instances into [`Culler::visible_buffer`] and counting them into
    /// [`Culler::args_buffer`]. The `queue.write_buffer` uploads of
    /// `compact` are ordered before any later submit, so recording here
    /// always sees the staged frame.
    pub fn record_compact(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_pass(encoder, None);
    }

    // qual:allow(test_quality, untested) reason: "test-util profiling hook reached only by the criterion culling bench (a separate crate), never under libtest"
    /// Test-only profiling hook: [`Culler::record_compact`] with pass
    /// timestamp writes, so the bench measures the dispatch inside the
    /// caller's encoder exactly where the production frame records it.
    #[cfg(feature = "test-util")]
    #[doc(hidden)]
    pub fn record_compact_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: wgpu::ComputePassTimestampWrites<'_>,
    ) {
        self.record_pass(encoder, Some(timestamp_writes));
    }

    /// The one compaction pass: one compute dispatch over the staged
    /// frame, with optional timestamp writes at the pass boundary
    /// (`None` for the production entry point).
    fn record_pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("culler pass"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let staged = self.staged_count.load(std::sync::atomic::Ordering::Relaxed);
        pass.dispatch_workgroups(cull_workgroup_count(staged), 1, 1);
    }
}
