//! Automatic ECS-lane execution with policy-driven residency.
//!
//! [`AutoLane`] is the piece the auto-GPU path was missing: it owns a
//! [`SmartBuffer`](crate::smart_buffer::SmartBuffer) plus a
//! [`DispatchConfig`](crate::dispatcher::DispatchConfig), and
//! [`execute`](AutoLane::execute) resolves the [`Platform`](crate::dispatcher::Platform)
//! from the element count, then drives residency itself — the caller never
//! touches dirty flags or sync calls:
//!
//! * **CPU verdict** — the closure runs on the CPU copy (the GPU side goes
//!   stale, which is correct: the CPU just mutated the data).
//! * **GPU verdict with pipeline + bind group** — uploads only if the CPU
//!   side is dirty, records the dispatch into the [`CommandSync`](crate::command_sync::CommandSync),
//!   flushes, then downloads only if the GPU side is dirty. The CPU copy
//!   ends authoritative either way.
//! * **GPU verdict without pipeline/bind group** — falls back to the CPU
//!   closure instead of dropping the work.
//!
//! This is the "auto" that [`SmartBuffer`](crate::smart_buffer::SmartBuffer)
//! alone does not provide (its residency decisions are manual) and that
//! `ornis-core`'s `GpuExecutor` stub never implemented (it always returns
//! `None`). Lane types still opt in at compile time through
//! [`LaneTarget`](ornis_core::LaneTarget); this module adds the runtime half.

use crate::command_sync::CommandSync;
use crate::dispatcher::{DispatchConfig, Platform, choose_platform};
use crate::smart_buffer::SmartBuffer;

/// An ECS lane buffer that executes on CPU or GPU by policy, keeping both
/// residency sides coherent without caller intervention.
pub struct AutoLane<T: bytemuck::Pod> {
    buf: SmartBuffer<T>,
    config: DispatchConfig,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl<T: bytemuck::Pod> AutoLane<T> {
    /// Wrap CPU-side `data`, uploading it once to a storage buffer usable as
    /// both `COPY_SRC` and `COPY_DST` so either verdict can proceed.
    pub fn new(
        data: Vec<T>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: DispatchConfig,
        label: &str,
    ) -> Self {
        let buf = SmartBuffer::new(
            data,
            device,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            label,
        );
        Self {
            buf,
            config,
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Which [`Platform`] this lane resolves to for `element_count` elements.
    pub fn platform_for(&self, element_count: usize) -> Platform {
        choose_platform(&self.config, element_count)
    }

    /// Authoritative CPU copy. Valid after [`execute`](Self::execute)
    /// regardless of which side ran — a GPU run downloads before returning.
    pub fn data(&self) -> &[T] {
        self.buf.cpu_data()
    }

    /// Element count of the lane.
    pub fn len(&self) -> usize {
        self.buf.cpu_data().len()
    }

    /// Whether the lane holds no elements.
    pub fn is_empty(&self) -> bool {
        self.buf.cpu_data().is_empty()
    }

    /// GPU buffer backing this lane, for bind-group construction.
    pub fn gpu_buffer(&self) -> Option<&wgpu::Buffer> {
        self.buf.gpu_buffer()
    }

    /// Run the lane: GPU dispatch or CPU closure, with residency handled.
    ///
    /// Returns `true` when the selected work completed and the CPU copy is
    /// authoritative. A GPU readback failure returns `false` and preserves
    /// the dirty flag so the caller can retry or report the device error.
    ///
    /// On success the CPU copy is authoritative and no dirty flags remain
    /// (a GPU run uploads first when the CPU side changed and downloads
    /// after; a CPU run leaves the GPU side stale by design — it will
    /// re-upload on the next GPU verdict).
    pub fn execute(
        &mut self,
        sync: &mut CommandSync,
        pipeline: Option<&wgpu::ComputePipeline>,
        bind_group: Option<&wgpu::BindGroup>,
        cpu_work: impl FnOnce(&mut [T]),
    ) -> bool {
        let n = self.buf.cpu_data().len();
        if n == 0 {
            return true;
        }
        let gpu_ready = matches!(self.platform_for(n), Platform::Gpu)
            && pipeline.is_some()
            && bind_group.is_some();
        if !gpu_ready {
            cpu_work(self.buf.cpu_data_mut());
            return true;
        }
        self.buf.sync_to_gpu(&self.queue);
        let wgc = (n as u32).div_ceil(self.config.workgroup_size);
        sync.dispatch_gpu(
            pipeline.expect("checked above"),
            bind_group.expect("checked above"),
            (wgc, 1, 1),
            &self.config.label,
        );
        sync.flush();
        self.buf.mark_gpu_dirty();
        self.buf.sync_to_cpu_blocking(&self.device, &self.queue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[ornis_macros::gpu_pipeline(
        workgroup_size = 64,
        storage(buf: [f32; 64], read_write),
        builtin(gid: global_invocation_id),
    )]
    fn scale_lane() {
        buf[gid.x] = buf[gid.x] * 2.0;
    }

    /// Adapter if the machine has one; `None` (skip the test) otherwise.
    /// Unlike [`WgpuContext::new_blocking`](crate::context::WgpuContext::new_blocking),
    /// this never panics, so GPU-path tests degrade to no-ops on headless CI.
    async fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .ok()
    }

    fn scale_pipeline(
        device: &wgpu::Device,
        buf: &wgpu::Buffer,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroup) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("auto_lane scale test"),
            source: wgpu::ShaderSource::Wgsl(scale_lane::wgsl_source().into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        (pipeline, bg)
    }

    fn lane_config(threshold: usize) -> DispatchConfig {
        DispatchConfig {
            target: crate::dispatcher::ExecutionTarget::Auto(threshold),
            workgroup_size: 64,
            label: "auto_lane test".to_string(),
        }
    }

    #[test]
    fn cpu_verdict_runs_closure_and_keeps_result() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        // Threshold far above the lane: CPU must win.
        let mut lane = AutoLane::new(
            vec![1.0f32, 2.0, 3.0, 4.0],
            &device,
            &queue,
            lane_config(1_000_000),
            "cpu test",
        );
        assert_eq!(lane.platform_for(4), Platform::Cpu);
        lane.execute(&mut sync, None, None, |data| {
            for x in data.iter_mut() {
                *x *= 3.0;
            }
        });
        assert_eq!(lane.data(), &[3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn gpu_verdict_dispatches_and_downloads() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        // Threshold below the lane: GPU must win (64 elements, threshold 16).
        let mut lane = AutoLane::new(
            vec![1.0f32; 64],
            &device,
            &queue,
            lane_config(16),
            "gpu test",
        );
        assert_eq!(lane.platform_for(64), Platform::Gpu);
        let (pipeline, bg) = scale_pipeline(&device, lane.gpu_buffer().expect("buffer exists"));
        // If this ran on CPU it would multiply by 3; GPU multiplies by 2.
        lane.execute(&mut sync, Some(&pipeline), Some(&bg), |data| {
            for x in data.iter_mut() {
                *x *= 3.0;
            }
        });
        assert_eq!(lane.data(), &vec![2.0f32; 64]);
    }

    #[test]
    fn gpu_verdict_without_pipeline_falls_back_to_cpu() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lane = AutoLane::new(
            vec![5.0f32; 64],
            &device,
            &queue,
            lane_config(16),
            "fallback test",
        );
        assert_eq!(lane.platform_for(64), Platform::Gpu);
        lane.execute(&mut sync, None, None, |data| {
            for x in data.iter_mut() {
                *x += 1.0;
            }
        });
        assert_eq!(lane.data(), &vec![6.0f32; 64]);
    }

    #[test]
    fn empty_lane_executes_nothing() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lane = AutoLane::new(Vec::<f32>::new(), &device, &queue, lane_config(1), "empty");
        lane.execute(&mut sync, None, None, |_| panic!("must not run"));
        assert!(sync.is_empty());
    }
}
