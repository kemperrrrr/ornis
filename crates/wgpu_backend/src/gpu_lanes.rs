//! Typed lane bridge: `SmartStore` component lanes to automatic GPU execution.
//!
//! [`GpuLanes`] closes the last manual gap in the auto-GPU path. [`AutoLane`]
//! already routes by policy and drives residency, but the caller still had
//! to extract lane data, size buffers, and build pipelines by hand. Here the
//! caller only provides, per call, a pipeline builder and a CPU fallback —
//! everything else is automatic:
//!
//! 1. the lane is read from the [`SmartStore`](ornis_core::SmartStore) by
//!    component type (missing lane → `None`, no work);
//! 2. a per-type slot (resident buffers + pipeline + bind group) is created
//!    once and reused while the lane length is stable; a length change
//!    rebuilds it (and only then does `build` run);
//! 3. [`AutoLane::execute`](crate::auto_lane::AutoLane::execute) picks CPU
//!    or GPU by the [`DispatchConfig`](crate::dispatcher::DispatchConfig);
//! 4. the authoritative result is written back into the store lane, so the
//!    ECS side observes the computation whichever side ran.
//!
//! The two per-call copies (lane → slot, slot → lane) are the price of the
//! bridge: the GPU path still wins once the kernel outweighs them, which is
//! exactly what the threshold policy decides.

use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::auto_lane::AutoLane;
use crate::command_sync::CommandSync;
use crate::dispatcher::DispatchConfig;

/// Per-component slot: resident lane plus its pipeline and bind group.
struct TypedSlot<T: bytemuck::Pod + Send + Sync + 'static> {
    lane: AutoLane<T>,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

/// Cache of per-type GPU slots, keyed by component `TypeId`.
pub struct GpuLanes {
    slots: HashMap<TypeId, Box<dyn Any>>,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuLanes {
    /// Create an empty bridge on the given device/queue.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            slots: HashMap::new(),
            device: device.clone(),
            queue: queue.clone(),
        }
    }

    /// Number of cached per-type slots (test hook).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Execute the `T` lane from `store`, automatically on CPU or GPU.
    ///
    /// `build` constructs the compute pipeline and bind group for the
    /// lane's GPU buffer; it runs only when the slot is (re)created.
    /// `cpu_work` is the CPU fallback — it also runs whenever the policy
    /// picks CPU. Returns `None` when the store has no `T` lane, otherwise
    /// whether the execution completed: a GPU readback failure yields
    /// `Some(false)` and skips the write-back, preserving the store lane
    /// and the slot's dirty flag for retry.
    pub fn execute<T, B, C>(
        &mut self,
        store: &mut ornis_core::SmartStore,
        sync: &mut CommandSync,
        config: &DispatchConfig,
        build: B,
        cpu_work: C,
    ) -> Option<bool>
    where
        T: bytemuck::Pod + Send + Sync + 'static,
        B: FnOnce(&wgpu::Device, &wgpu::Buffer) -> (wgpu::ComputePipeline, wgpu::BindGroup),
        C: FnOnce(&mut [T]),
    {
        let data: Vec<T> = store.read_lane::<T>()?.data.clone();
        if data.is_empty() {
            return Some(true);
        }

        let stale = self
            .slots
            .get(&TypeId::of::<T>())
            .and_then(|s| s.downcast_ref::<TypedSlot<T>>())
            .is_none_or(|s| s.lane.len() != data.len());
        if stale {
            let lane = AutoLane::new(
                data,
                &self.device,
                &self.queue,
                config.clone(),
                "gpu_lanes slot",
            );
            let buf = lane.gpu_buffer().expect("slot owns its buffer");
            let (pipeline, bind_group) = build(&self.device, buf);
            self.slots.insert(
                TypeId::of::<T>(),
                Box::new(TypedSlot {
                    lane,
                    pipeline,
                    bind_group,
                }),
            );
        }
        let slot = self
            .slots
            .get_mut(&TypeId::of::<T>())
            .and_then(|s| s.downcast_mut::<TypedSlot<T>>())
            .expect("slot just built");

        let ok = slot
            .lane
            .execute(sync, Some(&slot.pipeline), Some(&slot.bind_group), cpu_work);
        if !ok {
            return Some(false);
        }

        let mut lane = store.write_lane::<T>()?;
        lane.data.copy_from_slice(slot.lane.data());
        Some(true)
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
            label: Some("gpu_lanes scale test"),
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
            label: "gpu_lanes test".to_string(),
        }
    }

    fn filled_store(n: usize) -> ornis_core::SmartStore {
        let mut store = ornis_core::SmartStore::new();
        for i in 0..n {
            let e = store.create_entity();
            store.insert(e, i as f32);
        }
        store
    }

    fn lane_sum(store: &ornis_core::SmartStore) -> f32 {
        store.read_lane::<f32>().unwrap().data.iter().sum()
    }

    #[test]
    fn cpu_verdict_writes_back_to_store() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut store = filled_store(8);
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lanes = GpuLanes::new(&device, &queue);
        let out = lanes.execute(
            &mut store,
            &mut sync,
            &lane_config(1_000_000),
            scale_pipeline,
            |data: &mut [f32]| {
                for x in data.iter_mut() {
                    *x *= 3.0;
                }
            },
        );
        assert_eq!(out, Some(true));
        // 0+..+7 = 28, times 3.
        assert_eq!(lane_sum(&store), 84.0);
        assert_eq!(lanes.slot_count(), 1);
    }

    #[test]
    fn gpu_verdict_writes_back_to_store() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut store = filled_store(64);
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lanes = GpuLanes::new(&device, &queue);
        let out = lanes.execute(
            &mut store,
            &mut sync,
            &lane_config(16),
            scale_pipeline,
            |data: &mut [f32]| {
                for x in data.iter_mut() {
                    *x *= 3.0;
                }
            },
        );
        assert_eq!(out, Some(true));
        // GPU kernel (x2) must win over the CPU closure (x3): 0+..+63 = 2016.
        assert_eq!(lane_sum(&store), 4032.0);
    }

    #[test]
    fn slot_reused_across_runs() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut store = filled_store(64);
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lanes = GpuLanes::new(&device, &queue);
        let cfg = lane_config(16);
        let run =
            |lanes: &mut GpuLanes, store: &mut ornis_core::SmartStore, sync: &mut CommandSync| {
                lanes.execute(store, sync, &cfg, scale_pipeline, |data: &mut [f32]| {
                    for x in data.iter_mut() {
                        *x *= 3.0;
                    }
                })
            };
        assert_eq!(run(&mut lanes, &mut store, &mut sync), Some(true));
        assert_eq!(lanes.slot_count(), 1);
        // Same length: slot reused, values double again on GPU.
        assert_eq!(run(&mut lanes, &mut store, &mut sync), Some(true));
        assert_eq!(lanes.slot_count(), 1);
        assert_eq!(lane_sum(&store), 4032.0 * 2.0);
    }

    #[test]
    fn lane_growth_rebuilds_slot() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        // CPU verdict throughout (avoids sizing a kernel per length); the
        // rebuild is verdict-independent.
        let mut store = filled_store(4);
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lanes = GpuLanes::new(&device, &queue);
        let cfg = lane_config(1_000_000);
        let run =
            |lanes: &mut GpuLanes, store: &mut ornis_core::SmartStore, sync: &mut CommandSync| {
                lanes.execute(store, sync, &cfg, scale_pipeline, |data: &mut [f32]| {
                    for x in data.iter_mut() {
                        *x += 1.0;
                    }
                })
            };
        assert_eq!(run(&mut lanes, &mut store, &mut sync), Some(true));
        assert_eq!(lane_sum(&store), 6.0 + 4.0);
        for _ in 0..4 {
            let e = store.create_entity();
            store.insert(e, 10.0f32);
        }
        assert_eq!(run(&mut lanes, &mut store, &mut sync), Some(true));
        assert_eq!(lanes.slot_count(), 1);
        // Old 4 grew by +1 each, new 4 went 10 -> 11.
        assert_eq!(lane_sum(&store), (6.0 + 4.0 * 2.0) + 44.0);
    }

    #[test]
    fn missing_lane_returns_none() {
        let Some((device, queue)) = pollster::block_on(try_device()) else {
            return;
        };
        let mut store = ornis_core::SmartStore::new();
        let mut sync = CommandSync::new(device.clone(), queue.clone());
        let mut lanes = GpuLanes::new(&device, &queue);
        let out: Option<bool> = lanes.execute::<f32, _, _>(
            &mut store,
            &mut sync,
            &lane_config(1),
            scale_pipeline,
            |_| panic!("must not run"),
        );
        assert_eq!(out, None);
        assert_eq!(lanes.slot_count(), 0);
    }
}
