//! CPU/GPU dispatch decisions for smart-store workloads.
//!
//! [`Dispatcher`] (and the higher-level [`SmartDispatcher`]) compare an
//! operation's element count against a configurable threshold and route it
//! to CPU or GPU executors, falling back to CPU whenever no GPU executor is
//! wired up or the `gpu` feature is off.
//!
//! **Status:** the GPU route is a reserved extension point — `GpuExecutor`
//! is an experimental stub that performs no GPU work, so every dispatch
//! effectively executes on CPU today. Working GPU compute dispatch lives in
//! `ornis-wgpu-backend` (`CommandSync`).
use crate::component_store::ComponentStore;
use crate::pipeline::PipelineConfig;
use crate::smart_store::SmartStore;

/// Result of runtime dispatch decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    /// Run on CPU threads.
    Cpu,
    /// Run on the GPU.
    Gpu,
}

/// Runtime dispatcher that decides CPU vs GPU based on element count and threshold
#[derive(Debug, Clone, Copy)]
pub struct Dispatcher {
    cpu_threshold: usize,
    gpu_available: bool,
}

/// High-level dispatcher that combines CPU and GPU execution
impl Dispatcher {
    /// Create a new dispatcher with a CPU threshold and GPU availability
    pub fn new(cpu_threshold: usize, gpu_available: bool) -> Self {
        Self {
            cpu_threshold,
            gpu_available,
        }
    }

    /// Create from a PipelineConfig type
    pub fn from_config<T: PipelineConfig>(gpu_available: bool) -> Self {
        Self::new(T::THRESHOLD, gpu_available)
    }

    /// Decide execution target based on element count
    pub fn decide(&self, element_count: usize) -> ExecutionTarget {
        if element_count >= self.cpu_threshold && self.gpu_available {
            ExecutionTarget::Gpu
        } else {
            ExecutionTarget::Cpu
        }
    }

    /// Get the CPU threshold
    pub fn threshold(&self) -> usize {
        self.cpu_threshold
    }
}

/// CPU executor for running operations on component lanes
pub struct CpuExecutor;

impl CpuExecutor {
    /// Execute a read-only operation on a component lane
    pub fn execute<T, F, R>(store: &SmartStore, f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&ComponentStore<T>) -> R + Send,
        R: Send,
    {
        let lane = store.read_lane::<T>()?;
        Some(f(&lane))
    }

    /// Execute a mutable operation on a component lane
    pub fn execute_mut<T, F, R>(store: &SmartStore, f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&mut ComponentStore<T>) -> R + Send,
        R: Send,
    {
        let mut lane = store.write_lane::<T>()?;
        Some(f(&mut lane))
    }

    /// Execute a parallel read operation
    pub fn execute_par<T, F, R>(store: &SmartStore, f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&ComponentStore<T>) -> R + Send,
        R: Send,
    {
        let lane = store.read_lane::<T>()?;
        Some(f(&lane))
    }
}

/// GPU executor (requires gpu feature).
///
/// **Experimental — not production-ready.** This is a reserved extension
/// point: `execute` is a stub that performs no GPU work (and the `gpu`
/// feature of `ornis-core` does not compile against wgpu yet). Real GPU
/// compute lives in `ornis-wgpu-backend` (`CommandSync`); treat this type
/// as a placeholder for the future automatic ECS→GPU dispatch, not as a
/// working executor.
#[cfg(feature = "gpu")]
pub struct GpuExecutor {
    // GPU device and queue would be stored here
}

#[cfg(feature = "gpu")]
impl GpuExecutor {
    /// Create a new GPU executor
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {}
    }

    /// Execute a compute shader on a component lane.
    ///
    /// Stub: the `gpu` feature is a reserved extension point and ornis-core
    /// does not depend on wgpu yet, so this code path cannot even compile
    /// (let alone be exercised) — mutants here are untestable, skip.
    #[mutants::skip]
    pub fn execute<T, F, R>(&self, _store: &SmartStore, _f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&ComponentStore<T>) -> R + Send,
        R: Send,
    {
        // Would compile shader, create buffers, dispatch compute
        None
    }
}

/// High-level smart dispatcher that combines CPU and GPU execution.
///
/// Note: the GPU side is currently a stub — see `GpuExecutor` (behind the
/// `gpu` feature). When the dispatcher picks [`ExecutionTarget::Gpu`] the
/// work silently falls back to the CPU executor, so `SmartDispatcher` is
/// effectively CPU-only today.
pub struct SmartDispatcher {
    dispatcher: Dispatcher,
    #[cfg(feature = "gpu")]
    gpu_executor: Option<GpuExecutor>,
}

impl SmartDispatcher {
    /// Create from PipelineConfig
    pub fn new<T: PipelineConfig>(gpu_available: bool) -> Self {
        Self {
            dispatcher: Dispatcher::from_config::<T>(gpu_available),
            #[cfg(feature = "gpu")]
            gpu_executor: None,
        }
    }

    /// Create with explicit threshold
    pub fn with_threshold(cpu_threshold: usize, gpu_available: bool) -> Self {
        Self {
            dispatcher: Dispatcher::new(cpu_threshold, gpu_available),
            #[cfg(feature = "gpu")]
            gpu_executor: None,
        }
    }

    /// Set GPU executor (requires gpu feature).
    /// Stub only: `gpu_executor` is never read without the (uncompilable)
    /// `gpu` feature, so a `with ()` mutant is unobservable — skip.
    #[cfg(feature = "gpu")]
    #[mutants::skip]
    pub fn set_gpu_executor(&mut self, executor: GpuExecutor) {
        self.gpu_executor = Some(executor);
    }

    /// Execute a read-only operation, automatically choosing CPU/GPU
    pub fn execute_read<T, F, R>(&self, store: &SmartStore, element_count: usize, f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&ComponentStore<T>) -> R + Send,
        R: Send,
    {
        let target = self.dispatcher.decide(element_count);

        match target {
            ExecutionTarget::Cpu => CpuExecutor::execute::<T, _, R>(store, f),
            ExecutionTarget::Gpu => {
                #[cfg(feature = "gpu")]
                if let Some(ref gpu) = self.gpu_executor {
                    gpu.execute(store, f)
                } else {
                    // Fallback to CPU if GPU not set up
                    CpuExecutor::execute::<T, _, R>(store, f)
                }
                #[cfg(not(feature = "gpu"))]
                CpuExecutor::execute::<T, _, R>(store, f)
            }
        }
    }

    /// Execute a mutable operation (CPU only for now)
    pub fn execute_mut<T, F, R>(&self, store: &SmartStore, element_count: usize, f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&mut ComponentStore<T>) -> R + Send,
        R: Send,
    {
        let _target = self.dispatcher.decide(element_count);
        CpuExecutor::execute_mut::<T, _, R>(store, f)
    }

    /// Get the underlying dispatcher for manual decisions
    pub fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
    }
}

/// Trait for types that can be dispatched to CPU or GPU
/// Types whose workload size can be measured for dispatch decisions.
pub trait Dispatchable: 'static + Send + Sync {
    /// Returns how many elements of this type are currently live in `store`.
    fn element_count(&self, store: &SmartStore) -> usize;
}

impl<T: 'static + Send + Sync> Dispatchable for T {
    fn element_count(&self, store: &SmartStore) -> usize {
        store.read_lane::<T>().map(|lane| lane.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SmartStore;

    #[test]
    fn dispatcher_decides_cpu_for_small_data() {
        let dispatcher = Dispatcher::new(1000, true);
        assert_eq!(dispatcher.decide(100), ExecutionTarget::Cpu);
        assert_eq!(dispatcher.decide(500), ExecutionTarget::Cpu);
    }

    #[test]
    fn dispatcher_decides_gpu_for_large_data() {
        let dispatcher = Dispatcher::new(1000, true);
        assert_eq!(dispatcher.decide(1000), ExecutionTarget::Gpu);
        assert_eq!(dispatcher.decide(10000), ExecutionTarget::Gpu);
    }

    #[test]
    fn dispatcher_no_gpu_fallback() {
        let dispatcher = Dispatcher::new(1000, false);
        assert_eq!(dispatcher.decide(10000), ExecutionTarget::Cpu);
    }

    #[test]
    fn smart_dispatcher_cpu_execution() {
        let mut store = SmartStore::new();
        let dispatcher = SmartDispatcher::with_threshold(1000, false);

        for i in 0..100 {
            let entity = store.create_entity();
            store.insert::<f32>(entity, i as f32);
        }

        let result = dispatcher.execute_read::<f32, _, _>(&store, 100, |lane| lane.len());
        assert_eq!(result, Some(100));
    }

    #[test]
    fn cpu_executor_read() {
        let mut store = SmartStore::new();
        for i in 0..10 {
            let e = store.create_entity();
            store.insert(e, i as f32);
        }
        let sum = CpuExecutor::execute::<f32, _, _>(&store, |lane| lane.iter().sum::<f32>());
        assert_eq!(sum, Some((0..10).map(|i| i as f32).sum()));
    }

    #[test]
    fn cpu_executor_mut() {
        let mut store = SmartStore::new();
        for i in 0..10 {
            let e = store.create_entity();
            store.insert(e, i as f32);
        }
        let sum = CpuExecutor::execute_mut::<f32, _, _>(&store, |lane| {
            let mut s = 0.0;
            for v in lane.iter_mut() {
                s += *v;
                *v *= 2.0;
            }
            s
        });
        assert_eq!(sum, Some((0..10).map(|i| i as f32).sum()));
    }

    #[test]
    fn dispatcher_threshold_reported() {
        let dispatcher = Dispatcher::new(500, true);
        assert_eq!(dispatcher.threshold(), 500);
    }

    #[test]
    fn cpu_executor_execute_par_reads_lane() {
        let mut store = SmartStore::new();
        for i in 0..10 {
            let e = store.create_entity();
            store.insert(e, i as f32);
        }
        let sum = CpuExecutor::execute_par::<f32, _, _>(&store, |lane| lane.iter().sum::<f32>());
        assert_eq!(sum, Some((0..10).map(|i| i as f32).sum()));
    }

    #[test]
    fn smart_dispatcher_execute_mut_writes() {
        let mut store = SmartStore::new();
        let dispatcher = SmartDispatcher::with_threshold(1000, false);

        for i in 0..10 {
            let e = store.create_entity();
            store.insert(e, i as f32);
        }

        let result = dispatcher.execute_mut::<f32, _, _>(&store, 100, |lane| {
            let mut s = 0.0;
            for v in lane.iter_mut() {
                s += *v;
                *v += 100.0;
            }
            s
        });
        assert_eq!(result, Some((0..10).map(|i| i as f32).sum()));

        // The mutation must be visible afterwards.
        let back =
            dispatcher.execute_read::<f32, _, _>(&store, 100, |lane| lane.iter().sum::<f32>());
        assert_eq!(back, Some((0..10).map(|i| i as f32 + 100.0).sum()));
    }

    #[test]
    fn dispatchable_element_count() {
        let mut store = SmartStore::new();
        for i in 0..7 {
            let e = store.create_entity();
            store.insert(e, i as f32);
        }
        assert_eq!(0f32.element_count(&store), 7);

        // Unregistered type yields zero.
        assert_eq!(0u64.element_count(&store), 0);
    }
}
