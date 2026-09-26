//! CPU/GPU dispatch decisions for smart-store workloads.
//!
//! [`Dispatcher`] compares an operation's element count against a
//! configurable threshold and returns an advisory [`ExecutionTarget`].
//! [`SmartDispatcher`] is a CPU-only executor built on that decision:
//! every read/write runs through [`CpuExecutor`].
//!
//! # Layer boundary: no Device/Queue in core (owner decision 2026-09-21)
//!
//! Execution on a GPU requires owning a `wgpu::Device`/`Queue`, which
//! `ornis-core` must not own. The former `GpuExecutor` STUB promised
//! execution the layering forbids — it always returned `None` — so it was
//! removed together with the GPU branch of `SmartDispatcher`,
//! `SmartDispatcher::set_gpu_executor`, and the `gpu` feature.
//! [`Dispatcher::decide`] stays: it is a pure threshold comparison with
//! no device dependency.
//!
//! Working GPU compute dispatch lives in `ornis-wgpu-backend`
//! (`CommandSync`, `AutoLane`, `GpuLanes`): `CommandSync` records
//! `wgpu::ComputePipeline` dispatches and CPU closures, then `flush()`
//! submits them to the `Device`/`Queue`; `AutoLane` resolves the CPU/GPU
//! verdict from the element count and drives `SmartBuffer` residency
//! itself (upload-if-dirty → dispatch → flush → download-if-dirty) with
//! CPU fallback; `GpuLanes` bridges `SmartStore` component lanes to
//! `AutoLane` with per-type slot reuse. The CPU sends commands to where
//! the data lives — no eager PCIe copies.
use crate::component_store::ComponentStore;
use crate::pipeline::PipelineConfig;
use crate::smart_store::SmartStore;

/// Result of runtime dispatch decision.
///
/// Advisory only: [`Dispatcher::decide`] performs a pure threshold
/// comparison. `Gpu` means "large enough that the GPU may pay off" —
/// interpreting and executing that verdict lives in `ornis-wgpu-backend`,
/// never in this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    /// Run on CPU threads.
    Cpu,
    /// Advisory: workload is large enough to consider the GPU.
    Gpu,
}

/// Runtime dispatcher that decides CPU vs GPU based on element count and threshold.
///
/// Pure decision, no execution and no device dependency: safe to keep in
/// `ornis-core`.
#[derive(Debug, Clone, Copy)]
pub struct Dispatcher {
    cpu_threshold: usize,
    gpu: GpuAvailability,
}

/// Whether a GPU is available for dispatch (typed replacement for
/// `gpu_available: bool`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum GpuAvailability {
    /// No GPU; always stay on CPU.
    #[default]
    Unavailable,
    /// GPU may be used above the threshold.
    Available,
}

impl GpuAvailability {
    /// `true` for [`GpuAvailability::Available`].
    pub fn is_available(self) -> bool {
        matches!(self, Self::Available)
    }
}

impl From<bool> for GpuAvailability {
    /// Legacy `gpu_available: bool` polarity.
    fn from(available: bool) -> Self {
        if available {
            Self::Available
        } else {
            Self::Unavailable
        }
    }
}

impl From<GpuAvailability> for bool {
    /// Legacy `gpu_available: bool` polarity.
    fn from(g: GpuAvailability) -> bool {
        g.is_available()
    }
}

/// High-level dispatcher that combines CPU and GPU execution
impl Dispatcher {
    /// Create a new dispatcher with a CPU threshold and GPU availability.
    pub fn with_availability(cpu_threshold: usize, gpu: GpuAvailability) -> Self {
        Self { cpu_threshold, gpu }
    }

    /// Boolean-compat constructor (kept for tests and call sites).
    pub fn new(cpu_threshold: usize, gpu_available: bool) -> Self {
        Self::with_availability(cpu_threshold, GpuAvailability::from(gpu_available))
    }

    /// Create from a PipelineConfig type (typed).
    pub fn from_config_with<T: PipelineConfig>(gpu: GpuAvailability) -> Self {
        Self::with_availability(T::THRESHOLD, gpu)
    }

    /// Boolean-compat config constructor (kept for tests and call sites).
    pub fn from_config<T: PipelineConfig>(gpu_available: bool) -> Self {
        Self::new(T::THRESHOLD, gpu_available)
    }

    /// Decide execution target based on element count
    pub fn decide(&self, element_count: usize) -> ExecutionTarget {
        if element_count >= self.cpu_threshold && self.gpu.is_available() {
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

/// High-level CPU-only dispatcher over [`Dispatcher`] decisions.
///
/// The dispatch verdict stays advisory and observable via [`dispatcher`](Self::dispatcher),
/// but execution always runs on [`CpuExecutor`]. Real GPU execution lives
/// in `ornis-wgpu-backend` (`CommandSync` / `AutoLane` / `GpuLanes`),
/// which owns the `Device`/`Queue` this crate must not own.
pub struct SmartDispatcher {
    dispatcher: Dispatcher,
}

impl SmartDispatcher {
    /// Create from PipelineConfig
    pub fn new<T: PipelineConfig>(gpu_available: bool) -> Self {
        Self {
            dispatcher: Dispatcher::from_config::<T>(gpu_available),
        }
    }

    /// Create with explicit threshold
    pub fn with_threshold(cpu_threshold: usize, gpu_available: bool) -> Self {
        Self {
            dispatcher: Dispatcher::new(cpu_threshold, gpu_available),
        }
    }

    /// Execute a read-only operation on the CPU.
    ///
    /// `element_count` is still resolved through [`Dispatcher::decide`] so
    /// the threshold wiring stays observable, but the verdict is advisory:
    /// execution always runs on [`CpuExecutor`]. Real GPU dispatch is
    /// `ornis-wgpu-backend::CommandSync`.
    pub fn execute_read<T, F, R>(&self, store: &SmartStore, element_count: usize, f: F) -> Option<R>
    where
        T: 'static + Send + Sync,
        F: FnOnce(&ComponentStore<T>) -> R + Send,
        R: Send,
    {
        let _ = self.dispatcher.decide(element_count);
        CpuExecutor::execute::<T, _, R>(store, f)
    }

    /// Execute a mutable operation on the CPU.
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
    fn smart_dispatcher_large_count_stays_on_cpu() {
        let mut store = SmartStore::new();
        let dispatcher = SmartDispatcher::with_threshold(1000, true);

        for i in 0..100 {
            let entity = store.create_entity();
            store.insert::<f32>(entity, i as f32);
        }

        // The verdict is advisory Gpu, but execution is CPU-only.
        assert_eq!(dispatcher.dispatcher().decide(10_000), ExecutionTarget::Gpu);
        let result = dispatcher.execute_read::<f32, _, _>(&store, 10_000, |lane| lane.len());
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
