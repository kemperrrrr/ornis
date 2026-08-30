//! Zero-cost lane executors bridging `ornis_core` routing and recording.
//!
//! [`ExecuteLane`] is implemented by the [`CpuExecutor`] / [`GpuExecutor`]
//! ZSTs: the CPU lane records a closure, the GPU lane a compute dispatch
//! sized from the element count. Resolution through [`LaneTarget`]
//! monomorphizes away any runtime branch.

use crate::command_sync::CommandSync;
use crate::router::PipelineRouter;
use ornis_core::{LaneTarget, Route};

/// CPU executor ZST — monomorphizes to sequential/rayon dispatch.
pub struct CpuExecutor;

/// GPU executor ZST — monomorphizes to wgpu compute dispatch.
pub struct GpuExecutor;

/// Trait with zero-cost dispatch via ZSTs.
pub trait ExecuteLane {
    /// Records the workload into `sync` on this executor's lane.
    ///
    /// The GPU lane submits `pipeline`/`bind_group` as a compute dispatch
    /// sized from `element_count`; the CPU lane runs `cpu_work` instead.
    fn execute(
        &self,
        sync: &mut CommandSync,
        element_count: usize,
        pipeline: Option<&wgpu::ComputePipeline>,
        bind_group: Option<&wgpu::BindGroup>,
        cpu_work: Box<dyn FnOnce() + Send>,
        label: &str,
    );
}

impl ExecuteLane for CpuExecutor {
    fn execute(
        &self,
        sync: &mut CommandSync,
        _element_count: usize,
        _pipeline: Option<&wgpu::ComputePipeline>,
        _bind_group: Option<&wgpu::BindGroup>,
        cpu_work: Box<dyn FnOnce() + Send>,
        _label: &str,
    ) {
        sync.dispatch_cpu(cpu_work);
    }
}

impl ExecuteLane for GpuExecutor {
    fn execute(
        &self,
        sync: &mut CommandSync,
        element_count: usize,
        pipeline: Option<&wgpu::ComputePipeline>,
        bind_group: Option<&wgpu::BindGroup>,
        _cpu_work: Box<dyn FnOnce() + Send>,
        label: &str,
    ) {
        if let (Some(pipeline), Some(bg)) = (pipeline, bind_group) {
            let wgc = (element_count as u32).div_ceil(64);
            sync.dispatch_gpu(pipeline, bg, (wgc, 1, 1), label);
        }
    }
}

/// Static dispatch: resolves [`LaneTarget`] at compile time and calls the right executor.
/// No runtime branching — monomorphizes directly.
pub fn dispatch_lane<T, F>(
    sync: &mut CommandSync,
    element_count: usize,
    pipeline: Option<&wgpu::ComputePipeline>,
    bind_group: Option<&wgpu::BindGroup>,
    cpu_work: F,
) where
    T: LaneTarget,
    T::Target: Route,
    F: FnOnce() + Send + 'static,
{
    match PipelineRouter::resolve::<T>() {
        crate::dispatcher::Platform::Cpu => {
            CpuExecutor.execute(
                sync,
                element_count,
                pipeline,
                bind_group,
                Box::new(cpu_work),
                "",
            );
        }
        crate::dispatcher::Platform::Gpu => {
            GpuExecutor.execute(
                sync,
                element_count,
                pipeline,
                bind_group,
                Box::new(cpu_work),
                "",
            );
        }
    }
}

/// Type-level tag dispatch: use the ZST directly without a match.
/// Caller provides the executor ZST as a generic parameter.
pub fn dispatch_with<E, F>(
    executor: E,
    sync: &mut CommandSync,
    element_count: usize,
    pipeline: Option<&wgpu::ComputePipeline>,
    bind_group: Option<&wgpu::BindGroup>,
    cpu_work: F,
    label: &str,
) where
    E: ExecuteLane,
    F: FnOnce() + Send + 'static,
{
    executor.execute(
        sync,
        element_count,
        pipeline,
        bind_group,
        Box::new(cpu_work),
        label,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::{CpuLane, GpuLane, LaneTarget};

    struct CpuComponent;
    impl LaneTarget for CpuComponent {
        type Target = CpuLane;
    }

    struct GpuComponent;
    impl LaneTarget for GpuComponent {
        type Target = GpuLane;
    }

    #[test]
    fn cpu_executor_is_zst() {
        assert_eq!(std::mem::size_of::<CpuExecutor>(), 0);
    }

    #[test]
    fn gpu_executor_is_zst() {
        assert_eq!(std::mem::size_of::<GpuExecutor>(), 0);
    }

    #[test]
    fn dispatch_lane_compiles_for_cpu() {
        let ctx = crate::context::WgpuContext::new_blocking();
        let mut sync = crate::command_sync::CommandSync::new(ctx.device, ctx.queue);

        dispatch_lane::<CpuComponent, _>(&mut sync, 100, None, None, || {
            let _ = 1 + 1;
        });
        assert_eq!(sync.len(), 1);
    }

    #[test]
    fn dispatch_lane_compiles_for_gpu() {
        let ctx = crate::context::WgpuContext::new_blocking();
        let mut sync = crate::command_sync::CommandSync::new(ctx.device, ctx.queue);

        dispatch_lane::<GpuComponent, _>(&mut sync, 100, None, None, || {
            let _ = 1 + 1;
        });
        assert_eq!(sync.len(), 0);
    }

    #[test]
    fn zst_dispatch_with_executor() {
        let ctx = crate::context::WgpuContext::new_blocking();
        let mut sync = crate::command_sync::CommandSync::new(ctx.device, ctx.queue);

        dispatch_with(CpuExecutor, &mut sync, 10, None, None, || {}, "cpu");
        assert_eq!(sync.len(), 1);

        dispatch_with(GpuExecutor, &mut sync, 10, None, None, || {}, "gpu");
        assert_eq!(sync.len(), 1);
    }
}
