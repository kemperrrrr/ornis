//! Compile-time mapping from component lanes to execution platforms.
//!
//! [`PipelineRouter`] resolves a component's [`LaneTarget`] marker to a
//! concrete [`Platform`] — `Hybrid` and `Auto` currently route to GPU — and
//! maps a [`TargetDiscriminant`] back to an [`ExecutionTarget`]. Both paths
//! monomorphize to constants, with no runtime branching.

use crate::dispatcher::{ExecutionTarget, Platform};
use ornis_core::{LaneTarget, Route, TargetDiscriminant, lane_target_of};

/// Zero-cost bridge between [`LaneTarget`] (ZST) and [`ExecutionTarget`].
/// At runtime this compiles to a constant value — no branches.
pub struct PipelineRouter;

impl PipelineRouter {
    /// Resolve a component lane to an execution platform.
    /// Monomorphizes to `Platform::Cpu` or `Platform::Gpu` directly.
    pub fn resolve<T: LaneTarget>() -> Platform
    where
        T::Target: Route,
    {
        match lane_target_of::<T>() {
            TargetDiscriminant::Cpu => Platform::Cpu,
            TargetDiscriminant::Gpu => Platform::Gpu,
            TargetDiscriminant::Hybrid | TargetDiscriminant::Auto(_) => Platform::Gpu,
        }
    }

    /// Map a [`TargetDiscriminant`] to an [`ExecutionTarget`].
    pub fn execution_target(discriminant: TargetDiscriminant) -> ExecutionTarget {
        match discriminant {
            TargetDiscriminant::Cpu => ExecutionTarget::Cpu,
            TargetDiscriminant::Gpu => ExecutionTarget::Gpu,
            TargetDiscriminant::Hybrid | TargetDiscriminant::Auto(_) => {
                ExecutionTarget::Auto(10_000)
            }
        }
    }
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
    fn cpu_component_resolves_to_cpu() {
        assert_eq!(PipelineRouter::resolve::<CpuComponent>(), Platform::Cpu);
    }

    #[test]
    fn gpu_component_resolves_to_gpu() {
        assert_eq!(PipelineRouter::resolve::<GpuComponent>(), Platform::Gpu);
    }
}
