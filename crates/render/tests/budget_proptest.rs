//! S4 property (PLAN Приложение C): the scheduler either fits the pool
//! into the budget or refuses with an actionable error — never silently
//! exceeds it. Runs against random (technique x bloom x tail-culling x
//! surface size) configurations.

use ornis_render::{Budget, PassId, RenderGraph3D, Technique};
use proptest::prelude::*;

fn cfg(technique: Technique, bloom: bool, size: u32) -> RenderGraph3D {
    RenderGraph3D::new_with(
        wgpu::TextureFormat::Rgba8Unorm,
        (size, size),
        technique,
        bloom,
    )
}

proptest! {
    /// Culling only trailing passes keeps graph invariants intact
    /// (dropping a mid-chain writer would trip read-before-write).
    #[test]
    fn budget_holds_or_refuses(
        technique_idx in 0usize..3,
        bloom in proptest::bool::ANY,
        drop_tail in 0usize..4,
        size_idx in 0usize..3,
    ) {
        let technique = [Technique::Forward, Technique::Deferred, Technique::Hybrid]
            [technique_idx];
        let size = [64u32, 320, 1280][size_idx];
        let mut g3 = cfg(technique, bloom, size);

        let max_tail = if bloom { 3 } else { 1 };
        let drop_tail = drop_tail.min(max_tail);
        let total = g3.graph_mut().layout().passes.len();
        for i in 0..drop_tail.min(total) {
            let idx = total - 1 - i;
            g3.graph_mut().set_pass_enabled(PassId(idx as u32), false);
        }

        let planned = g3.graph_mut().try_layout().unwrap().planned_pool_bytes();

        // Exact budget always fits.
        g3.set_budget(Budget::gpu_textures(planned));
        prop_assert!(g3.graph_mut().try_layout().is_ok());

        // One byte less always refuses, with the real requirement.
        if planned > 0 {
            g3.set_budget(Budget::gpu_textures(planned - 1));
            let err = g3.graph_mut().try_layout().unwrap_err();
            prop_assert_eq!(err.required, planned);
            prop_assert_eq!(err.budget, planned - 1);
            prop_assert!(!err.offenders.is_empty());
        }

        // Unbounded restores the S3 behavior.
        g3.set_budget(Budget::unbounded());
        prop_assert!(g3.graph_mut().try_layout().is_ok());
    }
}
