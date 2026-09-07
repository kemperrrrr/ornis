//! E1/E2 (S5e) gates: schedule-driven rendering (passes projected as core
//! `Schedule` systems, `schedule_bridge`) and the buffers path (per-pass
//! encoders into `FrameCommandBuffers` + one ordered flush) must each be
//! pixel-identical to the sequential graph path. Runs headless — on CI
//! via lavapipe, locally on any adapter; skipped when no adapter is
//! found. The scene and harness live in `common` (shared with the S5b
//! parallel gate).

mod common;

#[test]
fn schedule_ordered_render_matches_sequential_pixels() {
    common::with_headless_scene(|scene| {
        // Schedule-driven: the E1 projection orders the same passes as
        // core `Schedule` systems; recording still goes through the
        // borrowed encoder (`render_schedule`).
        let sched_pixels = common::render_frame_pixels(scene, |plan, context| {
            plan.render_schedule(context, &scene.renderer, &scene.mesh, 1)
                .expect("E1 projection: production plans are typed");
        });
        assert_eq!(
            common::sequential_reference_pixels(scene),
            sched_pixels,
            "E1: schedule-driven rendering must be pixel-identical"
        );
    });
}

#[test]
fn buffers_path_matches_sequential_pixels() {
    common::with_headless_scene(|scene| {
        // E2: the encoder context as frame data — per-pass encoders land
        // in FrameCommandBuffers in registration order, one ordered
        // flush submits them. Same pixels as the borrowed-encoder path.
        let e2_pixels = common::render_frame_pixels(scene, |plan, context| {
            let buffers = ornis_render::gpu_resources::FrameCommandBuffers::default();
            plan.render_to_buffers(
                context.device,
                context.queue,
                context.target,
                &scene.renderer,
                &scene.mesh,
                1,
                &buffers,
            )
            .expect("E2 projection: production plans are typed");
            buffers.flush(context.queue);
        });
        assert_eq!(
            common::sequential_reference_pixels(scene),
            e2_pixels,
            "E2: buffers-path rendering must be pixel-identical"
        );
    });
}
