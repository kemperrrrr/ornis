//! E1 (S5e) gate: schedule-driven rendering (passes projected as core
//! `Schedule` systems, `schedule_bridge`) must be pixel-identical to the
//! sequential graph path. Runs headless — on CI via lavapipe, locally on
//! any adapter; skipped when no adapter is found. The scene and harness
//! live in `common` (shared with the S5b parallel gate).

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
