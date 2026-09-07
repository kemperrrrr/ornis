//! S5b gate (PLAN Appendix C): parallel command recording must be
//! pixel-identical to the sequential path. Runs headless — on CI via
//! lavapipe, locally on any adapter; skipped when no adapter is found.
//! The scene and harness live in `common` (shared with the E1 schedule
//! gate).

mod common;

#[test]
fn parallel_recording_matches_sequential_pixels() {
    common::with_headless_scene(|scene| {
        // Parallel: a level at a time (lighting ∥ forward inside), own
        // encoders per pass, single ordered submit.
        let par_pixels = common::render_frame_pixels(scene, |plan, context| {
            assert!(!plan.parallel_recording(), "sequential by default");
            plan.set_parallel_recording(true);
            plan.render(context, &scene.renderer, &scene.mesh, 1);
        });
        assert_eq!(
            common::sequential_reference_pixels(scene),
            par_pixels,
            "parallel recording must be pixel-identical"
        );
    });
}
