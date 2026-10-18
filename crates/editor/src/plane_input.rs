//! Plane-camera engine seam (PLAN §k, slice K2): resource install, resource
//! read-back, and the 2D input frame system.
//!
//! Kept out of [`crate::plane`] so the camera type stays a pure view model
//! and this module owns the schedule/resource contract. [`install_plane_camera`]
//! is idempotent (schedule-entry guard, like `install_editor`).

use std::sync::Mutex;

use ornis_core::{Engine, InputState, Resources, System, SystemAccess};

use crate::plane::{PlaneCamera, ViewportMode};

/// Installs the plane editing domain into `engine`.
///
/// Inserts the [`Mutex<PlaneCamera>`] resource (replacing any previous plane
/// camera), the [`ViewportMode`] resource (only when absent, so a host that
/// already entered 2D keeps its mode), and the `plane_camera_input` frame
/// system. Idempotent: a second call registers no second system (guarded on
/// the schedule entry, like `install_orbit_camera`/`install_editor`).
pub fn install_plane_camera(engine: &mut Engine, camera: PlaneCamera) {
    let _ = engine.world_mut().insert(Mutex::new(camera));
    if engine.world().resources().get::<ViewportMode>().is_none() {
        let _ = engine.world_mut().insert(ViewportMode::Perspective3d);
    }
    if engine.schedule().mermaid().contains(SYSTEM_NAME) {
        return;
    }
    engine.schedule_mut().add_system(PlaneCameraInput);
}

/// Clones the current plane camera from an engine resource.
///
/// Returns `None` when [`install_plane_camera`] has not been called.
pub fn read_plane_camera(engine: &Engine) -> Option<PlaneCamera> {
    engine
        .world()
        .resources()
        .get::<Mutex<PlaneCamera>>()
        .map(|camera| camera.lock().unwrap_or_else(|e| e.into_inner()).clone())
}

/// Schedule name of the plane-camera input system (idempotency guard).
const SYSTEM_NAME: &str = "plane_camera_input";

/// Frame system applying the plane input contract while in 2D.
///
/// No-op in [`ViewportMode::Perspective3d`] (the orbit camera owns the
/// input there); in [`ViewportMode::Plane2d`] middle/right drag pans and the
/// wheel zooms around the cursor, and the pick projection tracks `hh` even
/// though the projection variant never changes here. The left button never
/// moves this camera — it selects through `editor_maintain`.
struct PlaneCameraInput;

impl System for PlaneCameraInput {
    /// Schedule name of this system (matches [`SYSTEM_NAME`]).
    fn name(&self) -> &'static str {
        SYSTEM_NAME
    }

    /// Declares the honest access set: input and mode reads, plane-camera
    /// write (schedule enforcement panics on anything undeclared).
    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<InputState>()
            .reads::<ViewportMode>()
            .writes::<Mutex<PlaneCamera>>()
    }

    /// 2D behaviour: pan-and-zoom from the shared input snapshot.
    fn run(&self, resources: &Resources) {
        let mode = resources.get::<ViewportMode>().copied().unwrap_or_default();
        if !mode.is_plane() {
            return;
        }
        let (Some(input), Some(camera)) = (
            resources.get::<InputState>(),
            resources.get::<Mutex<PlaneCamera>>(),
        ) else {
            return;
        };
        camera
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply_input(input);
    }
}

#[cfg(test)]
mod tests {
    use glam::Vec2;
    use ornis_core::{Engine, InputState, MouseButton};

    use super::*;
    use crate::plane::PlaneCamera;

    /// Ortho rig: center origin, `hh = 5`, eye at 10, `800×600` viewport.
    fn camera() -> PlaneCamera {
        PlaneCamera::new(
            glam::Vec2::ZERO,
            ornis_core::Meters::new(5.0),
            ornis_core::Meters::new(10.0),
            ornis_core::Meters::new(0.1),
            ornis_core::Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("camera")
    }

    /// The input system only drives the camera in `Plane2d`; install is
    /// idempotent and entering 2D then leaving keeps the plane state.
    #[test]
    fn input_system_gates_on_mode_and_install_is_idempotent() {
        let mut engine = Engine::new();
        install_plane_camera(&mut engine, camera());
        install_plane_camera(&mut engine, camera());
        assert_eq!(
            engine
                .schedule()
                .mermaid()
                .matches("plane_camera_input")
                .count(),
            1
        );
        assert!(
            engine
                .world()
                .resources()
                .get::<Mutex<PlaneCamera>>()
                .is_some()
        );
        // Perspective3d: the system leaves the camera alone.
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine input resource");
            input.set_button(MouseButton::Right, true);
            input.set_pointer_anchor([400.0, 300.0]);
            input.set_pointer_position([500.0, 300.0]);
        }
        engine.run_frame(1.0 / 60.0);
        assert_eq!(
            read_plane_camera(&engine).expect("plane camera").center(),
            Vec2::ZERO
        );
        // Plane2d: a fresh drag pans (frame transients cleared above, so
        // move the pointer again to generate a new delta).
        let _ = engine.world_mut().insert(ViewportMode::Plane2d);
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine input resource");
            input.set_pointer_position([600.0, 300.0]);
        }
        engine.run_frame(1.0 / 60.0);
        let moved = read_plane_camera(&engine).expect("plane camera").center();
        assert!((moved - Vec2::new(-100.0 / 60.0, 0.0)).length() < 1e-4);
    }
}
