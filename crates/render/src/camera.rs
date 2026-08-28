//! Backend-neutral orbit camera and scheduled input consumer.
//!
//! The camera is useful to both the native showcase and the browser viewport:
//! platform adapters only update [`ornis_core::InputState`], while the shared
//! frame schedule consumes held left-button pointer deltas and wheel movement.
//! It owns no window, DOM or GPU state.

use std::sync::Mutex;

use glam::Vec3;
use ornis_core::{Engine, InputState, Resources, System, SystemAccess};

use crate::scene::CameraDesc;

/// Client-side orbit camera: azimuth/elevation around a target plus a zoom
/// radius. It is view state, not part of the server-authoritative scene.
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    target: Vec3,
    up: Vec3,
    azimuth: f32,
    elevation: f32,
    radius: f32,
    fov: f32,
    near: f32,
    far: f32,
}

impl OrbitCamera {
    const MIN_RADIUS: f32 = 0.5;
    const MAX_RADIUS: f32 = 1000.0;
    /// Keep elevation off the poles so `look_at` never degenerates.
    const ELEVATION_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
    const ROTATE_SPEED: f32 = 0.005;
    const ZOOM_SPEED: f32 = 0.001;

    /// Creates an orbit camera from a serialized look-at camera description.
    pub fn from_desc(cam: &CameraDesc) -> Self {
        let target = Vec3::from_array(cam.target);
        let offset = Vec3::from_array(cam.position) - target;
        let radius = offset.length().max(Self::MIN_RADIUS);
        // offset = radius * (cos(el)*cos(az), sin(el), cos(el)*sin(az))
        let elevation = (offset.y / radius).clamp(-1.0, 1.0).asin();
        let azimuth = offset.z.atan2(offset.x);
        Self {
            target,
            up: Vec3::from_array(cam.up),
            azimuth,
            elevation,
            radius,
            fov: cam.fov,
            near: cam.near,
            far: cam.far,
        }
    }

    /// Returns the current eye position around the orbit target.
    pub fn position(&self) -> Vec3 {
        let (ce, se) = (self.elevation.cos(), self.elevation.sin());
        let (ca, sa) = (self.azimuth.cos(), self.azimuth.sin());
        self.target + self.radius * Vec3::new(ce * ca, se, ce * sa)
    }

    /// Returns the look-at target, up vector, field of view, and clip planes.
    pub fn view_parameters(&self) -> (Vec3, Vec3, Vec3, f32, f32, f32) {
        (
            self.position(),
            self.target,
            self.up,
            self.fov,
            self.near,
            self.far,
        )
    }

    /// Applies the shared input contract: left-button drag rotates and wheel
    /// movement zooms. Transient deltas are cleared by [`Engine`] after the
    /// once-per-frame schedule consumes the same input resource.
    pub fn apply_input(&mut self, input: &InputState) {
        if input.mouse_button_down(0) {
            let [dx, dy] = input.pointer_delta();
            self.rotate(dx, dy);
        }
        self.zoom(input.wheel_delta());
    }

    fn rotate(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * Self::ROTATE_SPEED;
        self.elevation = (self.elevation + dy * Self::ROTATE_SPEED)
            .clamp(-Self::ELEVATION_LIMIT, Self::ELEVATION_LIMIT);
    }

    /// `delta_y` from a wheel event: positive scrolls down/away (zoom out).
    fn zoom(&mut self, delta_y: f32) {
        self.radius = (self.radius * (delta_y * Self::ZOOM_SPEED).exp())
            .clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
    }
}

/// Registers an [`OrbitCamera`] as a client-side resource and schedules its
/// once-per-frame [`InputState`] consumer.
///
/// The camera is intentionally stored in a mutex because systems receive a
/// shared `Resources` reference. This is a small view-state resource, not a
/// second authoritative world or a GPU representation.
pub fn install_orbit_camera(engine: &mut Engine, camera: OrbitCamera) {
    let _ = engine.world_mut().insert(Mutex::new(camera));
    engine.schedule_mut().add_system(OrbitCameraSystem);
}

/// Clones the current client-side orbit camera from an engine resource.
///
/// Returns `None` when [`install_orbit_camera`] has not been called. The
/// accessor is intended for the platform renderer after [`Engine::run_frame`]
/// has allowed the scheduled input consumer to update the camera.
pub fn read_orbit_camera(engine: &Engine) -> Option<OrbitCamera> {
    engine
        .world()
        .resources()
        .get::<Mutex<OrbitCamera>>()
        .map(|camera| camera.lock().expect("orbit camera lock").clone())
}

/// Once-per-frame system that applies the backend-neutral input snapshot to
/// the client-side orbit camera.
struct OrbitCameraSystem;

impl System for OrbitCameraSystem {
    fn name(&self) -> &'static str {
        "orbit_camera_input"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<InputState>()
            .writes::<Mutex<OrbitCamera>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(input) = resources.get::<InputState>() else {
            return;
        };
        let Some(camera) = resources.get::<Mutex<OrbitCamera>>() else {
            return;
        };
        camera
            .lock()
            .expect("orbit camera lock")
            .apply_input(input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> CameraDesc {
        CameraDesc {
            position: [0.0, 2.5, 9.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 60.0,
            near: 0.1,
            far: 100.0,
        }
    }

    #[test]
    fn orbit_camera_consumes_shared_input() {
        let mut orbit = OrbitCamera::from_desc(&camera());
        let initial = orbit.position();
        let mut input = InputState::new();
        input.set_mouse_button(0, true);
        input.set_pointer_position([10.0, 4.0]);
        input.add_wheel_delta(100.0);

        orbit.apply_input(&input);

        assert_ne!(orbit.position(), initial);
        assert!(orbit.position().length() > 0.5);
        assert_eq!(orbit.view_parameters().3, 60.0);
    }

    #[test]
    fn scheduled_camera_consumes_input_resource_once_per_frame() {
        let mut engine = Engine::new();
        let initial = OrbitCamera::from_desc(&camera()).position();
        install_orbit_camera(&mut engine, OrbitCamera::from_desc(&camera()));
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine input resource");
            input.set_mouse_button(0, true);
            input.set_pointer_position([10.0, 4.0]);
            input.add_wheel_delta(100.0);
        }

        engine.run_frame(0.0);

        let updated = read_orbit_camera(&engine)
            .expect("scheduled camera resource")
            .position();
        assert_ne!(updated, initial);
        let input = engine
            .world()
            .resources()
            .get::<InputState>()
            .expect("engine input resource");
        assert_eq!(input.pointer_delta(), [0.0, 0.0]);
        assert_eq!(input.wheel_delta(), 0.0);
    }

    #[test]
    fn elevation_is_clamped_away_from_poles() {
        let mut orbit = OrbitCamera::from_desc(&camera());
        let mut input = InputState::new();
        input.set_mouse_button(0, true);
        input.set_pointer_position([0.0, 100_000.0]);
        orbit.apply_input(&input);
        let position = orbit.position();
        assert!(position.y.abs() < position.length());
    }
}
