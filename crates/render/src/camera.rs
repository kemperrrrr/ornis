//! Backend-neutral orbit camera and input consumer.
//!
//! The camera is useful to both the native showcase and the browser viewport:
//! platform adapters only update [`ornis_core::InputState`], while this type
//! consumes held left-button pointer deltas and wheel movement at the frame
//! boundary. It owns no window, DOM or GPU state.

use glam::Vec3;
use ornis_core::InputState;

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
    /// movement zooms. Transient deltas are cleared by [`ornis_core::Engine`]
    /// after the schedule consumes the same input resource.
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
