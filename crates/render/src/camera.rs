//! Backend-neutral orbit camera and scheduled input consumer.
//!
//! The camera is useful to both the native showcase and the browser viewport:
//! platform adapters only update [`ornis_core::InputState`], while the shared
//! frame schedule consumes held left-button pointer deltas and wheel movement.
//! It owns no window, DOM or GPU state.

use std::sync::Mutex;

use glam::{Mat4, Vec3, Vec4};
use ornis_core::{Degrees, Engine, InputState, Meters, Resources, System, SystemAccess};

use ornis_assets::scene::CameraDesc;

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
    /// `|view · up|` above which the default up axis would be parallel.
    const VIEW_UP_PARALLEL: f32 = 0.999;
    /// Vertical field of view used by [`Self::looking_at`].
    pub const DEFAULT_FOV: Degrees = Degrees(60.0);
    /// Near clip distance used by [`Self::looking_at`].
    pub const DEFAULT_NEAR: Meters = Meters(0.1);
    /// Far clip distance used by [`Self::looking_at`].
    pub const DEFAULT_FAR: Meters = Meters(100.0);

    /// Creates an orbit camera from a serialized look-at camera description.
    pub fn from_desc(cam: &CameraDesc) -> Self {
        Self::looking_at(Vec3::from_array(cam.position), Vec3::from_array(cam.target))
            .with_up(Vec3::from_array(cam.up))
            .with_fov(Degrees::new(cam.fov))
            .with_clip(Meters::new(cam.near), Meters::new(cam.far))
    }

    /// Orbit camera aimed from `eye` at `target`.
    ///
    /// Field of view and clip planes start at [`Self::DEFAULT_FOV`],
    /// [`Self::DEFAULT_NEAR`] and [`Self::DEFAULT_FAR`]. Up is [`Vec3::Y`],
    /// or [`Vec3::Z`] when the view is parallel to Y so the basis does not
    /// collapse. A zero offset uses the minimum orbit radius along +X.
    pub fn looking_at(eye: Vec3, target: Vec3) -> Self {
        let offset = eye - target;
        let radius = offset.length().max(Self::MIN_RADIUS);
        // offset = radius * (cos(el)*cos(az), sin(el), cos(el)*sin(az))
        let elevation = (offset.y / radius).clamp(-1.0, 1.0).asin();
        let azimuth = offset.z.atan2(offset.x);
        Self {
            target,
            up: up_for_offset(offset),
            azimuth,
            elevation,
            radius,
            fov: Self::DEFAULT_FOV.get(),
            near: Self::DEFAULT_NEAR.get(),
            far: Self::DEFAULT_FAR.get(),
        }
    }

    /// Replaces the up axis. The caller is responsible for keeping it
    /// non-parallel to the view direction.
    pub fn with_up(mut self, up: Vec3) -> Self {
        self.up = up;
        self
    }

    /// Sets the vertical field of view in degrees.
    pub fn with_fov(mut self, fov: Degrees) -> Self {
        self.fov = fov.get();
        self
    }

    /// Sets the near and far clip distances.
    pub fn with_clip(mut self, near: Meters, far: Meters) -> Self {
        self.near = near.get();
        self.far = far.get();
        self
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
/// second authoritative world or a GPU representation. A repeat call replaces
/// the camera and does not register the input system twice.
pub fn install_orbit_camera(engine: &mut Engine, camera: OrbitCamera) {
    let _ = engine.world_mut().insert(Mutex::new(camera));
    if engine.schedule().mermaid().contains("orbit_camera_input") {
        return;
    }
    engine.schedule_mut().add_system(OrbitCameraSystem);
}

/// Up axis for an offset from the look-at target.
fn up_for_offset(offset: Vec3) -> Vec3 {
    let len = offset.length();
    if len > f32::EPSILON && (offset.y / len).abs() > OrbitCamera::VIEW_UP_PARALLEL {
        Vec3::Z
    } else {
        Vec3::Y
    }
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
        .map(|camera| camera.lock().unwrap_or_else(|e| e.into_inner()).clone())
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
            .unwrap_or_else(|e| e.into_inner())
            .apply_input(input);
    }
}

/// Frame view-projection from orbit view parameters and a surface size
/// (S7): the aspect falls back to 1.0 for a zero dimension, the
/// perspective is the DirectX-style projection of the legacy renderer.
///
/// Kept as a free function so `RenderSubmit::run` stays pure
/// orchestration (IOSP): guards, one lane read, one projection, uploads.
pub fn camera_view_projection(
    view: (Vec3, Vec3, Vec3, f32, f32, f32),
    surface_size: (u32, u32),
) -> (Mat4, Vec3) {
    let (cam_pos, cam_target, cam_up, fov, near, far) = view;
    let (w, h) = (surface_size.0 as f64, surface_size.1 as f64);
    let aspect = if h > 0.0 { w as f32 / h as f32 } else { 1.0 };
    let view_matrix = glam::camera::rh::view::look_at_mat4(cam_pos, cam_target, cam_up);
    let projection =
        glam::camera::rh::proj::directx::perspective(fov.to_radians(), aspect, near, far);
    (projection * view_matrix, cam_pos)
}

/// Six normalized world-space frustum planes for CPU-side sphere culling.
///
/// Read-only helper: built once per frame from the frame `view_proj` and
/// queried per entity via [`Frustum::sphere_visible`]. Planes follow
/// Gribb/Hartmann extraction from the matrix rows; the depth planes assume
/// the DirectX `0..1` convention of [`camera_view_projection`]
/// (`near = row2`, `far = row3 - row2`). A degenerate (zero-length or
/// non-finite) plane never culls — the query fails open.
#[derive(Clone, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    /// Extracts the six planes from `view_proj` and normalizes them.
    pub fn from_view_proj(view_proj: &Mat4) -> Self {
        /// Homogeneous (clip-w) row of a 4×4 view-projection matrix.
        const CLIP_W_ROW: usize = 3;
        let r0 = view_proj.row(0);
        let r1 = view_proj.row(1);
        let r2 = view_proj.row(2);
        let r3 = view_proj.row(CLIP_W_ROW);
        Self {
            planes: [
                r3 + r0, // left
                r3 - r0, // right
                r3 + r1, // bottom
                r3 - r1, // top
                r2,      // near (DirectX 0..1)
                r3 - r2, // far
            ]
            .map(|p| {
                let len = p.truncate().length();
                /// Squared-length floor for a usable frustum-plane normal.
                const PLANE_NORMAL_EPS: f32 = 1e-12;
                if len.is_finite() && len > PLANE_NORMAL_EPS {
                    p / len
                } else {
                    // Degenerate: mark with a NaN normal so the query
                    // fails open instead of culling the frame.
                    Vec4::NAN
                }
            }),
        }
    }

    /// Tests a world-space sphere against the six planes.
    ///
    /// Returns `false` only when the sphere is fully outside at least one
    /// plane with a valid normal; degenerate planes and non-finite
    /// inputs return `true` (never falsely cull).
    pub fn sphere_visible(&self, center: Vec3, radius: f32) -> bool {
        if !center.is_finite() || !radius.is_finite() || radius < 0.0 {
            return true;
        }
        for plane in &self.planes {
            if !plane.is_finite() {
                continue;
            }
            let distance = plane.truncate().dot(center) + plane.w;
            if distance < -radius {
                return false;
            }
        }
        true
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
            fov: OrbitCamera::DEFAULT_FOV.get(),
            near: OrbitCamera::DEFAULT_NEAR.get(),
            far: OrbitCamera::DEFAULT_FAR.get(),
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
        assert_eq!(orbit.view_parameters().3, OrbitCamera::DEFAULT_FOV.get());
    }

    #[test]
    fn looking_at_places_the_eye_and_names_the_defaults() {
        let eye = Vec3::new(2.5, 1.8, 3.5);
        let orbit = OrbitCamera::looking_at(eye, Vec3::Y).with_fov(Degrees(45.0));
        let (position, target, up, fov, near, far) = orbit.view_parameters();
        assert!((position - eye).length() < 1e-4);
        assert_eq!(target, Vec3::Y);
        assert_eq!(up, Vec3::Y);
        assert_eq!(fov, 45.0);
        assert_eq!(near, OrbitCamera::DEFAULT_NEAR.get());
        assert_eq!(far, OrbitCamera::DEFAULT_FAR.get());
        let overhead = OrbitCamera::looking_at(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
        assert_eq!(overhead.view_parameters().2, Vec3::Z);
    }

    #[test]
    fn install_orbit_camera_replaces_without_a_second_system() {
        let mut engine = Engine::new();
        let camera = OrbitCamera::from_desc(&camera());
        install_orbit_camera(&mut engine, camera.clone());
        install_orbit_camera(&mut engine, camera);
        assert_eq!(
            engine
                .schedule()
                .mermaid()
                .matches("orbit_camera_input")
                .count(),
            1
        );
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

    #[test]
    fn view_projection_aspect_fallback_and_finite_matrices() {
        let orbit = OrbitCamera::from_desc(&camera());
        let view = orbit.view_parameters();
        let (view_proj, cam_pos) = camera_view_projection(view, (1920, 1080));
        assert_eq!(cam_pos, view.0);
        assert!(view_proj.to_cols_array().iter().all(|c| c.is_finite()));
        // A zero dimension falls back to square aspect — no NaN, and the
        // projection differs from the wide-surface one.
        let (square, _) = camera_view_projection(view, (0, 0));
        assert!(square.to_cols_array().iter().all(|c| c.is_finite()));
        assert_ne!(square, view_proj);
    }
}
