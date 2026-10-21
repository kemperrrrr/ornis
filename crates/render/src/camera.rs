//! Backend-neutral orbit camera and scheduled input consumer.
//!
//! The camera is useful to both the native showcase and the browser viewport:
//! platform adapters only update [`ornis_core::InputState`], while the shared
//! frame schedule consumes held left-button pointer deltas and wheel movement.
//! It owns no window, DOM or GPU state.

use std::sync::Mutex;

use glam::{Mat4, Vec3, Vec4};
use ornis_core::{Degrees, Engine, InputState, Meters, Resources, System, SystemAccess};

use ornis_assets::scene::{CameraDesc, CameraProjection};

/// Device pixel ratio of the pointer backend feeding [`OrbitCamera`].
///
/// Browser adapters deliver pointer deltas in device pixels, while orbit
/// rotation is anchored to CSS pixels: on a hidpi surface (dpr 2) one CSS
/// pixel of mouse movement arrives as two device pixels, and without this
/// resource the rotation would spin twice as fast as on a 1x display.
/// The wasm render loop publishes the window's `devicePixelRatio`; the
/// world without the resource falls back to [`PointerDpr::default`]
/// (`1.0`). Note this default matches native only on 1x displays: the
/// native showcase feeds `CursorMoved` physical pixels and never inserts
/// this resource, so the camera divides them by `1.0` instead of the
/// display scale — on a HiDPI native monitor the orbit rotates
/// `scale_factor` times faster than on a 1x display (pre-existing native
/// behaviour, tracked separately by the render track — not a regression
/// of this type). Wheel zoom is not scaled — it is a discrete gesture,
/// not a positional delta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointerDpr(pub f32);

impl Default for PointerDpr {
    fn default() -> Self {
        Self(1.0)
    }
}

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
    projection: CameraProjection,
}

/// Named orbit-camera view parameters for one frame (K0).
///
/// Replaces the `(eye, target, up, fov, near, far)` tuple: the projection
/// travels alongside the look-at frame so [`camera_view_projection`]
/// renders the authored perspective/orthographic mode without a second
/// argument. Field of view and clip planes stay typed
/// ([`Degrees`]/[`Meters`], never bare `f32`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraView {
    /// Eye position in world units.
    pub eye: Vec3,
    /// Look-at target in world units.
    pub target: Vec3,
    /// Up direction (non-parallel to the view direction).
    pub up: Vec3,
    /// Projection: perspective foreshortening or an orthographic box.
    pub projection: CameraProjection,
    /// Vertical field of view in degrees (ignored in orthographic mode).
    pub fov: Degrees,
    /// Near clip distance in meters.
    pub near: Meters,
    /// Far clip distance in meters.
    pub far: Meters,
}

impl OrbitCamera {
    const MIN_RADIUS: f32 = 0.5;
    const MAX_RADIUS: f32 = 1000.0;
    /// Orthographic zoom bounds: the wheel moves `half_height` inside
    /// `[0.01, 1000]` m (same exponential speed as the radius zoom).
    const MIN_HALF_HEIGHT: Meters = Meters(0.01);
    const MAX_HALF_HEIGHT: Meters = Meters(1000.0);
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
    ///
    /// Eye, target, up, projection, field of view and clip planes come from
    /// `cam`. [`Self::looking_at`] only supplies the orbit basis.
    pub fn from_desc(cam: &CameraDesc) -> Self {
        Self {
            projection: cam.projection,
            ..Self::looking_at(cam.position, cam.target)
                .with_up(cam.up.get())
                .with_fov(cam.fov)
                .with_clip(cam.near, cam.far)
        }
    }

    /// Orbit camera aimed from `eye` at `target`.
    ///
    /// Field of view and clip planes start at [`Self::DEFAULT_FOV`],
    /// [`Self::DEFAULT_NEAR`] and [`Self::DEFAULT_FAR`], the projection at
    /// [`CameraProjection::Perspective`]. Up is [`Vec3::Y`],
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
            projection: CameraProjection::Perspective,
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

    /// Returns the look-at frame, projection, field of view and clip planes
    /// for one frame.
    pub fn view_parameters(&self) -> CameraView {
        CameraView {
            eye: self.position(),
            target: self.target,
            up: self.up,
            projection: self.projection,
            fov: Degrees::new(self.fov),
            near: Meters::new(self.near),
            far: Meters::new(self.far),
        }
    }

    /// Applies the shared input contract: left-button drag rotates and wheel
    /// movement zooms. Pointer deltas arrive in device pixels and are divided
    /// by `pointer_dpr` (see [`PointerDpr`]) so rotation speed is identical
    /// across display scales; a non-finite or non-positive dpr is treated as
    /// `1.0`. Transient deltas are cleared by [`Engine`] after the
    /// once-per-frame schedule consumes the same input resource.
    pub fn apply_input(&mut self, input: &InputState, pointer_dpr: f32) {
        if input.mouse_button_down(0) {
            let [dx, dy] = input.pointer_delta();
            let dpr = if pointer_dpr.is_finite() && pointer_dpr > 0.0 {
                pointer_dpr
            } else {
                1.0
            };
            self.rotate(dx / dpr, dy / dpr);
        }
        self.zoom(input.wheel_delta());
    }

    fn rotate(&mut self, dx: f32, dy: f32) {
        self.azimuth -= dx * Self::ROTATE_SPEED;
        self.elevation = (self.elevation + dy * Self::ROTATE_SPEED)
            .clamp(-Self::ELEVATION_LIMIT, Self::ELEVATION_LIMIT);
    }

    /// `delta_y` from a wheel event: positive scrolls down/away (zoom out).
    /// In orthographic mode the wheel moves `half_height` (same exponential
    /// speed, clamped to `[0.01, 1000]` m) and the orbit radius stays put;
    /// rotation is unchanged in both modes.
    fn zoom(&mut self, delta_y: f32) {
        match &mut self.projection {
            CameraProjection::Perspective => {
                self.radius = (self.radius * (delta_y * Self::ZOOM_SPEED).exp())
                    .clamp(Self::MIN_RADIUS, Self::MAX_RADIUS);
            }
            CameraProjection::Orthographic { half_height } => {
                let zoomed = half_height.get() * (delta_y * Self::ZOOM_SPEED).exp();
                *half_height = Meters::new(
                    zoomed.clamp(Self::MIN_HALF_HEIGHT.get(), Self::MAX_HALF_HEIGHT.get()),
                );
            }
        }
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
            .reads::<PointerDpr>()
            .writes::<Mutex<OrbitCamera>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(input) = resources.get::<InputState>() else {
            return;
        };
        let Some(camera) = resources.get::<Mutex<OrbitCamera>>() else {
            return;
        };
        let pointer_dpr = resources.get::<PointerDpr>().copied().unwrap_or_default().0;
        camera
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .apply_input(input, pointer_dpr);
    }
}

/// Frame view-projection from orbit view parameters and a surface size
/// (S7): the aspect falls back to 1.0 for a zero dimension, the
/// perspective is the DirectX-style projection of the legacy renderer,
/// and the orthographic branch is the same DirectX `0..1` box the shadow
/// maps use (`±half_height` vertically, aspect-scaled horizontally).
///
/// Kept as a free function so `RenderSubmit::run` stays pure
/// orchestration (IOSP): guards, one lane read, one projection, uploads.
pub fn camera_view_projection(view: &CameraView, surface_size: (u32, u32)) -> (Mat4, Vec3) {
    let (w, h) = (surface_size.0 as f64, surface_size.1 as f64);
    let aspect = if h > 0.0 { w as f32 / h as f32 } else { 1.0 };
    let view_matrix = glam::camera::rh::view::look_at_mat4(view.eye, view.target, view.up);
    let projection = match view.projection {
        CameraProjection::Perspective => glam::camera::rh::proj::directx::perspective(
            view.fov.get().to_radians(),
            aspect,
            view.near.get(),
            view.far.get(),
        ),
        CameraProjection::Orthographic { half_height } => {
            let half = half_height.get();
            glam::camera::rh::proj::directx::orthographic(
                -half * aspect,
                half * aspect,
                -half,
                half,
                view.near.get(),
                view.far.get(),
            )
        }
    };
    (projection * view_matrix, view.eye)
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

    /// Normalized frustum planes for the GPU culling uniform
    /// ([`crate::gpu_culling::cull_params`]): the same six `vec4`s the
    /// CPU test evaluates, so the compute pass and
    /// [`Frustum::sphere_visible`] agree by construction.
    pub fn planes(&self) -> [Vec4; 6] {
        self.planes
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
    use ornis_core::UnitVec3;

    fn camera() -> CameraDesc {
        CameraDesc {
            position: Vec3::new(0.0, 2.5, 9.0),
            target: Vec3::ZERO,
            up: UnitVec3::Y,
            fov: Degrees::new(60.0),
            near: Meters::new(0.1),
            far: Meters::new(100.0),
            projection: CameraProjection::Perspective,
        }
    }

    fn ortho_camera(half_height: Meters) -> CameraDesc {
        CameraDesc::try_orthographic_units(
            [Meters::new(0.0), Meters::new(0.0), Meters::new(9.0)],
            [Meters::new(0.0), Meters::new(0.0), Meters::new(0.0)],
            [0.0, 1.0, 0.0],
            half_height,
            Meters::new(0.1),
            Meters::new(100.0),
        )
        .expect("valid orthographic test camera")
    }

    /// NDC of a world point through a view-projection matrix.
    fn ndc_of(view_proj: &Mat4, point: Vec3) -> Vec3 {
        let clip = *view_proj * point.extend(1.0);
        (clip / clip.w).truncate()
    }

    /// Orthographic half-height of a [`CameraView`] built by the tests.
    fn ortho_half_height(view: &CameraView) -> Meters {
        match view.projection {
            CameraProjection::Orthographic { half_height } => half_height,
            CameraProjection::Perspective => panic!("expected an orthographic test view"),
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

        orbit.apply_input(&input, 1.0);

        assert_ne!(orbit.position(), initial);
        assert!(orbit.position().length() > 0.5);
        assert_eq!(
            orbit.view_parameters().fov.get(),
            OrbitCamera::DEFAULT_FOV.get()
        );
    }

    #[test]
    fn from_desc_copies_typed_fov_clip_and_up() {
        let desc = CameraDesc {
            position: Vec3::new(2.5, 1.8, 3.5),
            target: Vec3::Y,
            up: UnitVec3::Z,
            fov: Degrees::new(45.0),
            near: Meters::new(0.25),
            far: Meters::new(250.0),
            projection: CameraProjection::Perspective,
        };
        let orbit = OrbitCamera::from_desc(&desc);
        let view = orbit.view_parameters();
        assert!((view.eye - desc.position).length() < 1e-4);
        assert_eq!(view.target, desc.target);
        assert_eq!(view.up, desc.up.get());
        assert_eq!(view.fov, desc.fov);
        assert_eq!(view.near, desc.near);
        assert_eq!(view.far, desc.far);
        assert_eq!(view.projection, desc.projection);

        let ortho = ortho_camera(Meters::new(2.0));
        let orbit = OrbitCamera::from_desc(&ortho);
        let view = orbit.view_parameters();
        assert!((view.eye - ortho.position).length() < 1e-4);
        assert_eq!(view.projection, ortho.projection);
    }

    #[test]
    fn looking_at_places_the_eye_and_names_the_defaults() {
        let eye = Vec3::new(2.5, 1.8, 3.5);
        let orbit = OrbitCamera::looking_at(eye, Vec3::Y).with_fov(Degrees(45.0));
        let view = orbit.view_parameters();
        assert!((view.eye - eye).length() < 1e-4);
        assert_eq!(view.target, Vec3::Y);
        assert_eq!(view.up, Vec3::Y);
        assert_eq!(view.fov.get(), 45.0);
        assert_eq!(view.projection, CameraProjection::Perspective);
        assert_eq!(view.near, OrbitCamera::DEFAULT_NEAR);
        assert_eq!(view.far, OrbitCamera::DEFAULT_FAR);
        let overhead = OrbitCamera::looking_at(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO);
        assert_eq!(overhead.view_parameters().up, Vec3::Z);
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
        orbit.apply_input(&input, 1.0);
        let position = orbit.position();
        assert!(position.y.abs() < position.length());
    }

    /// Review (round 4): the scheduled input system divides device-pixel
    /// deltas by the [`PointerDpr`] resource — the same 40px drag yields
    /// half the rotation at dpr 2 — while a world without the resource
    /// (native, plain 1x browser) keeps the full 1x behaviour. Runs
    /// through `run_frame`, so the declared `PointerDpr` read is enforced
    /// by the access checker too.
    #[test]
    fn pointer_dpr_scales_rotation_for_device_pixel_backends() {
        let mut engine = Engine::new();
        install_orbit_camera(&mut engine, OrbitCamera::from_desc(&camera()));
        let drag = |input: &mut InputState| {
            input.set_mouse_button(0, true);
            input.set_pointer_anchor([0.0, 0.0]);
            input.set_pointer_position([40.0, 0.0]);
        };
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine input resource");
            drag(input);
        }
        let initial = read_orbit_camera(&engine).expect("camera resource").azimuth;
        engine.run_frame(0.0);
        let full = read_orbit_camera(&engine).expect("camera resource").azimuth - initial;
        assert!(full.abs() > 0.0, "a 40px drag must rotate at dpr 1");

        // Same drag with a 2x backend: half the rotation.
        let _ = engine.world_mut().resources_mut().insert(PointerDpr(2.0));
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine input resource");
            drag(input);
        }
        let before = read_orbit_camera(&engine).expect("camera resource").azimuth;
        engine.run_frame(0.0);
        let half = read_orbit_camera(&engine).expect("camera resource").azimuth - before;
        assert!(
            (full - 2.0 * half).abs() < 1e-5,
            "dpr 1 rotation {full} vs dpr 2 rotation {half}"
        );
    }

    /// Review (round 6): the dpr guard inside [`OrbitCamera::apply_input`]
    /// must treat every non-positive or non-finite scale as `1.0`, so
    /// `0.0`, `-1.0`, `NaN` and `INFINITY` rotate by exactly the same
    /// angles as an unguarded `1.0` call (kills the "condition = true"
    /// mutation, which would feed `dx / 0.0` straight into `rotate`).
    #[test]
    fn apply_input_treats_degenerate_pointer_dpr_as_one() {
        assert_eq!(PointerDpr::default().0, 1.0, "default lives in the type");
        let rotation_at = |dpr: f32| {
            let mut orbit = OrbitCamera::from_desc(&camera());
            let mut input = InputState::new();
            input.set_mouse_button(0, true);
            input.set_pointer_anchor([0.0, 0.0]);
            input.set_pointer_position([40.0, 25.0]);
            orbit.apply_input(&input, dpr);
            (orbit.azimuth, orbit.elevation)
        };
        let (reference_azimuth, reference_elevation) = rotation_at(1.0);
        let untouched = OrbitCamera::from_desc(&camera());
        assert_ne!(
            reference_azimuth, untouched.azimuth,
            "the drag must actually rotate at dpr 1"
        );
        for degenerate in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let (azimuth, elevation) = rotation_at(degenerate);
            assert!(
                (azimuth - reference_azimuth).abs() < 1e-6,
                "dpr {degenerate} azimuth {azimuth} != 1.0 azimuth {reference_azimuth}"
            );
            assert!(
                (elevation - reference_elevation).abs() < 1e-6,
                "dpr {degenerate} elevation {elevation} != 1.0 elevation {reference_elevation}"
            );
        }
    }

    #[test]
    fn view_projection_aspect_fallback_and_finite_matrices() {
        let orbit = OrbitCamera::from_desc(&camera());
        let view = orbit.view_parameters();
        let (view_proj, cam_pos) = camera_view_projection(&view, (1920, 1080));
        assert_eq!(cam_pos, view.eye);
        assert!(view_proj.to_cols_array().iter().all(|c| c.is_finite()));
        // A zero dimension falls back to square aspect — no NaN, and the
        // projection differs from the wide-surface one.
        let (square, _) = camera_view_projection(&view, (0, 0));
        assert!(square.to_cols_array().iter().all(|c| c.is_finite()));
        assert_ne!(square, view_proj);
        // The fallback is exactly aspect 1: a square surface agrees.
        let (unit, _) = camera_view_projection(&view, (10, 10));
        assert_eq!(square, unit);
    }

    #[test]
    fn orthographic_matrix_maps_view_box_edges_to_ndc_unit() {
        // `half_height = 2` on a 320×180 frame: the box rim at the target
        // depth lands on NDC ±1.
        let half = 2.0_f32;
        let orbit = OrbitCamera::from_desc(&ortho_camera(Meters::new(half)));
        let view = orbit.view_parameters();
        let (view_proj, _) = camera_view_projection(&view, (320, 180));
        let aspect = 320.0 / 180.0;
        for (point, expected) in [
            (Vec3::new(half * aspect, 0.0, 0.0), [1.0, 0.0]),
            (Vec3::new(-half * aspect, 0.0, 0.0), [-1.0, 0.0]),
            (Vec3::new(0.0, half, 0.0), [0.0, 1.0]),
            (Vec3::new(0.0, -half, 0.0), [0.0, -1.0]),
        ] {
            let ndc = ndc_of(&view_proj, point);
            assert!(
                (ndc.x - expected[0]).abs() < 1e-4
                    && (ndc.y - expected[1]).abs() < 1e-4
                    && (0.0..=1.0).contains(&ndc.z),
                "{point:?} -> {ndc:?}, expected rim {expected:?} inside the depth range"
            );
        }
    }

    #[test]
    fn orthographic_screen_size_is_distance_independent() {
        // The same world offset maps to the same NDC from two eye
        // distances; the perspective camera shrinks with distance.
        let half = Meters::new(2.0);
        let orbit = |eye_z: f32| {
            OrbitCamera::from_desc(
                &CameraDesc::try_orthographic_units(
                    [Meters::new(0.0), Meters::new(0.0), Meters::new(eye_z)],
                    [Meters::new(0.0), Meters::new(0.0), Meters::new(0.0)],
                    [0.0, 1.0, 0.0],
                    half,
                    Meters::new(0.1),
                    Meters::new(100.0),
                )
                .expect("valid ortho camera"),
            )
        };
        let (near_vp, _) = camera_view_projection(&orbit(9.0).view_parameters(), (320, 180));
        let (far_vp, _) = camera_view_projection(&orbit(20.0).view_parameters(), (320, 180));
        let point = Vec3::new(2.0, 1.5, 0.0);
        let (near_ndc, far_ndc) = (ndc_of(&near_vp, point), ndc_of(&far_vp, point));
        // Screen position (x/y) is identical; only the depth differs.
        assert!(
            (near_ndc.truncate() - far_ndc.truncate()).length() < 1e-6,
            "{near_ndc:?} vs {far_ndc:?}"
        );

        let persp = |eye_z: f32| {
            OrbitCamera::looking_at(Vec3::new(0.0, 0.0, eye_z), Vec3::ZERO)
                .with_fov(Degrees::new(60.0))
        };
        let (near_vp, _) = camera_view_projection(&persp(9.0).view_parameters(), (320, 180));
        let (far_vp, _) = camera_view_projection(&persp(20.0).view_parameters(), (320, 180));
        assert!(
            (ndc_of(&near_vp, point) - ndc_of(&far_vp, point)).length() > 0.1,
            "perspective must shrink with distance"
        );
    }

    #[test]
    fn orthographic_zero_surface_size_falls_back_to_unit_aspect() {
        let orbit = OrbitCamera::from_desc(&ortho_camera(Meters::new(2.0)));
        let view = orbit.view_parameters();
        let (zero, _) = camera_view_projection(&view, (0, 0));
        assert!(zero.to_cols_array().iter().all(|c| c.is_finite()));
        let (unit, _) = camera_view_projection(&view, (10, 10));
        assert_eq!(zero, unit);
    }

    #[test]
    fn orthographic_frustum_keeps_visible_and_culls_offscreen() {
        // `half_height = 2` on 320×180: half-width ≈ 3.56 at any depth.
        let orbit = OrbitCamera::from_desc(&ortho_camera(Meters::new(2.0)));
        let (view_proj, _) = camera_view_projection(&orbit.view_parameters(), (320, 180));
        let frustum = Frustum::from_view_proj(&view_proj);
        assert!(frustum.sphere_visible(Vec3::ZERO, 0.5));
        assert!(frustum.sphere_visible(Vec3::new(0.0, 0.0, -50.0), 0.5));
        assert!(!frustum.sphere_visible(Vec3::new(10.0, 0.0, 0.0), 0.5));
        assert!(!frustum.sphere_visible(Vec3::new(0.0, -10.0, 0.0), 0.5));
        // Boundary: the half-width edge sits at ±hh*aspect ≈ ±3.56.
        assert!(frustum.sphere_visible(Vec3::new(3.9, 0.0, 0.0), 0.5));
        assert!(!frustum.sphere_visible(Vec3::new(4.2, 0.0, 0.0), 0.5));
    }

    #[test]
    fn orthographic_wheel_zooms_half_height_not_radius() {
        let mut orbit = OrbitCamera::from_desc(&ortho_camera(Meters::new(2.0)));
        let before = orbit.view_parameters();
        let eye_before = before.eye;
        let mut input = InputState::new();
        input.add_wheel_delta(100.0);
        orbit.apply_input(&input, 1.0);
        let after = orbit.view_parameters();
        // The orbit radius (hence the eye) stays put in ortho mode.
        assert_eq!(after.eye, eye_before);
        let half_before = ortho_half_height(&before);
        let half_after = ortho_half_height(&after);
        assert!(
            half_after.get() > half_before.get(),
            "{} vs {}",
            half_after.get(),
            half_before.get()
        );
        // The zoom clamps instead of running away.
        for _ in 0..100 {
            let mut input = InputState::new();
            input.add_wheel_delta(100.0);
            orbit.apply_input(&input, 1.0);
        }
        let clamped = ortho_half_height(&orbit.view_parameters());
        assert_eq!(clamped.get(), 1000.0);
        // The zoom clamps at the bottom instead of running away.
        let mut orbit = OrbitCamera::from_desc(&ortho_camera(Meters::new(2.0)));
        for _ in 0..100 {
            let mut input = InputState::new();
            input.add_wheel_delta(-100.0);
            orbit.apply_input(&input, 1.0);
        }
        let clamped = ortho_half_height(&orbit.view_parameters());
        assert_eq!(clamped.get(), 0.01);
        // Rotation still works in ortho mode.
        let mut orbit = OrbitCamera::from_desc(&ortho_camera(Meters::new(2.0)));
        let mut input = InputState::new();
        input.set_mouse_button(0, true);
        input.set_pointer_position([10.0, 4.0]);
        orbit.apply_input(&input, 1.0);
        assert_ne!(orbit.position(), eye_before);
    }
}
