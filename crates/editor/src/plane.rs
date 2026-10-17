//! Plane editing camera and viewport mode (PLAN §k, slice K2).
//!
//! 2D games are degenerate 3D: an orthographic camera over the XY sheet
//! (`z = 0`, looking along `-Z`, up `+Y`). [`PlaneCamera`] owns that view as
//! `center + half_height` state plus the viewport size, and drives both K2
//! phases through one [`PlaneProjection`]: [`PlaneProjection::PerspectiveStandIn`]
//! renders a top-down perspective until the K0a orthographic camera lands,
//! [`PlaneProjection::Orthographic`] renders the true ortho box. Either way
//! `pixels_per_unit = viewport_h / (2·hh)` on the sheet, so pan, zoom and
//! the grid agree by construction. The left button never moves this camera —
//! it selects (E1 click); pan is middle/right drag, zoom is the wheel around
//! the cursor.

use std::sync::Mutex;

use glam::{Vec2, Vec3};
use ornis_assets::scene::CameraProjection;
use ornis_core::{
    Degrees, Engine, InputState, Meters, MouseButton, Resources, System, SystemAccess,
};

use crate::picking::ViewportCamera;

/// Viewport mode: full 3D orbit or the 2D plane sheet (enum, never a bool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViewportMode {
    /// Orbit camera with perspective foreshortening (default).
    #[default]
    Perspective3d,
    /// [`PlaneCamera`] pan-and-zoom over the XY sheet.
    Plane2d,
}

impl ViewportMode {
    /// Whether the plane camera drives the viewport.
    #[must_use]
    pub fn is_plane(self) -> bool {
        matches!(self, Self::Plane2d)
    }
}

/// Projection backing a [`PlaneCamera`]: one state, two render phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlaneProjection {
    /// Phase A stand-in: top-down perspective (`-Z`, up `+Y`) until the K0a
    /// orthographic camera lands. The eye height derives from `hh` so the
    /// sheet scale matches the ortho phase exactly.
    PerspectiveStandIn {
        /// Vertical field of view of the top-down perspective.
        fov_y: Degrees,
    },
    /// Phase B: the K0a orthographic box; the eye stays at a fixed height
    /// above the sheet while `hh` sets the scale.
    Orthographic {
        /// Eye height above the sheet (scale comes from `half_height`).
        eye_height: Meters,
    },
}

/// 2D plane camera: `center + half_height` over the XY sheet.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneCamera {
    center: Vec2,
    half_height: Meters,
    projection: PlaneProjection,
    near: Meters,
    far: Meters,
    viewport_px: [f32; 2],
}

impl PlaneCamera {
    /// Narrowest accepted visible half-height in meters (K0a clamp).
    pub const MIN_HALF_HEIGHT: Meters = Meters(0.01);
    /// Widest accepted visible half-height in meters (K0a clamp).
    pub const MAX_HALF_HEIGHT: Meters = Meters(1000.0);
    /// Wheel zoom speed: same exponential as the orbit camera.
    const ZOOM_SPEED: f32 = 0.001;
    /// Floor for a derived ortho eye height (degenerate eye/target pairs
    /// sit this far above the sheet instead of collapsing).
    const MIN_EYE_HEIGHT: Meters = Meters(0.01);
    /// Exclusive upper bound of the stand-in field of view in degrees.
    const FOV_MAX_DEG_EXCLUSIVE: f32 = 180.0;
    /// Half-angle scale: the sheet half-height spans `tan(fov/2)` per unit
    /// of eye height.
    const HALF_ANGLE: f32 = 0.5;
    /// Minimum viewport dimension in pixels (degenerate surfaces rejected).
    const MIN_VIEWPORT_PX: f32 = 1.0;
    /// Placeholder fov carried in tuple frames for the orthographic phase
    /// (the tuple has no projection slot; picking ignores it).
    const ORTHO_TUPLE_FOV_DEG: f32 = 60.0;

    /// Checked constructor: `None` on a non-finite center, a non-finite or
    /// non-positive half-height, an invalid projection payload (fov outside
    /// `(0, 180)` degrees, non-finite/non-positive eye height), `near`
    /// non-finite or `<= 0`, `far <= near`, or a sub-pixel viewport.
    #[must_use]
    pub fn new(
        center: Vec2,
        half_height: Meters,
        projection: PlaneProjection,
        near: Meters,
        far: Meters,
        viewport_px: [f32; 2],
    ) -> Option<Self> {
        if !center.is_finite() {
            return None;
        }
        if !half_height.is_finite() || half_height.get() <= 0.0 {
            return None;
        }
        match projection {
            PlaneProjection::PerspectiveStandIn { fov_y } => {
                if !fov_y.is_finite()
                    || fov_y.get() <= 0.0
                    || fov_y.get() >= Self::FOV_MAX_DEG_EXCLUSIVE
                {
                    return None;
                }
            }
            PlaneProjection::Orthographic { eye_height } => {
                if !eye_height.is_finite() || eye_height.get() <= 0.0 {
                    return None;
                }
            }
        }
        if !near.is_finite() || near.get() <= 0.0 || !far.is_finite() || far.get() <= near.get() {
            return None;
        }
        let [w, h] = viewport_px;
        if !w.is_finite()
            || !h.is_finite()
            || w < Self::MIN_VIEWPORT_PX
            || h < Self::MIN_VIEWPORT_PX
        {
            return None;
        }
        Some(Self {
            center,
            half_height,
            projection,
            near,
            far,
            viewport_px,
        })
    }

    /// Enters 2D from a tuple orbit frame: `center = target.xy`,
    /// `hh = |eye − target|·tan(fov/2)`, near/far carried over.
    ///
    /// Kept while tuple callers exist; K0a sides prefer
    /// [`Self::from_camera_view`], which also carries the projection.
    #[must_use]
    pub fn from_view_parameters(
        view: (Vec3, Vec3, Vec3, f32, f32, f32),
        projection: PlaneProjection,
        viewport_px: [f32; 2],
    ) -> Option<Self> {
        let (eye, target, _, fov_deg, near, far) = view;
        let half_height = Self::orbit_half_height(eye, target, fov_deg)?;
        Self::new(
            target.truncate(),
            half_height,
            projection,
            Meters::new(near),
            Meters::new(far),
            viewport_px,
        )
    }

    /// Orbit-derived half-height: `|eye − target|·tan(fov/2)`. `None` on a
    /// degenerate eye/target pair.
    // qual:allow(iosp) reason: "single ?-propagated validation feeding one closed-form geometric formula (no branches, no mutation); splitting further would scatter the hh derivation across shims"
    fn orbit_half_height(eye: Vec3, target: Vec3, fov_deg: f32) -> Option<Meters> {
        let distance = Self::checked_eye_distance(eye, target)?;
        Some(Meters::new(
            distance * (fov_deg.to_radians() * Self::HALF_ANGLE).tan(),
        ))
    }

    /// Eye/target distance for an orbit entry: `None` on a non-finite eye,
    /// target or distance.
    fn checked_eye_distance(eye: Vec3, target: Vec3) -> Option<f32> {
        if !eye.is_finite() || !target.is_finite() {
            return None;
        }
        let distance = (eye - target).length();
        distance.is_finite().then_some(distance)
    }

    /// Enters 2D from explicit look-at fields plus the K0a projection.
    ///
    /// The live path is always the orthographic sheet: an asset ortho keeps
    /// its half-height, a perspective orbit derives `hh` from the fov
    /// (`|eye − target|·tan(fov/2)`), and the eye sits at the orbit distance
    /// above the sheet. The [`PlaneProjection::PerspectiveStandIn`] never
    /// reaches this seam — its fov-derived eye height leaves the near/far
    /// clip window at the `hh` zoom extremes and empties the viewport.
    // qual:allow(srp, max_parameters=7) reason: "RAT-16 mandates explicit-param constructors without builders; K0a seam carries the full look-at frame as values, like CameraDesc::try_new_units"
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn from_camera_view(
        eye: Vec3,
        target: Vec3,
        projection: CameraProjection,
        fov: Degrees,
        near: Meters,
        far: Meters,
        viewport_px: [f32; 2],
    ) -> Option<Self> {
        let half_height = match projection {
            CameraProjection::Perspective => Self::orbit_half_height(eye, target, fov.get())?,
            CameraProjection::Orthographic { half_height } => half_height,
        };
        let eye_height = Self::checked_eye_height(eye, target)?;
        Self::new(
            target.truncate(),
            half_height,
            PlaneProjection::Orthographic { eye_height },
            near,
            far,
            viewport_px,
        )
    }

    /// Eye height above the sheet for an ortho entry: `None` on a
    /// non-finite eye/target pair, floored at [`Self::MIN_EYE_HEIGHT`].
    fn checked_eye_height(eye: Vec3, target: Vec3) -> Option<Meters> {
        if !eye.is_finite() || !target.is_finite() {
            return None;
        }
        Some(Meters::new(
            (eye - target).length().max(Self::MIN_EYE_HEIGHT.get()),
        ))
    }

    /// Sheet center in world units (XY).
    // qual:allow(dry, boilerplate) reason: "RAT-16 API surface (spec-mandated accessor); read cross-crate by the ornis-wasm wire state"
    #[must_use]
    pub fn center(&self) -> Vec2 {
        self.center
    }

    /// Visible half-height on the sheet in world units.
    // qual:allow(dry, boilerplate) reason: "RAT-16 API surface (spec-mandated accessor); read cross-crate by the ornis-wasm wire state"
    #[must_use]
    pub fn half_height(&self) -> Meters {
        self.half_height
    }

    /// Updates the viewport size; `false` on a degenerate size (state
    /// unchanged).
    pub fn set_viewport_px(&mut self, viewport_px: [f32; 2]) -> bool {
        let [w, h] = viewport_px;
        if !w.is_finite()
            || !h.is_finite()
            || w < Self::MIN_VIEWPORT_PX
            || h < Self::MIN_VIEWPORT_PX
        {
            return false;
        }
        self.viewport_px = viewport_px;
        true
    }

    /// Screen pixels per world unit on the sheet (`viewport_h / (2·hh)`),
    /// identical in both phases by construction.
    #[must_use]
    pub fn pixels_per_unit(&self) -> f32 {
        self.viewport_px[1] / (2.0 * self.half_height.get())
    }

    /// Viewport pixel → sheet point (exact on `z = 0` in both phases: the
    /// stand-in eye height is derived from `hh`, so the perspective
    /// foreshortening cancels at the sheet).
    #[must_use]
    pub fn screen_to_plane(&self, pointer_px: [f32; 2]) -> Vec2 {
        let [w, h] = self.viewport_px;
        let ppu = self.pixels_per_unit();
        Vec2::new(
            self.center.x + (pointer_px[0] - w / 2.0) / ppu,
            self.center.y + (h / 2.0 - pointer_px[1]) / ppu,
        )
    }

    /// Sheet point → viewport pixel (inverse of [`Self::screen_to_plane`]).
    #[must_use]
    pub fn plane_to_screen(&self, point: Vec2) -> [f32; 2] {
        let [w, h] = self.viewport_px;
        let ppu = self.pixels_per_unit();
        [
            w / 2.0 + (point.x - self.center.x) * ppu,
            h / 2.0 - (point.y - self.center.y) * ppu,
        ]
    }

    /// Pans by a screen drag in pixels: the world follows the cursor
    /// (screen `y` points down, sheet `y` points up).
    pub fn pan_pixels(&mut self, delta_px: [f32; 2]) {
        let ppu = self.pixels_per_unit();
        self.center.x -= delta_px[0] / ppu;
        self.center.y += delta_px[1] / ppu;
    }

    /// Zooms around the cursor: the sheet point under `pointer_px` stays
    /// put. Same exponential speed as the orbit wheel
    /// (`exp(delta·0.001)`), clamped to the K0a `[0.01, 1000]` m range.
    pub fn zoom_at(&mut self, wheel_delta: f32, pointer_px: [f32; 2]) {
        if !wheel_delta.is_finite() {
            return;
        }
        let zoomed = self.half_height.get() * (wheel_delta * Self::ZOOM_SPEED).exp();
        let clamped = zoomed.clamp(Self::MIN_HALF_HEIGHT.get(), Self::MAX_HALF_HEIGHT.get());
        let before = self.screen_to_plane(pointer_px);
        self.half_height = Meters::new(clamped);
        let after = self.screen_to_plane(pointer_px);
        self.center += before - after;
    }

    /// Applies the shared plane input contract: middle/right drag pans,
    /// the wheel zooms around the cursor. The left button is ignored — in
    /// 2D it only selects (E1 click).
    pub fn apply_input(&mut self, input: &InputState) {
        if input.button_down(MouseButton::Middle) || input.button_down(MouseButton::Right) {
            self.pan_pixels(input.pointer_delta());
        }
        let wheel = input.wheel_delta();
        if wheel != 0.0 {
            self.zoom_at(wheel, input.pointer_position());
        }
    }

    /// Eye height above the sheet: derived from `hh` in the stand-in phase,
    /// the fixed `eye_height` in the ortho phase.
    fn eye_height(&self) -> f32 {
        match self.projection {
            PlaneProjection::PerspectiveStandIn { fov_y } => {
                self.half_height.get() / (fov_y.get().to_radians() / 2.0).tan()
            }
            PlaneProjection::Orthographic { eye_height } => eye_height.get(),
        }
    }

    /// Field of view for tuple frames: the stand-in fov, or a placeholder
    /// in the ortho phase (the tuple has no projection slot; picking
    /// ignores it there — see [`Self::camera_view_params`]).
    fn tuple_fov_deg(&self) -> f32 {
        match self.projection {
            PlaneProjection::PerspectiveStandIn { fov_y } => fov_y.get(),
            PlaneProjection::Orthographic { .. } => Self::ORTHO_TUPLE_FOV_DEG,
        }
    }

    /// The same tuple frame `OrbitCamera::view_parameters` used to publish
    /// (for the WASM `update_camera` and [`ViewportCamera`] seams): the eye
    /// sits at `(cx, cy, eye_height)` looking at `(cx, cy, 0)`, up `+Y`.
    #[must_use]
    pub fn view_parameters(&self) -> (Vec3, Vec3, Vec3, f32, f32, f32) {
        let eye = Vec3::new(self.center.x, self.center.y, self.eye_height());
        let target = Vec3::new(self.center.x, self.center.y, 0.0);
        (
            eye,
            target,
            Vec3::Y,
            self.tuple_fov_deg(),
            self.near.get(),
            self.far.get(),
        )
    }

    /// Look-at fields plus the K0a projection for the render
    /// `camera_view_projection` seam (phase B): `(eye, target, up,
    /// projection, fov, near, far)`. The WASM side builds the render
    /// `CameraView` from these without an `ornis-render` edge here.
    #[must_use]
    pub fn camera_view_params(
        &self,
    ) -> (Vec3, Vec3, Vec3, CameraProjection, Degrees, Meters, Meters) {
        let (eye, target, up, _, _, _) = self.view_parameters();
        let (projection, fov) = match self.projection {
            PlaneProjection::PerspectiveStandIn { fov_y } => (CameraProjection::Perspective, fov_y),
            PlaneProjection::Orthographic { .. } => (
                CameraProjection::Orthographic {
                    half_height: self.half_height,
                },
                Degrees::new(Self::ORTHO_TUPLE_FOV_DEG),
            ),
        };
        (eye, target, up, projection, fov, self.near, self.far)
    }

    /// Picking camera for this plane view: top-down perspective in phase A,
    /// true orthographic in phase B.
    #[must_use]
    pub fn viewport_camera(&self) -> Option<ViewportCamera> {
        let (eye, target, up, _, _, _) = self.view_parameters();
        match self.projection {
            PlaneProjection::PerspectiveStandIn { fov_y } => ViewportCamera::new(
                eye,
                target,
                up,
                fov_y.get(),
                self.near.get(),
                self.far.get(),
                self.viewport_px,
            ),
            PlaneProjection::Orthographic { .. } => ViewportCamera::new_orthographic(
                eye,
                target,
                up,
                self.half_height,
                self.near.get(),
                self.far.get(),
                self.viewport_px,
            ),
        }
    }
}

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
    use super::*;

    /// Stand-in rig: center origin, `hh = 5`, `800×600` viewport.
    fn camera() -> PlaneCamera {
        PlaneCamera::new(
            Vec2::ZERO,
            Meters::new(5.0),
            PlaneProjection::PerspectiveStandIn {
                fov_y: Degrees::new(60.0),
            },
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test camera")
    }

    /// Ortho rig: same framing, true orthographic projection.
    fn ortho_camera() -> PlaneCamera {
        PlaneCamera::new(
            Vec2::ZERO,
            Meters::new(5.0),
            PlaneProjection::Orthographic {
                eye_height: Meters::new(10.0),
            },
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid ortho test camera")
    }

    /// `pixels_per_unit` is `viewport_h / (2·hh)` in both phases.
    #[test]
    fn pixels_per_unit_matches_half_height() {
        assert_eq!(camera().pixels_per_unit(), 60.0);
        assert_eq!(ortho_camera().pixels_per_unit(), 60.0);
    }

    /// The tuple frame looks straight down `-Z` with up `+Y`; the stand-in
    /// eye height derives from `hh`, the ortho one is fixed.
    #[test]
    fn view_parameters_look_down_with_up_y() {
        let (eye, target, up, fov, near, far) = camera().view_parameters();
        let expected_height = 5.0 / (60.0f32.to_radians() * 0.5).tan();
        assert!((eye - Vec3::new(0.0, 0.0, expected_height)).length() < 1e-4);
        assert_eq!(target, Vec3::ZERO);
        assert_eq!(up, Vec3::Y);
        assert_eq!((fov, near, far), (60.0, 0.1, 100.0));
        let (ortho_eye, _, _, _, _, _) = ortho_camera().view_parameters();
        assert_eq!(ortho_eye, Vec3::new(0.0, 0.0, 10.0));
    }

    /// `screen_to_plane`/`plane_to_screen` round-trip, and the center pixel
    /// maps to the camera center.
    #[test]
    fn screen_plane_round_trip() {
        let camera = camera();
        assert_eq!(camera.screen_to_plane([400.0, 300.0]), Vec2::ZERO);
        for point in [Vec2::new(3.0, -2.0), Vec2::new(-7.5, 4.25)] {
            let back = camera.screen_to_plane(camera.plane_to_screen(point));
            assert!((back - point).length() < 1e-4, "{back:?} vs {point:?}");
        }
        // Corners: half-width = aspect·hh = (800/600)·5.
        let corner = camera.screen_to_plane([0.0, 0.0]);
        assert!((corner - Vec2::new(-20.0 / 3.0, 5.0)).length() < 1e-4);
    }

    /// `pan_pixels` moves the world with the cursor: `screen_to_plane`
    /// shifts by `(−dx, +dy)/ppu`.
    #[test]
    fn pan_moves_world_with_cursor() {
        let mut camera = camera();
        let before = camera.screen_to_plane([400.0, 300.0]);
        camera.pan_pixels([120.0, -30.0]);
        let after = camera.screen_to_plane([400.0, 300.0]);
        assert!((after - before - Vec2::new(-2.0, -0.5)).length() < 1e-5);
    }

    /// `zoom_at` keeps the sheet point under the cursor fixed and clamps
    /// `hh` to the K0a range.
    #[test]
    fn zoom_keeps_cursor_anchor_and_clamps() {
        let mut camera = camera();
        let anchor = [500.0, 200.0];
        let before = camera.screen_to_plane(anchor);
        camera.zoom_at(120.0, anchor);
        assert!((camera.screen_to_plane(anchor) - before).length() < 1e-4);
        assert!(camera.half_height().get() > 5.0);
        // Zoom out hard: clamps at 1000 m, the anchor still holds.
        for _ in 0..200 {
            camera.zoom_at(120.0, anchor);
        }
        assert_eq!(camera.half_height().get(), 1000.0);
        assert!((camera.screen_to_plane(anchor) - before).length() < 1e-3);
        // Zoom in hard: clamps at 0.01 m.
        for _ in 0..400 {
            camera.zoom_at(-120.0, anchor);
        }
        assert_eq!(camera.half_height().get(), 0.01);
        // Non-finite wheels are ignored.
        camera.zoom_at(f32::NAN, anchor);
        assert_eq!(camera.half_height().get(), 0.01);
    }

    /// `apply_input` pans on middle/right drag, zooms on the wheel, and
    /// ignores the left button (it selects, never moves the camera).
    #[test]
    fn apply_input_pans_and_zooms_but_ignores_left() {
        let mut camera = camera();
        let mut input = InputState::new();
        input.set_pointer_anchor([400.0, 300.0]);
        // Left drag: no motion.
        input.set_button(MouseButton::Left, true);
        input.set_pointer_position([500.0, 300.0]);
        camera.apply_input(&input);
        assert_eq!(camera.center(), Vec2::ZERO);
        input.set_button(MouseButton::Left, false);
        // Right drag of 60 px right: world moves 1 unit left at 60 ppu.
        input.set_button(MouseButton::Right, true);
        input.set_pointer_position([460.0, 300.0]);
        camera.apply_input(&input);
        assert!((camera.center() - Vec2::new(-1.0, 0.0)).length() < 1e-5);
        // Middle drag works the same (deltas accumulate without a frame
        // clear: [60, 0] carried over plus [0, 60] new).
        input.set_button(MouseButton::Right, false);
        input.set_button(MouseButton::Middle, true);
        input.set_pointer_position([460.0, 360.0]);
        camera.apply_input(&input);
        assert!((camera.center() - Vec2::new(-2.0, 1.0)).length() < 1e-5);
        // Wheel zooms around the cursor.
        input.set_button(MouseButton::Middle, false);
        let anchor = input.pointer_position();
        let before = camera.screen_to_plane(anchor);
        input.add_wheel_delta(100.0);
        camera.apply_input(&input);
        assert!(camera.half_height().get() > 5.0);
        assert!((camera.screen_to_plane(anchor) - before).length() < 1e-4);
    }

    /// Gate: `screen_to_plane` agrees with `pointer_on_plane` over the
    /// picking camera in both phases, corners included.
    #[test]
    fn screen_to_plane_agrees_with_pointer_on_plane() {
        use crate::picking::{WorkPlane, pointer_on_plane};
        for camera in [camera(), ortho_camera()] {
            let viewport = camera.viewport_camera().expect("picking camera");
            for px in [0.0, 200.0, 400.0, 799.0] {
                for py in [0.0, 150.0, 300.0, 599.0] {
                    let via_plane = camera.screen_to_plane([px, py]);
                    let via_pick =
                        pointer_on_plane(&viewport, [px, py], WorkPlane::XY).expect("plane hit");
                    assert!(
                        (via_pick.truncate() - via_plane).length() < 1e-3,
                        "{px},{py}: {via_pick:?} vs {via_plane:?}"
                    );
                }
            }
        }
    }

    /// `from_view_parameters` centers on `target.xy` with the orbit-derived
    /// half-height; `from_camera_view` always enters the orthographic sheet
    /// (asset half-height, or the fov-derived one from a perspective orbit).
    #[test]
    fn from_orbit_frames_keeps_scale() {
        let plane = PlaneCamera::from_view_parameters(
            (
                Vec3::new(1.0, 2.0, 9.0),
                Vec3::new(1.0, 2.0, 0.0),
                Vec3::Y,
                60.0,
                0.1,
                100.0,
            ),
            PlaneProjection::PerspectiveStandIn {
                fov_y: Degrees::new(60.0),
            },
            [800.0, 600.0],
        )
        .expect("plane from orbit");
        assert_eq!(plane.center(), Vec2::new(1.0, 2.0));
        let expected = 9.0 * (60.0f32.to_radians() * 0.5).tan();
        assert!((plane.half_height().get() - expected).abs() < 1e-4);

        let ortho = PlaneCamera::from_camera_view(
            Vec3::new(1.0, 2.0, 9.0),
            Vec3::new(1.0, 2.0, 0.0),
            CameraProjection::Orthographic {
                half_height: Meters::new(2.0),
            },
            Degrees::new(60.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("plane from ortho view");
        assert_eq!(ortho.half_height(), Meters::new(2.0));
        assert_eq!(ortho.center(), Vec2::new(1.0, 2.0));
        // A perspective orbit enters as the orthographic sheet with the
        // fov-derived half-height (the stand-in never reaches the live path).
        let from_perspective = PlaneCamera::from_camera_view(
            Vec3::new(1.0, 2.0, 9.0),
            Vec3::new(1.0, 2.0, 0.0),
            CameraProjection::Perspective,
            Degrees::new(60.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("plane from perspective orbit");
        assert_eq!(
            from_perspective.camera_view_params().3,
            CameraProjection::Orthographic {
                half_height: Meters::new(expected)
            }
        );
        // Degenerate inputs are rejected.
        assert!(
            PlaneCamera::new(
                Vec2::new(f32::NAN, 0.0),
                Meters::new(5.0),
                PlaneProjection::PerspectiveStandIn {
                    fov_y: Degrees::new(60.0)
                },
                Meters::new(0.1),
                Meters::new(100.0),
                [800.0, 600.0],
            )
            .is_none()
        );
        let mut rejects_viewport = camera();
        assert!(!rejects_viewport.set_viewport_px([0.0, 600.0]));
        assert!(rejects_viewport.set_viewport_px([1024.0, 768.0]));
        assert_eq!(rejects_viewport.viewport_px, [1024.0, 768.0]);
    }

    /// The input system only drives the camera in `Plane2d`; install is
    /// idempotent and entering 2D then leaving keeps the plane state.
    #[test]
    fn input_system_gates_on_mode_and_install_is_idempotent() {
        use std::sync::Mutex;
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

    /// `camera_view_params` agrees with `view_parameters` on the look-at
    /// frame and carries the K0a projection (stand-in: perspective with the
    /// fov; ortho: asset half-height).
    #[test]
    fn camera_view_params_match_tuple_frame() {
        use ornis_assets::scene::CameraProjection;
        let camera = camera();
        let (eye, target, up, projection, fov, near, far) = camera.camera_view_params();
        let (t_eye, t_target, t_up, t_fov, t_near, t_far) = camera.view_parameters();
        assert_eq!((eye, target, up), (t_eye, t_target, t_up));
        assert_eq!(projection, CameraProjection::Perspective);
        assert_eq!(fov, Degrees::new(60.0));
        assert_eq!(fov.get(), t_fov);
        assert_eq!((near.get(), far.get()), (t_near, t_far));
        let ortho = ortho_camera();
        let (_, _, _, projection, _, _, _) = ortho.camera_view_params();
        assert_eq!(
            projection,
            CameraProjection::Orthographic {
                half_height: Meters::new(5.0)
            }
        );
    }

    /// Live 2D entry (review RAT-16 blockers 1–2): `from_camera_view` from a
    /// perspective orbit yields the orthographic sheet on both seams, and
    /// picking keeps hitting `z = 0` at both `hh` zoom clamp edges — the eye
    /// height is the orbit distance, so the sheet never leaves the near/far
    /// window the way the fov-derived stand-in eye height did.
    #[test]
    fn live_entry_is_orthographic_and_picks_at_zoom_clamps() {
        use crate::picking::PickProjection;
        let mut camera = PlaneCamera::from_camera_view(
            Vec3::new(0.0, 0.0, 9.0),
            Vec3::ZERO,
            CameraProjection::Perspective,
            Degrees::new(60.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("live entry from a perspective orbit");
        assert_eq!(
            camera.camera_view_params().3,
            CameraProjection::Orthographic {
                half_height: Meters::new(9.0 * (60.0f32.to_radians() * 0.5).tan())
            }
        );
        assert!(matches!(
            camera
                .viewport_camera()
                .expect("picking camera")
                .projection(),
            PickProjection::Orthographic { .. }
        ));
        let center = [400.0, 300.0];
        // Zoom out to the 1000 m clamp, then in to the 0.01 m clamp.
        for _ in 0..200 {
            camera.zoom_at(120.0, center);
        }
        assert_eq!(camera.half_height().get(), 1000.0);
        assert_sheet_stays_pickable(&camera);
        for _ in 0..400 {
            camera.zoom_at(-120.0, center);
        }
        assert_eq!(camera.half_height().get(), 0.01);
        assert_sheet_stays_pickable(&camera);
    }

    /// Every viewport pixel resolves onto the sheet at both zoom clamps,
    /// matching [`PlaneCamera::screen_to_plane`] (the exactness the picking
    /// pipeline relies on), and the center hit stays inside the pick
    /// camera's near/far window.
    fn assert_sheet_stays_pickable(camera: &PlaneCamera) {
        use crate::picking::{WorkPlane, pointer_on_plane};
        let viewport = camera.viewport_camera().expect("picking camera");
        for px in [0.0, 400.0, 799.0] {
            for py in [0.0, 300.0, 599.0] {
                let via_pick =
                    pointer_on_plane(&viewport, [px, py], WorkPlane::XY).expect("plane hit");
                assert!((via_pick.z).abs() < 1e-4, "{px},{py}: z={}", via_pick.z);
                assert!(
                    (via_pick.truncate() - camera.screen_to_plane([px, py])).length() < 1e-3,
                    "{px},{py}: {via_pick:?}"
                );
            }
        }
        let center_hit =
            pointer_on_plane(&viewport, [400.0, 300.0], WorkPlane::XY).expect("center plane hit");
        let depth = (center_hit - viewport.position()).length();
        assert!(
            depth >= viewport.near() && depth <= viewport.far(),
            "center depth {depth} outside [{}, {}]",
            viewport.near(),
            viewport.far()
        );
    }
}
