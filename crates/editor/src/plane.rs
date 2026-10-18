//! Plane editing camera and viewport mode (PLAN §k, slice K2).
//!
//! 2D games are degenerate 3D: an orthographic camera over the XY sheet
//! (`z = 0`, looking along `-Z`, up `+Y`). [`PlaneCamera`] owns that view as
//! `center + half_height` state plus the viewport size. The K0a orthographic
//! camera is the only live projection (the phase-A perspective stand-in was
//! removed once `CameraProjection` landed in master): `pixels_per_unit =
//! viewport_h / (2·hh)` on the sheet, so pan, zoom and the grid agree by
//! construction. The left button never moves this camera — it selects (E1
//! click); pan is middle/right drag, zoom is the wheel around the cursor.

use glam::{Vec2, Vec3};
use ornis_assets::scene::CameraProjection;
use ornis_core::{Degrees, InputState, Meters, MouseButton};

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

/// 2D plane camera: `center + half_height` over the XY sheet.
///
/// The projection is always the K0a orthographic box: the eye sits at a
/// fixed `eye_height` above the sheet while `half_height` sets the scale.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneCamera {
    center: Vec2,
    half_height: Meters,
    eye_height: Meters,
    near: Meters,
    far: Meters,
    /// Current viewport size in pixels (the surface the sheet maps onto).
    pub viewport_px: [f32; 2],
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
    /// Fixed eye stand-off above the orbit target level for ortho entry.
    /// Ortho scale is `half_height`, so the eye height never affects
    /// framing — it only sets the clip window. The stand-off is
    /// independent of the orbit radius so geometry above the orbit (a
    /// raised target with a close orbit used to put the eye below the
    /// scene) stays inside the band; geometry above
    /// `max(target.z, 0) + ORTHO_EYE_STANDOFF − near` remains clipped
    /// until the scene bounding box is plumbed in (accepted K2 limit).
    const ORTHO_EYE_STANDOFF: Meters = Meters(100.0);
    /// Half-angle scale: the sheet half-height spans `tan(fov/2)` per unit
    /// of eye height.
    const HALF_ANGLE: f32 = 0.5;
    /// Minimum viewport dimension in pixels (degenerate surfaces rejected).
    const MIN_VIEWPORT_PX: f32 = 1.0;
    /// Placeholder fov carried in tuple frames for the orthographic phase
    /// (the tuple has no projection slot; picking ignores it).
    const ORTHO_TUPLE_FOV_DEG: f32 = 60.0;
    /// Clip-window factor: when the orbit-derived near/far would clip the
    /// sheet or the target level, shrink `near` to
    /// `ORTHO_EYE_STANDOFF·CLIP_MARGIN` (below both distances) and grow
    /// `far` to `eye_height·CLIP_MARGIN_INV` (above the sheet).
    const CLIP_MARGIN: f32 = 0.1;
    /// Reciprocal of [`Self::CLIP_MARGIN`] for the far-clip expansion.
    const CLIP_MARGIN_INV: f32 = 10.0;

    /// Checked constructor: `None` on a non-finite center, a non-finite or
    /// non-positive half-height or eye height, `near` non-finite or `<= 0`,
    /// `far <= near`, or a sub-pixel viewport.
    // qual:allow(srp, max_parameters=6) reason: "RAT-16 mandates explicit-param constructors without builders; the plane camera carries center, scale, eye height, clip and viewport as values"
    #[must_use]
    pub fn new(
        center: Vec2,
        half_height: Meters,
        eye_height: Meters,
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
        if !eye_height.is_finite() || eye_height.get() <= 0.0 {
            return None;
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
            eye_height,
            near,
            far,
            viewport_px,
        })
    }

    /// Orbit-derived half-height: `|eye − target|·tan(fov/2)`. `None` on a
    /// degenerate eye/target pair.
    fn orbit_half_height(eye: Vec3, target: Vec3, fov_deg: f32) -> Option<Meters> {
        Self::checked_eye_distance(eye, target)
            .map(|distance| Meters::new(distance * (fov_deg.to_radians() * Self::HALF_ANGLE).tan()))
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
    /// (`|eye − target|·tan(fov/2)`), and the derived `hh` is clamped to the
    /// K0a range. The eye sits at a fixed `ORTHO_EYE_STANDOFF`
    /// above `max(target.z, 0)` — independent of the orbit radius, so
    /// geometry above the orbit stays inside the clip window — and near/far
    /// are widened when needed to cover the sheet and the target level.
    ///
    /// Coverage is a fixed band, not the scene: geometry above
    /// `max(target.z, 0) + ORTHO_EYE_STANDOFF − near` still sits above the
    /// eye and stays clipped in 2D (an accepted K2 limitation — the scene
    /// bounding box never enters the eye-height derivation).
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
            CameraProjection::Orthographic { half_height } => {
                // Reject non-finite frames on the ortho path too.
                Self::checked_eye_distance(eye, target)?;
                half_height
            }
        };
        let half_height = Meters::new(
            half_height
                .get()
                .clamp(Self::MIN_HALF_HEIGHT.get(), Self::MAX_HALF_HEIGHT.get()),
        );
        let eye_height = Self::checked_eye_height(target);
        let (near, far) = Self::sheet_clip_window(near, far, eye_height);
        Self::new(
            target.truncate(),
            half_height,
            eye_height,
            near,
            far,
            viewport_px,
        )
    }

    /// Eye height above the sheet: a fixed [`Self::ORTHO_EYE_STANDOFF`]
    /// above the orbit target level `max(target.z, 0)`, floored at
    /// [`Self::MIN_EYE_HEIGHT`]. Independent of the orbit radius so a
    /// raised target with a close orbit keeps its geometry below the eye.
    fn checked_eye_height(target: Vec3) -> Meters {
        Meters::new(
            (target.z.max(0.0) + Self::ORTHO_EYE_STANDOFF.get()).max(Self::MIN_EYE_HEIGHT.get()),
        )
    }

    /// Widens the orbit near/far window when it would clip the sheet at
    /// distance `eye_height` or geometry just below the eye: `near` shrinks
    /// to a small fraction of the eye stand-off, `far` grows above the
    /// sheet, so `z = 0` and the target level always lie inside
    /// `[near, far]`.
    fn sheet_clip_window(near: Meters, far: Meters, eye_height: Meters) -> (Meters, Meters) {
        let near = near
            .get()
            .min(Self::ORTHO_EYE_STANDOFF.get() * Self::CLIP_MARGIN)
            .max(f32::EPSILON);
        let far = far.get().max(eye_height.get() * Self::CLIP_MARGIN_INV);
        (Meters::new(near), Meters::new(far))
    }

    /// Sheet center in world units (XY).
    #[must_use]
    pub fn center(&self) -> Vec2 {
        self.center
    }

    /// Visible half-height on the sheet in world units.
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

    /// Screen pixels per world unit on the sheet (`viewport_h / (2·hh)`).
    #[must_use]
    pub fn pixels_per_unit(&self) -> f32 {
        self.viewport_px[1] / (2.0 * self.half_height.get())
    }

    /// Viewport pixel → sheet point (exact on `z = 0`: the ortho rays are
    /// parallel and the projection is linear).
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

    /// Eye height above the sheet (fixed at construction).
    fn eye_height(&self) -> f32 {
        self.eye_height.get()
    }

    /// The same tuple frame `OrbitCamera::view_parameters` used to publish
    /// (for the WASM `update_camera` and [`ViewportCamera`] seams): the eye
    /// sits at `(cx, cy, eye_height)` looking at `(cx, cy, 0)`, up `+Y`.
    /// The fov is a placeholder (the ortho projection ignores it).
    #[must_use]
    pub fn view_parameters(&self) -> (Vec3, Vec3, Vec3, f32, f32, f32) {
        let eye = Vec3::new(self.center.x, self.center.y, self.eye_height());
        let target = Vec3::new(self.center.x, self.center.y, 0.0);
        (
            eye,
            target,
            Vec3::Y,
            Self::ORTHO_TUPLE_FOV_DEG,
            self.near.get(),
            self.far.get(),
        )
    }

    /// Look-at fields plus the K0a projection for the render
    /// `camera_view_projection` seam: `(eye, target, up, projection, fov,
    /// near, far)`. The WASM side builds the render `CameraView` from these
    /// without an `ornis-render` edge here.
    #[must_use]
    pub fn camera_view_params(
        &self,
    ) -> (Vec3, Vec3, Vec3, CameraProjection, Degrees, Meters, Meters) {
        let (eye, target, up, _, _, _) = self.view_parameters();
        (
            eye,
            target,
            up,
            CameraProjection::Orthographic {
                half_height: self.half_height,
            },
            Degrees::new(Self::ORTHO_TUPLE_FOV_DEG),
            self.near,
            self.far,
        )
    }

    /// Picking camera for this plane view: true orthographic over the sheet.
    #[must_use]
    pub fn viewport_camera(&self) -> Option<ViewportCamera> {
        let (eye, target, up, _, _, _) = self.view_parameters();
        ViewportCamera::new_orthographic(
            eye,
            target,
            up,
            self.half_height,
            self.near.get(),
            self.far.get(),
            self.viewport_px,
        )
    }
}

// Engine-resource install/read and the 2D input frame system live in
// `crate::plane_input` (idempotent install, `plane_camera_input` schedule
// entry, pan/zoom gated on `ViewportMode::Plane2d`).

#[cfg(test)]
mod tests {
    use super::*;

    /// Ortho rig: center origin, `hh = 5`, eye at 10, `800×600` viewport.
    fn camera() -> PlaneCamera {
        PlaneCamera::new(
            Vec2::ZERO,
            Meters::new(5.0),
            Meters::new(10.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test camera")
    }

    /// `pixels_per_unit` is `viewport_h / (2·hh)`.
    #[test]
    fn pixels_per_unit_matches_half_height() {
        assert_eq!(camera().pixels_per_unit(), 60.0);
    }

    /// The tuple frame looks straight down `-Z` with up `+Y`; the eye sits
    /// at the fixed `eye_height`.
    #[test]
    fn view_parameters_look_down_with_up_y() {
        let (eye, target, up, fov, near, far) = camera().view_parameters();
        assert_eq!(eye, Vec3::new(0.0, 0.0, 10.0));
        assert_eq!(target, Vec3::ZERO);
        assert_eq!(up, Vec3::Y);
        assert_eq!((fov, near, far), (60.0, 0.1, 100.0));
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
    /// picking camera, corners included.
    #[test]
    fn screen_to_plane_agrees_with_pointer_on_plane() {
        use crate::picking::{WorkPlane, pointer_on_plane};
        let camera = camera();
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

    /// `from_camera_view` centers on `target.xy`: an asset ortho keeps its
    /// half-height (clamped), a perspective orbit derives it from the fov.
    #[test]
    fn from_orbit_frames_keeps_scale() {
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
        // fov-derived half-height.
        let expected = 9.0 * (60.0f32.to_radians() * 0.5).tan();
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
        assert_eq!(from_perspective.center(), Vec2::new(1.0, 2.0));
        assert!((from_perspective.half_height().get() - expected).abs() < 1e-4);
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
                Meters::new(10.0),
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

    /// `camera_view_params` agrees with `view_parameters` on the look-at
    /// frame and always carries the K0a orthographic projection.
    #[test]
    fn camera_view_params_match_tuple_frame() {
        let camera = camera();
        let (eye, target, up, projection, fov, near, far) = camera.camera_view_params();
        let (t_eye, t_target, t_up, t_fov, t_near, t_far) = camera.view_parameters();
        assert_eq!((eye, target, up), (t_eye, t_target, t_up));
        assert_eq!(
            projection,
            CameraProjection::Orthographic {
                half_height: Meters::new(5.0)
            }
        );
        assert_eq!(fov, Degrees::new(60.0));
        assert_eq!(fov.get(), t_fov);
        assert_eq!((near.get(), far.get()), (t_near, t_far));
    }

    /// Live 2D entry: `from_camera_view` from a perspective orbit yields
    /// the orthographic sheet on both seams, and picking keeps hitting
    /// `z = 0` at both `hh` zoom clamp edges.
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
    /// matching [`PlaneCamera::screen_to_plane`], and the center hit stays
    /// inside the pick camera's near/far window.
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

    /// Shared entry for the fixed-stand-off tests: a perspective orbit
    /// at the given eye/target heights over a flat sheet.
    fn perspective_orbit(eye_z: f32, target_z: f32) -> PlaneCamera {
        PlaneCamera::from_camera_view(
            Vec3::new(0.0, 0.0, eye_z),
            Vec3::new(0.0, 0.0, target_z),
            CameraProjection::Perspective,
            Degrees::new(60.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("perspective orbit entry")
    }

    /// Review blocker (Editor track): the eye sits at a fixed stand-off
    /// above the target level, not at the orbit radius, so geometry above
    /// a close orbit stays inside the clip window and under the pick ray.
    #[test]
    fn fixed_eye_height_covers_geometry_above_close_orbit() {
        // (а) target.z = 0, orbit radius 0.5: the eye is at the fixed
        // stand-off (100), not at 0.5 — an object at z = 2 that used to
        // sit behind the eye is now in front of it, in-window.
        let camera = perspective_orbit(0.5, 0.0);
        let (eye, _, _, _, _, near, far) = camera.camera_view_params();
        assert!(
            (eye.z - PlaneCamera::ORTHO_EYE_STANDOFF.get()).abs() < 1e-3,
            "eye.z = {} (fixed stand-off, not the 0.5 radius)",
            eye.z
        );
        // Geometry at z = 2 lies between the eye and the sheet, in-window.
        let above_depth = (Vec3::new(0.0, 0.0, 2.0) - eye).length();
        assert!(
            above_depth >= near.get() && above_depth <= far.get(),
            "object at z=2 depth {above_depth} outside [{}, {}]",
            near.get(),
            far.get()
        );
        // The same object is pickable: the ortho ray through its screen
        // position (center) carries it inside the pick window.
        let viewport = camera.viewport_camera().expect("picking camera");
        let hit = crate::picking::pointer_on_plane(
            &viewport,
            [400.0, 300.0],
            crate::picking::WorkPlane::XY,
        )
        .expect("sheet hit");
        let point_depth = (Vec3::new(hit.x, hit.y, 2.0) - viewport.position()).length();
        assert!(
            point_depth >= viewport.near() && point_depth <= viewport.far(),
            "picked z=2 point depth {point_depth} outside [{}, {}]",
            viewport.near(),
            viewport.far()
        );
    }

    /// Review blocker (Editor track): a raised target with a close orbit
    /// keeps the target level inside the clip window — the eye stands a
    /// fixed stand-off above `target.z`, never a radius-length behind it.
    #[test]
    fn raised_target_close_orbit_keeps_target_in_clip_window() {
        // (б) target.z = 5, orbit radius 0.05: the eye is at 5 + 100 = 105,
        // so the object at z = 5 is a full stand-off in front of it — in
        // the window even though the radius is far smaller than the input
        // near (0.1).
        let camera = perspective_orbit(5.05, 5.0);
        let sheet_height = camera.eye_height();
        let (eye, _, _, _, _, near, far) = camera.camera_view_params();
        assert!(
            (eye.z - (5.0 + PlaneCamera::ORTHO_EYE_STANDOFF.get())).abs() < 1e-3,
            "eye.z = {} (target.z + stand-off, not 5.05)",
            eye.z
        );
        let target = Vec3::new(0.0, 0.0, 5.0);
        let depth = (target - eye).length();
        assert!(
            depth >= near.get() && depth <= far.get(),
            "target depth {depth} outside [{}, {}]",
            near.get(),
            far.get()
        );
        // The sheet below stays covered too (sheet depth = eye height).
        assert!(
            sheet_height >= near.get() && sheet_height <= far.get(),
            "sheet depth {sheet_height} outside [{}, {}]",
            near.get(),
            far.get()
        );
    }

    /// An orbit with `|eye − target| < near` (degenerate close orbit)
    /// still lands the sheet inside the widened `[near, far]` window.
    #[test]
    fn close_orbit_entry_widens_clip_window_to_cover_sheet() {
        let camera = PlaneCamera::from_camera_view(
            Vec3::new(0.0, 0.0, 0.05),
            Vec3::ZERO,
            CameraProjection::Perspective,
            Degrees::new(60.0),
            Meters::new(0.1),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("close orbit entry");
        let (_, _, _, _, _, near, far) = camera.camera_view_params();
        let eye_height = camera.eye_height();
        assert!(
            near.get() < eye_height && far.get() > eye_height,
            "sheet at {eye_height} outside [{}, {}]",
            near.get(),
            far.get()
        );
    }

    /// The derived `hh` is clamped to the K0a range on entry.
    #[test]
    fn derived_half_height_is_clamped_on_entry() {
        // A very wide fov around a far orbit would exceed MAX_HALF_HEIGHT.
        let wide = PlaneCamera::from_camera_view(
            Vec3::new(0.0, 0.0, 9000.0),
            Vec3::ZERO,
            CameraProjection::Perspective,
            Degrees::new(179.0),
            Meters::new(0.1),
            Meters::new(100000.0),
            [800.0, 600.0],
        )
        .expect("wide entry");
        assert_eq!(wide.half_height(), PlaneCamera::MAX_HALF_HEIGHT);
        // A tiny distance / fov would fall below MIN_HALF_HEIGHT.
        let tiny = PlaneCamera::from_camera_view(
            Vec3::new(0.0, 0.0, 0.001),
            Vec3::ZERO,
            CameraProjection::Perspective,
            Degrees::new(0.01),
            Meters::new(0.0001),
            Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("tiny entry");
        assert_eq!(tiny.half_height(), PlaneCamera::MIN_HALF_HEIGHT);
    }
}
