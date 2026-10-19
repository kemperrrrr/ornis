//! Viewport mode switching, picking-camera publication and the grid
//! overlay wire state (K2 plane editing).
//!
//! The render loop owns the replica engine, but [`set_viewport_mode`] is
//! called from JS outside the frame closure. Both run on the single
//! browser thread, so the export only queues a mode request and the loop
//! applies it on the next frame via [`apply_mode_request`] — no shared
//! engine pointer, no unsafe. The frame loop then publishes the picking
//! camera before the schedule ([`sync_picking_camera`]) and refreshes the
//! overlay JSON after it ([`refresh_viewport_state_json`]).

use std::cell::{Cell, RefCell};
use std::sync::Mutex;

use ornis_core::{Engine, InputState, PositiveF32};
use ornis_editor::{
    PlaneCamera, PlaneGrid, ViewportMode, install_plane_camera, plane_grid, read_plane_camera,
};
use ornis_render::{OrbitCamera, install_orbit_camera, read_orbit_camera};
use wasm_bindgen::prelude::*;

thread_local! {
    /// Active viewport mode (perspective orbit or 2D plane sheet).
    pub(crate) static VIEWPORT_MODE: Cell<ViewportMode> =
        const { Cell::new(ViewportMode::Perspective3d) };
    /// Pending mode request from `set_viewport_mode`, applied by the loop.
    pub(crate) static PENDING_MODE: Cell<Option<ModeRequest>> = const { Cell::new(None) };
    /// Orbit camera saved on entry into 2D, restored on exit.
    static SAVED_ORBIT: RefCell<Option<OrbitCamera>> = const { RefCell::new(None) };
    /// Latest viewport-state wire JSON, refreshed every frame.
    static LAST_STATE_JSON: RefCell<String> =
        const { RefCell::new(String::new()) };
}

/// Mode request queued by [`set_viewport_mode`] for the frame loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModeRequest {
    /// Enter the 2D plane sheet (saves the orbit camera).
    Plane2d,
    /// Return to the perspective orbit (restores the saved camera).
    Perspective3d,
}

impl ModeRequest {
    /// Wire name of the requested mode, for the frame-loop switch log.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Plane2d => "plane2d",
            Self::Perspective3d => "perspective3d",
        }
    }
}

/// Minimum grid step on screen for the overlay wire state.
const MIN_GRID_STEP_PX: f32 = 48.0;

/// Wire state for the grid overlay and cursor readout, shared by the live
/// `editor/` overlay and the `editor-v2` mock (`applyViewState` accepts this
/// exact shape). In `perspective3d` the plane fields are all `null`.
#[derive(serde::Serialize)]
struct ViewportStateWire {
    /// `"perspective3d"` or `"plane2d"`.
    mode: &'static str,
    /// Sheet center in world units (2D only).
    center: Option<[f32; 2]>,
    /// Visible half-height in world units (2D only).
    half_height: Option<f32>,
    /// Screen pixels per world unit on the sheet (2D only).
    pixels_per_unit: Option<f32>,
    /// Grid numbers for the overlay (2D only).
    grid: Option<PlaneGrid>,
    /// Sheet point under the cursor (2D only).
    pointer_world: Option<[f32; 2]>,
    /// Device pixel ratio the overlay must divide by (Rust-clamped to
    /// `>= 1`), so the grid math and the backing-store size share one
    /// value at every display scale — including browser zoom below 100%.
    dpr: f32,
}

/// Builds the [`ViewportStateWire`] JSON: pure over the mode, an optional
/// plane camera, an optional cursor and the display scale, so host tests
/// pin the shape without a browser.
fn build_viewport_state_json(
    mode: ViewportMode,
    plane: Option<&PlaneCamera>,
    pointer_px: Option<[f32; 2]>,
    dpr: f32,
) -> String {
    // Active only in `plane2d` once a plane camera exists (before the first
    // frame it is still `perspective3d` with null plane fields).
    let camera = if mode.is_plane() { plane } else { None };
    let state = ViewportStateWire {
        mode: if camera.is_some() {
            "plane2d"
        } else {
            "perspective3d"
        },
        center: camera.map(|camera| camera.center().to_array()),
        half_height: camera.map(|camera| camera.half_height().get()),
        pixels_per_unit: camera.map(PlaneCamera::pixels_per_unit),
        grid: camera.map(|camera| plane_grid(camera, PositiveF32::expect_valid(MIN_GRID_STEP_PX))),
        pointer_world: camera
            .zip(pointer_px)
            .map(|(camera, pointer)| camera.screen_to_plane(pointer).to_array()),
        dpr,
    };
    serde_json::to_string(&state).unwrap_or_else(|_| "{\"mode\":\"perspective3d\"}".into())
}

/// Queues a viewport mode switch for the frame loop: orbit
/// (`"perspective3d"`) or the 2D plane sheet (`"plane2d"`). Returns `false`
/// on an unknown mode.
///
/// Entering 2D saves the orbit camera and derives the plane camera from its
/// current view (K0a projection included); leaving restores the saved orbit
/// (`install_orbit_camera` replaces the camera without a second system).
/// The loop applies the request on the next frame, where the replica engine
/// lives — this export never touches the engine directly.
// qual:api — wasm-bindgen export; called from the browser page, invisible to the Rust call graph.
#[wasm_bindgen]
pub fn set_viewport_mode(mode: &str) -> bool {
    match mode {
        "plane2d" => {
            PENDING_MODE.with(|cell| cell.set(Some(ModeRequest::Plane2d)));
            true
        }
        "perspective3d" => {
            PENDING_MODE.with(|cell| cell.set(Some(ModeRequest::Perspective3d)));
            true
        }
        _ => false,
    }
}

/// Applies a viewport [`ModeRequest`] against an engine: entering 2D saves
/// the orbit camera and derives the plane camera from its current view at
/// `surface_px` (K0a projection included), leaving restores the saved orbit
/// (`install_orbit_camera` replaces the camera without a second system).
/// Pure engine wiring (no DOM/GPU), so host tests pin the switch. Returns
/// whether the switch applied (entering without an orbit camera no-ops;
/// re-entering 2D while already in 2D is a no-op so the saved orbit is
/// never overwritten by a rotated one).
pub(crate) fn apply_mode_request(
    engine: &mut Engine,
    request: ModeRequest,
    surface_px: (u32, u32),
) -> bool {
    match request {
        ModeRequest::Plane2d => {
            // Early exit: already in 2D — do not overwrite SAVED_ORBIT
            // with the (possibly rotated) hidden orbit camera.
            if VIEWPORT_MODE.with(|cell| cell.get()).is_plane() {
                return false;
            }
            let Some(orbit) = read_orbit_camera(engine) else {
                return false;
            };
            let view = orbit.view_parameters();
            let plane = PlaneCamera::from_camera_view(
                view.eye,
                view.target,
                view.projection,
                view.fov,
                view.near,
                view.far,
                [surface_px.0 as f32, surface_px.1 as f32],
            );
            let Some(plane) = plane else { return false };
            SAVED_ORBIT.with(|saved| *saved.borrow_mut() = Some(orbit));
            install_plane_camera(engine, plane);
            let _ = engine.world_mut().insert(ViewportMode::Plane2d);
            // Remove the orbit camera resource so `orbit_camera_input`
            // no-ops in 2D (left-drag and the wheel must not rotate the
            // hidden orbit camera).
            let _ = engine
                .world_mut()
                .resources_mut()
                .remove::<Mutex<OrbitCamera>>();
            VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Plane2d));
            true
        }
        ModeRequest::Perspective3d => {
            if let Some(saved) = SAVED_ORBIT.with(|saved| saved.borrow_mut().take()) {
                install_orbit_camera(engine, saved);
            }
            let _ = engine.world_mut().insert(ViewportMode::Perspective3d);
            VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Perspective3d));
            true
        }
    }
}

/// Latest viewport-state wire JSON for the grid overlay and cursor
/// readout, refreshed by the frame loop every frame.
// qual:api — wasm-bindgen export; called from the browser overlay, invisible to the Rust call graph.
#[wasm_bindgen]
pub fn viewport_state_json() -> String {
    let cached = LAST_STATE_JSON.with(|cell| cell.borrow().clone());
    if cached.is_empty() {
        build_viewport_state_json(
            ViewportMode::Perspective3d,
            None,
            None,
            crate::device_pixel_ratio(),
        )
    } else {
        cached
    }
}

/// Pure plan for one viewport sync: given the mode, an optional plane
/// camera, an optional orbit camera, the surface size and the pointer
/// position and the display scale, returns the picking camera to publish
/// (if any) and the wire JSON to cache. No DOM/GPU/engine state, so host
/// tests pin the switch including the 3D-without-orbit edge (JSON resets
/// to `perspective3d`).
fn viewport_sync_plan(
    mode: ViewportMode,
    plane: Option<&PlaneCamera>,
    orbit: Option<&OrbitCamera>,
    size_px: (u32, u32),
    pointer_px: Option<[f32; 2]>,
    dpr: f32,
) -> (Option<ornis_editor::ViewportCamera>, String) {
    let size = [size_px.0 as f32, size_px.1 as f32];
    if mode.is_plane() {
        let picking = plane.and_then(PlaneCamera::viewport_camera);
        let json = build_viewport_state_json(mode, plane, pointer_px, dpr);
        (picking, json)
    } else {
        let picking = orbit.and_then(|orbit| {
            let view = orbit.view_parameters();
            ornis_editor::ViewportCamera::from_camera_view(
                view.eye,
                view.target,
                view.up,
                view.projection,
                view.fov,
                view.near,
                view.far,
                size,
            )
        });
        // Always refresh the JSON in 3D — including without an orbit
        // camera, so a previous plane2d overlay never sticks.
        let json = build_viewport_state_json(mode, None, None, dpr);
        (picking, json)
    }
}

/// Pure plan for the sheet-viewport size sync: in `plane2d` the plane
/// camera's pixel viewport tracks the surface (`handle_resize` and the
/// entering size from `set_viewport_mode` both land here). Returns the
/// viewport to write, or `None` when nothing must change (3D mode, no
/// plane camera, or the viewport already matches `size_px`). No engine or
/// DOM state, so host tests pin the branch.
fn plane_viewport_sync_plan(
    mode: ViewportMode,
    plane: Option<&PlaneCamera>,
    size_px: (u32, u32),
) -> Option<[f32; 2]> {
    if !mode.is_plane() {
        return None;
    }
    plane.and_then(|plane| {
        let size = [size_px.0 as f32, size_px.1 as f32];
        (!plane.viewport_matches(size)).then_some(size)
    })
}

/// Publishes the mode resource and the picking camera for the current
/// frame: runs BEFORE the schedule so the frame's pick systems see this
/// frame's camera (and the plane camera's pixel viewport tracks the
/// surface in 2D). Resource and thread-local I/O only — every decision
/// lives in the pure [`plane_viewport_sync_plan`] / [`viewport_sync_plan`]
/// seams, and both halves are driven by one host test on a bare engine.
///
/// The companion [`refresh_viewport_state_json`] caches the wire JSON for
/// the overlay AFTER the schedule, once pan/zoom input has been applied;
/// publishing it here would leave the overlay one frame behind. The
/// selection log is a separate browser edge, called by the frame loop
/// right after this one.
///
/// `replace_scene` (via `FrameState::apply_live_scene`) only destroys
/// scene entities and republishes lights: the `ViewportCamera` resource
/// and the `Selected`/`Hovered` lanes survive a live update (marks on
/// removed entities vanish with them; replacements start unselected).
pub(crate) fn sync_picking_camera(engine: &mut Engine, size: (u32, u32)) {
    let mode = VIEWPORT_MODE.with(|cell| cell.get());
    let _ = engine.world_mut().insert(mode);
    let plane = read_plane_camera(engine);
    if let Some(viewport_px) = plane_viewport_sync_plan(mode, plane.as_ref(), size)
        && let Some(mutex) = engine.world().resources().get::<Mutex<PlaneCamera>>()
    {
        mutex
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_viewport_px(viewport_px);
    }
    let plane = read_plane_camera(engine);
    let orbit = read_orbit_camera(engine);
    let (picking, _) = viewport_sync_plan(
        mode,
        plane.as_ref(),
        orbit.as_ref(),
        size,
        None,
        crate::device_pixel_ratio(),
    );
    match picking {
        Some(picking) => {
            let _ = engine.world_mut().insert(picking);
        }
        None => {
            // 3D without an orbit camera (or before the first camera
            // exists): drop a stale ortho sheet camera so picking never
            // keeps the 2D view after leaving `plane2d`.
            let _ = engine
                .world_mut()
                .resources_mut()
                .remove::<ornis_editor::ViewportCamera>();
        }
    }
}

/// Refreshes the wire JSON cache for the grid overlay AFTER the schedule
/// ran: pan/zoom input mutates the plane camera inside `frame(DT)`, so
/// reading the cameras here makes the overlay track the current frame
/// instead of lagging one behind. Resource and thread-local I/O only —
/// the JSON content comes from the pure [`viewport_sync_plan`] seam,
/// host-tested together with [`sync_picking_camera`].
pub(crate) fn refresh_viewport_state_json(engine: &mut Engine, size: (u32, u32)) {
    let mode = VIEWPORT_MODE.with(|cell| cell.get());
    let plane = read_plane_camera(engine);
    let orbit = read_orbit_camera(engine);
    let pointer = engine
        .world()
        .resources()
        .get::<InputState>()
        .map(|input| input.pointer_position());
    let (_, json) = viewport_sync_plan(
        mode,
        plane.as_ref(),
        orbit.as_ref(),
        size,
        pointer,
        crate::device_pixel_ratio(),
    );
    LAST_STATE_JSON.with(|cell| {
        *cell.borrow_mut() = json;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn viewport_state_wire_matches_mock_shape() {
        // Perspective: the plane fields are all null (the editor-v2 mock
        // `applyViewState` keys off this exact shape).
        let v: serde_json::Value = serde_json::from_str(&build_viewport_state_json(
            ViewportMode::Perspective3d,
            None,
            None,
            1.0,
        ))
        .expect("wire is JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
        assert_eq!(v["dpr"], serde_json::json!(1.0));
        for field in [
            "center",
            "half_height",
            "pixels_per_unit",
            "grid",
            "pointer_world",
        ] {
            assert!(v[field].is_null(), "{field} must be null, got {}", v[field]);
        }
        // Plane: the grid and cursor ride the same JSON the live overlay
        // reads in `editor/viewport.js`.
        let plane = PlaneCamera::new(
            glam::Vec2::ZERO,
            ornis_core::Meters::new(5.0),
            ornis_core::Meters::new(10.0),
            ornis_core::Meters::new(0.1),
            ornis_core::Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test plane camera");
        let v: serde_json::Value = serde_json::from_str(&build_viewport_state_json(
            ViewportMode::Plane2d,
            Some(&plane),
            Some([400.0, 300.0]),
            2.0,
        ))
        .expect("wire is JSON");
        assert_eq!(v["mode"], serde_json::json!("plane2d"));
        assert_eq!(v["center"], serde_json::json!([0.0, 0.0]));
        assert_eq!(v["half_height"], serde_json::json!(5.0));
        assert_eq!(v["pixels_per_unit"], serde_json::json!(60.0));
        assert_eq!(v["grid"]["minor_step"], serde_json::json!(1.0));
        assert_eq!(v["grid"]["major_every"], serde_json::json!(10));
        assert_eq!(v["grid"]["origin_px"], serde_json::json!([400.0, 300.0]));
        assert_eq!(v["pointer_world"], serde_json::json!([0.0, 0.0]));
        // The display scale rides the wire verbatim: the overlay must
        // divide by the same value the backing store multiplied with
        // (parity test for a fractional/low devicePixelRatio).
        assert_eq!(v["dpr"], serde_json::json!(2.0));
        // No plane camera yet (before the first frame): nulls, plane flag
        // notwithstanding.
        let v: serde_json::Value = serde_json::from_str(&build_viewport_state_json(
            ViewportMode::Plane2d,
            None,
            None,
            1.0,
        ))
        .expect("wire is JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
    }

    #[test]
    fn viewport_mode_export_accepts_known_modes_only() {
        // Host-safe: queue only, no engine access (the loop applies it).
        assert!(set_viewport_mode("plane2d"));
        assert!(set_viewport_mode("perspective3d"));
        assert!(!set_viewport_mode("orbit"));
        assert!(!set_viewport_mode(""));
        // A queued request does not rewrite the cached wire JSON by itself.
        PENDING_MODE.with(|cell| cell.set(None));
    }

    #[test]
    fn viewport_state_json_export_serves_cached_or_default_shape() {
        LAST_STATE_JSON.with(|cell| cell.borrow_mut().clear());
        let v: serde_json::Value =
            serde_json::from_str(&viewport_state_json()).expect("wire is JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
        LAST_STATE_JSON.with(|cell| {
            *cell.borrow_mut() = build_viewport_state_json(
                ViewportMode::Perspective3d,
                None,
                None,
                crate::device_pixel_ratio(),
            );
        });
        assert_eq!(
            viewport_state_json(),
            build_viewport_state_json(
                ViewportMode::Perspective3d,
                None,
                None,
                crate::device_pixel_ratio(),
            )
        );
        LAST_STATE_JSON.with(|cell| cell.borrow_mut().clear());
    }

    #[test]
    fn mode_request_enters_and_leaves_plane() {
        VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Perspective3d));
        SAVED_ORBIT.with(|cell| *cell.borrow_mut() = None);
        let mut engine = Engine::new();
        // Entering without an orbit camera no-ops.
        assert!(!apply_mode_request(
            &mut engine,
            ModeRequest::Plane2d,
            (800, 600)
        ));
        install_orbit_camera(
            &mut engine,
            OrbitCamera::looking_at(glam::Vec3::new(0.0, 0.0, 9.0), glam::Vec3::ZERO),
        );
        assert!(apply_mode_request(
            &mut engine,
            ModeRequest::Plane2d,
            (800, 600)
        ));
        assert_eq!(
            engine.world().resources().get::<ViewportMode>().copied(),
            Some(ViewportMode::Plane2d)
        );
        let plane = read_plane_camera(&engine).expect("plane camera installed");
        assert!((plane.center() - glam::Vec2::ZERO).length() < 1e-4);
        // Review RAT-16 blocker 1: entering from the perspective orbit
        // installs the orthographic sheet (fov-derived half-height), never
        // the perspective stand-in.
        let (_, _, _, projection, _, near, far) = plane.camera_view_params();
        let expected_hh = 9.0 * (60.0f32.to_radians() * 0.5).tan();
        assert_eq!(
            projection,
            ornis_assets::scene::CameraProjection::Orthographic {
                half_height: ornis_core::Meters::new(expected_hh)
            }
        );
        assert!((plane.half_height().get() - expected_hh).abs() < 1e-4);
        assert_eq!(near.get(), 0.1);
        // far grows to eye_height·CLIP_MARGIN_INV: the eye sits at the
        // fixed stand-off (100) above the sheet, so the sheet covers.
        assert_eq!(far.get(), 1000.0);
        // Re-entering while already in 2D is a no-op: the early exit must
        // protect SAVED_ORBIT from being overwritten by a hidden orbit and
        // the sheet must not be rebuilt from it. Re-installing a distinctly
        // positioned orbit makes the branch observable — without it
        // `read_orbit_camera` would return `None` and the no-op would be
        // indistinguishable from the missing-orbit no-op (the `Mutex` is
        // removed on entry, so a bare re-call returns `false` either way).
        install_orbit_camera(
            &mut engine,
            OrbitCamera::looking_at(
                glam::Vec3::new(3.0, 4.0, 9.0),
                glam::Vec3::new(1.0, 1.0, 0.0),
            ),
        );
        let saved_before =
            SAVED_ORBIT.with(|cell| cell.borrow().as_ref().map(|orbit| orbit.position()));
        let sheet_before = read_plane_camera(&engine).expect("plane camera");
        assert!(!apply_mode_request(
            &mut engine,
            ModeRequest::Plane2d,
            (800, 600)
        ));
        let saved_after =
            SAVED_ORBIT.with(|cell| cell.borrow().as_ref().map(|orbit| orbit.position()));
        let original = saved_before.expect("SAVED_ORBIT written on entry");
        assert!(
            (original - glam::Vec3::new(0.0, 0.0, 9.0)).length() < 1e-3,
            "SAVED_ORBIT must keep the original entry orbit, got {original:?}"
        );
        assert_eq!(
            saved_after, saved_before,
            "re-entering 2D must not overwrite SAVED_ORBIT"
        );
        let sheet_after = read_plane_camera(&engine).expect("plane camera");
        assert_eq!(
            sheet_after, sheet_before,
            "re-entering 2D must not rebuild the sheet from the hidden orbit"
        );
        // Leaving restores the saved orbit and flips the mode back.
        assert!(apply_mode_request(
            &mut engine,
            ModeRequest::Perspective3d,
            (800, 600)
        ));
        assert_eq!(
            engine.world().resources().get::<ViewportMode>().copied(),
            Some(ViewportMode::Perspective3d)
        );
        let orbit = read_orbit_camera(&engine).expect("orbit restored");
        assert!((orbit.position() - glam::Vec3::new(0.0, 0.0, 9.0)).length() < 1e-3);
        VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Perspective3d));
        SAVED_ORBIT.with(|cell| *cell.borrow_mut() = None);
    }

    /// `viewport_sync_plan` pins the pure sync branch: 2D publishes the
    /// plane picking camera and plane JSON; 3D publishes the orbit picking
    /// camera and resets the JSON; 3D without an orbit camera returns a
    /// null picking camera but still resets the JSON (so a previous
    /// plane2d overlay never sticks).
    #[test]
    fn viewport_sync_plan_switches_cleanly() {
        let plane = PlaneCamera::new(
            glam::Vec2::ZERO,
            ornis_core::Meters::new(5.0),
            ornis_core::Meters::new(10.0),
            ornis_core::Meters::new(0.1),
            ornis_core::Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test plane camera");
        let orbit = OrbitCamera::looking_at(glam::Vec3::new(0.0, 0.0, 9.0), glam::Vec3::ZERO);

        // 2D with plane: picking camera present, plane JSON; the display
        // scale passes through the plan into the wire untouched.
        let (picking, json) = viewport_sync_plan(
            ViewportMode::Plane2d,
            Some(&plane),
            Some(&orbit),
            (800, 600),
            None,
            1.5,
        );
        assert!(picking.is_some(), "2D publishes the plane picking camera");
        let v: serde_json::Value = serde_json::from_str(&json).expect("JSON");
        assert_eq!(v["mode"], serde_json::json!("plane2d"));
        assert_eq!(v["dpr"], serde_json::json!(1.5));

        // 3D with orbit: picking camera present, perspective JSON (nulls).
        let (picking, json) = viewport_sync_plan(
            ViewportMode::Perspective3d,
            Some(&plane),
            Some(&orbit),
            (800, 600),
            None,
            1.0,
        );
        assert!(picking.is_some(), "3D publishes the orbit picking camera");
        let v: serde_json::Value = serde_json::from_str(&json).expect("JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
        assert!(v["center"].is_null());

        // 3D without an orbit camera: null picking camera, JSON still
        // resets to perspective3d (does not leave a stale plane overlay).
        let (picking, json) = viewport_sync_plan(
            ViewportMode::Perspective3d,
            Some(&plane),
            None,
            (800, 600),
            None,
            1.0,
        );
        assert!(
            picking.is_none(),
            "3D without orbit publishes no picking camera"
        );
        let v: serde_json::Value = serde_json::from_str(&json).expect("JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
        assert!(v["center"].is_null());

        // 2D without a plane camera: null picking camera, perspective JSON.
        let (picking, json) = viewport_sync_plan(
            ViewportMode::Plane2d,
            None,
            Some(&orbit),
            (800, 600),
            None,
            1.0,
        );
        assert!(
            picking.is_none(),
            "2D without plane publishes no picking camera"
        );
        let v: serde_json::Value = serde_json::from_str(&json).expect("JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
    }

    /// `plane_viewport_sync_plan` pins the sheet-viewport size sync: in 2D
    /// a viewport that lags the surface is rewritten to `size_px`, a fresh
    /// one is left alone, and neither 3D nor a missing plane camera ever
    /// writes the sheet viewport.
    #[test]
    fn plane_viewport_sync_plan_targets_surface_size() {
        let plane = PlaneCamera::new(
            glam::Vec2::ZERO,
            ornis_core::Meters::new(5.0),
            ornis_core::Meters::new(10.0),
            ornis_core::Meters::new(0.1),
            ornis_core::Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test plane camera");
        // 2D + stale surface: rewrite to the new size.
        assert_eq!(
            plane_viewport_sync_plan(ViewportMode::Plane2d, Some(&plane), (1024, 768)),
            Some([1024.0, 768.0])
        );
        // 2D + already fresh: nothing to write.
        assert_eq!(
            plane_viewport_sync_plan(ViewportMode::Plane2d, Some(&plane), (800, 600)),
            None
        );
        // 3D never touches the sheet viewport.
        assert_eq!(
            plane_viewport_sync_plan(ViewportMode::Perspective3d, Some(&plane), (1024, 768)),
            None
        );
        // Plane mode without a plane camera: nothing to write.
        assert_eq!(
            plane_viewport_sync_plan(ViewportMode::Plane2d, None, (1024, 768)),
            None
        );
    }

    /// `sync_picking_camera` + `refresh_viewport_state_json` end-to-end
    /// on a bare engine: the pre-frame half publishes the mode and
    /// picking-camera resources and keeps the sheet viewport in step with
    /// the surface in 2D (and leaves it alone in 3D); the post-frame half
    /// caches the matching wire JSON. This is what makes the seam
    /// host-tested — both bodies are resource I/O over the plans pinned
    /// above.
    #[test]
    fn viewport_sync_publishes_camera_and_json() {
        VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Plane2d));
        LAST_STATE_JSON.with(|cell| cell.borrow_mut().clear());
        let mut engine = Engine::new();
        let plane = PlaneCamera::new(
            glam::Vec2::ZERO,
            ornis_core::Meters::new(5.0),
            ornis_core::Meters::new(10.0),
            ornis_core::Meters::new(0.1),
            ornis_core::Meters::new(100.0),
            [800.0, 600.0],
        )
        .expect("valid test plane camera");
        install_plane_camera(&mut engine, plane);

        // Plane mode: stale 800×600 viewport vs surface 1024×768 → synced.
        sync_picking_camera(&mut engine, (1024, 768));
        refresh_viewport_state_json(&mut engine, (1024, 768));
        let synced = read_plane_camera(&engine).expect("plane camera");
        assert!(synced.viewport_matches([1024.0, 768.0]));
        assert_eq!(
            engine.world().resources().get::<ViewportMode>().copied(),
            Some(ViewportMode::Plane2d)
        );
        assert!(
            engine
                .world()
                .resources()
                .get::<ornis_editor::ViewportCamera>()
                .is_some(),
            "plane picking camera published"
        );
        let v: serde_json::Value = serde_json::from_str(&viewport_state_json()).expect("JSON");
        assert_eq!(v["mode"], serde_json::json!("plane2d"));
        assert_eq!(v["center"], serde_json::json!([0.0, 0.0]));

        // 3D: mode flips, JSON resets to perspective3d (null plane fields),
        // and the sheet viewport stays where 2D left it.
        VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Perspective3d));
        install_orbit_camera(
            &mut engine,
            OrbitCamera::looking_at(glam::Vec3::new(0.0, 0.0, 9.0), glam::Vec3::ZERO),
        );
        sync_picking_camera(&mut engine, (640, 480));
        refresh_viewport_state_json(&mut engine, (640, 480));
        assert_eq!(
            engine.world().resources().get::<ViewportMode>().copied(),
            Some(ViewportMode::Perspective3d)
        );
        let v: serde_json::Value = serde_json::from_str(&viewport_state_json()).expect("JSON");
        assert_eq!(v["mode"], serde_json::json!("perspective3d"));
        assert!(v["center"].is_null());
        let plane = read_plane_camera(&engine).expect("plane camera");
        assert!(
            plane.viewport_matches([1024.0, 768.0]),
            "3D leaves 2D state"
        );

        // Without an orbit camera in 3D the stale ortho sheet camera is
        // removed, so picking never keeps the 2D view.
        let _ = engine
            .world_mut()
            .resources_mut()
            .remove::<std::sync::Mutex<OrbitCamera>>();
        sync_picking_camera(&mut engine, (640, 480));
        assert!(
            engine
                .world()
                .resources()
                .get::<ornis_editor::ViewportCamera>()
                .is_none(),
            "stale sheet camera removed when 3D has no orbit camera"
        );

        VIEWPORT_MODE.with(|cell| cell.set(ViewportMode::Perspective3d));
        LAST_STATE_JSON.with(|cell| cell.borrow_mut().clear());
    }
}
