//! Gizmo tool/space bindings for the replica viewport (PLAN §i, E4): the
//! toolbar queues mode/space/undo requests through `wasm_bindgen` exports
//! and the frame loop applies them on the single browser thread. Split out
//! of `viewport` so the mode-switching and grid-overlay wire state stays
//! separate from the gizmo toolbar state.

use std::cell::Cell;

use ornis_core::Engine;
use ornis_editor::{GizmoMode, GizmoSpace, apply_gizmo_visibility};
use wasm_bindgen::prelude::*;

thread_local! {
    /// Active gizmo mode (select tool = `Off`).
    pub(crate) static GIZMO_MODE: Cell<GizmoMode> = const { Cell::new(GizmoMode::Move) };
    /// Active gizmo axis space.
    pub(crate) static GIZMO_SPACE: Cell<GizmoSpace> = const { Cell::new(GizmoSpace::World) };
    /// Pending gizmo mode/space request from the toolbar.
    pub(crate) static PENDING_GIZMO: Cell<Option<(GizmoMode, GizmoSpace)>> =
        const { Cell::new(None) };
    /// One-step undo request from the toolbar.
    pub(crate) static PENDING_UNDO: Cell<bool> = const { Cell::new(false) };
}

/// Active gizmo mode (read by the frame loop's drag/picking step).
pub(crate) fn gizmo_mode() -> GizmoMode {
    GIZMO_MODE.with(Cell::get)
}

/// Active gizmo space (read by the frame loop's drag step).
pub(crate) fn gizmo_space() -> GizmoSpace {
    GIZMO_SPACE.with(Cell::get)
}

/// Gizmo mode for handle *visibility*: `Off` in Play (chrome off), so every
/// handle collapses; otherwise the active tool. The drag step still reads
/// [`gizmo_mode`] — this only governs what the replica draws, so a Play
/// session never shows authoring handles over the simulating scene.
pub(crate) fn visible_gizmo_mode() -> GizmoMode {
    if crate::mode::editor_mode().chrome_runs() {
        gizmo_mode()
    } else {
        GizmoMode::Off
    }
}

/// Applies a queued toolbar request: updates the active mode/space and
/// repositions/hides the replica handles. Returns whether a request was
/// applied.
pub(crate) fn apply_pending_gizmo(engine: &mut Engine) -> bool {
    let Some((mode, space)) = PENDING_GIZMO.with(Cell::take) else {
        return false;
    };
    GIZMO_MODE.with(|cell| cell.set(mode));
    GIZMO_SPACE.with(|cell| cell.set(space));
    if let Some(store) = engine.world_mut().store_mut() {
        apply_gizmo_visibility(store, visible_gizmo_mode(), space);
    }
    true
}

/// Takes the pending one-step undo request.
pub(crate) fn take_pending_undo() -> bool {
    PENDING_UNDO.with(Cell::take)
}

/// Selects the gizmo tool: `"move"`, `"rotate"`, `"scale"`, or `"off"`
/// (the select tool hides the handles and disables gizmo dragging).
///
/// The loop applies it on the next frame (same single-threaded queue as
/// [`super::set_viewport_mode`]); returns `false` on an unknown tool.
#[wasm_bindgen]
pub fn set_gizmo_mode(mode: &str) -> bool {
    let parsed = match mode {
        "move" => GizmoMode::Move,
        "rotate" => GizmoMode::Rotate,
        "scale" => GizmoMode::Scale,
        "off" | "select" => GizmoMode::Off,
        _ => return false,
    };
    PENDING_GIZMO.with(|cell| {
        // The queued request is the base, not the applied state: two
        // requests in one frame (space, then mode) must not lose the
        // space change.
        let (_, space) = cell.get().unwrap_or((gizmo_mode(), gizmo_space()));
        cell.set(Some((parsed, space)));
    });
    true
}

/// Selects world or local gizmo axes: `"world"` or `"local"`; the loop
/// applies it on the next frame. Returns `false` on an unknown space.
#[wasm_bindgen]
pub fn set_gizmo_space(space: &str) -> bool {
    let parsed = match space {
        "world" => GizmoSpace::World,
        "local" => GizmoSpace::Local,
        _ => return false,
    };
    PENDING_GIZMO.with(|cell| {
        // The queued request is the base, not the applied state: two
        // requests in one frame (space, then mode) must not lose the
        // space change.
        let (mode, _) = cell.get().unwrap_or((gizmo_mode(), gizmo_space()));
        cell.set(Some((mode, parsed)));
    });
    true
}

/// Queues one-step undo of the last gizmo drag. The loop sends the stored
/// pre-drag transform through `set_component`.
#[wasm_bindgen]
pub fn undo_gizmo() -> bool {
    PENDING_UNDO.with(|cell| cell.set(true));
    true
}
