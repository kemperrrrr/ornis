//! Replica play/edit mode (PLAN §i, slice E4): the browser mirrors the
//! authoritative editor mode so viewport chrome (picking, gizmo drag) runs
//! only in Edit/Paused, never in Play. Like the gizmo tool/space, the mode
//! is a UI-relayed state: JS writes it through [`set_editor_mode`] on every
//! `mode_changed`, and the frame loop gates chrome off it. Without this a
//! drag in Play could commit an edit the server discards on Stop.

use std::cell::Cell;
use std::sync::Arc;

use ornis_core::{Engine, RunCondition};
use ornis_editor::EditorMode;
use wasm_bindgen::prelude::*;

thread_local! {
    /// Authoritative editor mode mirrored from the server (E4); defaults to
    /// Edit so a freshly booted replica authors rather than simulates.
    static EDITOR_MODE: Cell<EditorMode> = const { Cell::new(EditorMode::Edit) };
}

/// Active editor mode (read by the frame loop's manual gizmo-drag gate).
pub(crate) fn editor_mode() -> EditorMode {
    EDITOR_MODE.with(Cell::get)
}

/// Mirrors the authoritative editor mode (E4): `"edit"`, `"play"` or
/// `"paused"`. Returns `false` on an unknown wire name, keeping the last
/// mode. In Play the replica gates picking and gizmo drag off.
// qual:api — wasm-bindgen export; called from the browser transport.
#[wasm_bindgen]
pub fn set_editor_mode(mode: &str) -> bool {
    let parsed = match mode {
        "edit" => EditorMode::Edit,
        "play" => EditorMode::Play,
        "paused" => EditorMode::Paused,
        _ => return false,
    };
    EDITOR_MODE.with(|cell| cell.set(parsed));
    true
}

/// Inserts the [`EditorMode`] resource and gates the replica's picking
/// system (`editor_maintain`) to Edit/Paused.
///
/// The replica runs no simulation and lays gizmo handles out manually, so
/// only the picking system needs a run condition; the manual gizmo drag
/// reads [`editor_mode`] directly. The condition reads the shared resource
/// (not the thread-local) so it stays correct under the parallel scheduler;
/// [`sync_editor_mode`] refreshes that resource from the JS setter each
/// frame. Idempotent: the resource is inserted only when absent.
pub(crate) fn install_replica_chrome_gate(engine: &mut Engine) {
    if engine.world().resources().get::<EditorMode>().is_none() {
        let _ = engine.world_mut().insert(EditorMode::default());
    }
    let chrome: RunCondition = Arc::new(|resources| {
        resources
            .get::<EditorMode>()
            .copied()
            .unwrap_or_default()
            .chrome_runs()
    });
    let _ = engine
        .schedule_mut()
        .set_run_condition("editor_maintain", Some(chrome));
}

/// Refreshes the [`EditorMode`] resource from the JS-relayed thread-local so
/// the next frame's picking gate sees the current mode.
pub(crate) fn sync_editor_mode(engine: &mut Engine) {
    let _ = engine.world_mut().insert(editor_mode());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::{Resources, System, SystemAccess};
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn set_editor_mode_parses_known_modes_only() {
        assert!(set_editor_mode("play"));
        assert_eq!(editor_mode(), EditorMode::Play);
        assert!(set_editor_mode("paused"));
        assert_eq!(editor_mode(), EditorMode::Paused);
        assert!(set_editor_mode("edit"));
        assert_eq!(editor_mode(), EditorMode::Edit);
        assert!(!set_editor_mode("bogus"));
        assert_eq!(
            editor_mode(),
            EditorMode::Edit,
            "unknown keeps the last mode"
        );
    }

    /// Stub chrome system that counts runs, named like the real picking
    /// system so the gate targets it.
    struct CountingMaintain(Arc<AtomicU64>);

    impl System for CountingMaintain {
        fn name(&self) -> &'static str {
            "editor_maintain"
        }
        fn access(&self) -> SystemAccess {
            SystemAccess::new()
        }
        fn run(&self, _resources: &Resources) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// The picking gate runs chrome in Edit/Paused and skips it in Play,
    /// reading the synced resource (correct under the parallel scheduler).
    #[test]
    fn chrome_gate_runs_picking_only_outside_play() {
        let runs = Arc::new(AtomicU64::new(0));
        let mut engine = Engine::new();
        engine
            .schedule_mut()
            .add_system(CountingMaintain(runs.clone()));
        install_replica_chrome_gate(&mut engine);

        set_editor_mode("edit");
        sync_editor_mode(&mut engine);
        engine.run_frame(1.0 / 60.0);
        assert_eq!(runs.load(Ordering::SeqCst), 1, "Edit runs chrome");

        set_editor_mode("play");
        sync_editor_mode(&mut engine);
        engine.run_frame(1.0 / 60.0);
        assert_eq!(runs.load(Ordering::SeqCst), 1, "Play skips chrome");

        set_editor_mode("paused");
        sync_editor_mode(&mut engine);
        engine.run_frame(1.0 / 60.0);
        assert_eq!(runs.load(Ordering::SeqCst), 2, "Paused runs chrome");

        // Restore the default for the other tests on this thread.
        set_editor_mode("edit");
    }
}
