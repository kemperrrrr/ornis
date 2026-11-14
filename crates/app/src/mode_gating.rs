//! Play/edit system-set gating for the editor session (PLAN §i, slice E4).
//!
//! [`install_editor_mode_gating`] flips two system sets off the
//! [`EditorMode`] resource using the scheduler's run conditions: the
//! *simulation* set runs only in [`EditorMode::Play`], the *editor chrome*
//! set runs in [`EditorMode::Edit`]/[`EditorMode::Paused`]. The whole fixed
//! plan is simulation, so it is gated wholesale (systems added later inherit
//! the gate); the variable-rate plan mixes simulation and chrome, so its
//! members are gated by name.
//!
//! Only the editor session calls this — other hosts (showcase runner, tests)
//! never install the gate, so their systems run unconditionally. The editor
//! viewport draws from the `TransformDesc` snapshot, not `GlobalTransform`,
//! so freezing the fixed plan (including transform propagation) does not
//! stale the editor's visuals.

use std::sync::Arc;

use ornis_core::{Engine, RunCondition};
use ornis_editor::EditorMode;

/// Variable-rate simulation systems gated to `Play` (the fixed plan is
/// gated wholesale separately). Best-effort: a host may not install every
/// member (e.g. `audio_step` needs an audio device); the behavioural
/// session tests are the safety net against a missed gate.
const SIMULATION_SYSTEMS: [&str; 8] = [
    "player_input",
    "transform_update",
    "body_to_transform",
    "anim_sample",
    "skel_sample",
    "skel_skin_cpu",
    "audio_listener_sync",
    "audio_step",
];

/// Editor-chrome systems gated to `Edit`/`Paused` (disabled in `Play`).
const CHROME_SYSTEMS: [&str; 2] = ["editor_maintain", "gizmo_maintain"];

/// Installs the [`EditorMode`] resource (defaulting to [`EditorMode::Edit`])
/// and gates the simulation and editor-chrome system sets off it.
///
/// Idempotent: the resource is inserted only when absent and the conditions
/// overwrite any previous ones, so a second call re-applies the same gates.
/// The fixed plan is gated as a whole; variable-rate members are gated by
/// name (unknown names are skipped — see [`SIMULATION_SYSTEMS`]).
pub fn install_editor_mode_gating(engine: &mut Engine) {
    if engine.world().resources().get::<EditorMode>().is_none() {
        let _ = engine.world_mut().insert(EditorMode::default());
    }
    let simulation: RunCondition = Arc::new(|resources| {
        resources
            .get::<EditorMode>()
            .copied()
            .unwrap_or_default()
            .simulation_runs()
    });
    let chrome: RunCondition = Arc::new(|resources| {
        resources
            .get::<EditorMode>()
            .copied()
            .unwrap_or_default()
            .chrome_runs()
    });
    engine
        .fixed_schedule_mut()
        .set_schedule_condition(simulation.clone());
    for name in SIMULATION_SYSTEMS {
        let _ = engine
            .schedule_mut()
            .set_run_condition(name, simulation.clone());
    }
    for name in CHROME_SYSTEMS {
        let _ = engine
            .schedule_mut()
            .set_run_condition(name, chrome.clone());
    }
}
