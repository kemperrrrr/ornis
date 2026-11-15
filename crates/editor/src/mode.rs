//! Editor play/edit modes (PLAN §i, slice E4).
//!
//! [`EditorMode`] is the single resource the host flips to switch between
//! authoring and simulation. The scheduler gates two system sets off it
//! (see the host's mode-gating install): the *simulation* set (physics,
//! gameplay, animation, audio) runs only in [`EditorMode::Play`], while the
//! *editor chrome* set (picking, gizmos) runs in [`EditorMode::Edit`] and
//! [`EditorMode::Paused`]. The resource carries no scene state — the
//! save/restore of the authored scene on Play/Stop lives in the session.

/// The editor's run mode: authoring, simulating, or frozen mid-play.
///
/// Defaults to [`EditorMode::Edit`] so a freshly installed world authors
/// rather than simulates. The mode is a plain `Copy` resource read by the
/// scheduler's run conditions every frame; flipping it is the whole
/// play/edit switch (the session separately saves/restores the scene).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum EditorMode {
    /// Authoring: picking/chrome/gizmo run, simulation is stopped.
    #[default]
    Edit,
    /// Simulating: the simulation set runs, editor chrome is disabled.
    Play,
    /// Frozen mid-play: simulation is stopped but the current (play) state
    /// is kept and selection/inspector stay available.
    Paused,
}

impl EditorMode {
    /// Whether the simulation set should run this frame (only `Play`).
    #[must_use]
    pub fn simulation_runs(self) -> bool {
        matches!(self, Self::Play)
    }

    /// Whether the editor chrome set should run this frame (`Edit`/`Paused`).
    #[must_use]
    pub fn chrome_runs(self) -> bool {
        !self.simulation_runs()
    }

    /// Whether this is the authoring mode.
    #[must_use]
    pub fn is_edit(self) -> bool {
        matches!(self, Self::Edit)
    }

    /// Whether this is the simulating mode.
    #[must_use]
    pub fn is_play(self) -> bool {
        matches!(self, Self::Play)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two sets are exact complements across every mode: simulation
    /// runs only in Play, chrome runs in Edit and Paused.
    #[test]
    fn sets_are_complementary_across_modes() {
        for mode in [EditorMode::Edit, EditorMode::Play, EditorMode::Paused] {
            assert_ne!(
                mode.simulation_runs(),
                mode.chrome_runs(),
                "{mode:?}: exactly one set runs"
            );
        }
        assert!(EditorMode::Play.simulation_runs());
        assert!(!EditorMode::Edit.simulation_runs());
        assert!(!EditorMode::Paused.simulation_runs());
        assert!(EditorMode::Edit.chrome_runs());
        assert!(EditorMode::Paused.chrome_runs());
        assert!(!EditorMode::Play.chrome_runs());
    }

    /// Default is Edit; the mode predicates agree with the variant checks.
    #[test]
    fn default_is_edit_and_predicates_agree() {
        assert_eq!(EditorMode::default(), EditorMode::Edit);
        assert!(EditorMode::Edit.is_edit());
        assert!(EditorMode::Play.is_play());
        assert!(EditorMode::Paused.chrome_runs());
        assert!(!EditorMode::Edit.is_play());
        assert!(!EditorMode::Play.is_edit());
    }
}
