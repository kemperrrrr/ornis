//! Viewport domain of the editor as engine systems (PLAN §i, slice E0).
//!
//! Selection, hover and chrome (gizmos, grid) live in the authoritative
//! world as ordinary entities carrying the marker components below, so the
//! browser replica observes them for free through `/api/scene`. Panels and
//! the inspector stay web-side; this crate never grows transport code
//! (`editor-backend` owns that edge).
//!
//! DAG position: `binary → ornis-app(session) → ornis-editor → {ornis-core}`.
//! The crate depends on `ornis-core` only (the [`Engine`], its schedules
//! and the [`SmartStore`] lanes the markers live in) plus `serde` for the
//! [`EditorMarks`] snapshot view. Deliberately absent in E0: `ornis-assets`
//! (no scene I/O here — save/load filters live in `ornis-app`, which owns
//! the `Scene` mapping), `ornis-gameplay` (no input intent yet), and
//! `ornis-render`/`ornis-physics` (picking in E1 will read the CPU-side
//! render extraction; until then a render edge would be a dead dependency).
//! The one hard rule this crate enables: editor chrome never touches the
//! solver or the scene file — both filters key off [`is_editor_only`], the
//! single predicate every host must use instead of re-deriving the check
//! (the "forgotten filter" bug class).

#![warn(missing_docs)]

use ornis_core::{Engine, Entity, Resources, SmartStore, System, SystemAccess};

/// Marks an entity as editor chrome (gizmo handle, grid, selection proxy).
///
/// Entities carrying this marker are ordinary world citizens for the
/// snapshot path ([`EditorMarks`] labels them, the replica draws them) but
/// are excluded from physics sync-in and from scene save/serialization: a
/// gizmo must neither fall under gravity nor persist into the scene file.
/// Hosts must not re-derive this check — call [`is_editor_only`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EditorOnly;

/// Marks an entity as currently selected in the viewport.
///
/// Presence is the mark (no payload): the replica renders the outline from
/// the [`EditorMarks`] snapshot label, and the inspector edits the entity
/// through the existing `set_component` path. Picking (E1) will set and
/// clear this component; E0 only defines it and carries it in snapshots.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Selected;

/// Marks the entity currently under the viewport pointer.
///
/// Presence is the mark (no payload), mirroring [`Selected`]: a transient
/// hover highlight for the replica. Set by picking (E1); defined and
/// snapshotted already in E0 so the wire format is stable from the start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Hovered;

/// Snapshot view of one entity's editor marks for `GET /api/scene`.
///
/// Serializes canonically as the per-entity `editor` object, always present
/// (so the replica can rely on the shape without probing):
///
/// ```json
/// {"editor_only": false, "selected": true, "hovered": false}
/// ```
///
/// `editor_only` mirrors [`EditorOnly`] (chrome replicates, but never
/// saves); `selected`/`hovered` mirror [`Selected`]/[`Hovered`]. Read via
/// [`EditorMarks::of`]; the authoritative world keeps presence-based lanes,
/// this struct is only the wire projection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EditorMarks {
    /// The entity carries [`EditorOnly`]: viewport chrome, not scene content.
    pub editor_only: bool,
    /// The entity carries [`Selected`]: the replica draws the outline.
    pub selected: bool,
    /// The entity carries [`Hovered`]: the replica draws the hover highlight.
    pub hovered: bool,
}

impl EditorMarks {
    /// Reads the editor marks of `entity` from the store.
    ///
    /// A missing lane counts as absent (a world without [`install_editor`]
    /// simply reports no marks), so this is safe on any store.
    pub fn of(store: &SmartStore, entity: Entity) -> Self {
        Self {
            editor_only: is_editor_only(store, entity),
            selected: has_marker::<Selected>(store, entity),
            hovered: has_marker::<Hovered>(store, entity),
        }
    }
}

/// Central `EditorOnly` predicate behind every host-side filter.
///
/// Physics sync-in, scene save/serialization and any future consumer must
/// call this instead of reading the lane directly, so the exclusion rule
/// has exactly one definition. A store without the lane (no
/// [`install_editor`]) reports `false` for every entity.
pub fn is_editor_only(store: &SmartStore, entity: Entity) -> bool {
    has_marker::<EditorOnly>(store, entity)
}

/// Presence check for a unit marker lane; a missing lane means absent.
fn has_marker<T: 'static + Send + Sync>(store: &SmartStore, entity: Entity) -> bool {
    store
        .read_lane::<T>()
        .is_some_and(|lane| lane.contains(entity))
}

/// Installs the editor viewport domain into `engine`.
///
/// Registers the [`EditorOnly`]/[`Selected`]/[`Hovered`] lanes up front (so
/// snapshot and filter reads never observe a missing lane on an installed
/// engine) and adds the `editor_maintain` frame-system skeleton. The system
/// is intentionally a no-op in E0: it reserves the schedule slot (and its
/// access declaration) that E1 picking will fill, keeping the DAG stable
/// across slices.
///
/// Idempotent: a second call on the same engine registers nothing twice
/// (guarded on the `editor_maintain` schedule entry, like the
/// `install_gameplay_physics_bridge` family).
pub fn install_editor(engine: &mut Engine) {
    if engine.schedule().mermaid().contains(SYSTEM_NAME) {
        return;
    }
    if let Some(store) = engine.world_mut().store_mut() {
        store.register::<EditorOnly>();
        store.register::<Selected>();
        store.register::<Hovered>();
    }
    engine.schedule_mut().add_system(EditorMaintain);
}

/// Schedule name of the E0 skeleton system (also the idempotency guard).
const SYSTEM_NAME: &str = "editor_maintain";

/// Frame-system skeleton for the editor viewport domain.
///
/// E0 reserves the slot only (no behaviour): E1 picking will read input
/// and the render extraction here and maintain [`Selected`]/[`Hovered`].
/// Declares the marker lanes up front so the schedule DAG already carries
/// the edges future slices need.
struct EditorMaintain;

impl System for EditorMaintain {
    /// Schedule name of this system (matches [`SYSTEM_NAME`]).
    fn name(&self) -> &'static str {
        SYSTEM_NAME
    }

    /// Declares the editor marker lanes (read) over the shared store.
    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<EditorOnly>()
            .reads_lane::<Selected>()
            .reads_lane::<Hovered>()
    }

    /// E0 skeleton: observes nothing, writes nothing.
    fn run(&self, _resources: &Resources) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `install_editor` registers all marker lanes and the skeleton system,
    /// is idempotent, and an empty frame runs cleanly.
    #[test]
    fn install_registers_lanes_and_system_idempotently() {
        let mut engine = Engine::new();
        install_editor(&mut engine);
        install_editor(&mut engine);
        let store = engine.world().store().expect("world store");
        assert!(store.read_lane::<EditorOnly>().is_some());
        assert!(store.read_lane::<Selected>().is_some());
        assert!(store.read_lane::<Hovered>().is_some());
        let occurrences = engine.schedule().mermaid().matches(SYSTEM_NAME).count();
        assert_eq!(occurrences, 1, "exactly one {SYSTEM_NAME}");
        engine.run_frame(1.0 / 60.0);
    }

    /// Marker presence round-trips through [`EditorMarks::of`]; a store
    /// without installation reports no marks instead of failing.
    #[test]
    fn marks_reflect_lane_presence() {
        let mut engine = Engine::new();
        let plain = engine.world().store().expect("store").create_entity();
        assert_eq!(
            EditorMarks::of(engine.world().store().expect("store"), plain),
            EditorMarks::default()
        );
        assert!(!is_editor_only(
            engine.world().store().expect("store"),
            plain
        ));

        install_editor(&mut engine);
        let store = engine.world_mut().store_mut().expect("store");
        let gizmo = store.create_entity();
        store.insert(gizmo, EditorOnly);
        store.insert(gizmo, Selected);
        let marks = EditorMarks::of(store, gizmo);
        assert_eq!(
            marks,
            EditorMarks {
                editor_only: true,
                selected: true,
                hovered: false,
            }
        );
        assert!(is_editor_only(store, gizmo));
        assert!(!is_editor_only(store, plain));
    }

    /// Declared accesses cover the marker lanes (schedule enforcement).
    #[test]
    fn system_declares_marker_lanes() {
        use std::any::TypeId;
        let access = EditorMaintain.access();
        for id in [
            TypeId::of::<EditorOnly>(),
            TypeId::of::<Selected>(),
            TypeId::of::<Hovered>(),
        ] {
            assert!(access.reads_lanes.contains(&id));
        }
    }
}
