//! Host wiring for the transform gizmo (PLAN §i, slice E3): the
//! authoritative layout system and the replica-side visibility helper.
//!
//! The pure handle identity and drag math stay in [`crate::gizmo`]; this
//! module owns every write into `TransformDesc`, so both hosts place the
//! same handles from the same [`layout`](crate::gizmo::layout) canon.

use crate::Selected;
use crate::gizmo::{GizmoHandle, GizmoMode, GizmoSpace, layout};
use ornis_assets::scene::TransformDesc;
use ornis_core::units::UnitQuat;
use ornis_core::{Engine, Entity, Resources, SmartStore, System, SystemAccess};

/// Schedule name of the authoritative gizmo layout system.
const GIZMO_SYSTEM_NAME: &str = "gizmo_maintain";

/// Hidden-handle transform (zero scale keeps it out of rendering/picking).
fn hidden_transform() -> TransformDesc {
    TransformDesc {
        translation: glam::Vec3::ZERO,
        rotation: UnitQuat::IDENTITY,
        scale: glam::Vec3::ZERO,
    }
}

/// Rewrites every handle transform: `keep` selects the visible set, the
/// rest collapse to zero scale. `selection` drives the pose (its T/R/S is
/// the gizmo origin); a missing selection hides everything.
fn write_layouts(
    store: &SmartStore,
    selection: Option<&TransformDesc>,
    keep: impl Fn(GizmoHandle) -> bool,
) {
    let Some(handles) = store.read_lane::<GizmoHandle>() else {
        return;
    };
    let updates: Vec<(Entity, TransformDesc)> = handles
        .entities
        .iter()
        .zip(handles.data.iter())
        .map(|(entity, handle)| {
            let transform = match (selection, keep(*handle)) {
                (Some(selection), true) => layout(selection, *handle).transform,
                _ => hidden_transform(),
            };
            (*entity, transform)
        })
        .collect();
    drop(handles);
    let Some(mut lane) = store.write_lane::<TransformDesc>() else {
        return;
    };
    for (entity, transform) in updates {
        lane.insert(entity, transform);
    }
}

/// Selection transform the handles are laid out around.
fn selection_transform(store: &SmartStore) -> Option<TransformDesc> {
    let selected = store
        .read_lane::<Selected>()
        .and_then(|lane| lane.entities.first().copied())?;
    store.read_lane::<TransformDesc>()?.get(selected).cloned()
}

/// Installs the authoritative gizmo layout system.
///
/// The session calls this once; the replica does not (its handles arrive
/// through `/api/scene` and are positioned locally by
/// [`apply_gizmo_visibility`]). Idempotent like [`crate::install_editor`].
pub fn install_gizmo_authoring(engine: &mut Engine) {
    if engine.schedule().mermaid().contains(GIZMO_SYSTEM_NAME) {
        return;
    }
    if let Some(store) = engine.world_mut().store_mut() {
        store.register::<GizmoHandle>();
    }
    engine.schedule_mut().add_system(GizmoMaintain);
}

/// Lays handles around the selected entity every frame.
///
/// Handles with no selection (or a dead one) collapse to zero scale; the
/// session normally despawns them on clear, so this is the defensive path.
struct GizmoMaintain;

impl System for GizmoMaintain {
    fn name(&self) -> &'static str {
        GIZMO_SYSTEM_NAME
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<Selected>()
            .reads_lane::<GizmoHandle>()
            .reads_lane::<TransformDesc>()
            .writes_lane::<TransformDesc>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let selection = selection_transform(store);
        write_layouts(store, selection.as_ref(), |_| true);
    }
}

/// Replica-side handle visibility/placement for the active mode and space.
///
/// The authoritative snapshot carries every handle; the viewport shows only
/// the active `(mode, space)` set and collapses the rest. Also called after
/// each drag preview step so the handles follow the moving selection.
pub fn apply_gizmo_visibility(store: &mut SmartStore, mode: GizmoMode, space: GizmoSpace) {
    let selection = selection_transform(store);
    write_layouts(store, selection.as_ref(), |handle| {
        handle.matches(mode, space)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gizmo::{BAR_THICKNESS, MOVE_LENGTH};
    use crate::{EditorOnly, install_editor, set_selection};
    use glam::Vec3;

    /// Store with editor lanes, one selected scene entity and two handles.
    fn fixture() -> (Engine, Entity, Entity, Entity) {
        let mut engine = Engine::new();
        install_editor(&mut engine);
        let store = engine.world_mut().store_mut().expect("store");
        store.register::<TransformDesc>();
        let scene = store.create_entity();
        store.insert(
            scene,
            TransformDesc::from_translation(Vec3::new(1.0, 2.0, 3.0)),
        );
        let active = store.create_entity();
        store.insert(
            active,
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::World,
                axis: crate::GizmoAxis::X,
            },
        );
        store.insert(active, EditorOnly);
        store.insert(active, TransformDesc::IDENTITY);
        let inactive = store.create_entity();
        store.insert(
            inactive,
            GizmoHandle {
                mode: GizmoMode::Rotate,
                space: GizmoSpace::World,
                axis: crate::GizmoAxis::Y,
            },
        );
        store.insert(inactive, EditorOnly);
        store.insert(inactive, TransformDesc::IDENTITY);
        set_selection(store, Some(scene));
        (engine, scene, active, inactive)
    }

    #[test]
    fn visibility_positions_the_active_set_and_hides_the_rest() {
        let (mut engine, _scene, active, inactive) = fixture();
        let store = engine.world_mut().store_mut().expect("store");
        apply_gizmo_visibility(store, GizmoMode::Move, GizmoSpace::World);
        let lane = store.read_lane::<TransformDesc>().expect("lane");
        let active_transform = lane.get(active).expect("active transform");
        let expected = layout(
            &TransformDesc::from_translation(Vec3::new(1.0, 2.0, 3.0)),
            GizmoHandle {
                mode: GizmoMode::Move,
                space: GizmoSpace::World,
                axis: crate::GizmoAxis::X,
            },
        )
        .transform;
        assert!((active_transform.translation - expected.translation).length() < 1e-6);
        let hidden = lane.get(inactive).expect("inactive transform");
        assert_eq!(hidden.scale, Vec3::ZERO);
    }

    #[test]
    fn authoring_system_layouts_handles_from_the_selection() {
        let (mut engine, _scene, active, _inactive) = fixture();
        install_gizmo_authoring(&mut engine);
        engine.run_frame(0.0);
        let store = engine.world().store().expect("store");
        let lane = store.read_lane::<TransformDesc>().expect("lane");
        let transform = lane.get(active).expect("active transform");
        assert!((transform.translation - Vec3::new(1.55, 2.0, 3.0)).length() < 1e-5);
        assert_eq!(
            transform.scale,
            Vec3::new(MOVE_LENGTH, BAR_THICKNESS, BAR_THICKNESS)
        );
    }
}
