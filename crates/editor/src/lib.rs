//! Viewport domain of the editor as engine systems (PLAN §i, slices E0–E1).
//!
//! Selection, hover and chrome (gizmos, grid) live in the authoritative
//! world as ordinary entities carrying the marker components below, so the
//! browser replica observes them for free through `/api/scene`. Panels and
//! the inspector stay web-side; this crate never grows transport code
//! (`editor-backend` owns that edge).
//!
//! DAG position: `binary → ornis-app(session) → ornis-editor → {ornis-core,
//! ornis-assets}`. The `ornis-assets` edge is the picking read of
//! `TransformDesc`/`MeshDesc` (E1); there is deliberately no `ornis-render`
//! edge (no GPU/wgpu types — picking mirrors the extraction canon over the
//! lanes) and no `ornis-gameplay` edge (no input intent, only the shared
//! `InputState` snapshot). The one hard rule this crate enables: editor
//! chrome never touches the solver or the scene file — both filters key off
//! [`is_editor_only`], the single predicate every host must use instead of
//! re-deriving the check (the "forgotten filter" bug class).
//!
//! Picking (E1) runs where the camera lives: the authoritative world stays
//! camera-agnostic, while each viewport side holds a [`ViewportCamera`]
//! resource (converted from its orbit camera) and the `editor_maintain`
//! system turns press edges into [`Selected`] and pointer rest into
//! [`Hovered`]. Chrome ([`EditorOnly`]) replicates for drawing but is not
//! clickable in E1 — gizmo picking arrives with E3 (see
//! [`ChromePolicy`]).

#![warn(missing_docs)]

use std::sync::Mutex;

use ornis_assets::scene::{MeshDesc, TransformDesc};
use ornis_core::{
    Engine, Entity, InputState, MouseButton, Resources, SmartStore, System, SystemAccess,
};

/// Viewport picking: rays, hits and store intersection (slice E1).
pub mod picking;

pub use picking::{
    ChromePolicy, PickEpsilon, PickError, PickHit, PickOutcome, PickRay, ViewportCamera,
    pick_closest, pick_ray,
};

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
/// through the existing `set_component` path. Maintained by the click
/// pipeline ([`editor_pick_click`]): a hit selects exclusively, a miss
/// clears.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Selected;

/// Marks the entity currently under the viewport pointer.
///
/// Presence is the mark (no payload), mirroring [`Selected`]: a transient
/// hover highlight for the replica. Maintained by [`editor_pick_hover`]
/// from the pointer rest position (no press needed).
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

/// Edge state behind the click/hover pipelines.
///
/// `prev_pressed` turns the held left-button level into a press edge (one
/// selection update per click — a held drag keeps orbiting the camera
/// without re-picking); `prev_hover` skips hover lane writes while the
/// pointer rests on the same entity. Stored behind a `Mutex` like the orbit
/// camera: systems observe shared `Resources`, so interior mutability is
/// the scheduled-mutation path.
#[derive(Debug, Default)]
pub struct EditorPickState {
    /// Left-button level observed on the previous frame.
    prev_pressed: bool,
    /// Entity carrying [`Hovered`] after the previous hover pass.
    prev_hover: Option<Entity>,
}

/// Locks pick state, recovering from a poisoned mutex (a picking pass never
/// panics, so poisoning only comes from an unrelated panic elsewhere).
fn lock_pick_state(state: &Mutex<EditorPickState>) -> std::sync::MutexGuard<'_, EditorPickState> {
    state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Click pipeline: press edge → pick → exclusive [`Selected`].
///
/// Returns `None` when there is no press edge (the common frame — no state
/// changes); otherwise `Some` with the applied outcome. Failures
/// ([`PickError::NoCamera`] on a camera-agnostic host, [`PickError::NoInput`],
/// [`PickError::EmptyScene`], [`PickError::DegenerateView`],
/// [`PickError::OutsideViewport`]) change nothing — errors never mutate the
/// selection. A hit selects exclusively, a miss clears the selection.
/// Chrome never wins (see [`ChromePolicy`]); near/far clip comes from the
/// viewport camera.
pub fn editor_pick_click(resources: &Resources) -> Option<Result<PickOutcome, PickError>> {
    let Some(input) = resources.get::<InputState>() else {
        return Some(Err(PickError::NoInput));
    };
    let pressed = input.button_down(MouseButton::Left);
    let pointer = input.pointer_position();
    let previous = resources
        .get::<Mutex<EditorPickState>>()
        .is_some_and(|state| lock_pick_state(state).prev_pressed);
    if let Some(state) = resources.get::<Mutex<EditorPickState>>() {
        lock_pick_state(state).prev_pressed = pressed;
    }
    if !pressed || previous {
        return None;
    }
    Some(click_outcome(resources, pointer))
}

/// Hover pipeline: pointer rest → exclusive [`Hovered`] (no press needed).
///
/// Same failure contract as [`editor_pick_click`] (errors change nothing);
/// a pointer outside the viewport clears the hover instead of erroring, so
/// leaving the canvas never strands a highlight. Lane writes are skipped
/// while the hovered set is unchanged.
pub fn editor_pick_hover(resources: &Resources) -> Result<PickOutcome, PickError> {
    let Some(input) = resources.get::<InputState>() else {
        return Err(PickError::NoInput);
    };
    let Some(store) = resources.get::<SmartStore>() else {
        return Err(PickError::NoInput);
    };
    let Some(camera) = resources.get::<ViewportCamera>() else {
        return Err(PickError::NoCamera);
    };
    let outcome = match pick_ray(camera, input.pointer_position()) {
        Err(PickError::OutsideViewport) => PickOutcome::Miss,
        Err(other) => return Err(other),
        Ok(ray) => match picking::pick_closest(store, &ray, ChromePolicy::SkipChrome)? {
            Some(hit) if within_clip(camera, hit.distance()) => PickOutcome::Hit(hit),
            _ => PickOutcome::Miss,
        },
    };
    apply_hover(resources, store, &outcome);
    Ok(outcome)
}

/// Runs one click query and applies it to the [`Selected`] lane.
fn click_outcome(resources: &Resources, pointer: [f32; 2]) -> Result<PickOutcome, PickError> {
    let Some(store) = resources.get::<SmartStore>() else {
        return Err(PickError::NoInput);
    };
    let Some(camera) = resources.get::<ViewportCamera>() else {
        return Err(PickError::NoCamera);
    };
    let ray = pick_ray(camera, pointer)?;
    let outcome = match picking::pick_closest(store, &ray, ChromePolicy::SkipChrome)? {
        Some(hit) if within_clip(camera, hit.distance()) => PickOutcome::Hit(hit),
        _ => PickOutcome::Miss,
    };
    apply_selection(store, &outcome);
    Ok(outcome)
}

/// Whether a hit distance survives the camera near/far clip.
fn within_clip(camera: &ViewportCamera, distance: ornis_core::PositiveF32) -> bool {
    let d = distance.get();
    d >= camera.near() && d <= camera.far()
}

/// Applies a click outcome to the [`Selected`] lane: exclusive hit, clear
/// on miss. Idempotent (an unchanged set performs no lane writes).
fn apply_selection(store: &SmartStore, outcome: &PickOutcome) {
    match outcome {
        PickOutcome::Hit(hit) => {
            let already = store
                .read_lane::<Selected>()
                .is_some_and(|lane| lane.entities.len() == 1 && lane.contains(hit.entity()));
            if already {
                return;
            }
            if let Some(mut lane) = store.write_lane::<Selected>() {
                for entity in lane.entities.clone() {
                    if entity != hit.entity() {
                        lane.remove(entity);
                    }
                }
                if !lane.contains(hit.entity()) {
                    lane.insert(hit.entity(), Selected);
                }
            }
        }
        PickOutcome::Miss => {
            let empty = store
                .read_lane::<Selected>()
                .is_none_or(|lane| lane.entities.is_empty());
            if empty {
                return;
            }
            if let Some(mut lane) = store.write_lane::<Selected>() {
                for entity in lane.entities.clone() {
                    lane.remove(entity);
                }
            }
        }
    }
}

/// Applies a hover outcome to the [`Hovered`] lane, tracking the previous
/// hover in [`EditorPickState`] so resting frames skip lane writes.
fn apply_hover(resources: &Resources, store: &SmartStore, outcome: &PickOutcome) {
    let wanted = match outcome {
        PickOutcome::Hit(hit) => Some(hit.entity()),
        PickOutcome::Miss => None,
    };
    let known = resources
        .get::<Mutex<EditorPickState>>()
        .map(|state| lock_pick_state(state).prev_hover);
    if known.is_some_and(|prev| prev == wanted) {
        return;
    }
    if wanted.is_none() {
        let empty = store
            .read_lane::<Hovered>()
            .is_none_or(|lane| lane.entities.is_empty());
        if empty {
            set_prev_hover(resources, None);
            return;
        }
    }
    if let Some(mut lane) = store.write_lane::<Hovered>() {
        for entity in lane.entities.clone() {
            lane.remove(entity);
        }
        if let Some(entity) = wanted {
            lane.insert(entity, Hovered);
        }
    }
    set_prev_hover(resources, wanted);
}

/// Records the hovered entity in the pick state, if the resource exists.
fn set_prev_hover(resources: &Resources, hover: Option<Entity>) {
    if let Some(state) = resources.get::<Mutex<EditorPickState>>() {
        lock_pick_state(state).prev_hover = hover;
    }
}

/// Installs the editor viewport domain into `engine`.
///
/// Registers the [`EditorOnly`]/[`Selected`]/[`Hovered`] lanes up front (so
/// snapshot and filter reads never observe a missing lane on an installed
/// engine), installs the [`EditorPickState`] edge resource, and adds the
/// `editor_maintain` frame system (E1: hover + click picking). The
/// authoritative host stays camera-agnostic: no [`ViewportCamera`] is
/// installed here — each viewport side inserts its own, and picking no-ops
/// until one exists.
///
/// Idempotent: a second call on the same engine registers nothing twice
/// (guarded on the `editor_maintain` schedule entry, like the
/// `install_gameplay_physics_bridge` family) and never clobbers existing
/// pick state.
pub fn install_editor(engine: &mut Engine) {
    if engine.schedule().mermaid().contains(SYSTEM_NAME) {
        return;
    }
    if let Some(store) = engine.world_mut().store_mut() {
        store.register::<EditorOnly>();
        store.register::<Selected>();
        store.register::<Hovered>();
    }
    if engine
        .world()
        .resources()
        .get::<Mutex<EditorPickState>>()
        .is_none()
    {
        let _ = engine
            .world_mut()
            .insert(Mutex::new(EditorPickState::default()));
    }
    engine.schedule_mut().add_system(EditorMaintain);
}

/// Schedule name of the editor maintain system (also the idempotency guard).
const SYSTEM_NAME: &str = "editor_maintain";

/// Frame system for the editor viewport domain (E1: hover + click picking).
///
/// Hover updates [`Hovered`] from the pointer rest position every frame;
/// click turns the left-button press edge into an exclusive [`Selected`]
/// (hit) or a clear (miss). Reads input, the client-side camera and the
/// render lanes; writes only the marker lanes and the pick edge state.
/// Camera-absent hosts (authoritative server) no-op through
/// [`PickError::NoCamera`].
struct EditorMaintain;

impl System for EditorMaintain {
    /// Schedule name of this system (matches [`SYSTEM_NAME`]).
    fn name(&self) -> &'static str {
        SYSTEM_NAME
    }

    /// Declares the honest access set: input/camera/store resources plus
    /// render-lane reads and marker-lane writes (schedule enforcement
    /// panics on anything undeclared, so this list is the behaviour).
    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<InputState>()
            .reads::<ViewportCamera>()
            .reads::<SmartStore>()
            .writes::<Mutex<EditorPickState>>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<EditorOnly>()
            .writes_lane::<Selected>()
            .writes_lane::<Hovered>()
    }

    /// E1 behaviour: hover pass, then click pass; errors are no-ops.
    fn run(&self, resources: &Resources) {
        let _ = editor_pick_hover(resources);
        let _ = editor_pick_click(resources);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use ornis_core::PositiveF32;

    /// Test camera matching the picking-module rig: eye `(0, 0, 6)`.
    fn camera() -> ViewportCamera {
        ViewportCamera::new(
            Vec3::new(0.0, 0.0, 6.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
            [200.0, 200.0],
        )
        .expect("valid test camera")
    }

    /// Engine with the editor domain, a viewport camera and renderable
    /// lanes (mirrors the session construction without the session).
    fn pick_engine() -> Engine {
        let mut engine = Engine::new();
        install_editor(&mut engine);
        let _ = engine.world_mut().insert(camera());
        if engine.world().resources().get::<InputState>().is_none() {
            let _ = engine.world_mut().insert(InputState::default());
        }
        let store = engine.world_mut().store_mut().expect("world store");
        store.register::<TransformDesc>();
        store.register::<MeshDesc>();
        engine
    }

    /// Spawns a sphere scene entity (never chrome).
    fn spawn_scene(engine: &mut Engine, translation: [f32; 3]) -> Entity {
        let store = engine.world_mut().store_mut().expect("world store");
        let entity = store.create_entity();
        store.insert(
            entity,
            TransformDesc::from_translation(glam::Vec3::from_array(translation)),
        );
        store.insert(
            entity,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            },
        );
        entity
    }

    /// Presses or releases the left button at `pointer`, then runs a frame.
    fn click_frame(engine: &mut Engine, pressed: bool, pointer: [f32; 2]) {
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("input resource");
            input.set_button(MouseButton::Left, pressed);
            input.set_pointer_position(pointer);
        }
        engine.run_frame(1.0 / 60.0);
    }

    /// Whether `entity` currently carries the marker lane `T`.
    fn has<T: 'static + Send + Sync>(engine: &Engine, entity: Entity) -> bool {
        engine
            .world()
            .store()
            .expect("world store")
            .read_lane::<T>()
            .is_some_and(|lane| lane.contains(entity))
    }

    /// `install_editor` registers all marker lanes and the picking system,
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
        assert!(
            engine
                .world()
                .resources()
                .get::<Mutex<EditorPickState>>()
                .is_some()
        );
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

    /// Declared accesses cover the honest E1 set (schedule enforcement).
    #[test]
    fn system_declares_honest_access() {
        use std::any::TypeId;
        let access = EditorMaintain.access();
        for id in [
            TypeId::of::<InputState>(),
            TypeId::of::<ViewportCamera>(),
            TypeId::of::<SmartStore>(),
            TypeId::of::<Mutex<EditorPickState>>(),
        ] {
            assert!(access.reads.contains(&id) || access.writes.contains(&id));
        }
        for id in [
            TypeId::of::<TransformDesc>(),
            TypeId::of::<MeshDesc>(),
            TypeId::of::<EditorOnly>(),
        ] {
            assert!(access.reads_lanes.contains(&id));
        }
        for id in [TypeId::of::<Selected>(), TypeId::of::<Hovered>()] {
            assert!(access.writes_lanes.contains(&id));
        }
    }

    /// Click selects the nearest entity, a later click moves the selection
    /// off the previous one, and a miss clears it.
    #[test]
    fn click_selects_moves_and_clears() {
        let mut engine = pick_engine();
        let near = spawn_scene(&mut engine, [0.0, 0.0, 0.0]);
        let far = spawn_scene(&mut engine, [0.0, 0.0, -5.0]);

        click_frame(&mut engine, true, [100.0, 100.0]);
        assert!(has::<Selected>(&engine, near));
        assert!(!has::<Selected>(&engine, far));

        // Release re-arms the edge; holding never re-picks.
        click_frame(&mut engine, false, [100.0, 100.0]);
        assert!(has::<Selected>(&engine, near));

        // Move the winner off-ray: the next click selects the runner-up and
        // deselects the previous entity.
        {
            let store = engine.world().store().expect("world store");
            let mut lane = store.write_lane::<TransformDesc>().expect("transform lane");
            lane.get_mut(near).expect("near transform").translation = Vec3::new(100.0, 0.0, 0.0);
        }
        click_frame(&mut engine, true, [100.0, 100.0]);
        assert!(!has::<Selected>(&engine, near));
        assert!(has::<Selected>(&engine, far));

        // A click into empty viewport clears the selection.
        click_frame(&mut engine, false, [100.0, 100.0]);
        click_frame(&mut engine, true, [199.0, 199.0]);
        assert!(!has::<Selected>(&engine, near));
        assert!(!has::<Selected>(&engine, far));
    }

    /// Chrome in front of scene content never steals a click in E1.
    #[test]
    fn chrome_never_steals_click() {
        let mut engine = pick_engine();
        let scene = spawn_scene(&mut engine, [0.0, 0.0, 0.0]);
        let gizmo = spawn_scene(&mut engine, [0.0, 0.0, 3.0]);
        engine
            .world()
            .store()
            .expect("world store")
            .write_lane::<EditorOnly>()
            .expect("chrome lane")
            .insert(gizmo, EditorOnly);

        click_frame(&mut engine, true, [100.0, 100.0]);
        assert!(has::<Selected>(&engine, scene));
        assert!(!has::<Selected>(&engine, gizmo));
    }

    /// Hover follows the pointer rest position without any press and never
    /// touches the selection.
    #[test]
    fn hover_follows_pointer_without_press() {
        let mut engine = pick_engine();
        let target = spawn_scene(&mut engine, [0.0, 0.0, 0.0]);

        click_frame(&mut engine, false, [100.0, 100.0]);
        assert!(has::<Hovered>(&engine, target));
        assert!(!has::<Selected>(&engine, target));

        click_frame(&mut engine, false, [199.0, 199.0]);
        assert!(!has::<Hovered>(&engine, target));
    }

    /// Without a viewport camera the authoritative host no-ops: no panic,
    /// no selection, typed `NoCamera` on the direct call.
    #[test]
    fn click_without_camera_is_noop() {
        let mut engine = Engine::new();
        install_editor(&mut engine);
        if engine.world().resources().get::<InputState>().is_none() {
            let _ = engine.world_mut().insert(InputState::default());
        }
        let store = engine.world_mut().store_mut().expect("world store");
        store.register::<TransformDesc>();
        store.register::<MeshDesc>();
        let entity = store.create_entity();
        store.insert(entity, TransformDesc::IDENTITY);
        store.insert(
            entity,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            },
        );
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("input resource");
            input.set_button(MouseButton::Left, true);
            input.set_pointer_position([100.0, 100.0]);
        }
        engine.run_frame(1.0 / 60.0);
        assert!(!has::<Selected>(&engine, entity));
        // Re-arm the edge (the frame above consumed it), then the direct
        // call reports the typed failure.
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("input resource");
            input.set_button(MouseButton::Left, false);
        }
        engine.run_frame(1.0 / 60.0);
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("input resource");
            input.set_button(MouseButton::Left, true);
        }
        assert_eq!(
            editor_pick_click(engine.world().resources()),
            Some(Err(PickError::NoCamera))
        );
    }
}
