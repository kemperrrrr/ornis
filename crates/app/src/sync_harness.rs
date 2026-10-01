//! Generic reflection-driven physics sync harness.
//!
//! The typed bridges in the parent module (`velocity_to_body`,
//! `body_to_transform`) are the optimized SI route: hand-written systems
//! over typed lanes. This module re-expresses the same shape declaratively
//! so the Nth physics engine costs mappings, not new bridge systems:
//! [`SyncMapping`] (source lane + path → destination lane + path) executed
//! by [`apply_sync_mappings`] over the [`ComponentRegistry`] field surface.
//!
//! Boundaries (deliberate):
//!
//! * Tooling-grade throughput — every hop round-trips the canonical JSON
//!   form (exact for `f32` via `ryu`). The hot SI path stays typed; the
//!   harness is the generic route for other engines and scenes where the
//!   bridge cost is acceptable.
//! * Lifecycle stays out: entities lacking the destination component are
//!   skipped (counted), never constructed — create components with
//!   [`Mutation::Set`](ornis_core::mutation::Mutation::Set) first. The one
//!   intentional divergence from the typed `body_to_transform`, which
//!   inserts a missing [`Position`](ornis_gameplay::Position).
//! * Units stay in the typed world: mappings must already agree in units
//!   (m/s → m/s, meters → meters). Reflection carries values, the
//!   integrator owns their meaning.
//! * Phases stay in the schedule: intent mappings belong in the fixed
//!   schedule, placement mappings in the frame schedule —
//!   [`install_sync_harness`] takes both lists separately.
//!
//! [`RigidBody`](ornis_physics::RigidBody) is not serializable by design
//! (mass-triple invariants, non-serde [`Shape`](ornis_physics::Shape)), so
//! it enters the registry through a hand-written [`FieldSurface`](ornis_core::FieldSurface):
//! `position`/`orientation`/`body_type` readable, `velocity` writable with
//! the static-body guard (a static skips with `Ok(false)`, mirroring the
//! typed bridge). Register it with [`register_rigid_body_fields`].

use std::any::TypeId;

use glam::{Quat, Vec3};
use ornis_core::{
    ComponentName, ComponentRegistry, Engine, Entity, FieldMeta, FieldPath, FieldSegment,
    FieldSurface, RegistryError, Resources, SmartStore, System, SystemAccess,
};
use ornis_physics::{BodyType, RigidBody};
use serde_json::{Value, json};

/// Spatial components in a velocity / position vector.
const VEC3_COMPONENTS: usize = 3;

/// Registry name of the [`RigidBody`] field surface.
pub const RIGID_BODY_NAME: &str = "RigidBody";

/// Addressable [`RigidBody`] schema: solver-owned pose readable, intent
/// (`velocity`) writable with the static-body guard.
static RIGID_BODY_FIELDS: &[FieldMeta] = &[
    FieldMeta {
        name: "position",
        path: "position",
        type_name: "[f32; 3]",
        writable: false,
    },
    FieldMeta {
        name: "orientation",
        path: "orientation",
        type_name: "[f32; 4]",
        writable: false,
    },
    FieldMeta {
        name: "velocity",
        path: "velocity",
        type_name: "[f32; 3]",
        writable: true,
    },
    FieldMeta {
        name: "body_type",
        path: "body_type",
        type_name: "BodyType",
        writable: false,
    },
];

fn vec3_json(v: Vec3) -> Value {
    json!([v.x, v.y, v.z])
}

fn quat_json(q: Quat) -> Value {
    json!([q.x, q.y, q.z, q.w])
}

fn body_type_json(body_type: BodyType) -> Value {
    Value::String(
        match body_type {
            BodyType::Static => "Static",
            BodyType::Dynamic => "Dynamic",
            BodyType::Kinematic => "Kinematic",
        }
        .to_string(),
    )
}

/// Runs `f` against the entity's body (`None` — no such body);
/// an `Err(())` address becomes [`RegistryError::UnknownField`].
fn with_body(
    store: &SmartStore,
    entity: Entity,
    component: &ComponentName,
    path: &FieldPath,
    f: impl FnOnce(&RigidBody) -> Result<Value, ()>,
) -> Result<Option<Value>, RegistryError> {
    let Some(lane) = store.read_lane::<RigidBody>() else {
        return Ok(None);
    };
    let Some(body) = lane.get(entity) else {
        return Ok(None);
    };
    f(body).map(Some).map_err(|()| RegistryError::UnknownField {
        component: component.clone(),
        path: path.clone(),
    })
}

/// Reads one whitelisted [`RigidBody`] path (`None` — no such body).
fn get_rigid_field(
    store: &SmartStore,
    entity: Entity,
    component: &ComponentName,
    path: &FieldPath,
) -> Result<Option<Value>, RegistryError> {
    with_body(store, entity, component, path, |body| {
        let mut segments = path.segments();
        let head = segments.next().ok_or(())?;
        let rest: Vec<_> = segments.collect();
        match (head, rest.as_slice()) {
            (FieldSegment::Field("position"), []) => Ok(vec3_json(body.position)),
            (FieldSegment::Field("position"), [FieldSegment::Index(i)]) => {
                Ok(json!(body.position.to_array().get(*i).copied().ok_or(())?))
            }
            (FieldSegment::Field("orientation"), []) => Ok(quat_json(body.orientation)),
            (FieldSegment::Field("orientation"), [FieldSegment::Index(i)]) => Ok(json!(
                body.orientation.to_array().get(*i).copied().ok_or(())?
            )),
            (FieldSegment::Field("velocity"), []) => Ok(vec3_json(body.velocity)),
            (FieldSegment::Field("velocity"), [FieldSegment::Index(i)]) => {
                Ok(json!(body.velocity.to_array().get(*i).copied().ok_or(())?))
            }
            (FieldSegment::Field("body_type"), []) => Ok(body_type_json(body.body_type)),
            _ => Err(()),
        }
    })
}

/// Writes `velocity` (whole or indexed) unless the body is static.
/// Position, orientation and `body_type` are solver-owned: read-only.
fn set_rigid_field(
    store: &SmartStore,
    entity: Entity,
    component: &ComponentName,
    path: &FieldPath,
    value: &Value,
) -> Result<bool, RegistryError> {
    let unknown = || RegistryError::UnknownField {
        component: component.clone(),
        path: path.clone(),
    };
    let mut segments = path.segments();
    let writable = match (segments.next(), segments.collect::<Vec<_>>().as_slice()) {
        (Some(FieldSegment::Field("velocity")), []) => None,
        (Some(FieldSegment::Field("velocity")), [FieldSegment::Index(i)]) => Some(*i),
        _ => return Err(unknown()),
    };
    let Some(mut lane) = store.write_lane::<RigidBody>() else {
        return Err(RegistryError::MissingLane {
            component: component.clone(),
        });
    };
    let Some(body) = lane.get_mut(entity) else {
        return Err(RegistryError::MissingComponent {
            component: component.clone(),
            id: entity.id(),
            generation: entity.generation(),
        });
    };
    // Only kinematic/dynamic bodies follow external intent; static bodies
    // remain editor-controlled (mirrors `velocity_to_body`).
    if body.body_type == BodyType::Static {
        return Ok(false);
    }
    match writable {
        None => {
            let xyz: [f32; VEC3_COMPONENTS] =
                serde_json::from_value(value.clone()).map_err(RegistryError::from_json)?;
            body.velocity = Vec3::from_array(xyz);
        }
        Some(i) => {
            let v: f32 = serde_json::from_value(value.clone()).map_err(RegistryError::from_json)?;
            let mut xyz = body.velocity.to_array();
            *xyz.get_mut(i).ok_or_else(unknown)? = v;
            body.velocity = Vec3::from_array(xyz);
        }
    }
    Ok(true)
}

/// Registers the [`RigidBody`] field surface (`position`/`orientation`/
/// `body_type` readable, `velocity` writable with the static guard).
/// Whole-value JSON stays unsupported — bodies enter the world typed.
pub fn register_rigid_body_fields(registry: &mut ComponentRegistry) -> &mut ComponentRegistry {
    registry.register_field_surface::<RigidBody>(
        RIGID_BODY_NAME,
        FieldSurface {
            fields: RIGID_BODY_FIELDS,
            get_field: get_rigid_field,
            set_field: set_rigid_field,
        },
    )
}

/// One declarative field copy: source lane + path → destination lane +
/// path, applied per entity carrying both (destination-absent entities
/// are skipped). Values travel through the canonical JSON form, so both
/// ends must already agree in units.
#[derive(Debug, Clone)]
pub struct SyncMapping {
    /// Source component (registry name).
    pub src_component: ComponentName,
    /// Source path inside the canonical form (`velocity.0`).
    pub src_path: FieldPath,
    /// Destination component (registry name).
    pub dst_component: ComponentName,
    /// Destination path inside the canonical form.
    pub dst_path: FieldPath,
}

impl SyncMapping {
    /// Declares one field copy.
    pub fn new(
        src_component: ComponentName,
        src_path: FieldPath,
        dst_component: ComponentName,
        dst_path: FieldPath,
    ) -> Self {
        Self {
            src_component,
            src_path,
            dst_component,
            dst_path,
        }
    }
}

/// Outcome of [`apply_sync_mappings`]: writes performed vs entities
/// skipped (destination-absent, source-empty, or guard-skipped).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SyncReport {
    /// Field writes performed.
    pub applied: usize,
    /// Entities skipped without error.
    pub skipped: usize,
}

/// Applies `mappings` in order, deterministically (dense source order).
/// A missing lane on either side skips the mapping — level-triggered sync
/// semantics: there is nothing to carry yet. Configuration bugs
/// (unknown component at a present lane, unknown field) fail fast.
pub fn apply_sync_mappings(
    store: &SmartStore,
    registry: &ComponentRegistry,
    mappings: &[SyncMapping],
) -> Result<SyncReport, RegistryError> {
    let mut report = SyncReport::default();
    for mapping in mappings {
        let Some(src) = registry.by_component(&mapping.src_component) else {
            return Err(RegistryError::UnknownComponent(
                mapping.src_component.clone(),
            ));
        };
        let Some(dst) = registry.by_component(&mapping.dst_component) else {
            return Err(RegistryError::UnknownComponent(
                mapping.dst_component.clone(),
            ));
        };
        // Read phase (shared): collect values first so the write phase
        // never meets a borrowed lane.
        let mut batch = Vec::new();
        let mut read_error = None;
        src.visit_entities(store, &mut |entity| {
            if read_error.is_some() || !dst.contains(store, entity) {
                return;
            }
            match src.get_field(store, entity, &mapping.src_path) {
                Ok(Some(value)) => batch.push((entity, value)),
                Ok(None) => {}
                Err(error) => read_error = Some(error),
            }
        });
        if let Some(error) = read_error {
            return Err(error);
        }
        // Write phase.
        for (entity, value) in batch {
            match dst.set_field(store, entity, &mapping.dst_path, &value) {
                Ok(true) => report.applied += 1,
                Ok(false) => report.skipped += 1,
                Err(RegistryError::MissingLane { .. }) => report.skipped += 1,
                Err(RegistryError::MissingComponent { .. }) => report.skipped += 1,
                Err(error) => return Err(error),
            }
        }
    }
    Ok(report)
}

/// [`apply_sync_mappings`] as a schedule system, with precise lane access
/// declared from the registry (`reads_lane_id`/`writes_lane_id`).
/// Construction resolves every mapping component eagerly, so unknown
/// names fail at install, never mid-frame.
pub struct SyncHarnessSystem {
    registry: &'static ComponentRegistry,
    mappings: Vec<SyncMapping>,
    label: &'static str,
    reads: Vec<TypeId>,
    writes: Vec<TypeId>,
}

impl SyncHarnessSystem {
    /// Builds the system, resolving mapping components now (fail fast).
    ///
    /// # Errors
    /// [`RegistryError::UnknownComponent`] for an unregistered mapping end.
    pub fn new(
        registry: &'static ComponentRegistry,
        label: &'static str,
        mappings: Vec<SyncMapping>,
    ) -> Result<Self, RegistryError> {
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        for mapping in &mappings {
            let Some(src) = registry.by_component(&mapping.src_component) else {
                return Err(RegistryError::UnknownComponent(
                    mapping.src_component.clone(),
                ));
            };
            let Some(dst) = registry.by_component(&mapping.dst_component) else {
                return Err(RegistryError::UnknownComponent(
                    mapping.dst_component.clone(),
                ));
            };
            reads.push(src.type_id());
            writes.push(dst.type_id());
        }
        Ok(Self {
            registry,
            mappings,
            label,
            reads,
            writes,
        })
    }
}

impl System for SyncHarnessSystem {
    fn name(&self) -> &'static str {
        self.label
    }

    fn access(&self) -> SystemAccess {
        let mut access = SystemAccess::new().reads::<SmartStore>();
        for id in &self.reads {
            access = access.reads_lane_id(*id);
        }
        for id in &self.writes {
            access = access.writes_lane_id(*id);
        }
        access
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        apply_sync_mappings(store, self.registry, &self.mappings)
            .expect("sync mappings validated at install");
    }
}

/// Installs the generic harness: `fixed` mappings into the fixed schedule
/// (`sync_harness_fixed`), `frame` mappings into the frame schedule
/// (`sync_harness_frame`). Empty lists install nothing.
///
/// The registry must outlive the engine (`&'static`, e.g. the host's
/// `LazyLock` registry) because systems own it.
pub fn install_sync_harness(
    engine: &mut Engine,
    registry: &'static ComponentRegistry,
    fixed: Vec<SyncMapping>,
    frame: Vec<SyncMapping>,
) -> Result<(), RegistryError> {
    if !fixed.is_empty() {
        let system = SyncHarnessSystem::new(registry, "sync_harness_fixed", fixed)?;
        engine.fixed_schedule_mut().add_system(system);
    }
    if !frame.is_empty() {
        let system = SyncHarnessSystem::new(registry, "sync_harness_frame", frame)?;
        engine.schedule_mut().add_system(system);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_assets::scene::TransformDesc;
    use ornis_gameplay::{Position, Velocity};

    use crate::{BodyToTransformSystem, VelocityToBodySystem, install_gameplay_physics_bridge};

    fn rigid_body_name() -> ComponentName {
        ComponentName::from_static(RIGID_BODY_NAME)
    }

    fn harness_registry() -> ComponentRegistry {
        let mut registry = ComponentRegistry::new();
        registry.register_component::<Velocity>();
        registry.register_component::<Position>();
        registry.register::<TransformDesc>("Transform");
        register_rigid_body_fields(&mut registry);
        registry
    }

    fn intent_mappings() -> Vec<SyncMapping> {
        vec![
            SyncMapping::new(
                ComponentName::from_static("Velocity"),
                FieldPath::from_static("0"),
                ComponentName::from_static(RIGID_BODY_NAME),
                FieldPath::from_static("velocity.0"),
            ),
            SyncMapping::new(
                ComponentName::from_static("Velocity"),
                FieldPath::from_static("2"),
                ComponentName::from_static(RIGID_BODY_NAME),
                FieldPath::from_static("velocity.2"),
            ),
        ]
    }

    fn placement_mappings() -> Vec<SyncMapping> {
        let pairs = [
            ("position.0", "Position", "0"),
            ("position.1", "Position", "1"),
            ("position.2", "Position", "2"),
            ("position.0", "Transform", "translation.0"),
            ("position.1", "Transform", "translation.1"),
            ("position.2", "Transform", "translation.2"),
            ("orientation.0", "Transform", "rotation.0"),
            ("orientation.1", "Transform", "rotation.1"),
            ("orientation.2", "Transform", "rotation.2"),
            ("orientation.3", "Transform", "rotation.3"),
        ];
        pairs
            .into_iter()
            .map(|(src, dst_component, dst)| {
                SyncMapping::new(
                    ComponentName::from_static(RIGID_BODY_NAME),
                    FieldPath::from_static(src),
                    ComponentName::from_static(dst_component),
                    FieldPath::from_static(dst),
                )
            })
            .collect()
    }

    fn static_leak_registry() -> &'static ComponentRegistry {
        use std::sync::LazyLock;
        static REGISTRY: LazyLock<ComponentRegistry> = LazyLock::new(harness_registry);
        &REGISTRY
    }

    fn seed_scene(store: &mut SmartStore) -> [Entity; 5] {
        let dynamic = store.create_entity();
        store.insert(dynamic, Velocity(Vec3::new(3.0, 0.0, 4.0)));
        store.insert(dynamic, RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        store.insert(dynamic, Position(Vec3::ZERO));
        store.insert(
            dynamic,
            TransformDesc {
                translation: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        );

        let stuck = store.create_entity();
        store.insert(stuck, Velocity(Vec3::new(5.0, 0.0, 6.0)));
        let mut heavy = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        heavy.make_static();
        store.insert(stuck, heavy);
        store.insert(stuck, Position(Vec3::ZERO));
        store.insert(
            stuck,
            TransformDesc {
                translation: [0.0; 3],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
            },
        );

        let disembodied = store.create_entity();
        store.insert(disembodied, Velocity(Vec3::new(1.0, 2.0, 3.0)));

        let novelocity = store.create_entity();
        store.insert(novelocity, RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        store.insert(novelocity, Position(Vec3::ZERO));

        let nodesc = store.create_entity();
        store.insert(nodesc, RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        store.insert(nodesc, Position(Vec3::ZERO));

        [dynamic, stuck, disembodied, novelocity, nodesc]
    }

    fn snapshot(store: &SmartStore) -> Vec<(Vec3, Vec3, Vec3, Option<TransformDesc>)> {
        let bodies = store.read_lane::<RigidBody>().unwrap();
        let positions = store.read_lane::<Position>().unwrap();
        let descs = store.read_lane::<TransformDesc>().unwrap();
        let mut out = Vec::new();
        for entity in &bodies.entities {
            let body = bodies.get(*entity).unwrap();
            out.push((
                body.velocity,
                body.position,
                positions.get(*entity).map(|p| p.0).unwrap_or(Vec3::NAN),
                descs.get(*entity).cloned(),
            ));
        }
        out
    }

    #[test]
    fn harness_matches_typed_bridges_through_the_schedule() {
        // Twin engines: typed bridges vs declarative harness, same seed.
        let mut typed = Engine::new();
        install_gameplay_physics_bridge(&mut typed);
        seed_scene(typed.world_mut().store_mut().unwrap());

        let mut generic = Engine::new();
        install_sync_harness(
            &mut generic,
            static_leak_registry(),
            intent_mappings(),
            placement_mappings(),
        )
        .unwrap();
        let entities = seed_scene(generic.world_mut().store_mut().unwrap());

        typed.run_frame(1.0 / 60.0);
        generic.run_frame(1.0 / 60.0);

        let left = snapshot(typed.world().store().unwrap());
        let right = snapshot(generic.world().store().unwrap());
        assert_eq!(left.len(), right.len());
        for (index, (a, b)) in left.iter().zip(right.iter()).enumerate() {
            assert_eq!(a.0, b.0, "velocity of body {index}");
            assert_eq!(a.1, b.1, "position of body {index}");
            assert_eq!(a.2, b.2, "gameplay position of body {index}");
            assert_eq!(
                a.3.as_ref().map(|d| (d.translation, d.rotation)),
                b.3.as_ref().map(|d| (d.translation, d.rotation)),
                "transform of body {index}"
            );
        }

        // Static body kept its solver velocity in both worlds.
        let lane = generic
            .world()
            .store()
            .unwrap()
            .read_lane::<RigidBody>()
            .unwrap();
        assert_eq!(lane.get(entities[1]).unwrap().velocity, Vec3::ZERO);
    }

    #[test]
    fn harness_runs_typed_systems_side_by_side() {
        // The harness and the typed systems agree when driven directly
        // (unit-level parity without the schedule).
        let registry = harness_registry();

        let mut typed_store = SmartStore::new();
        seed_scene(&mut typed_store);
        let mut typed_resources = Resources::new();
        typed_resources.insert(typed_store);
        let typed_resources = typed_resources;
        VelocityToBodySystem.run(&typed_resources);
        BodyToTransformSystem.run(&typed_resources);
        let typed = snapshot(typed_resources.get::<SmartStore>().unwrap());

        let mut generic_store = SmartStore::new();
        seed_scene(&mut generic_store);
        let report = apply_sync_mappings(&generic_store, &registry, &intent_mappings()).unwrap();
        assert!(report.applied > 0);
        let report = apply_sync_mappings(&generic_store, &registry, &placement_mappings()).unwrap();
        assert!(report.applied > 0);
        let generic = snapshot(&generic_store);

        assert_eq!(typed.len(), generic.len());
        for (a, b) in typed.iter().zip(generic.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert_eq!(a.2, b.2);
        }
    }

    #[test]
    fn rigid_surface_guards_and_names_errors() {
        let registry = harness_registry();
        let meta = registry.by_component(&rigid_body_name()).unwrap();
        assert_eq!(meta.fields().len(), 4);
        assert!(
            meta.fields()
                .iter()
                .any(|f| f.name == "velocity" && f.writable)
        );
        assert!(
            meta.fields()
                .iter()
                .all(|f| f.writable == (f.name == "velocity"))
        );

        let mut heavy = RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0);
        heavy.make_static();
        // Custom surface writes need the lane: insert typed first.
        let mut owned = SmartStore::new();
        let stuck = {
            let e = owned.create_entity();
            owned.insert(e, heavy);
            e
        };
        let written = meta
            .set_field(
                &owned,
                stuck,
                &FieldPath::from_static("velocity.0"),
                &serde_json::json!(9.0),
            )
            .unwrap();
        assert!(!written, "static body skips velocity writes");
        assert_eq!(
            meta.get_field(&owned, stuck, &FieldPath::from_static("body_type"))
                .unwrap(),
            Some(serde_json::json!("Static"))
        );
        // Solver-owned pose is read-only.
        let denied = meta.set_field(
            &owned,
            stuck,
            &FieldPath::from_static("position.0"),
            &serde_json::json!(1.0),
        );
        assert!(matches!(denied, Err(RegistryError::UnknownField { .. })));
    }

    #[test]
    fn install_fails_fast_on_unknown_component() {
        let registry = harness_registry();
        let leaked: &'static ComponentRegistry = Box::leak(Box::new(registry));
        let bad = vec![SyncMapping::new(
            ComponentName::from_static("Ghost"),
            FieldPath::from_static("x"),
            ComponentName::from_static("Position"),
            FieldPath::from_static("0"),
        )];
        let mut engine = Engine::new();
        let error =
            install_sync_harness(&mut engine, leaked, bad, Vec::new()).expect_err("bad src");
        assert!(matches!(error, RegistryError::UnknownComponent(_)));
    }
}
