//! Domain systems that connect the core frame host to builtin physics and
//! backend-neutral render extraction.
//!
//! The runtime keeps `BuiltinPhysicsEngine` as a domain representation while
//! `TransformDesc` and `RigidBody` remain ECS components in the logical
//! [`ornis_core::World`]. Physics systems make the sync-in/step/sync-out
//! boundary explicit; render extraction turns the same ECS lanes into a
//! backend-neutral snapshot. GPU resource ownership and editor protocol
//! details remain outside this module.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Mutex;

use glam::{Quat, Vec3};
use ornis_core::{
    ComponentStore, Engine, Entity, Resources, SmartStore, System, SystemAccess, Time,
};
use ornis_physics::{BodyHandle, BodyType, BuiltinPhysicsEngine, PhysicsEngine, RigidBody};
use ornis_render::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_render::{RenderExtracted, install_render_extract};

/// Physics domain state registered in a core [`Engine`] as a resource.
///
/// The builtin solver owns its optimized body array and the map keeps the
/// association with generational ECS entities. ECS `RigidBody` components are
/// synchronized at the system boundary rather than exposing physics' internal
/// vector to other domains.
pub struct PhysicsRuntime {
    solver: BuiltinPhysicsEngine,
    bindings: HashMap<Entity, BodyHandle>,
    changed: bool,
}

impl PhysicsRuntime {
    /// Creates a physics runtime with world-space gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            solver: BuiltinPhysicsEngine::new(gravity),
            bindings: HashMap::new(),
            changed: false,
        }
    }

    fn sync_in(
        &mut self,
        bodies: &ComponentStore<RigidBody>,
        transforms: Option<&ComponentStore<TransformDesc>>,
    ) {
        self.remove_stale_bindings(bodies);

        for (&entity, source) in bodies.entities.iter().zip(&bodies.data) {
            if let Some(&handle) = self.bindings.get(&entity) {
                self.sync_external_pose(
                    handle,
                    source,
                    transforms.and_then(|lane| lane.get(entity)),
                );
                continue;
            }

            let mut body = source.clone();
            if let Some(transform) = transforms.and_then(|lane| lane.get(entity)) {
                apply_transform_to_body(&mut body, transform);
            }
            let handle = self.solver.add_body(body);
            self.bindings.insert(entity, handle);
        }
    }

    fn remove_stale_bindings(&mut self, bodies: &ComponentStore<RigidBody>) {
        let mut stale: Vec<(Entity, BodyHandle)> = self
            .bindings
            .iter()
            .filter(|(entity, _)| !bodies.contains(**entity))
            .map(|(&entity, &handle)| (entity, handle))
            .collect();
        stale.sort_unstable_by_key(|&(_, handle)| Reverse(handle));

        for (entity, handle) in stale {
            let last = self.bindings.len().saturating_sub(1);
            let moved = if handle < last {
                self.bindings
                    .iter()
                    .find_map(|(&candidate, &bound)| (bound == last).then_some(candidate))
            } else {
                None
            };
            self.solver.remove_body(handle);
            self.bindings.remove(&entity);
            if let Some(moved) = moved {
                self.bindings.insert(moved, handle);
            }
            self.changed = true;
        }
    }

    fn sync_external_pose(
        &mut self,
        handle: BodyHandle,
        source: &RigidBody,
        transform: Option<&TransformDesc>,
    ) {
        let Some(body) = self.solver.get_body_mut(handle) else {
            return;
        };
        // Static and kinematic bodies are editor-controlled. Dynamic bodies
        // are authoritative in the solver after their initial registration.
        if matches!(body.body_type, BodyType::Static | BodyType::Kinematic)
            && let Some(transform) = transform
        {
            apply_transform_to_body(body, transform);
        }
        // A newly edited body role/material is reflected at the next sync
        // only when the ECS source differs from the solver representation.
        if body.body_type != source.body_type {
            *body = source.clone();
            if let Some(transform) = transform {
                apply_transform_to_body(body, transform);
            }
        }
    }

    fn step(&mut self, delta_seconds: f32) {
        let before: Vec<(Entity, Vec3, Quat)> = self
            .bindings
            .iter()
            .filter_map(|(&entity, &handle)| {
                self.solver
                    .get_body(handle)
                    .map(|body| (entity, body.position, body.orientation))
            })
            .collect();

        if delta_seconds > 0.0 {
            self.solver.step(delta_seconds);
        }

        self.changed |= before.iter().any(|(entity, position, orientation)| {
            let Some(&handle) = self.bindings.get(entity) else {
                return true;
            };
            let Some(body) = self.solver.get_body(handle) else {
                return true;
            };
            body.position != *position || body.orientation != *orientation
        });
    }

    fn sync_out(
        &mut self,
        bodies: &mut ComponentStore<RigidBody>,
        transforms: &mut ComponentStore<TransformDesc>,
    ) {
        for (&entity, &handle) in &self.bindings {
            let Some(body) = self.solver.get_body(handle) else {
                continue;
            };
            if let Some(destination) = bodies.get_mut(entity) {
                *destination = body.clone();
            }
            if let Some(destination) = transforms.get_mut(entity) {
                destination.translation = body.position.to_array();
                destination.rotation = [
                    body.orientation.x,
                    body.orientation.y,
                    body.orientation.z,
                    body.orientation.w,
                ];
            }
        }
    }

    pub(crate) fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }
}

/// Installs the physics resource and its sync/step/sync systems in `engine`.
///
/// The ECS must contain `RigidBody` and `TransformDesc` lanes for entities
/// that should participate. The editor registers static rigid bodies for
/// renderable scene entities; callers can insert dynamic bodies before the
/// first frame or through their own domain command.
pub fn install_physics(engine: &mut Engine, gravity: Vec3) {
    let _ = engine
        .world_mut()
        .insert(Mutex::new(PhysicsRuntime::new(gravity)));
    engine
        .schedule_mut()
        .add_system(PhysicsSyncIn)
        .add_system(PhysicsStep)
        .add_system(PhysicsSyncOut);
}

/// ECS → physics synchronization system.
struct PhysicsSyncIn;

impl System for PhysicsSyncIn {
    fn name(&self) -> &'static str {
        "physics_sync_in"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<RigidBody>()
            .reads_lane::<TransformDesc>()
            .writes::<Mutex<PhysicsRuntime>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(body_lane) = store.read_lane::<RigidBody>() else {
            return;
        };
        let transforms = store.read_lane::<TransformDesc>();
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let mut runtime = runtime_resource.lock().expect("physics runtime lock");
        runtime.sync_in(&body_lane, transforms.as_deref());
    }
}

/// Advances the physics domain with the frame delta.
struct PhysicsStep;

impl System for PhysicsStep {
    fn name(&self) -> &'static str {
        "physics_step"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<Time>()
            .writes::<Mutex<PhysicsRuntime>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(time) = resources.get::<Time>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        runtime_resource
            .lock()
            .expect("physics runtime lock")
            .step(time.delta_seconds());
    }
}

/// Physics → ECS synchronization system.
struct PhysicsSyncOut;

impl System for PhysicsSyncOut {
    fn name(&self) -> &'static str {
        "physics_sync_out"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .writes::<SmartStore>()
            .reads::<Mutex<PhysicsRuntime>>()
            .writes_lane::<RigidBody>()
            .writes_lane::<TransformDesc>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let Some(mut body_lane) = store.write_lane::<RigidBody>() else {
            return;
        };
        let Some(mut transform_lane) = store.write_lane::<TransformDesc>() else {
            return;
        };
        runtime_resource
            .lock()
            .expect("physics runtime lock")
            .sync_out(&mut body_lane, &mut transform_lane);
    }
}

fn normalized_rotation(rotation: [f32; 4]) -> Quat {
    let orientation = Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    let length_squared = orientation.length_squared();
    if length_squared.is_finite() && length_squared > 1e-12 {
        orientation.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Applies an ECS transform to a physics body's pose.
pub(crate) fn apply_transform_to_body(body: &mut RigidBody, transform: &TransformDesc) {
    body.position = Vec3::from_array(transform.translation);
    body.orientation = normalized_rotation(transform.rotation);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamic_body(position: Vec3) -> RigidBody {
        RigidBody::new_sphere(position, 0.5, 1.0)
    }

    fn transform(position: Vec3) -> TransformDesc {
        TransformDesc {
            translation: position.to_array(),
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }

    #[test]
    fn installed_systems_move_dynamic_body_back_into_ecs() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, dynamic_body(Vec3::ZERO));
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, transform(Vec3::ZERO));
        install_physics(&mut engine, Vec3::new(0.0, -9.81, 0.0));

        engine.run_frame(1.0 / 60.0);

        let store = engine.world().store().expect("world store");
        let lane = store.read_lane::<TransformDesc>().expect("transform lane");
        assert!(lane.get(entity).expect("entity transform").translation[1] < 0.0);
    }

    #[test]
    fn systems_preserve_static_body_pose() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            RigidBody::new_sphere(Vec3::new(2.0, 3.0, 4.0), 0.5, 0.0),
        );
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, transform(Vec3::new(2.0, 3.0, 4.0)));
        install_physics(&mut engine, Vec3::new(0.0, -9.81, 0.0));

        engine.run_frame(1.0 / 60.0);

        let store = engine.world().store().expect("world store");
        let lane = store.read_lane::<TransformDesc>().expect("transform lane");
        assert_eq!(
            lane.get(entity).expect("entity transform").translation,
            [2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn removing_component_removes_physics_binding() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, dynamic_body(Vec3::ZERO));
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, transform(Vec3::ZERO));
        install_physics(&mut engine, Vec3::ZERO);
        engine.run_frame(1.0 / 60.0);

        let removed = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .write_lane::<RigidBody>()
            .expect("rigid-body lane")
            .remove(entity);
        assert!(removed.is_some());
        engine.run_frame(1.0 / 60.0);

        let runtime = engine
            .world()
            .resources()
            .get::<Mutex<PhysicsRuntime>>()
            .expect("physics resource")
            .lock()
            .expect("physics runtime lock");
        assert_eq!(runtime.bindings.len(), 0);
    }

    #[test]
    fn removing_middle_body_preserves_swap_remove_bindings() {
        let mut engine = Engine::new();
        let mut entities = Vec::new();
        for i in 0..4 {
            let entity = engine
                .world_mut()
                .store_mut()
                .expect("world store")
                .create_entity();
            engine
                .world_mut()
                .store_mut()
                .expect("world store")
                .insert(entity, dynamic_body(Vec3::new(i as f32, 0.0, 0.0)));
            engine
                .world_mut()
                .store_mut()
                .expect("world store")
                .insert(entity, transform(Vec3::new(i as f32, 0.0, 0.0)));
            entities.push(entity);
        }
        install_physics(&mut engine, Vec3::ZERO);
        engine.run_frame(0.0);

        let removed = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .write_lane::<RigidBody>()
            .expect("rigid-body lane")
            .remove(entities[1]);
        assert!(removed.is_some());
        engine.run_frame(0.0);

        let runtime = engine
            .world()
            .resources()
            .get::<Mutex<PhysicsRuntime>>()
            .expect("physics resource")
            .lock()
            .expect("physics runtime lock");
        assert_eq!(runtime.bindings.get(&entities[0]), Some(&0));
        assert_eq!(runtime.bindings.get(&entities[2]), Some(&2));
        assert_eq!(runtime.bindings.get(&entities[3]), Some(&1));
    }

    #[test]
    fn render_extract_collects_complete_ecs_entities() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, transform(Vec3::new(1.0, 2.0, 3.0)));
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MeshDesc::Sphere {
                radius: 2.0,
                segments: 48,
                rings: 32,
            },
        );
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MaterialDesc::Metal {
                base_color: [0.9, 0.8, 0.2],
                roughness: 0.2,
            },
        );
        install_render_extract(&mut engine);

        engine.run_frame(0.0);

        let extracted = engine
            .world()
            .resources()
            .get::<Mutex<RenderExtracted>>()
            .expect("render extraction resource")
            .lock()
            .expect("render extraction lock")
            .clone();
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(extracted.materials.len(), 1);
        assert_eq!(extracted.instances.len(), 1);
        assert_eq!(extracted.instances[0].material_index, 0);
        assert_eq!(
            extracted.instances[0].model_matrix.w_axis.truncate(),
            Vec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn render_extract_skips_entities_without_complete_render_components() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, transform(Vec3::ZERO));
        install_render_extract(&mut engine);

        engine.run_frame(0.0);

        let extracted = engine
            .world()
            .resources()
            .get::<Mutex<RenderExtracted>>()
            .expect("render extraction resource")
            .lock()
            .expect("render extraction lock");
        assert!(extracted.instances.is_empty());
        assert!(extracted.materials.is_empty());
    }

    #[test]
    fn physics_accesses_are_declared_for_schedule_enforcement() {
        assert!(
            PhysicsSyncIn
                .access()
                .reads_lanes
                .contains(&std::any::TypeId::of::<RigidBody>())
        );
        assert!(
            PhysicsSyncOut
                .access()
                .writes_lanes
                .contains(&std::any::TypeId::of::<TransformDesc>())
        );
    }
}
