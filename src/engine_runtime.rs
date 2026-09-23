//! Domain systems that connect the core frame host to sequential-impulse physics,
//! XPBD soft bodies and backend-neutral render extraction.
//!
//! The runtime keeps `SequentialImpulseEngine` as a domain representation while
//! `TransformDesc` and `RigidBody` remain ECS components in the logical
//! [`ornis_core::World`]. Physics systems make the sync-in/step/sync-out
//! boundary explicit; render extraction turns the same ECS lanes into a
//! backend-neutral snapshot. GPU resource ownership and editor protocol
//! details remain outside this module.
//!
//! Soft bodies (PLAN B2/D1.5) ride the same seam through a dedicated
//! [`XpbdEngine`]: entities carrying a [`SoftBody`] lane component are bound
//! to solver handles, stepped with the fixed clock, and written back as
//! world-space [`MeshDesc::Custom`] soups (identity transform) that the
//! existing extraction path draws unchanged.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Mutex;

use glam::{Quat, Vec3};
#[cfg(test)]
use ornis_assets::scene::MaterialDesc;
use ornis_assets::scene::{MeshDesc, TransformDesc};
use ornis_core::{
    ComponentStore, Engine, Entity, FixedTime, Resources, SmartStore, System, SystemAccess,
};
use ornis_physics::{
    BodyHandle, BodyType, PhysicsEngine, RigidBody, SoftBody, SoftHandle, SolverKind, XpbdEngine,
};
#[cfg(test)]
use ornis_render::extract_render_data;

/// Physics domain state registered in a core [`Engine`] as a resource.
///
/// The sequential-impulse solver owns its optimized body array and the map keeps the
/// association with generational ECS entities. ECS `RigidBody` components are
/// synchronized at the system boundary rather than exposing physics' internal
/// vector to other domains. The common core engine host accumulates
/// render-frame time and invokes this domain at a bounded fixed 60 Hz
/// timestep.
///
/// Soft bodies live in a second solver ([`XpbdEngine`], PLAN B2/D1) with its
/// own entity bindings: the rigid orchestrator does not know particles, so
/// sharing one solver would silently drop the coupling. Mesh upload reads
/// the soft solver directly (world-space soup, identity transform).
pub struct PhysicsRuntime {
    solver: ornis_physics::Engine,
    bindings: HashMap<Entity, BodyHandle>,
    soft_solver: XpbdEngine,
    soft_bindings: HashMap<Entity, SoftHandle>,
    changed: bool,
}

impl PhysicsRuntime {
    /// Creates a physics runtime with world-space gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            solver: ornis_physics::Engine::new(SolverKind::SequentialImpulse, gravity),
            bindings: HashMap::new(),
            soft_solver: XpbdEngine::new(gravity),
            soft_bindings: HashMap::new(),
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
        // are authoritative in the solver after their initial registration,
        // except for velocity: gameplay intent (`velocity_to_body`) writes
        // it into the ECS lane, so it must be forwarded here or the bridge
        // dead-ends and `sync_out` pins the pose back every tick.
        if matches!(body.body_type, BodyType::Dynamic) {
            body.velocity = source.velocity;
            body.angular_velocity = source.angular_velocity;
        }
        if matches!(body.body_type, BodyType::Static | BodyType::Kinematic)
            && let Some(transform) = transform
        {
            apply_transform_to_body(body, transform);
        }
        // A newly edited body role/filter is reflected at the next sync
        // when the ECS source differs from the solver representation.
        if body.body_type != source.body_type
            || body.collision_layer != source.collision_layer
            || body.collision_mask != source.collision_mask
            || body.is_trigger != source.is_trigger
        {
            *body = source.clone();
            if let Some(transform) = transform {
                apply_transform_to_body(body, transform);
            }
        }
    }

    /// Advances the physics solver by one host-selected fixed update.
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

        self.solver.step(delta_seconds);
        self.soft_solver.step(delta_seconds);

        self.changed |= before.iter().any(|(entity, position, orientation)| {
            let Some(&handle) = self.bindings.get(entity) else {
                return true;
            };
            let Some(body) = self.solver.get_body(handle) else {
                return true;
            };
            body.position != *position || body.orientation != *orientation
        });
        self.changed |= !self.soft_bindings.is_empty();
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

    /// Binds newly added [`SoftBody`] lane components into the soft solver.
    /// Solver state is authoritative after registration (no per-step pose
    /// sync in D1 — there is no gameplay intent for particles yet).
    fn sync_soft_in(&mut self, soft: &ComponentStore<SoftBody>) {
        self.remove_stale_soft_bindings(soft);
        for (&entity, source) in soft.entities.iter().zip(&soft.data) {
            if self.soft_bindings.contains_key(&entity) {
                continue;
            }
            let handle = self.soft_solver.add_soft_body(source.clone());
            self.soft_bindings.insert(entity, handle);
        }
    }

    /// Drops bindings whose lane component is gone, remapping the
    /// swap-remove survivor exactly like the rigid path.
    fn remove_stale_soft_bindings(&mut self, soft: &ComponentStore<SoftBody>) {
        let mut stale: Vec<(Entity, SoftHandle)> = self
            .soft_bindings
            .iter()
            .filter(|(entity, _)| !soft.contains(**entity))
            .map(|(&entity, &handle)| (entity, handle))
            .collect();
        stale.sort_unstable_by_key(|&(_, handle)| Reverse(handle));

        for (entity, handle) in stale {
            let last = self.soft_bindings.len().saturating_sub(1);
            let moved = if handle < last {
                self.soft_bindings
                    .iter()
                    .find_map(|(&candidate, &bound)| (bound == last).then_some(candidate))
            } else {
                None
            };
            self.soft_solver.remove_soft_body(handle);
            self.soft_bindings.remove(&entity);
            if let Some(moved) = moved {
                self.soft_bindings.insert(moved, handle);
            }
            self.changed = true;
        }
    }

    /// Writes solver particle positions into [`MeshDesc::Custom`] soups and
    /// pins the entity transform to identity (particles are already
    /// world-space — the existing extraction path draws them unchanged).
    /// Bodies without a render surface (chains) are skipped: line
    /// rendering is out of D1 scope.
    fn sync_soft_out(
        &mut self,
        meshes: &mut ComponentStore<MeshDesc>,
        transforms: &mut ComponentStore<TransformDesc>,
    ) {
        for (&entity, &handle) in &self.soft_bindings {
            let Some(body) = self.soft_solver.get_soft_body(handle) else {
                continue;
            };
            if body.surface.is_empty() {
                continue;
            }
            let positions: Vec<[f32; 3]> = body
                .positions_snapshot()
                .iter()
                .map(Vec3::to_array)
                .collect();
            let indices: Vec<u32> = body
                .surface
                .iter()
                .flat_map(|tri| [tri[0] as u32, tri[1] as u32, tri[2] as u32])
                .collect();
            let desc = MeshDesc::Custom { positions, indices };
            if let Some(slot) = meshes.get_mut(entity) {
                *slot = desc;
            } else {
                meshes.insert(entity, desc);
            }
            if let Some(transform) = transforms.get_mut(entity) {
                transform.translation = [0.0, 0.0, 0.0];
                transform.rotation = [0.0, 0.0, 0.0, 1.0];
                transform.scale = [1.0, 1.0, 1.0];
            }
        }
    }
}

/// Installs the physics resource and its sync/step/sync systems in `engine`.
///
/// The ECS must contain `RigidBody` and `TransformDesc` lanes for entities
/// that should participate. The editor registers static rigid bodies for
/// renderable scene entities; callers can insert dynamic bodies before the
/// first frame or through their own domain command. The three systems are
/// registered in reverse order in the engine's fixed schedule, so each
/// host-selected fixed update performs sync-in → step → sync-out. The
/// variable-rate schedule (including render extraction) runs after all fixed
/// updates for the frame.
pub fn install_physics(engine: &mut Engine, gravity: Vec3) {
    let _ = engine
        .world_mut()
        .insert(Mutex::new(PhysicsRuntime::new(gravity)));
    // Prepends land at the front, so registration runs in reverse: the
    // final fixed order is sync-in → soft-sync-in → step → sync-out →
    // soft-sync-out (mesh upload last, reading settled solver state).
    engine
        .fixed_schedule_mut()
        .prepend_system(SoftSyncOut)
        .prepend_system(PhysicsSyncOut)
        .prepend_system(PhysicsStep)
        .prepend_system(SoftSyncIn)
        .prepend_system(PhysicsSyncIn);
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

/// Advances the physics domain by the host-selected fixed step.
struct PhysicsStep;

impl System for PhysicsStep {
    fn name(&self) -> &'static str {
        "physics_step"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<FixedTime>()
            .writes::<Mutex<PhysicsRuntime>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(time) = resources.get::<FixedTime>() else {
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

/// ECS → soft-solver synchronization system (PLAN B2/D1.5).
struct SoftSyncIn;

impl System for SoftSyncIn {
    fn name(&self) -> &'static str {
        "soft_sync_in"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<SoftBody>()
            .writes::<Mutex<PhysicsRuntime>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(soft_lane) = store.read_lane::<SoftBody>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let mut runtime = runtime_resource.lock().expect("physics runtime lock");
        runtime.sync_soft_in(&soft_lane);
    }
}

/// Soft-solver → mesh synchronization system (PLAN B2/D1.5): rewrites
/// [`MeshDesc::Custom`] soups from solver particles every fixed update.
struct SoftSyncOut;

impl System for SoftSyncOut {
    fn name(&self) -> &'static str {
        "soft_sync_out"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .writes::<SmartStore>()
            .reads::<Mutex<PhysicsRuntime>>()
            .writes_lane::<MeshDesc>()
            .writes_lane::<TransformDesc>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let Some(mut mesh_lane) = store.write_lane::<MeshDesc>() else {
            return;
        };
        let Some(mut transform_lane) = store.write_lane::<TransformDesc>() else {
            return;
        };
        runtime_resource
            .lock()
            .expect("physics runtime lock")
            .sync_soft_out(&mut mesh_lane, &mut transform_lane);
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
    fn physics_accumulator_runs_fixed_steps_after_partial_frames() {
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

        let fixed_delta = FixedTime::default().delta_seconds();
        engine.run_frame(fixed_delta * 0.51);
        let halfway = engine
            .world()
            .store()
            .expect("world store")
            .read_lane::<TransformDesc>()
            .expect("transform lane")
            .get(entity)
            .expect("entity transform")
            .translation[1];
        assert_eq!(halfway, 0.0);

        engine.run_frame(fixed_delta * 0.51);
        let after_step = engine
            .world()
            .store()
            .expect("world store")
            .read_lane::<TransformDesc>()
            .expect("transform lane")
            .get(entity)
            .expect("entity transform")
            .translation[1];
        assert!(after_step < 0.0);
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
        let fixed_delta = FixedTime::default().delta_seconds();
        engine.run_frame(fixed_delta);

        let removed = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .write_lane::<RigidBody>()
            .expect("rigid-body lane")
            .remove(entities[1]);
        assert!(removed.is_some());
        engine.run_frame(fixed_delta);

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
                emission: [0.0, 0.0, 0.0],
            },
        );

        engine.run_frame(0.0);

        // X4: direct lane read — no scheduled snapshot anymore.
        let extracted = extract_render_data(engine.world().store().expect("world store"));
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
    fn physics_sync_output_is_visible_to_render_lane_reads() {
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
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MeshDesc::Sphere {
                radius: 0.5,
                segments: 16,
                rings: 8,
            },
        );
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MaterialDesc::Dielectric {
                base_color: [0.5, 0.5, 0.5],
                roughness: 0.5,
                emission: [0.0, 0.0, 0.0],
            },
        );
        install_physics(&mut engine, Vec3::new(0.0, -9.81, 0.0));

        engine.run_frame(1.0 / 60.0);

        let transform_lane = engine
            .world()
            .store()
            .expect("world store")
            .read_lane::<TransformDesc>()
            .expect("transform lane");
        let transform = transform_lane.get(entity).expect("entity transform");
        assert!(transform.translation[1] < 0.0);
        // X4: the render side reads the same lanes directly — the physics
        // sync output must be visible to that read after the frame.
        let extracted = extract_render_data(engine.world().store().expect("world store"));
        assert_eq!(
            extracted.instances[0].model_matrix.w_axis.truncate().y,
            transform.translation[1]
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

        engine.run_frame(0.0);

        let extracted = extract_render_data(engine.world().store().expect("world store"));
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
        assert!(
            SoftSyncIn
                .access()
                .reads_lanes
                .contains(&std::any::TypeId::of::<SoftBody>())
        );
        assert!(
            SoftSyncOut
                .access()
                .writes_lanes
                .contains(&std::any::TypeId::of::<MeshDesc>())
        );
    }

    /// PLAN B2/D1.5: a cloth entity's `MeshDesc::Custom` tracks solver
    /// particles every frame, its transform is pinned to identity
    /// (world-space soup), and the existing extraction path draws it.
    #[test]
    fn soft_body_mesh_tracks_solver_particles() {
        use ornis_physics::{ClothPin, SoftBody};

        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        let origin = Vec3::new(0.0, 2.0, 0.0);
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            SoftBody::cloth_grid(origin, 4, 4, 0.25, 1.0, 0.0, 0.0, 1e-4, ClothPin::TopRow),
        );
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MeshDesc::Custom {
                positions: Vec::new(),
                indices: Vec::new(),
            },
        );
        // Deliberately non-identity: the bridge must reset it (world soup).
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .insert(entity, transform(Vec3::new(9.0, 9.0, 9.0)));
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MaterialDesc::Dielectric {
                base_color: [0.5, 0.5, 0.5],
                roughness: 0.5,
                emission: [0.0, 0.0, 0.0],
            },
        );
        install_physics(&mut engine, Vec3::new(0.0, -9.81, 0.0));

        // Kick the free particles sideways: a hanging sheet at rest is an
        // exact equilibrium (nothing would move), so the solver run is
        // proven by the resulting pendulum swing, not by gravity alone.
        engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .write_lane::<SoftBody>()
            .expect("soft lane")
            .get_mut(entity)
            .expect("entity soft body")
            .particles
            .iter_mut()
            .filter(|p| p.inv_mass > 0.0)
            .for_each(|p| p.velocity.x = 1.5);
        for _ in 0..60 {
            engine.run_frame(1.0 / 60.0);
        }

        let store = engine.world().store().expect("world store");
        let mesh_lane = store.read_lane::<MeshDesc>().expect("mesh lane");
        let mesh = mesh_lane.get(entity).expect("entity mesh");
        let (positions, indices) = mesh.as_custom().expect("still a Custom soup");
        assert_eq!(positions.len(), 16, "one vertex per particle");
        assert_eq!(indices.len(), 2 * 3 * 3 * 3, "two tris per cell");
        // Pinned corner never moved; a free particle swung sideways.
        assert_eq!(positions[0], origin.to_array());
        let swung = positions
            .iter()
            .any(|p| (p[0] - origin.to_array()[0]).abs() > 0.05);
        assert!(swung, "free cloth swung under its initial kick");
        let transform_lane = store.read_lane::<TransformDesc>().expect("transform lane");
        let transform = transform_lane.get(entity).expect("entity transform");
        assert_eq!(transform.translation, [0.0, 0.0, 0.0]);
        assert_eq!(transform.rotation, [0.0, 0.0, 0.0, 1.0]);
        // The pre-existing extraction path draws the soup unchanged.
        let extracted = extract_render_data(store);
        assert_eq!(extracted.custom_meshes.len(), 1);
        assert_eq!(extracted.custom_meshes[0].vertices.len(), 16);
        assert_eq!(
            extracted.custom_meshes[0].instance.model_matrix,
            glam::Mat4::IDENTITY
        );
    }
}
