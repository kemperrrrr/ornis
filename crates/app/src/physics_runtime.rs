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
//! Soft bodies (PLAN B2/D1.5) ride the same seam through the [`Engine`]
//! orchestrator on its XPBD path: entities carrying a [`SoftBody`] lane
//! component promote the world onto [`SolverKind::Xpbd`], are bound to
//! solver handles, stepped with the fixed clock together with the rigid
//! bodies (so soft↔rigid coupling runs in the one substep loop), and
//! written back as world-space [`MeshDesc::Custom`] soups (identity
//! transform) that the existing extraction path draws unchanged.

use std::cmp::Reverse;
use std::collections::HashMap;
use std::sync::Mutex;

use glam::{Quat, Vec3};
#[cfg(test)]
use ornis_assets::scene::MaterialDesc;
use ornis_assets::scene::{MeshDesc, TransformDesc};
use ornis_core::{
    ChildOf, ComponentStore, Engine, Entity, FixedTime, GlobalTransform, Resources, SmartStore,
    System, SystemAccess, Transform, UnitQuat,
};
use ornis_editor::EditorOnly;
use ornis_physics::soft_render::{MIN_TUBE_SIDES, tube_indices, tube_positions};
use ornis_physics::{
    BodyHandle, BodyType, PhysicsEngine, RigidBody, SoftBody, SoftHandle, SolverKind,
};
#[cfg(test)]
use ornis_render::extract_render_data;

/// Spatial components in a world-space position.
const VEC3_COMPONENTS: usize = 3;

/// Render parameters for a chain/rope soft body (PLAN B2/D1 leftover #3).
///
/// Entities with a soft-solver binding, an empty [`SoftBody::surface`] and
/// this lane component upload a tube soup along the solver particles every
/// frame (see [`PhysicsRuntime::sync_soft_out`]). `radius` is the tube radius
/// in world units, `sides` the cross-section resolution (minimum 3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeMesh {
    /// Tube radius in world units (must be finite and positive).
    pub radius: f32,
    /// Cross-section sides (minimum 3).
    pub sides: u32,
}

/// Physics domain state registered in a core [`Engine`] as a resource.
///
/// The sequential-impulse solver owns its optimized body array and the map keeps the
/// association with generational ECS entities. ECS `RigidBody` components are
/// synchronized at the system boundary rather than exposing physics' internal
/// vector to other domains. The common core engine host accumulates
/// render-frame time and invokes this domain at a bounded fixed 60 Hz
/// timestep.
///
/// Soft bodies live in the same orchestrator ([`SolverKind::Xpbd`], PLAN
/// B2/D1): the first [`SoftBody`] lane component promotes the world onto
/// the XPBD path, so rigid + soft share one substep loop and soft↔rigid
/// coupling actually runs. Mesh upload reads the orchestrator directly
/// (world-space soup, identity transform).
pub struct PhysicsRuntime {
    solver: ornis_physics::Engine,
    bindings: HashMap<Entity, BodyHandle>,
    soft_bindings: HashMap<Entity, SoftHandle>,
    gravity: Vec3,
    changed: bool,
}

impl PhysicsRuntime {
    /// Creates a physics runtime with world-space gravity.
    pub fn new(gravity: Vec3) -> Self {
        Self {
            solver: ornis_physics::Engine::new(SolverKind::SequentialImpulse, gravity),
            bindings: HashMap::new(),
            soft_bindings: HashMap::new(),
            gravity,
            changed: false,
        }
    }

    /// Syncs ECS rigid bodies into the solver, skipping editor chrome.
    ///
    /// Entities carrying [`EditorOnly`] never reach the solver (a gizmo must
    /// not fall under gravity) and lose a stale binding if marked later —
    /// the single [`ornis_editor::is_editor_only`] rule, threaded through
    /// as a lane instead of a second query so the hot loop stays one pass.
    fn sync_in(
        &mut self,
        bodies: &ComponentStore<RigidBody>,
        globals: Option<&ComponentStore<GlobalTransform>>,
        transforms: Option<&ComponentStore<TransformDesc>>,
        excluded: Option<&ComponentStore<EditorOnly>>,
    ) {
        self.remove_stale_bindings(bodies, excluded);

        for (&entity, source) in bodies.entities.iter().zip(&bodies.data) {
            if excluded.is_some_and(|lane| lane.contains(entity)) {
                continue;
            }
            if let Some(&handle) = self.bindings.get(&entity) {
                self.sync_external_pose(handle, source, world_pose(globals, transforms, entity));
                continue;
            }

            let mut body = source.clone();
            if let Some((position, orientation)) = world_pose(globals, transforms, entity) {
                apply_world_pose(&mut body, position, orientation);
            }
            let handle = self.solver.add_body(body);
            self.bindings.insert(entity, handle);
        }
    }

    /// Drops bindings whose lane component is gone — or whose entity
    /// became editor chrome ([`EditorOnly`]): marking an entity later
    /// evicts its body on the next sync instead of simulating it once more.
    fn remove_stale_bindings(
        &mut self,
        bodies: &ComponentStore<RigidBody>,
        excluded: Option<&ComponentStore<EditorOnly>>,
    ) {
        let mut stale: Vec<(Entity, BodyHandle)> = self
            .bindings
            .iter()
            .filter(|(entity, _)| {
                !bodies.contains(**entity) || excluded.is_some_and(|lane| lane.contains(**entity))
            })
            .map(|(&entity, &handle)| (entity, handle))
            .collect();
        stale.sort_unstable_by_key(|&(_, handle)| Reverse(handle));

        for (entity, handle) in stale {
            let last = self.bindings.len().saturating_sub(1);
            let moved = if handle.index() < last {
                self.bindings
                    .iter()
                    .find_map(|(&candidate, &bound)| (bound.index() == last).then_some(candidate))
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
        pose: Option<(Vec3, Quat)>,
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
            && let Some((position, orientation)) = pose
        {
            apply_world_pose(body, position, orientation);
        }
        // A newly edited body role/filter is reflected at the next sync
        // when the ECS source differs from the solver representation.
        if body.body_type != source.body_type
            || body.collision_layer != source.collision_layer
            || body.collision_mask != source.collision_mask
            || body.is_trigger != source.is_trigger
        {
            *body = source.clone();
            if let Some((position, orientation)) = pose {
                apply_world_pose(body, position, orientation);
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

    fn sync_out(&mut self, store: &SmartStore) {
        let poses: Vec<(Entity, RigidBody)> = self
            .bindings
            .iter()
            .filter_map(|(&entity, &handle)| {
                self.solver
                    .get_body(handle)
                    .map(|body| (entity, body.clone()))
            })
            .collect();
        let parents: Vec<(Entity, Entity)> = store
            .read_lane::<ChildOf>()
            .map(|lane| {
                lane.entities
                    .iter()
                    .zip(lane.data.iter())
                    .map(|(&child, link)| (child, link.parent()))
                    .collect()
            })
            .unwrap_or_default();
        let globals: Vec<(Entity, GlobalTransform)> = store
            .read_lane::<GlobalTransform>()
            .map(|lane| {
                lane.entities
                    .iter()
                    .zip(lane.data.iter())
                    .map(|(&entity, pose)| (entity, *pose))
                    .collect()
            })
            .unwrap_or_default();
        let locals: Vec<(Entity, Transform)> = store
            .read_lane::<Transform>()
            .map(|lane| {
                lane.entities
                    .iter()
                    .zip(lane.data.iter())
                    .map(|(&entity, pose)| (entity, *pose))
                    .collect()
            })
            .unwrap_or_default();
        if let Some(mut bodies) = store.write_lane::<RigidBody>() {
            for (entity, body) in &poses {
                if let Some(destination) = bodies.get_mut(*entity) {
                    *destination = body.clone();
                }
            }
        }
        if let Some(mut transforms) = store.write_lane::<TransformDesc>() {
            for (entity, body) in &poses {
                if let Some(destination) = transforms.get_mut(*entity) {
                    destination.translation = body.position;
                    if let Some(rotation) = UnitQuat::normalize(body.orientation) {
                        destination.rotation = rotation;
                    }
                }
            }
        }
        if let Some(mut lane) = store.write_lane::<Transform>() {
            for (entity, body) in &poses {
                let Some(previous) = locals
                    .iter()
                    .find_map(|(candidate, pose)| (*candidate == *entity).then_some(*pose))
                else {
                    continue;
                };
                let world_rotation =
                    UnitQuat::normalize(body.orientation).unwrap_or(previous.rotation);
                let parent_global = parents
                    .iter()
                    .find_map(|(child, parent)| (*child == *entity).then_some(*parent))
                    .and_then(|parent| {
                        globals
                            .iter()
                            .find_map(|(candidate, pose)| (*candidate == parent).then_some(*pose))
                    });
                let local = match parent_global {
                    Some(parent) => parent.to_local(body.position, world_rotation, previous),
                    None => Transform {
                        translation: body.position,
                        rotation: world_rotation,
                        scale: previous.scale,
                    },
                };
                if let Some(destination) = lane.get_mut(*entity) {
                    *destination = local;
                }
            }
        }
    }

    pub(crate) fn take_changed(&mut self) -> bool {
        std::mem::take(&mut self.changed)
    }

    /// Binds newly added [`SoftBody`] lane components into the solver.
    /// The first soft body promotes the world onto [`SolverKind::Xpbd`]
    /// (one-way: the promotion rebuilds the rigid scene 1:1, handles stay
    /// valid); solver state is authoritative after registration (no
    /// per-step pose sync in D1 — there is no gameplay intent for
    /// particles yet).
    fn sync_soft_in(&mut self, soft: &ComponentStore<SoftBody>) {
        self.remove_stale_soft_bindings(soft);
        if soft.entities.is_empty() {
            return;
        }
        if self.solver.kind() != SolverKind::Xpbd {
            self.solver.set_solver_kind(SolverKind::Xpbd, self.gravity);
        }
        for (&entity, source) in soft.entities.iter().zip(&soft.data) {
            if self.soft_bindings.contains_key(&entity) {
                continue;
            }
            let handle = self.solver.add_soft_body(source.clone());
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
            let moved = if handle.index() < last {
                self.soft_bindings
                    .iter()
                    .find_map(|(&candidate, &bound)| (bound.index() == last).then_some(candidate))
            } else {
                None
            };
            self.solver.remove_soft_body(handle);
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
    /// Bodies with a render `surface` (cloth) upload it directly; bodies
    /// with an empty `surface` but a [`RopeMesh`] lane upload a tube soup
    /// along the chain (positions every frame, indices rebuilt). Chains
    /// without [`RopeMesh`] are still skipped.
    fn sync_soft_out(
        &mut self,
        meshes: &mut ComponentStore<MeshDesc>,
        transforms: &mut ComponentStore<TransformDesc>,
        ropes: Option<&ComponentStore<RopeMesh>>,
    ) -> Vec<Entity> {
        let mut pinned = Vec::new();
        for (&entity, &handle) in &self.soft_bindings {
            let Some(body) = self.solver.get_soft_body(handle) else {
                continue;
            };
            if body.surface.is_empty() {
                let Some(ropes) = ropes else { continue };
                let Some(rope) = ropes.get(entity) else {
                    continue;
                };
                if !rope.radius.is_finite() || rope.radius <= 0.0 || rope.sides < MIN_TUBE_SIDES {
                    continue;
                }
                let count = body.particles.len();
                if count < 2 {
                    continue;
                }
                let positions = tube_positions(&body.particles, rope.radius, rope.sides);
                if positions.is_empty() {
                    continue;
                }
                let indices = tube_indices(count, rope.sides);
                let desc = MeshDesc::Custom { positions, indices };
                if let Some(slot) = meshes.get_mut(entity) {
                    *slot = desc;
                } else {
                    meshes.insert(entity, desc);
                }
                if let Some(transform) = transforms.get_mut(entity) {
                    transform.translation = glam::Vec3::ZERO;
                    transform.rotation = ornis_core::units::UnitQuat::IDENTITY;
                    transform.scale = glam::Vec3::ONE;
                }
                pinned.push(entity);
                continue;
            }
            let positions: Vec<[f32; VEC3_COMPONENTS]> = body
                .positions_snapshot()
                .iter()
                .map(Vec3::to_array)
                .collect();
            let indices: Vec<u32> = body
                .surface
                .iter()
                .flat_map(|tri| [tri[0].as_u32(), tri[1].as_u32(), tri[2].as_u32()])
                .collect();
            let desc = MeshDesc::Custom { positions, indices };
            if let Some(slot) = meshes.get_mut(entity) {
                *slot = desc;
            } else {
                meshes.insert(entity, desc);
            }
            if let Some(transform) = transforms.get_mut(entity) {
                transform.translation = glam::Vec3::ZERO;
                transform.rotation = ornis_core::units::UnitQuat::IDENTITY;
                transform.scale = glam::Vec3::ONE;
            }
            pinned.push(entity);
        }
        pinned
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
    // Last prepend runs first: world poses exist before sync-in reads them.
    ornis_core::install_fixed_transform_propagation(engine);
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
            .reads_lane::<GlobalTransform>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<EditorOnly>()
            .writes::<Mutex<PhysicsRuntime>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(body_lane) = store.read_lane::<RigidBody>() else {
            return;
        };
        let globals = store.read_lane::<GlobalTransform>();
        let transforms = store.read_lane::<TransformDesc>();
        let excluded = store.read_lane::<EditorOnly>();
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let mut runtime = runtime_resource.lock().unwrap_or_else(|e| e.into_inner());
        runtime.sync_in(
            &body_lane,
            globals.as_deref(),
            transforms.as_deref(),
            excluded.as_deref(),
        );
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
            .unwrap_or_else(|e| e.into_inner())
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
            .reads_lane::<ChildOf>()
            .reads_lane::<GlobalTransform>()
            .reads_lane::<Transform>()
            .writes_lane::<Transform>()
            .writes_lane::<TransformDesc>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        runtime_resource
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .sync_out(store);
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
        let mut runtime = runtime_resource.lock().unwrap_or_else(|e| e.into_inner());
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
            .reads_lane::<RopeMesh>()
            .writes_lane::<MeshDesc>()
            .writes_lane::<TransformDesc>()
            .writes_lane::<Transform>()
            .writes_lane::<GlobalTransform>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let pinned = {
            let Some(mut mesh_lane) = store.write_lane::<MeshDesc>() else {
                return;
            };
            let Some(mut transform_lane) = store.write_lane::<TransformDesc>() else {
                return;
            };
            let ropes = store.read_lane::<RopeMesh>();
            runtime_resource
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .sync_soft_out(&mut mesh_lane, &mut transform_lane, ropes.as_deref())
        };
        for entity in pinned {
            pin_hierarchy_identity(store, entity);
        }
    }
}

/// World pose for physics sync-in.
///
/// [`GlobalTransform`] wins. Entities that have not been propagated yet
/// fall back to [`TransformDesc`], which flat scenes store in world space.
fn world_pose(
    globals: Option<&ComponentStore<GlobalTransform>>,
    transforms: Option<&ComponentStore<TransformDesc>>,
    entity: Entity,
) -> Option<(Vec3, Quat)> {
    if let Some(global) = globals.and_then(|lane| lane.get(entity)) {
        return Some((global.translation, global.rotation.get()));
    }
    transforms
        .and_then(|lane| lane.get(entity))
        .map(|transform| (transform.translation, transform.rotation.get()))
}

/// Applies a world pose to a physics body's position and orientation.
fn apply_world_pose(body: &mut RigidBody, position: Vec3, orientation: Quat) {
    body.position = position;
    body.orientation = orientation;
}

/// Pins local and world TRS to identity so a world-space soft mesh is not
/// transformed a second time. [`TransformDesc`] is pinned by the caller.
fn pin_hierarchy_identity(store: &SmartStore, entity: Entity) {
    if let Some(mut lane) = store.write_lane::<Transform>()
        && let Some(transform) = lane.get_mut(entity)
    {
        *transform = Transform::IDENTITY;
    }
    if let Some(mut lane) = store.write_lane::<GlobalTransform>()
        && let Some(transform) = lane.get_mut(entity)
    {
        *transform = GlobalTransform::IDENTITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::units::{Clamped01, PositiveF32};

    fn dynamic_body(position: Vec3) -> RigidBody {
        RigidBody::new_sphere(position, 0.5, 1.0)
    }

    fn transform(position: Vec3) -> TransformDesc {
        TransformDesc::from_translation(position)
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
            lane.get(entity)
                .expect("entity transform")
                .translation
                .to_array(),
            [2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn physics_sync_reads_world_pose_and_writes_local() {
        use ornis_core::{ChildOf, GlobalTransform, Transform};
        let mut engine = Engine::new();
        let store = engine.world_mut().store_mut().expect("store");
        let parent = store.create_entity();
        let child = store.create_entity();
        store.insert(
            parent,
            Transform::from_translation(Vec3::new(8.0, 0.0, 0.0)),
        );
        store.insert(child, Transform::from_translation(Vec3::ZERO));
        store.insert(child, TransformDesc::from_translation(Vec3::ZERO));
        store.insert(child, ChildOf(parent));
        store.insert(child, RigidBody::new_sphere(Vec3::ZERO, 0.5, 0.0));
        install_physics(&mut engine, Vec3::new(0.0, -9.81, 0.0));

        engine.run_frame(1.0 / 60.0);

        let store = engine.world().store().expect("store");
        let world = store
            .read_lane::<TransformDesc>()
            .expect("desc")
            .get(child)
            .expect("child desc")
            .translation;
        let local = store
            .read_lane::<Transform>()
            .expect("local")
            .get(child)
            .expect("child local")
            .translation;
        assert!(
            (world.x - 8.0).abs() < 1e-3,
            "sync-in must read the propagated world pose, got {world:?}"
        );
        assert!(
            local.x.abs() < 1e-3,
            "sync-out must keep the parent-relative local pose, got {local:?}"
        );
        let global = store
            .read_lane::<GlobalTransform>()
            .expect("global")
            .get(child)
            .expect("child global")
            .translation;
        assert!((global.x - 8.0).abs() < 1e-3);
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
        assert_eq!(
            runtime.bindings.get(&entities[0]),
            Some(&BodyHandle::from_raw(0))
        );
        assert_eq!(
            runtime.bindings.get(&entities[2]),
            Some(&BodyHandle::from_raw(2))
        );
        assert_eq!(
            runtime.bindings.get(&entities[3]),
            Some(&BodyHandle::from_raw(1))
        );
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
                radius: PositiveF32::expect_valid(2.0),
                segments: 48,
                rings: 32,
            },
        );
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MaterialDesc::Metal {
                base_color: [0.9, 0.8, 0.2],
                roughness: Clamped01::new(0.2),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(1.0),
            },
        );

        engine.run_frame(0.0);

        // X4: direct lane read — no scheduled snapshot anymore.
        let extracted = extract_render_data(engine.world().store().expect("world store"));
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(extracted.materials.len(), 1);
        assert_eq!(extracted.instances.len(), 1);
        assert_eq!(
            extracted.instances[0].material_index,
            ornis_render::MaterialIdx::from_raw(0)
        );
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
                radius: PositiveF32::expect_valid(0.5),
                segments: 16,
                rings: 8,
            },
        );
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            MaterialDesc::Dielectric {
                base_color: [0.5, 0.5, 0.5],
                roughness: Clamped01::new(0.5),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(0.0),
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
            PhysicsSyncIn
                .access()
                .reads_lanes
                .contains(&std::any::TypeId::of::<EditorOnly>()),
            "sync-in must declare the EditorOnly exclusion lane"
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
        assert!(
            SoftSyncOut
                .access()
                .reads_lanes
                .contains(&std::any::TypeId::of::<RopeMesh>())
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
                roughness: Clamped01::new(0.5),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(0.0),
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
        assert_eq!(transform.translation.to_array(), [0.0, 0.0, 0.0]);
        assert_eq!(transform.rotation_array(), [0.0, 0.0, 0.0, 1.0]);
        // The pre-existing extraction path draws the soup unchanged.
        let extracted = extract_render_data(store);
        assert_eq!(extracted.custom_meshes.len(), 1);
        assert_eq!(extracted.custom_meshes[0].vertices.len(), 16);
        assert_eq!(
            extracted.custom_meshes[0].instance.model_matrix,
            glam::Mat4::IDENTITY
        );
    }

    /// PLAN B2/D1 leftover #3: a chain entity (empty `surface`) with a
    /// [`RopeMesh`] lane uploads a tube soup tracking the solver particles
    /// every frame, its transform pinned to identity (world-space soup), and
    /// the existing extraction path draws it.
    #[test]
    fn rope_mesh_uploads_tube_soup_tracking_solver() {
        use ornis_physics::SoftBody;

        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("world store")
            .create_entity();
        let origin = Vec3::new(0.0, 2.0, 0.0);
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            SoftBody::chain(origin, Vec3::NEG_Y, 6, 0.25, 1.0, 0.0),
        );
        engine.world_mut().store_mut().expect("world store").insert(
            entity,
            RopeMesh {
                radius: 0.05,
                sides: 6,
            },
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
                roughness: Clamped01::new(0.5),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(0.0),
            },
        );
        install_physics(&mut engine, Vec3::new(0.0, -9.81, 0.0));

        // Same kick as the cloth test: a hanging chain at rest is an exact
        // equilibrium, so motion is proven by the resulting pendulum swing.
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
        assert_eq!(positions.len(), 6 * 6, "one ring of 6 per particle");
        assert_eq!(indices.len(), (6 - 1) * 6 * 6, "two tris per side quad");
        assert!(positions.len() > 6, "tube soup outnumbers the particles");
        // Pinned particle never moved: its whole ring hugs the origin.
        for v in &positions[0..6] {
            let dist = Vec3::from_array(*v).distance(origin);
            assert!((dist - 0.05).abs() < 1e-3, "pinned ring at radius: {v:?}");
        }
        // A free particle swung sideways under its initial kick.
        let swung = positions
            .iter()
            .any(|p| (p[0] - origin.to_array()[0]).abs() > 0.05);
        assert!(swung, "free chain swung under its initial kick");
        let transform_lane = store.read_lane::<TransformDesc>().expect("transform lane");
        let transform = transform_lane.get(entity).expect("entity transform");
        assert_eq!(transform.translation.to_array(), [0.0, 0.0, 0.0]);
        assert_eq!(transform.rotation_array(), [0.0, 0.0, 0.0, 1.0]);
        // The pre-existing extraction path draws the soup unchanged.
        let extracted = extract_render_data(store);
        assert_eq!(extracted.custom_meshes.len(), 1);
        assert_eq!(extracted.custom_meshes[0].vertices.len(), 6 * 6);
        assert_eq!(
            extracted.custom_meshes[0].instance.model_matrix,
            glam::Mat4::IDENTITY
        );
    }
}
