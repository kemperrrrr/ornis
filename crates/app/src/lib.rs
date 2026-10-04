//! Unified gameplay runtime: single [`World`]/[`Engine`]/[`ornis_core::Schedule`] host.
//!
//! This crate bridges the backend-neutral gameplay systems in `ornis-core`
//! with domain specifics (physics [`RigidBody`] and render
//! [`TransformDesc`]/[`MeshDesc`]/[`MaterialDesc`]) so that the schedule
//! plans physics, render and gameplay as one DAG over a single world.
//! [`GameWorld`](game_world::GameWorld) is the single scene-backed world
//! type — extraction reads its lanes directly, never a second world copy.

use glam::Vec3;
use ornis_animation::{AnimClip, AnimPlayer, AnimSampleSystem, SkelSampleSystem, SkelSkinSystem};
use ornis_assets::scene::TransformDesc;
use ornis_core::{
    Engine, Entity, FixedTime, GlobalTransform, InputState, Resources, SmartStore, System,
    SystemAccess, Transform, World,
};
use ornis_physics::RigidBody;

pub use ornis_gameplay::{GameplayPlugin, Position, Velocity, install_gameplay};

/// Pure `ornis-gltf` mirror → `ornis-animation` converters.
pub mod anim_wiring;
pub mod game_world;
/// Sequential-impulse/XPBD physics runtime over the unified world.
pub mod physics_runtime;
/// Headless editor session: commands in, snapshots/events out.
pub mod session;
pub mod sync_harness;

pub use game_world::{EntityMut, GameWorld, ModelInstance, PlaybackError, ReplicaGameWorld, Spawn};
pub use ornis_animation::{AnimatorAccess, try_animator};

/// Inserts the flat authored pose as both local [`Transform`] and world
/// [`GlobalTransform`].
///
/// Today's scene and glTF spawns are still one level deep, so local and
/// world match. Hierarchy propagation overwrites [`GlobalTransform`] once
/// a [`ornis_core::ChildOf`] link exists.
pub(crate) fn insert_flat_pose(store: &mut SmartStore, entity: Entity, desc: &TransformDesc) {
    let local = Transform {
        translation: desc.translation,
        rotation: desc.rotation,
        scale: desc.scale,
    };
    store.insert(entity, local);
    store.insert(entity, GlobalTransform::from_local(local));
}

/// Installs the unified runtime into `engine`.
///
/// Registers:
/// * core gameplay systems (`player_input`, `physics_push`, `transform_update`)
///   via [`install_gameplay`];
/// * physics bridge systems that propagate gameplay intent into [`RigidBody`]
///   and back.
///
/// The single [`Engine`] then drives the frame:
///
/// ```text
/// fixed:  gameplay physics_push + physics sync/step + bridge
/// frame:  player_input + transform_update + body_to_transform
/// ```
/// Installs the cross-domain bridges `Velocity → RigidBody` (fixed) and
/// `RigidBody → Position/TransformDesc` (frame) so that browser `InputState`
/// intent reaches the authoritative physics solver in the same [`Engine`]
/// DAG. This is the cross-domain runtime seam: gameplay writes `Velocity`,
/// physics consumes it, and render extracts the final pose.
///
/// Idempotent with respect to [`install_unified_runtime`]: calling either
/// helper installs the same bridge systems exactly once per `Engine`.
pub fn install_gameplay_physics_bridge(engine: &mut Engine) {
    if engine
        .fixed_schedule()
        .mermaid()
        .contains("velocity_to_body")
    {
        return;
    }
    engine
        .fixed_schedule_mut()
        .prepend_system(VelocityToBodySystem);
    engine.schedule_mut().add_system(BodyToTransformSystem);
    // Intent must reach the solver in the same fixed update. Explicit edges
    // only split levels forward along registration order (S3 contract), so a
    // backward `order_before` would be silently ignored — registration must
    // put the writer first: `prepend` lands intent ahead of `sync_in`, and
    // the RaW on the `RigidBody` lane keeps it there across recomputes.
    // Best-effort: hosts without physics simply have no `physics_sync_in`.
    let _ = engine
        .fixed_schedule_mut()
        .try_order_before("velocity_to_body", "physics_sync_in");
}

/// Installs object animation (`anim_sample`, physics bodies skipped) into
/// the frame schedule after the body-pose propagation.
///
/// Idempotent like [`install_gameplay_physics_bridge`]: a second call on
/// the same engine is a no-op. Physics-authoritative entities keep the
/// generic sampler honest — `RigidBody` is visible here, unlike in
/// `ornis-anim` itself (which stays physics-free by design).
pub fn install_object_animation(engine: &mut Engine) {
    if engine.schedule().mermaid().contains("anim_sample") {
        return;
    }
    let Some(store) = engine.world_mut().store_mut() else {
        return;
    };
    store.register::<AnimPlayer>();
    store.register_cold::<AnimClip>();
    engine
        .schedule_mut()
        .add_system(AnimSampleSystem::<RigidBody>::new());
    // Poses propagate body → animation: the sampler overwrites what the
    // bridge wrote, never the reverse. Best-effort without the bridge.
    let _ = engine
        .schedule_mut()
        .try_order_before("body_to_transform", "anim_sample");
}

/// Installs skeletal animation (`skel_sample`, `skel_skin_cpu`) into
/// the frame schedule after the object sampler.
///
/// Idempotent like [`install_object_animation`]: a second call on the
/// same engine is a no-op. Ordering follows the animation crate contract:
/// `anim_sample → skel_sample → skel_skin_cpu` (disjoint lanes, explicit
/// edges for determinism). No-op until an entity carries skeletal lanes.
pub fn install_skeletal_animation(engine: &mut Engine) {
    if engine.schedule().mermaid().contains("skel_sample") {
        return;
    }
    engine.schedule_mut().add_system(SkelSampleSystem::new());
    engine.schedule_mut().add_system(SkelSkinSystem::new());
    let _ = engine
        .schedule_mut()
        .try_order_before("anim_sample", "skel_sample");
    let _ = engine
        .schedule_mut()
        .try_order_before("skel_sample", "skel_skin_cpu");
}

pub fn install_unified_runtime(engine: &mut Engine) {
    // Core gameplay (player_input @ frame, physics_push @ fixed, transform_update @ frame)
    install_gameplay(engine);

    // Bridge gameplay velocity/position with physics bodies so that
    // the single schedule plans them together.
    install_gameplay_physics_bridge(engine);
}

/// Writes [`Velocity`] (gameplay intent) into kinematic/dynamic [`RigidBody`]s.
///
/// Fixed-rate so that catch-up frames apply the same intent once per substep,
/// not once per variable frame.
///
/// The typed hot path: [`sync_harness`] re-expresses this mapping
/// declaratively for Nth engines, this system stays the optimized SI route.
pub(crate) struct VelocityToBodySystem;

impl System for VelocityToBodySystem {
    fn name(&self) -> &'static str {
        "velocity_to_body"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<FixedTime>()
            .reads::<SmartStore>()
            .reads_lane::<Velocity>()
            .writes_lane::<RigidBody>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(vel_lane) = store.read_lane::<Velocity>() else {
            return;
        };
        let snapshot: Vec<(ornis_core::Entity, Vec3)> = vel_lane
            .entities
            .iter()
            .zip(&vel_lane.data)
            .map(|(&e, v)| (e, v.0))
            .collect();
        drop(vel_lane);
        let Some(mut body_lane) = store.write_lane::<RigidBody>() else {
            return;
        };
        for (entity, vel) in snapshot {
            if let Some(body) = body_lane.get_mut(entity) {
                // Only kinematic/dynamic bodies follow gameplay intent; static
                // bodies remain editor-controlled.
                if body.body_type != ornis_physics::BodyType::Static {
                    body.velocity.x = vel.x;
                    body.velocity.z = vel.z;
                    // Preserve vertical velocity for gravity/jump.
                }
            }
        }
    }
}

/// Propagates physics body positions back into gameplay [`Position`] and
/// render [`TransformDesc`] lanes.
///
/// The typed hot path: [`sync_harness`] re-expresses this mapping
/// declaratively for Nth engines, this system stays the optimized SI route.
pub(crate) struct BodyToTransformSystem;

impl System for BodyToTransformSystem {
    fn name(&self) -> &'static str {
        "body_to_transform"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<RigidBody>()
            .writes_lane::<Position>()
            .writes_lane::<TransformDesc>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(body_lane) = store.read_lane::<RigidBody>() else {
            return;
        };
        let snapshot: Vec<(ornis_core::Entity, Vec3, glam::Quat)> = body_lane
            .entities
            .iter()
            .zip(&body_lane.data)
            .map(|(&e, b)| (e, b.position, b.orientation))
            .collect();
        drop(body_lane);
        if let Some(mut pos_lane) = store.write_lane::<Position>() {
            for (entity, position, _) in &snapshot {
                if let Some(pos) = pos_lane.get_mut(*entity) {
                    pos.0 = *position;
                } else {
                    pos_lane.insert(*entity, Position(*position));
                }
            }
        }
        if let Some(mut desc_lane) = store.write_lane::<TransformDesc>() {
            for (entity, position, orientation) in snapshot {
                if let Some(desc) = desc_lane.get_mut(entity) {
                    desc.translation = position;
                    if let Some(rotation) = ornis_core::units::UnitQuat::normalize(orientation) {
                        desc.rotation = rotation;
                    }
                }
            }
        }
    }
}

/// Applies browser [`InputState`] received over WS without polling.
///
/// The editor backend forwards decoded `InputState` snapshots as a resource
/// update; this helper is the server-side counterpart used by both the
/// native and editor-only runtimes.
pub fn apply_browser_input(world: &mut World, input: InputState) {
    if let Some(slot) = world.resources_mut().get_mut::<InputState>() {
        *slot = input;
    } else {
        let _ = world.insert(input);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
    #[allow(unused_imports)]
    use ornis_core::Entity;
    use ornis_core::units::{Clamped01, PositiveF32};
    use ornis_core::{Engine, World};

    /// Thin view over the unified [`World`]: no second `Engine` copy required.
    struct UnifiedView<'a> {
        world: &'a World,
    }

    impl<'a> UnifiedView<'a> {
        fn new(world: &'a World) -> Self {
            Self { world }
        }

        fn renderable_count(&self) -> usize {
            let Some(store) = self.world.store() else {
                return 0;
            };
            let Some(transforms) = store.read_lane::<TransformDesc>() else {
                return 0;
            };
            let Some(meshes) = store.read_lane::<MeshDesc>() else {
                return 0;
            };
            let Some(materials) = store.read_lane::<MaterialDesc>() else {
                return 0;
            };
            let mut count = 0;
            for &entity in &transforms.entities {
                if meshes.get(entity).is_some() && materials.get(entity).is_some() {
                    count += 1;
                }
            }
            count
        }

        fn render_snapshot(&self) -> Vec<(TransformDesc, MeshDesc, MaterialDesc)> {
            let Some(store) = self.world.store() else {
                return Vec::new();
            };
            let Some(transforms) = store.read_lane::<TransformDesc>() else {
                return Vec::new();
            };
            let Some(meshes) = store.read_lane::<MeshDesc>() else {
                return Vec::new();
            };
            let Some(materials) = store.read_lane::<MaterialDesc>() else {
                return Vec::new();
            };
            let mut out = Vec::new();
            for (&entity, transform) in transforms.entities.iter().zip(&transforms.data) {
                let (Some(mesh), Some(material)) = (meshes.get(entity), materials.get(entity))
                else {
                    continue;
                };
                out.push((transform.clone(), mesh.clone(), material.clone()));
            }
            out
        }
    }

    #[test]
    fn unified_view_counts_renderables_without_copy() {
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        let entity = engine.world().store().unwrap().create_entity();
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, TransformDesc::IDENTITY);
        engine.world_mut().store_mut().unwrap().insert(
            entity,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            },
        );
        engine.world_mut().store_mut().unwrap().insert(
            entity,
            MaterialDesc::Dielectric {
                base_color: [1.0, 0.0, 0.0],
                roughness: Clamped01::new(0.5),
                emission: [0.0, 0.0, 0.0],
            },
        );
        let view = UnifiedView::new(engine.world());
        assert_eq!(view.renderable_count(), 1);
        assert_eq!(view.render_snapshot().len(), 1);
    }

    #[test]
    fn unified_runtime_schedules_gameplay_physics_render() {
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        // Core gameplay + bridge + unified extract
        assert!(engine.schedule().len() >= 3);
        assert!(!engine.fixed_schedule().is_empty());
        let mermaid = engine.schedule().mermaid();
        assert!(mermaid.contains("player_input"));
        assert!(mermaid.contains("body_to_transform") || mermaid.contains("transform_update"));
    }

    #[test]
    fn velocity_to_body_propagates_to_rigid_body() {
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        let e = engine.world().store().unwrap().create_entity();
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(e, Velocity(Vec3::new(3.0, 0.0, 4.0)));
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(e, RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        engine.run_frame(1.0 / 60.0);
        let store = engine.world().store().unwrap();
        let lane = store.read_lane::<RigidBody>().unwrap();
        let body = lane.get(e).unwrap();
        assert!((body.velocity.x - 3.0).abs() < 1e-4);
        assert!((body.velocity.z - 4.0).abs() < 1e-4);
    }

    #[test]
    fn apply_browser_input_replaces_resource() {
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        let mut input = InputState::new();
        input.set_key(87, true);
        input.set_pointer_position([100.0, 200.0]);
        apply_browser_input(engine.world_mut(), input.clone());
        let stored = engine.world().resources().get::<InputState>().unwrap();
        assert!(stored.key_down(87));
        assert_eq!(stored.pointer_position(), [100.0, 200.0]);
    }

    #[test]
    fn browser_wasd_input_drives_player_through_gameplay_to_physics() {
        use ornis_core::InputState;
        use ornis_gameplay::Position;
        let mut engine = Engine::new();
        ornis_gameplay::install_gameplay(&mut engine);
        install_gameplay_physics_bridge(&mut engine);
        struct MiniPhysics;
        impl ornis_core::System for MiniPhysics {
            fn name(&self) -> &'static str {
                "mini_physics_integrate"
            }
            fn access(&self) -> ornis_core::SystemAccess {
                ornis_core::SystemAccess::new()
                    .reads::<ornis_core::FixedTime>()
                    .reads::<ornis_core::SmartStore>()
                    .reads_lane::<RigidBody>()
                    .writes_lane::<RigidBody>()
            }
            fn run(&self, resources: &ornis_core::Resources) {
                let dt = resources
                    .get::<ornis_core::FixedTime>()
                    .unwrap()
                    .delta_seconds();
                let store = resources.get::<ornis_core::SmartStore>().unwrap();
                let snap: Vec<(ornis_core::Entity, Vec3)> = store
                    .read_lane::<RigidBody>()
                    .map(|lane| {
                        lane.entities
                            .iter()
                            .zip(&lane.data)
                            .map(|(&e, b)| (e, b.velocity))
                            .collect()
                    })
                    .unwrap_or_default();
                if let Some(mut lane) = store.write_lane::<RigidBody>() {
                    for (e, vel) in snap {
                        if let Some(b) = lane.get_mut(e) {
                            b.position += vel * dt;
                        }
                    }
                }
            }
        }
        engine.fixed_schedule_mut().add_system(MiniPhysics);
        let e = engine.world().store().unwrap().create_entity();
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(e, ornis_gameplay::Player);
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(e, Position(Vec3::ZERO));
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(e, TransformDesc::IDENTITY);
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(e, RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
        let mut input = InputState::new();
        input.set_key(87, true);
        apply_browser_input(engine.world_mut(), input);
        engine.run_frame(1.0 / 60.0);
        engine.run_frame(1.0 / 60.0);
        let store = engine.world().store().unwrap();
        let body = store
            .read_lane::<RigidBody>()
            .unwrap()
            .get(e)
            .unwrap()
            .clone();
        let pos = store.read_lane::<Position>().unwrap().get(e).unwrap().0;
        let t = store
            .read_lane::<TransformDesc>()
            .unwrap()
            .get(e)
            .unwrap()
            .clone();
        assert!(
            body.velocity.z < -1.0,
            "velocity not propagated: {:?}",
            body.velocity
        );
        assert!(pos.z < -0.01, "Position not moved: {:?}", pos);
        assert!(
            t.translation[2] < -0.01,
            "TransformDesc not moved: {:?}",
            t.translation
        );
        assert!(body.position.z < -0.01);
    }
}
