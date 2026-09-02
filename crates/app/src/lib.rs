//! Unified gameplay runtime: single [`World`]/[`Engine`]/[`Schedule`] host.
//!
//! This crate bridges the backend-neutral gameplay systems in `ornis-core`
//! with domain specifics (physics [`RigidBody`] and render
//! [`TransformDesc`]/[`MeshDesc`]/[`MaterialDesc`]) so that the schedule
//! plans physics, render and gameplay as one DAG over a single world.
//! `RenderWorld` remains as an optional thin view — extraction is no longer
//! a mandatory copy boundary.

use std::sync::Mutex;

use glam::Vec3;
use ornis_core::{
    Engine, FixedTime, InputState, Resources, SmartStore, System, SystemAccess, World,
};
use ornis_physics::RigidBody;
use ornis_render::scene::{MaterialDesc, MeshDesc, TransformDesc};

pub use ornis_core::{GameplayPlugin, Position, Velocity, install_gameplay};

/// Thin view over the unified [`World`]: no second `Engine` copy required.
///
/// Wraps [`ornis_core::RenderWorldView`] and adds render-specific projection
/// that reads [`TransformDesc`]/[`MeshDesc`]/[`MaterialDesc`] lanes directly
/// from the authoritative world.
pub struct UnifiedView<'a> {
    world: &'a World,
}

impl<'a> UnifiedView<'a> {
    /// Creates a view over the unified world.
    pub fn new(world: &'a World) -> Self {
        Self { world }
    }

    /// Returns the underlying world.
    pub fn world(&self) -> &World {
        self.world
    }

    /// Number of renderable entities (entities with all three render lanes).
    pub fn renderable_count(&self) -> usize {
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

    /// Projects render snapshot without serialization.
    pub fn render_snapshot(&self) -> Vec<(TransformDesc, MeshDesc, MaterialDesc)> {
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
            let (Some(mesh), Some(material)) = (meshes.get(entity), materials.get(entity)) else {
                continue;
            };
            out.push((transform.clone(), mesh.clone(), material.clone()));
        }
        out
    }
}

/// Installs the unified runtime into `engine`.
///
/// Registers:
/// * core gameplay systems (`player_input`, `physics_push`, `transform_update`)
///   via [`install_gameplay`];
/// * physics bridge systems that propagate gameplay intent into [`RigidBody`]
///   and back;
/// * optional render extraction as a scheduled system (not a separate world copy).
///
/// The single [`Engine`] then drives the frame:
///
/// ```text
/// fixed:  gameplay physics_push + physics sync/step + bridge
/// frame:  player_input + transform_update + render extract
/// ```
pub fn install_unified_runtime(engine: &mut Engine) {
    // Core gameplay (player_input @ frame, physics_push @ fixed, transform_update @ frame)
    install_gameplay(engine);

    // Bridge gameplay velocity/position with physics bodies so that
    // the single schedule plans them together.
    engine.fixed_schedule_mut().add_system(VelocityToBodySystem);
    engine.schedule_mut().add_system(BodyToTransformSystem);

    // Render extraction as a schedule system on the same world — not a
    // second RenderWorld copy. The extracted snapshot lives as a resource.
    let _ = engine
        .world_mut()
        .insert(Mutex::new(UnifiedRenderExtracted::default()));
    engine.schedule_mut().add_system(UnifiedRenderExtractSystem);
}

/// CPU-side render data extracted from the unified world.
#[derive(Clone, Debug, Default)]
pub struct UnifiedRenderExtracted {
    /// Extracted materials in instance order.
    pub materials: Vec<ornis_core::OpenPBRMaterial>,
    /// Per-entity instance count.
    pub instance_count: usize,
    /// Whether extraction saw renderable entities.
    pub has_content: bool,
}

/// Writes [`Velocity`] (gameplay intent) into kinematic/dynamic [`RigidBody`]s.
///
/// Fixed-rate so that catch-up frames apply the same intent once per substep,
/// not once per variable frame.
struct VelocityToBodySystem;

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
struct BodyToTransformSystem;

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
                    desc.translation = position.to_array();
                    desc.rotation = [orientation.x, orientation.y, orientation.z, orientation.w];
                }
            }
        }
    }
}

struct UnifiedRenderExtractSystem;

impl System for UnifiedRenderExtractSystem {
    fn name(&self) -> &'static str {
        "unified_render_extract"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<TransformDesc>()
            .reads_lane::<MeshDesc>()
            .reads_lane::<MaterialDesc>()
            .writes::<Mutex<UnifiedRenderExtracted>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(out) = resources.get::<Mutex<UnifiedRenderExtracted>>() else {
            return;
        };
        let transforms = store.read_lane::<TransformDesc>();
        let meshes = store.read_lane::<MeshDesc>();
        let materials = store.read_lane::<MaterialDesc>();
        let (Some(t), Some(m), Some(ma)) = (transforms, meshes, materials) else {
            *out.lock().expect("unified extract lock") = UnifiedRenderExtracted::default();
            return;
        };
        let mut count = 0;
        for &entity in &t.entities {
            if m.get(entity).is_some() && ma.get(entity).is_some() {
                count += 1;
            }
        }
        // Material conversion borrowed from render extraction logic.
        let mut extracted_materials = Vec::with_capacity(count);
        for (&entity, _) in t.entities.iter().zip(&t.data) {
            let (Some(_mesh), Some(material)) = (m.get(entity), ma.get(entity)) else {
                continue;
            };
            let gpu = match material {
                MaterialDesc::Dielectric {
                    base_color,
                    roughness,
                } => {
                    let mut o = ornis_core::OpenPBRMaterial::dielectric();
                    o.base.color_rgb(*base_color);
                    o.specular.roughness(*roughness);
                    o
                }
                MaterialDesc::Metal {
                    base_color,
                    roughness,
                } => {
                    let mut o = ornis_core::OpenPBRMaterial::metal();
                    o.base.color_rgb(*base_color);
                    o.specular.roughness(*roughness);
                    o
                }
                MaterialDesc::Coat {
                    base_color,
                    coat_weight,
                    coat_roughness,
                } => {
                    let mut o = ornis_core::OpenPBRMaterial::coat();
                    o.base.color_rgb(*base_color);
                    o.coat.weight(*coat_weight);
                    o.coat.roughness(*coat_roughness);
                    o
                }
            };
            extracted_materials.push(gpu);
        }
        *out.lock().expect("unified extract lock") = UnifiedRenderExtracted {
            materials: extracted_materials,
            instance_count: count,
            has_content: count > 0,
        };
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
    use ornis_core::Engine;
    #[allow(unused_imports)]
    use ornis_core::Entity;
    use ornis_render::scene::{MaterialDesc, MeshDesc, TransformDesc};

    #[test]
    fn unified_view_counts_renderables_without_copy() {
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        let entity = engine.world().store().unwrap().create_entity();
        engine.world_mut().store_mut().unwrap().insert(
            entity,
            TransformDesc {
                translation: [0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
            },
        );
        engine.world_mut().store_mut().unwrap().insert(
            entity,
            MeshDesc::Sphere {
                radius: 1.0,
                segments: 16,
                rings: 8,
            },
        );
        engine.world_mut().store_mut().unwrap().insert(
            entity,
            MaterialDesc::Dielectric {
                base_color: [1.0, 0.0, 0.0],
                roughness: 0.5,
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
}
