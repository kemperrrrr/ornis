//! Single game world: one type, two roles.
//!
//! [`GameWorld`] is the only scene-backed world type: it owns one [`Engine`] plus the scene entity list and a monotonic mutation counter, and serves both the authoritative native/editor host and the browser replica — the two instances differ only in their [`SceneRole`] phantom parameter ([`Authoritative`](ornis_core::Authoritative) vs [`Replica`](ornis_core::Replica)) and in how scenes cross the serialization boundary (IDEAS §28), never in the world layout itself.
//!
//! System registration uses the staged [`Engine`](ornis_core::Engine) plan
//! directly ([`Stage`](ornis_core::Stage) via
//! [`Engine::add_stage_system`](ornis_core::Engine::add_stage_system) /
//! [`Engine::stage_schedule_mut`](ornis_core::Engine::stage_schedule_mut)):
//! `Input` systems consume the per-frame input, `Gameplay` systems run
//! intent and physics at the fixed step, `PostFrame` runs the variable
//! script tick, propagates poses and extracts audio/render views.

use std::marker::PhantomData;

use ornis_assets::scene::{EntityDesc, Scene};
use ornis_core::{
    Authoritative, Color, Engine, Entity, LinearRgb, Replica, SceneEntities, SceneRole,
    SceneVersion, Seconds,
};
use ornis_render::extraction::{RenderLights, extract_render_data};
use ornis_render::{DirectionalLight, FrameUpload, OrbitCamera, install_orbit_camera};

/// Single scene-backed game world: one [`Engine`] plus its scene entities.
///
/// The native showcase and the editor server run the authoritative instance
/// ([`Authoritative`](ornis_core::Authoritative), the default: a bare
/// `GameWorld` keeps meaning the authoritative world); the browser viewport
/// runs a [`ReplicaGameWorld`] populated from serialized snapshots across
/// the boundary (IDEAS §28) — construct it with
/// [`ReplicaGameWorld::new_replica`]/[`ReplicaGameWorld::from_scene_replica`],
/// never by reusing the authoritative constructors. Physics, audio,
/// scripting and GPU state stay specialized resources installed by the
/// platform through [`GameWorld::engine_mut`], never duplicated here.
/// That `engine_mut` (a `&mut Engine<Running>` late-registration hatch, in
/// the `install_*` family style) is the deliberate legacy seam: the 50+
/// platform installers are not migrated to the builder, so scene-backed
/// setup keeps registering between frames here.
///
/// Scene membership lives in a [`SceneEntities`](ornis_core::SceneEntities)
/// newtype and the mutation counter in a
/// [`SceneVersion`](ornis_core::SceneVersion): both serialize as the plain
/// underlying JSON numbers, so snapshots and the WASM boundary are unchanged
/// — the `/api/*` transport stays `u64` on the wire and `SceneVersion`
/// converts with `From` at the edges.
pub struct GameWorld<Role: SceneRole = Authoritative> {
    engine: Engine,
    entities: SceneEntities,
    version: SceneVersion,
    role: PhantomData<Role>,
    /// Loaded-but-unspawned assets by handle (parse once, spawn many).
    assets: std::collections::HashMap<AssetHandle, std::rc::Rc<ornis_gltf::LoadedScene>>,
    /// Next asset handle.
    next_asset: u64,
}

/// Handle to a loaded asset in [`GameWorld`]: parse once via
/// [`GameWorld::load_asset`], instantiate with [`GameWorld::spawn_asset`]
/// any number of times, drop the id to release (last `Rc` wins).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetHandle(u64);

/// Browser replica of the scene-backed world: same layout as the
/// authoritative [`GameWorld`], only the transport differs (snapshots cross
/// the serialization boundary instead of shared memory).
pub type ReplicaGameWorld = GameWorld<Replica>;

impl<Role: SceneRole> Default for GameWorld<Role> {
    fn default() -> Self {
        Self::new_in_role()
    }
}

impl GameWorld {
    /// Creates an empty authoritative world with no scene entities.
    pub fn new() -> Self {
        Self::new_in_role()
    }

    /// Creates an authoritative world populated from a serialized scene
    /// description.
    pub fn from_scene(scene: &Scene) -> Self {
        let mut world = Self::new();
        world.replace_scene(scene);
        world
    }

    /// One-shot convenience: [`GameWorld::load_asset`] plus
    /// [`GameWorld::spawn_asset`]. Prefer the split form when the same
    /// asset spawns more than once.
    ///
    /// # Errors
    ///
    /// Same as the two calls it wraps.
    pub fn spawn_gltf(&mut self, path: &std::path::Path) -> Result<SpawnedGltf, SpawnError> {
        let handle = self.load_asset(path)?;
        self.spawn_asset(handle)
    }

    /// Loads an asset file without spawning anything: parse once, spawn
    /// many times via [`GameWorld::spawn_asset`]. The format dispatches
    /// by extension (glTF today; FBX is a future arm, not user code).
    ///
    /// # Errors
    ///
    /// [`SpawnError`] when the format is unsupported or the file cannot
    /// be read or parsed; nothing is retained.
    pub fn load_asset(&mut self, path: &std::path::Path) -> Result<AssetHandle, SpawnError> {
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let loaded = match extension.as_str() {
            "glb" | "gltf" => ornis_gltf::load_path(path)?,
            _ => {
                return Err(SpawnError::UnsupportedFormat {
                    path: path.display().to_string(),
                    extension,
                });
            }
        };
        let handle = AssetHandle(self.next_asset);
        self.next_asset += 1;
        self.assets.insert(handle, std::rc::Rc::new(loaded));
        Ok(handle)
    }

    /// Spawns a loaded asset into the world: mesh entities plus animation
    /// wiring (skeleton roots, named clip playlists, a character-root
    /// [`Animator`](ornis_animation::Animator)) with the needed sampler
    /// systems installed. Loading never starts a skeletal clip — call
    /// [`AnimatorMut::play`](ornis_animation::AnimatorMut::play) with the
    /// clip name.
    ///
    /// # Errors
    ///
    /// [`SpawnError::UnknownAsset`] for a stale handle.
    pub fn spawn_asset(&mut self, handle: AssetHandle) -> Result<SpawnedGltf, SpawnError> {
        let Some(loaded) = self.assets.get(&handle).cloned() else {
            return Err(SpawnError::UnknownAsset(handle.0));
        };
        Ok(self.spawn_loaded(&loaded))
    }

    /// Spawns a scene loaded through [`AssetServer::load`]
    /// (`ornis_assets`): the unified asset path. glTF scenes keep their
    /// loader data on the server and get the same mesh + animation wiring
    /// as [`GameWorld::spawn_asset`]; other scene formats (RON) spawn
    /// their entities with no animation wiring. Loading never autoplays.
    ///
    /// Minimal bridge for the asset track: the world-local registry
    /// ([`GameWorld::load_asset`]/[`AssetHandle`]) is still here and is to
    /// be retired by the core track in favour of this entry point.
    ///
    /// # Errors
    ///
    /// [`SpawnError::UnknownAsset`] when `handle` is not (or no longer)
    /// loaded on `assets`; the world is untouched.
    ///
    /// [`AssetServer::load`]: ornis_assets::AssetServer::load
    pub fn spawn_scene(
        &mut self,
        assets: &ornis_assets::AssetServer,
        handle: &ornis_assets::Handle<Scene>,
    ) -> Result<SpawnedGltf, SpawnError> {
        if let Some(loaded) = assets.loaded_scene(handle.id()) {
            return Ok(self.spawn_loaded(loaded));
        }
        let Some(scene) = assets.get(handle) else {
            return Err(SpawnError::UnknownAsset(handle.id().index()));
        };
        let store = self
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        store.register::<ornis_assets::scene::TransformDesc>();
        store.register::<ornis_assets::scene::MeshDesc>();
        store.register::<ornis_assets::scene::MaterialDesc>();
        let mesh_entities = scene
            .entities
            .iter()
            .map(|desc| {
                let entity = store.create_entity();
                store.insert(entity, desc.transform.clone());
                store.insert(entity, desc.mesh.clone());
                store.insert(entity, desc.material.clone());
                entity
            })
            .collect();
        Ok(SpawnedGltf {
            mesh_entities,
            wiring: crate::anim_wiring::AnimationWiring {
                roots: Vec::new(),
                skel_playlists: Vec::new(),
                anim_playlists: Vec::new(),
                clips: std::collections::HashMap::new(),
            },
        })
    }

    /// Shared body of [`GameWorld::spawn_asset`] and
    /// [`GameWorld::spawn_scene`]: mesh entities plus animation wiring,
    /// installing the sampler systems the wiring needs.
    fn spawn_loaded(&mut self, loaded: &ornis_gltf::LoadedScene) -> SpawnedGltf {
        let spawn = {
            let store = self
                .engine_mut()
                .world_mut()
                .store_mut()
                .expect("engine always carries a store");
            crate::anim_wiring::spawn_gltf_world(store, loaded)
        };
        let wiring = {
            let store = self
                .engine_mut()
                .world_mut()
                .store_mut()
                .expect("engine always carries a store");
            crate::anim_wiring::wire_loaded_animation(store, loaded, &spawn)
        };
        if !wiring.roots.is_empty() || !wiring.skel_playlists.is_empty() {
            crate::install_skeletal_animation(self.engine_mut());
        }
        if !wiring.anim_playlists.is_empty() {
            crate::install_object_animation(self.engine_mut());
        }
        SpawnedGltf {
            mesh_entities: spawn.entities,
            wiring,
        }
    }

    /// Starts the entity's animation player (skeletal or object).
    /// Returns whether a player was found. Loading never autoplays —
    /// hosts start playback explicitly, like a video player.
    pub fn play_animation(&mut self, entity: Entity) -> bool {
        let Some(store) = self.engine_mut().world_mut().store_mut() else {
            return false;
        };
        crate::anim_wiring::set_playing(store, entity, true)
    }

    /// Pauses the entity's animation player at the current clock.
    /// Returns whether a player was found.
    pub fn pause_animation(&mut self, entity: Entity) -> bool {
        let Some(store) = self.engine_mut().world_mut().store_mut() else {
            return false;
        };
        crate::anim_wiring::set_playing(store, entity, false)
    }

    /// Stops the entity's animation player: pauses and rewinds to zero.
    /// Returns whether a player was found.
    pub fn stop_animation(&mut self, entity: Entity) -> bool {
        let Some(store) = self.engine_mut().world_mut().store_mut() else {
            return false;
        };
        crate::anim_wiring::set_playing(store, entity, false);
        crate::anim_wiring::rewind_player(store, entity)
    }
}

/// What [`GameWorld::spawn_gltf`] created: mesh entities plus the
/// animation wiring over them.
pub struct SpawnedGltf {
    /// Mesh entities in load order.
    pub mesh_entities: Vec<Entity>,
    /// Skeleton roots, named clip playlists, and the character-root animator.
    pub wiring: crate::anim_wiring::AnimationWiring,
}

/// Failure to [`GameWorld::spawn_gltf`]: the world is untouched.
#[derive(Debug, thiserror::Error)]
pub enum SpawnError {
    /// The file cannot be read or parsed (loader message preserved).
    #[error(transparent)]
    Import(#[from] ornis_gltf::ImportError),
    /// Extension dispatch knows no loader for this format (glTF today).
    #[error("unsupported asset format for {path}: .{extension}")]
    UnsupportedFormat {
        /// Requested path.
        path: String,
        /// Lowercased extension (empty when absent).
        extension: String,
    },
    /// Stale asset handle (never loaded or already released).
    #[error("unknown asset #{0}")]
    UnknownAsset(u64),
}

impl GameWorld<Replica> {
    /// Creates an empty replica world with no scene entities.
    pub fn new_replica() -> Self {
        Self::new_in_role()
    }

    /// Creates a replica world populated from a serialized scene
    /// description (the snapshot side of the serialization boundary).
    pub fn from_scene_replica(scene: &Scene) -> Self {
        let mut world = Self::new_replica();
        world.replace_scene(scene);
        world
    }
}

impl<Role: SceneRole> GameWorld<Role> {
    fn new_in_role() -> Self {
        Self {
            engine: Engine::new(),
            entities: SceneEntities::new(),
            version: SceneVersion::ZERO,
            role: PhantomData,
            assets: std::collections::HashMap::new(),
            next_asset: 0,
        }
    }

    /// Monotonic scene-mutation counter: bumped by every
    /// [`Self::replace_scene`] (and therefore once by [`Self::from_scene`]).
    /// Frame execution alone never bumps it.
    pub fn version(&self) -> SceneVersion {
        self.version
    }

    /// Returns the shared engine for read-only inspection.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Returns the shared engine for platform setup (orbit camera, physics,
    /// gameplay, audio, GPU resources) and custom systems.
    ///
    /// Scene replacement and frame execution should normally use
    /// [`Self::replace_scene`] and [`Self::frame`] so the entity list and
    /// extraction resource remain consistent.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Number of scene entities currently represented in the ECS.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the handles of entities populated from the current scene.
    ///
    /// The slice excludes auxiliary entities inserted through
    /// [`Self::engine_mut`], which lets a platform attach hidden runtime
    /// components such as physics bodies without changing the scene count.
    pub fn entities(&self) -> &[Entity] {
        self.entities.as_slice()
    }

    /// Returns the scene-entity membership as a
    /// [`SceneEntities`](ornis_core::SceneEntities) newtype.
    ///
    /// Same handles as [`Self::entities`], kept distinct from auxiliary
    /// runtime entities at the type level.
    pub fn scene_entities(&self) -> &SceneEntities {
        &self.entities
    }

    /// Replaces the scene entities with `scene.entities` and publishes the
    /// scene lighting as the [`RenderLights`] resource.
    ///
    /// The camera stays frame/view state owned by the caller; lights and
    /// ambient are world state now. Bumps [`Self::version`] so snapshot
    /// clients can cheaply detect the replacement.
    pub fn replace_scene(&mut self, scene: &Scene) {
        let previous = std::mem::take(&mut self.entities);
        if let Some(store) = self.engine.world().store() {
            for entity in previous.iter().copied() {
                if store.is_alive(entity) {
                    store.destroy_entity(entity);
                }
            }
        }
        self.entities = insert_scene_entities(&mut self.engine, &scene.entities).into();
        self.publish_render_lights(RenderLights::from_scene(scene));
        self.version.bump();
    }

    /// Sets the linear ambient color.
    ///
    /// Alpha is not part of the ambient channel. Creates an empty
    /// [`RenderLights`] rig when the world has none, and leaves lights
    /// already published by [`Self::spawn`] or [`Self::replace_scene`] in
    /// place. [`Self::replace_scene`] replaces the whole rig, including
    /// this ambient.
    pub fn set_ambient(&mut self, color: Color) {
        self.ensure_render_lights().ambient = color.to_linear_rgb().as_array();
    }

    /// Places `value` into the world.
    ///
    /// [`DirectionalLight`] is appended to the [`RenderLights`] rig.
    /// [`OrbitCamera`] replaces the client-side view and registers its
    /// input system once. Neither call returns a success flag: a light
    /// the renderer cannot upload is reported by that rig, and the camera
    /// install does not fail.
    pub fn spawn<S: Spawn>(&mut self, value: S) -> S::Output {
        value.spawn_into(self)
    }

    /// Reads the frame payload directly from the component lanes.
    ///
    /// Equivalent to calling [`extract_render_data`] on this world's
    /// store; provided for callers that own the [`GameWorld`].
    pub fn frame_upload(&self) -> FrameUpload {
        match self.engine.world().store() {
            Some(store) => extract_render_data(store),
            None => FrameUpload::default(),
        }
    }

    /// Runs one frame and returns its CPU-side render payload.
    ///
    /// The engine advances bounded fixed steps first, then the
    /// once-per-frame schedule; extraction observes the final poses.
    /// Platform GPU presentation (`RenderFrame3D` / WASM adapter) consumes
    /// the returned [`FrameUpload`] through its own existing path.
    pub fn frame(&mut self, delta_seconds: f32) -> FrameUpload {
        self.frame_secs(Seconds::new(delta_seconds))
    }

    /// Runs one frame with a [`Seconds`] delta and returns its CPU-side
    /// render payload (same contract as [`Self::frame`]: finite and
    /// non-negative).
    pub fn frame_secs(&mut self, delta: Seconds) -> FrameUpload {
        self.engine.run_frame_secs(delta);
        self.frame_upload()
    }

    /// Single write of the light rig. [`Self::new`] does not publish one,
    /// so a world that never calls [`Self::replace_scene`], [`Self::set_ambient`]
    /// or [`Self::spawn`] stays without lights until the platform's fallback.
    fn publish_render_lights(&mut self, lights: RenderLights) {
        let _ = self.engine.world_mut().insert(lights);
    }

    fn ensure_render_lights(&mut self) -> &mut RenderLights {
        if self
            .engine
            .world()
            .resources()
            .get::<RenderLights>()
            .is_none()
        {
            self.publish_render_lights(RenderLights {
                ambient: LinearRgb::BLACK.as_array(),
                lights: Vec::new(),
                ..RenderLights::default()
            });
        }
        self.engine
            .world_mut()
            .resources_mut()
            .get_mut::<RenderLights>()
            .expect("RenderLights was just published")
    }
}

fn insert_scene_entities(engine: &mut Engine, entities: &[EntityDesc]) -> Vec<Entity> {
    let Some(store) = engine.world_mut().store_mut() else {
        return Vec::new();
    };
    let mut handles = Vec::with_capacity(entities.len());
    for entity in entities {
        let handle = store.create_entity();
        store.insert(handle, entity.transform.clone());
        store.insert(handle, entity.mesh.clone());
        store.insert(handle, entity.material.clone());
        handles.push(handle);
    }
    handles
}

/// Value [`GameWorld::spawn`] can place into the world.
///
/// Infallible values use `Output = ()`. A fallible spawn uses
/// `Output = Result<_, _>` rather than a `bool` or a count.
pub trait Spawn {
    /// What a spawn hands back.
    type Output;

    /// Inserts `self` into `world`.
    fn spawn_into<Role: SceneRole>(self, world: &mut GameWorld<Role>) -> Self::Output;
}

impl Spawn for DirectionalLight {
    type Output = ();

    fn spawn_into<Role: SceneRole>(self, world: &mut GameWorld<Role>) -> Self::Output {
        let desc = self.to_light_desc();
        world.ensure_render_lights().lights.push(desc);
    }
}

impl Spawn for OrbitCamera {
    type Output = ();

    fn spawn_into<Role: SceneRole>(self, world: &mut GameWorld<Role>) -> Self::Output {
        install_orbit_camera(world.engine_mut(), self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use ornis_assets::scene::{
        CameraDesc, LightDesc, MaterialDesc, MeshDesc, ShadowCast, TransformDesc,
    };
    use ornis_core::units::{Clamped01, PositiveF32};
    use ornis_core::{Color, Degrees, Lux, Stage as CoreStage, Time, UnitVec3};
    use ornis_physics::RigidBody;
    use ornis_render::{DirectionalLight, OrbitCamera, read_orbit_camera};

    /// Minimal probe system for staged-plan lookups.
    struct StageProbe(&'static str);

    impl ornis_core::System for StageProbe {
        fn name(&self) -> &'static str {
            self.0
        }
        fn access(&self) -> ornis_core::SystemAccess {
            ornis_core::SystemAccess::new()
        }
        fn run(&self, _: &ornis_core::Resources) {}
    }

    fn two_sphere_scene() -> Scene {
        let entity = |name: &str, x: f32| EntityDesc {
            name: name.into(),
            transform: TransformDesc::from_translation(glam::Vec3::new(x, 0.0, 0.0)),
            mesh: MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 16,
                rings: 8,
            },
            material: MaterialDesc::Dielectric {
                base_color: [0.8, 0.2, 0.2],
                roughness: Clamped01::new(0.5),
                emission: [0.0, 0.0, 0.0],
            },
        };
        Scene {
            name: "runtime-test".into(),
            entities: vec![entity("a", -1.0), entity("b", 1.0)],
            lights: Vec::new(),
            camera: CameraDesc {
                position: glam::Vec3::new(0.0, 2.5, 9.0),
                target: glam::Vec3::ZERO,
                up: ornis_core::units::UnitVec3::Y,
                fov: ornis_core::units::Degrees::new(60.0),
                near: ornis_core::units::Meters::new(0.1),
                far: ornis_core::units::Meters::new(100.0),
            },
            ambient: [0.1, 0.1, 0.1],
        }
    }

    fn scene() -> Scene {
        Scene {
            name: "test".into(),
            entities: vec![EntityDesc {
                name: "sphere".into(),
                transform: TransformDesc::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)),
                mesh: MeshDesc::Sphere {
                    radius: PositiveF32::expect_valid(2.0),
                    segments: 48,
                    rings: 32,
                },
                material: MaterialDesc::Metal {
                    base_color: [0.9, 0.8, 0.2],
                    roughness: Clamped01::new(0.2),
                    emission: [0.0, 0.0, 0.0],
                },
            }],
            lights: Vec::new(),
            camera: CameraDesc {
                position: glam::Vec3::new(0.0, 2.5, 9.0),
                target: glam::Vec3::ZERO,
                up: ornis_core::units::UnitVec3::Y,
                fov: ornis_core::units::Degrees::new(60.0),
                near: ornis_core::units::Meters::new(0.1),
                far: ornis_core::units::Meters::new(100.0),
            },
            ambient: [0.1, 0.1, 0.1],
        }
    }

    #[test]
    fn frame_runs_shared_host_and_extracts_scene() {
        let scene = two_sphere_scene();
        let mut world = GameWorld::from_scene(&scene);
        assert_eq!(world.entity_count(), 2);
        assert_eq!(world.entities().len(), 2);

        let upload = world.frame(1.0 / 60.0);
        assert_eq!(upload.instances.len(), 2);
        // Identical materials dedup to one table entry (render extraction canon).
        assert_eq!(upload.materials.len(), 1);
        assert_eq!(upload.instances[0].material_index.as_u32(), 0);
        assert_eq!(upload.instances[1].material_index.as_u32(), 0);
        let time = world
            .engine()
            .world()
            .resources()
            .get::<Time>()
            .expect("engine publishes Time");
        assert_eq!(time.frame(), 1);
    }

    #[test]
    fn empty_frame_advances_clock_without_content() {
        let mut world = GameWorld::new();
        let upload = world.frame(1.0 / 60.0);
        assert!(upload.instances.is_empty());
        assert!(upload.materials.is_empty());
        let time = world
            .engine()
            .world()
            .resources()
            .get::<Time>()
            .expect("engine publishes Time");
        assert_eq!(time.frame(), 1);
    }

    #[test]
    fn hidden_floor_never_enters_frame_upload() {
        let scene = two_sphere_scene();
        let mut world = GameWorld::from_scene(&scene);
        let floor = {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            let floor = store.create_entity();
            store.insert(
                floor,
                RigidBody::new_box(Vec3::new(0.0, -2.0, 0.0), Vec3::new(20.0, 1.0, 20.0), 0.0),
            );
            floor
        };
        assert!(
            world
                .engine()
                .world()
                .store()
                .expect("store")
                .is_alive(floor)
        );
        assert_eq!(world.entity_count(), 2);

        let upload = world.frame(1.0 / 60.0);
        assert_eq!(upload.instances.len(), 2);
        assert!(upload.custom_meshes.is_empty());
    }

    fn render_lights(world: &GameWorld) -> RenderLights {
        world
            .engine()
            .world()
            .resources()
            .get::<RenderLights>()
            .expect("RenderLights")
            .clone()
    }

    #[test]
    fn typed_light_ambient_and_camera_publish_one_rig() {
        let mut world = GameWorld::new();
        let ambient = Color::hex("#1A1A26").expect("hex");
        world.set_ambient(ambient);
        world.spawn(DirectionalLight {
            direction: UnitVec3::new(Vec3::new(-1.0, -1.0, -1.0)).expect("direction"),
            illuminance: Lux(0.6),
            color: Color::WHITE,
            ..Default::default()
        });
        world.spawn(
            OrbitCamera::looking_at(Vec3::new(2.5, 1.8, 3.5), Vec3::Y).with_fov(Degrees(45.0)),
        );

        let rig = render_lights(&world);
        assert_eq!(rig.lights.len(), 1);
        assert_eq!(rig.ambient, ambient.to_linear_rgb().as_array());
        let camera = read_orbit_camera(world.engine()).expect("camera");
        assert_eq!(camera.view_parameters().3, 45.0);

        world.spawn(OrbitCamera::looking_at(
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::ZERO,
        ));
        assert_eq!(
            world
                .engine()
                .schedule()
                .mermaid()
                .matches("orbit_camera_input")
                .count(),
            1
        );
    }

    #[test]
    fn replace_scene_replaces_spawned_lights() {
        let mut world = GameWorld::new();
        world.set_ambient(Color::WHITE);
        world.spawn(DirectionalLight {
            direction: UnitVec3::Y,
            illuminance: Lux(0.6),
            color: Color::WHITE,
            ..Default::default()
        });
        world.replace_scene(&scene());
        let rig = render_lights(&world);
        assert!(rig.lights.is_empty());
        assert_eq!(rig.ambient, [0.1, 0.1, 0.1]);
    }

    #[test]
    fn spawn_after_replace_appends_one_light_and_keeps_ambient() {
        let mut world = GameWorld::from_scene(&scene());
        world.spawn(DirectionalLight {
            direction: UnitVec3::Y,
            illuminance: Lux(1.0),
            color: Color::WHITE,
            ..Default::default()
        });
        let rig = render_lights(&world);
        assert_eq!(rig.lights.len(), 1);
        assert_eq!(rig.ambient, [0.1, 0.1, 0.1]);
    }

    #[test]
    fn set_ambient_after_from_scene_keeps_scene_lights() {
        let mut lit = scene();
        lit.lights.push(LightDesc::Directional {
            direction: UnitVec3::Y,
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow: ShadowCast::Disabled,
        });
        let mut world = GameWorld::from_scene(&lit);
        world.set_ambient(Color::WHITE);
        let rig = render_lights(&world);
        assert_eq!(rig.lights.len(), 1);
        assert_eq!(rig.ambient, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn replace_scene_swaps_entities() {
        let mut world = GameWorld::from_scene(&two_sphere_scene());
        let empty = Scene {
            entities: Vec::new(),
            ..two_sphere_scene()
        };
        world.replace_scene(&empty);
        assert_eq!(world.entity_count(), 0);
        assert!(world.frame(0.0).instances.is_empty());
    }

    #[test]
    fn version_counts_scene_replacements_only() {
        let mut world = GameWorld::new();
        assert_eq!(world.version(), 0);
        world.frame(1.0 / 60.0);
        assert_eq!(world.version(), 0);
        world.replace_scene(&two_sphere_scene());
        assert_eq!(world.version(), 1);
        world.replace_scene(&two_sphere_scene());
        assert_eq!(world.version(), 2);
        let from_scene = GameWorld::from_scene(&two_sphere_scene());
        assert_eq!(from_scene.version(), 1);
    }

    #[test]
    fn replica_role_shares_layout_with_typed_version_and_secs_frame() {
        use ornis_core::{SceneVersion, Seconds};
        // A bare `GameWorld` stays authoritative; the browser replica is
        // the same layout under the `Replica` role.
        let authoritative = GameWorld::from_scene(&two_sphere_scene());
        let mut replica: ReplicaGameWorld =
            ReplicaGameWorld::from_scene_replica(&two_sphere_scene());
        assert_eq!(authoritative.version(), SceneVersion::new(1));
        assert_eq!(replica.version(), 1);
        assert_eq!(replica.scene_entities().len(), 2);
        assert_eq!(replica.entities(), authoritative.entities());
        let upload = replica.frame_secs(Seconds::new(1.0 / 60.0));
        assert_eq!(upload.instances.len(), 2);
        // Frame execution alone never bumps the scene version.
        assert_eq!(replica.version(), SceneVersion::new(1));
    }

    #[test]
    fn core_stage_names_are_stable() {
        assert_eq!(CoreStage::PreUpdate.name(), "pre_update");
        assert_eq!(CoreStage::Input.name(), "input");
        assert_eq!(CoreStage::Gameplay.name(), "gameplay");
        assert_eq!(CoreStage::PostFrame.name(), "post_frame");
    }

    #[test]
    fn each_registered_system_lives_in_its_staged_plan() {
        use crate::install_unified_runtime;
        let mut engine = Engine::new();
        install_unified_runtime(&mut engine);
        // Every system the unified runtime registers must sit in its
        // staged plan (`Gameplay` delegates to the fixed schedule,
        // `PostFrame` to the once-per-frame schedule).
        let gameplay = engine.stage_schedule(CoreStage::Gameplay).mermaid();
        let post_frame = engine.stage_schedule(CoreStage::PostFrame).mermaid();
        for name in ["physics_push", "velocity_to_body"] {
            assert!(
                gameplay.contains(name),
                "{name} should be registered in the Gameplay stage"
            );
        }
        for name in ["player_input", "transform_update", "body_to_transform"] {
            assert!(
                post_frame.contains(name),
                "{name} should be registered in the PostFrame stage"
            );
        }
    }

    #[test]
    fn stage_registration_channels_route_to_real_plans() {
        use ornis_core::Engine;
        let mut engine = Engine::new();
        // Each registrable system goes through the staged channels:
        // `add_stage_system` and `stage_schedule_mut`.
        engine.add_stage_system(CoreStage::Input, StageProbe("player_input"));
        engine
            .stage_schedule_mut(CoreStage::Gameplay)
            .add_system(StageProbe("physics_step"));

        assert_eq!(engine.stage_schedule(CoreStage::Input).len(), 1);
        assert!(
            engine
                .stage_schedule(CoreStage::Input)
                .mermaid()
                .contains("player_input")
        );
        let gameplay = engine.stage_schedule(CoreStage::Gameplay);
        assert_eq!(gameplay.len(), engine.fixed_schedule().len());
        assert!(gameplay.mermaid().contains("physics_step"));
        assert!(engine.stage_schedule(CoreStage::PreUpdate).is_empty());
    }

    #[test]
    fn game_world_extracts_scene_entities_from_lanes() {
        let mut world = GameWorld::from_scene(&scene());
        assert_eq!(world.entity_count(), 1);
        world.frame(0.0);

        let extracted = world.frame_upload();
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(extracted.materials.len(), 1);
        assert_eq!(extracted.instances.len(), 1);
        assert_eq!(extracted.instances[0].material_index.as_u32(), 0);
        assert_eq!(
            extracted.instances[0].model_matrix.w_axis.truncate(),
            Vec3::new(1.0, 2.0, 3.0)
        );
    }

    #[test]
    fn replacing_scene_destroys_previous_entities_before_extraction() {
        let mut world = GameWorld::from_scene(&scene());
        let empty = Scene {
            entities: Vec::new(),
            ..scene()
        };
        world.replace_scene(&empty);
        world.frame(0.0);

        assert_eq!(world.entity_count(), 0);
        assert!(world.frame_upload().instances.is_empty());
    }

    #[test]
    fn frame_upload_matches_the_direct_lane_canon() {
        // X1/X4 (Extract-free) data gate: `extract_render_data` is the
        // single canon — `GameWorld::frame_upload` and a plain canon
        // call on the same store must agree for a scene mixing all three
        // material kinds and varied tessellation (instances field-wise
        // over the glam matrices, materials by `Debug` bytes).
        let varied = Scene {
            name: "oracle".into(),
            entities: vec![
                EntityDesc {
                    name: "dielectric".into(),
                    transform: TransformDesc::from_translation(glam::Vec3::new(1.0, 2.0, 3.0)),
                    mesh: MeshDesc::Sphere {
                        radius: PositiveF32::expect_valid(2.0),
                        segments: 24,
                        rings: 16,
                    },
                    material: MaterialDesc::Dielectric {
                        base_color: [0.8, 0.2, 0.2],
                        roughness: Clamped01::new(0.4),
                        emission: [0.0, 0.0, 0.0],
                    },
                },
                EntityDesc {
                    name: "metal".into(),
                    transform: TransformDesc::from_arrays(
                        [-1.0, 0.0, 2.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [2.0, 2.0, 2.0],
                    ),
                    mesh: MeshDesc::Sphere {
                        radius: PositiveF32::expect_valid(0.5),
                        segments: 48,
                        rings: 32,
                    },
                    material: MaterialDesc::Metal {
                        base_color: [0.9, 0.8, 0.2],
                        roughness: Clamped01::new(0.2),
                        emission: [0.0, 0.0, 0.0],
                    },
                },
                EntityDesc {
                    name: "coat".into(),
                    transform: TransformDesc::from_arrays(
                        [0.0, 5.0, -3.0],
                        [0.3, 0.2, 0.1, 0.9],
                        [1.0, 1.0, 1.0],
                    ),
                    mesh: MeshDesc::Sphere {
                        radius: PositiveF32::expect_valid(1.0),
                        segments: 32,
                        rings: 24,
                    },
                    material: MaterialDesc::Coat {
                        base_color: [0.2, 0.4, 0.9],
                        coat_weight: Clamped01::new(0.7),
                        coat_roughness: Clamped01::new(0.1),
                        emission: [0.0, 0.0, 0.0],
                    },
                },
            ],
            lights: Vec::new(),
            camera: CameraDesc {
                position: glam::Vec3::new(0.0, 2.5, 9.0),
                target: glam::Vec3::ZERO,
                up: ornis_core::units::UnitVec3::Y,
                fov: ornis_core::units::Degrees::new(60.0),
                near: ornis_core::units::Meters::new(0.1),
                far: ornis_core::units::Meters::new(100.0),
            },
            ambient: [0.1, 0.1, 0.1],
        };
        let mut world = GameWorld::from_scene(&varied);
        world.frame(0.0);

        // X1/X4 canon: one direct lane read covers all material kinds;
        // `GameWorld::frame_upload` must agree with the plain canon
        // call on the same store.
        let snapshot = world.frame_upload();
        let direct = extract_render_data(world.engine().world().store().expect("store"));
        assert_eq!(snapshot.mesh_params, (48, 32));
        assert_eq!(snapshot.instances.len(), 3);
        assert_eq!(direct.mesh_params, snapshot.mesh_params);
        assert_eq!(direct.instances.len(), snapshot.instances.len());
        for (direct, snapshot) in direct.instances.iter().zip(&snapshot.instances) {
            assert_eq!(direct.model_matrix, snapshot.model_matrix);
            assert_eq!(direct.normal_matrix, snapshot.normal_matrix);
            assert_eq!(direct.material_index, snapshot.material_index);
        }
        assert_eq!(direct.materials.len(), snapshot.materials.len());
        for (direct, snapshot) in direct.materials.iter().zip(&snapshot.materials) {
            assert_eq!(format!("{direct:?}"), format!("{snapshot:?}"));
        }
    }

    #[test]
    fn max_mesh_params_is_the_extraction_mesh_canon() {
        // X2 canon tie: the mesh re-create criterion must be exactly the
        // oracle's `mesh_params` — max tessellation over COMPLETE entities
        // only, floor (32, 24). The incomplete entity (mesh + material,
        // no transform, 96/64) must not push the maximum.
        use ornis_render::{extract_render_data as canon, max_mesh_params};
        let complete = Scene {
            name: "canon".into(),
            entities: vec![
                EntityDesc {
                    name: "fine".into(),
                    transform: TransformDesc::IDENTITY,
                    mesh: MeshDesc::Sphere {
                        radius: PositiveF32::expect_valid(1.0),
                        segments: 48,
                        rings: 32,
                    },
                    material: MaterialDesc::Metal {
                        base_color: [0.9, 0.8, 0.2],
                        roughness: Clamped01::new(0.2),
                        emission: [0.0, 0.0, 0.0],
                    },
                },
                EntityDesc {
                    name: "coarse".into(),
                    transform: TransformDesc::from_translation(glam::Vec3::new(2.0, 0.0, 0.0)),
                    mesh: MeshDesc::Sphere {
                        radius: PositiveF32::expect_valid(1.0),
                        segments: 16,
                        rings: 12,
                    },
                    material: MaterialDesc::Dielectric {
                        base_color: [0.2, 0.8, 0.2],
                        roughness: Clamped01::new(0.5),
                        emission: [0.0, 0.0, 0.0],
                    },
                },
            ],
            lights: Vec::new(),
            camera: CameraDesc {
                position: glam::Vec3::new(0.0, 2.5, 9.0),
                target: glam::Vec3::ZERO,
                up: ornis_core::units::UnitVec3::Y,
                fov: ornis_core::units::Degrees::new(60.0),
                near: ornis_core::units::Meters::new(0.1),
                far: ornis_core::units::Meters::new(100.0),
            },
            ambient: [0.1, 0.1, 0.1],
        };
        let mut world = GameWorld::from_scene(&complete);
        // Incomplete entity: mesh + material without a transform lane
        // entry — a high tessellation that must NOT move the maximum.
        let store = world.engine_mut().world_mut().store_mut().expect("store");
        let incomplete = store.create_entity();
        store.insert(
            incomplete,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 96,
                rings: 64,
            },
        );
        store.insert(
            incomplete,
            MaterialDesc::Coat {
                base_color: [0.2, 0.4, 0.9],
                coat_weight: Clamped01::new(0.7),
                coat_roughness: Clamped01::new(0.1),
                emission: [0.0, 0.0, 0.0],
            },
        );
        world.frame(0.0);

        let store = world.engine().world().store().expect("store");
        let extracted = canon(store);
        assert_eq!(extracted.mesh_params, (48, 32));
        assert_eq!(max_mesh_params(store), extracted.mesh_params);
    }

    #[test]
    fn replace_scene_publishes_scene_lighting_as_resource() {
        // X3: the scene loader owns lights/ambient — replacing a scene
        // must publish them as the `RenderLights` resource.
        use ornis_assets::scene::LightDesc;
        let world = GameWorld::from_scene(&Scene {
            lights: vec![LightDesc::Directional {
                direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.0, -1.0, 0.0))
                    .expect("non-zero direction"),
                intensity: 2.0,
                color: [1.0, 0.9, 0.8],
                shadow: ShadowCast::Disabled,
            }],
            ambient: [0.2, 0.2, 0.2],
            ..scene()
        });
        let lights = world
            .engine()
            .world()
            .resources()
            .get::<RenderLights>()
            .expect("scene loader publishes RenderLights");
        assert_eq!(lights.ambient, [0.2, 0.2, 0.2]);
        assert_eq!(lights.lights.len(), 1);
        assert!(matches!(
            lights.set_lights_args().as_slice(),
            [LightDesc::Directional {
                direction: d,
                intensity: v,
                color: [1.0, 0.9, 0.8],
                shadow: ShadowCast::Disabled,
            }] if *v == 2.0 && d.as_array() == [0.0, -1.0, 0.0]
        ));
    }

    /// Scene-first facade: empty world, spawned asset, explicit light and
    /// camera — no engine, store, or installer calls. The world starts
    /// dark (no silent rig) and skeletal playback waits for a named `play`.
    #[test]
    fn facade_spawns_asset_with_explicit_light_and_camera() {
        let mut world = GameWorld::new();
        assert!(
            world
                .engine()
                .world()
                .resources()
                .get::<RenderLights>()
                .is_none(),
            "world starts dark"
        );
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let spawned = world.spawn_gltf(&starter).expect("starter loads");
        assert!(!spawned.mesh_entities.is_empty());
        assert!(!spawned.wiring.roots.is_empty());
        assert!(!spawned.wiring.skel_playlists.is_empty());

        world.set_ambient(Color::linear_rgb(0.1, 0.1, 0.15));
        world.spawn(DirectionalLight {
            direction: UnitVec3::new(Vec3::new(1.0, 1.0, 1.0)).expect("direction"),
            illuminance: Lux(0.6),
            color: Color::WHITE,
            ..Default::default()
        });
        let rig = world
            .engine()
            .world()
            .resources()
            .get::<RenderLights>()
            .expect("explicit light");
        assert_eq!(rig.ambient, [0.1, 0.1, 0.15]);
        assert_eq!(rig.lights.len(), 1);

        world.spawn(
            OrbitCamera::looking_at(Vec3::new(2.5, 1.8, 3.5), Vec3::new(0.0, 1.0, 0.0))
                .with_fov(Degrees(45.0)),
        );
        assert!(
            world
                .engine()
                .world()
                .resources()
                .get::<std::sync::Mutex<ornis_render::camera::OrbitCamera>>()
                .is_some(),
            "orbit camera installed"
        );

        let missing = world.spawn_gltf(std::path::Path::new("nope.glb"));
        assert!(missing.is_err(), "missing file rejects");
    }

    /// The unified asset path: `AssetServer::load::<Scene>` + `spawn_scene`
    /// matches the world-local `spawn_gltf` result (entities and wiring),
    /// and the same handle spawns more than once.
    #[test]
    fn spawn_scene_from_asset_server_matches_spawn_gltf() {
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mut assets = ornis_assets::AssetServer::new();
        let mannequin: ornis_assets::Handle<Scene> = assets.load(&starter).expect("starter loads");

        let mut world = GameWorld::new();
        let hero = world
            .spawn_scene(&assets, &mannequin)
            .expect("loaded handle spawns");
        let mut reference = GameWorld::new();
        let legacy = reference.spawn_gltf(&starter).expect("starter loads");
        assert_eq!(hero.mesh_entities.len(), legacy.mesh_entities.len());
        assert_eq!(hero.wiring.roots.len(), legacy.wiring.roots.len());
        assert_eq!(
            hero.wiring.skel_playlists.len(),
            legacy.wiring.skel_playlists.len()
        );
        assert!(!hero.wiring.roots.is_empty());
        assert!(
            hero.wiring.clips.contains_key("Walk_Loop"),
            "spawn_scene indexes skeletal clips by name"
        );
        assert_eq!(hero.wiring.clips.len(), legacy.wiring.clips.len());

        let again = world
            .spawn_scene(&assets, &mannequin)
            .expect("parse once, spawn many");
        assert_eq!(again.mesh_entities.len(), hero.mesh_entities.len());
    }

    /// RON scenes spawn their entities with empty wiring; unloaded
    /// handles reject without touching the world.
    #[test]
    fn spawn_scene_ron_and_unknown_handle() {
        let ron_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/scene.ron");
        let mut assets = ornis_assets::AssetServer::new();
        let handle: ornis_assets::Handle<Scene> = assets.load(&ron_path).expect("ron loads");
        let expected = assets.get(&handle).expect("loaded").entities.len();
        let mut world = GameWorld::new();
        let spawned = world.spawn_scene(&assets, &handle).expect("ron spawns");
        assert_eq!(spawned.mesh_entities.len(), expected);
        assert_eq!(spawned.wiring.added(), 0);

        assert!(assets.unload(&handle));
        assert!(matches!(
            world.spawn_scene(&assets, &handle),
            Err(SpawnError::UnknownAsset(_))
        ));
    }

    /// `Walk_Loop` is chosen by name (not the baked TPose, not the first
    /// moving clip) and joints visibly travel: pose snapshots over 90
    /// frames take more than one distinct value.
    #[test]
    fn walk_loop_moves_joints_over_time() {
        use ornis_animation::{JointPose, SkelPlayer, try_animator};
        let mut world = GameWorld::new();
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let spawned = world.spawn_gltf(&starter).expect("starter loads");
        assert!(spawned.wiring.skel_playlists.len() > 1);
        let walk = spawned
            .wiring
            .clips
            .get("Walk_Loop")
            .copied()
            .expect("Walk_Loop");
        assert_ne!(walk.0, spawned.wiring.skel_playlists[0]);

        let hero = spawned.mesh_entities[0];
        {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            assert!(
                store
                    .read_lane::<SkelPlayer>()
                    .is_none_or(|lane| lane.is_empty()),
                "no skeletal cursor until play"
            );
            try_animator(store, hero)
                .expect("animator")
                .play("Walk_Loop")
                .expect("Walk_Loop");
        }

        let store = world.engine().world().store().expect("store");
        {
            let players = store.read_lane::<SkelPlayer>().expect("players");
            let played: Vec<_> = players
                .entities
                .iter()
                .map(|e| players.get(*e).unwrap().clip)
                .collect();
            assert!(!played.is_empty());
            assert!(
                played.iter().all(|clip| *clip == walk),
                "play must select Walk_Loop, not the static first clip"
            );
        }

        let mut seen = std::collections::HashSet::new();
        for _ in 0..90 {
            world.engine_mut().run_frame(1.0 / 60.0);
            let store = world.engine().world().store().expect("store");
            let poses = store.read_lane::<JointPose>().expect("poses");
            // Whole pose: the no-root-motion export keeps the root joint
            // static, motion lives in the descendants.
            let pose = poses
                .entities
                .iter()
                .find_map(|e| poses.get(*e))
                .expect("pose");
            seen.insert(format!("{:?}", pose.matrices));
        }
        assert!(seen.len() > 1, "joints must travel, TPose would repeat");
    }

    /// Playback controls behave like a video player: play starts a
    /// paused player, pause holds the clock, stop pauses and rewinds.
    #[test]
    fn playback_controls_play_pause_stop() {
        use ornis_animation::{ClipId, SkelPlayer};
        use ornis_core::{Clamped01, Seconds};
        let mut world = GameWorld::new();
        let store = world.engine_mut().world_mut().store_mut().expect("store");
        let entity = store.create_entity();
        store.insert(
            entity,
            SkelPlayer {
                clip: ClipId(entity),
                time: Seconds::new(5.0),
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: false,
                looping: true,
            },
        );
        assert!(world.play_animation(entity));
        assert!(!world.pause_animation(Entity::new(999)));
        assert!(world.pause_animation(entity));
        assert!(world.stop_animation(entity));
        assert!(!world.play_animation(Entity::new(999)));

        let store = world.engine().world().store().expect("store");
        let player = *store
            .read_lane::<SkelPlayer>()
            .expect("lane")
            .get(entity)
            .expect("player");
        assert!(!player.playing);
        assert_eq!(player.time, Seconds::ZERO);
    }
}
