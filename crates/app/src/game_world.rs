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
    Authoritative, Color, Engine, Entity, Position, Replica, SceneEntities, SceneRole,
    SceneVersion, Seconds, UnitQuat,
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
    /// Asset registry for this world. Users load through [`Self::load`] or
    /// [`Self::assets_mut`] rather than constructing a standalone server.
    assets: ornis_assets::AssetServer,
    /// Window title. Default is `Ornis Engine`; [`Self::set_title`] replaces it.
    title: String,
    role: PhantomData<Role>,
}

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

    /// Spawns a RON [`Scene`] loaded through [`Self::assets_mut`] and returns
    /// one root entity.
    ///
    /// Mesh entities are spawned flat (no animator). A glTF [`Model`](ornis_assets::Model)
    /// spawns through [`GameWorld::spawn`] of its [`Handle`](ornis_assets::Handle).
    /// Loading never starts a clip.
    ///
    /// # Errors
    ///
    /// [`AssetError::UnknownHandle`](ornis_assets::AssetError::UnknownHandle)
    /// when `handle` is not loaded. The world is untouched.
    pub fn spawn_scene(
        &mut self,
        handle: &ornis_assets::Handle<Scene>,
    ) -> Result<Entity, ornis_assets::AssetError> {
        let Some(scene) = self.assets.get(handle).cloned() else {
            return Err(ornis_assets::AssetError::UnknownHandle {
                index: handle.id().index(),
            });
        };
        let store = self
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        store.register::<ornis_assets::scene::TransformDesc>();
        store.register::<ornis_assets::scene::MeshDesc>();
        store.register::<ornis_assets::scene::MaterialDesc>();
        let root = store.create_entity();
        for desc in &scene.entities {
            let entity = store.create_entity();
            store.insert(entity, desc.transform.clone());
            crate::insert_flat_pose(store, entity, &desc.transform);
            store.insert(entity, desc.mesh.clone());
            store.insert(entity, desc.material.clone());
        }
        Ok(root)
    }

    /// Starts the entity's animation player (skeletal or object).
    ///
    /// Loading never autoplays. Named skeletal clips start through
    /// [`EntityMut`] and [`AnimatorAccess::animator`](ornis_animation::AnimatorAccess::animator).
    ///
    /// # Errors
    ///
    /// [`PlaybackError`] when `entity` has no player.
    pub fn play_animation(&mut self, entity: Entity) -> Result<(), PlaybackError> {
        let store = self
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        if crate::anim_wiring::set_playing(store, entity, true) {
            Ok(())
        } else {
            Err(PlaybackError)
        }
    }

    /// Pauses the entity's animation player at the current clock.
    ///
    /// # Errors
    ///
    /// [`PlaybackError`] when `entity` has no player.
    pub fn pause_animation(&mut self, entity: Entity) -> Result<(), PlaybackError> {
        let store = self
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        if crate::anim_wiring::set_playing(store, entity, false) {
            Ok(())
        } else {
            Err(PlaybackError)
        }
    }

    /// Stops the entity's animation player: pauses and rewinds to zero.
    ///
    /// # Errors
    ///
    /// [`PlaybackError`] when `entity` has no player.
    pub fn stop_animation(&mut self, entity: Entity) -> Result<(), PlaybackError> {
        let store = self
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        let found = crate::anim_wiring::set_playing(store, entity, false);
        let rewound = crate::anim_wiring::rewind_player(store, entity);
        if found || rewound {
            Ok(())
        } else {
            Err(PlaybackError)
        }
    }
}

/// The entity has no skeletal or object animation player.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("entity has no animation player")]
pub struct PlaybackError;

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
            assets: ornis_assets::AssetServer::new(),
            title: "Ornis Engine".to_owned(),
            role: PhantomData,
        }
    }

    /// The asset registry owned by this world.
    pub fn assets(&self) -> &ornis_assets::AssetServer {
        &self.assets
    }

    /// The asset registry owned by this world.
    ///
    /// Load with [`Self::load`] or `assets_mut().load::<T>(path)`. The returned
    /// [`Handle`](ornis_assets::Handle) is owned, so the borrow ends before
    /// [`Self::spawn`] or [`GameWorld::spawn_scene`].
    pub fn assets_mut(&mut self) -> &mut ornis_assets::AssetServer {
        &mut self.assets
    }

    /// Loads the asset at `path` as `A`.
    ///
    /// Same contract as [`AssetServer::load`](ornis_assets::AssetServer::load):
    /// the type parameter selects the asset and the extension selects the
    /// importer. Delegates to [`Self::assets_mut`].
    ///
    /// # Errors
    ///
    /// The same [`AssetError`](ornis_assets::AssetError) values as
    /// [`AssetServer::load`](ornis_assets::AssetServer::load).
    pub fn load<A: ornis_assets::Asset>(
        &mut self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<ornis_assets::Handle<A>, ornis_assets::AssetError> {
        self.assets_mut().load(path)
    }

    /// Stores the window title.
    ///
    /// The default, set by [`GameWorld::new`] and
    /// [`ReplicaGameWorld::new_replica`], is `Ornis Engine`.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.title = title.into();
    }

    /// Window title stored by [`Self::set_title`].
    pub fn title(&self) -> &str {
        &self.title
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

    /// Sets the ambient [`Color`].
    ///
    /// The resource stores `color` as given. The GPU upload reads linear
    /// RGB and drops alpha. Creates an empty [`RenderLights`] rig when the
    /// world has none, and leaves lights already published by
    /// [`Self::spawn`] or [`Self::replace_scene`] in place.
    /// [`Self::replace_scene`] replaces the whole rig, including this
    /// ambient.
    pub fn set_ambient(&mut self, color: Color) {
        self.ensure_render_lights().ambient = color;
    }

    /// Places `value` into the world.
    ///
    /// [`DirectionalLight`] is appended to the [`RenderLights`] rig.
    /// [`OrbitCamera`] replaces the client-side view and registers its
    /// input system once. Neither call returns a success flag: a light
    /// the renderer cannot upload is reported by that rig, and the camera
    /// install does not fail. A [`ModelSpawn`] returns the character root,
    /// or [`SpawnModelError`] when the handle is not loaded or the parent
    /// is dead — those paths create no entities.
    pub fn spawn<S: Spawn>(&mut self, value: S) -> S::Output {
        value.spawn_into(self)
    }

    /// Borrows `entity` for component edits, including [`Animator`](ornis_animation::Animator)
    /// playback via [`AnimatorAccess`](ornis_animation::AnimatorAccess).
    pub fn entity_mut(&mut self, entity: Entity) -> EntityMut<'_> {
        let store = self
            .engine
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        EntityMut { store, entity }
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
        ornis_core::install_transform_propagation(self.engine_mut());
        self.engine.run_frame_secs(delta);
        self.frame_upload()
    }

    /// Parents `child` under `parent`, updating [`ChildOf`](ornis_core::ChildOf)
    /// and the parent's [`Children`](ornis_core::Children) together.
    ///
    /// This is the world entry point for parenting. [`SmartStore`](ornis_core::SmartStore)
    /// has no insert hook, so a raw `insert(ChildOf)` leaves the cache stale
    /// until [`reconcile_children`](ornis_core::reconcile_children).
    ///
    /// # Errors
    ///
    /// [`HierarchyError`](ornis_core::HierarchyError) when either entity is
    /// dead, they are the same entity, or the link would cycle.
    pub fn set_parent(
        &mut self,
        child: Entity,
        parent: Entity,
    ) -> Result<(), ornis_core::HierarchyError> {
        let store = self
            .engine
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        ornis_core::set_parent(store, child, parent)
    }

    /// Detaches `child` from its parent and drops the matching cache entry.
    pub fn clear_parent(&mut self, child: Entity) {
        let Some(store) = self.engine.world_mut().store_mut() else {
            return;
        };
        ornis_core::clear_parent(store, child);
    }

    /// Destroys `entity` and its descendants, and drops them from the scene
    /// list and from the parent's [`Children`](ornis_core::Children) cache.
    pub fn despawn_recursive(&mut self, entity: Entity) {
        {
            let Some(store) = self.engine.world_mut().store_mut() else {
                return;
            };
            ornis_core::despawn_recursive(store, entity);
        }
        let Some(store) = self.engine.world().store() else {
            return;
        };
        let live: Vec<Entity> = self
            .entities
            .iter()
            .copied()
            .filter(|entity| store.is_alive(*entity))
            .collect();
        if live.len() != self.entities.len() {
            self.entities = live.into();
            self.version.bump();
        }
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
                ambient: Color::BLACK,
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
        crate::insert_flat_pose(store, handle, &entity.transform);
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
/// [`ModelSpawn`] returns [`Result<Entity, SpawnModelError>`]: an unknown
/// handle or a dead parent creates nothing. [`GameWorld::load`] (or
/// [`GameWorld::assets_mut`]) returns the handle and ends the borrow before
/// `spawn`, so the two calls do not overlap.
/// The implementor copies asset data out of [`GameWorld::assets`] before
/// [`GameWorld::engine_mut`], because both methods borrow the whole world.
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

impl Spawn for ModelSpawn {
    type Output = Result<Entity, SpawnModelError>;

    fn spawn_into<Role: SceneRole>(self, world: &mut GameWorld<Role>) -> Self::Output {
        let Some(model) = world.assets().get(&self.model).cloned() else {
            return Err(SpawnModelError::UnknownHandle {
                index: self.model.id().index(),
            });
        };
        if let Some(parent) = self.parent {
            let alive = world
                .engine()
                .world()
                .store()
                .is_some_and(|store| store.is_alive(parent));
            if !alive {
                return Err(SpawnModelError::InvalidParent { parent });
            }
        }
        Ok(spawn_model_hierarchy(
            world,
            self.model,
            &model,
            self.root_transform(),
            self.parent,
        ))
    }
}

/// Why [`GameWorld::spawn`] of a [`ModelSpawn`] failed.
///
/// The world is unchanged: no entities are created. A failed
/// [`GameWorld::load`] never produces a handle, so spawn only sees a handle
/// that is absent from this world's asset server. A dead `parent` is
/// rejected the same way, before the hierarchy is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SpawnModelError {
    /// The handle is not present on this world's asset server (never loaded
    /// here, already unloaded, or the dangling [`Default`] handle).
    #[error("model handle {index} is not loaded")]
    UnknownHandle {
        /// Raw [`AssetId::index`](ornis_assets::AssetId::index).
        index: u64,
    },
    /// `parent` is not a live entity.
    #[error("model parent {parent:?} is not alive")]
    InvalidParent {
        /// The entity [`ModelSpawn::parent`] named.
        parent: Entity,
    },
}

/// Placement of one [`Model`](ornis_assets::Model) in the world.
///
/// [`GameWorld::spawn`] builds the node hierarchy under a synthetic root.
/// `position`, `rotation`, and `scale` become that root's local
/// [`Transform`](ornis_core::Transform). `parent`, when set, is applied with
/// [`set_parent`](ornis_core::set_parent). The live node map is a separate
/// [`ModelInstance`] component on the root.
///
/// [`Default`] is the origin, identity rotation, unit scale, no parent, and
/// a dangling model handle. Spawning it returns
/// [`SpawnModelError::UnknownHandle`] and creates no entities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelSpawn {
    /// Asset to instantiate. Required; the default handle is not loaded.
    pub model: ornis_assets::Handle<ornis_assets::Model>,
    /// Root translation, in meters. Default is the origin.
    pub position: Position,
    /// Root orientation. Default is the identity.
    pub rotation: UnitQuat,
    /// Root scale per axis. Default is `(1, 1, 1)`, not zero.
    pub scale: glam::Vec3,
    /// Parent of the synthetic root. Default is none.
    pub parent: Option<Entity>,
}

impl ModelSpawn {
    /// Places `model` at the origin with identity rotation and unit scale.
    pub fn new(model: ornis_assets::Handle<ornis_assets::Model>) -> Self {
        Self {
            model,
            ..Self::default()
        }
    }

    fn root_transform(self) -> ornis_core::Transform {
        ornis_core::Transform {
            translation: self.position.get(),
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

impl Default for ModelSpawn {
    /// Origin, identity rotation, unit scale, no parent, dangling model handle.
    fn default() -> Self {
        Self {
            model: ornis_assets::Handle::dangling(),
            position: Position::default(),
            rotation: UnitQuat::IDENTITY,
            scale: glam::Vec3::ONE,
            parent: None,
        }
    }
}

/// Live instance of one [`Model`](ornis_assets::Model).
///
/// `nodes[i]` is the entity spawned for [`NodeIdx`](ornis_assets::NodeIdx)`(i)`.
/// The component sits on the synthetic root [`GameWorld::spawn`] returns.
/// Object-animation tracks and skin lookups resolve a node through this
/// vec; joint clips still sample the skin-order joint index.
#[derive(Debug, Clone)]
pub struct ModelInstance {
    /// Handle the instance was spawned from.
    pub model: ornis_assets::Handle<ornis_assets::Model>,
    /// One entity per model node, in [`Model::nodes`](ornis_assets::Model::nodes) order.
    pub nodes: Vec<Entity>,
}

fn spawn_model_hierarchy<Role: SceneRole>(
    world: &mut GameWorld<Role>,
    handle: ornis_assets::Handle<ornis_assets::Model>,
    model: &ornis_assets::Model,
    local: ornis_core::Transform,
    parent: Option<Entity>,
) -> Entity {
    let (root, skeletal, object) = {
        let store = world
            .engine_mut()
            .world_mut()
            .store_mut()
            .expect("engine always carries a store");
        let root = insert_model_root(store, model, local);
        if let Some(parent) = parent {
            ornis_core::set_parent(store, root, parent)
                .expect("parent was alive before the model was spawned");
        }
        let nodes = insert_model_nodes(store, root, model);
        let primitives = insert_model_primitives(store, &nodes, model);
        store.insert(
            root,
            ModelInstance {
                model: handle,
                nodes: nodes.clone(),
            },
        );
        let spawn = crate::anim_wiring::GltfSpawn {
            entities: primitives,
            node_to_entity: nodes
                .iter()
                .enumerate()
                .map(|(index, entity)| (ornis_assets::NodeIdx(index as u32), *entity))
                .collect(),
            scene_root: Some(root),
        };
        let wiring = crate::anim_wiring::wire_loaded_animation(store, model, &spawn);
        for entity in wiring
            .roots
            .iter()
            .chain(wiring.skel_playlists.iter())
            .chain(wiring.anim_playlists.iter())
        {
            ornis_core::set_parent(store, *entity, root)
                .expect("animation entity stays in the model subtree");
        }
        let skeletal = !wiring.roots.is_empty() || !wiring.skel_playlists.is_empty();
        let object = !wiring.anim_playlists.is_empty();
        (root, skeletal, object)
    };
    if skeletal {
        crate::install_skeletal_animation(world.engine_mut());
    }
    if object {
        crate::install_object_animation(world.engine_mut());
    }
    root
}

fn insert_model_root(
    store: &mut ornis_core::SmartStore,
    model: &ornis_assets::Model,
    local: ornis_core::Transform,
) -> Entity {
    let root = store.create_entity();
    store.insert(root, local);
    store.insert(root, ornis_core::GlobalTransform::from_local(local));
    store.insert(root, ornis_core::Name(model.name.clone()));
    root
}

fn insert_model_nodes(
    store: &mut ornis_core::SmartStore,
    root: Entity,
    model: &ornis_assets::Model,
) -> Vec<Entity> {
    let mut nodes = Vec::with_capacity(model.nodes.len());
    for node in &model.nodes {
        let entity = store.create_entity();
        store.insert(entity, node.local);
        store.insert(entity, ornis_core::GlobalTransform::from_local(node.local));
        if let Some(name) = &node.name {
            store.insert(entity, ornis_core::Name(name.clone()));
        }
        let parent = node
            .parent
            .map(|index| nodes[index.index()])
            .unwrap_or(root);
        ornis_core::set_parent(store, entity, parent).expect("model node parent");
        nodes.push(entity);
    }
    nodes
}

fn insert_model_primitives(
    store: &mut ornis_core::SmartStore,
    nodes: &[Entity],
    model: &ornis_assets::Model,
) -> Vec<Entity> {
    let flat = ornis_assets::scene_from_model(model);
    let mut primitives = Vec::with_capacity(model.primitives.len());
    for (primitive, desc) in model.primitives.iter().zip(flat.entities.iter()) {
        let entity = store.create_entity();
        store.insert(entity, ornis_core::Transform::IDENTITY);
        store.insert(entity, ornis_core::GlobalTransform::IDENTITY);
        store.insert(entity, desc.mesh.clone());
        store.insert(entity, desc.material.clone());
        let parent = nodes[primitive.node.index()];
        ornis_core::set_parent(store, entity, parent).expect("primitive parent");
        primitives.push(entity);
    }
    primitives
}

/// Mutable access to one entity inside a [`GameWorld`].
///
/// [`AnimatorAccess`](ornis_animation::AnimatorAccess) resolves
/// `world.entity_mut(hero).animator()?.play("Walk_Loop")?` against the
/// character root [`GameWorld::spawn`] of a [`ModelSpawn`] returned.
pub struct EntityMut<'a> {
    store: &'a mut ornis_core::SmartStore,
    entity: Entity,
}

impl EntityMut<'_> {
    /// Parents this entity under `parent` and updates both hierarchy caches.
    ///
    /// Same contract as [`GameWorld::set_parent`]: the link goes through
    /// [`set_parent`](ornis_core::set_parent), not a raw [`ChildOf`](ornis_core::ChildOf)
    /// insert.
    ///
    /// # Errors
    ///
    /// [`HierarchyError`](ornis_core::HierarchyError) when either entity is
    /// dead, they are the same entity, or the link would cycle.
    pub fn set_parent(&mut self, parent: Entity) -> Result<(), ornis_core::HierarchyError> {
        ornis_core::set_parent(self.store, self.entity, parent)
    }

    /// Detaches this entity from its parent and drops the matching cache entry.
    pub fn clear_parent(&mut self) {
        ornis_core::clear_parent(self.store, self.entity);
    }
}

impl ornis_animation::AnimatorAccess for EntityMut<'_> {
    fn anim_store_mut(&mut self) -> &mut ornis_core::SmartStore {
        self.store
    }

    fn anim_entity(&self) -> Entity {
        self.entity
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
        assert_eq!(rig.ambient, ambient);
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
        assert_eq!(rig.ambient, Color::linear_rgb(0.1, 0.1, 0.1));
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
        assert_eq!(rig.ambient, Color::linear_rgb(0.1, 0.1, 0.1));
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
        assert_eq!(rig.ambient, Color::WHITE);
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
        assert_eq!(lights.ambient, Color::linear_rgb(0.2, 0.2, 0.2));
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

    fn mesh_entities(world: &GameWorld) -> Vec<Entity> {
        let Some(store) = world.engine().world().store() else {
            return Vec::new();
        };
        let Some(lane) = store.read_lane::<MeshDesc>() else {
            return Vec::new();
        };
        lane.entities.clone()
    }

    /// Scene-first facade: empty world, one character root, explicit light
    /// and camera. The world starts dark, and the animator sits on the
    /// root `spawn` returned.
    #[test]
    fn facade_spawns_asset_with_explicit_light_and_camera() {
        use ornis_animation::{Animator, AnimatorError, try_animator};
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
        let mannequin: ornis_assets::Handle<ornis_assets::Model> =
            world.assets_mut().load(&starter).expect("starter loads");
        let hero = world
            .spawn(ModelSpawn::new(mannequin))
            .expect("starter spawns");
        let meshes = mesh_entities(&world);
        assert!(!meshes.is_empty());
        assert_ne!(hero, meshes[0], "character root is not the first mesh");
        {
            let store = world.engine().world().store().expect("store");
            let anim = store
                .read_lane::<Animator>()
                .expect("animator lane")
                .get(hero)
                .expect("animator on the character root")
                .clone();
            assert!(!anim.targets().is_empty());
            assert!(anim.clip("Walk_Loop").is_some());
        }
        {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            assert!(matches!(
                try_animator(store, meshes[0]),
                Err(AnimatorError::Missing)
            ));
        }

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
        assert_eq!(rig.ambient, Color::linear_rgb(0.1, 0.1, 0.15));
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

        let missing = world
            .assets_mut()
            .load::<ornis_assets::Model>(std::path::Path::new("nope.glb"));
        assert!(missing.is_err(), "missing file rejects");
    }

    /// The same loaded handle spawns more than once. Each spawn returns
    /// its own character root, and mesh entities accumulate. Another world
    /// loads the same file through its own server.
    #[test]
    fn spawn_scene_same_handle_spawns_another_root() {
        use ornis_animation::AnimatorAccess;
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mut world = GameWorld::new();
        let mannequin: ornis_assets::Handle<ornis_assets::Model> =
            world.assets_mut().load(&starter).expect("starter loads");
        let hero = world
            .spawn(ModelSpawn::new(mannequin))
            .expect("starter spawns");
        let meshes = mesh_entities(&world);
        assert!(!meshes.is_empty());
        world
            .entity_mut(hero)
            .animator()
            .expect("animator on the root")
            .play("Walk_Loop")
            .expect("Walk_Loop");

        let mut other = GameWorld::new();
        let other_handle: ornis_assets::Handle<ornis_assets::Model> = other
            .assets_mut()
            .load(&starter)
            .expect("other world loads");
        let other_hero = other
            .spawn(ModelSpawn::new(other_handle))
            .expect("starter spawns");
        assert_eq!(mesh_entities(&other).len(), meshes.len());
        other
            .entity_mut(other_hero)
            .animator()
            .expect("animator on the other root")
            .play("Walk_Loop")
            .expect("Walk_Loop");

        let again = world
            .spawn(ModelSpawn::new(mannequin))
            .expect("starter spawns");
        assert_ne!(again, hero);
        assert_eq!(mesh_entities(&world).len(), meshes.len() * 2);
    }

    /// RON scenes return one empty root plus their mesh entities and no
    /// animator. An unloaded handle rejects without touching the world.
    #[test]
    fn spawn_scene_ron_and_unknown_handle() {
        use ornis_animation::{AnimatorError, try_animator};
        let ron_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/scene.ron");
        let mut world = GameWorld::new();
        let handle: ornis_assets::Handle<Scene> =
            world.assets_mut().load(&ron_path).expect("ron loads");
        let expected = world.assets().get(&handle).expect("loaded").entities.len();
        let root = world.spawn_scene(&handle).expect("ron spawns");
        let meshes = mesh_entities(&world);
        assert_eq!(meshes.len(), expected);
        assert!(!meshes.contains(&root));
        assert!(
            world
                .engine()
                .world()
                .store()
                .expect("store")
                .is_alive(root)
        );
        {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            assert!(matches!(
                try_animator(store, root),
                Err(AnimatorError::Missing)
            ));
        }

        let before = mesh_entities(&world).len();
        assert!(world.assets_mut().unload(&handle));
        assert!(matches!(
            world.spawn_scene(&handle),
            Err(ornis_assets::AssetError::UnknownHandle { .. })
        ));
        assert_eq!(mesh_entities(&world).len(), before);
    }

    /// `Walk_Loop` is chosen by name (not the baked TPose, not the first
    /// moving clip) and joints visibly travel: pose snapshots over 90
    /// frames take more than one distinct value. The animator is the
    /// character root, not the first mesh.
    #[test]
    fn walk_loop_moves_joints_over_time() {
        use ornis_animation::{Animator, AnimatorAccess, AnimatorError, JointPose, SkelPlayer};
        let mut world = GameWorld::new();
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mannequin: ornis_assets::Handle<ornis_assets::Model> =
            world.assets_mut().load(&starter).expect("starter loads");
        let hero = world
            .spawn(ModelSpawn::new(mannequin))
            .expect("starter spawns");
        let meshes = mesh_entities(&world);
        assert_ne!(hero, meshes[0]);

        let walk = {
            let store = world.engine().world().store().expect("store");
            let anim = store
                .read_lane::<Animator>()
                .expect("animator lane")
                .get(hero)
                .expect("animator on the root")
                .clone();
            assert!(anim.clips().len() > 1);
            assert!(!anim.targets().is_empty());
            let names = store.read_lane::<crate::session::Name>().expect("names");
            let first_clip = names.entities.iter().find_map(|entity| {
                let name = &names.get(*entity)?.0;
                anim.clips().get(name).map(|id| id.0)
            });
            let walk = anim.clip("Walk_Loop").expect("Walk_Loop");
            assert_ne!(
                walk.0,
                first_clip.expect("first indexed clip"),
                "Walk_Loop is not the baked first clip"
            );
            walk
        };
        {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            assert!(
                store
                    .read_lane::<SkelPlayer>()
                    .is_none_or(|lane| lane.is_empty()),
                "no skeletal cursor until play"
            );
            assert!(matches!(
                ornis_animation::try_animator(store, meshes[0]),
                Err(AnimatorError::Missing)
            ));
        }
        world
            .entity_mut(hero)
            .animator()
            .expect("animator")
            .play("Walk_Loop")
            .expect("Walk_Loop");

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
        assert!(world.play_animation(entity).is_ok());
        assert!(world.pause_animation(Entity::new(999)).is_err());
        assert!(world.pause_animation(entity).is_ok());
        assert!(world.stop_animation(entity).is_ok());
        assert!(world.play_animation(Entity::new(999)).is_err());

        let store = world.engine().world().store().expect("store");
        let player = *store
            .read_lane::<SkelPlayer>()
            .expect("lane")
            .get(entity)
            .expect("player");
        assert!(!player.playing);
        assert_eq!(player.time, Seconds::ZERO);
    }

    #[test]
    fn set_parent_keeps_the_children_cache_aligned() {
        use ornis_core::{ChildOf, Children};
        let mut world = GameWorld::from_scene(&two_sphere_scene());
        let entities = world.entities().to_vec();
        world.set_parent(entities[1], entities[0]).expect("parent");
        {
            let store = world.engine().world().store().expect("store");
            assert_eq!(
                store
                    .read_lane::<ChildOf>()
                    .expect("links")
                    .get(entities[1])
                    .copied(),
                Some(ChildOf(entities[0]))
            );
            assert_eq!(
                store
                    .read_lane::<Children>()
                    .expect("cache")
                    .get(entities[0])
                    .expect("list")
                    .as_slice(),
                &[entities[1]]
            );
        }
        world.entity_mut(entities[1]).clear_parent();
        let store = world.engine().world().store().expect("store");
        assert!(
            store
                .read_lane::<ChildOf>()
                .is_none_or(|lane| lane.get(entities[1]).is_none())
        );
        assert!(
            store
                .read_lane::<Children>()
                .expect("cache")
                .get(entities[0])
                .is_none_or(|list| list.as_slice().is_empty())
        );
    }

    #[test]
    fn despawn_recursive_drops_the_scene_subtree() {
        let mut world = GameWorld::from_scene(&two_sphere_scene());
        let entities = world.entities().to_vec();
        assert!(entities.len() >= 2);
        world
            .entity_mut(entities[1])
            .set_parent(entities[0])
            .expect("parent");
        world.despawn_recursive(entities[0]);
        assert_eq!(world.entity_count(), entities.len() - 2);
        let store = world.engine().world().store().expect("store");
        assert!(!store.is_alive(entities[0]));
        assert!(!store.is_alive(entities[1]));
    }

    /// Hierarchical model spawn: one synthetic root, node parents from the
    /// model, primitive children, world pose after propagation, and a
    /// `Walk_Loop` animator on that root. Despawn removes the subtree.
    #[test]
    fn spawned_mannequin_keeps_node_parents_and_world_pose() {
        use ornis_animation::AnimatorAccess;
        use ornis_core::{ChildOf, GlobalTransform, Transform, propagate_transforms};
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mut world = GameWorld::new();
        let mannequin: ornis_assets::Handle<ornis_assets::Model> =
            world.assets_mut().load(&starter).expect("starter loads");
        let hero = world
            .spawn(ModelSpawn::new(mannequin))
            .expect("starter spawns");
        {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            propagate_transforms(store);
        }
        let model = world
            .assets()
            .get(&mannequin)
            .expect("model retained")
            .clone();
        let store = world.engine().world().store().expect("store");
        let instance = store
            .read_lane::<ModelInstance>()
            .expect("instance lane")
            .get(hero)
            .expect("instance on the root")
            .clone();
        assert!(
            store
                .read_lane::<ChildOf>()
                .is_none_or(|lane| lane.get(hero).is_none()),
            "synthetic root has no parent"
        );
        assert_eq!(instance.nodes.len(), model.nodes.len());
        let parents = store.read_lane::<ChildOf>().expect("parents");
        for (index, node) in model.nodes.iter().enumerate() {
            let entity = instance.nodes[index];
            let expected = node
                .parent
                .map(|parent| instance.nodes[parent.index()])
                .unwrap_or(hero);
            assert_eq!(parents.get(entity).copied(), Some(ChildOf(expected)));
        }
        let hand = model.node_by_name("hand_r").expect("hand_r");
        let arm = model.node_by_name("lowerarm_r").expect("lowerarm_r");
        assert_eq!(model.nodes[hand.index()].parent, Some(arm));
        assert_eq!(
            parents.get(instance.nodes[hand.index()]).copied(),
            Some(ChildOf(instance.nodes[arm.index()]))
        );
        let globals = store.read_lane::<GlobalTransform>().expect("globals");
        for index in [hand, arm, model.roots[0]] {
            let global = *globals
                .get(instance.nodes[index.index()])
                .expect("node global");
            let expected = model.world_transform(index);
            assert!(
                (global.translation - expected.translation).length() < 1e-3,
                "translation {} vs {}",
                global.translation,
                expected.translation
            );
            assert!(
                (global.scale - expected.scale).length() < 1e-3,
                "scale {:?} vs {:?}",
                global.scale,
                expected.scale
            );
            assert!(global.rotation.get().dot(expected.rotation.get()).abs() > 1.0 - 1e-3);
        }
        let meshes = mesh_entities(&world);
        assert_eq!(meshes.len(), model.primitives.len());
        let locals = store.read_lane::<Transform>().expect("locals");
        for (mesh, primitive) in meshes.iter().zip(model.primitives.iter()) {
            assert_eq!(
                parents.get(*mesh).copied(),
                Some(ChildOf(instance.nodes[primitive.node.index()]))
            );
            assert_eq!(locals.get(*mesh).copied(), Some(Transform::IDENTITY));
        }
        let hand_entity = instance.nodes[hand.index()];
        let arm_entity = instance.nodes[arm.index()];
        let primitive = meshes[0];
        drop(parents);
        drop(globals);
        drop(locals);
        world
            .entity_mut(hero)
            .animator()
            .expect("animator on the root")
            .play("Walk_Loop")
            .expect("Walk_Loop");
        world.despawn_recursive(hero);
        let store = world.engine().world().store().expect("store");
        assert!(!store.is_alive(hero));
        assert!(!store.is_alive(hand_entity));
        assert!(!store.is_alive(arm_entity));
        assert!(!store.is_alive(primitive));
    }

    fn transform_count(world: &GameWorld) -> usize {
        world
            .engine()
            .world()
            .store()
            .and_then(|store| {
                store
                    .read_lane::<ornis_core::Transform>()
                    .map(|lane| lane.len())
            })
            .unwrap_or(0)
    }

    /// An unknown, dangling, or dead-parent spawn is an error. Scene entity
    /// count and the transform lane stay as they were.
    #[test]
    fn spawn_unknown_model_handle_leaves_the_world_untouched() {
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mut world = GameWorld::from_scene(&two_sphere_scene());
        let before = world.entity_count();
        let transforms_before = transform_count(&world);
        let handle: ornis_assets::Handle<ornis_assets::Model> =
            world.assets_mut().load(&starter).expect("starter loads");
        assert!(world.assets_mut().unload(&handle));
        let err = world
            .spawn(ModelSpawn {
                model: handle,
                ..Default::default()
            })
            .expect_err("unknown handle");
        assert_eq!(
            err,
            SpawnModelError::UnknownHandle {
                index: handle.id().index(),
            }
        );
        let dangling = world.spawn(ModelSpawn::default()).expect_err("dangling");
        assert_eq!(dangling, SpawnModelError::UnknownHandle { index: 0 });
        let loaded = world
            .assets_mut()
            .load::<ornis_assets::Model>(&starter)
            .expect("reload");
        let dead = Entity::new(u32::MAX);
        let parent_err = world
            .spawn(ModelSpawn {
                model: loaded,
                parent: Some(dead),
                ..Default::default()
            })
            .expect_err("dead parent");
        assert_eq!(parent_err, SpawnModelError::InvalidParent { parent: dead });
        assert_eq!(world.entity_count(), before);
        assert_eq!(transform_count(&world), transforms_before);
        assert!(
            world
                .engine()
                .world()
                .store()
                .expect("store")
                .read_lane::<ModelInstance>()
                .is_none_or(|lane| lane.is_empty())
        );
    }

    /// Root local TRS is the spawn placement. After propagation the world
    /// pose is the parent pose composed with that local TRS.
    #[test]
    fn model_spawn_applies_position_rotation_scale_and_parent() {
        use ornis_core::{ChildOf, GlobalTransform, Transform, UnitQuat, propagate_transforms};
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mut world = GameWorld::new();
        let mannequin: ornis_assets::Handle<ornis_assets::Model> =
            world.assets_mut().load(&starter).expect("starter loads");
        let parent_local = Transform::from_translation(glam::Vec3::new(10.0, 0.0, 0.0));
        let parent = {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            let parent = store.create_entity();
            store.insert(parent, parent_local);
            store.insert(parent, GlobalTransform::from_local(parent_local));
            parent
        };
        let position = Position::new(1.0, 2.0, 3.0);
        let rotation =
            UnitQuat::normalize(glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2))
                .expect("yaw");
        let scale = glam::Vec3::new(2.0, 3.0, 4.0);
        let hero = world
            .spawn(ModelSpawn {
                model: mannequin,
                position,
                rotation,
                scale,
                parent: Some(parent),
            })
            .expect("placed");
        {
            let store = world.engine().world().store().expect("store");
            let local = store
                .read_lane::<Transform>()
                .expect("locals")
                .get(hero)
                .copied()
                .expect("root local");
            assert_eq!(local.translation, position.get());
            assert_eq!(local.rotation, rotation);
            assert_eq!(local.scale, scale);
            assert_eq!(
                store
                    .read_lane::<ChildOf>()
                    .expect("parents")
                    .get(hero)
                    .copied(),
                Some(ChildOf(parent))
            );
        }
        {
            let store = world.engine_mut().world_mut().store_mut().expect("store");
            propagate_transforms(store);
        }
        let global = world
            .engine()
            .world()
            .store()
            .expect("store")
            .read_lane::<GlobalTransform>()
            .expect("globals")
            .get(hero)
            .copied()
            .expect("root global");
        let expected = GlobalTransform::from_local(parent_local).mul_local(Transform {
            translation: position.get(),
            rotation,
            scale,
        });
        assert!((global.translation - expected.translation).length() < 1e-4);
        assert!((global.scale - expected.scale).length() < 1e-4);
        assert!(global.rotation.get().dot(expected.rotation.get()).abs() > 1.0 - 1e-4);
    }

    /// `load` is the world's asset server. A missing file rejects and a
    /// loaded handle resolves.
    #[test]
    fn load_delegates_to_the_asset_server() {
        let starter = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../assets/starter/ual1_standard.glb");
        let mut world = GameWorld::new();
        let handle = world
            .load::<ornis_assets::Model>(&starter)
            .expect("starter loads");
        assert!(world.assets().get(&handle).is_some());
        let missing = world.load::<ornis_assets::Model>(std::path::Path::new("nope.glb"));
        assert!(missing.is_err());
    }

    /// The window title defaults to `Ornis Engine` and `set_title` stores
    /// the replacement on both roles.
    #[test]
    fn set_title_stores_the_window_title() {
        let mut world = GameWorld::new();
        assert_eq!(world.title(), "Ornis Engine");
        world.set_title("Ornis — Animation Demo");
        assert_eq!(world.title(), "Ornis — Animation Demo");
        let replica = ReplicaGameWorld::new_replica();
        assert_eq!(replica.title(), "Ornis Engine");
    }
}
