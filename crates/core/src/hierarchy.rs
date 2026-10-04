//! Scene hierarchy: parent links, local [`Transform`], and world
//! [`GlobalTransform`].
//!
//! [`ChildOf`] is the parent pointer. [`Children`] is a cache of those
//! links, updated by [`set_parent`], [`clear_parent`], and
//! [`despawn_recursive`], and rebuilt by [`reconcile_children`] (also the
//! first step of [`propagate_transforms`]) so a raw `ChildOf` insert is
//! consistent before the tree walk. [`PropagateTransforms`] writes each
//! entity's world TRS from the root downward. The product is TRS only:
//! a rotated non-uniform parent scale does not produce shear.

use std::collections::{HashMap, HashSet};

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::Engine;
use crate::entity::Entity;
use crate::schedule::{Resources, System, SystemAccess};
use crate::smart_store::SmartStore;
use crate::units::UnitQuat;

/// Parent scale below this is treated as degenerate when converting a
/// world pose back into local translation.
const SCALE_EPSILON: f32 = 1e-8;

/// Upper bound on ancestor walks. A longer chain is treated as a cycle.
const HIERARCHY_WALK_LIMIT: usize = 4096;

/// Display name shared by editor entities and asset/glTF nodes.
///
/// Serde form is the string itself (`"hero"`), not a wrapped object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Name(pub String);

/// Local TRS relative to the parent. On a root this is also the world pose
/// until [`propagate_transforms`] copies it into [`GlobalTransform`].
///
/// Wire form matches the scene descriptor: `translation` and `scale` are
/// `[f32; 3]`, `rotation` is `[x, y, z, w]` and is normalized on load.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "TransformWire", into = "TransformWire")]
pub struct Transform {
    /// Translation relative to the parent, in world units.
    pub translation: Vec3,
    /// Orientation relative to the parent.
    pub rotation: UnitQuat,
    /// Scale relative to the parent, per axis.
    pub scale: Vec3,
}

/// Serde stand-in so the quaternion stays a unit newtype in Rust and an
/// array on the wire.
#[derive(Serialize, Deserialize)]
struct TransformWire {
    translation: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
}

impl From<Transform> for TransformWire {
    fn from(value: Transform) -> Self {
        Self {
            translation: value.translation.to_array(),
            rotation: value.rotation.get().to_array(),
            scale: value.scale.to_array(),
        }
    }
}

impl From<GlobalTransform> for TransformWire {
    fn from(value: GlobalTransform) -> Self {
        Self::from(Transform {
            translation: value.translation,
            rotation: value.rotation,
            scale: value.scale,
        })
    }
}

impl From<TransformWire> for GlobalTransform {
    fn from(value: TransformWire) -> Self {
        Self::from_local(Transform::from(value))
    }
}

impl From<TransformWire> for Transform {
    fn from(value: TransformWire) -> Self {
        Self {
            translation: Vec3::from_array(value.translation),
            rotation: UnitQuat::normalize(Quat::from_array(value.rotation))
                .unwrap_or(UnitQuat::IDENTITY),
            scale: Vec3::from_array(value.scale),
        }
    }
}

impl Transform {
    /// Origin, identity rotation, unit scale.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: UnitQuat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// Translation with identity rotation and unit scale.
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: UnitQuat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    /// Column-major TRS matrix (scale, then rotation, then translation).
    pub fn to_mat4(self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation.get(),
            self.translation,
        )
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// World-space TRS written by [`propagate_transforms`].
///
/// Same fields and wire form as [`Transform`]. Readers (render, physics)
/// use this pose; [`Transform`] stays parent-relative.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(from = "TransformWire", into = "TransformWire")]
pub struct GlobalTransform {
    /// World translation.
    pub translation: Vec3,
    /// World orientation.
    pub rotation: UnitQuat,
    /// World scale, per axis (shear is not represented).
    pub scale: Vec3,
}

impl GlobalTransform {
    /// Origin, identity rotation, unit scale.
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: UnitQuat::IDENTITY,
        scale: Vec3::ONE,
    };

    /// World pose of a root whose local TRS is `local`.
    pub const fn from_local(local: Transform) -> Self {
        Self {
            translation: local.translation,
            rotation: local.rotation,
            scale: local.scale,
        }
    }

    /// `self * local` as TRS: scale multiplies per axis, rotation
    /// composes, translation is rotated and scaled in parent space.
    ///
    /// A non-uniform parent scale combined with a child rotation would
    /// introduce shear in a full matrix product. That shear is dropped;
    /// the result stays a TRS.
    pub fn mul_local(self, local: Transform) -> Self {
        let rotation = UnitQuat::normalize(self.rotation.get() * local.rotation.get())
            .unwrap_or(self.rotation);
        let scale = self.scale * local.scale;
        let translation = self.translation + self.rotation.get() * (self.scale * local.translation);
        Self {
            translation,
            rotation,
            scale,
        }
    }

    /// Local TRS of an entity whose world translation/rotation are
    /// `world_translation` / `world_rotation`, given this parent pose.
    ///
    /// Scale is copied from `previous` (physics does not author scale).
    /// A near-zero parent scale axis keeps that axis of
    /// `previous.translation`.
    pub fn to_local(
        self,
        world_translation: Vec3,
        world_rotation: UnitQuat,
        previous: Transform,
    ) -> Transform {
        let delta = world_translation - self.translation;
        let unrotated = self.rotation.get().inverse() * delta;
        let translation = Vec3::new(
            div_axis(unrotated.x, self.scale.x, previous.translation.x),
            div_axis(unrotated.y, self.scale.y, previous.translation.y),
            div_axis(unrotated.z, self.scale.z, previous.translation.z),
        );
        let rotation = UnitQuat::normalize(self.rotation.get().inverse() * world_rotation.get())
            .unwrap_or(previous.rotation);
        Transform {
            translation,
            rotation,
            scale: previous.scale,
        }
    }

    /// Column-major world matrix.
    pub fn to_mat4(self) -> glam::Mat4 {
        glam::Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation.get(),
            self.translation,
        )
    }
}

impl Default for GlobalTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

fn div_axis(value: f32, scale: f32, fallback: f32) -> f32 {
    if scale.abs() < SCALE_EPSILON {
        fallback
    } else {
        value / scale
    }
}

/// Parent pointer. The parent's [`Children`] cache lists this entity.
///
/// [`set_parent`] and [`clear_parent`] write this link and the cache
/// together. [`SmartStore`] has no insert hook, so
/// [`SmartStore::insert`] of [`ChildOf`] leaves
/// [`Children`] stale until [`reconcile_children`] or
/// [`propagate_transforms`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildOf(pub Entity);

impl ChildOf {
    /// The parent entity.
    pub const fn parent(self) -> Entity {
        self.0
    }
}

/// Ordered children of one entity. A cache of [`ChildOf`] links.
///
/// [`set_parent`], [`clear_parent`], [`despawn_recursive`], and
/// [`reconcile_children`] keep it aligned with [`ChildOf`]. Replacing the
/// vec by hand, or inserting [`ChildOf`] through [`SmartStore::insert`],
/// is repaired on the next reconcile ([`propagate_transforms`] reconciles
/// first). [`SmartStore`] has no insert hook that could do this itself.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Children(pub Vec<Entity>);

impl Children {
    /// Entities in child order.
    pub fn as_slice(&self) -> &[Entity] {
        &self.0
    }
}

/// Why a parent link was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum HierarchyError {
    /// `child` or `parent` is not a live entity.
    #[error("hierarchy entity is not alive")]
    Dead,
    /// An entity cannot parent itself.
    #[error("entity cannot be its own parent")]
    SelfParent,
    /// The new parent is in the child's subtree.
    #[error("parent link would cycle")]
    Cycle,
}

/// Makes `parent` the parent of `child` and updates both [`Children`] caches.
///
/// # Errors
///
/// [`HierarchyError::Dead`] when either handle is not alive,
/// [`HierarchyError::SelfParent`] when they are the same entity, and
/// [`HierarchyError::Cycle`] when `parent` is under `child`.
pub fn set_parent(
    store: &mut SmartStore,
    child: Entity,
    parent: Entity,
) -> Result<(), HierarchyError> {
    if !store.is_alive(child) || !store.is_alive(parent) {
        return Err(HierarchyError::Dead);
    }
    if child == parent {
        return Err(HierarchyError::SelfParent);
    }
    store.register::<ChildOf>();
    store.register::<Children>();
    if would_cycle(store, child, parent) {
        return Err(HierarchyError::Cycle);
    }
    clear_parent_link(store, child);
    store
        .write_lane::<ChildOf>()
        .expect("ChildOf lane is registered")
        .insert(child, ChildOf(parent));
    let mut children = store
        .write_lane::<Children>()
        .expect("Children lane is registered");
    if let Some(list) = children.get_mut(parent) {
        if !list.0.contains(&child) {
            list.0.push(child);
        }
    } else {
        children.insert(parent, Children(vec![child]));
    }
    Ok(())
}

/// Removes `child`'s parent link and the matching [`Children`] entry.
pub fn clear_parent(store: &SmartStore, child: Entity) {
    clear_parent_link(store, child);
}

fn clear_parent_link(store: &SmartStore, child: Entity) {
    let parent = store
        .read_lane::<ChildOf>()
        .and_then(|lane| lane.get(child).copied())
        .map(ChildOf::parent);
    if let Some(mut links) = store.write_lane::<ChildOf>() {
        links.remove(child);
    }
    let Some(parent) = parent else {
        return;
    };
    if let Some(mut children) = store.write_lane::<Children>()
        && let Some(list) = children.get_mut(parent)
    {
        list.0.retain(|entity| *entity != child);
    }
}

fn would_cycle(store: &SmartStore, child: Entity, parent: Entity) -> bool {
    let mut cursor = Some(parent);
    for _ in 0..HIERARCHY_WALK_LIMIT {
        let Some(entity) = cursor else {
            return false;
        };
        if entity == child {
            return true;
        }
        cursor = store
            .read_lane::<ChildOf>()
            .and_then(|lane| lane.get(entity).copied())
            .map(ChildOf::parent);
    }
    true
}

/// Rebuilds every [`Children`] list from the [`ChildOf`] lane.
///
/// Existing child order is kept for links that are still present; new
/// links are appended. Parents with no remaining children lose the cache
/// component. Dead parents are skipped.
pub fn reconcile_children(store: &mut SmartStore) {
    store.register::<ChildOf>();
    store.register::<Children>();
    reconcile_registered(store);
}

/// Destroys `root` and every descendant.
///
/// The root is removed from its parent's [`Children`] first. Descendants
/// come from both the cache and any [`ChildOf`] that still points at a
/// doomed entity, so a stale cache cannot leave a child alive.
pub fn despawn_recursive(store: &SmartStore, root: Entity) {
    if !store.is_alive(root) {
        return;
    }
    clear_parent_link(store, root);
    let mut doomed = Vec::new();
    let mut stack = vec![root];
    let mut seen = HashSet::new();
    while let Some(entity) = stack.pop() {
        if !seen.insert(entity) {
            continue;
        }
        doomed.push(entity);
        let mut next = Vec::new();
        if let Some(children) = store.read_lane::<Children>()
            && let Some(list) = children.get(entity)
        {
            next.extend(list.0.iter().copied());
        }
        if let Some(links) = store.read_lane::<ChildOf>() {
            for (&child, link) in links.entities.iter().zip(links.data.iter()) {
                if link.parent() == entity {
                    next.push(child);
                }
            }
        }
        stack.extend(next);
    }
    for entity in doomed.into_iter().rev() {
        if store.is_alive(entity) {
            store.destroy_entity(entity);
        }
    }
}

/// Registers the hierarchy lanes on `store`.
pub fn register_hierarchy(store: &mut SmartStore) {
    store.register::<Name>();
    store.register::<Transform>();
    store.register::<GlobalTransform>();
    store.register::<ChildOf>();
    store.register::<Children>();
}

/// Walks the tree roots-first and writes [`GlobalTransform`] for every
/// [`Transform`].
///
/// Entities whose parent is missing, dead, or itself has no [`Transform`]
/// are roots (`global = local`). A cycle is broken by treating the
/// remaining entities as roots so the walk finishes.
pub fn propagate_transforms(store: &mut SmartStore) {
    register_hierarchy(store);
    propagate_registered(store);
}

/// Walks a hierarchy whose lanes are already registered.
///
/// Same roots-first [`GlobalTransform`] write as [`propagate_transforms`],
/// without creating lanes. A store that has never held a [`Transform`]
/// returns immediately. Samplers call this after publishing a local pose
/// so children observe it in the same frame.
pub fn propagate_registered(store: &SmartStore) {
    propagate_in(store);
}

fn propagate_in(store: &SmartStore) {
    reconcile_registered(store);
    let Some(locals) = store.read_lane::<Transform>() else {
        return;
    };
    let local_map: HashMap<Entity, Transform> = locals
        .entities
        .iter()
        .zip(locals.data.iter())
        .map(|(&entity, transform)| (entity, *transform))
        .collect();
    drop(locals);
    let parents: HashMap<Entity, Entity> = store
        .read_lane::<ChildOf>()
        .map(|lane| {
            lane.entities
                .iter()
                .zip(lane.data.iter())
                .filter_map(|(&child, link)| {
                    let parent = link.parent();
                    (store.is_alive(parent) && local_map.contains_key(&parent))
                        .then_some((child, parent))
                })
                .collect()
        })
        .unwrap_or_default();
    let mut pending: Vec<Entity> = local_map.keys().copied().collect();
    let mut globals: HashMap<Entity, GlobalTransform> = HashMap::new();
    let mut guard = pending.len().saturating_add(1);
    while !pending.is_empty() && guard > 0 {
        guard -= 1;
        let batch = pending.len();
        let mut deferred = Vec::new();
        for entity in pending {
            let parent = parents.get(&entity).copied();
            let ready = match parent {
                Some(parent) => globals.contains_key(&parent),
                None => true,
            };
            if !ready {
                deferred.push(entity);
                continue;
            }
            let local = local_map[&entity];
            let global = match parent.and_then(|parent| globals.get(&parent)) {
                Some(parent_global) => parent_global.mul_local(local),
                None => GlobalTransform::from_local(local),
            };
            globals.insert(entity, global);
        }
        if deferred.len() == batch {
            for entity in deferred {
                globals.insert(entity, GlobalTransform::from_local(local_map[&entity]));
            }
            break;
        }
        pending = deferred;
    }
    let Some(mut lane) = store.write_lane::<GlobalTransform>() else {
        return;
    };
    for (entity, global) in globals {
        lane.insert(entity, global);
    }
}

/// [`reconcile_children`] without registering lanes (system path).
fn reconcile_registered(store: &SmartStore) {
    if store.read_lane::<ChildOf>().is_none() && store.read_lane::<Children>().is_none() {
        return;
    }
    let mut grouped: HashMap<Entity, Vec<Entity>> = HashMap::new();
    if let Some(links) = store.read_lane::<ChildOf>() {
        for (&child, link) in links.entities.iter().zip(links.data.iter()) {
            let parent = link.parent();
            if parent != child && store.is_alive(parent) && store.is_alive(child) {
                grouped.entry(parent).or_default().push(child);
            }
        }
    }
    let previous: Vec<(Entity, Vec<Entity>)> = store
        .read_lane::<Children>()
        .map(|lane| {
            lane.entities
                .iter()
                .zip(lane.data.iter())
                .map(|(&entity, children)| (entity, children.0.clone()))
                .collect()
        })
        .unwrap_or_default();
    let mut ordered: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (parent, old) in previous {
        let Some(live) = grouped.get(&parent) else {
            continue;
        };
        let mut kept: Vec<Entity> = old
            .into_iter()
            .filter(|child| live.contains(child))
            .collect();
        for child in live {
            if !kept.contains(child) {
                kept.push(*child);
            }
        }
        ordered.insert(parent, kept);
    }
    for (parent, live) in &grouped {
        ordered.entry(*parent).or_insert_with(|| live.clone());
    }
    let Some(mut children) = store.write_lane::<Children>() else {
        return;
    };
    let stale: Vec<Entity> = children
        .entities
        .iter()
        .copied()
        .filter(|entity| !ordered.contains_key(entity))
        .collect();
    for entity in stale {
        children.remove(entity);
    }
    for (parent, list) in ordered {
        children.insert(parent, Children(list));
    }
}

/// Frame-schedule system: parent-relative [`Transform`] → [`GlobalTransform`].
///
/// Install it on the frame schedule before render extraction and on the
/// fixed schedule before physics sync ([`install_transform_propagation`],
/// [`install_fixed_transform_propagation`]).
pub struct PropagateTransforms;

impl System for PropagateTransforms {
    fn name(&self) -> &'static str {
        "propagate_transforms"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<Transform>()
            .reads_lane::<ChildOf>()
            .writes_lane::<Children>()
            .writes_lane::<GlobalTransform>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        propagate_in(store);
    }
}

/// Registers hierarchy lanes and prepends [`PropagateTransforms`] on the
/// frame schedule so it runs before render extraction.
pub fn install_transform_propagation(engine: &mut Engine) {
    if let Some(store) = engine.world_mut().store_mut() {
        register_hierarchy(store);
    }
    if !engine.schedule().mermaid().contains("propagate_transforms") {
        engine.schedule_mut().prepend_system(PropagateTransforms);
    }
}

/// Registers hierarchy lanes and prepends [`PropagateTransforms`] on the
/// fixed schedule so it runs before physics sync.
///
/// Call this after physics systems are prepended: the last prepend is the
/// first system to run.
pub fn install_fixed_transform_propagation(engine: &mut Engine) {
    if let Some(store) = engine.world_mut().store_mut() {
        register_hierarchy(store);
    }
    if !engine
        .fixed_schedule()
        .mermaid()
        .contains("propagate_transforms")
    {
        engine
            .fixed_schedule_mut()
            .prepend_system(PropagateTransforms);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn entities(store: &mut SmartStore, count: usize) -> Vec<Entity> {
        (0..count).map(|_| store.create_entity()).collect()
    }

    #[test]
    fn transform_defaults_to_identity_and_round_trips_json() {
        let transform = Transform::default();
        assert_eq!(transform, Transform::IDENTITY);
        let json = serde_json::to_string(&transform).expect("serialize");
        let back: Transform = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, Transform::IDENTITY);
        let name = Name("hero".to_string());
        assert_eq!(serde_json::to_string(&name).expect("name"), "\"hero\"");
        let tilted = Transform {
            translation: Vec3::new(1.0, 2.0, 3.0),
            rotation: UnitQuat::normalize(Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)).expect("normalizes"),
            scale: Vec3::new(2.0, 2.0, 2.0),
        };
        let restored: Transform =
            serde_json::from_str(&serde_json::to_string(&tilted).expect("json")).expect("back");
        assert_eq!(restored.translation, tilted.translation);
        assert_eq!(restored.scale, tilted.scale);
        assert_eq!(restored.rotation, UnitQuat::IDENTITY);
    }

    #[test]
    fn propagation_composes_three_levels_and_reparent_rewrites_world() {
        let mut store = SmartStore::new();
        let ids = entities(&mut store, 3);
        let (root, mid, leaf) = (ids[0], ids[1], ids[2]);
        store.insert(root, Transform::from_translation(Vec3::new(10.0, 0.0, 0.0)));
        store.insert(mid, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)));
        store.insert(leaf, Transform::from_translation(Vec3::new(1.0, 0.0, 0.0)));
        set_parent(&mut store, mid, root).expect("mid");
        set_parent(&mut store, leaf, mid).expect("leaf");
        propagate_transforms(&mut store);
        let globals = store.read_lane::<GlobalTransform>().expect("globals");
        assert_eq!(
            globals.get(leaf).expect("leaf").translation,
            Vec3::new(12.0, 0.0, 0.0)
        );
        drop(globals);

        let other = store.create_entity();
        store.insert(other, Transform::IDENTITY);
        set_parent(&mut store, leaf, other).expect("reparent");
        let children = store.read_lane::<Children>().expect("children");
        assert!(children.get(mid).is_none_or(|list| !list.0.contains(&leaf)));
        assert!(children.get(other).expect("other").0.contains(&leaf));
        assert_eq!(
            store
                .read_lane::<ChildOf>()
                .expect("links")
                .get(leaf)
                .expect("link")
                .parent(),
            other
        );
        drop(children);
        propagate_transforms(&mut store);
        let globals = store.read_lane::<GlobalTransform>().expect("globals");
        assert_eq!(
            globals.get(leaf).expect("leaf").translation,
            Vec3::new(1.0, 0.0, 0.0)
        );
    }

    #[test]
    fn parent_rotation_and_scale_move_the_child_in_world() {
        let mut store = SmartStore::new();
        let parent = store.create_entity();
        let child = store.create_entity();
        let rotation =
            UnitQuat::normalize(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2)).expect("yaw");
        store.insert(
            parent,
            Transform {
                translation: Vec3::ZERO,
                rotation,
                scale: Vec3::splat(2.0),
            },
        );
        store.insert(child, Transform::from_translation(Vec3::X));
        set_parent(&mut store, child, parent).expect("parent");
        propagate_transforms(&mut store);
        let global = *store
            .read_lane::<GlobalTransform>()
            .expect("globals")
            .get(child)
            .expect("child");
        let expected = rotation.get() * (Vec3::splat(2.0) * Vec3::X);
        assert!((global.translation - expected).length() < 1e-4);
        assert_eq!(global.scale, Vec3::splat(2.0));
    }

    #[test]
    fn children_cache_follows_links_and_raw_inserts_reconcile() {
        let mut store = SmartStore::new();
        let parent = store.create_entity();
        let child = store.create_entity();
        set_parent(&mut store, child, parent).expect("link");
        assert_eq!(
            store
                .read_lane::<Children>()
                .expect("cache")
                .get(parent)
                .expect("list")
                .as_slice(),
            &[child]
        );
        clear_parent(&store, child);
        assert!(
            store
                .read_lane::<Children>()
                .expect("cache")
                .get(parent)
                .is_none_or(|list| list.as_slice().is_empty())
        );
        assert!(
            store
                .read_lane::<ChildOf>()
                .is_none_or(|lane| lane.get(child).is_none())
        );

        store.insert(child, ChildOf(parent));
        reconcile_children(&mut store);
        assert_eq!(
            store
                .read_lane::<Children>()
                .expect("cache")
                .get(parent)
                .expect("rebuilt")
                .as_slice(),
            &[child]
        );
    }

    #[test]
    fn despawn_recursive_removes_the_subtree_and_the_parent_link() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        let mid = store.create_entity();
        let leaf = store.create_entity();
        let keeper = store.create_entity();
        set_parent(&mut store, mid, root).expect("mid");
        set_parent(&mut store, leaf, mid).expect("leaf");
        set_parent(&mut store, root, keeper).expect("root");
        despawn_recursive(&store, root);
        assert!(!store.is_alive(root));
        assert!(!store.is_alive(mid));
        assert!(!store.is_alive(leaf));
        assert!(store.is_alive(keeper));
        let children = store.read_lane::<Children>().expect("cache");
        assert!(
            children
                .get(keeper)
                .is_none_or(|list| list.as_slice().is_empty())
        );
    }

    #[test]
    fn set_parent_rejects_cycles_and_self_links() {
        let mut store = SmartStore::new();
        let parent = store.create_entity();
        let child = store.create_entity();
        set_parent(&mut store, child, parent).expect("link");
        assert_eq!(
            set_parent(&mut store, parent, child),
            Err(HierarchyError::Cycle)
        );
        assert_eq!(
            set_parent(&mut store, parent, parent),
            Err(HierarchyError::SelfParent)
        );
        assert_eq!(
            set_parent(&mut store, parent, Entity::new(99)),
            Err(HierarchyError::Dead)
        );
    }

    #[test]
    fn world_pose_converts_back_to_local_under_a_parent() {
        let parent = GlobalTransform::from_local(Transform {
            translation: Vec3::new(10.0, 0.0, 0.0),
            rotation: UnitQuat::IDENTITY,
            scale: Vec3::splat(2.0),
        });
        let previous = Transform::from_translation(Vec3::ZERO);
        let local = parent.to_local(Vec3::new(12.0, 0.0, 0.0), UnitQuat::IDENTITY, previous);
        assert_eq!(local.translation, Vec3::X);
        assert_eq!(local.scale, Vec3::ONE);
    }
}
