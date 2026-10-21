//! Pure `ornis-gltf` mirror → `ornis-animation` converters, plus the
//! shared spawn/wire routines built on them: [`spawn_gltf_world`](crate::anim_wiring::spawn_gltf_world)
//! (mesh entities from a loaded scene) and [`wire_loaded_animation`](crate::anim_wiring::wire_loaded_animation)
//! (skeleton roots, named clip playlists, a character-root [`Animator`](ornis_animation::Animator),
//! paused object players, skinned meshes).
//!
//! Field-for-field mapping with no I/O beyond the caller's store: joint
//! beyond the output clips: joint indices pass through
//! [`JointId::from_raw`](ornis_animation::JointId), rotations arrive as
//! `(x, y, z, w)` unit quaternions via [`Quat::from_xyzw`](glam::Quat), and
//! object tracks resolve through the caller-supplied node→entity map
//! (unmapped tracks are dropped and counted, never stubbed).

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    AnimClip, AnimFade, AnimPlayer, AnimTrack, Animator, AnimatorError, ClipId, JointId, JointMask,
    JointPose, JointTrack, JointTrs, Key, KeyTrack, SkelClip, SkelFade, SkelLayer, SkelPlayer,
    Skeleton, SkinnedMesh, try_animator,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::{Entity, Seconds, SmartStore};
use ornis_gltf::{
    LoadedAnimClip, LoadedInterpolation, LoadedJointTrack, LoadedKeyTrack, LoadedMesh,
    LoadedSkelClip, LoadedSkin, Model, NodeIdx,
};

use super::session::Name;

/// Converts a skeletal mirror clip to a live [`SkelClip`].
///
/// Duration and joint indices copy verbatim; every channel maps key-for-key
/// with interpolation preserved. Rotation arrays are `(x, y, z, w)` unit
/// quaternions on the mirror side (normalized at import) and convert with
/// [`Quat::from_xyzw`] exactly — no silent renormalization here.
/// [`LoadedJointTrack::node`](ornis_gltf::LoadedJointTrack::node) resolves
/// through `node_to_entity` onto [`JointTrack::node`](ornis_animation::JointTrack::node)
/// so sampling can write that node's local transform. A missing entry
/// leaves `node` empty; the joint index is unchanged.
pub fn skel_clip_from_loaded(
    clip: &LoadedSkelClip,
    node_to_entity: &HashMap<NodeIdx, Entity>,
) -> SkelClip {
    SkelClip {
        name: clip.name.clone(),
        duration: clip.duration,
        tracks: clip
            .tracks
            .iter()
            .map(|track| joint_track_from_loaded(track, node_to_entity))
            .collect(),
    }
}

/// Converts an object mirror clip to a live [`AnimClip`], resolving
/// [`NodeIdx`] through `node_to_entity` to world entities.
///
/// Returns the clip plus the count of tracks dropped for lack of a mapping.
/// The clip name comes from the caller (the mirror name is not consumed), and
/// `duration`/`looping` copy the mirror verbatim. Mesh-less parents stay in
/// the clip; this flat spawn drops a track only when `node_to_entity` has
/// no entity for that node.
pub fn anim_clip_from_loaded(
    name: String,
    clip: &LoadedAnimClip,
    node_to_entity: &HashMap<NodeIdx, Entity>,
) -> (AnimClip, usize) {
    let mut dropped = 0usize;
    let mut tracks = Vec::with_capacity(clip.tracks.len());
    for track in &clip.tracks {
        let Some(&entity) = node_to_entity.get(&track.node) else {
            dropped += 1;
            continue;
        };
        tracks.push(AnimTrack {
            entity,
            rest: joint_trs_from_loaded(&track.rest),
            translation: vec3_track_from_loaded(&track.translation),
            rotation: quat_track_from_loaded(&track.rotation),
            scale: vec3_track_from_loaded(&track.scale),
        });
    }
    (
        AnimClip {
            name,
            duration: clip.duration,
            looping: clip.looping,
            tracks,
        },
        dropped,
    )
}

/// Converts an authored mirror rest transform to a live [`JointTrs`].
///
/// Plain arrays in, `glam` values out: translation and scale verbatim,
/// rotation `(x, y, z, w)` as stored.
pub fn joint_trs_from_loaded(rest: &ornis_gltf::LoadedJointRest) -> JointTrs {
    JointTrs {
        translation: Vec3::from_array(rest.translation),
        rotation: Quat::from_array(rest.rotation),
        scale: Vec3::from_array(rest.scale),
    }
}

/// Converts a mirror skin to a live [`Skeleton`].
///
/// Parents map `-1` → root (`None`) via [`JointId::from_raw_opt`]; bind
/// inverses are glTF column-major (`m[col][row]`) and convert with
/// [`Mat4::from_cols_array_2d`]; names copy verbatim (diagnostics only).
pub fn skeleton_from_loaded(skin: &LoadedSkin) -> Skeleton {
    let rest: Vec<JointTrs> = skin.rest.iter().map(joint_trs_from_loaded).collect();
    Skeleton::new(
        skin.parents
            .iter()
            .map(|&raw| JointId::from_raw_opt(raw))
            .collect(),
        skin.inverse_bind
            .iter()
            .map(Mat4::from_cols_array_2d)
            .collect(),
        skin.joint_names.clone(),
    )
    .with_rest(rest)
}

/// Builds a live [`SkinnedMesh`] from mirror bind data, or [`None`] when the
/// mesh carries no skin influences.
///
/// `None` fires when either `joints` or `weights` is absent (unskinned
/// primitive). Normals/uvs fall back to the mirror `resolved_*` rebuilds
/// (recomputed normals, box-projected uvs) so a skinned primitive without
/// source attributes still skins honestly.
pub fn skinned_mesh_from_loaded(skeleton: Entity, mesh: &LoadedMesh) -> Option<SkinnedMesh> {
    let (Some(joints), Some(weights)) = (mesh.joints.clone(), mesh.weights.clone()) else {
        return None;
    };
    Some(SkinnedMesh::new(
        skeleton,
        joints,
        weights,
        mesh.positions.clone(),
        mesh.resolved_normals(),
        mesh.resolved_uvs(),
        mesh.indices.clone(),
    ))
}

/// Maps one mirror joint track (joint index, hierarchy node, three channels).
fn joint_track_from_loaded(
    track: &LoadedJointTrack,
    node_to_entity: &HashMap<NodeIdx, Entity>,
) -> JointTrack {
    JointTrack {
        joint: JointId::from_raw(track.joint),
        node: node_to_entity.get(&track.node).copied(),
        translation: vec3_track_from_loaded(&track.translation),
        rotation: quat_track_from_loaded(&track.rotation),
        scale: vec3_track_from_loaded(&track.scale),
    }
}

/// Maps a `[f32; 3]` mirror channel to a [`Vec3`] track.
fn vec3_track_from_loaded(track: &LoadedKeyTrack<[f32; 3]>) -> KeyTrack<Vec3> {
    let keys = track
        .keys
        .iter()
        .map(|key| Key {
            time: key.time,
            value: Vec3::from_array(key.value),
        })
        .collect();
    with_interpolation(keys, &track.interpolation, Vec3::from_array)
}

/// Maps a `(x, y, z, w)` mirror channel to a [`Quat`] track, exactly.
fn quat_track_from_loaded(track: &LoadedKeyTrack<[f32; 4]>) -> KeyTrack<Quat> {
    let keys = track
        .keys
        .iter()
        .map(|key| Key {
            time: key.time,
            value: Quat::from_xyzw(key.value[0], key.value[1], key.value[2], key.value[3]),
        })
        .collect();
    with_interpolation(keys, &track.interpolation, |value| {
        Quat::from_xyzw(value[0], value[1], value[2], value[3])
    })
}

/// Wraps converted keys with the mirror interpolation (linear/stepped/cubic),
/// mapping cubic tangents with the same value conversion as the keys.
fn with_interpolation<T, M: Copy>(
    keys: Vec<Key<T>>,
    interpolation: &LoadedInterpolation<M>,
    convert: impl Fn(M) -> T,
) -> KeyTrack<T> {
    match interpolation {
        LoadedInterpolation::Linear => KeyTrack::linear(keys),
        LoadedInterpolation::Step => KeyTrack::stepped(keys),
        LoadedInterpolation::Cubic {
            in_tangents,
            out_tangents,
        } => KeyTrack::cubic(
            keys,
            in_tangents
                .iter()
                .map(|tangent| convert(*tangent))
                .collect(),
            out_tangents
                .iter()
                .map(|tangent| convert(*tangent))
                .collect(),
        ),
    }
}

/// World entities spawned from one [`Model`]: mesh entities in primitive
/// order plus the node → world map animation wiring resolves through.
pub struct GltfSpawn {
    /// Mesh entities, 1:1 with [`Model::primitives`], in primitive order.
    pub entities: Vec<Entity>,
    /// [`NodeIdx`] → world entity.
    ///
    /// [`spawn_gltf_world`] records the first primitive of a node and omits
    /// mesh-less nodes, so object tracks on those nodes are dropped. The
    /// hierarchical [`GameWorld::spawn`](crate::GameWorld::spawn) of a
    /// [`ModelSpawn`](crate::ModelSpawn) fills this map from
    /// [`ModelInstance`](crate::ModelInstance) (every node, parent before
    /// child) so those tracks bind to the node entity.
    pub node_to_entity: HashMap<NodeIdx, Entity>,
    /// Character root that receives [`Animator`].
    ///
    /// `None` (what [`spawn_gltf_world`] leaves) tells
    /// [`wire_loaded_animation`] to fall back to the first mesh entity.
    /// Set this before wiring when the real scene root is a different
    /// entity — that is the hook for hierarchical [`ModelSpawn`](crate::ModelSpawn) spawn.
    /// The entity must already exist.
    pub scene_root: Option<Entity>,
}

/// Entities created by [`wire_loaded_animation`] that outlive mesh
/// entities: skeleton roots and clip playlists (for `alive` bookkeeping
/// and version bumps at the call site).
pub struct AnimationWiring {
    /// One skeleton root per skin, in skin order.
    pub roots: Vec<Entity>,
    /// One playlist entity per skeletal clip, in clip order.
    pub skel_playlists: Vec<Entity>,
    /// One playlist entity per object clip, in clip order.
    pub anim_playlists: Vec<Entity>,
    /// Skeletal clip name → playlist. The first name wins on duplicates.
    pub clips: HashMap<String, ClipId>,
}

impl AnimationWiring {
    /// Entities the call site must track (roots + playlists).
    pub fn added(&self) -> usize {
        self.roots.len() + self.skel_playlists.len() + self.anim_playlists.len()
    }

    /// Inserts an [`Animator`] on `scene_root` for this wiring's skeletal clips.
    ///
    /// Hierarchical model spawn passes the synthetic character root.
    /// Skeleton roots are not given a [`SkelPlayer`] until
    /// [`AnimatorMut::play`](ornis_animation::AnimatorMut::play). Calling
    /// this again on the same entity replaces the component; it does not
    /// clear an animator previously attached to a different entity.
    pub fn attach_animator(&self, store: &mut SmartStore, scene_root: Entity) {
        attach_animator(store, scene_root, self.clips.clone(), self.roots.clone());
    }
}

/// Inserts an [`Animator`] on `scene_root` that indexes `clips` and drives
/// `targets` (skeleton roots).
///
/// No [`SkelPlayer`] is created. Targets stay empty until a named `play`.
pub fn attach_animator(
    store: &mut SmartStore,
    scene_root: Entity,
    clips: HashMap<String, ClipId>,
    targets: Vec<Entity>,
) {
    store.register::<Animator>();
    store.insert(scene_root, Animator::new(clips, targets));
}

/// Spawns the mesh entities of a loaded glTF [`Model`]: `TransformDesc` +
/// `MeshDesc::Custom` + `MaterialDesc` per primitive, at the node's world
/// TRS, no physics bodies (pure visual spawn — the solver never sees these
/// entities).
///
/// This is the flat layout. One entity per node with local [`Transform`](ornis_core::Transform)
/// and [`ornis_core::ChildOf`] is the Core track's spawn, not this function.
///
/// Lanes are registered idempotently; entities follow primitive order so
/// the caller can zip them back against [`Model::primitives`].
pub fn spawn_gltf_world(store: &mut SmartStore, model: &Model) -> GltfSpawn {
    store.register::<TransformDesc>();
    store.register::<MeshDesc>();
    store.register::<MaterialDesc>();
    let scene = ornis_assets::scene_from_model(model);
    let mut entities = Vec::with_capacity(scene.entities.len());
    let mut node_to_entity = HashMap::new();
    for (desc, primitive) in scene.entities.iter().zip(model.primitives.iter()) {
        let entity = store.create_entity();
        store.insert(entity, desc.transform.clone());
        crate::insert_flat_pose(store, entity, &desc.transform);
        store.insert(entity, desc.mesh.clone());
        store.insert(entity, desc.material.clone());
        node_to_entity.entry(primitive.node).or_insert(entity);
        entities.push(entity);
    }
    GltfSpawn {
        entities,
        node_to_entity,
        scene_root: None,
    }
}

/// Sets `playing` on the entity's player (skeletal first, then
/// object). Returns whether a player was found. Players are fellow
/// components, not commands: the engine never touches them, only hosts do.
pub fn set_playing(store: &SmartStore, entity: Entity, playing: bool) -> bool {
    if let Some(mut lane) = store.write_lane::<SkelPlayer>()
        && let Some(player) = lane.get_mut(entity)
    {
        player.playing = playing;
        return true;
    }
    if let Some(mut lane) = store.write_lane::<AnimPlayer>()
        && let Some(player) = lane.get_mut(entity)
    {
        player.playing = playing;
        return true;
    }
    false
}

/// Rewinds the entity's player clock to zero (keeps the playing state).
/// Returns whether a player was found.
pub fn rewind_player(store: &SmartStore, entity: Entity) -> bool {
    if let Some(mut lane) = store.write_lane::<SkelPlayer>()
        && let Some(player) = lane.get_mut(entity)
    {
        player.time = Seconds::ZERO;
        return true;
    }
    if let Some(mut lane) = store.write_lane::<AnimPlayer>()
        && let Some(player) = lane.get_mut(entity)
    {
        player.time = 0.0;
        return true;
    }
    false
}

/// Why named playback ([`play_crossfade`]) could not start.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PlaybackError {
    /// The entity has no [`Animator`] catalog and no [`AnimPlayer`] cursor.
    /// The [`GameWorld`](crate::game_world::GameWorld) playback entries
    /// also report it when the world carries no engine store — an
    /// invariant that never happens in a running app.
    #[error("entity has no animation player")]
    Missing,
    /// No indexed clip matches the requested name.
    #[error("unknown clip `{name}`")]
    UnknownClip {
        /// Requested clip name.
        name: String,
    },
}

/// Starts `name` on `entity`, fading from whatever plays there now.
///
/// The host-facing driver over both playback topologies
/// (`docs/animation-design.md` §5 «Дальше», п. 10): a character root
/// carrying an [`Animator`] catalog starts the named skeletal clip on
/// every target through
/// [`AnimatorMut::play_crossfade`](ornis_animation::AnimatorMut::play_crossfade)
/// — one fade implementation — and an object entity with an [`AnimPlayer`]
/// cursor receives a typed [`AnimFade`]
/// crossfade to the named clip ([`fade_to_clip`]). A non-positive
/// `duration` cuts without a fade, like
/// [`AnimatorMut::play`](ornis_animation::AnimatorMut::play), and so
/// does a start over a fade still in flight — one fade per entity.
///
/// Object names resolve through the playlist's [`Name`] — what
/// [`wire_loaded_animation`] writes from the glTF animation name. Among
/// same-named playlists the earliest-created one that tracks `entity`
/// wins (the "first name wins" rule of the skeletal catalog).
///
/// # Errors
///
/// [`PlaybackError::Missing`] when `entity` carries neither catalog nor
/// cursor (hand-built skeletal cursors without an [`Animator`] start
/// through [`set_playing`] instead). [`PlaybackError::UnknownClip`] when
/// `name` is not indexed.
pub fn play_crossfade(
    store: &mut SmartStore,
    entity: Entity,
    name: &str,
    duration: Seconds,
) -> Result<(), PlaybackError> {
    if let Ok(mut animator) = try_animator(store, entity) {
        return animator
            .play_crossfade(name, duration)
            .map_err(|err| match err {
                AnimatorError::UnknownClip { name } => PlaybackError::UnknownClip { name },
                AnimatorError::Missing => PlaybackError::Missing,
            });
    }
    if store
        .read_lane::<AnimPlayer>()
        .is_none_or(|players| players.get(entity).is_none())
    {
        return Err(PlaybackError::Missing);
    }
    let clip =
        object_clip_named(store, entity, name).ok_or_else(|| PlaybackError::UnknownClip {
            name: name.to_owned(),
        })?;
    fade_to_clip(store, entity, clip, duration)
}

/// Object playlist holding an [`AnimClip`] that tracks `entity`, indexed
/// by the playlist [`Name`]. The earliest-created match wins.
fn object_clip_named(store: &SmartStore, entity: Entity, name: &str) -> Option<ClipId> {
    let names = store.read_lane::<Name>()?;
    let clips = store.read_cold_lane::<AnimClip>()?;
    names
        .entities
        .iter()
        .zip(&names.data)
        .filter(|(_, label)| label.0 == name)
        .map(|(playlist, _)| *playlist)
        .filter(|playlist| {
            clips
                .get(*playlist)
                .is_some_and(|clip| clip.tracks.iter().any(|track| track.entity == entity))
        })
        .min_by_key(|playlist| playlist.id())
        .map(ClipId)
}

/// Crossfades `entity`'s object cursor to `clip` — the typed [`AnimFade`]
/// input, so hosts never install the lane component themselves.
///
/// Snapshots the current [`AnimPlayer`] into [`AnimFade::from`] (its
/// clock keeps advancing until the fade retires) and installs a fresh
/// cursor on `clip` from time zero — `speed` and `weight` reset to `1.0`,
/// the
/// [`AnimatorMut::play_crossfade`](ornis_animation::AnimatorMut::play_crossfade)
/// semantics for the object topology. A non-positive `duration`, a paused
/// cursor, or a start while a fade is still in flight cuts without one and
/// drops any [`AnimFade`] already on the entity: one fade per entity, and
/// it only starts from a settled playing cursor.
///
/// The caller owns `clip`'s validity — [`play_crossfade`] resolves names
/// through the playlist [`Name`] — and nothing here checks that `clip`
/// tracks `entity`.
///
/// # Errors
///
/// [`PlaybackError::Missing`] when `entity` has no [`AnimPlayer`].
pub fn fade_to_clip(
    store: &mut SmartStore,
    entity: Entity,
    clip: ClipId,
    duration: Seconds,
) -> Result<(), PlaybackError> {
    let outgoing = store
        .read_lane::<AnimPlayer>()
        .and_then(|players| players.get(entity).copied())
        .ok_or(PlaybackError::Missing)?;
    store.register::<AnimPlayer>();
    store.register::<AnimFade>();
    let fading = fade_in_flight(store, entity);
    if outgoing.playing && duration.get() > 0.0 && !fading {
        store.insert(
            entity,
            AnimFade {
                from: outgoing,
                duration,
                elapsed: Seconds::ZERO,
            },
        );
    } else if let Some(mut fades) = store.write_lane::<AnimFade>() {
        fades.remove(entity);
    }
    store.insert(
        entity,
        AnimPlayer {
            clip,
            time: 0.0,
            speed: 1.0,
            weight: 1.0,
            playing: true,
        },
    );
    Ok(())
}

/// Whether `entity` already carries an [`AnimFade`] in flight.
fn fade_in_flight(store: &SmartStore, entity: Entity) -> bool {
    store
        .read_lane::<AnimFade>()
        .is_some_and(|fades| fades.get(entity).is_some())
}

/// Installs the overlay `layer` on a skeleton root — the typed
/// [`SkelLayer`] input, so hosts never install the lane component
/// themselves. At most one layer per root: `layer` replaces whatever
/// sits there.
///
/// Returns whether `root` carries a [`Skeleton`] (the layer's anchor);
/// nothing is written without one.
// qual:api — typed `SkelLayer` input for hosts; in-tree callers are tests
// until the layer stack (step 3 of the gate plan) lands.
pub fn set_skel_layer(store: &mut SmartStore, root: Entity, layer: SkelLayer) -> bool {
    if store
        .read_lane::<Skeleton>()
        .is_none_or(|skeletons| skeletons.get(root).is_none())
    {
        return false;
    }
    store.register::<SkelLayer>();
    store.insert(root, layer);
    true
}

/// Wires animation from a loaded glTF scene into an already-spawned
/// world: skeleton roots (`Skeleton` + [`JointPose`]), cold clip
/// playlists named with the glTF animation name, an [`Animator`] on the
/// character root when any skeletal clip exists, and [`SkinnedMesh`] lanes
/// on skinned mesh entities.
///
/// Skeletal players are not created. The animator map is clip name →
/// playlist, and playback starts only at
/// [`AnimatorMut::play`](ornis_animation::AnimatorMut::play). Object
/// players stay paused on the entities their tracks name (those tracks
/// are the target, not a guessed clip); hosts start them with
/// [`set_playing`]. [`GltfSpawn::scene_root`], when set, receives the
/// animator; otherwise the first mesh entity does, which is the stand-in
/// until hierarchical model spawn supplies the synthetic character root.
///
/// `spawn.entities` is one entity per primitive, in primitive order (the
/// flat [`spawn_gltf_world`] list, or the primitive children of a
/// hierarchical spawn). `spawn.node_to_entity` resolves [`NodeIdx`] to the
/// entity object tracks write: a mesh entity for the flat spawn, the node
/// entity from [`ModelInstance`](crate::ModelInstance) for the hierarchy.
/// Joint clips still sample [`LoadedJointTrack::joint`](ornis_gltf::LoadedJointTrack::joint);
/// [`LoadedJointTrack::node`](ornis_gltf::LoadedJointTrack::node) is the
/// hierarchy address (mapped onto the live track, not a second joint index)
/// so sampling writes that node's local transform. Sources without
/// skins or clips wire nothing.
/// Lanes are registered idempotently.
pub fn wire_loaded_animation(
    store: &mut SmartStore,
    loaded: &Model,
    spawn: &GltfSpawn,
) -> AnimationWiring {
    use ornis_animation::{AnimClip, SkelClip};
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register::<SkinnedMesh>();
    store.register::<SkelPlayer>();
    store.register::<SkelLayer>();
    store.register::<SkelFade>();
    store.register_cold::<SkelClip>();
    store.register_cold::<JointMask>();
    store.register::<AnimPlayer>();
    store.register::<AnimFade>();
    store.register_cold::<AnimClip>();

    let mut wiring = AnimationWiring {
        roots: Vec::with_capacity(loaded.skins.len()),
        skel_playlists: Vec::with_capacity(loaded.skel_clips.len()),
        anim_playlists: Vec::with_capacity(loaded.anim_clips.len()),
        clips: HashMap::new(),
    };
    if loaded.skins.is_empty() && loaded.skel_clips.is_empty() && loaded.anim_clips.is_empty() {
        return wiring;
    }
    // Skeleton roots, one per skin.
    for (index, skin) in loaded.skins.iter().enumerate() {
        let skeleton = skeleton_from_loaded(skin);
        let joints = skeleton.joint_count();
        let root = store.create_entity();
        store.insert(root, Name(format!("skeleton_{index}")));
        store.insert(root, skeleton);
        store.insert(root, JointPose::identity(joints));
        wiring.roots.push(root);
    }
    // Skeletal playlists, one per clip and shared across roots. No player
    // is inserted: the character-root [`Animator`] starts a clip by name.
    // An empty glTF name falls back to `skel_clip_{index}` so `play` has a key.
    for (index, clip) in loaded.skel_clips.iter().enumerate() {
        let name = if clip.name.is_empty() {
            format!("skel_clip_{index}")
        } else {
            clip.name.clone()
        };
        let live = skel_clip_from_loaded(clip, &spawn.node_to_entity);
        let playlist = store.create_entity();
        store.insert(playlist, Name(name.clone()));
        store.insert_cold(playlist, live);
        wiring.skel_playlists.push(playlist);
        wiring.clips.entry(name).or_insert(ClipId(playlist));
    }
    // Skinned meshes: `SkinnedMesh.skeleton` points at the skin's root.
    // The skin index lives on the node. `None` (unskinned primitive) keeps
    // the regular `Custom` mesh.
    for (primitive, entity) in loaded.primitives.iter().zip(spawn.entities.iter()) {
        let Some(skin) = loaded
            .nodes
            .get(primitive.node.index())
            .and_then(|node| node.skin)
        else {
            continue;
        };
        let Some(&root) = wiring.roots.get(skin) else {
            continue;
        };
        let Some(skinned) = skinned_mesh_from_loaded(root, &primitive.mesh) else {
            continue;
        };
        store.insert(*entity, skinned);
    }
    // Object playlists, one per clip; every mapped track entity gets a
    // paused player (last clip wins on shared entities — one player
    // lane per entity). The track names the target, so this is not a
    // guessed clip; hosts start it with [`set_playing`].
    for clip in &loaded.anim_clips {
        let (live, dropped) = anim_clip_from_loaded(clip.name.clone(), clip, &spawn.node_to_entity);
        // Dropped (unmapped) tracks stay dropped: the count is covered
        // by converter unit tests, nothing is stored at runtime.
        let _ = dropped;
        let targets: Vec<Entity> = live.tracks.iter().map(|track| track.entity).collect();
        let playlist = store.create_entity();
        store.insert(playlist, Name(live.name.clone()));
        store.insert_cold(playlist, live);
        wiring.anim_playlists.push(playlist);
        for target in &targets {
            store.insert(
                *target,
                AnimPlayer {
                    clip: ClipId(playlist),
                    time: 0.0,
                    speed: 1.0,
                    weight: 1.0,
                    playing: false,
                },
            );
        }
    }
    if !wiring.clips.is_empty()
        && let Some(root) = spawn.scene_root.or_else(|| spawn.entities.first().copied())
    {
        wiring.attach_animator(store, root);
    }
    wiring
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_animation::{AnimatorError, Interpolation, try_animator};
    use ornis_core::{Clamped01, Seconds};
    use ornis_gltf::{LoadedAnimTrack, LoadedKey, LoadedSkelClip, LoadedSkin};

    fn bare_mesh() -> ornis_gltf::LoadedMesh {
        ornis_gltf::LoadedMesh {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0],
            normals: None,
            uvs: None,
            joints: None,
            weights: None,
        }
    }

    fn bare_material() -> ornis_gltf::LoadedMaterial {
        ornis_gltf::LoadedMaterial {
            base_color: [1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emission: [0.0, 0.0, 0.0],
            textures: Default::default(),
        }
    }

    /// One primitive on `node`, identity local pose.
    fn primitive_on(node: NodeIdx) -> ornis_gltf::ModelPrimitive {
        ornis_gltf::ModelPrimitive {
            node,
            mesh: bare_mesh(),
            material: bare_material(),
        }
    }

    /// Spawn maps loader-local node handles to world entities, 1:1 and in
    /// load order, with placement components inserted.
    #[test]
    fn spawn_maps_nodes_to_world_entities() {
        use ornis_assets::scene::TransformDesc;
        let loaded = ornis_gltf::Model {
            name: "two".to_string(),
            nodes: vec![
                ornis_gltf::ModelNode {
                    name: Some("part_0".into()),
                    parent: None,
                    local: ornis_core::Transform::IDENTITY,
                    primitives: vec![0],
                    skin: None,
                },
                ornis_gltf::ModelNode {
                    name: Some("part_1".into()),
                    parent: None,
                    local: ornis_core::Transform::IDENTITY,
                    primitives: vec![1],
                    skin: None,
                },
            ],
            roots: vec![NodeIdx(0), NodeIdx(1)],
            primitives: vec![primitive_on(NodeIdx(0)), primitive_on(NodeIdx(1))],
            skins: Vec::new(),
            skel_clips: Vec::new(),
            anim_clips: Vec::new(),
            stats: ornis_gltf::ImportStats::default(),
        };
        let mut store = SmartStore::new();
        let spawn = spawn_gltf_world(&mut store, &loaded);
        assert_eq!(spawn.entities.len(), 2);
        assert_eq!(spawn.node_to_entity[&NodeIdx(0)], spawn.entities[0]);
        assert_eq!(spawn.node_to_entity[&NodeIdx(1)], spawn.entities[1]);
        for entity in &spawn.entities {
            let lane = store.read_lane::<TransformDesc>().unwrap();
            assert!(lane.get(*entity).is_some());
        }
    }

    /// Mirror joint track with one linear key per channel.
    fn joint_fixture() -> LoadedJointTrack {
        LoadedJointTrack {
            joint: 3,
            node: NodeIdx(3),
            translation: LoadedKeyTrack::linear(vec![LoadedKey {
                time: 0.5,
                value: [1.0, 2.0, 3.0],
            }]),
            rotation: LoadedKeyTrack::linear(vec![LoadedKey {
                time: 0.5,
                value: [0.0, 0.0, 0.0, 1.0],
            }]),
            scale: LoadedKeyTrack::linear(vec![LoadedKey {
                time: 0.5,
                value: [1.0, 1.0, 1.0],
            }]),
        }
    }

    #[test]
    fn skel_clip_converts_joint_and_channels_verbatim() {
        let clip = LoadedSkelClip {
            name: String::new(),
            duration: 2.5,
            tracks: vec![joint_fixture()],
        };
        let live = skel_clip_from_loaded(&clip, &HashMap::new());
        assert_eq!(live.duration, 2.5);
        assert_eq!(live.tracks.len(), 1);
        let track = &live.tracks[0];
        assert_eq!(track.joint, JointId::from_raw(3));
        assert!(track.node.is_none(), "unmapped joint node stays unbound");
        assert_eq!(
            track.translation.keys,
            vec![Key {
                time: 0.5,
                value: Vec3::new(1.0, 2.0, 3.0)
            }]
        );
        assert_eq!(track.translation.interpolation, Interpolation::Linear);
        assert_eq!(track.rotation.keys[0].value, Quat::IDENTITY);
        assert_eq!(track.scale.keys[0].value, Vec3::ONE);
    }

    #[test]
    fn skel_clip_maps_joint_node_through_node_to_entity() {
        let clip = LoadedSkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![joint_fixture()],
        };
        let store = SmartStore::new();
        let entity = store.create_entity();
        let mut map = HashMap::new();
        map.insert(NodeIdx(3), entity);
        let live = skel_clip_from_loaded(&clip, &map);
        assert_eq!(live.tracks[0].node, Some(entity));
        assert_eq!(live.tracks[0].joint, JointId::from_raw(3));
    }

    #[test]
    fn stepped_interpolation_survives_on_all_channels() {
        let clip = LoadedSkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![LoadedJointTrack {
                joint: 0,
                node: NodeIdx(0),
                translation: LoadedKeyTrack::stepped(vec![LoadedKey {
                    time: 0.0,
                    value: [4.0, 5.0, 6.0],
                }]),
                rotation: LoadedKeyTrack::stepped(vec![LoadedKey {
                    time: 0.0,
                    value: [0.0, 0.0, 0.0, 1.0],
                }]),
                scale: LoadedKeyTrack::stepped(vec![LoadedKey {
                    time: 0.0,
                    value: [2.0, 2.0, 2.0],
                }]),
            }],
        };
        let live = skel_clip_from_loaded(&clip, &HashMap::new());
        let track = &live.tracks[0];
        assert_eq!(track.translation.interpolation, Interpolation::Step);
        assert_eq!(track.rotation.interpolation, Interpolation::Step);
        assert_eq!(track.scale.interpolation, Interpolation::Step);
        // Step holds the earlier key: sampling mid-span returns the first value.
        assert_eq!(
            track.translation.sample(0.99),
            Some(Vec3::new(4.0, 5.0, 6.0))
        );
    }

    #[test]
    fn cubic_interpolation_converts_tangents_on_all_joint_channels() {
        use ornis_gltf::LoadedKeyTrack as MirrorTrack;
        let clip = LoadedSkelClip {
            name: String::new(),
            duration: 2.0,
            tracks: vec![LoadedJointTrack {
                joint: 0,
                node: NodeIdx(0),
                translation: MirrorTrack::cubic(
                    vec![
                        LoadedKey {
                            time: 0.0,
                            value: [0.0, 0.0, 0.0],
                        },
                        LoadedKey {
                            time: 2.0,
                            value: [2.0, 0.0, 0.0],
                        },
                    ],
                    vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    vec![[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ),
                rotation: MirrorTrack::cubic(
                    vec![
                        LoadedKey {
                            time: 0.0,
                            value: [0.0, 0.0, 0.0, 1.0],
                        },
                        LoadedKey {
                            time: 1.0,
                            value: [0.0, 0.0, 0.0, 1.0],
                        },
                    ],
                    vec![[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                    vec![[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                ),
                scale: MirrorTrack::cubic(Vec::new(), Vec::new(), Vec::new()),
            }],
        };
        let live = skel_clip_from_loaded(&clip, &HashMap::new());
        let track = &live.tracks[0];
        // Full round-trip: keys plus tangents convert value-for-value.
        assert_eq!(
            track.translation,
            KeyTrack::cubic(
                vec![
                    Key {
                        time: 0.0,
                        value: Vec3::ZERO,
                    },
                    Key {
                        time: 2.0,
                        value: Vec3::new(2.0, 0.0, 0.0),
                    },
                ],
                vec![Vec3::ZERO, Vec3::ZERO],
                vec![Vec3::new(2.0, 0.0, 0.0), Vec3::ZERO],
            )
        );
        assert!(matches!(
            track.rotation.interpolation,
            Interpolation::Cubic { .. }
        ));
        assert_eq!(track.rotation.keys[0].value, Quat::IDENTITY);
        // Tangent shapes the midpoint (1.0 + 0.125 · m0·dt = 1.5 at s = 0.5).
        let mid = track.translation.sample(1.0).expect("mid key");
        assert!(
            (mid.x - 1.5).abs() < 1e-6,
            "converted tangent must shape the midpoint, got {mid:?}"
        );
        // Cubic rotation samples stay unit.
        let spun = track.rotation.sample(0.5).expect("mid key");
        assert!((spun.length() - 1.0).abs() < 1e-5);
        // Empty cubic round-trips to an empty cubic (channel reads identity).
        assert!(track.scale.keys.is_empty());
        assert!(matches!(
            track.scale.interpolation,
            Interpolation::Cubic { .. }
        ));
        assert_eq!(track.scale.sample(0.5), None);
    }

    #[test]
    fn cubic_object_tracks_convert_through_node_mapping() {
        use ornis_gltf::LoadedKeyTrack as MirrorTrack;
        let loader = NodeIdx(0);
        let world = Entity::new(41);
        let clip = LoadedAnimClip {
            name: "cubic".to_string(),
            duration: 1.0,
            looping: true,
            tracks: vec![LoadedAnimTrack {
                node: loader,
                rest: ornis_gltf::LoadedJointRest::default(),
                translation: MirrorTrack::cubic(
                    vec![
                        LoadedKey {
                            time: 0.0,
                            value: [0.0, 0.0, 0.0],
                        },
                        LoadedKey {
                            time: 1.0,
                            value: [1.0, 1.0, 1.0],
                        },
                    ],
                    vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                ),
                rotation: MirrorTrack::cubic(Vec::new(), Vec::new(), Vec::new()),
                scale: MirrorTrack::cubic(Vec::new(), Vec::new(), Vec::new()),
            }],
        };
        let map = HashMap::from([(loader, world)]);
        let (live, dropped) = anim_clip_from_loaded("cubic".to_string(), &clip, &map);
        assert_eq!(dropped, 0);
        assert_eq!(live.tracks.len(), 1);
        assert_eq!(live.tracks[0].entity, world);
        // Zero tangents degrade to smoothstep: 0.5 at the midpoint.
        let mid = live.tracks[0].translation.sample(0.5).expect("mid key");
        assert!(
            (mid - Vec3::splat(0.5)).length() < 1e-6,
            "object cubic must smoothstep, got {mid:?}"
        );
        assert_eq!(live.tracks[0].rotation.sample(0.5), None);
    }

    #[test]
    fn rotation_quat_maps_xyzw_exactly() {
        use std::f32::consts::FRAC_1_SQRT_2;
        let clip = LoadedSkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![LoadedJointTrack {
                joint: 1,
                node: NodeIdx(1),
                translation: LoadedKeyTrack::linear(Vec::new()),
                rotation: LoadedKeyTrack::linear(vec![LoadedKey {
                    time: 1.0,
                    value: [0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2],
                }]),
                scale: LoadedKeyTrack::linear(Vec::new()),
            }],
        };
        let live = skel_clip_from_loaded(&clip, &HashMap::new());
        let got = live.tracks[0].rotation.keys[0].value;
        let want = Quat::from_xyzw(0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
        assert_eq!(got, want);
    }

    #[test]
    fn anim_clip_applies_node_mapping_and_counts_dropped() {
        let loader_a = NodeIdx(0);
        let loader_b = NodeIdx(1);
        let world_a = Entity::new(41);
        let clip = LoadedAnimClip {
            name: "mirror-name-ignored".to_string(),
            duration: 3.0,
            looping: true,
            tracks: vec![object_track(loader_a), object_track(loader_b)],
        };
        let map = HashMap::from([(loader_a, world_a)]);
        let (live, dropped) = anim_clip_from_loaded("walk".to_string(), &clip, &map);
        assert_eq!(live.name, "walk");
        assert_eq!(live.duration, 3.0);
        assert!(live.looping);
        assert_eq!(live.tracks.len(), 1);
        assert_eq!(live.tracks[0].entity, world_a);
        assert_eq!(live.tracks[0].translation.keys[0].value, Vec3::X);
        assert_eq!(dropped, 1);
    }

    #[test]
    fn anim_clip_empty_when_nothing_maps() {
        let clip = LoadedAnimClip {
            name: "x".to_string(),
            duration: 1.0,
            looping: false,
            tracks: vec![object_track(NodeIdx(7))],
        };
        let (live, dropped) = anim_clip_from_loaded("x".to_string(), &clip, &HashMap::new());
        assert!(live.tracks.is_empty());
        assert_eq!(dropped, 1);
        assert!(!live.looping);
    }

    #[test]
    fn skeleton_maps_parents_binds_and_names() {
        let skin = LoadedSkin {
            parents: vec![-1, 0],
            inverse_bind: vec![
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [5.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            joint_names: vec!["root".to_string(), "tip".to_string()],
            rest: vec![
                ornis_gltf::LoadedJointRest::default(),
                ornis_gltf::LoadedJointRest {
                    translation: [0.0, 3.0, 0.0],
                    ..Default::default()
                },
            ],
        };
        let live = skeleton_from_loaded(&skin);
        assert_eq!(live.rest[1].translation, glam::Vec3::new(0.0, 3.0, 0.0));
        assert_eq!(live.parents.len(), 2);
        assert_eq!(live.inverse_bind.len(), 2);
        assert_eq!(live.parents[0], None);
        assert_eq!(live.parents[1], Some(JointId::from_raw(0)));
        assert_eq!(
            live.inverse_bind[0],
            Mat4::from_cols_array_2d(&skin.inverse_bind[0])
        );
        assert_eq!(
            live.joint_names.as_ref(),
            &["root".to_string(), "tip".to_string()][..]
        );
        assert_eq!(live.validate(), Ok(2));
    }

    #[test]
    fn skinned_mesh_none_without_skin_data() {
        let skeleton = Entity::new(9);
        let bare = LoadedMesh {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0],
            normals: Some(vec![[0.0, 0.0, 1.0]]),
            uvs: Some(vec![[0.0, 0.0]]),
            joints: None,
            weights: None,
        };
        assert!(skinned_mesh_from_loaded(skeleton, &bare).is_none());
        let half = LoadedMesh {
            joints: Some(vec![[0, 0, 0, 0]]),
            weights: None,
            ..bare.clone()
        };
        assert!(skinned_mesh_from_loaded(skeleton, &half).is_none());
    }

    #[test]
    fn skinned_mesh_carries_bind_data_with_resolved_fallbacks() {
        let skeleton = Entity::new(9);
        let mesh = LoadedMesh {
            positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            indices: vec![0, 1, 2],
            normals: None,
            uvs: None,
            joints: Some(vec![[0, 0, 0, 0]; 3]),
            weights: Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]),
        };
        let live = skinned_mesh_from_loaded(skeleton, &mesh).expect("skinned mesh");
        assert_eq!(live.skeleton, skeleton);
        assert_eq!(live.joints, vec![[0, 0, 0, 0]; 3]);
        assert_eq!(live.weights, vec![[1.0, 0.0, 0.0, 0.0]; 3]);
        assert_eq!(live.vertex_count(), 3);
        // Fallbacks: recomputed normals, box-projected uvs (same as upload path).
        assert_eq!(live.normals.as_ref(), &mesh.resolved_normals()[..]);
        assert_eq!(live.uvs.as_ref(), &mesh.resolved_uvs()[..]);
        assert_eq!(live.indices.as_ref(), &[0, 1, 2][..]);
    }

    #[test]
    fn empty_clips_convert_to_empty_clips() {
        let skel = LoadedSkelClip {
            name: String::new(),
            duration: 0.0,
            tracks: Vec::new(),
        };
        let live_skel = skel_clip_from_loaded(&skel, &HashMap::new());
        assert_eq!(live_skel.duration, 0.0);
        assert!(live_skel.tracks.is_empty());

        let anim = LoadedAnimClip {
            name: "empty".to_string(),
            duration: 0.0,
            looping: true,
            tracks: Vec::new(),
        };
        let (live_anim, dropped) =
            anim_clip_from_loaded("empty".to_string(), &anim, &HashMap::new());
        assert!(live_anim.tracks.is_empty());
        assert_eq!(dropped, 0);
        assert_eq!(live_anim.duration, 0.0);
    }
    /// One object track with a single translation key (`X` at `t = 0`).
    fn object_track(node: NodeIdx) -> LoadedAnimTrack {
        LoadedAnimTrack {
            node,
            rest: ornis_gltf::LoadedJointRest::default(),
            translation: LoadedKeyTrack::linear(vec![LoadedKey {
                time: 0.0,
                value: [1.0, 0.0, 0.0],
            }]),
            rotation: LoadedKeyTrack::linear(Vec::new()),
            scale: LoadedKeyTrack::linear(Vec::new()),
        }
    }

    fn named_clip(name: &str) -> LoadedSkelClip {
        LoadedSkelClip {
            name: name.to_string(),
            duration: 1.0,
            tracks: Vec::new(),
        }
    }

    fn skinned_scene(clips: Vec<LoadedSkelClip>) -> ornis_gltf::Model {
        let mut primitive = primitive_on(NodeIdx(0));
        primitive.mesh.joints = Some(vec![[0, 0, 0, 0]]);
        primitive.mesh.weights = Some(vec![[1.0, 0.0, 0.0, 0.0]]);
        ornis_gltf::Model {
            name: "character".into(),
            nodes: vec![ornis_gltf::ModelNode {
                name: Some("part_0".into()),
                parent: None,
                local: ornis_core::Transform::IDENTITY,
                primitives: vec![0],
                skin: Some(0),
            }],
            roots: vec![NodeIdx(0)],
            primitives: vec![primitive],
            skins: vec![LoadedSkin {
                parents: vec![-1],
                inverse_bind: vec![[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]],
                joint_names: vec!["root".into()],
                rest: vec![ornis_gltf::LoadedJointRest::default()],
            }],
            skel_clips: clips,
            anim_clips: Vec::new(),
            stats: ornis_gltf::ImportStats::default(),
        }
    }

    /// Clip names index the character root. The first clip is not selected,
    /// and a duplicate name keeps the earlier playlist.
    #[test]
    fn wiring_indexes_clips_by_name_without_a_player() {
        let loaded = skinned_scene(vec![
            named_clip("Crouch_Fwd_Loop"),
            named_clip("Walk_Loop"),
            named_clip(""),
            named_clip("Walk_Loop"),
        ]);
        let mut store = SmartStore::new();
        let spawn = spawn_gltf_world(&mut store, &loaded);
        let wiring = wire_loaded_animation(&mut store, &loaded, &spawn);

        assert!(
            store
                .read_lane::<SkelPlayer>()
                .is_none_or(|lane| lane.is_empty()),
            "skeletal players stay empty until play"
        );
        let hero = spawn.entities[0];
        let animator = store
            .read_lane::<Animator>()
            .expect("animator lane")
            .get(hero)
            .cloned()
            .expect("animator on the first mesh");
        assert_eq!(
            animator.clip("Crouch_Fwd_Loop").map(|id| id.0),
            Some(wiring.skel_playlists[0])
        );
        assert_eq!(
            animator.clip("Walk_Loop").map(|id| id.0),
            Some(wiring.skel_playlists[1]),
            "duplicate Walk_Loop keeps the earlier playlist"
        );
        assert_eq!(
            animator.clip("skel_clip_2").map(|id| id.0),
            Some(wiring.skel_playlists[2])
        );
        let names = store.read_lane::<Name>().expect("names");
        assert_eq!(names.get(wiring.skel_playlists[1]).unwrap().0, "Walk_Loop");
        drop(names);

        let err = try_animator(&mut store, hero)
            .unwrap()
            .play("Nope")
            .unwrap_err();
        assert_eq!(
            err,
            AnimatorError::UnknownClip {
                name: "Nope".into()
            }
        );
        try_animator(&mut store, hero)
            .unwrap()
            .play("Walk_Loop")
            .unwrap();
        let cursor = store
            .read_lane::<SkelPlayer>()
            .unwrap()
            .get(wiring.roots[0])
            .copied()
            .unwrap();
        assert!(cursor.playing);
        assert_eq!(cursor.clip.0, wiring.skel_playlists[1]);
        assert_eq!(cursor.time, Seconds::ZERO);
        assert_eq!(cursor.weight, Clamped01::ONE);
    }

    /// An explicit scene root receives the animator instead of the first mesh.
    #[test]
    fn scene_root_override_receives_the_animator() {
        let loaded = skinned_scene(vec![named_clip("Walk_Loop")]);
        let mut store = SmartStore::new();
        let mut spawn = spawn_gltf_world(&mut store, &loaded);
        let root = store.create_entity();
        spawn.scene_root = Some(root);
        wire_loaded_animation(&mut store, &loaded, &spawn);
        let animators = store.read_lane::<Animator>().unwrap();
        assert!(animators.get(root).is_some());
        assert!(animators.get(spawn.entities[0]).is_none());
    }

    /// Object playlist named `name` with one track on `target`, plus a
    /// mid-flight player (`playing`, clock at `2.0`) so fades are visible.
    fn object_scene(store: &mut SmartStore, name: &str, target: Entity) -> ClipId {
        let playlist = store.create_entity();
        store.insert(playlist, Name(name.to_string()));
        store.insert_cold(
            playlist,
            AnimClip {
                name: name.to_string(),
                duration: 1.0,
                looping: true,
                tracks: vec![AnimTrack {
                    entity: target,
                    rest: JointTrs::IDENTITY,
                    translation: KeyTrack::linear(Vec::new()),
                    rotation: KeyTrack::linear(Vec::new()),
                    scale: KeyTrack::linear(Vec::new()),
                }],
            },
        );
        let clip = ClipId(playlist);
        store.insert(
            target,
            AnimPlayer {
                clip,
                time: 2.0,
                speed: 1.0,
                weight: 1.0,
                playing: true,
            },
        );
        clip
    }

    /// `Walk_Loop` fixture for the playback tests: a wired single-clip
    /// world with its hero, skeleton root and the clip's playlist id.
    fn walk_loop_hero() -> (SmartStore, Entity, Entity, ClipId) {
        let loaded = skinned_scene(vec![named_clip("Walk_Loop")]);
        let mut store = SmartStore::new();
        let spawn = spawn_gltf_world(&mut store, &loaded);
        let wiring = wire_loaded_animation(&mut store, &loaded, &spawn);
        (
            store,
            spawn.entities[0],
            wiring.roots[0],
            wiring.clips["Walk_Loop"],
        )
    }

    /// Object fixture for the fade tests: a playing mid-flight cursor on
    /// `target` plus two named clips (`Wave`, `Close`) to retarget between.
    fn fade_scene() -> (SmartStore, Entity, ClipId, ClipId) {
        let mut store = SmartStore::new();
        let target = store.create_entity();
        let wave = object_scene(&mut store, "Wave", target);
        let close = object_scene(&mut store, "Close", target);
        (store, target, wave, close)
    }

    /// The facade starts named skeletal clips on every target and fades
    /// replays from the outgoing cursor — the `AnimatorMut` behavior,
    /// reached without touching the lanes by hand.
    #[test]
    fn play_crossfade_starts_named_skeletal_clips_and_fades_replays() {
        use ornis_animation::SkelFade;
        let (mut store, hero, root, clip) = walk_loop_hero();

        play_crossfade(&mut store, hero, "Walk_Loop", Seconds::new(0.25)).expect("Walk_Loop");
        let cursor = store
            .read_lane::<SkelPlayer>()
            .unwrap()
            .get(root)
            .copied()
            .expect("cursor on the skeleton root");
        assert!(cursor.playing);
        assert_eq!(cursor.clip, clip);
        assert!(
            store
                .read_lane::<SkelFade>()
                .is_none_or(|fades| fades.get(root).is_none()),
            "a first start has nothing to fade from"
        );

        play_crossfade(&mut store, hero, "Walk_Loop", Seconds::new(0.25)).expect("replay");
        let fade = store
            .read_lane::<SkelFade>()
            .unwrap()
            .get(root)
            .copied()
            .expect("fade on replay");
        assert_eq!(fade.from.clip, clip);
        assert_eq!(fade.duration, Seconds::new(0.25));
    }

    /// One fade per root: a start over a `SkelFade` still in flight cuts
    /// to the fresh cursor at once — the in-flight fade is dropped,
    /// never restarted or stacked.
    #[test]
    fn play_crossfade_cuts_over_an_inflight_skel_fade() {
        use ornis_animation::SkelFade;
        let (mut store, hero, root, clip) = walk_loop_hero();

        play_crossfade(&mut store, hero, "Walk_Loop", Seconds::new(0.25)).expect("start");
        play_crossfade(&mut store, hero, "Walk_Loop", Seconds::new(0.25)).expect("fade");
        assert!(store.read_lane::<SkelFade>().unwrap().get(root).is_some());

        play_crossfade(&mut store, hero, "Walk_Loop", Seconds::new(0.25)).expect("cut over fade");
        assert!(
            store
                .read_lane::<SkelFade>()
                .is_none_or(|fades| fades.get(root).is_none()),
            "an in-flight fade is cut, not restarted"
        );
        let cursor = store
            .read_lane::<SkelPlayer>()
            .unwrap()
            .get(root)
            .copied()
            .expect("cursor");
        assert_eq!(cursor.clip, clip);
        assert_eq!(cursor.time, Seconds::ZERO);
        assert!(cursor.playing);
    }

    /// The object branch resolves the playlist by `Name` (the earliest
    /// created wins on duplicates) and writes `AnimFade` plus a fresh
    /// `AnimPlayer` — the typed inputs, not raw lane inserts.
    #[test]
    fn play_crossfade_fades_object_players_through_playlist_names() {
        let mut store = SmartStore::new();
        let target = store.create_entity();
        let clip = object_scene(&mut store, "Wave", target);
        let second = object_scene(&mut store, "Wave", target);
        assert_ne!(clip, second);

        play_crossfade(&mut store, target, "Wave", Seconds::new(0.5)).expect("Wave");
        let player = store
            .read_lane::<AnimPlayer>()
            .unwrap()
            .get(target)
            .copied()
            .expect("player");
        assert_eq!(player.clip, clip, "the earliest-created playlist wins");
        assert_eq!(player.time, 0.0);
        assert!(player.playing);
        let fade = store
            .read_lane::<AnimFade>()
            .unwrap()
            .get(target)
            .copied()
            .expect("fade");
        assert_eq!(fade.from.clip, second, "the outgoing cursor snapshot");
        assert_eq!(fade.from.time, 2.0);
        assert_eq!(fade.duration, Seconds::new(0.5));
        assert_eq!(fade.elapsed, Seconds::ZERO);
    }

    /// Unknown names and catalog-less entities fail with typed errors.
    #[test]
    fn play_crossfade_reports_unknown_clips_and_missing_players() {
        let mut store = SmartStore::new();
        let target = store.create_entity();
        object_scene(&mut store, "Wave", target);
        assert_eq!(
            play_crossfade(&mut store, target, "Nope", Seconds::new(0.5)),
            Err(PlaybackError::UnknownClip {
                name: "Nope".into()
            })
        );
        assert_eq!(
            play_crossfade(&mut store, Entity::new(999), "Wave", Seconds::new(0.5)),
            Err(PlaybackError::Missing)
        );

        // A clip that exists but does not track the entity is not its name.
        let bystander = store.create_entity();
        store.insert(
            bystander,
            AnimPlayer {
                clip: ClipId(store.create_entity()),
                time: 0.0,
                speed: 1.0,
                weight: 1.0,
                playing: true,
            },
        );
        assert_eq!(
            play_crossfade(&mut store, bystander, "Wave", Seconds::new(0.5)),
            Err(PlaybackError::UnknownClip {
                name: "Wave".into()
            })
        );
    }

    /// `fade_to_clip` cuts without a fade on a paused cursor or a
    /// non-positive duration and drops any fade in flight — the
    /// `AnimatorMut::start` rules for the object topology.
    #[test]
    fn fade_to_clip_cuts_without_a_fade_when_paused_or_zero() {
        let (mut store, target, outgoing, incoming) = fade_scene();

        // Zero duration: instant cut, no fade.
        fade_to_clip(&mut store, target, incoming, Seconds::ZERO).expect("cut");
        assert!(
            store
                .read_lane::<AnimFade>()
                .is_none_or(|fades| fades.get(target).is_none())
        );
        assert_eq!(
            store
                .read_lane::<AnimPlayer>()
                .unwrap()
                .get(target)
                .unwrap()
                .clip,
            incoming
        );

        // Paused cursor: no fade even with a positive duration.
        store
            .write_lane::<AnimPlayer>()
            .unwrap()
            .get_mut(target)
            .unwrap()
            .playing = false;
        fade_to_clip(&mut store, target, outgoing, Seconds::new(0.5)).expect("cut");
        assert!(
            store
                .read_lane::<AnimFade>()
                .is_none_or(|fades| fades.get(target).is_none())
        );

        // An in-flight fade dies on the next cut.
        fade_to_clip(&mut store, target, incoming, Seconds::new(0.5)).expect("fade");
        assert!(store.read_lane::<AnimFade>().unwrap().get(target).is_some());
        fade_to_clip(&mut store, target, outgoing, Seconds::ZERO).expect("cut");
        assert!(
            store
                .read_lane::<AnimFade>()
                .is_none_or(|fades| fades.get(target).is_none())
        );

        assert_eq!(
            fade_to_clip(&mut store, Entity::new(999), incoming, Seconds::ZERO),
            Err(PlaybackError::Missing)
        );
    }

    /// One fade per entity: a start over a fade still in flight cuts to
    /// the new clip at once — the in-flight `AnimFade` is dropped, never
    /// restarted or stacked (`AnimatorMut::start` rule, object topology).
    #[test]
    fn fade_to_clip_cuts_over_an_inflight_fade() {
        let (mut store, target, wave, close) = fade_scene();

        fade_to_clip(&mut store, target, wave, Seconds::new(0.5)).expect("fade");
        assert!(store.read_lane::<AnimFade>().unwrap().get(target).is_some());

        fade_to_clip(&mut store, target, close, Seconds::new(0.5)).expect("cut over fade");
        assert!(
            store
                .read_lane::<AnimFade>()
                .is_none_or(|fades| fades.get(target).is_none()),
            "an in-flight fade is cut, not restarted"
        );
        let player = store
            .read_lane::<AnimPlayer>()
            .unwrap()
            .get(target)
            .copied()
            .expect("player");
        assert_eq!(player.clip, close);
        assert_eq!(player.time, 0.0, "the cut lands on a fresh cursor");
    }

    /// `set_skel_layer` anchors on a `Skeleton` root: nothing is written
    /// without one, one layer per root replaces in place.
    #[test]
    fn set_skel_layer_requires_a_skeleton_root() {
        use ornis_animation::MaskId;
        let mut store = SmartStore::new();
        let root = store.create_entity();
        let playlist = store.create_entity();
        let layer = SkelLayer {
            player: SkelPlayer {
                clip: ClipId(playlist),
                time: Seconds::ZERO,
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            },
            mask: MaskId(playlist),
        };

        assert!(!set_skel_layer(&mut store, root, layer), "no Skeleton yet");
        assert!(
            store
                .read_lane::<SkelLayer>()
                .is_none_or(|layers| layers.get(root).is_none()),
            "nothing is written without the anchor"
        );

        store.insert(
            root,
            Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["joint".into()]),
        );
        assert!(set_skel_layer(&mut store, root, layer));
        assert_eq!(
            store.read_lane::<SkelLayer>().unwrap().get(root),
            Some(&layer)
        );
    }
}
