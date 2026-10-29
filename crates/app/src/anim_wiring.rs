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
    AnimClip, AnimEvent, AnimFade, AnimPlayer, AnimTrack, Animator, AnimatorError, BasePull,
    ClipEvents, ClipId, ClipSyncMarks, ControlRig, Effector, FREEFORM_MAX_POINTS, IkStack, JointId,
    JointMask, JointPose, JointTrack, JointTrs, Key, KeyTrack, MachineCursor, MaskId, RetargetMap,
    SkelClip, SkelFade, SkelInertia, SkelLayerStack, SkelPlayer, SkelTree, SkelTreePlayer,
    Skeleton, SkinnedMesh, SwitchCursor, SyncMark, TreeChild, TreeChild2d, TreeNodeDef, TreeParam,
    TreeTransition, fade_in_flight, freeform_cache, switch_selection, try_animator,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::{ChildOf, Clamped01, Entity, GlobalTransform, Seconds, SmartStore};
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
/// object) at its held clock. Players are fellow components, not
/// commands: the engine never touches them, only hosts do.
///
/// On a root carrying [`SkelTree`] this writes the ignored [`SkelPlayer`]
/// lane — the tree drives the root — so hosts should use the `GameWorld`
/// status entries (`resume_animation`/`pause_animation`/`stop_animation`),
/// which answer [`PlaybackError::TreeDriven`] instead. Returns whether a
/// player was found.
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
    /// The [`ClipId`] is not a live playlist with a track for the entity:
    /// a stale or fabricated handle, or a clip that does not animate it.
    #[error("clip {clip:?} does not track the entity")]
    InvalidClip {
        /// Rejected clip handle.
        clip: ClipId,
    },
    /// The root is driven by an [`SkelTree`]: the single-player playback
    /// entries do not apply while the tree owns the root. Drive the tree
    /// through [`set_tree_param`], or take the root back for the player
    /// with [`clear_skel_tree`] first.
    #[error("entity is driven by an animation tree")]
    TreeDriven,
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
/// [`PlaybackError::TreeDriven`] when `entity` — or any target of its
/// [`Animator`] catalog — is tree-driven ([`set_skel_tree`]): the tree
/// owns the root's playback, and no fade or cursor is written. [`PlaybackError::Missing`] when `entity` carries
/// neither catalog nor cursor (hand-built skeletal cursors without an
/// [`Animator`] start through [`set_playing`] instead).
/// [`PlaybackError::UnknownClip`] when `name` is not indexed.
/// [`PlaybackError::InvalidClip`] is unreachable here: object names
/// resolve only to playlists that track `entity`.
pub fn play_crossfade(
    store: &mut SmartStore,
    entity: Entity,
    name: &str,
    duration: Seconds,
) -> Result<(), PlaybackError> {
    if tree_driven(store, entity) {
        return Err(PlaybackError::TreeDriven);
    }
    // An `Animator` writes its player lanes onto the target roots — those
    // must not be tree-driven either.
    let animated_targets = store.read_lane::<Animator>().and_then(|animators| {
        animators
            .get(entity)
            .map(|animator| animator.targets().to_vec())
    });
    if let Some(targets) = animated_targets
        && targets.iter().any(|target| tree_driven(store, *target))
    {
        return Err(PlaybackError::TreeDriven);
    }
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

/// Whether `entity` is driven by an [`SkelTree`] ([`set_skel_tree`]): the
/// tree owns the root's playback until [`clear_skel_tree`].
pub(crate) fn tree_driven(store: &SmartStore, entity: Entity) -> bool {
    store
        .read_lane::<SkelTreePlayer>()
        .is_some_and(|players| players.contains(entity))
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
/// The caller passes `clip` by handle; [`fade_to_clip`] rejects anything
/// that is not a live playlist tracking `entity` with
/// [`PlaybackError::InvalidClip`], and [`play_crossfade`] resolves names
/// through the playlist [`Name`] — both paths keep `entity`'s cursor
/// intact on failure.
///
/// # Errors
///
/// [`PlaybackError::TreeDriven`] when `entity` is tree-driven
/// ([`set_skel_tree`]): the tree owns the root's playback, and no cursor
/// is written. [`PlaybackError::Missing`] when `entity` has no
/// [`AnimPlayer`]. [`PlaybackError::InvalidClip`] when `clip` is not a
/// live playlist with a track for `entity`.
pub fn fade_to_clip(
    store: &mut SmartStore,
    entity: Entity,
    clip: ClipId,
    duration: Seconds,
) -> Result<(), PlaybackError> {
    if tree_driven(store, entity) {
        return Err(PlaybackError::TreeDriven);
    }
    let outgoing = store
        .read_lane::<AnimPlayer>()
        .and_then(|players| players.get(entity).copied())
        .ok_or(PlaybackError::Missing)?;
    let clips = store.read_cold_lane::<AnimClip>();
    let valid = clips
        .as_deref()
        .and_then(|clips| clips.get(clip.0))
        .is_some_and(|playlist| playlist.tracks.iter().any(|track| track.entity == entity));
    drop(clips);
    if !valid {
        return Err(PlaybackError::InvalidClip { clip });
    }
    store.register::<AnimPlayer>();
    store.register::<AnimFade>();
    let fading = fade_in_flight::<AnimFade>(store, entity);
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

/// Installs `stack` as the complete overlay layer stack on a skeleton
/// root — the typed [`SkelLayerStack`] input, so hosts never install the
/// lane component themselves. Set semantics: `stack` replaces whatever
/// layers sat on the root (an empty stack clears them); layers blend
/// bottom-up over the base pose (see [`SkelLayerStack`]).
///
/// Returns whether `root` carries a [`Skeleton`] (the stack's anchor);
/// nothing is written without one.
pub fn set_skel_layers(store: &mut SmartStore, root: Entity, stack: SkelLayerStack) -> bool {
    if store
        .read_lane::<Skeleton>()
        .is_none_or(|skeletons| skeletons.get(root).is_none())
    {
        return false;
    }
    store.register::<SkelLayerStack>();
    store.insert(root, stack);
    true
}

/// Creates the cold-lane entity holding `mask` (design §8.6): the
/// [`MaskId`] handle [`SkelLayer`](ornis_animation::SkelLayer)s point
/// at. The weight count meets the skeleton at sample time (a mismatch is
/// bad skin, counted) — authoring stays permissive.
pub fn create_skel_mask(store: &mut SmartStore, mask: JointMask) -> MaskId {
    store.register_cold::<JointMask>();
    let entity = store.create_entity();
    store.insert_cold(entity, mask);
    MaskId(entity)
}

/// Installs the [`IkStack`] of `root` — the typed effector input
/// (design §8.2, §8.6, §8.7). Set semantics: `effectors` and `base`
/// replace whatever IK sat on the root (an empty set clears it);
/// effectors apply in vector order after the tree/layer blend and
/// before the ragdoll blend, and `base` (the §8.7 pelvis pull) leans
/// into unreachable two-bone targets first.
///
/// Returns whether anything is written: `root` must carry a
/// [`Skeleton`], every effector must pass [`Effector::installable`]
/// and `base` its in-range joint (one bad entry rejects the whole
/// set).
pub fn set_skel_ik(
    store: &mut SmartStore,
    root: Entity,
    effectors: Vec<Effector>,
    base: Option<BasePull>,
) -> bool {
    let Some(skeletons) = store.read_lane::<Skeleton>() else {
        return false;
    };
    let Some(skeleton) = skeletons.get(root) else {
        return false;
    };
    let count = skeleton.parents.len();
    drop(skeletons);
    let valid = effectors.iter().all(|effector| effector.installable(count))
        && base.is_none_or(|base| base.installable(count));
    if !valid {
        return false;
    }
    store.register::<IkStack>();
    if effectors.is_empty() {
        // The documented clear («an empty stack clears it»): the root
        // goes back to "no stack, never touched" (design §8.2).
        if let Some(mut stacks) = store.write_lane::<IkStack>() {
            stacks.remove(root);
        }
    } else {
        store.insert(root, IkStack { effectors, base });
    }
    true
}

/// Installs the [`IkStack`] of `root` from world-space goals (design
/// §8.8, the control rig): `rig`'s targets are world points, mapped
/// through the world matrix of `root`'s frame into its model space and
/// then installed like [`set_skel_ik`] — same set semantics and
/// validation (one bad entry rejects the whole set; an empty goal list
/// clears the stack).
///
/// The world matrix is `root`'s own [`GlobalTransform`] or the nearest
/// ancestor's (`ChildOf` chain): a wired skeleton root carries no local
/// transform and lives in its parent's frame. `root` under no
/// [`GlobalTransform`] is not written — clearing the stack in that state
/// goes through [`set_skel_ik`] with an empty set. Non-finite mapped
/// goals (a singular root matrix) fail [`Effector::installable`] and
/// reject the whole set, keeping the previously installed stack.
///
/// The installed goals are model-space, so they go stale as the root
/// moves — call again from the update loop (one inverse plus one point
/// per goal).
pub fn set_control_rig(store: &mut SmartStore, root: Entity, rig: ControlRig) -> bool {
    let Some(root_world) = root_world_matrix(store, root) else {
        return false;
    };
    let stack = rig.into_stack(root_world);
    set_skel_ik(store, root, stack.effectors, stack.base)
}

/// The world matrix of `root`'s frame: its own `GlobalTransform`, or the
/// nearest ancestor's along `ChildOf` (a skeleton root without a local
/// transform lives in its parent's frame). `None` when the chain carries
/// no [`GlobalTransform`]. The walk stays within
/// [`HIERARCHY_WALK_LIMIT`](ornis_core::HIERARCHY_WALK_LIMIT):
/// [`set_parent`](ornis_core::set_parent) keeps the hierarchy acyclic,
/// and the bound holds even for raw `ChildOf` writes.
fn root_world_matrix(store: &SmartStore, root: Entity) -> Option<Mat4> {
    let globals = store.read_lane::<GlobalTransform>()?;
    let parents = store.read_lane::<ChildOf>();
    let mut cursor = Some(root);
    for _ in 0..ornis_core::HIERARCHY_WALK_LIMIT {
        let entity = cursor?;
        if let Some(global) = globals.get(entity) {
            return Some(global.to_mat4());
        }
        cursor = parents
            .as_ref()
            .and_then(|parents| parents.get(entity))
            .map(|parent| parent.0);
    }
    None
}

/// Installs the [`RetargetMap`] of `target` — the explicit «source joint →
/// target joint» correspondence for playing `source`-authored clips on the
/// target skeleton (design §8.3). Set semantics: `joints` replaces whatever
/// map sat on the target (all-`None` transfers nothing and holds every
/// joint at rest); sampled channels land on the mapped target joints with
/// the translation scaled by the rest-bone length ratio.
///
/// Returns whether anything is written: both roots must carry a
/// [`Skeleton`], `joints` must name exactly one entry per source joint and
/// every `Some` must be in range of the target (one bad entry rejects the
/// whole map).
pub fn set_retarget_map(
    store: &mut SmartStore,
    target: Entity,
    source: Entity,
    joints: Vec<Option<JointId>>,
) -> bool {
    let Some(skeletons) = store.read_lane::<Skeleton>() else {
        return false;
    };
    let Some(source_skeleton) = skeletons.get(source) else {
        return false;
    };
    let Some(target_skeleton) = skeletons.get(target) else {
        return false;
    };
    let Some(map) = RetargetMap::bake(source_skeleton, target_skeleton, joints) else {
        return false;
    };
    drop(skeletons);
    store.register_cold::<RetargetMap>();
    store.insert_cold(target, map);
    true
}

/// Whether `children` form a complete row-major lattice for
/// [`TreeNodeDef::Blend2d`]: rows of one `position[1]` with the same
/// strictly ascending `position[0]` x positions, rows themselves
/// strictly ascending in `position[1]`. Empty reads as rest and passes.
fn blend2d_lattice_ok(children: &[TreeChild2d]) -> bool {
    let Some(first) = children.first() else {
        return true;
    };
    let cols = children
        .iter()
        .take_while(|child| child.position[1] == first.position[1])
        .count();
    children.len().is_multiple_of(cols)
        && children.chunks(cols).enumerate().all(|(row, line)| {
            line.iter().enumerate().all(|(col, child)| {
                child.position[1] == line[0].position[1]
                    && child.position[0] == children[col].position[0]
                    && (col == 0 || children[col - 1].position[0] < child.position[0])
                    && (row == 0 || children[(row - 1) * cols].position[1] < line[0].position[1])
            })
        })
}

/// Whether `tree` is structurally sound and installable: non-empty
/// nodes, finite parameter defaults, blend axes that name a parameter,
/// children at strictly ascending finite positions pointing at earlier
/// nodes, and leaf [`ClipId`]s that name a live [`SkelClip`] playlist —
/// the same handle check [`fade_to_clip`] applies to its `clip`. An
/// unknown handle is refused here instead of bad-skinning the root every
/// frame.
/// [`TreeNodeDef::Blend1d`] arm of [`tree_is_valid`]: the axis in range,
/// strictly ascending finite positions and children among the earlier
/// nodes.
fn blend1d_node_valid(tree: &SkelTree, axis: usize, children: &[TreeChild], index: usize) -> bool {
    axis < tree.params.len()
        && children
            .windows(2)
            .all(|pair| pair[0].position < pair[1].position)
        && children
            .iter()
            .all(|child| child.node < index && child.position.is_finite())
}

/// [`TreeNodeDef::Blend2d`] arm of [`tree_is_valid`]: distinct axes in
/// range, finite positions, children among the earlier nodes and an
/// installable lattice.
fn blend2d_node_valid(
    tree: &SkelTree,
    axes: [usize; 2],
    children: &[TreeChild2d],
    index: usize,
) -> bool {
    axes[0] != axes[1]
        && axes.iter().all(|axis| *axis < tree.params.len())
        && children
            .iter()
            .all(|child| child.node < index && child.position.iter().all(|axis| axis.is_finite()))
        && blend2d_lattice_ok(children)
}

/// [`TreeNodeDef::Freeform2d`] arm of [`tree_is_valid`]: distinct axes
/// in range, children among the earlier nodes at finite **distinct**
/// positions (the triangulation needs them) and no more than
/// [`FREEFORM_MAX_POINTS`] of them.
fn freeform2d_node_valid(
    tree: &SkelTree,
    axes: [usize; 2],
    children: &[TreeChild2d],
    index: usize,
) -> bool {
    axes[0] != axes[1]
        && axes.iter().all(|axis| *axis < tree.params.len())
        && children.len() <= FREEFORM_MAX_POINTS
        && children.iter().enumerate().all(|(slot, child)| {
            child.node < index
                && child.position.iter().all(|axis| axis.is_finite())
                && children[..slot]
                    .iter()
                    .all(|other| other.position != child.position)
        })
}

/// [`TreeNodeDef::Machine`] arm of [`tree_is_valid`]: non-empty states
/// among the earlier nodes, an in-range initial state and in-range finite
/// transitions.
fn machine_node_valid(
    tree: &SkelTree,
    states: &[usize],
    initial: usize,
    transitions: &[TreeTransition],
    index: usize,
) -> bool {
    !states.is_empty()
        && initial < states.len()
        && states.iter().all(|state| *state < index)
        && transitions.iter().all(|transition| {
            transition.from < states.len()
                && transition.to < states.len()
                && transition.when.param < tree.params.len()
                && transition.when.value.is_finite()
                && transition.fade.get().is_finite()
        })
}

/// [`TreeNodeDef::Switch`] arm of [`tree_is_valid`]: a finite fade, the
/// parameter in range and non-empty children among the earlier nodes.
fn switch_node_valid(
    tree: &SkelTree,
    param: usize,
    children: &[usize],
    fade: Seconds,
    index: usize,
) -> bool {
    param < tree.params.len()
        && !children.is_empty()
        && children.iter().all(|child| *child < index)
        && fade.get().is_finite()
}

fn tree_is_valid(store: &SmartStore, tree: &SkelTree) -> bool {
    let clips = store.read_cold_lane::<SkelClip>();
    !tree.nodes.is_empty()
        && tree.params.iter().all(|param| param.default.is_finite())
        && tree
            .nodes
            .iter()
            .enumerate()
            .all(|(index, node)| match node {
                TreeNodeDef::Clip { clip } => clips
                    .as_deref()
                    .and_then(|clips| clips.get(clip.0))
                    .is_some(),
                TreeNodeDef::Blend1d { axis, children } => {
                    blend1d_node_valid(tree, *axis, children, index)
                }
                TreeNodeDef::Blend2d { axes, children } => {
                    blend2d_node_valid(tree, *axes, children, index)
                }
                TreeNodeDef::Freeform2d { axes, children } => {
                    freeform2d_node_valid(tree, *axes, children, index)
                }
                TreeNodeDef::Machine {
                    states,
                    initial,
                    transitions,
                } => machine_node_valid(tree, states, *initial, transitions, index),
                TreeNodeDef::Switch {
                    param,
                    children,
                    fade,
                } => switch_node_valid(tree, *param, children, *fade, index),
                _ => false,
            })
}

/// Installs `tree` as the animation tree on a skeleton root — the typed
/// [`SkelTree`] input, so hosts never install the tree lanes themselves.
/// Set semantics: `tree` replaces whatever tree sat on the root. Leaf
/// cursors ([`SkelTreePlayer::leaves`], one per [`TreeNodeDef::Clip`] in
/// node order) start at time zero with `speed = 1`, full weight and
/// looping playback; parameters take their [`TreeParam::default`] values
/// until [`set_tree_param`]. A root with a tree is driven by it — a
/// [`SkelPlayer`] on the same entity is ignored (its clock does not
/// advance), any [`SkelFade`] still in flight on the root is dropped
/// (a fade cannot meaningfully cross into a tree), and the playback
/// entries report [`PlaybackError::TreeDriven`] until [`clear_skel_tree`]
/// takes the root back.
///
/// Returns whether `root` carries a [`Skeleton`] (the tree's anchor) and
/// `tree` passes `tree_is_valid`; nothing is written otherwise.
pub fn set_skel_tree(store: &mut SmartStore, root: Entity, tree: SkelTree) -> bool {
    if store
        .read_lane::<Skeleton>()
        .is_none_or(|skeletons| skeletons.get(root).is_none())
    {
        return false;
    }
    if !tree_is_valid(store, &tree) {
        return false;
    }
    store.register_cold::<SkelTree>();
    store.register::<SkelTreePlayer>();
    let mut player = SkelTreePlayer::default();
    player.params = tree.params.iter().map(|param| param.default).collect();
    player.leaves = tree
        .nodes
        .iter()
        .filter_map(|node| match node {
            TreeNodeDef::Clip { clip } => Some(SkelPlayer {
                clip: *clip,
                time: Seconds::new(0.0),
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            }),
            _ => None,
        })
        .collect();
    player.machines = tree
        .nodes
        .iter()
        .filter_map(|node| match node {
            TreeNodeDef::Machine { initial, .. } => Some(MachineCursor {
                state: *initial,
                from: None,
                elapsed: Seconds::ZERO,
                duration: Seconds::ZERO,
            }),
            _ => None,
        })
        .collect();
    player.switches = tree
        .nodes
        .iter()
        .filter_map(|node| match node {
            TreeNodeDef::Switch {
                param,
                children,
                fade: _,
            } => Some(SwitchCursor {
                child: switch_selection(tree.params[*param].default, children.len()),
                from: None,
                elapsed: Seconds::ZERO,
                duration: Seconds::ZERO,
            }),
            _ => None,
        })
        .collect();
    player.freeforms = tree
        .nodes
        .iter()
        .filter_map(|node| match node {
            TreeNodeDef::Freeform2d { children, .. } => Some(freeform_cache(children)),
            _ => None,
        })
        .collect();
    store.insert_cold(root, tree);
    store.insert(root, player);
    // A fade is meaningless on a tree root: drop one still in flight from
    // player playback — it would tick in `advance_fades` and be ignored.
    if let Some(mut fades) = store.write_lane::<SkelFade>() {
        fades.remove(root);
    }
    true
}

/// Sets tree parameter `name` on a skeleton root to `value` — the typed
/// host input for [`SkelTree`] axes (the key is [`TreeParam::name`]). A
/// value outside an axis's covered range simply clamps at the end inputs
/// during blend1d evaluation.
///
/// Returns `false` — and writes nothing — when `root` has no [`SkelTree`]
/// or [`SkelTreePlayer`], names no parameter, or `value` is not finite
/// (NaN/±inf would poison the blended pose silently).
pub fn set_tree_param(store: &mut SmartStore, root: Entity, name: &str, value: f32) -> bool {
    if !value.is_finite() {
        return false;
    }
    let trees = store.read_cold_lane::<SkelTree>();
    let Some(index) = trees
        .as_deref()
        .and_then(|trees| trees.get(root))
        .and_then(|tree| {
            tree.params
                .iter()
                .position(|param: &TreeParam| param.name == name)
        })
    else {
        return false;
    };
    let Some(mut players) = store.write_lane::<SkelTreePlayer>() else {
        return false;
    };
    let Some(player) = players.get_mut(root) else {
        return false;
    };
    let Some(slot) = player.params.get_mut(index) else {
        return false;
    };
    *slot = value;
    true
}

/// Installs sync marks for `clip` — the typed [`ClipSyncMarks`] input,
/// so hosts never install the lane component themselves. Set semantics:
/// `marks` replaces whatever marks sat on the clip; empty `marks` clears
/// them. Marks are authoring data: when a
/// [`SkelTree`] machine transition fires, the
/// incoming state's looping leaf cursors re-phase so that same-named
/// marks of the leaving and arriving clips land on the same wall-clock
/// hit (`docs/animation-design.md` §7.5).
///
/// Returns `false` — and writes nothing — when `clip` is not a live
/// [`SkelClip`] playlist or any mark is malformed: an empty name, or an
/// `at` outside `[0, 1)` / non-finite.
pub fn set_clip_sync_marks(store: &mut SmartStore, clip: ClipId, marks: Vec<SyncMark>) -> bool {
    let clips = store.read_cold_lane::<SkelClip>();
    let known = clips
        .as_deref()
        .and_then(|clips| clips.get(clip.0))
        .is_some();
    drop(clips);
    if !known {
        return false;
    }
    if !marks
        .iter()
        .all(|mark| !mark.name.is_empty() && mark.at.is_finite() && (0.0..1.0).contains(&mark.at))
    {
        return false;
    }
    store.register_cold::<ClipSyncMarks>();
    store.insert_cold(clip.0, ClipSyncMarks { marks });
    true
}

/// Whether every authored mark-like entry is legal: non-empty name and
/// a finite `at` in `0..1` — the validation contract of clip mark data.
fn marks_authorable<'a>(mut entries: impl Iterator<Item = (&'a str, f32)>) -> bool {
    entries.all(|(name, at)| !name.is_empty() && at.is_finite() && (0.0..1.0).contains(&at))
}

/// Installs clip events for `clip` — the typed [`ClipEvents`] input, the
/// counterpart of [`set_clip_sync_marks`] for the events lane (design
/// §8.1). `clip` must be a live playlist entity ([`SkelClip`] or
/// [`AnimClip`]); `events` replaces any previous set (an empty list
/// clears the lane). Returns `false` — changing nothing — for an unknown
/// clip or an invalid event (empty name, `at` outside `0..1` or
/// non-finite).
pub fn set_clip_events(store: &mut SmartStore, clip: ClipId, events: Vec<AnimEvent>) -> bool {
    let known = {
        let skel = store.read_cold_lane::<SkelClip>();
        let obj = store.read_cold_lane::<AnimClip>();
        skel.as_deref()
            .is_some_and(|clips| clips.get(clip.0).is_some())
            || obj
                .as_deref()
                .is_some_and(|clips| clips.get(clip.0).is_some())
    };
    let pairs = events.iter().map(|event| (event.name.as_str(), event.at));
    let ok = known && marks_authorable(pairs);
    if ok {
        store.insert_cold(clip.0, ClipEvents { events });
    }
    ok
}

/// Removes the [`SkelTree`] and its [`SkelTreePlayer`] from a skeleton
/// root — the way back to single-player playback. Set semantics: after
/// the call the root is no longer tree-driven, whatever tree was on it.
/// The playback entries accept the root again
/// ([`PlaybackError::TreeDriven`] is gone).
///
/// Returns whether `root` carried a tree to remove.
pub fn clear_skel_tree(store: &mut SmartStore, root: Entity) -> bool {
    clear_root_lanes::<SkelTree, SkelTreePlayer>(store, root)
}

/// Hot-lane twin of [`clear_root_lanes`]: removes `root` from the hot
/// lanes `A` and `B`, reporting whether either held it.
pub(crate) fn clear_root_hot_lanes<A: 'static + Send + Sync, B: 'static + Send + Sync>(
    store: &mut SmartStore,
    root: Entity,
) -> bool {
    let had = store
        .read_lane::<A>()
        .is_some_and(|lane| lane.contains(root))
        || store
            .read_lane::<B>()
            .is_some_and(|lane| lane.contains(root));
    if let Some(mut lane) = store.write_lane::<A>() {
        lane.remove(root);
    }
    if let Some(mut lane) = store.write_lane::<B>() {
        lane.remove(root);
    }
    had
}

/// Shared clear shape of the typed facades: removes `root` from the cold
/// lane `C` and the hot lane `H`, reporting whether either held it.
pub(crate) fn clear_root_lanes<C: 'static + Send + Sync, H: 'static + Send + Sync>(
    store: &mut SmartStore,
    root: Entity,
) -> bool {
    let had = store
        .read_cold_lane::<C>()
        .is_some_and(|lane| lane.contains(root))
        || store
            .read_lane::<H>()
            .is_some_and(|lane| lane.contains(root));
    if let Some(mut lane) = store.write_cold_lane::<C>() {
        lane.remove(root);
    }
    if let Some(mut lane) = store.write_lane::<H>() {
        lane.remove(root);
    }
    had
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
    store.register::<SkelLayerStack>();
    store.register::<SkelTreePlayer>();
    store.register::<SkelFade>();
    store.register::<SkelInertia>();
    store.register_cold::<SkelClip>();
    store.register_cold::<JointMask>();
    store.register_cold::<SkelTree>();
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
    use ornis_animation::{
        AnimatorError, Interpolation, LookAt, SkelLayer, TreeCmp, TreeCond, TreeParam,
        TreeTransition, TwoBoneIk, try_animator,
    };
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

    /// A one-joint `Skeleton` anchor with identity bind — the component
    /// the facade fixtures attach their lanes to.
    fn skeleton_anchor() -> Skeleton {
        Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["joint".into()])
    }

    /// Layer fixture: a bare `Skeleton` root — the `set_skel_layers`
    /// anchor and the home of parameter-only trees.
    fn layer_root(store: &mut SmartStore) -> Entity {
        let root = store.create_entity();
        store.insert(root, skeleton_anchor());
        root
    }

    /// Tree fixture: a `Skeleton` root with one live playlist clip
    /// (`walk`) — the world the tree-facade tests start from.
    fn tree_hero(store: &mut SmartStore) -> (Entity, ClipId) {
        store.register_cold::<SkelClip>();
        let root = store.create_entity();
        store.insert(root, skeleton_anchor());
        let clip = store.create_entity();
        store.insert_cold(
            clip,
            SkelClip {
                name: "walk".to_string(),
                duration: 1.0,
                tracks: Vec::new(),
            },
        );
        (root, ClipId(clip))
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

    /// A [`ClipId`] that is not a live playlist tracking `entity` is
    /// rejected with [`PlaybackError::InvalidClip`] and writes nothing.
    #[test]
    fn fade_to_clip_rejects_an_invalid_clip_id() {
        let (mut store, target, _wave, _close) = fade_scene();
        let before = store
            .read_lane::<AnimPlayer>()
            .unwrap()
            .get(target)
            .copied()
            .expect("fade_scene installs a cursor");

        // A playlist of someone else's: valid handle, wrong entity.
        let foreign_root = store.create_entity();
        let foreign = object_scene(&mut store, "Idle", foreign_root);
        let err = fade_to_clip(&mut store, target, foreign, Seconds::new(0.5))
            .expect_err("a clip that does not track the entity is invalid");
        assert_eq!(err, PlaybackError::InvalidClip { clip: foreign });

        // A fabricated handle: no playlist at that entity at all.
        let missing = ClipId(store.create_entity());
        let err = fade_to_clip(&mut store, target, missing, Seconds::new(0.5))
            .expect_err("a handle with no playlist is invalid");
        assert_eq!(err, PlaybackError::InvalidClip { clip: missing });

        assert_eq!(
            store
                .read_lane::<AnimPlayer>()
                .unwrap()
                .get(target)
                .copied()
                .expect("cursor stays"),
            before,
            "an invalid clip writes nothing"
        );
        assert!(
            store
                .read_lane::<AnimFade>()
                .is_none_or(|fades| fades.get(target).is_none()),
            "an invalid clip installs no fade"
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

    /// `set_skel_layers` anchors on a `Skeleton` root: nothing is written
    /// without one, and the call replaces the whole stack in place.
    #[test]
    fn set_skel_layers_requires_a_skeleton_root() {
        use ornis_animation::MaskId;
        let mut store = SmartStore::new();
        let root = store.create_entity();
        let playlist = store.create_entity();
        let stack = SkelLayerStack {
            layers: vec![SkelLayer {
                player: SkelPlayer {
                    clip: ClipId(playlist),
                    time: Seconds::ZERO,
                    speed: Seconds::new(1.0),
                    weight: Clamped01::ONE,
                    playing: true,
                    looping: true,
                },
                mask: MaskId(playlist),
            }],
        };
        assert!(
            !set_skel_layers(&mut store, root, stack.clone()),
            "no Skeleton yet"
        );
        assert!(
            store
                .read_lane::<SkelLayerStack>()
                .is_none_or(|layers| layers.get(root).is_none()),
            "nothing is written without the anchor"
        );
        store.insert(root, skeleton_anchor());
        assert!(set_skel_layers(&mut store, root, stack.clone()));
        assert_eq!(
            store.read_lane::<SkelLayerStack>().unwrap().get(root),
            Some(&stack)
        );
    }

    /// Every `set_skel_layers` call installs a fresh stack: the second call
    /// wipes the first (no append, no merge), an empty one clears.
    #[test]
    fn set_skel_layers_replaces_the_whole_stack() {
        use ornis_animation::MaskId;
        let mut store = SmartStore::new();
        let root = layer_root(&mut store);
        let mask_entity = store.create_entity();
        let clip_a = store.create_entity();
        let clip_b = store.create_entity();
        let one = |clip: Entity| SkelLayer {
            player: SkelPlayer {
                clip: ClipId(clip),
                time: Seconds::ZERO,
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            },
            mask: MaskId(mask_entity),
        };
        let pair = SkelLayerStack {
            layers: vec![one(clip_a), one(clip_b)],
        };
        assert!(set_skel_layers(&mut store, root, pair));
        let two = store
            .read_lane::<SkelLayerStack>()
            .and_then(|lane| lane.get(root).cloned())
            .expect("stack kept");
        assert_eq!(two.layers.len(), 2, "both layers installed");

        let solo = SkelLayerStack {
            layers: vec![two.layers[0]],
        };
        assert!(set_skel_layers(&mut store, root, solo.clone()));
        let replaced = store
            .read_lane::<SkelLayerStack>()
            .and_then(|lane| lane.get(root).cloned())
            .expect("stack kept");
        assert_eq!(replaced, solo, "the fresh stack replaces, never merges");

        assert!(set_skel_layers(&mut store, root, SkelLayerStack::default()));
        assert_eq!(
            store
                .read_lane::<SkelLayerStack>()
                .and_then(|lane| lane.get(root).cloned()),
            Some(SkelLayerStack::default()),
            "an empty stack clears every layer"
        );
    }

    /// Three-joint chain anchor for [`set_skel_ik`] fixtures.
    fn ik_root(store: &mut SmartStore) -> Entity {
        let root = store.create_entity();
        store.insert(
            root,
            Skeleton::new(
                vec![None, Some(JointId::from_raw(0)), Some(JointId::from_raw(1))],
                vec![Mat4::IDENTITY; 3],
                vec!["a".into(), "b".into(), "c".into()],
            ),
        );
        root
    }

    /// `set_skel_ik` requires the skeleton anchor, validates every
    /// effector against it (one bad entry rejects the set), replaces
    /// the stack wholesale and clears with an empty list.
    #[test]
    fn set_skel_ik_validates_against_the_skeleton() {
        let chain = TwoBoneIk {
            root: JointId::from_raw(0),
            mid: JointId::from_raw(1),
            end: JointId::from_raw(2),
            target: Vec3::new(1.0, 1.0, 0.0),
            weight: Clamped01::ONE,
        };
        let constraint = Effector::TwoBone(chain);
        let mut store = SmartStore::new();
        let bare = store.create_entity();
        assert!(!set_skel_ik(&mut store, bare, vec![constraint], None));
        let root = ik_root(&mut store);
        let out_of_range = Effector::TwoBone(TwoBoneIk {
            end: JointId::from_raw(9),
            ..chain
        });
        assert!(!set_skel_ik(&mut store, root, vec![out_of_range], None));
        assert!(store.read_lane::<IkStack>().is_none());
        assert!(set_skel_ik(&mut store, root, vec![constraint], None));
        let lane = store.read_lane::<IkStack>().expect("ik lane");
        let effectors = lane.get(root).map(|stack| stack.effectors.clone());
        assert_eq!(effectors, Some(vec![constraint]));
        drop(lane);
        assert!(set_skel_ik(&mut store, root, Vec::new(), None));
        let lane = store.read_lane::<IkStack>().expect("ik lane");
        assert!(
            lane.get(root).is_none(),
            "an empty set clears the stack («an empty stack clears it»)"
        );
    }

    /// A zero look-at axis is not an aim: the facade writes nothing.
    /// A finite axis on an in-range joint installs.
    #[test]
    fn set_skel_ik_rejects_a_zero_look_at_aim() {
        let mut store = SmartStore::new();
        let root = ik_root(&mut store);
        let look = LookAt {
            joint: JointId::from_raw(0),
            target: Vec3::Y,
            aim: Vec3::ZERO,
            weight: Clamped01::ONE,
        };
        assert!(!set_skel_ik(
            &mut store,
            root,
            vec![Effector::LookAt(look)],
            None
        ));
        assert!(store.read_lane::<IkStack>().is_none());
        let aimed = LookAt {
            aim: Vec3::X,
            ..look
        };
        assert!(set_skel_ik(
            &mut store,
            root,
            vec![Effector::LookAt(aimed)],
            None
        ));
        let lane = store.read_lane::<IkStack>().expect("ik lane");
        assert_eq!(
            lane.get(root).map(|stack| stack.effectors.clone()),
            Some(vec![Effector::LookAt(aimed)])
        );
    }

    /// The §8.7 base pull rides the stack and validates against the
    /// skeleton like the effectors do (an out-of-range base rejects the
    /// whole set).
    #[test]
    fn set_skel_ik_validates_the_base_pull() {
        let mut store = SmartStore::new();
        let root = ik_root(&mut store);
        let chain = Effector::TwoBone(TwoBoneIk {
            root: JointId::from_raw(0),
            mid: JointId::from_raw(1),
            end: JointId::from_raw(2),
            target: Vec3::new(1.0, 1.0, 0.0),
            weight: Clamped01::ONE,
        });
        let out_of_range = BasePull {
            joint: JointId::from_raw(9),
            weight: Clamped01::ONE,
        };
        assert!(!set_skel_ik(
            &mut store,
            root,
            vec![chain],
            Some(out_of_range)
        ));
        assert!(store.read_lane::<IkStack>().is_none());
        let pelvis = BasePull {
            joint: JointId::from_raw(0),
            weight: Clamped01::ONE,
        };
        assert!(set_skel_ik(&mut store, root, vec![chain], Some(pelvis)));
        let lane = store.read_lane::<IkStack>().expect("ik lane");
        assert_eq!(lane.get(root).map(|stack| stack.base), Some(Some(pelvis)));
    }

    /// The §8.8 facade: world goals land as model-space goals through
    /// the root's `GlobalTransform`; without one nothing is written.
    #[test]
    fn set_control_rig_maps_goals_through_the_root_global() {
        let mut store = SmartStore::new();
        let root = ik_root(&mut store);
        let rig = ControlRig {
            goals: vec![Effector::TwoBone(TwoBoneIk {
                root: JointId::from_raw(0),
                mid: JointId::from_raw(1),
                end: JointId::from_raw(2),
                target: Vec3::new(12.0, 1.0, 0.0),
                weight: Clamped01::ONE,
            })],
            base: None,
        };
        // No `GlobalTransform` yet — nothing is written.
        assert!(!set_control_rig(&mut store, root, rig.clone()));
        assert!(store.read_lane::<IkStack>().is_none());
        let mut global = GlobalTransform::IDENTITY;
        global.translation = Vec3::new(10.0, 0.0, 0.0);
        store.insert(root, global);
        // The lane exists, but this root has no entry — nothing is written.
        let bare = store.create_entity();
        assert!(!set_control_rig(&mut store, bare, ControlRig::default()));
        assert!(set_control_rig(&mut store, root, rig));
        let lane = store.read_lane::<IkStack>().expect("ik lane");
        let stack = lane.get(root).expect("stack");
        assert_eq!(stack.base, None);
        match stack.effectors[0] {
            Effector::TwoBone(chain) => {
                assert!(
                    (chain.target - Vec3::new(2.0, 1.0, 0.0)).length() < 1e-4,
                    "target at {:?}",
                    chain.target
                );
            }
            _ => panic!("the goal stays a chain"),
        }
    }

    fn retarget_skeleton(store: &mut SmartStore, rest: Vec3) -> Entity {
        let root = store.create_entity();
        store.insert(
            root,
            Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec![String::new()]).with_rest(vec![
                JointTrs {
                    translation: rest,
                    ..JointTrs::IDENTITY
                },
            ]),
        );
        root
    }

    /// `set_retarget_map` requires both skeleton anchors, validates the
    /// mapping against them (one bad entry rejects the map), bakes the
    /// rest-length scale and replaces the map wholesale.
    #[test]
    fn set_retarget_map_validates_and_bakes() {
        let mut store = SmartStore::new();
        let target = retarget_skeleton(&mut store, Vec3::X * 2.0);
        let source = retarget_skeleton(&mut store, Vec3::X);
        let bare = store.create_entity();
        assert!(!set_retarget_map(
            &mut store,
            target,
            bare,
            vec![Some(JointId::from_raw(0))]
        ));
        assert!(!set_retarget_map(
            &mut store,
            bare,
            source,
            vec![Some(JointId::from_raw(0))]
        ));
        assert!(!set_retarget_map(&mut store, target, source, Vec::new()));
        assert!(!set_retarget_map(
            &mut store,
            target,
            source,
            vec![Some(JointId::from_raw(2))]
        ));
        assert!(store.read_cold_lane::<RetargetMap>().is_none());
        assert!(set_retarget_map(
            &mut store,
            target,
            source,
            vec![Some(JointId::from_raw(0))]
        ));
        let lane = store
            .read_cold_lane::<RetargetMap>()
            .expect("retarget lane");
        let map = lane.get(target).expect("map kept");
        assert_eq!(map.joints, vec![Some(JointId::from_raw(0))]);
        assert_eq!(map.scales, vec![2.0]);
    }

    /// `set_skel_tree` refuses a root without `Skeleton`, then installs the
    /// cold def and a player whose leaves follow the clip nodes and whose
    /// parameters take their defaults.
    #[test]
    fn set_skel_tree_installs_tree_lanes() {
        use ornis_animation::{SkelTree, SkelTreePlayer, TreeChild, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip },
                TreeNodeDef::Blend1d {
                    axis: 0,
                    children: vec![TreeChild {
                        position: 0.0,
                        node: 0,
                    }],
                },
            ],
            params: vec![TreeParam {
                name: "walk".to_string(),
                default: 0.25,
            }],
        };
        {
            let bare = store.create_entity();
            assert!(
                !set_skel_tree(&mut store, bare, tree.clone()),
                "no Skeleton yet"
            );
        }
        assert!(set_skel_tree(&mut store, root, tree.clone()));
        assert_eq!(
            store.read_cold_lane::<SkelTree>().unwrap().get(root),
            Some(&tree)
        );
        let player = store
            .read_lane::<SkelTreePlayer>()
            .unwrap()
            .get(root)
            .unwrap()
            .clone();
        assert_eq!(player.params, vec![0.25]);
        assert_eq!(player.leaves.len(), 1);
        assert_eq!(player.leaves[0].clip, clip);
        assert!(player.leaves[0].playing, "leaves start playing");
    }

    /// [`set_skel_tree`] drops a [`SkelFade`] still in flight on the
    /// root: a fade cannot meaningfully cross into a tree and would tick
    /// there ignored.
    #[test]
    fn set_skel_tree_drops_an_inflight_fade() {
        use ornis_animation::{SkelFade, SkelTree, TreeNodeDef};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        store.insert(
            root,
            SkelFade {
                from: SkelPlayer {
                    clip,
                    time: Seconds::ZERO,
                    speed: Seconds::new(1.0),
                    weight: Clamped01::ONE,
                    playing: true,
                    looping: true,
                },
                duration: Seconds::new(0.2),
                elapsed: Seconds::ZERO,
            },
        );
        assert!(set_skel_tree(
            &mut store,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Clip { clip }],
                params: Vec::new(),
            }
        ));
        assert!(
            store
                .read_lane::<SkelFade>()
                .is_none_or(|fades| fades.get(root).is_none()),
            "the in-flight fade is dropped, not left ticking"
        );
    }

    /// `set_tree_param` writes by [`TreeParam::name`], refuses unknown
    /// names and roots without a tree.
    #[test]
    fn set_tree_param_addresses_by_name() {
        use ornis_animation::{SkelTree, SkelTreePlayer, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let root = layer_root(&mut store);
        assert!(
            !set_tree_param(&mut store, root, "walk", 1.0),
            "no tree yet"
        );
        assert!(set_skel_tree(
            &mut store,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Blend1d {
                    axis: 0,
                    children: Vec::new(),
                }],
                params: vec![TreeParam {
                    name: "walk".to_string(),
                    default: 0.0,
                }],
            }
        ));
        assert!(
            !set_tree_param(&mut store, root, "jump", 1.0),
            "unknown name"
        );
        assert!(set_tree_param(&mut store, root, "walk", 1.0));
        let player = store
            .read_lane::<SkelTreePlayer>()
            .unwrap()
            .get(root)
            .unwrap()
            .clone();
        assert_eq!(player.params, vec![1.0]);
    }

    /// [`set_skel_tree`] refuses uninstallable trees instead of
    /// bad-skinning the root every frame: an unknown leaf handle (the
    /// check [`fade_to_clip`] applies), no root node, and a non-finite
    /// parameter default.
    #[test]
    fn set_skel_tree_rejects_uninstallable_trees() {
        use ornis_animation::{SkelTree, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        let leaf = |clip| TreeNodeDef::Clip { clip };
        {
            let unknown = leaf(ClipId(store.create_entity()));
            let tree = SkelTree {
                nodes: vec![unknown],
                params: Vec::new(),
            };
            assert!(
                !set_skel_tree(&mut store, root, tree),
                "unknown leaf handle"
            );
        }
        {
            let tree = SkelTree {
                nodes: Vec::new(),
                params: Vec::new(),
            };
            assert!(!set_skel_tree(&mut store, root, tree), "no root node");
        }
        {
            let tree = SkelTree {
                nodes: vec![leaf(clip)],
                params: vec![TreeParam {
                    name: "walk".to_string(),
                    default: f32::NAN,
                }],
            };
            assert!(!set_skel_tree(&mut store, root, tree), "non-finite default");
        }
        {
            let tree = SkelTree {
                nodes: vec![leaf(clip)],
                params: Vec::new(),
            };
            assert!(
                set_skel_tree(&mut store, root, tree),
                "the sound tree installs"
            );
        }
    }

    /// [`set_skel_tree`] refuses broken blend nodes: an axis that names no
    /// parameter, a child not earlier than its parent, positions not
    /// strictly ascending, and a non-finite position.
    #[test]
    fn set_skel_tree_rejects_broken_blends() {
        use ornis_animation::{SkelTree, TreeChild, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        let leaf = || TreeNodeDef::Clip { clip };
        let blend = |axis, children| TreeNodeDef::Blend1d { axis, children };
        let param = TreeParam {
            name: "walk".to_string(),
            default: 0.0,
        };
        {
            let tree = SkelTree {
                nodes: vec![
                    leaf(),
                    blend(
                        1,
                        vec![TreeChild {
                            position: 0.0,
                            node: 0,
                        }],
                    ),
                ],
                params: vec![param.clone()],
            };
            assert!(!set_skel_tree(&mut store, root, tree), "axis names nothing");
        }
        {
            let tree = SkelTree {
                nodes: vec![
                    leaf(),
                    blend(
                        0,
                        vec![TreeChild {
                            position: 0.0,
                            node: 1,
                        }],
                    ),
                ],
                params: vec![param.clone()],
            };
            assert!(!set_skel_tree(&mut store, root, tree), "child not earlier");
        }
        {
            let tree = SkelTree {
                nodes: vec![
                    leaf(),
                    leaf(),
                    blend(
                        0,
                        vec![
                            TreeChild {
                                position: 1.0,
                                node: 0,
                            },
                            TreeChild {
                                position: 1.0,
                                node: 1,
                            },
                        ],
                    ),
                ],
                params: vec![param.clone()],
            };
            assert!(
                !set_skel_tree(&mut store, root, tree),
                "positions not strictly ascending"
            );
        }
        {
            let tree = SkelTree {
                nodes: vec![
                    leaf(),
                    blend(
                        0,
                        vec![TreeChild {
                            position: f32::INFINITY,
                            node: 0,
                        }],
                    ),
                ],
                params: vec![param],
            };
            assert!(
                !set_skel_tree(&mut store, root, tree),
                "non-finite position"
            );
        }
    }

    /// Builds a sound two-state machine tree: one leaf and the machine,
    /// one `walk` parameter.
    fn machine_tree(
        clip: ClipId,
        initial: usize,
        transitions: Vec<TreeTransition>,
    ) -> ornis_animation::SkelTree {
        ornis_animation::SkelTree {
            nodes: vec![
                ornis_animation::TreeNodeDef::Clip { clip },
                ornis_animation::TreeNodeDef::Clip { clip },
                ornis_animation::TreeNodeDef::Machine {
                    states: vec![0, 1],
                    initial,
                    transitions,
                },
            ],
            params: vec![TreeParam {
                name: "walk".to_string(),
                default: 0.0,
            }],
        }
    }

    /// One machine-`TreeNodeDef` factory so the tests name their shapes.
    fn machine_node(
        states: Vec<usize>,
        initial: usize,
        transitions: Vec<TreeTransition>,
    ) -> ornis_animation::TreeNodeDef {
        ornis_animation::TreeNodeDef::Machine {
            states,
            initial,
            transitions,
        }
    }

    /// `from -> to` while the `param` guard exceeds `value`.
    fn handoff(from: usize, to: usize, param: usize, value: f32, fade: f32) -> TreeTransition {
        TreeTransition {
            from,
            to,
            when: TreeCond {
                param,
                op: TreeCmp::Greater,
                value,
            },
            fade: Seconds::new(fade),
        }
    }

    /// Installs `machine` next to a leaf and reports the facade verdict.
    fn machine_install_result(
        machine: ornis_animation::TreeNodeDef,
        params: Vec<TreeParam>,
    ) -> bool {
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        set_skel_tree(
            &mut store,
            root,
            ornis_animation::SkelTree {
                nodes: vec![ornis_animation::TreeNodeDef::Clip { clip }, machine],
                params,
            },
        )
    }

    /// The `MachineCursor` `set_skel_tree` leaves on a machine root;
    /// `reset` first drives the cursor mid-transition and re-installs.
    fn machine_cursor_after(reset: bool) -> MachineCursor {
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        assert!(
            install_machine(&mut store, root, clip),
            "the machine installs"
        );
        if reset {
            fly_machine_cursor(&mut store, root);
            assert!(install_machine(&mut store, root, clip), "re-installs");
        }
        cursor_of(&store, root)
    }

    /// Installs the handoff machine tree of [`machine_tree`].
    fn install_machine(store: &mut SmartStore, root: Entity, clip: ClipId) -> bool {
        set_skel_tree(
            store,
            root,
            machine_tree(clip, 0, vec![handoff(0, 1, 0, 0.5, 1.0)]),
        )
    }

    /// Puts the machine cursor mid-transition by hand — the lane state
    /// that a re-install must overwrite.
    fn fly_machine_cursor(store: &mut SmartStore, root: Entity) {
        let mut players = store
            .write_lane::<SkelTreePlayer>()
            .expect("lane registered");
        players.get_mut(root).expect("player kept").machines[0] = MachineCursor {
            state: 1,
            from: Some(0),
            elapsed: Seconds::new(0.25),
            duration: Seconds::new(1.0),
        };
    }

    fn cursor_of(store: &SmartStore, root: Entity) -> MachineCursor {
        store
            .read_lane::<SkelTreePlayer>()
            .expect("lane registered")
            .get(root)
            .expect("player kept")
            .machines[0]
    }

    #[test]
    fn machine_with_empty_states_is_refused() {
        assert!(
            !machine_install_result(machine_node(Vec::new(), 0, Vec::new()), Vec::new()),
            "empty states"
        );
    }

    #[test]
    fn machine_with_initial_out_of_range_is_refused() {
        assert!(
            !machine_install_result(machine_node(vec![0, 0], 5, Vec::new()), Vec::new()),
            "initial out of range"
        );
    }

    #[test]
    fn machine_with_a_forward_state_is_refused() {
        assert!(
            !machine_install_result(machine_node(vec![1], 0, Vec::new()), Vec::new()),
            "state not earlier"
        );
    }

    #[test]
    fn machine_with_from_out_of_range_is_refused() {
        assert!(
            !machine_install_result(
                machine_node(vec![0], 0, vec![handoff(2, 0, 0, 0.0, 0.0)]),
                vec![walk_param()]
            ),
            "from out of range"
        );
    }

    #[test]
    fn machine_with_to_out_of_range_is_refused() {
        assert!(
            !machine_install_result(
                machine_node(
                    vec![0],
                    0,
                    vec![handoff(0, 0, 0, 0.0, 0.0), handoff(0, 2, 0, 0.0, 0.0)]
                ),
                vec![walk_param()]
            ),
            "to out of range"
        );
    }

    #[test]
    fn machine_with_an_unknown_guard_is_refused() {
        assert!(
            !machine_install_result(
                machine_node(vec![0], 0, vec![handoff(0, 0, 3, 0.0, 0.0)]),
                Vec::new()
            ),
            "unknown guard"
        );
    }

    #[test]
    fn machine_with_a_non_finite_guard_value_is_refused() {
        assert!(
            !machine_install_result(
                machine_node(vec![0], 0, vec![handoff(0, 0, 0, f32::NAN, 0.0)]),
                vec![walk_param()]
            ),
            "non-finite value"
        );
    }

    #[test]
    fn machine_with_a_non_finite_fade_is_refused() {
        assert!(
            !machine_install_result(
                machine_node(vec![0], 0, vec![handoff(0, 0, 0, 0.0, f32::NAN)]),
                vec![walk_param()]
            ),
            "non-finite fade"
        );
    }

    #[test]
    fn a_sound_machine_installs() {
        assert!(machine_install_result(
            machine_node(vec![0], 0, Vec::new()),
            Vec::new()
        ));
    }

    /// The `walk` guard parameter of the machine fixtures.
    fn walk_param() -> TreeParam {
        TreeParam {
            name: "walk".to_string(),
            default: 0.0,
        }
    }

    #[test]
    fn set_skel_tree_seeds_machine_cursors_at_initial() {
        assert_eq!(
            machine_cursor_after(false),
            MachineCursor {
                state: 0,
                from: None,
                elapsed: Seconds::ZERO,
                duration: Seconds::ZERO,
            }
        );
    }

    #[test]
    fn set_skel_tree_resets_a_machine_mid_transition() {
        assert_eq!(
            machine_cursor_after(true),
            MachineCursor {
                state: 0,
                from: None,
                elapsed: Seconds::ZERO,
                duration: Seconds::ZERO,
            }
        );
    }

    /// The `SwitchCursor` `set_skel_tree` seeds on a switch root: the
    /// standing child is the rounded default-parameter selection (§7.7).
    #[test]
    fn set_skel_tree_seeds_switch_cursors_at_default_selection() {
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        assert!(set_skel_tree(
            &mut store,
            root,
            ornis_animation::SkelTree {
                nodes: vec![
                    ornis_animation::TreeNodeDef::Clip { clip },
                    ornis_animation::TreeNodeDef::Clip { clip },
                    ornis_animation::TreeNodeDef::Switch {
                        param: 0,
                        children: vec![0, 1],
                        fade: Seconds::new(1.0),
                    },
                ],
                params: vec![TreeParam {
                    name: "pick".to_string(),
                    default: 0.6,
                }],
            },
        ));
        let cursor = {
            let players = store
                .read_lane::<SkelTreePlayer>()
                .expect("lane registered");
            players.get(root).expect("player kept").switches[0]
        };
        assert_eq!(
            cursor,
            SwitchCursor {
                child: 1,
                from: None,
                elapsed: Seconds::ZERO,
                duration: Seconds::ZERO,
            }
        );
    }

    /// One install-time [`FreeformCache`] per freeform node, in node
    /// order — two freeform nodes seed two caches (Ядро's r-39-fix
    /// request).
    #[test]
    fn set_skel_tree_seeds_one_cache_per_freeform_node() {
        use ornis_animation::{SkelTree, TreeChild2d, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        let cell = |x: f32, y: f32, node: usize| TreeChild2d {
            position: [x, y],
            node,
        };
        assert!(set_skel_tree(
            &mut store,
            root,
            SkelTree {
                nodes: vec![
                    TreeNodeDef::Clip { clip },
                    TreeNodeDef::Clip { clip },
                    TreeNodeDef::Clip { clip },
                    TreeNodeDef::Freeform2d {
                        axes: [0, 1],
                        children: vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 1), cell(0.0, 2.0, 2)],
                    },
                    TreeNodeDef::Freeform2d {
                        axes: [0, 1],
                        children: vec![cell(0.0, 0.0, 0), cell(4.0, 0.0, 1), cell(0.0, 4.0, 2)],
                    },
                ],
                params: vec![
                    TreeParam {
                        name: "x".to_string(),
                        default: 0.5,
                    },
                    TreeParam {
                        name: "y".to_string(),
                        default: 0.5,
                    },
                ],
            },
        ));
        let freeforms = {
            let players = store
                .read_lane::<SkelTreePlayer>()
                .expect("lane registered");
            players.get(root).expect("player kept").freeforms.clone()
        };
        assert_eq!(freeforms.len(), 2, "one cache per freeform node");
        for cache in &freeforms {
            assert!(!cache.triples().is_empty(), "a triangle survived");
        }
    }

    /// One sync mark for [`set_clip_sync_marks`] fixtures.
    fn sync_mark(name: &str, at: f32) -> SyncMark {
        SyncMark {
            name: name.to_string(),
            at,
        }
    }

    /// The `set_clip_sync_marks` verdict on `marks`: for the live clip of
    /// a tree fixture, and for a made-up clip id.
    fn marks_verdict(marks: Vec<SyncMark>) -> (bool, bool) {
        let mut store = SmartStore::new();
        let (_root, clip) = tree_hero(&mut store);
        (
            set_clip_sync_marks(&mut store, clip, marks.clone()),
            set_clip_sync_marks(&mut store, ClipId(Entity::new(999)), marks),
        )
    }

    #[test]
    fn set_clip_sync_marks_accepts_valid_marks() {
        assert_eq!(marks_verdict(vec![sync_mark("foot", 0.5)]), (true, false));
    }

    #[test]
    fn set_clip_sync_marks_rejects_an_empty_name() {
        assert_eq!(marks_verdict(vec![sync_mark("", 0.5)]), (false, false));
    }

    #[test]
    fn set_clip_sync_marks_rejects_an_out_of_range_at() {
        assert_eq!(marks_verdict(vec![sync_mark("foot", 1.5)]), (false, false));
    }

    /// Marks of `clip` in the cold lane — `None` when the lane holds
    /// none.
    fn lane_marks(store: &SmartStore, clip: ClipId) -> Option<Vec<SyncMark>> {
        store
            .read_cold_lane::<ClipSyncMarks>()
            .as_deref()
            .and_then(|lane| lane.get(clip.0))
            .map(|entry| entry.marks.clone())
            .filter(|marks| !marks.is_empty())
    }

    /// The `set_clip_sync_marks` verdict on `marks` and what the lane
    /// holds for the live clip afterwards.
    fn lane_after(marks: Vec<SyncMark>) -> (bool, Option<Vec<SyncMark>>) {
        let mut store = SmartStore::new();
        let (_root, clip) = tree_hero(&mut store);
        let set = set_clip_sync_marks(&mut store, clip, marks);
        (set, lane_marks(&store, clip))
    }

    /// The clear verdict of an empty list after `marks` were set, and
    /// the lane afterwards.
    fn lane_after_clear(marks: Vec<SyncMark>) -> (bool, Option<Vec<SyncMark>>) {
        let mut store = SmartStore::new();
        let (_root, clip) = tree_hero(&mut store);
        set_clip_sync_marks(&mut store, clip, marks);
        let cleared = set_clip_sync_marks(&mut store, clip, Vec::new());
        (cleared, lane_marks(&store, clip))
    }

    #[test]
    fn set_clip_sync_marks_stores_marks_in_the_lane() {
        let marks = vec![sync_mark("foot", 0.5)];
        assert_eq!(lane_after(marks.clone()), (true, Some(marks)));
    }

    #[test]
    fn an_empty_list_clears_the_lane() {
        assert_eq!(lane_after_clear(vec![sync_mark("foot", 0.5)]), (true, None));
    }

    /// One clip event for [`set_clip_events`] fixtures.
    fn clip_event(name: &str, at: f32) -> AnimEvent {
        AnimEvent {
            name: name.to_string(),
            at,
        }
    }

    /// The `set_clip_events` verdict on `events`: for the live clip of a
    /// tree fixture, and for a made-up clip id.
    fn events_verdict(events: Vec<AnimEvent>) -> (bool, bool) {
        let mut store = SmartStore::new();
        let (_root, clip) = tree_hero(&mut store);
        (
            set_clip_events(&mut store, clip, events.clone()),
            set_clip_events(&mut store, ClipId(Entity::new(999)), events),
        )
    }

    #[test]
    fn set_clip_events_accepts_valid_events() {
        assert_eq!(events_verdict(vec![clip_event("foot", 0.5)]), (true, false));
    }

    #[test]
    fn set_clip_events_rejects_an_empty_name() {
        assert_eq!(events_verdict(vec![clip_event("", 0.5)]), (false, false));
    }

    #[test]
    fn set_clip_events_rejects_an_out_of_range_at() {
        assert_eq!(
            events_verdict(vec![clip_event("foot", 1.5)]),
            (false, false)
        );
    }

    #[test]
    fn set_clip_events_stores_events_in_the_lane() {
        let mut store = SmartStore::new();
        let (_root, clip) = tree_hero(&mut store);
        let events = vec![clip_event("foot", 0.5)];
        assert!(set_clip_events(&mut store, clip, events.clone()));
        let lane = store
            .read_cold_lane::<ClipEvents>()
            .expect("events lane exists");
        assert_eq!(
            lane.get(clip.0).map(|entry| entry.events.clone()),
            Some(events)
        );
    }

    #[test]
    fn set_clip_events_also_accepts_object_clips() {
        let mut store = SmartStore::new();
        let clip_entity = store.create_entity();
        store.insert_cold(
            clip_entity,
            AnimClip {
                name: "spin".to_string(),
                duration: 1.0,
                looping: true,
                tracks: Vec::new(),
            },
        );
        let clip = ClipId(clip_entity);
        let tick = vec![clip_event("tick", 0.5)];
        assert!(set_clip_events(&mut store, clip, tick));
        let bad_tick = vec![clip_event("tick", 2.0)];
        assert!(!set_clip_events(&mut store, clip, bad_tick));
    }

    #[test]
    fn set_clip_sync_marks_rejects_a_non_finite_at() {
        assert_eq!(
            marks_verdict(vec![sync_mark("foot", f32::NAN)]),
            (false, false)
        );
    }

    #[test]
    fn set_tree_param_rejects_non_finite_values() {
        use ornis_animation::{SkelTree, SkelTreePlayer, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let root = layer_root(&mut store);
        assert!(set_skel_tree(
            &mut store,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Blend1d {
                    axis: 0,
                    children: Vec::new(),
                }],
                params: vec![TreeParam {
                    name: "walk".to_string(),
                    default: 0.5,
                }],
            }
        ));
        assert!(!set_tree_param(&mut store, root, "walk", f32::NAN));
        assert!(!set_tree_param(&mut store, root, "walk", f32::INFINITY));
        let player = store
            .read_lane::<SkelTreePlayer>()
            .unwrap()
            .get(root)
            .unwrap()
            .clone();
        assert_eq!(player.params, vec![0.5], "nothing is written on refusal");
    }

    /// [`clear_skel_tree`] takes the root back for single-player
    /// playback: both tree lanes are gone, and a second clear finds
    /// nothing.
    #[test]
    fn clear_skel_tree_takes_the_root_back() {
        use ornis_animation::{SkelTree, SkelTreePlayer, TreeNodeDef};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        assert!(!clear_skel_tree(&mut store, root), "no tree to remove");
        assert!(set_skel_tree(
            &mut store,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Clip { clip }],
                params: Vec::new(),
            }
        ));
        assert!(clear_skel_tree(&mut store, root));
        assert!(
            store
                .read_cold_lane::<SkelTree>()
                .is_none_or(|trees| trees.get(root).is_none())
        );
        assert!(
            store
                .read_lane::<SkelTreePlayer>()
                .is_none_or(|players| players.get(root).is_none())
        );
        assert!(!clear_skel_tree(&mut store, root), "already clear");
    }

    /// [`set_skel_tree`] refuses broken blend2d lattices — rows of
    /// different width, ragged x positions across rows, x not strictly
    /// ascending, y not strictly ascending — and accepts the complete
    /// one.
    #[test]
    fn set_skel_tree_rejects_broken_blend2d_lattices() {
        use ornis_animation::{SkelTree, TreeChild2d, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        let leaf = || TreeNodeDef::Clip { clip };
        let cell = |x: f32, y: f32, node: usize| TreeChild2d {
            position: [x, y],
            node,
        };
        let blend = |axes: [usize; 2], children| TreeNodeDef::Blend2d { axes, children };
        let params = vec![
            TreeParam {
                name: "x".to_string(),
                default: 0.0,
            },
            TreeParam {
                name: "y".to_string(),
                default: 0.0,
            },
        ];
        let tree = |axes: [usize; 2], children| SkelTree {
            nodes: vec![leaf(), leaf(), blend(axes, children)],
            params: params.clone(),
        };
        {
            let ragged = tree(
                [0, 1],
                vec![cell(0.0, 0.0, 0), cell(1.0, 0.0, 1), cell(0.0, 1.0, 1)],
            );
            assert!(!set_skel_tree(&mut store, root, ragged), "ragged rows");
        }
        {
            let mismatched = tree(
                [0, 1],
                vec![
                    cell(0.0, 0.0, 0),
                    cell(1.0, 0.0, 1),
                    cell(0.5, 1.0, 1),
                    cell(1.5, 1.0, 1),
                ],
            );
            assert!(
                !set_skel_tree(&mut store, root, mismatched),
                "x positions differ across rows"
            );
        }
        {
            let flat_x = tree(
                [0, 1],
                vec![
                    cell(1.0, 0.0, 0),
                    cell(1.0, 0.0, 1),
                    cell(1.0, 1.0, 1),
                    cell(1.0, 1.0, 1),
                ],
            );
            assert!(
                !set_skel_tree(&mut store, root, flat_x),
                "x not strictly ascending"
            );
        }
        {
            // Two full rows, y descending: the refusal must fire on the
            // per-row y check, not on x (an all-equal-y shape collapses
            // into one wide row and would refuse on x instead).
            let descending_y = tree(
                [0, 1],
                vec![
                    cell(0.0, 1.0, 0),
                    cell(1.0, 1.0, 1),
                    cell(0.0, 0.0, 1),
                    cell(1.0, 0.0, 1),
                ],
            );
            assert!(
                !set_skel_tree(&mut store, root, descending_y),
                "y not strictly ascending"
            );
        }
        {
            let same_axis = tree([0, 0], vec![cell(0.0, 0.0, 0), cell(1.0, 0.0, 1)]);
            assert!(
                !set_skel_tree(&mut store, root, same_axis),
                "both axes on one parameter"
            );
        }
        {
            let sound = tree(
                [0, 1],
                vec![
                    cell(0.0, 0.0, 0),
                    cell(1.0, 0.0, 1),
                    cell(0.0, 1.0, 1),
                    cell(1.0, 1.0, 1),
                ],
            );
            assert!(
                set_skel_tree(&mut store, root, sound),
                "the lattice installs"
            );
        }
    }

    /// [`set_skel_tree`] refuses broken `Freeform2d` nodes — equal
    /// axes, an axis outside the parameters, a repeated or non-finite
    /// point, more than [`FREEFORM_MAX_POINTS`] points, a child not
    /// earlier than its parent — and writes **nothing** on refusal: no
    /// `SkelTreePlayer` appears for the root.
    #[test]
    fn set_skel_tree_rejects_broken_freeform_nodes() {
        use ornis_animation::{SkelTree, TreeChild2d, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let (root, clip) = tree_hero(&mut store);
        let params = vec![
            TreeParam {
                name: "speed".to_string(),
                default: 0.0,
            },
            TreeParam {
                name: "dir".to_string(),
                default: 0.0,
            },
        ];
        let cell = |x: f32, y: f32, node: usize| TreeChild2d {
            position: [x, y],
            node,
        };
        let freeform = |axes: [usize; 2], children| TreeNodeDef::Freeform2d { axes, children };
        let tree = |axes: [usize; 2], children| SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip },
                TreeNodeDef::Clip { clip },
                TreeNodeDef::Clip { clip },
                freeform(axes, children),
            ],
            params: params.clone(),
        };
        let refuses = |store: &mut SmartStore, tree: SkelTree, why: &str| {
            assert!(!set_skel_tree(store, root, tree), "{why} accepted");
            assert!(
                store
                    .read_lane::<SkelTreePlayer>()
                    .is_none_or(|players| players.get(root).is_none()),
                "{why} wrote a player"
            );
        };
        refuses(
            &mut store,
            tree(
                [0, 0],
                vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 1), cell(0.0, 2.0, 2)],
            ),
            "equal axes",
        );
        refuses(
            &mut store,
            tree(
                [0, 2],
                vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 1), cell(0.0, 2.0, 2)],
            ),
            "axis out of range",
        );
        refuses(
            &mut store,
            tree(
                [0, 1],
                vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 1), cell(0.0, 0.0, 2)],
            ),
            "repeated point",
        );
        refuses(
            &mut store,
            tree(
                [0, 1],
                vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 1), cell(f32::NAN, 2.0, 2)],
            ),
            "non-finite point",
        );
        {
            let points = (0..FREEFORM_MAX_POINTS + 1)
                .map(|i| cell(i as f32, (i * i % 5) as f32, i))
                .collect::<Vec<_>>();
            refuses(
                &mut store,
                SkelTree {
                    nodes: std::iter::repeat_n(TreeNodeDef::Clip { clip }, FREEFORM_MAX_POINTS + 1)
                        .chain(std::iter::once(freeform([0, 1], points)))
                        .collect(),
                    params: params.clone(),
                },
                "more than FREEFORM_MAX_POINTS points",
            );
        }
        refuses(
            &mut store,
            tree(
                [0, 1],
                vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 3), cell(0.0, 2.0, 2)],
            ),
            "child not earlier than parent",
        );
        {
            let sound = tree(
                [0, 1],
                vec![cell(0.0, 0.0, 0), cell(2.0, 0.0, 1), cell(0.0, 2.0, 2)],
            );
            assert!(
                set_skel_tree(&mut store, root, sound),
                "the sound tree installs"
            );
        }
    }

    /// The playback entries refuse a tree-driven root with
    /// [`PlaybackError::TreeDriven`] — no silent no-op — and
    /// [`clear_skel_tree`] hands the root back to them.
    #[test]
    fn playback_entries_error_on_tree_root() {
        use ornis_animation::{SkelTree, TreeNodeDef, TreeParam};
        let mut store = SmartStore::new();
        let root = layer_root(&mut store);
        let playlist = store.create_entity();
        assert!(set_skel_tree(
            &mut store,
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Blend1d {
                    axis: 0,
                    children: Vec::new(),
                }],
                params: vec![TreeParam {
                    name: "walk".to_string(),
                    default: 0.0,
                }],
            }
        ));
        let fade = fade_to_clip(&mut store, root, ClipId(playlist), Seconds::new(0.1));
        assert_eq!(fade, Err(PlaybackError::TreeDriven), "fade_to_clip refuses");
        let play = play_crossfade(&mut store, root, "walk", Seconds::new(0.1));
        assert_eq!(
            play,
            Err(PlaybackError::TreeDriven),
            "play_crossfade refuses"
        );
        assert!(clear_skel_tree(&mut store, root));
        let after = fade_to_clip(&mut store, root, ClipId(playlist), Seconds::new(0.1));
        assert_eq!(
            after,
            Err(PlaybackError::Missing),
            "the player verdict is back"
        );
    }
}
