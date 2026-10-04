//! Pure `ornis-gltf` mirror → `ornis-animation` converters, plus the
//! shared spawn/wire routines built on them: [`spawn_gltf_world`]
//! (mesh entities from a loaded scene) and [`wire_loaded_animation`]
//! (skeleton roots, named clip playlists, a character-root [`Animator`],
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
    AnimClip, AnimPlayer, AnimTrack, Animator, ClipId, JointId, JointPose, JointTrack, Key,
    KeyTrack, SkelClip, SkelPlayer, Skeleton, SkinnedMesh,
};
use ornis_assets::scene::{MaterialDesc, MeshDesc, TransformDesc};
use ornis_core::{Entity, Seconds, SmartStore};
use ornis_gltf::{
    LoadedAnimClip, LoadedInterpolation, LoadedJointTrack, LoadedKeyTrack, LoadedMesh,
    LoadedSkelClip, LoadedSkin,
};

use super::session::Name;

/// Converts a skeletal mirror clip to a live [`SkelClip`].
///
/// Duration and joint indices copy verbatim; every channel maps key-for-key
/// with interpolation preserved. Rotation arrays are `(x, y, z, w)` unit
/// quaternions on the mirror side (normalized at import) and convert with
/// [`Quat::from_xyzw`] exactly — no silent renormalization here.
pub fn skel_clip_from_loaded(clip: &LoadedSkelClip) -> SkelClip {
    SkelClip {
        name: clip.name.clone(),
        duration: clip.duration,
        tracks: clip.tracks.iter().map(joint_track_from_loaded).collect(),
    }
}

/// Converts an object mirror clip to a live [`AnimClip`], resolving loader-local
/// node handles through `node_to_entity` to world entities.
///
/// Returns the clip plus the count of tracks dropped for lack of a mapping.
/// The clip name comes from the caller (the mirror name is not consumed), and
/// `duration`/`looping` copy the mirror verbatim.
pub fn anim_clip_from_loaded(
    name: String,
    clip: &LoadedAnimClip,
    node_to_entity: &HashMap<Entity, Entity>,
) -> (AnimClip, usize) {
    let mut dropped = 0usize;
    let mut tracks = Vec::with_capacity(clip.tracks.len());
    for track in &clip.tracks {
        let Some(&entity) = node_to_entity.get(&track.entity) else {
            dropped += 1;
            continue;
        };
        tracks.push(AnimTrack {
            entity,
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

/// Converts a mirror skin to a live [`Skeleton`].
///
/// Parents map `-1` → root (`None`) via [`JointId::from_raw_opt`]; bind
/// inverses are glTF column-major (`m[col][row]`) and convert with
/// [`Mat4::from_cols_array_2d`]; names copy verbatim (diagnostics only).
pub fn skeleton_from_loaded(skin: &LoadedSkin) -> Skeleton {
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

/// Maps one mirror joint track (joint index plus three channels).
fn joint_track_from_loaded(track: &LoadedJointTrack) -> JointTrack {
    JointTrack {
        joint: JointId::from_raw(track.joint),
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

/// World entities spawned from one [`LoadedScene`](ornis_gltf::LoadedScene):
/// mesh entities in load order plus the loader-local node → world map
/// animation wiring resolves through.
pub struct GltfSpawn {
    /// Mesh entities, 1:1 with the loaded entities, in load order.
    pub entities: Vec<Entity>,
    /// Loader-local node handle (`Entity::new(node)`) → world entity.
    /// First mesh entity wins per node.
    pub node_to_entity: std::collections::HashMap<Entity, Entity>,
    /// Character root that receives [`Animator`].
    ///
    /// `None` (what [`spawn_gltf_world`] leaves) tells
    /// [`wire_loaded_animation`] to fall back to the first mesh entity.
    /// Set this before wiring when the real scene root is a different
    /// entity — that is the hook for `GameWorld::spawn_scene` (Core track).
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
    /// Scene spawn (`GameWorld::spawn_scene`, Core track) should pass the
    /// character root. Skeleton roots are not given a [`SkelPlayer`] until
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

/// Spawns the mesh entities of a loaded glTF scene: `TransformDesc` +
/// `MeshDesc::Custom` + `MaterialDesc` per primitive, no physics bodies
/// (pure visual spawn — the solver never sees these entities).
///
/// Lanes are registered idempotently; entities follow load order so the
/// caller can zip them back against the loaded entities.
pub fn spawn_gltf_world(store: &mut SmartStore, loaded: &ornis_gltf::LoadedScene) -> GltfSpawn {
    store.register::<TransformDesc>();
    store.register::<MeshDesc>();
    store.register::<MaterialDesc>();
    let scene = ornis_assets::import::scene_from_gltf(loaded);
    let mut entities = Vec::with_capacity(scene.entities.len());
    let mut node_to_entity = std::collections::HashMap::new();
    for (desc, loaded_entity) in scene.entities.iter().zip(loaded.entities.iter()) {
        let entity = store.create_entity();
        store.insert(entity, desc.transform.clone());
        store.insert(entity, desc.mesh.clone());
        store.insert(entity, desc.material.clone());
        node_to_entity
            .entry(Entity::new(loaded_entity.node))
            .or_insert(entity);
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
/// until `GameWorld::spawn_scene` returns the character root.
///
/// `spawn.entities` must be the load-order mesh entities (see
/// [`spawn_gltf_world`]); `spawn.node_to_entity` resolves loader-local
/// node handles to them. Sources without skins or clips wire nothing.
/// Lanes are registered idempotently.
pub fn wire_loaded_animation(
    store: &mut SmartStore,
    loaded: &ornis_gltf::LoadedScene,
    spawn: &GltfSpawn,
) -> AnimationWiring {
    use ornis_animation::{AnimClip, SkelClip};
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register::<SkinnedMesh>();
    store.register::<SkelPlayer>();
    store.register_cold::<SkelClip>();
    store.register::<AnimPlayer>();
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
        let live = skel_clip_from_loaded(clip);
        let playlist = store.create_entity();
        store.insert(playlist, Name(name.clone()));
        store.insert_cold(playlist, live);
        wiring.skel_playlists.push(playlist);
        wiring.clips.entry(name).or_insert(ClipId(playlist));
    }
    // Skinned meshes: `SkinnedMesh.skeleton` points at the skin's root.
    // `None` (unskinned primitive) keeps the regular `Custom` mesh.
    for (desc, entity) in loaded.entities.iter().zip(spawn.entities.iter()) {
        let Some(skin) = desc.skin else {
            continue;
        };
        let Some(&root) = wiring.roots.get(skin) else {
            continue;
        };
        let Some(skinned) = skinned_mesh_from_loaded(root, &desc.mesh) else {
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

    /// Minimal unskinned entity mirror (node index drives the map).
    fn entity_fixture(node: u32) -> ornis_gltf::LoadedEntity {
        ornis_gltf::LoadedEntity {
            name: format!("part_{node}"),
            node,
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            mesh: ornis_gltf::LoadedMesh {
                positions: vec![[0.0, 0.0, 0.0]],
                indices: vec![0],
                normals: None,
                uvs: None,
                joints: None,
                weights: None,
            },
            skin: None,
            material: ornis_gltf::LoadedMaterial {
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                emission: [0.0, 0.0, 0.0],
                base_color_texture: None,
                metallic_roughness_texture: None,
                emissive_texture: None,
            },
        }
    }

    /// Spawn maps loader-local node handles to world entities, 1:1 and in
    /// load order, with placement components inserted.
    #[test]
    fn spawn_maps_nodes_to_world_entities() {
        use ornis_assets::scene::TransformDesc;
        let loaded = ornis_gltf::LoadedScene {
            name: "two".to_string(),
            entities: vec![entity_fixture(0), entity_fixture(7)],
            skins: Vec::new(),
            skel_clips: Vec::new(),
            anim_clips: Vec::new(),
            stats: ornis_gltf::ImportStats::default(),
        };
        let mut store = SmartStore::new();
        let spawn = spawn_gltf_world(&mut store, &loaded);
        assert_eq!(spawn.entities.len(), 2);
        assert_eq!(spawn.node_to_entity[&Entity::new(0)], spawn.entities[0]);
        assert_eq!(spawn.node_to_entity[&Entity::new(7)], spawn.entities[1]);
        for entity in &spawn.entities {
            let lane = store.read_lane::<TransformDesc>().unwrap();
            assert!(lane.get(*entity).is_some());
        }
    }

    /// Mirror joint track with one linear key per channel.
    fn joint_fixture() -> LoadedJointTrack {
        LoadedJointTrack {
            joint: 3,
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
        let live = skel_clip_from_loaded(&clip);
        assert_eq!(live.duration, 2.5);
        assert_eq!(live.tracks.len(), 1);
        let track = &live.tracks[0];
        assert_eq!(track.joint, JointId::from_raw(3));
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
    fn stepped_interpolation_survives_on_all_channels() {
        let clip = LoadedSkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![LoadedJointTrack {
                joint: 0,
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
        let live = skel_clip_from_loaded(&clip);
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
        let live = skel_clip_from_loaded(&clip);
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
        let loader = Entity::new(0);
        let world = Entity::new(41);
        let clip = LoadedAnimClip {
            name: "cubic".to_string(),
            duration: 1.0,
            looping: true,
            tracks: vec![LoadedAnimTrack {
                entity: loader,
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
                translation: LoadedKeyTrack::linear(Vec::new()),
                rotation: LoadedKeyTrack::linear(vec![LoadedKey {
                    time: 1.0,
                    value: [0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2],
                }]),
                scale: LoadedKeyTrack::linear(Vec::new()),
            }],
        };
        let live = skel_clip_from_loaded(&clip);
        let got = live.tracks[0].rotation.keys[0].value;
        let want = Quat::from_xyzw(0.0, FRAC_1_SQRT_2, 0.0, FRAC_1_SQRT_2);
        assert_eq!(got, want);
    }

    #[test]
    fn anim_clip_applies_node_mapping_and_counts_dropped() {
        let loader_a = Entity::new(0);
        let loader_b = Entity::new(1);
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
            tracks: vec![object_track(Entity::new(7))],
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
        };
        let live = skeleton_from_loaded(&skin);
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
        let live_skel = skel_clip_from_loaded(&skel);
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
    fn object_track(entity: Entity) -> LoadedAnimTrack {
        LoadedAnimTrack {
            entity,
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

    fn skinned_scene(clips: Vec<LoadedSkelClip>) -> ornis_gltf::LoadedScene {
        let mut mesh = entity_fixture(0);
        mesh.skin = Some(0);
        ornis_gltf::LoadedScene {
            name: "character".into(),
            entities: vec![mesh],
            skins: vec![LoadedSkin {
                parents: vec![-1],
                inverse_bind: vec![[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]],
                joint_names: vec!["root".into()],
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
}
