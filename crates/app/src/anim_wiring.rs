//! Pure `ornis-gltf` mirror → `ornis-animation` converters.
//!
//! Field-for-field mapping with no I/O, no ECS access, and no allocation
//! beyond the output clips: joint indices pass through
//! [`JointId::from_raw`](ornis_animation::JointId), rotations arrive as
//! `(x, y, z, w)` unit quaternions via [`Quat::from_xyzw`](glam::Quat), and
//! object tracks resolve through the caller-supplied node→entity map
//! (unmapped tracks are dropped and counted, never stubbed).

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    AnimClip, AnimTrack, JointId, JointTrack, Key, KeyTrack, SkelClip, Skeleton, SkinnedMesh,
};
use ornis_core::Entity;
use ornis_gltf::{
    LoadedAnimClip, LoadedInterpolation, LoadedJointTrack, LoadedKeyTrack, LoadedMesh,
    LoadedSkelClip, LoadedSkin,
};

/// Converts a skeletal mirror clip to a live [`SkelClip`].
///
/// Duration and joint indices copy verbatim; every channel maps key-for-key
/// with interpolation preserved. Rotation arrays are `(x, y, z, w)` unit
/// quaternions on the mirror side (normalized at import) and convert with
/// [`Quat::from_xyzw`] exactly — no silent renormalization here.
pub fn skel_clip_from_loaded(clip: &LoadedSkelClip) -> SkelClip {
    SkelClip {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_animation::Interpolation;
    use ornis_gltf::{LoadedAnimTrack, LoadedKey};

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
}
