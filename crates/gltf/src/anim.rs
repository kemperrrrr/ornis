//! Animation-clip assembly from glTF `animations[]` (design §4).
//!
//! Consumes sampler `input` times plus `output` TRS values per
//! `channel.target.path`; routes skinned nodes (joints of the imported
//! skins) to joint tracks and the rest to object tracks. `LINEAR` maps to
//! [`LoadedKeyTrack::linear`], `STEP` to [`LoadedKeyTrack::stepped`];
//! `CUBICSPLINE` channels skip honestly with a counter, morph-target
//! channels are ignored (documented in the crate skip-rules table).
//!
//! Clip containers mirror `ornis-animation` field-for-field (plain arrays,
//! no `glam` dependency — the same seam as [`LoadedSkin`](crate::LoadedSkin),
//! whose wiring feeds the animation builders): joint indices map through
//! `JointId::from_raw`, rotations are `(x, y, z, w)` unit quaternions,
//! object entities arrive as [`Entity`] handles keyed by node index.

use std::collections::HashMap;

use gltf::Document;
use gltf::animation::Property;
use gltf::animation::util::ReadOutputs;
use ornis_core::Entity;

use crate::ImportStats;

/// Squared length below which a quaternion is treated as degenerate.
const DEGENERATE_LEN2: f32 = 1e-12;

/// How to blend between the keys of a [`LoadedKeyTrack`].
///
/// Mirrors `ornis-animation` `Interpolation` field-for-field; the wiring
/// maps variants one-to-one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadedInterpolation {
    /// Blend between surrounding keys (`lerp`/`slerp`).
    Linear,
    /// Hold the earlier key (glTF `STEP`).
    Step,
}

/// One animation key: the channel value at `time` seconds from clip start.
///
/// Mirrors `ornis-animation` `Key`: keys inside a [`LoadedKeyTrack`] are
/// sorted by ascending `time` (input order is preserved verbatim).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoadedKey<T> {
    /// Seconds from the clip start.
    pub time: f32,
    /// Channel value at `time` (`[f32; 3]` for translation/scale,
    /// `(x, y, z, w)` unit quaternion for rotation).
    pub value: T,
}

/// One sorted channel of keys (translation, rotation or scale).
///
/// Mirrors `ornis-animation` `KeyTrack`: an empty track on a joint channel
/// means identity, on an object channel it means "leave untouched".
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedKeyTrack<T> {
    /// Keys sorted by ascending `time`.
    pub keys: Vec<LoadedKey<T>>,
    /// How to blend between keys.
    pub interpolation: LoadedInterpolation,
}

impl<T> LoadedKeyTrack<T> {
    /// Builds a linearly blended track.
    pub fn linear(keys: Vec<LoadedKey<T>>) -> Self {
        Self {
            keys,
            interpolation: LoadedInterpolation::Linear,
        }
    }

    /// Builds a hold-previous track (glTF `STEP`).
    pub fn stepped(keys: Vec<LoadedKey<T>>) -> Self {
        Self {
            keys,
            interpolation: LoadedInterpolation::Step,
        }
    }
}

/// One joint channel of a [`LoadedSkelClip`]: sorted keys per transform lane.
///
/// Mirrors `ornis-animation` `JointTrack` field-for-field: `joint` maps
/// through `JointId::from_raw`, empty lanes read as identity (zero
/// translation, unit rotation, unit scale) so untracked joints hold the
/// bind offset instead of collapsing the chain.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedJointTrack {
    /// Animated joint index into the skin order (see [`node_to_joint_map`]).
    pub joint: u32,
    /// Translation keys; empty means zero translation.
    pub translation: LoadedKeyTrack<[f32; 3]>,
    /// Rotation keys (`(x, y, z, w)` unit quaternions); empty means identity.
    pub rotation: LoadedKeyTrack<[f32; 4]>,
    /// Scale keys; empty means unit scale.
    pub scale: LoadedKeyTrack<[f32; 3]>,
}

/// Skeletal animation clip: duration plus per-joint tracks.
///
/// Mirrors `ornis-animation` `SkelClip` (which has no name lane either);
/// time wraps past `duration` by contract.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedSkelClip {
    /// Clip length in seconds (maximum input time over assembled keys).
    pub duration: f32,
    /// Per-joint tracks; first track per joint wins on duplicates.
    pub tracks: Vec<LoadedJointTrack>,
}

/// Object-space tracks for one animated node inside a [`LoadedAnimClip`].
///
/// Mirrors `ornis-animation` `AnimTrack`: `entity` is the loader-local
/// node handle ([`Entity::new`] keyed by node index — the node→entity
/// mapping is consumed here and never leaks raw indices); empty lanes mean
/// "leave untouched" (so entity size survives when no scale track exists).
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedAnimTrack {
    /// Animated node as a loader-local entity handle.
    pub entity: Entity,
    /// Translation keys; empty means "leave translation untouched".
    pub translation: LoadedKeyTrack<[f32; 3]>,
    /// Rotation keys; empty means "leave rotation untouched".
    pub rotation: LoadedKeyTrack<[f32; 4]>,
    /// Scale keys; empty means "leave scale untouched".
    pub scale: LoadedKeyTrack<[f32; 3]>,
}

/// Shareable object-animation clip: name, duration, loop flag plus tracks.
///
/// Mirrors `ornis-animation` `AnimClip` field-for-field; the loader always
/// sets `looping` (clips loop, players gate playback).
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedAnimClip {
    /// Human-readable label: glTF animation name, else `clip_{index}`.
    pub name: String,
    /// Clip length in seconds (maximum input time over assembled keys).
    pub duration: f32,
    /// Whether playback wraps (`true`: the loader convention).
    pub looping: bool,
    /// Per-node tracks in node-index order.
    pub tracks: Vec<LoadedAnimTrack>,
}

/// Builds the node→joint map from the document skins (first skin wins).
///
/// Mirrors the skin import source data (`skin.joints()` order): nodes that
/// are joints of a skin map to their joint position, everything else
/// animates as an object node. Skins without joints contribute nothing.
pub fn node_to_joint_map(document: &Document) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for skin in document.skins() {
        for (position, node) in skin.joints().enumerate() {
            map.entry(node.index()).or_insert(position);
        }
    }
    map
}

/// Assembles one [`LoadedSkelClip`] plus one [`LoadedAnimClip`] per glTF
/// animation.
///
/// `node_to_joint` comes from [`node_to_joint_map`]: channels targeting
/// those nodes become [`LoadedJointTrack`]s, the rest become
/// [`LoadedAnimTrack`]s — the node→entity/joint mapping is consumed here
/// and never leaves the loader.
///
/// Each clip `duration` is the maximum input time over its assembled keys.
/// A side with no tracks emits nothing; an animation with neither side
/// bumps [`ImportStats::skipped_clips`]. `CUBICSPLINE` channels bump
/// [`ImportStats::skipped_cubicspline`]; morph-target channels are ignored.
/// Malformed channels (unreadable or count-mismatched accessors) skip
/// silently. Duplicate channels for the same node and path keep the first.
pub fn assemble_clips(
    document: &Document,
    buffers: &[Vec<u8>],
    node_to_joint: &HashMap<usize, usize>,
    stats: &mut ImportStats,
) -> (Vec<LoadedSkelClip>, Vec<LoadedAnimClip>) {
    let mut skel_clips = Vec::new();
    let mut anim_clips = Vec::new();
    for (index, animation) in document.animations().enumerate() {
        let name = animation
            .name()
            .map(str::to_string)
            .unwrap_or_else(|| format!("clip_{index}"));
        let mut assembly = ClipAssembly::default();
        for channel in animation.channels() {
            if matches!(channel.target().property(), Property::MorphTargetWeights) {
                continue;
            }
            let node = channel.target().node().index();
            let Some(keys) = read_channel(&channel, buffers, stats) else {
                continue;
            };
            assembly.push(node, keys, node_to_joint.contains_key(&node));
        }
        let (joint_tracks, object_tracks) = assembly.into_tracks(node_to_joint);
        let duration = track_duration(&joint_tracks, &object_tracks);
        let mut emitted = false;
        if !joint_tracks.is_empty() {
            skel_clips.push(LoadedSkelClip {
                duration,
                tracks: joint_tracks,
            });
            emitted = true;
        }
        if !object_tracks.is_empty() {
            anim_clips.push(LoadedAnimClip {
                name,
                duration,
                looping: true,
                tracks: object_tracks,
            });
            emitted = true;
        }
        if !emitted {
            stats.skipped_clips += 1;
        }
    }
    (skel_clips, anim_clips)
}

/// Per-animation accumulator: at most one key track per node and path.
#[derive(Debug, Default)]
struct ClipAssembly {
    /// Node index → partial joint channels (only for skinned nodes).
    joints: HashMap<usize, JointPart>,
    /// Node index → partial object channels (only for unskinned nodes).
    entities: HashMap<usize, ObjectPart>,
}

/// Partial joint channels for one node (each path filled at most once).
#[derive(Debug, Default)]
struct JointPart {
    /// Translation keys (`None` = path not seen).
    translation: Option<LoadedKeyTrack<[f32; 3]>>,
    /// Rotation keys (`None` = path not seen).
    rotation: Option<LoadedKeyTrack<[f32; 4]>>,
    /// Scale keys (`None` = path not seen).
    scale: Option<LoadedKeyTrack<[f32; 3]>>,
}

/// Partial object channels for one node (each path filled at most once).
#[derive(Debug, Default)]
struct ObjectPart {
    /// Translation keys (`None` = path not seen).
    translation: Option<LoadedKeyTrack<[f32; 3]>>,
    /// Rotation keys (`None` = path not seen).
    rotation: Option<LoadedKeyTrack<[f32; 4]>>,
    /// Scale keys (`None` = path not seen).
    scale: Option<LoadedKeyTrack<[f32; 3]>>,
}

/// One decoded channel, dispatched by target path.
#[derive(Debug)]
enum ChannelKeys {
    /// Translation keys.
    Translation(LoadedKeyTrack<[f32; 3]>),
    /// Rotation keys.
    Rotation(LoadedKeyTrack<[f32; 4]>),
    /// Scale keys.
    Scale(LoadedKeyTrack<[f32; 3]>),
}

impl ClipAssembly {
    /// Pushes one decoded channel onto the routed side; duplicates for the
    /// same node and path keep the first (glTF forbids them, the loader
    /// stays panic-free regardless).
    fn push(&mut self, node: usize, keys: ChannelKeys, is_joint: bool) {
        match keys {
            ChannelKeys::Translation(track) => {
                if is_joint {
                    self.joints
                        .entry(node)
                        .or_default()
                        .translation
                        .get_or_insert(track);
                } else {
                    self.entities
                        .entry(node)
                        .or_default()
                        .translation
                        .get_or_insert(track);
                }
            }
            ChannelKeys::Rotation(track) => {
                if is_joint {
                    self.joints
                        .entry(node)
                        .or_default()
                        .rotation
                        .get_or_insert(track);
                } else {
                    self.entities
                        .entry(node)
                        .or_default()
                        .rotation
                        .get_or_insert(track);
                }
            }
            ChannelKeys::Scale(track) => {
                if is_joint {
                    self.joints
                        .entry(node)
                        .or_default()
                        .scale
                        .get_or_insert(track);
                } else {
                    self.entities
                        .entry(node)
                        .or_default()
                        .scale
                        .get_or_insert(track);
                }
            }
        }
    }

    /// Splits staged parts into joint and object tracks (both in ascending
    /// node order; joint indices resolve through `node_to_joint`).
    fn into_tracks(
        self,
        node_to_joint: &HashMap<usize, usize>,
    ) -> (Vec<LoadedJointTrack>, Vec<LoadedAnimTrack>) {
        let mut staged: Vec<(usize, JointPart)> = self.joints.into_iter().collect();
        staged.sort_by_key(|(node, _)| *node);
        let mut joints = Vec::with_capacity(staged.len());
        for (node, part) in staged {
            let Some(&joint) = node_to_joint.get(&node) else {
                continue;
            };
            joints.push(LoadedJointTrack {
                // Empty joint channel = identity (NOT "untouched" like
                // object tracks): absent paths stay empty, and the wiring
                // reads them as zero translation / unit rotation / unit
                // scale instead of collapsing the chain.
                joint: joint as u32,
                translation: part
                    .translation
                    .unwrap_or_else(|| LoadedKeyTrack::linear(Vec::new())),
                rotation: part
                    .rotation
                    .unwrap_or_else(|| LoadedKeyTrack::linear(Vec::new())),
                scale: part
                    .scale
                    .unwrap_or_else(|| LoadedKeyTrack::linear(Vec::new())),
            });
        }
        let mut staged: Vec<(usize, ObjectPart)> = self.entities.into_iter().collect();
        staged.sort_by_key(|(node, _)| *node);
        let mut entities = Vec::with_capacity(staged.len());
        for (node, part) in staged {
            // Empty object channel = "leave untouched" (design §1.1): absent
            // paths stay empty tracks via the constructors below.
            entities.push(LoadedAnimTrack {
                entity: Entity::new(node as u32),
                translation: part
                    .translation
                    .unwrap_or_else(|| LoadedKeyTrack::linear(Vec::new())),
                rotation: part
                    .rotation
                    .unwrap_or_else(|| LoadedKeyTrack::linear(Vec::new())),
                scale: part
                    .scale
                    .unwrap_or_else(|| LoadedKeyTrack::linear(Vec::new())),
            });
        }
        (joints, entities)
    }
}

/// Maximum key time over both track lists (`0.0` when all are empty).
fn track_duration(joints: &[LoadedJointTrack], objects: &[LoadedAnimTrack]) -> f32 {
    let mut max = 0.0f32;
    for track in joints {
        max = max.max(track_max(&track.translation.keys));
        max = max.max(track_max(&track.rotation.keys));
        max = max.max(track_max(&track.scale.keys));
    }
    for track in objects {
        max = max.max(track_max(&track.translation.keys));
        max = max.max(track_max(&track.rotation.keys));
        max = max.max(track_max(&track.scale.keys));
    }
    max
}

/// Maximum `time` over one key list (`0.0` when empty).
fn track_max<T>(keys: &[LoadedKey<T>]) -> f32 {
    keys.iter().map(|key| key.time).fold(0.0f32, f32::max)
}

/// Decodes one channel to path-dispatched keys (`None` = skip honestly).
///
/// `CUBICSPLINE` bumps [`ImportStats::skipped_cubicspline`]; unreadable or
/// count-mismatched accessors return `None`. Rotation outputs decode
/// through `into_f32` (normalized integers included) and normalize to unit
/// length, falling back to identity on degenerate input — never NaN.
fn read_channel(
    channel: &gltf::animation::Channel<'_>,
    buffers: &[Vec<u8>],
    stats: &mut ImportStats,
) -> Option<ChannelKeys> {
    // Accessor indices come from `Gltf::from_slice` validation, so the
    // `property()` / `interpolation()` / `node()` unwraps below only fire
    // on hand-built nonsense that validation already rejected as `Parse`.
    let sampler = channel.sampler();
    if matches!(
        sampler.interpolation(),
        gltf::animation::Interpolation::CubicSpline
    ) {
        stats.skipped_cubicspline += 1;
        return None;
    }
    let stepped = matches!(
        sampler.interpolation(),
        gltf::animation::Interpolation::Step
    );
    let reader = channel.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
    let times: Vec<f32> = reader.read_inputs()?.collect();
    if times.is_empty() {
        return None;
    }
    match channel.target().property() {
        Property::Translation => {
            let ReadOutputs::Translations(outputs) = reader.read_outputs()? else {
                return None;
            };
            let values: Vec<[f32; 3]> = outputs.collect();
            check_counts(times.len(), values.len())?;
            let keys = times
                .into_iter()
                .zip(values)
                .map(|(time, value)| LoadedKey { time, value })
                .collect();
            Some(ChannelKeys::Translation(make_vec_track(keys, stepped)))
        }
        Property::Scale => {
            let ReadOutputs::Scales(outputs) = reader.read_outputs()? else {
                return None;
            };
            let values: Vec<[f32; 3]> = outputs.collect();
            check_counts(times.len(), values.len())?;
            let keys = times
                .into_iter()
                .zip(values)
                .map(|(time, value)| LoadedKey { time, value })
                .collect();
            Some(ChannelKeys::Scale(make_vec_track(keys, stepped)))
        }
        Property::Rotation => {
            if !valid_rotation_output(&sampler.output()) {
                return None;
            }
            let ReadOutputs::Rotations(outputs) = reader.read_outputs()? else {
                return None;
            };
            let values: Vec<[f32; 4]> = outputs.into_f32().collect();
            check_counts(times.len(), values.len())?;
            let keys = times
                .into_iter()
                .zip(values)
                .map(|(time, value)| LoadedKey {
                    time,
                    value: normalize_quat(value),
                })
                .collect();
            Some(ChannelKeys::Rotation(make_quat_track(keys, stepped)))
        }
        Property::MorphTargetWeights => None,
    }
}

/// Width guard before `read_outputs`: rotation storage outside the decodable
/// set would hit the backend `unreachable!()`, so a hostile file skips here.
fn valid_rotation_output(output: &gltf::accessor::Accessor<'_>) -> bool {
    use gltf::accessor::DataType;
    matches!(
        output.data_type(),
        DataType::I8 | DataType::U8 | DataType::I16 | DataType::U16 | DataType::F32
    )
}

/// Input/output count agreement; `None` (skip) on mismatch or empty output.
fn check_counts(inputs: usize, outputs: usize) -> Option<()> {
    if outputs == 0 || inputs != outputs {
        None
    } else {
        Some(())
    }
}

/// Builds a translation/scale track through the constructors only.
fn make_vec_track(keys: Vec<LoadedKey<[f32; 3]>>, stepped: bool) -> LoadedKeyTrack<[f32; 3]> {
    if stepped {
        LoadedKeyTrack::stepped(keys)
    } else {
        LoadedKeyTrack::linear(keys)
    }
}

/// Builds a rotation track through the constructors only.
fn make_quat_track(keys: Vec<LoadedKey<[f32; 4]>>, stepped: bool) -> LoadedKeyTrack<[f32; 4]> {
    if stepped {
        LoadedKeyTrack::stepped(keys)
    } else {
        LoadedKeyTrack::linear(keys)
    }
}

/// Normalizes a raw `(x, y, z, w)` quaternion, identity on degenerate input.
fn normalize_quat(raw: [f32; 4]) -> [f32; 4] {
    let length_squared: f32 = raw.iter().map(|component| component * component).sum();
    if length_squared.is_finite() && length_squared > DEGENERATE_LEN2 {
        let length = length_squared.sqrt();
        [
            raw[0] / length,
            raw[1] / length,
            raw[2] / length,
            raw[3] / length,
        ]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}
