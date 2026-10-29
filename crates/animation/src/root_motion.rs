//! Root motion: the per-frame [`RootMotionDelta`] the played clip carries
//! (design §8.4, gate B — the sampler writes the delta, the host applies
//! it: to the transform, to the simulation, or C-style — accumulate in the
//! variable pass and consume on the fixed tick).

use glam::{Quat, Vec3};
use ornis_core::Entity;

use crate::events::CursorStep;
use crate::{JointId, JointTrack, RetargetMap, SkelClip, Skeleton};

/// Steps below this read as no step (float noise).
const STEP_EPSILON: f32 = 1.0e-6;

/// The walk accumulates at most this many within-a-loop pieces and then
/// stops early — a runaway cursor step cannot spin the sampler.
const MAX_PIECES: u32 = 256;

/// One frame's root-motion delta from the played clip (design §8.4 B).
///
/// The motion of the skeleton's root joint over this frame's cursor step,
/// in the skeleton's model space: [`Self::translation`] is the joint's
/// position delta and [`Self::rotation`] the orientation delta
/// (`rotation * previous == current` at full weight). The sampler refreshes
/// the hot lane on the character root every frame (identity while paused or
/// clip-less); the host applies it — to the transform, to the simulation,
/// or C-style: accumulate in the variable pass and consume on the fixed
/// tick (§8.4 C).
///
/// The recorded delta is the pose's root contributors folded the same way
/// the pose is: crossfade mix, then the tree's effective-weight fold, then
/// overlay layers bottom-up (§8.4). A retarget scales the translation by
/// the mapped root bone's rest-length ratio.
///
/// `Clone + Send + Sync` per the hot-lane contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootMotionDelta {
    /// Root-joint translation delta of this frame, skeleton model space.
    pub translation: Vec3,
    /// Root-joint rotation delta of this frame: `rotation * previous ==
    /// current` at full weight.
    pub rotation: Quat,
}

impl RootMotionDelta {
    /// The no-motion delta (a paused or clip-less cursor).
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
    };

    /// Pulls the delta toward none by `weight`: translation scales,
    /// rotation slerps from identity. A non-positive or non-finite
    /// `weight` is no motion — the same answer as a zero player weight.
    pub(crate) fn scaled(self, weight: f32) -> Self {
        if !weight.is_finite() || weight <= 0.0 {
            return Self::IDENTITY;
        }
        Self {
            translation: self.translation * weight,
            rotation: Quat::IDENTITY.slerp(self.rotation, weight),
        }
    }

    /// `self·(1 − progress) + to·progress`, the crossfade and the layer
    /// overlay (a layer lerps the accumulated delta toward its own).
    /// A non-finite or non-positive `progress` keeps `self`; `progress ≥ 1`
    /// takes `to`.
    pub(crate) fn mix(self, to: Self, progress: f32) -> Self {
        if !progress.is_finite() || progress <= 0.0 {
            return self;
        }
        if progress >= 1.0 {
            return to;
        }
        Self {
            translation: self.translation.lerp(to.translation, progress),
            rotation: self.rotation.slerp(to.rotation, progress),
        }
    }

    /// Folds `other` into a running blend at relative `weight`, the same
    /// renormalized share as [`crate::blend::BlendedPose::fold_pose`].
    /// `total` starts at `0.0`. Non-positive weights are a no-op, so the
    /// first live contributor takes the accumulator whole.
    pub(crate) fn fold(&mut self, other: Self, weight: f32, total: &mut f32) {
        if !weight.is_finite() || weight <= 0.0 {
            return;
        }
        let next = *total + weight;
        let share = weight / next;
        *total = next;
        *self = self.mix(other, share);
    }
}

/// One overlay layer's root-motion contribution, recorded while the layer
/// cursor steps and applied after the crossfade mix (§8.4: layers sit on
/// top of the blended base, bottom-up).
///
/// `delta` is the layer clip's full (weight-1) root motion; `weight` is
/// `player.weight · mask[root]`, the same factor the pose overlay uses.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RootMotionPart {
    /// Skeleton root the layer overlays.
    pub(crate) entity: Entity,
    /// Overlay factor in `0..=1` (`player.weight · mask[root]`).
    pub(crate) weight: f32,
    /// Full root-motion delta of the layer clip over this frame.
    pub(crate) delta: RootMotionDelta,
}

/// Integrates `root`'s channels over `cursor`'s step — the frame's
/// [`RootMotionDelta`] (design §8.4 B).
///
/// Looping clips wrap at the ends without losing motion at the seam (a
/// locomotion loop keeps walking forward through the track's snap-back),
/// clamped clips stop at the boundary, reverse cursors retrace. `weight`
/// scales the delta toward none (`0` holds the rest pose and moves
/// nothing). A missing clip, root joint or track reads as no motion, as
/// does a step past [`MAX_PIECES`] clip loops after the pieces already
/// walked.
///
/// `map` resolves the target root onto the source clip (§8.4): the first
/// track whose source joint the map sends to `root` wins (the same
/// first-track rule as [`crate::blend::sample_joints`]), and its
/// translation delta is scaled by that pair's rest-length ratio. Without
/// a map the track is the one authored on `root` itself. A map that names
/// no source bone for `root` is no motion — the seam stays explicit.
pub(crate) fn root_motion(
    clip: Option<&SkelClip>,
    cursor: Option<CursorStep>,
    root: Option<JointId>,
    map: Option<&RetargetMap>,
    weight: f32,
) -> RootMotionDelta {
    let (Some(clip), Some(cursor), Some(root)) = (clip, cursor, root) else {
        return RootMotionDelta::IDENTITY;
    };
    if cursor.duration <= 0.0 {
        return RootMotionDelta::IDENTITY;
    }
    let Some((track, scale)) = root_track(clip, root, map) else {
        return RootMotionDelta::IDENTITY;
    };
    let mut translation = Vec3::ZERO;
    let mut rotation = Quat::IDENTITY;
    walk(track, cursor, &mut translation, &mut rotation);
    RootMotionDelta {
        translation: translation * scale,
        rotation,
    }
    .scaled(weight)
}

/// The first clip track that drives `root`, plus the translation scale
/// the retarget applies to it. `None` when no track does.
fn root_track<'a>(
    clip: &'a SkelClip,
    root: JointId,
    map: Option<&RetargetMap>,
) -> Option<(&'a JointTrack, f32)> {
    clip.tracks.iter().find_map(|track| {
        let scale = track_scale(track, root, map)?;
        Some((track, scale))
    })
}

/// Translation scale of `track` when it drives the target `root`: `1.0`
/// with no map (and only when the track is authored on `root`), the
/// mapped pair's scale otherwise. `None` when this track does not drive
/// `root`.
fn track_scale(track: &JointTrack, root: JointId, map: Option<&RetargetMap>) -> Option<f32> {
    let Some(map) = map else {
        return (track.joint == root).then_some(1.0);
    };
    let source = track.joint.index();
    let target = map.joints.get(source).copied().flatten()?;
    if target != root {
        return None;
    }
    Some(
        map.scales
            .get(source)
            .copied()
            .filter(|scale| scale.is_finite())
            .unwrap_or(1.0),
    )
}

/// Accumulates every piece of the cursor step into the running deltas.
fn walk(track: &JointTrack, cursor: CursorStep, translation: &mut Vec3, rotation: &mut Quat) {
    let mut at = cursor.prev;
    let mut left = cursor.step;
    for _ in 0..MAX_PIECES {
        let Some((from, to, next, used)) = piece(at, left, cursor.duration, cursor.looping) else {
            break;
        };
        accumulate(track, from, to, translation, rotation);
        at = next;
        left -= used;
    }
}

/// The next walk piece — `from → to` inside one loop — plus where the
/// walk continues (`next_at`) and how much of the step it consumed;
/// [`None`] at the end of the step.
fn piece(at: f32, left: f32, duration: f32, looping: bool) -> Option<(f32, f32, f32, f32)> {
    if left > STEP_EPSILON {
        return forward(at, left, duration, looping);
    }
    if left < -STEP_EPSILON {
        return backward(at, left, duration, looping);
    }
    None
}

/// Forward walk piece: ends the step inside the loop, wraps at the
/// duration (looping) or stops at it (clamped).
fn forward(at: f32, left: f32, duration: f32, looping: bool) -> Option<(f32, f32, f32, f32)> {
    let headroom = duration - at;
    if left < headroom {
        return Some((at, at + left, at + left, left));
    }
    if looping {
        return Some((at, duration, 0.0, headroom));
    }
    Some((at, duration, duration, left))
}

/// Backward walk piece (negative step): retraces inside the loop, wraps
/// down through zero (looping) or stops at it (clamped).
fn backward(at: f32, left: f32, duration: f32, looping: bool) -> Option<(f32, f32, f32, f32)> {
    let headroom = at;
    if -left < headroom {
        return Some((at, at + left, at + left, left));
    }
    if looping {
        return Some((at, 0.0, duration, -headroom));
    }
    Some((at, 0.0, 0.0, left))
}

/// Accumulates one within-a-loop piece `from → to` of `track`'s channels
/// into the running `translation`/`rotation` deltas.
fn accumulate(track: &JointTrack, from: f32, to: f32, translation: &mut Vec3, rotation: &mut Quat) {
    let from_t = track.translation.sample(from).unwrap_or(Vec3::ZERO);
    let to_t = track.translation.sample(to).unwrap_or(Vec3::ZERO);
    *translation += to_t - from_t;
    let from_r = track.rotation.sample(from).unwrap_or(Quat::IDENTITY);
    let to_r = track.rotation.sample(to).unwrap_or(Quat::IDENTITY);
    *rotation = (to_r * from_r.inverse()) * *rotation;
}

/// The skeleton's root joint that carries root motion (§8.4): the first
/// joint without a parent. [`None`] for a skeleton without one.
pub(crate) fn root_joint(skeleton: &Skeleton) -> Option<JointId> {
    skeleton
        .parents
        .iter()
        .position(|parent| parent.is_none())
        .map(JointId::from)
}

#[cfg(test)]
mod tests {
    use glam::Mat4;
    use ornis_core::{Clamped01, Entity, Seconds, SmartStore};

    use super::*;
    use crate::{
        ClipId, JointMask, JointTrack, Key, KeyTrack, MaskId, RetargetMap, SkelFade, SkelLayer,
        SkelLayerStack, SkelPlayer, SkelTree, SkelTreePlayer, TreeChild, TreeNodeDef, TreeParam,
        advance_skel_players,
    };

    fn clip_over(translation: KeyTrack<Vec3>, rotation: KeyTrack<Quat>) -> SkelClip {
        SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![JointTrack {
                joint: JointId::from_raw(0),
                node: None,
                translation,
                rotation,
                scale: KeyTrack::linear(vec![]),
            }],
        }
    }

    fn walking() -> SkelClip {
        clip_over(
            KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 1.0,
                    value: Vec3::X,
                },
            ]),
            KeyTrack::linear(vec![]),
        )
    }

    fn step_of(prev: f32, step: f32, looping: bool) -> Option<CursorStep> {
        Some(CursorStep {
            prev,
            step,
            duration: 1.0,
            looping,
            weight: 1.0,
        })
    }

    fn run(clip: &SkelClip, cursor: Option<CursorStep>, weight: f32) -> RootMotionDelta {
        root_motion(Some(clip), cursor, Some(JointId::from_raw(0)), None, weight)
    }

    #[test]
    fn forward_step_translates_by_the_track() {
        let delta = run(&walking(), step_of(0.25, 0.5, false), 1.0);
        assert!((delta.translation - Vec3::X * 0.5).length() < 1e-5);
        assert_eq!(delta.rotation, Quat::IDENTITY);
    }

    #[test]
    fn clamped_step_stops_at_the_end() {
        let delta = run(&walking(), step_of(0.8, 0.5, false), 1.0);
        assert!((delta.translation - Vec3::X * 0.2).length() < 1e-5);
    }

    #[test]
    fn looping_step_keeps_walking_through_the_seam() {
        // A locomotion loop: the track snaps back at the loop point, but
        // the character keeps walking forward.
        let delta = run(&walking(), step_of(0.9, 0.2, true), 1.0);
        assert!((delta.translation - Vec3::X * 0.2).length() < 1e-5);
    }

    #[test]
    fn reverse_step_retraces_the_track() {
        let delta = run(&walking(), step_of(0.3, -0.2, false), 1.0);
        assert!((delta.translation - Vec3::X * -0.2).length() < 1e-5);
    }

    #[test]
    fn rotation_delta_composes_the_turn() {
        let quarter = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let clip = clip_over(
            KeyTrack::linear(vec![]),
            KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Quat::IDENTITY,
                },
                Key {
                    time: 1.0,
                    value: quarter,
                },
            ]),
        );
        let delta = run(&clip, step_of(0.0, 1.0, false), 1.0);
        assert!((delta.rotation - quarter).length() < 1e-5);
    }

    #[test]
    fn weight_scales_the_delta_to_none() {
        let half = run(&walking(), step_of(0.25, 0.5, false), 0.5);
        assert!((half.translation - Vec3::X * 0.25).length() < 1e-5);
        let none = run(&walking(), step_of(0.25, 0.5, false), 0.0);
        assert_eq!(none, RootMotionDelta::IDENTITY);
    }

    #[test]
    fn missing_pieces_read_as_no_motion() {
        let root = Some(JointId::from_raw(0));
        let cursor = step_of(0.0, 0.5, false);
        assert_eq!(
            root_motion(None, cursor, root, None, 1.0),
            RootMotionDelta::IDENTITY
        );
        assert_eq!(
            root_motion(Some(&walking()), None, root, None, 1.0),
            RootMotionDelta::IDENTITY
        );
        assert_eq!(
            root_motion(Some(&walking()), cursor, None, None, 1.0),
            RootMotionDelta::IDENTITY
        );
        let empty = clip_over(KeyTrack::linear(vec![]), KeyTrack::linear(vec![]));
        assert_eq!(
            root_motion(Some(&empty), cursor, root, None, 1.0),
            RootMotionDelta::IDENTITY
        );
    }

    #[test]
    fn advance_records_root_motion_and_resets_it_when_the_cursor_stops() {
        let mut store = SmartStore::new();
        let playlist = store.create_entity();
        store.insert_cold(playlist, walking());
        let root = store.create_entity();
        store.insert(
            root,
            Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec![String::new()]),
        );
        store.insert(
            root,
            SkelPlayer {
                clip: ClipId(playlist),
                time: Seconds::new(0.0),
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: false,
            },
        );
        // The lane is installed explicitly (like `AnimEventQueue`): the host
        // records identity on the root, the sampler overwrites it per frame.
        store.insert(root, RootMotionDelta::IDENTITY);
        advance_skel_players(&store, 0.5);
        let delta = store
            .read_lane::<RootMotionDelta>()
            .and_then(|lane| lane.get(root).copied())
            .expect("a playing cursor records its root motion");
        assert!((delta.translation - Vec3::X * 0.5).length() < 1e-5);
        // Paused: the record resets to identity instead of holding the
        // last frame's motion (design §8.4).
        store
            .write_lane::<SkelPlayer>()
            .expect("player lane")
            .get_mut(root)
            .expect("player")
            .playing = false;
        advance_skel_players(&store, 0.5);
        let paused = store
            .read_lane::<RootMotionDelta>()
            .and_then(|lane| lane.get(root).copied());
        assert_eq!(paused, Some(RootMotionDelta::IDENTITY));
        // Clip-less cursors reset the same way.
        let mut players = store.write_lane::<SkelPlayer>().expect("player lane");
        let player = players.get_mut(root).expect("player");
        player.playing = true;
        player.clip = ClipId(Entity::new(99));
        drop(players);
        advance_skel_players(&store, 0.5);
        let clipless = store
            .read_lane::<RootMotionDelta>()
            .and_then(|lane| lane.get(root).copied());
        assert_eq!(clipless, Some(RootMotionDelta::IDENTITY));
    }

    fn sideways() -> SkelClip {
        clip_over(
            KeyTrack::linear(vec![
                Key {
                    time: 0.0,
                    value: Vec3::ZERO,
                },
                Key {
                    time: 1.0,
                    value: Vec3::Y,
                },
            ]),
            KeyTrack::linear(vec![]),
        )
    }

    fn one_bone() -> Skeleton {
        Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec![String::new()])
    }

    fn player_on(clip: Entity, weight: f32) -> SkelPlayer {
        SkelPlayer {
            clip: ClipId(clip),
            time: Seconds::ZERO,
            speed: Seconds::new(1.0),
            weight: Clamped01::new(weight),
            playing: true,
            looping: false,
        }
    }

    fn motion_of(store: &SmartStore, root: Entity) -> Vec3 {
        store
            .read_lane::<RootMotionDelta>()
            .and_then(|lane| lane.get(root).copied())
            .expect("opt-in root motion")
            .translation
    }

    fn near(got: Vec3, want: Vec3) {
        assert!((got - want).length() < 1e-4, "got {got:?} want {want:?}");
    }

    /// Walk on +X and strafe on +Y, both opting into root motion.
    fn two_clip_root() -> (SmartStore, Entity, Entity, Entity) {
        let mut store = SmartStore::new();
        let walk = store.create_entity();
        store.insert_cold(walk, walking());
        let side = store.create_entity();
        store.insert_cold(side, sideways());
        let root = store.create_entity();
        store.insert(root, one_bone());
        store.insert(root, RootMotionDelta::IDENTITY);
        (store, root, walk, side)
    }

    #[test]
    fn retarget_scales_the_source_track_mapped_onto_the_root() {
        let clip = SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: vec![JointTrack {
                joint: JointId::from_raw(1),
                node: None,
                translation: KeyTrack::linear(vec![
                    Key {
                        time: 0.0,
                        value: Vec3::ZERO,
                    },
                    Key {
                        time: 1.0,
                        value: Vec3::X,
                    },
                ]),
                rotation: KeyTrack::linear(vec![]),
                scale: KeyTrack::linear(vec![]),
            }],
        };
        let map = RetargetMap {
            joints: vec![None, Some(JointId::from_raw(0))],
            scales: vec![1.0, 2.0],
        };
        let cursor = step_of(0.0, 0.5, false);
        let scaled = root_motion(
            Some(&clip),
            cursor,
            Some(JointId::from_raw(0)),
            Some(&map),
            1.0,
        );
        near(scaled.translation, Vec3::X);
        let unmapped = root_motion(Some(&clip), cursor, Some(JointId::from_raw(0)), None, 1.0);
        assert_eq!(unmapped, RootMotionDelta::IDENTITY);
    }

    #[test]
    fn crossfade_mixes_root_motion_by_the_pre_tick_share() {
        let (mut store, root, walk, side) = two_clip_root();
        store.insert(root, player_on(side, 1.0));
        store.insert(
            root,
            SkelFade {
                from: player_on(walk, 1.0),
                duration: Seconds::new(1.0),
                elapsed: Seconds::ZERO,
            },
        );
        advance_skel_players(&store, 0.5);
        // Share 0 at the start of the step: the outgoing walk owns the delta.
        near(motion_of(&store, root), Vec3::X * 0.5);
        advance_skel_players(&store, 0.5);
        // The clock is now at 0.5, so the next step mixes the two halves.
        near(motion_of(&store, root), Vec3::X * 0.25 + Vec3::Y * 0.25);
    }

    #[test]
    fn a_paused_outgoing_clip_holds_root_motion_at_share_zero() {
        let (mut store, root, walk, side) = two_clip_root();
        store.insert(root, player_on(side, 1.0));
        let mut held = player_on(walk, 1.0);
        held.playing = false;
        store.insert(
            root,
            SkelFade {
                from: held,
                duration: Seconds::new(1.0),
                elapsed: Seconds::ZERO,
            },
        );
        advance_skel_players(&store, 0.5);
        near(motion_of(&store, root), Vec3::ZERO);
    }

    #[test]
    fn an_overlay_layer_blends_root_motion_over_the_base() {
        let (mut store, root, walk, side) = two_clip_root();
        store.insert(root, player_on(walk, 1.0));
        let mask = store.create_entity();
        store.insert_cold(
            mask,
            JointMask {
                name: String::new(),
                weights: vec![Clamped01::ONE],
            },
        );
        let mut layer = player_on(side, 0.5);
        store.insert(
            root,
            SkelLayerStack {
                layers: vec![SkelLayer {
                    player: layer,
                    mask: MaskId(mask),
                }],
            },
        );
        advance_skel_players(&store, 0.5);
        // Base walks +X by 0.5; the layer strafes +Y by 0.5 at half weight.
        near(motion_of(&store, root), Vec3::X * 0.25 + Vec3::Y * 0.25);
        layer.weight = Clamped01::ONE;
        layer.playing = false;
        store
            .write_lane::<SkelLayerStack>()
            .expect("layers")
            .get_mut(root)
            .expect("stack")
            .layers[0]
            .player = layer;
        // Rewind the base cursor so the next step is another half unit.
        store
            .write_lane::<SkelPlayer>()
            .expect("players")
            .get_mut(root)
            .expect("player")
            .time = Seconds::ZERO;
        advance_skel_players(&store, 0.5);
        // A held full-mask layer pulls the delta to none: the pose is the
        // layer's, and the layer did not move.
        near(motion_of(&store, root), Vec3::ZERO);
    }

    #[test]
    fn a_full_mask_layer_sits_on_top_of_the_crossfade_mix() {
        let (mut store, root, walk, side) = two_clip_root();
        let idle = store.create_entity();
        store.insert_cold(
            idle,
            clip_over(KeyTrack::linear(vec![]), KeyTrack::linear(vec![])),
        );
        store.insert(root, player_on(side, 1.0));
        store.insert(
            root,
            SkelFade {
                from: player_on(walk, 1.0),
                duration: Seconds::new(1.0),
                elapsed: Seconds::new(0.5),
            },
        );
        let mask = store.create_entity();
        store.insert_cold(
            mask,
            JointMask {
                name: String::new(),
                weights: vec![Clamped01::ONE],
            },
        );
        store.insert(
            root,
            SkelLayerStack {
                layers: vec![SkelLayer {
                    player: player_on(idle, 1.0),
                    mask: MaskId(mask),
                }],
            },
        );
        advance_skel_players(&store, 0.5);
        // The mix of walk and strafe is non-zero; the held layer replaces it.
        near(motion_of(&store, root), Vec3::ZERO);
    }

    #[test]
    fn a_zero_root_mask_leaves_the_base_delta() {
        let (mut store, root, walk, side) = two_clip_root();
        store.insert(root, player_on(walk, 1.0));
        let mask = store.create_entity();
        store.insert_cold(
            mask,
            JointMask {
                name: String::new(),
                weights: vec![Clamped01::ZERO],
            },
        );
        store.insert(
            root,
            SkelLayerStack {
                layers: vec![SkelLayer {
                    player: player_on(side, 1.0),
                    mask: MaskId(mask),
                }],
            },
        );
        advance_skel_players(&store, 0.5);
        near(motion_of(&store, root), Vec3::X * 0.5);
    }

    #[test]
    fn a_blend_tree_folds_leaf_root_motion_and_a_broken_tree_holds() {
        let (mut store, root, walk, side) = two_clip_root();
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip { clip: ClipId(walk) },
                TreeNodeDef::Clip { clip: ClipId(side) },
                TreeNodeDef::Blend1d {
                    axis: 0,
                    children: vec![
                        TreeChild {
                            position: 0.0,
                            node: 0,
                        },
                        TreeChild {
                            position: 1.0,
                            node: 1,
                        },
                    ],
                },
            ],
            params: vec![TreeParam {
                name: "gait".to_string(),
                default: 0.5,
            }],
        };
        store.insert_cold(root, tree);
        store.insert(
            root,
            SkelTreePlayer {
                params: vec![0.5],
                leaves: vec![player_on(walk, 1.0), player_on(side, 1.0)],
                machines: Vec::new(),
                switches: Vec::new(),
                freeforms: Vec::new(),
            },
        );
        advance_skel_players(&store, 0.5);
        near(motion_of(&store, root), Vec3::X * 0.25 + Vec3::Y * 0.25);

        // A child that is not an earlier node breaks the contract: the pose
        // is bad skin, so the delta stays at the frame's identity reset
        // instead of falling back to the leaves' own weights.
        store
            .write_cold_lane::<SkelTree>()
            .expect("trees")
            .get_mut(root)
            .expect("tree")
            .nodes[2] = TreeNodeDef::Blend1d {
            axis: 0,
            children: vec![TreeChild {
                position: 0.0,
                node: 5,
            }],
        };
        advance_skel_players(&store, 0.5);
        near(motion_of(&store, root), Vec3::ZERO);
    }

    #[test]
    fn a_full_mask_layer_replaces_tree_root_motion() {
        let (mut store, root, walk, side) = two_clip_root();
        store.insert_cold(
            root,
            SkelTree {
                nodes: vec![TreeNodeDef::Clip { clip: ClipId(walk) }],
                params: Vec::new(),
            },
        );
        store.insert(
            root,
            SkelTreePlayer {
                params: Vec::new(),
                leaves: vec![player_on(walk, 1.0)],
                machines: Vec::new(),
                switches: Vec::new(),
                freeforms: Vec::new(),
            },
        );
        let mask = store.create_entity();
        store.insert_cold(
            mask,
            JointMask {
                name: String::new(),
                weights: vec![Clamped01::ONE],
            },
        );
        store.insert(
            root,
            SkelLayerStack {
                layers: vec![SkelLayer {
                    player: player_on(side, 1.0),
                    mask: MaskId(mask),
                }],
            },
        );
        advance_skel_players(&store, 0.5);
        near(motion_of(&store, root), Vec3::Y * 0.5);
    }
}
