//! Clip events (`docs/animation-design.md` §8.1): named normalized
//! positions in a clip and the per-frame fired queue an application
//! reads and drains. A separate module from the players because
//! authoring data, crossing detection and the hot output queue are
//! different responsibilities.

use ornis_core::{ColdComponentStore, ComponentStore, Entity};

use crate::ClipId;

/// One authoring event of a clip: a named mark at a normalized position
/// in the clip's loop, in `[0, 1)`.
///
/// Events are data ([`ClipEvents`], keyed by [`ClipId`](crate::ClipId)). A
/// cursor that crosses `at` while it advances fires the event once into
/// the owner's [`AnimEventQueue`] (design §8.1): start-exclusive,
/// end-inclusive — a cursor that starts exactly on a mark does not fire
/// it (put marks slightly above `0.0` to signal clip starts). Names are
/// free-form strings (`"footstep"`, `"hit"`); the queue carries the name
/// and the clip, not the mark itself.
#[derive(Debug, Clone, PartialEq)]
pub struct AnimEvent {
    /// Name reported in [`AnimEventOut::event`]; non-empty (checked at
    /// the authoring setter).
    pub name: String,
    /// Position within one loop of the clip: `0.0 <= at < 1.0`.
    pub at: f32,
}

/// Events of one clip — cold authoring data keyed by [`ClipId`](crate::ClipId)
/// in the [`ColdComponentStore`] lane. Events are optional: a clip
/// without them (or a cursor whose owner has no [`AnimEventQueue`]) never
/// fires anything. The lane is written through `set_clip_events` (app
/// wiring); marks with a non-finite `at` or outside `0..1` never fire.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClipEvents {
    /// Events in author order; fire-order ties fall back to this order.
    pub events: Vec<AnimEvent>,
}

/// One fired event of a frame (hot output item of the queue).
#[derive(Debug, Clone, PartialEq)]
pub struct AnimEventOut {
    /// The clip whose cursor crossed the mark ([`AnimEventOut`] of several
    /// clips can share one owner; clip identity disambiguates them).
    pub clip: ClipId,
    /// The [`AnimEvent::name`] that fired.
    pub event: String,
    /// Effective pose contribution of the stepping cursor: its own blend
    /// weight ([`crate::SkelPlayer::weight`] / [`crate::AnimPlayer::weight`])
    /// times the tree path weights and machine-state activity of a tree
    /// leaf, times the crossfade share of an incoming (`progress`) or
    /// outgoing (`1 - progress`) cursor ([`crate::blend::fade_progress`]).
    /// Every stepping cursor fires — zero weight included — so the host,
    /// not the sampler, filters "audible enough" (design §8.1).
    pub weight: f32,
}

/// Per-frame queue of fired events on the cursor owner entity (hot lane:
/// `Clone + Send + Sync` per the contract).
///
/// Installed explicitly via `SmartStore::insert` on every entity that
/// should report events (the player root — tree leaves and machine
/// states aggregate into their root's queue). Cursors whose owner has no
/// queue fire nothing: events are an opt-in output, never a side effect.
/// The application reads and **drains** the queue every frame; without a
/// drain the vector grows unboundedly (design §8.1).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AnimEventQueue {
    /// Events fired since the last drain, in fire order.
    pub fired: Vec<AnimEventOut>,
}

impl AnimEventQueue {
    /// Drains the fired events of the frame in fire order; the queue is
    /// empty afterwards. Read-and-clear in one step.
    pub fn drain(&mut self) -> std::vec::Drain<'_, AnimEventOut> {
        self.fired.drain(..)
    }
}

/// The cursor step one advance just performed — the input of the
/// crossing detection and the emit glue ([`CursorStep`] keeps those
/// signatures argument-lean).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CursorStep {
    /// Cursor time before the step, seconds.
    pub prev: f32,
    /// Signed cursor step `dt * speed`, seconds.
    pub step: f32,
    /// Clip duration in seconds.
    pub duration: f32,
    /// Whether the cursor wraps past the duration.
    pub looping: bool,
    /// Effective blend weight of the stepping cursor (the cursor's own
    /// weight times any tree/fade share the caller scales in via
    /// [`CursorStep::scaled`]), carried into [`AnimEventOut::weight`]
    /// (design §8.1: every stepping cursor fires, the host filters).
    pub weight: f32,
}

impl CursorStep {
    /// The same step with its blend weight scaled by `scale` — the
    /// effective pose contribution of this cursor (tree path weights,
    /// machine state activity, crossfade share; design §8.1).
    pub(crate) fn scaled(mut self, scale: f32) -> Self {
        self.weight *= scale;
        self
    }
}

/// The unwrapped time of `pos`' first crossing of the oriented cursor
/// step `(prev, prev + step]` — start excluded, end included — or
/// `None` when the step never reaches the mark. `pos` lies in
/// `[0, duration)`.
///
/// Looping steps map `pos + k * duration` into the step for the first
/// reachable integer `k` (a step spanning several loops crosses each
/// mark at most once, at its first pass). Non-looping steps clamp the
/// end to `[0, duration]`, matching [`crate::advance_player_time`].
fn first_crossing(pos: f32, prev: f32, step: f32, duration: f32, looping: bool) -> Option<f32> {
    let end = prev + step;
    if looping {
        if step > 0.0 {
            let k = ((prev - pos) / duration).floor() as i64 + 1;
            let t = pos + (k as f32) * duration;
            (t <= end).then_some(t)
        } else {
            let k = ((prev - pos) / duration).ceil() as i64 - 1;
            let t = pos + (k as f32) * duration;
            (t >= end).then_some(t)
        }
    } else if step > 0.0 {
        (prev < pos && pos <= end.min(duration)).then_some(pos)
    } else {
        (end.max(0.0) <= pos && pos < prev).then_some(pos)
    }
}

/// Appends the events crossed by one cursor step to `out` in fire order:
/// by crossing time (forward or backward), ties by author order. Each
/// event fires at most once per step (design §8.1).
fn fire_step(out: &mut Vec<AnimEventOut>, events: &ClipEvents, clip: ClipId, cursor: CursorStep) {
    let CursorStep {
        prev,
        step,
        duration,
        looping,
        weight,
    } = cursor;
    if duration <= 0.0 || step == 0.0 || !prev.is_finite() || !step.is_finite() {
        return;
    }
    let mut hits: Vec<(f32, usize)> = Vec::new();
    for (index, event) in events.events.iter().enumerate() {
        if !event.at.is_finite() || !(0.0..1.0).contains(&event.at) {
            continue;
        }
        if let Some(t) = first_crossing(event.at * duration, prev, step, duration, looping) {
            hits.push(((t - prev) / step, index));
        }
    }
    hits.sort_by(|a, b| a.0.total_cmp(&b.0));
    out.extend(hits.into_iter().map(|(_, index)| AnimEventOut {
        clip,
        event: events.events[index].name.clone(),
        weight,
    }));
}

/// Appends the events crossed by one cursor step of `owner`'s `clip`
/// cursor to the owner's [`AnimEventQueue`] — the glue every player
/// advance calls right after stepping the cursor. A missing queue lane,
/// a missing queue on `owner`, or missing [`ClipEvents`] of `clip` each
/// mean "fire nothing".
pub(crate) fn emit_step(
    queues: Option<&mut ComponentStore<AnimEventQueue>>,
    events: Option<&ColdComponentStore<ClipEvents>>,
    owner: Entity,
    clip: ClipId,
    cursor: CursorStep,
) {
    let (Some(queues), Some(events)) = (queues, events) else {
        return;
    };
    let Some(clip_events) = events.get(clip.0) else {
        return;
    };
    let Some(queue) = queues.get_mut(owner) else {
        return;
    };
    fire_step(&mut queue.fired, clip_events, clip, cursor);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::{Clamped01, Seconds, SmartStore};

    use crate::{
        MachineCursor, SkelClip, SkelFade, SkelPlayer, SkelTree, SkelTreePlayer, TreeChild,
        TreeChild2d, TreeNodeDef, TreeParam, advance_skel_players,
    };

    fn marks(pairs: &[(&str, f32)]) -> ClipEvents {
        ClipEvents {
            events: pairs
                .iter()
                .map(|&(name, at)| AnimEvent {
                    name: name.to_owned(),
                    at,
                })
                .collect(),
        }
    }

    fn fired(
        events: &ClipEvents,
        prev: f32,
        step: f32,
        duration: f32,
        looping: bool,
    ) -> Vec<String> {
        let mut out = Vec::new();
        fire_step(
            &mut out,
            events,
            ClipId(Entity::new(0)),
            CursorStep {
                prev,
                step,
                duration,
                looping,
                weight: 1.0,
            },
        );
        out.into_iter().map(|item| item.event).collect()
    }

    #[test]
    fn event_carries_the_cursor_blend_weight() {
        let events = marks(&[("foot", 0.5)]);
        let mut out = Vec::new();
        fire_step(
            &mut out,
            &events,
            ClipId(Entity::new(0)),
            CursorStep {
                prev: 0.25,
                step: 0.25,
                duration: 1.0,
                looping: false,
                weight: 0.0,
            },
        );
        // Every stepping cursor fires — zero weight included; the host
        // filters on `weight` (design §8.1).
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].weight, 0.0);
    }

    fn clip_firing() -> SkelClip {
        SkelClip {
            name: String::new(),
            duration: 1.0,
            tracks: Vec::new(),
        }
    }

    fn stepping(clip: ClipId) -> SkelPlayer {
        SkelPlayer {
            clip,
            time: Seconds::new(0.0),
            speed: Seconds::new(1.0),
            weight: Clamped01::ONE,
            playing: true,
            looping: false,
        }
    }

    fn mark(store: &mut SmartStore, name: &str, at: f32) -> ClipId {
        let playlist = store.create_entity();
        store.insert_cold(playlist, clip_firing());
        store.insert_cold(
            playlist,
            ClipEvents {
                events: vec![AnimEvent {
                    name: name.to_owned(),
                    at,
                }],
            },
        );
        ClipId(playlist)
    }

    fn fired_weights(store: &SmartStore, root: Entity) -> Vec<(String, f32)> {
        let queue = store.read_lane::<AnimEventQueue>().expect("queue lane");
        queue
            .get(root)
            .expect("root queue")
            .fired
            .iter()
            .map(|event| (event.event.clone(), event.weight))
            .collect()
    }

    /// Runs one §8.1 tree case end to end: clips "a"/"b" firing at `0.5`,
    /// `tree` + `player` on the root, one 0.5 s advance — then answers the
    /// fired `(event, weight)` pairs. `build` receives both playlist ids so
    /// the tree can cite them.
    fn tree_fired(
        build: impl FnOnce(ClipId, ClipId) -> (SkelTree, SkelTreePlayer),
    ) -> Vec<(String, f32)> {
        let mut store = SmartStore::new();
        let clip_a = mark(&mut store, "a", 0.5);
        let clip_b = mark(&mut store, "b", 0.5);
        let (tree, player) = build(clip_a, clip_b);
        let root = store.create_entity();
        store.insert_cold(root, tree);
        store.insert(root, player);
        store.insert(root, AnimEventQueue::default());
        advance_skel_players(&store, 0.5);
        fired_weights(&store, root)
    }

    #[test]
    fn tree_blend_weight_scales_the_leaf_event() {
        // Both leaves step and fire, but the parameter puts blend weight
        // only on the first leaf — the host filters on the event weight
        // (design §8.1).
        assert_eq!(
            tree_fired(|clip_a, clip_b| (
                SkelTree {
                    nodes: vec![
                        TreeNodeDef::Clip { clip: clip_a },
                        TreeNodeDef::Clip { clip: clip_b },
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
                        name: "x".to_owned(),
                        default: 0.0,
                    }],
                },
                SkelTreePlayer {
                    params: vec![0.0],
                    leaves: vec![stepping(clip_a), stepping(clip_b)],
                    machines: Vec::new(),
                },
            )),
            vec![("a".to_owned(), 1.0), ("b".to_owned(), 0.0)]
        );
    }

    #[test]
    fn machine_inactive_state_event_is_zero() {
        // Only the standing state contributes to the pose.
        assert_eq!(
            tree_fired(|clip_a, clip_b| (
                SkelTree {
                    nodes: vec![
                        TreeNodeDef::Clip { clip: clip_a },
                        TreeNodeDef::Clip { clip: clip_b },
                        TreeNodeDef::Machine {
                            states: vec![0, 1],
                            initial: 0,
                            transitions: Vec::new(),
                        },
                    ],
                    params: Vec::new(),
                },
                SkelTreePlayer {
                    params: Vec::new(),
                    leaves: vec![stepping(clip_a), stepping(clip_b)],
                    machines: vec![MachineCursor {
                        state: 0,
                        from: None,
                        elapsed: Seconds::new(0.0),
                        duration: Seconds::new(0.0),
                    }],
                },
            )),
            vec![("a".to_owned(), 1.0), ("b".to_owned(), 0.0)]
        );
    }

    #[test]
    fn crossfade_share_scales_incoming_and_outgoing_events() {
        let mut store = SmartStore::new();
        let clip_in = mark(&mut store, "in", 0.5);
        let clip_out = mark(&mut store, "out", 0.5);
        let root = store.create_entity();
        store.insert(root, stepping(clip_in));
        store.insert(
            root,
            SkelFade {
                from: stepping(clip_out),
                duration: Seconds::new(1.0),
                elapsed: Seconds::new(0.5),
            },
        );
        store.insert(root, AnimEventQueue::default());
        advance_skel_players(&store, 0.5);
        // Mid-fade: the incoming cursor carries its progress share and the
        // outgoing one the complement (design §8.1).
        assert_eq!(
            fired_weights(&store, root),
            vec![("in".to_owned(), 0.5), ("out".to_owned(), 0.5)]
        );
    }

    #[test]
    fn machine_transition_splits_the_event_weight() {
        // Mid-transition the leaving state carries `1 − progress` and the
        // heading one `progress` (design §8.1).
        assert_eq!(
            tree_fired(|clip_a, clip_b| (
                SkelTree {
                    nodes: vec![
                        TreeNodeDef::Clip { clip: clip_a },
                        TreeNodeDef::Clip { clip: clip_b },
                        TreeNodeDef::Machine {
                            states: vec![0, 1],
                            initial: 0,
                            transitions: Vec::new(),
                        },
                    ],
                    params: Vec::new(),
                },
                SkelTreePlayer {
                    params: Vec::new(),
                    leaves: vec![stepping(clip_a), stepping(clip_b)],
                    machines: vec![MachineCursor {
                        state: 1,
                        from: Some(0),
                        elapsed: Seconds::new(0.5),
                        duration: Seconds::new(1.0),
                    }],
                },
            )),
            vec![("a".to_owned(), 0.5), ("b".to_owned(), 0.5)]
        );
    }

    #[test]
    fn tree_blend2d_weight_scales_the_leaf_event() {
        // The cell midpoint spreads bilinearly: one quarter to the (0,0)
        // corner, three quarters across the rest (design §8.1).
        assert_eq!(
            tree_fired(|clip_a, clip_b| (
                SkelTree {
                    nodes: vec![
                        TreeNodeDef::Clip { clip: clip_a },
                        TreeNodeDef::Clip { clip: clip_b },
                        TreeNodeDef::Blend2d {
                            axes: [0, 1],
                            children: vec![
                                TreeChild2d {
                                    position: [0.0, 0.0],
                                    node: 0,
                                },
                                TreeChild2d {
                                    position: [1.0, 0.0],
                                    node: 1,
                                },
                                TreeChild2d {
                                    position: [0.0, 1.0],
                                    node: 1,
                                },
                                TreeChild2d {
                                    position: [1.0, 1.0],
                                    node: 1,
                                },
                            ],
                        },
                    ],
                    params: vec![
                        TreeParam {
                            name: "x".to_owned(),
                            default: 0.0,
                        },
                        TreeParam {
                            name: "y".to_owned(),
                            default: 0.0,
                        },
                    ],
                },
                SkelTreePlayer {
                    params: vec![0.5, 0.5],
                    leaves: vec![stepping(clip_a), stepping(clip_b)],
                    machines: Vec::new(),
                },
            )),
            vec![("a".to_owned(), 0.25), ("b".to_owned(), 0.75)]
        );
    }

    #[test]
    fn forward_step_fires_a_crossed_mark_once() {
        let events = marks(&[("foot", 0.5)]);
        assert_eq!(fired(&events, 0.25, 0.25, 1.0, false), vec!["foot"]);
        // Already behind the mark: no re-fire.
        assert_eq!(fired(&events, 0.5, 0.25, 1.0, false), Vec::<String>::new());
        // Never reached this step.
        assert_eq!(fired(&events, 0.0, 0.25, 1.0, false), Vec::<String>::new());
    }

    #[test]
    fn backward_step_fires_in_reverse_crossing_order() {
        let events = marks(&[("early", 0.25), ("late", 0.75)]);
        // Backwards from 0.9 to 0.1: `late` is passed before `early`.
        assert_eq!(fired(&events, 0.9, -0.8, 1.0, false), vec!["late", "early"]);
    }

    #[test]
    fn forward_wrap_fires_tail_before_head() {
        let events = marks(&[("head", 0.05), ("tail", 0.95)]);
        // 0.9 → 1.1 wrapped to 0.1: the tail mark is passed first.
        assert_eq!(fired(&events, 0.9, 0.2, 1.0, true), vec!["tail", "head"]);
    }

    #[test]
    fn backward_wrap_fires_head_after_the_tail() {
        let events = marks(&[("head", 0.05), ("tail", 0.95)]);
        // 0.1 → -0.1 wrapped to 0.9: downward through 0.0 passes `head`
        // first, then `tail` on the wrapped segment.
        assert_eq!(fired(&events, 0.1, -0.2, 1.0, true), vec!["head", "tail"]);
    }

    #[test]
    fn a_full_loop_step_fires_each_mark_exactly_once() {
        let events = marks(&[("a", 0.25), ("b", 0.75)]);
        // One full loop from 0.5: `b` is passed before the wrap, `a`
        // after it — crossing order, each exactly once.
        assert_eq!(fired(&events, 0.5, 1.0, 1.0, true), vec!["b", "a"]);
    }

    #[test]
    fn non_looping_steps_clamp_without_wrapping() {
        let events = marks(&[("early", 0.2), ("late", 0.8)]);
        // Clamped at the end: only the reached mark fires, no wrap.
        assert_eq!(fired(&events, 0.7, 0.5, 1.0, false), vec!["late"]);
        // Clamped at the start going backwards: only `early` fires.
        assert_eq!(fired(&events, 0.3, -0.5, 1.0, false), vec!["early"]);
    }

    #[test]
    fn a_mark_at_zero_fires_only_on_an_arrival_or_wrap() {
        let events = marks(&[("start", 0.0)]);
        // A cursor starting on the mark does not cross it.
        assert_eq!(fired(&events, 0.0, 0.25, 1.0, true), Vec::<String>::new());
        // The wrap passes it.
        assert_eq!(fired(&events, 0.9, 0.2, 1.0, true), vec!["start"]);
        // A backward arrival at the start passes it.
        assert_eq!(fired(&events, 0.3, -0.3, 1.0, false), vec!["start"]);
    }

    #[test]
    fn fire_order_breaks_ties_by_author_order() {
        let events = marks(&[("first", 0.5), ("second", 0.5)]);
        assert_eq!(
            fired(&events, 0.25, 0.25, 1.0, false),
            vec!["first", "second"]
        );
    }

    #[test]
    fn zero_or_degenerate_steps_fire_nothing() {
        let events = marks(&[("foot", 0.5)]);
        assert_eq!(fired(&events, 0.25, 0.0, 1.0, true), Vec::<String>::new());
        assert_eq!(fired(&events, 0.25, 0.25, 0.0, true), Vec::<String>::new());
    }

    #[test]
    fn invalid_marks_never_fire() {
        let events = marks(&[("bad-low", -0.5), ("bad-high", 1.5), ("good", 0.5)]);
        assert_eq!(fired(&events, 0.25, 0.25, 1.0, false), vec!["good"]);
    }

    /// Fixture: a looping skel player at `time` on a 1s clip plus the
    /// events of `pairs`, all in a fresh store.
    fn player_store(time: f32, pairs: &[(&str, f32)]) -> (SmartStore, Entity) {
        let mut store = SmartStore::new();
        let clip = store.create_entity();
        store.insert_cold(
            clip,
            SkelClip {
                name: "walk".to_owned(),
                duration: 1.0,
                tracks: Vec::new(),
            },
        );
        store.insert_cold(clip, marks(pairs));
        let root = store.create_entity();
        store.insert(
            root,
            SkelPlayer {
                clip: ClipId(clip),
                time: Seconds::new(time),
                speed: Seconds::new(1.0),
                weight: Clamped01::new(1.0),
                playing: true,
                looping: true,
            },
        );
        store.insert(root, AnimEventQueue::default());
        (store, root)
    }

    #[test]
    fn advanced_players_fill_the_queue_and_drain_empties_it() {
        let (store, root) = player_store(0.25, &[("foot", 0.5)]);
        advance_skel_players(&store, 0.25);
        {
            let queues = store.read_lane::<AnimEventQueue>().expect("queue lane");
            let queue = queues.get(root).expect("queue on root");
            assert_eq!(queue.fired.len(), 1);
            assert_eq!(queue.fired[0].event, "foot");
        }
        // A paused cursor fires nothing; drain empties the queue.
        let mut queues = store.write_lane::<AnimEventQueue>().expect("queue lane");
        assert_eq!(queues.get_mut(root).expect("queue").drain().count(), 1);
        drop(queues);
        {
            let mut players = store.write_lane::<SkelPlayer>().expect("player lane");
            players.get_mut(root).expect("player").playing = false;
        }
        advance_skel_players(&store, 0.25);
        let queues = store.read_lane::<AnimEventQueue>().expect("queue lane");
        assert_eq!(queues.get(root).expect("queue").fired.len(), 0);
    }

    #[test]
    fn cursors_without_a_queue_or_events_fire_nothing() {
        let mut store = SmartStore::new();
        let clip = store.create_entity();
        store.insert_cold(
            clip,
            SkelClip {
                name: "walk".to_owned(),
                duration: 1.0,
                tracks: Vec::new(),
            },
        );
        store.insert_cold(clip, marks(&[("foot", 0.5)]));
        let root = store.create_entity();
        store.insert(
            root,
            SkelPlayer {
                clip: ClipId(clip),
                time: Seconds::new(0.25),
                speed: Seconds::new(1.0),
                weight: Clamped01::new(1.0),
                playing: true,
                looping: true,
            },
        );
        // No `AnimEventQueue` on the root: the advance must not panic or
        // invent one.
        advance_skel_players(&store, 0.25);
        assert!(store.read_lane::<AnimEventQueue>().is_none());
    }
}
