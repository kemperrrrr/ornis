//! Contact-hooks pins: library one-way platform ([`OneWayPlatform`]),
//! conveyor belt (friction + surface-velocity override), impact-force
//! readout (`accumulated_impulse` / `approach_speed`), solver flags
//! (`READ_ONLY` sensing without solving), sensor-pair filtering
//! (`filter_intersection_pair`), manifold reduction
//! ([`ContactView::retain_points`]), bool-compatible flags, and the
//! no-op-hook bit-identity gate (hooks attached but default = legacy path
//! exactly).

use std::sync::{Arc, Mutex};

use glam::Vec3;
use ornis_physics::engine::{
    ContactHooks, ContactView, ModifyContext, OneWayPlatform, PairFilterContext, SolverFlags,
};
use ornis_physics::{
    BodyHandle, ContactEventKind, PhysicsEngine, RigidBody, SequentialImpulseEngine,
    TriggerEventKind,
};

/// A ball shot upward from below must cross the platform, then fall back
/// and come to rest standing on top of it.
#[test]
fn one_way_platform_passes_from_below_holds_from_above() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let platform = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(2.0, 0.1, 2.0),
        0.0,
    ));
    let mut ball = RigidBody::new_sphere(Vec3::new(0.0, -3.0, 0.0), 0.5, 1.0);
    ball.velocity = Vec3::new(0.0, 10.0, 0.0);
    let ball_h = physics.add_body(ball);
    physics.set_contact_hooks(Some(Box::new(OneWayPlatform::new(platform))));

    // Phase 1: the rising ball must cross the platform plane (top at 0.1).
    let mut crossed = false;
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
        if physics.get_body(ball_h).expect("ball live").position.y > 1.5 {
            crossed = true;
            break;
        }
    }
    assert!(
        crossed,
        "ball shot up at 10 m/s must pass the one-way platform"
    );

    // Phase 2: it falls back and must land on top (rest center 0.1 + 0.5).
    for _ in 0..900 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(ball_h).expect("ball live");
    assert!(
        (b.position.y - 0.6).abs() < 0.05,
        "ball must rest on the platform top, got y={}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.2,
        "ball must settle, got v={:?}",
        b.velocity
    );
}

/// Conveyor belt: a friction + surface-velocity override on the ground pair
/// must carry a resting box along +X; without hooks the box stays put.
struct Belt {
    ground: BodyHandle,
    speed: f32,
}

impl ContactHooks for Belt {
    fn filter_pair(
        &self,
        _a: BodyHandle,
        _b: BodyHandle,
        _ctx: &PairFilterContext<'_>,
    ) -> SolverFlags {
        SolverFlags::COMPUTE_IMPULSES
    }

    fn modify_contact(&self, contact: &mut ContactView, _ctx: &ModifyContext<'_>) {
        if contact.body_a == self.ground || contact.body_b == self.ground {
            contact.friction = 1.5;
            contact.surface_velocity = Vec3::new(self.speed, 0.0, 0.0);
        }
    }
}

fn belt_scene() -> (SequentialImpulseEngine, BodyHandle) {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    let klein = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    (physics, klein)
}

#[test]
fn conveyor_belt_carries_box_control_stays() {
    // Control: no hooks, the box just lands and stays near x = 0.
    let (mut control, control_box) = belt_scene();
    for _ in 0..180 {
        control.step(1.0 / 60.0);
    }
    let c = control.get_body(control_box).expect("box live");
    assert!(
        c.position.x.abs() < 0.05,
        "control box must not drift, got x={}",
        c.position.x
    );

    // Belt at 2 m/s for 3 s: the box must ride along +X.
    let (mut physics, klein) = belt_scene();
    let ground = BodyHandle::from_raw(0);
    physics.set_contact_hooks(Some(Box::new(Belt { ground, speed: 2.0 })));
    for _ in 0..180 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(klein).expect("box live");
    assert!(
        b.position.x > 2.0,
        "belt must carry the box along +X, got x={}",
        b.position.x
    );
    assert!(
        (b.position.y - 0.5).abs() < 0.1,
        "box must stay on the belt, got y={}",
        b.position.y
    );
}

/// A sphere dropped from 5 m must register a hard impact in the hook
/// (approach well above the 1 m/s hit floor) and build up resting
/// support impulse afterwards; the probe values are asserted through a
/// hook that records the hardest pre-solve approach and the largest
/// warm-started normal impulse into a shared tape drained after the run.
#[test]
fn contact_force_probe_sees_impact_and_support() {
    #[derive(Debug, Default)]
    struct Tape {
        max_approach: f32,
        max_impulse: f32,
    }

    struct Taped(Arc<Mutex<Tape>>);

    impl ContactHooks for Taped {
        fn modify_contact(&self, contact: &mut ContactView, _ctx: &ModifyContext<'_>) {
            let mut t = self.0.lock().expect("tape lock");
            t.max_approach = t.max_approach.max(contact.approach_speed);
            t.max_impulse = t.max_impulse.max(contact.accumulated_impulse);
        }
    }

    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5, 1.0));
    let tape = Arc::new(Mutex::new(Tape::default()));
    physics.set_contact_hooks(Some(Box::new(Taped(Arc::clone(&tape)))));
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    let t = tape.lock().expect("tape lock");
    // Free fall from ~4.5 m of gap: impact near 9 m/s, far above the hit floor.
    assert!(
        t.max_approach > 4.0,
        "hook must see the hard impact, got {}",
        t.max_approach
    );
    assert!(
        t.max_impulse > 1e-6,
        "hook must see resting support impulse, got {}",
        t.max_impulse
    );
}

/// No-op hook (all defaults) must be bit-identical to no hooks: same
/// scene, same steps, exact float equality on every body.
struct Noop;

impl ContactHooks for Noop {}

fn snapshot_scene() -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    for i in 0..3 {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5 + i as f32 * 1.02, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
    }
    let mut fast = RigidBody::new_box(Vec3::new(-3.0, 6.0, 0.0), Vec3::splat(0.4), 1.0);
    fast.velocity = Vec3::new(0.0, -20.0, 0.0);
    physics.add_body(fast);
    physics
}

fn render(physics: &SequentialImpulseEngine) -> Vec<u32> {
    let mut out = Vec::new();
    for b in &physics.bodies {
        for x in b
            .position
            .to_array()
            .into_iter()
            .chain(b.orientation.to_array())
            .chain(b.velocity.to_array())
            .chain(b.angular_velocity.to_array())
        {
            out.push(x.to_bits());
        }
    }
    out
}

#[test]
fn noop_hook_is_bit_identical_to_no_hooks() {
    let mut bare = snapshot_scene();
    for _ in 0..120 {
        bare.step(1.0 / 60.0);
    }
    let mut hooked = snapshot_scene();
    hooked.set_contact_hooks(Some(Box::new(Noop)));
    for _ in 0..120 {
        hooked.step(1.0 / 60.0);
    }
    assert_eq!(render(&bare), render(&hooked));
}

/// `READ_ONLY` senses without solving: the manifold is built (the hook
/// observes every substep with a zeroed impulse), begin events still fire,
/// but no impulse ever holds the body — it falls straight through.
#[derive(Debug, Default)]
struct ReadOnlyTape {
    calls: usize,
    max_approach: f32,
    max_impulse: f32,
}

struct ReadOnly {
    ground: BodyHandle,
    tape: Arc<Mutex<ReadOnlyTape>>,
}

impl ContactHooks for ReadOnly {
    fn filter_pair(
        &self,
        a: BodyHandle,
        _b: BodyHandle,
        _ctx: &PairFilterContext<'_>,
    ) -> SolverFlags {
        if a == self.ground {
            SolverFlags::READ_ONLY
        } else {
            SolverFlags::COMPUTE_IMPULSES
        }
    }

    fn modify_contact(&self, contact: &mut ContactView, _ctx: &ModifyContext<'_>) {
        let mut t = self.tape.lock().expect("tape lock");
        t.calls += 1;
        t.max_approach = t.max_approach.max(contact.approach_speed);
        t.max_impulse = t.max_impulse.max(contact.accumulated_impulse);
    }
}

#[test]
fn read_only_pair_senses_without_solving() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let ground = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    let klein = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let tape = Arc::new(Mutex::new(ReadOnlyTape::default()));
    physics.set_contact_hooks(Some(Box::new(ReadOnly {
        ground,
        tape: Arc::clone(&tape),
    })));
    for _ in 0..180 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(klein).expect("box live");
    assert!(
        b.position.y < -0.5,
        "read-only ground must not hold the box, got y={}",
        b.position.y
    );
    let t = tape.lock().expect("tape lock");
    assert!(
        t.calls > 0,
        "read-only manifold must still be built (modify runs)"
    );
    assert!(
        t.max_approach > 1.0,
        "hook must see the fall, got {}",
        t.max_approach
    );
    assert_eq!(
        t.max_impulse, 0.0,
        "read-only pair must never accumulate impulse"
    );
    let events = physics.drain_contact_events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e.kind, ContactEventKind::Begin)),
        "read-only touch must still emit Begin, got {events:?}"
    );
}

/// Sensor veto: a hook rejecting the intersection pair suppresses the
/// trigger enter event the same scene reports without hooks.
struct BlockIntersections;

impl ContactHooks for BlockIntersections {
    fn filter_intersection_pair(
        &self,
        _a: BodyHandle,
        _b: BodyHandle,
        _ctx: &PairFilterContext<'_>,
    ) -> bool {
        false
    }
}

fn sensor_scene() -> (SequentialImpulseEngine, BodyHandle, BodyHandle) {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let sensor =
        physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(1.0), 0.0).with_trigger(true));
    let ball = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
    (physics, sensor, ball)
}

#[test]
fn intersection_filter_blocks_sensor_events() {
    let (mut blocked, _sensor, _ball) = sensor_scene();
    blocked.set_contact_hooks(Some(Box::new(BlockIntersections)));
    for _ in 0..5 {
        blocked.step(1.0 / 60.0);
    }
    assert!(
        blocked.drain_trigger_events().is_empty(),
        "blocked sensor pair must emit no events"
    );

    let (mut control, _sensor, _ball) = sensor_scene();
    for _ in 0..5 {
        control.step(1.0 / 60.0);
    }
    let events = control.drain_trigger_events();
    assert!(
        events.iter().any(|e| e.kind == TriggerEventKind::Entered),
        "unfiltered sensor pair must emit Entered, got {events:?}"
    );
}

/// Manifold reduction: a hook keeping only the deepest point must observe
/// a full 4-point face contact, shrink every manifold to one lane, and
/// still leave the stack standing.
#[derive(Debug, Default)]
struct ReduceTape {
    max_before: usize,
    min_after: usize,
}

struct ReduceToDeepest {
    tape: Arc<Mutex<ReduceTape>>,
}

impl ContactHooks for ReduceToDeepest {
    fn modify_contact(&self, contact: &mut ContactView, _ctx: &ModifyContext<'_>) {
        let before = contact.points.len();
        let deepest = contact
            .points
            .iter()
            .map(|p| p.penetration)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut kept = false;
        contact.retain_points(|p| {
            if !kept && p.penetration >= deepest {
                kept = true;
                true
            } else {
                false
            }
        });
        let mut t = self.tape.lock().expect("tape lock");
        t.max_before = t.max_before.max(before);
        t.min_after = if t.min_after == 0 {
            contact.points.len()
        } else {
            t.min_after.min(contact.points.len())
        };
    }
}

#[test]
fn retain_points_shrinks_face_contact_to_one_lane() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    let klein = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let tape = Arc::new(Mutex::new(ReduceTape::default()));
    physics.set_contact_hooks(Some(Box::new(ReduceToDeepest {
        tape: Arc::clone(&tape),
    })));
    for _ in 0..240 {
        physics.step(1.0 / 60.0);
    }
    let t = tape.lock().expect("tape lock");
    assert_eq!(
        t.max_before, 4,
        "resting face contact must build a 4-point manifold, saw {}",
        t.max_before
    );
    assert_eq!(
        t.min_after, 1,
        "every manifold must shrink to one lane, saw min {}",
        t.min_after
    );
    let b = physics.get_body(klein).expect("box live");
    assert!(
        (b.position.y - 0.5).abs() < 0.1,
        "single-lane stack must stand, got y={}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.3,
        "single-lane stack must settle, got v={:?}",
        b.velocity
    );
}

/// Legacy `bool` polarity survives on the flag set: `true` solves,
/// `false` skips.
#[test]
fn solver_flags_stay_bool_compatible() {
    assert_eq!(SolverFlags::from(true), SolverFlags::COMPUTE_IMPULSES);
    assert_eq!(SolverFlags::from(false), SolverFlags::SKIP);
    assert!(bool::from(SolverFlags::COMPUTE_IMPULSES));
    assert!(bool::from(SolverFlags::READ_ONLY));
    assert!(!bool::from(SolverFlags::SKIP));
}
