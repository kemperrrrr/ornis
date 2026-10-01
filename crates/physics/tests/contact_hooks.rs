//! Contact-hooks pins: one-way platform (`filter_pair`), conveyor belt
//! (friction + surface-velocity override), impact-force readout
//! (`accumulated_impulse` / `approach_speed`), and the no-op-hook
//! bit-identity gate (hooks attached but default = legacy path exactly).

use std::sync::Mutex;

use glam::Vec3;
use ornis_physics::engine::{ContactHooks, ContactView, ModifyContext, PairFilterContext};
use ornis_physics::{BodyHandle, PhysicsEngine, RigidBody, SequentialImpulseEngine};

/// One-way platform: pairs with the platform pass through while the other
/// body moves up fast, collide otherwise.
struct OneWay {
    platform: BodyHandle,
}

impl ContactHooks for OneWay {
    fn filter_pair(&self, a: BodyHandle, b: BodyHandle, ctx: &PairFilterContext<'_>) -> bool {
        let other = if a == self.platform {
            ctx.body_b
        } else if b == self.platform {
            ctx.body_a
        } else {
            return true;
        };
        other.velocity.y <= 1.0
    }
}

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
    physics.set_contact_hooks(Some(Box::new(OneWay { platform })));

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
    fn filter_pair(&self, _a: BodyHandle, _b: BodyHandle, _ctx: &PairFilterContext<'_>) -> bool {
        true
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
    use std::sync::Arc;

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
