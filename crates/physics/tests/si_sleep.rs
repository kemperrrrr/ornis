//! SI sleep pins: island sleep, waking and kinematic/teleport interaction.

use glam::Vec3;
use ornis_physics::{BodyType, PhysicsEngine, RigidBody, SequentialImpulseEngine};

/// Settled-scene economics (100k-tiled probe, StepTiming verdict): an
/// interior resting grid sleeps whole (~14 steps, longest island timer
/// 0.6 s) and the fully-sleeping fast path then reports zero phase
/// work, while the first active step runs real substeps. Dynamics sit
/// strictly inside tile coverage so no edge body tips off and falls
/// forever (the 10k probe's 160 perpetual awake are that scene
/// overhang, not solver jitter).
#[test]
fn settled_grid_sleeps_and_costs_less_than_active() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    for tx in -1..=1 {
        for tz in -1..=1 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(tx as f32 * 10.0, -0.5, tz as f32 * 10.0),
                Vec3::new(5.0, 0.5, 5.0),
                0.0,
            ));
        }
    }
    let mut dynamics = Vec::new();
    for gx in -2..=2 {
        for gz in -2..=2 {
            dynamics.push(physics.add_body(RigidBody::new_box(
                Vec3::new(gx as f32 * 2.0, 0.4, gz as f32 * 2.0),
                Vec3::splat(0.4),
                1.0,
            )));
        }
    }
    physics.step(1.0 / 60.0);
    assert!(
        dynamics.iter().all(|&h| !physics.is_asleep(h)),
        "first step is active: nothing sleeps yet"
    );
    assert!(
        physics.step_timing().substeps > 0,
        "first step runs real substeps"
    );
    for _ in 0..240 {
        physics.step(1.0 / 60.0);
    }
    assert!(
        dynamics.iter().all(|&h| physics.is_asleep(h)),
        "interior resting grid must sleep whole"
    );
    let timing = physics.step_timing();
    assert_eq!(
        timing.substeps, 0,
        "fully-sleeping fast path reports zero phase work, got {timing:?}"
    );
    for &h in &dynamics {
        let b = physics.get_body(h).unwrap();
        assert!(
            (b.position.y - 0.4).abs() < 0.1,
            "settled body rest height, got {}",
            b.position.y
        );
        assert!(
            b.velocity.length() < 0.05,
            "settled body not quiet: {:?}",
            b.velocity
        );
    }
}

/// World-scale sleep: statics are born asleep (they never move), dynamics
/// are born awake. Every frozen-pair skip keys off this from step one.
#[test]
fn statics_are_born_asleep_dynamics_awake() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let floor = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let free = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    assert!(physics.is_asleep(floor), "static must be born asleep");
    assert!(!physics.is_asleep(free), "dynamic must be born awake");
}

/// World-scale sleep: a settled stack on a static floor survives a drop
/// impact with the floor still asleep (wake_island on statics is a
/// no-op) while the struck boxes wake and move.
#[test]
fn static_floor_stays_asleep_under_impact() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let floor = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let mut stack = Vec::new();
    for i in 0..3 {
        stack.push(physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5 + i as f32, 0.0),
            Vec3::splat(0.5),
            1.0,
        )));
    }
    for _ in 0..150 {
        physics.step(1.0 / 60.0);
    }
    for &h in &stack {
        assert!(physics.is_asleep(h), "stack must settle before the drop");
    }
    let top_y = physics.get_body(stack[2]).unwrap().position.y;
    let mut drop = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    drop.velocity = Vec3::new(0.0, -20.0, 0.0);
    physics.add_body(drop);
    // Rigid resting stacks barely displace under load — the observable
    // is the transient wake while the impact churns through, not the
    // final pose (it re-sleeps where it stood).
    let mut woke = false;
    for _ in 0..90 {
        physics.step(1.0 / 60.0);
        woke |= !physics.is_asleep(stack[2]);
    }
    assert!(
        physics.is_asleep(floor),
        "static floor must stay asleep through the impact"
    );
    let moved = physics.get_body(stack[2]).unwrap().position.y;
    assert!(
        woke,
        "struck stack must transiently wake (top {top_y} -> {moved})"
    );
}

/// World-scale sleep: a driven kinematic wall plows into a sleeping box
/// — the sleeper must wake and be pushed, never ghosted through. (Before
/// the asleep-flag cleanup, kinematic contacts were dropped in the
/// active-manifold filter and sleepers were intangible to drivers.)
#[test]
fn kinematic_wall_wakes_and_pushes_sleeper() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let sleeper = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..90 {
        physics.step(1.0 / 60.0);
    }
    assert!(
        physics.is_asleep(sleeper),
        "box must settle before the plow"
    );
    let mut wall = RigidBody::new_box(Vec3::new(-3.0, 0.5, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
    wall.body_type = BodyType::Kinematic;
    let wall_h = physics.add_body(wall);
    // Driven body, done consistently: the driver sets positions AND the
    // matching velocity field (approach wake, margins and CCD all read
    // velocities — a zero-velocity teleport is invisible to them).
    for _ in 0..120 {
        let w = physics.get_body_mut(wall_h).unwrap();
        w.velocity = Vec3::new(2.0, 0.0, 0.0);
        w.position.x += 2.0 / 60.0;
        physics.step(1.0 / 60.0);
    }
    let pushed = physics.get_body(sleeper).unwrap().position.x;
    assert!(
        pushed > 0.5,
        "kinematic wall must push the sleeper (x={pushed}), not ghost through"
    );
}

/// Penetration wake, isolated: a box spawned 5 cm deep inside a sleeper
/// with zero velocities must still wake it — the approach test is blind
/// here (no velocity field), overlap is the only signal. (Teleporting
/// drivers hit this path every frame.)
#[test]
fn teleport_overlap_wakes_sleeper_without_velocity() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let sleeper = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
    }
    assert!(physics.is_asleep(sleeper), "box must sleep in zero-g");
    // Spawn overlapping: intruder bottom 5 cm inside the sleeper top.
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.45, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..5 {
        physics.step(1.0 / 60.0);
    }
    assert!(
        !physics.is_asleep(sleeper),
        "deep overlap must wake the sleeper even at zero approach speed"
    );
}

/// Kinematic CCD deferred with proof: thin SLEEPING victim (4 cm) vs a
/// fast kinematic wall (10 m/s) at full 12 substeps. The speculative
/// margin provably covers travel (margin >= v*dt always), so the
/// discrete path already carries the victim without tunneling (rode
/// 7.5 m here) — the kinematic sweep (`solve_kinematic_sweep`) only
/// answers for larger per-step segments (teleports), not here: each
/// 1/6 m step sits below the travel gate.
#[test]
fn fast_kinematic_plow_carries_thin_sleeper() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let victim = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.02, 0.5, 0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
    }
    assert!(physics.is_asleep(victim), "victim must sleep in zero-g");
    let mut wall = RigidBody::new_box(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
    wall.body_type = BodyType::Kinematic;
    let wall_h = physics.add_body(wall);
    for _ in 0..60 {
        let w = physics.get_body_mut(wall_h).unwrap();
        w.velocity = Vec3::new(10.0, 0.0, 0.0);
        w.position.x += 10.0 / 60.0;
        physics.step(1.0 / 60.0);
    }
    let vx = physics.get_body(victim).unwrap().position.x;
    assert!(vx > 1.0, "victim must ride the plow, not tunnel (x={vx})");
}

/// Kinematic sweep, pass-through case: a wall TELEPORTED 1 m per step
/// with a zero velocity field crosses a thin sleeping victim. The
/// discrete phase can never see it (no end pose overlaps within any
/// margin), so without the sweep the victim would stay asleep at x=0 —
/// this test fails on the pre-sweep code (negative control verified by
/// stashing the sweep). With the sweep the victim wakes and takes the
/// normal approach one-shot, while the driver pose stays owned by the
/// driver.
#[test]
fn teleported_kinematic_wall_cannot_tunnel_thin_sleeper() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let victim = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.02, 0.5, 0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
    }
    assert!(physics.is_asleep(victim), "victim must sleep in zero-g");
    let mut wall = RigidBody::new_box(Vec3::new(-3.0, 0.0, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
    wall.body_type = BodyType::Kinematic;
    // Zero velocity field on purpose: the driver teleports only.
    let wall_h = physics.add_body(wall);
    for _ in 0..6 {
        physics.get_body_mut(wall_h).unwrap().position.x += 1.0;
        physics.step(1.0 / 60.0);
    }
    // Driver ownership: the sweep never moves the mover.
    let wall_x = physics.get_body(wall_h).unwrap().position.x;
    assert!(
        (wall_x - 3.0).abs() < 1e-4,
        "sweep must not move the driver (x={wall_x})"
    );
    assert!(
        !physics.is_asleep(victim),
        "teleport pass-through must wake the victim"
    );
    let v = physics.get_body(victim).unwrap();
    assert!(
        v.position.x > 1.0,
        "victim must be carried forward, not left at x={}",
        v.position.x
    );
    assert!(
        v.velocity.x > 0.0,
        "victim must take the normal approach (vx={})",
        v.velocity.x
    );
}

/// Kinematic sweep, below-gate case (response parity): a SMALL teleport
/// that ends overlapping a resting victim settles positionally — no
/// sweep, no temp velocity, no launch. A zero-velocity nudge into rest
/// imparts no momentum (Box2D parity); the penetration backstop still
/// wakes the victim.
#[test]
fn small_teleport_into_rest_settles_without_launch() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let victim = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
    }
    assert!(physics.is_asleep(victim), "victim must sleep in zero-g");
    let mut wall = RigidBody::new_box(Vec3::new(-1.08, 0.5, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
    wall.body_type = BodyType::Kinematic;
    let wall_h = physics.add_body(wall);
    // 10 cm nudge (below the 25 cm travel gate) ending 2 cm deep in the
    // victim's side (wall front at -0.48 vs victim back at -0.5).
    physics.get_body_mut(wall_h).unwrap().position.x = -0.98;
    for _ in 0..5 {
        physics.step(1.0 / 60.0);
    }
    let v = physics.get_body(victim).unwrap();
    assert!(
        !physics.is_asleep(victim),
        "overlap must wake the victim (penetration backstop)"
    );
    assert!(
        v.velocity.length() < 1.0,
        "resting nudge must not launch the victim (v={})",
        v.velocity.length()
    );
}

/// Kinematic sweep, above-gate case (impact parity): a LARGE teleport
/// ending inside the victim is the honest equivalent of a fast-wall
/// impact at the implied speed — the victim takes the hit. This pins the
/// gate boundary together with `small_teleport_into_rest_settles_without_launch`.
#[test]
fn large_teleport_into_rest_hits_like_fast_wall() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let victim = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(1.0 / 60.0);
    }
    assert!(physics.is_asleep(victim), "victim must sleep in zero-g");
    let mut wall = RigidBody::new_box(Vec3::new(-3.0, 0.5, 0.0), Vec3::new(0.5, 1.0, 1.0), 1.0);
    wall.body_type = BodyType::Kinematic;
    let wall_h = physics.add_body(wall);
    // 2.02 m jump (above the 25 cm gate) ending 2 cm deep in the victim.
    physics.get_body_mut(wall_h).unwrap().position.x = -0.98;
    for _ in 0..5 {
        physics.step(1.0 / 60.0);
    }
    let v = physics.get_body(victim).unwrap();
    assert!(!physics.is_asleep(victim), "impact must wake the victim");
    assert!(
        v.velocity.x > 5.0,
        "above-gate teleport must hit like a fast wall (vx={})",
        v.velocity.x
    );
}

/// Parked kinematics cost nothing: an unmoved zero-velocity kinematic
/// plus a sleeping dynamic still take the fully-sleeping fast path —
/// the driver snapshot must not invent phantom motion.
#[test]
fn parked_kinematic_keeps_sleeping_fast_path() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let victim = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    let mut wall = RigidBody::new_box(Vec3::new(5.0, 0.5, 0.0), Vec3::splat(0.5), 1.0);
    wall.body_type = BodyType::Kinematic;
    physics.add_body(wall);
    for _ in 0..40 {
        physics.step(1.0 / 60.0);
    }
    assert!(
        physics.is_asleep(victim),
        "parked driver must not disturb the sleeper"
    );
}
