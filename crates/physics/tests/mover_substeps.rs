//! R2 kinematic movers + R7 explicit substep counts.
//!
//! Movers are engine-driven kinematic platforms: the host declares a
//! per-step displacement, the engine moves the platform and carries
//! contacting passengers through the Box3D-style plane solve with a slop.
//! `step_with_substeps(dt, n)` is the Box3D `b3World_Step(dt, subStepCount)`
//! parity entry point (`1..=64`, explicit refusal outside).

use glam::Vec3;
use ornis_physics::{
    BodyHandle, BodyType, Engine, MoverError, MoverHandle, PhysicsEngine, RigidBody,
    SequentialImpulseEngine, SolverKind, StepError,
};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn kinematic_box(position: Vec3, half_extents: Vec3) -> RigidBody {
    let mut body = RigidBody::new_box(position, half_extents, 1.0);
    body.body_type = BodyType::Kinematic;
    body
}

/// Platform (top at y=0) plus a rider box resting exactly on it.
fn platform_with_rider() -> (SequentialImpulseEngine, BodyHandle, BodyHandle) {
    let mut physics = SequentialImpulseEngine::new(GRAVITY);
    let platform = physics.add_body(kinematic_box(
        Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(2.0, 0.25, 2.0),
    ));
    let rider = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    (physics, platform, rider)
}

fn attach_mover(
    physics: &mut SequentialImpulseEngine,
    platform: BodyHandle,
    displacement: Vec3,
) -> ornis_physics::MoverHandle {
    let mover = physics
        .add_mover(platform)
        .expect("kinematic platform takes a mover");
    physics
        .set_mover_displacement(mover, displacement)
        .expect("finite displacement stores");
    physics.set_mover_active(mover, true).expect("valid handle");
    mover
}

/// Horizontal ride: the platform moves 2 m/s, the rider must travel with
/// it — no slip, no sink, no penetration.
#[test]
fn mover_carries_box_without_slip_or_sink() {
    let (mut physics, platform, rider) = platform_with_rider();
    for _ in 0..30 {
        physics.step(DT);
    }
    attach_mover(&mut physics, platform, Vec3::new(2.0 * DT, 0.0, 0.0));
    for _ in 0..60 {
        physics.step(DT);
    }
    let platform_pos = physics.get_body(platform).unwrap().position;
    let rider_pos = physics.get_body(rider).unwrap().position;
    assert!(
        (platform_pos.x - 2.0).abs() < 0.05,
        "platform must travel its 2 m displacement, x={}",
        platform_pos.x
    );
    assert!(
        (rider_pos.x - platform_pos.x).abs() < 0.05,
        "rider must not slip: rider x={}, platform x={}",
        rider_pos.x,
        platform_pos.x
    );
    assert!(
        (rider_pos.y - 0.5).abs() < 0.05,
        "rider must not sink or launch: y={}",
        rider_pos.y
    );
    let penetration = (platform_pos.y + 0.25) - (rider_pos.y - 0.5);
    // Awake-rest band: the NGS position slop is 2 cm and an awake contact
    // sags on top of it (an asleep control rests at exactly 0) — the mover
    // must stay inside that band, never press through it.
    assert!(
        penetration <= 0.05,
        "no press-through beyond the awake-rest band: pen={penetration}"
    );
    assert!(
        !physics.is_asleep(rider),
        "the riding passenger stays awake while the mover displaces"
    );
}

/// Vertical lift: the rider rises with the platform, positionally — no
/// bounce (velocity stays small) and no press-through.
#[test]
fn mover_lift_carries_without_bounce() {
    let (mut physics, platform, rider) = platform_with_rider();
    for _ in 0..30 {
        physics.step(DT);
    }
    attach_mover(&mut physics, platform, Vec3::new(0.0, 1.0 * DT, 0.0));
    for _ in 0..60 {
        physics.step(DT);
    }
    let platform_pos = physics.get_body(platform).unwrap().position;
    let rider = physics.get_body(rider).unwrap();
    assert!(
        (platform_pos.y - 0.75).abs() < 0.05,
        "lift must rise 1 m, y={}",
        platform_pos.y
    );
    assert!(
        ((rider.position.y - platform_pos.y) - 0.75).abs() < 0.05,
        "rider offset must hold: rider y={}",
        rider.position.y
    );
    assert!(
        rider.velocity.length() < 0.5,
        "positional carry must not bounce the rider: {:?}",
        rider.velocity
    );
    let penetration = (platform_pos.y + 0.25) - (rider.position.y - 0.5);
    assert!(
        penetration <= 0.05,
        "lift must not press into the rider: pen={penetration}"
    );
}

/// Stopping the platform: the passenger stays put and the island re-sleeps
/// (zero displacement holds position without pinning wakefulness).
#[test]
fn mover_stop_lets_passenger_sleep() {
    let (mut physics, platform, rider) = platform_with_rider();
    for _ in 0..30 {
        physics.step(DT);
    }
    let mover = attach_mover(&mut physics, platform, Vec3::new(2.0 * DT, 0.0, 0.0));
    for _ in 0..30 {
        physics.step(DT);
    }
    // Stop: zero displacement, still active (holding, not driving).
    physics
        .set_mover_displacement(mover, Vec3::ZERO)
        .expect("zero stores");
    let stop_x = physics.get_body(rider).unwrap().position.x;
    let mut slept_at = None;
    for step in 0..600 {
        physics.step(DT);
        if physics.is_asleep(rider) {
            slept_at = Some(step);
            break;
        }
    }
    assert!(
        slept_at.is_some(),
        "the passenger must re-sleep after the stop"
    );
    let rider_pos = physics.get_body(rider).unwrap().position;
    assert!(
        (rider_pos.x - stop_x).abs() < 0.1,
        "the stopped passenger stays: stop x={stop_x}, now x={}",
        rider_pos.x
    );
    assert!(
        (rider_pos.y - 0.5).abs() < 0.05,
        "the stopped passenger rests: y={}",
        rider_pos.y
    );
}

/// Fast (above-gate) mover displacement flows through CCD travel: the
/// platform pose is exact, a thin sleeping victim in the path wakes and is
/// pushed instead of tunneled.
#[test]
fn mover_fast_displacement_uses_ccd_travel() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let victim = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(0.02, 0.5, 0.5),
        1.0,
    ));
    for _ in 0..30 {
        physics.step(DT);
    }
    assert!(physics.is_asleep(victim), "victim must sleep in zero-g");
    let platform = physics.add_body(kinematic_box(
        Vec3::new(-3.0, 0.0, 0.0),
        Vec3::new(0.5, 1.0, 1.0),
    ));
    attach_mover(&mut physics, platform, Vec3::new(1.0, 0.0, 0.0));
    for _ in 0..6 {
        physics.step(DT);
    }
    let platform_x = physics.get_body(platform).unwrap().position.x;
    assert!(
        (platform_x - 3.0).abs() < 1e-4,
        "the mover owns the platform pose: x={platform_x}"
    );
    assert!(
        !physics.is_asleep(victim),
        "the pass-through must wake the victim"
    );
    let victim_body = physics.get_body(victim).unwrap();
    assert!(
        victim_body.position.x > 1.0,
        "the victim rides the fast platform, x={}",
        victim_body.position.x
    );
}

/// Mover admission is explicit: dynamics and bad handles are refused, and
/// `remove_body` drops the mover driving the removed platform.
#[test]
fn mover_admission_is_explicit() {
    let mut physics = SequentialImpulseEngine::new(GRAVITY);
    let dynamic = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    assert_eq!(
        physics.add_mover(dynamic),
        Err(MoverError::NotKinematic {
            handle: dynamic.index()
        })
    );
    let platform = physics.add_body(kinematic_box(Vec3::ZERO, Vec3::splat(1.0)));
    let mover = physics.add_mover(platform).expect("kinematic attaches");
    physics.remove_body(platform);
    assert_eq!(
        physics.mover_count(),
        0,
        "removing the platform drops its mover"
    );
    assert!(
        physics.mover(mover).is_none(),
        "the dropped mover handle reads back empty"
    );
    assert_eq!(
        physics.set_mover_displacement(MoverHandle::from_raw(3), Vec3::X),
        Err(MoverError::UnknownMover { handle: 3 })
    );
}

/// Out-of-range counts are refused explicitly — never clamped silently —
/// and a valid call restores the configured cap afterwards.
#[test]
fn step_with_substeps_rejects_out_of_range() {
    let mut physics = SequentialImpulseEngine::new(GRAVITY);
    assert_eq!(
        physics.step_with_substeps(DT, 0),
        Err(StepError::BadSubstepCount {
            got: 0,
            min: 1,
            max: 64
        })
    );
    assert_eq!(
        physics.step_with_substeps(DT, 65),
        Err(StepError::BadSubstepCount {
            got: 65,
            min: 1,
            max: 64
        })
    );
    assert_eq!(physics.substeps(), 12, "refusals leave the cap alone");
    physics.step_with_substeps(DT, 1).expect("1 is valid");
    physics.step_with_substeps(DT, 64).expect("64 is valid");
    assert_eq!(physics.substeps(), 12, "valid calls restore the cap");
}

/// `step_with_substeps(dt, 12)` is bit-identical to `step(dt)` at the
/// default cap: the new entry point adds no drift to the default path.
#[test]
fn step_with_substeps_matches_step_at_default_cap() {
    fn scene() -> SequentialImpulseEngine {
        let mut physics = SequentialImpulseEngine::new(GRAVITY);
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
        let mut fast = RigidBody::new_box(Vec3::new(-3.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(0.0, -30.0, 0.0);
        physics.add_body(fast);
        physics
    }
    fn snapshot(physics: &SequentialImpulseEngine) -> Vec<[u32; 13]> {
        physics
            .bodies
            .iter()
            .map(|b| {
                let mut words = [0u32; 13];
                for (k, x) in b
                    .position
                    .to_array()
                    .into_iter()
                    .chain(b.orientation.to_array())
                    .chain(b.velocity.to_array())
                    .chain(b.angular_velocity.to_array())
                    .enumerate()
                {
                    words[k] = x.to_bits();
                }
                words
            })
            .collect()
    }
    let mut via_step = scene();
    for _ in 0..30 {
        via_step.step(DT);
    }
    let mut via_override = scene();
    for _ in 0..30 {
        via_override
            .step_with_substeps(DT, 12)
            .expect("12 is valid");
    }
    assert_eq!(snapshot(&via_step), snapshot(&via_override));
    assert_eq!(
        via_step.step_timing().substeps,
        via_override.step_timing().substeps
    );
}

/// The requested count reaches the solver: a fast scene runs exactly the
/// asked substeps (adaptive only trims below the cap, never above it).
#[test]
fn step_with_substeps_applies_requested_count() {
    fn fast_scene() -> SequentialImpulseEngine {
        let mut physics = SequentialImpulseEngine::new(GRAVITY);
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        let mut fast = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(0.0, -40.0, 0.0);
        physics.add_body(fast);
        physics
    }
    for n in [1, 4, 12] {
        let mut physics = fast_scene();
        physics.step_with_substeps(DT, n).expect("valid count");
        assert_eq!(
            physics.step_timing().substeps,
            n,
            "requested {n} substeps must run"
        );
        assert_eq!(physics.last_substep_shed(), 0, "tiny scene sheds nothing");
    }
}

/// Stiff settling scene (3-box tower): fewer substeps per step mean deeper
/// peak penetration — monotone in the count.
#[test]
fn more_substeps_less_penetration_monotone() {
    fn peak_sink(substeps: u32) -> f32 {
        let mut physics = SequentialImpulseEngine::new(GRAVITY);
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        for level in 0..3 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, 0.4 + level as f32 * 0.82, 0.0),
                Vec3::splat(0.4),
                1.0,
            ));
        }
        let mut peak = 0.0f32;
        for _ in 0..240 {
            physics
                .step_with_substeps(DT, substeps)
                .expect("valid count");
            // Bottom-box sink below its exact rest height (0.4).
            peak = peak.max(0.4 - physics.bodies[1].position.y);
        }
        peak
    }
    let pen1 = peak_sink(1);
    let pen2 = peak_sink(2);
    let pen4 = peak_sink(4);
    eprintln!("SUBSTEP-PEN n=1:{pen1:.5} n=2:{pen2:.5} n=4:{pen4:.5}");
    assert!(
        pen1 >= pen2 && pen2 >= pen4,
        "penetration must shrink monotone with substeps: {pen1} / {pen2} / {pen4}"
    );
    assert!(
        pen1 > pen4,
        "1 vs 4 substeps must differ measurably: {pen1} vs {pen4}"
    );
}

/// Orchestrator parity: the `Engine` seam forwards movers and the substep
/// override on Single/SequentialImpulse, and refuses both explicitly
/// elsewhere.
#[test]
fn engine_forwards_mover_and_substeps() {
    let mut engine = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let platform = engine.add_body(kinematic_box(
        Vec3::new(0.0, -0.25, 0.0),
        Vec3::new(2.0, 0.25, 2.0),
    ));
    let rider = engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    for _ in 0..30 {
        engine.step(DT);
    }
    let mover = engine.add_mover(platform).expect("SI mode hosts movers");
    engine
        .set_mover_displacement(mover, Vec3::new(2.0 * DT, 0.0, 0.0))
        .expect("valid displacement");
    engine.set_mover_active(mover, true).expect("valid handle");
    for _ in 0..60 {
        engine.step(DT);
    }
    let platform_x = engine.get_body(platform).unwrap().position.x;
    let rider_x = engine.get_body(rider).unwrap().position.x;
    assert!(
        (platform_x - 2.0).abs() < 0.05 && (rider_x - platform_x).abs() < 0.05,
        "orchestrated ride: platform x={platform_x}, rider x={rider_x}"
    );
    engine.step_with_substeps(DT, 4).expect("SI mode honors n");
    assert_eq!(
        engine.step_with_substeps(DT, 0),
        Err(StepError::BadSubstepCount {
            got: 0,
            min: 1,
            max: 64
        })
    );

    let mut avbd = Engine::new(SolverKind::Avbd, GRAVITY);
    let body = avbd.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    assert!(
        matches!(avbd.add_mover(body), Err(MoverError::Unsupported { .. })),
        "non-SI solvers refuse movers explicitly"
    );
    assert!(
        matches!(
            avbd.step_with_substeps(DT, 4),
            Err(StepError::Unsupported { .. })
        ),
        "non-SI solvers refuse the substep override explicitly"
    );
}
