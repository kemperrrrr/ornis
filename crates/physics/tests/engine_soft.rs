//! `Engine` soft-body gate (D1, design (b)): soft bodies step, sleep and
//! report events through the orchestrator on its XPBD path; off it they
//! park order-preserving but frozen. Handles stay stable across solver
//! migrations (the `global_handles_survive_islands_rebuilds` precedent).

use glam::Vec3;
use ornis_physics::xpbd::SoftContactEvent;
use ornis_physics::{
    AxisConfig, Engine, JointKind, PhysicsEngine, RigidBody, RoutingKind, SoftBody, SoftHandle,
    SolverKind,
};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn floor() -> RigidBody {
    RigidBody::new_box(Vec3::new(0.0, -5.0, 0.0), Vec3::splat(5.0), 0.0)
}

fn damped_cube(origin: Vec3) -> SoftBody {
    let mut cube = SoftBody::soft_cube(origin, 1.0, 1.0, 0.0, 0.0);
    cube.damping = 4.0;
    cube
}

/// The orchestrator steps soft bodies on the XPBD path: a hanging chain
/// keeps its length instead of stretching like rubber.
#[test]
fn soft_chain_holds_length_through_orchestrator() {
    let mut engine = Engine::new(SolverKind::Xpbd, GRAVITY);
    let rope = engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 6, 0.5, 1.0, 0.0));
    assert_eq!(engine.soft_body_count(), 1);
    for _ in 0..180 {
        engine.step(DT);
    }
    let body = engine.get_soft_body(rope).expect("rope survives");
    let end = body.particles.last().expect("nonempty").position;
    assert!(
        (end.y + 2.5).abs() < 0.1,
        "free end height {}, want ~-2.5",
        end.y
    );
    let mut worst = 0.0f32;
    for c in &body.constraints {
        let d =
            (body.particles[c.a.index()].position - body.particles[c.b.index()].position).length();
        worst = worst.max((d - c.rest).abs() / c.rest);
    }
    assert!(worst < 0.02, "max link stretch {worst}, want <2%");
}

/// Soft↔rigid coupling runs in the one solver: a dropped cube rests on
/// the floor instead of falling through.
#[test]
fn soft_rigid_coupling_through_orchestrator() {
    let mut engine = Engine::new(SolverKind::Xpbd, GRAVITY);
    engine.add_body(floor());
    let cube = engine.add_soft_body(damped_cube(Vec3::new(-0.5, 2.0, -0.5)));
    for _ in 0..240 {
        engine.step(DT);
    }
    let body = engine.get_soft_body(cube).expect("cube survives");
    let min_y = body
        .particles
        .iter()
        .map(|p| p.position.y)
        .fold(f32::INFINITY, f32::min);
    assert!(
        min_y > -0.05 && min_y < 0.25,
        "bottom layer at {min_y}, want ~0.1"
    );
    assert!(
        body.particles.iter().all(|p| p.position.is_finite()),
        "no NaN in landed cube"
    );
}

/// Sleep is visible at the orchestrator level: a settled cube freezes with
/// zeroed velocities, and an explicit wake thaws it.
#[test]
fn soft_sleep_and_wake_through_orchestrator() {
    let mut engine = Engine::new(SolverKind::Xpbd, GRAVITY);
    engine.add_body(floor());
    let cube = engine.add_soft_body(damped_cube(Vec3::new(-0.5, 2.0, -0.5)));
    assert!(!engine.is_soft_asleep(cube), "fresh bodies start awake");
    for _ in 0..300 {
        engine.step(DT);
    }
    assert!(engine.is_soft_asleep(cube), "settled cube must sleep");
    assert!(
        engine
            .get_soft_body(cube)
            .expect("cube survives")
            .particles
            .iter()
            .all(|p| p.velocity.length() < 1e-6),
        "sleep zeroes soft velocities"
    );
    engine.wake_soft_body(cube);
    assert!(!engine.is_soft_asleep(cube), "explicit wake thaws the cube");
    assert!(
        !engine.is_soft_asleep(SoftHandle::from_raw(u32::MAX)),
        "bad handle sleeps never"
    );
}

/// Soft contact events drain at the orchestrator level: touchdown emits
/// one Begin per pair, resting emits no flicker, teleporting away emits
/// the End.
#[test]
fn soft_contact_events_through_orchestrator() {
    use ornis_physics::trigger::ContactEventKind;

    let mut engine = Engine::new(SolverKind::Xpbd, GRAVITY);
    let ground = engine.add_body(floor());
    let cube = engine.add_soft_body(damped_cube(Vec3::new(-0.5, 2.0, -0.5)));
    let mut saw_begin = false;
    for _ in 0..240 {
        engine.step(DT);
        if engine.drain_soft_contact_events().into_iter().any(is_begin) {
            saw_begin = true;
            break;
        }
    }
    assert!(saw_begin, "touchdown must emit Begin");
    for _ in 0..30 {
        engine.step(DT);
    }
    let rest: Vec<SoftContactEvent> = engine.drain_soft_contact_events();
    assert!(
        rest.is_empty(),
        "resting contact must not flicker, got {rest:?}"
    );
    {
        let body = engine.get_soft_body_mut(cube).expect("cube mut");
        for p in &mut body.particles {
            p.position += Vec3::new(0.0, 5.0, 0.0);
        }
    }
    engine.wake_soft_body(cube);
    engine.step(DT);
    let ends: Vec<SoftContactEvent> = engine.drain_soft_contact_events();
    assert!(
        ends.iter()
            .any(|e| e.soft == cube && e.body == ground && e.kind == ContactEventKind::End),
        "teleport away must emit End, got {ends:?}"
    );

    fn is_begin(e: SoftContactEvent) -> bool {
        use ornis_physics::trigger::ContactEventKind;
        e.kind == ContactEventKind::Begin
    }
}

/// Handles stay stable across solver migrations and parked bodies freeze:
/// SI → XPBD → SI → XPBD keeps rigid and soft handles, and soft motion
/// only happens on the XPBD path.
#[test]
fn soft_handles_survive_solver_migration() {
    let mut engine = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let fall = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 5.0, 0.0), 0.5, 1.0));
    let cloth = engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 4, 0.5, 1.0, 0.0));
    let cube = engine.add_soft_body(damped_cube(Vec3::new(2.0, 3.0, 0.0)));
    assert_eq!(engine.soft_body_count(), 2);

    // Parked under SequentialImpulse: stepping moves the rigid body but
    // leaves every particle exactly where it was.
    let parked: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("parked cloth readable")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();
    for _ in 0..30 {
        engine.step(DT);
    }
    let still: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("parked cloth readable")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();
    assert_eq!(parked, still, "parked soft must not step");
    assert!(!engine.is_soft_asleep(cloth), "parked bodies never sleep");
    assert!(
        engine.drain_soft_contact_events().is_empty(),
        "parked bodies emit nothing"
    );
    assert!(
        engine.get_body(fall).expect("rigid survives").position.y < 5.0,
        "rigid keeps stepping while soft parks"
    );

    // Onto XPBD: handles stable, soft starts falling.
    engine.set_solver_kind(SolverKind::Xpbd, GRAVITY);
    assert_eq!(engine.kind(), SolverKind::Xpbd);
    assert_eq!(engine.soft_body_count(), 2);
    assert!(engine.get_soft_body(cloth).is_some(), "cloth handle stable");
    assert!(engine.get_soft_body(cube).is_some(), "cube handle stable");
    assert!(engine.get_body(fall).is_some(), "rigid handle stable");
    for _ in 0..60 {
        engine.step(DT);
    }
    let swinging = engine
        .get_soft_body(cloth)
        .expect("live cloth")
        .particles
        .iter()
        .map(|p| p.position)
        .collect::<Vec<_>>();
    assert_ne!(parked, swinging, "unparked soft must step");

    // Back to SequentialImpulse: everything stable, soft frozen again.
    engine.set_solver_kind(SolverKind::SequentialImpulse, GRAVITY);
    assert_eq!(engine.soft_body_count(), 2);
    assert!(engine.get_soft_body(cloth).is_some(), "cloth handle stable");
    assert!(engine.get_soft_body(cube).is_some(), "cube handle stable");
    let frozen: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("reparked cloth")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();
    assert_eq!(frozen, swinging, "migration preserves particle poses");
    for _ in 0..30 {
        engine.step(DT);
    }
    let refrozen: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("reparked cloth")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();
    assert_eq!(frozen, refrozen, "reparked soft must not step");
}

/// Soft removal follows the same swap-remove discipline as rigid bodies:
/// the tail moves into the hole, and invalid handles are a no-op.
#[test]
fn soft_remove_remaps_survivor_through_orchestrator() {
    let mut engine = Engine::new(SolverKind::Xpbd, GRAVITY);
    let a = engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 2, 0.5, 1.0, 0.0));
    let b = engine.add_soft_body(SoftBody::chain(Vec3::X, Vec3::NEG_Y, 2, 0.5, 1.0, 0.0));
    let c = engine.add_soft_body(SoftBody::chain(Vec3::NEG_X, Vec3::NEG_Y, 2, 0.5, 1.0, 0.0));
    assert_eq!(
        (a, b, c),
        (
            SoftHandle::from(0usize),
            SoftHandle::from(1usize),
            SoftHandle::from(2usize)
        )
    );
    engine.remove_soft_body(b);
    assert_eq!(engine.soft_body_count(), 2);
    assert!(
        engine.get_soft_body(b).is_some(),
        "tail survivor moves into the hole"
    );
    assert!(
        engine.get_soft_body(b).expect("survivor").particles[0]
            .position
            .x
            < -0.5,
        "hole holds the old tail (x = -1)"
    );
    engine.remove_soft_body(SoftHandle::from_raw(u32::MAX));
    assert_eq!(engine.soft_body_count(), 2, "invalid removal is a no-op");
}

/// Migrating onto XPBD drops joints the XPBD engine cannot solve
/// (wheel/gear/six-DOF): survivors keep their relative order, bodies and
/// soft bodies are untouched.
#[test]
fn unsupported_joints_drop_on_xpbd_migration() {
    let mut engine = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let a = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.5, 1.0));
    let b = engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, -2.0, 0.0), 0.5, 1.0));
    let _rod = engine
        .add_joint(
            a,
            b,
            JointKind::Distance {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("distance joint accepted");
    let _free = engine
        .add_joint(
            a,
            b,
            JointKind::SixDof {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                linear: [AxisConfig::Free; 3],
                angular: [AxisConfig::Free; 3],
            },
        )
        .expect("sixdof joint accepted");
    assert_eq!(engine.joint_count(), 2);
    let soft = engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 3, 0.5, 1.0, 0.0));

    engine.set_solver_kind(SolverKind::Xpbd, GRAVITY);
    assert_eq!(
        engine.joint_count(),
        1,
        "sixdof must drop, distance survives"
    );
    assert!(engine.get_body(a).is_some() && engine.get_body(b).is_some());
    assert!(
        engine.get_soft_body(soft).is_some(),
        "soft untouched by joint filtering"
    );
    // The survivor keeps stepping: the rod still holds its length.
    for _ in 0..120 {
        engine.step(DT);
    }
    let (pa, pb) = (
        engine.get_body(a).expect("a").position,
        engine.get_body(b).expect("b").position,
    );
    let len = (pb - pa).length();
    assert!((len - 2.0).abs() < 0.3, "migrated rod must hold, got {len}");
}

/// Islands routing parks live soft bodies without losing them: the parked
/// body stays readable and frozen, and collapses back onto XPBD unpark it
/// in handle order with stepping resumed.
#[test]
fn soft_parks_across_islands_routing() {
    let mut engine = Engine::new(SolverKind::Xpbd, GRAVITY);
    engine.add_body(floor());
    let cloth = engine.add_soft_body(SoftBody::chain(Vec3::ZERO, Vec3::NEG_Y, 4, 0.5, 1.0, 0.0));
    for _ in 0..30 {
        engine.step(DT);
    }
    let live: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("live cloth")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();

    engine.set_routing(RoutingKind::Islands);
    assert_eq!(engine.soft_body_count(), 1, "parked soft stays registered");
    assert!(
        engine.get_soft_body(cloth).is_some(),
        "parked handle stable"
    );
    for _ in 0..10 {
        engine.step(DT);
    }
    let parked: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("parked cloth readable")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();
    assert_eq!(live, parked, "Islands routing freezes soft bodies");

    engine.set_routing(RoutingKind::Single);
    assert_eq!(engine.kind(), SolverKind::Xpbd);
    assert!(
        engine.get_soft_body(cloth).is_some(),
        "unparked handle stable"
    );
    // The chain settled (and slept) before parking, so kick it awake: this
    // proves the unparked body steps again instead of staying frozen.
    {
        let body = engine.get_soft_body_mut(cloth).expect("unparked cloth mut");
        for p in body.particles.iter_mut().filter(|p| p.inv_mass > 0.0) {
            p.velocity.x = 1.5;
        }
    }
    engine.wake_soft_body(cloth);
    engine.step(DT);
    let resumed: Vec<Vec3> = engine
        .get_soft_body(cloth)
        .expect("unparked cloth")
        .particles
        .iter()
        .map(|p| p.position)
        .collect();
    assert_ne!(parked, resumed, "unparked soft must step again");
}
