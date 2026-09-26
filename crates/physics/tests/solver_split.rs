//! M3 Islands routing gate: the split registry (both engines alive, one
//! solver per contact island) settles like a single solver, migrates with
//! hysteresis, wakes sane after sleep, keeps joints, and reruns bit-identical.

use glam::Vec3;
use ornis_physics::{
    BodyHandle, Engine, JointHandle, JointKind, LocalAvbdBody, LocalAvbdJoint, LocalSiBody,
    LocalSiJoint, PhysicsEngine, RigidBody, RoutingKind, SolverKind,
};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn floor() -> RigidBody {
    RigidBody::new_box(Vec3::new(0.0, -0.5, 0.0), Vec3::new(50.0, 0.5, 50.0), 0.0)
}

fn box_at(x: f32, y: f32) -> RigidBody {
    RigidBody::new_box(Vec3::new(x, y, 0.0), Vec3::splat(0.5), 1.0)
}

/// 2-stack at x=0 plus a lone migrant at x=5, dropped from rest.
fn build_scene(engine: &mut Engine) -> (BodyHandle, BodyHandle, BodyHandle) {
    let f = engine.add_body(floor());
    let a = engine.add_body(box_at(0.0, 0.6));
    let b = engine.add_body(box_at(0.0, 1.7));
    let _ = (f, a);
    (f, a, b)
}

#[test]
fn split_drop_settle_matches_single() {
    let mut single = Engine::new(SolverKind::Avbd, GRAVITY);
    build_scene(&mut single);
    for _ in 0..600 {
        single.step(DT);
    }
    let mut split = Engine::new(SolverKind::Avbd, GRAVITY);
    build_scene(&mut split);
    split.set_routing(RoutingKind::Islands);
    for _ in 0..600 {
        split.step(DT);
    }
    // Same rest state (behavioral tolerance — trajectories differ by solver).
    for h in 1..3usize {
        let ps = single.get_body(BodyHandle::from(h)).unwrap().position;
        let px = split.get_body(BodyHandle::from(h)).unwrap().position;
        assert!(
            (ps.y - px.y).abs() < 0.15,
            "body {h}: single y={} split y={}",
            ps.y,
            px.y
        );
    }
    // Settled islands rest in SequentialImpulse; the floor lives in both natively.
    let m = split.split_metrics().expect("metrics under Islands");
    assert!(m.migrations > 0, "expected at least one routing migration");
    assert_eq!(
        m.avbd_bodies, 0,
        "all calm bodies must rest in SequentialImpulse"
    );
}

#[test]
fn split_migration_bounded_under_drive() {
    let mut engine = Engine::new(SolverKind::Avbd, GRAVITY);
    engine.add_body(floor());
    let migrant = engine.add_body(box_at(5.0, 0.6));
    engine.set_routing(RoutingKind::Islands);
    for step in 0..900 {
        if step % 90 == 0 && step > 0 {
            // Scripted deterministic kick (host edit must wake).
            if let Some(b) = engine.get_body_mut(migrant) {
                b.velocity = Vec3::new(0.0, 4.0, 0.0);
            }
        }
        engine.step(DT);
    }
    let m = engine.split_metrics().expect("metrics under Islands");
    // ~2 rebuilds per drive cycle (wake + sleep) over 10 cycles, plus the
    // initial settle. Thrash would read in the hundreds.
    assert!(
        (5..=40).contains(&m.migrations),
        "migrations={} outside hysteresis bound",
        m.migrations
    );
    let b = engine.get_body(migrant).unwrap();
    assert!(b.position.y.is_finite() && b.position.y > -1.0);
}

#[test]
fn split_sleep_migrate_wakes_sane() {
    let mut engine = Engine::new(SolverKind::Avbd, GRAVITY);
    engine.add_body(floor());
    let h = engine.add_body(box_at(0.0, 0.6));
    engine.set_routing(RoutingKind::Islands);
    for _ in 0..400 {
        engine.step(DT);
    }
    let m = engine.split_metrics().expect("metrics under Islands");
    assert_eq!(
        m.avbd_bodies, 0,
        "settled body must sleep in SequentialImpulse"
    );
    assert!(m.migrations > 0);
    // Kick the migrated sleeper via the host-edit path.
    engine.get_body_mut(h).unwrap().velocity = Vec3::new(0.0, 5.0, 0.0);
    for _ in 0..120 {
        engine.step(DT);
    }
    // Zombie rule: pulled mass model stays solvable, motion is sane.
    let b = engine.get_body(h).unwrap();
    assert!(b.inv_mass > 0.0, "woken body lost its mass model");
    assert!(b.position.y > 0.0 && b.position.y < 10.0);
    assert!(b.velocity.length() < 20.0);
}

#[test]
fn split_jointed_pair_survives() {
    let mut engine = Engine::new(SolverKind::Avbd, GRAVITY);
    engine.add_body(floor());
    let a = engine.add_body(box_at(0.0, 2.0));
    let b = engine.add_body(box_at(0.0, 3.2));
    engine
        .add_joint(
            a,
            b,
            JointKind::Distance {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid joint");
    engine.set_routing(RoutingKind::Islands);
    for _ in 0..600 {
        engine.step(DT);
    }
    // Joint pins its island to one solver: the link never stretches, the
    // pair settles, migrations still happen around it.
    let (pa, pb) = (
        engine.get_body(a).unwrap().position,
        engine.get_body(b).unwrap().position,
    );
    let dist = (pa - pb).length();
    assert!(
        (dist - 1.2).abs() < 0.3,
        "jointed pair drifted apart: {dist}"
    );
    assert!(pa.y > 0.0 && pb.y > pa.y, "pair must rest stacked");
    let m = engine.split_metrics().expect("metrics under Islands");
    assert!(m.migrations > 0, "expected routing activity");
}

#[test]
fn split_rerun_deterministic() {
    let run = || {
        let mut engine = Engine::new(SolverKind::Avbd, GRAVITY);
        build_scene(&mut engine);
        engine.set_routing(RoutingKind::Islands);
        for _ in 0..240 {
            engine.step(DT);
        }
        (0..3usize)
            .map(|h| engine.get_body(BodyHandle::from(h)).unwrap().position)
            .collect::<Vec<_>>()
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b, "Islands routing must rerun bit-identical");
}

/// Handle-space separation: local indices round-trip losslessly inside
/// their own space and reinterpret into the same `u32` globally. A local
/// AVBD index must never compare equal to (or be usable as) an SI index —
/// the types simply do not convert.
#[test]
fn local_handle_spaces_round_trip_losslessly() {
    let avbd = LocalAvbdBody::from_raw(3);
    assert_eq!(avbd.index(), 3);
    assert_eq!(avbd.as_u32(), 3);
    assert_eq!(BodyHandle::from(avbd).as_u32(), 3);
    assert_eq!(LocalAvbdBody::from(BodyHandle::from_raw(3)), avbd);

    let si = LocalSiBody::from(7usize);
    assert_eq!(usize::from(si), 7);
    assert_eq!(BodyHandle::from(si).index(), 7);
    assert_eq!(LocalSiBody::from(BodyHandle::from(si)), si);

    let aj = LocalAvbdJoint::from_raw(1);
    assert_eq!(JointHandle::from(aj).as_u32(), 1);
    assert_eq!(LocalAvbdJoint::from(JointHandle::from(aj)), aj);
    let sj = LocalSiJoint::from(2u32);
    assert_eq!(u32::from(sj), 2);
    assert_eq!(LocalSiJoint::from(JointHandle::from(sj)), sj);
}

/// Global handles are registry slots: structural rebuilds under Islands
/// (migration + an added body) keep every pre-existing global resolving
/// to the same body, and the static floor never moves.
#[test]
fn global_handles_survive_islands_rebuilds() {
    let mut engine = Engine::new(SolverKind::Avbd, GRAVITY);
    let (f, a, b) = build_scene(&mut engine);
    engine.set_routing(RoutingKind::Islands);
    for _ in 0..120 {
        engine.step(DT);
    }
    // Structural edit forces a full rebuild with live locals on both sides.
    let c = engine.add_body(box_at(5.0, 0.6));
    for _ in 0..240 {
        engine.step(DT);
    }
    for h in [f, a, b, c] {
        let body = engine.get_body(h).unwrap_or_else(|| panic!("global {h:?} lost"));
        assert!(body.position.is_finite(), "global {h:?} went non-finite");
    }
    let floor = engine.get_body(f).unwrap();
    assert_eq!(floor.position, Vec3::new(0.0, -0.5, 0.0));
    assert!(engine.split_metrics().is_some());
}
