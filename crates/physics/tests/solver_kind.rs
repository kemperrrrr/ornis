//! `Engine` orchestrator gate: dispatch behind one seam, lossless migration
//! across `SolverKind` switches (handles stay valid, joints survive).

use glam::Vec3;
use ornis_physics::{Engine, JointKind, PhysicsEngine, RigidBody, SolverKind};

const DT: f32 = 1.0 / 60.0;

fn box_at(y: f32) -> RigidBody {
    RigidBody::new_box(Vec3::new(0.0, y, 0.0), Vec3::splat(0.5), 1.0)
}

#[test]
fn engine_dispatch_steps_both_solvers() {
    for kind in [SolverKind::Builtin, SolverKind::Avbd] {
        let mut engine = Engine::new(kind, Vec3::new(0.0, -9.81, 0.0));
        assert_eq!(engine.kind(), kind);
        let h = engine.add_body(box_at(5.0));
        for _ in 0..60 {
            engine.step(DT);
        }
        let b = engine.get_body(h).unwrap();
        assert!(
            b.position.y < 4.0,
            "{kind:?} must integrate gravity, got y={}",
            b.position.y
        );
    }
}

#[test]
fn engine_migration_keeps_bodies_joints_handles() {
    let mut engine = Engine::new(SolverKind::Builtin, Vec3::new(0.0, -9.81, 0.0));
    let a = engine.add_body(box_at(2.0));
    let b = engine.add_body(box_at(4.0));
    let _j = engine
        .add_joint(
            a,
            b,
            JointKind::Distance {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid joint");
    for _ in 0..30 {
        engine.step(DT);
    }
    let (pa, pb) = (
        engine.get_body(a).unwrap().position,
        engine.get_body(b).unwrap().position,
    );
    // Builtin -> Avbd: poses, handles and the joint survive.
    engine.set_solver_kind(SolverKind::Avbd, Vec3::new(0.0, -9.81, 0.0));
    assert_eq!(engine.kind(), SolverKind::Avbd);
    assert_eq!(engine.get_body(a).unwrap().position, pa);
    assert_eq!(engine.get_body(b).unwrap().position, pb);
    for _ in 0..120 {
        engine.step(DT);
    }
    let (qa, qb) = (
        engine.get_body(a).unwrap().position,
        engine.get_body(b).unwrap().position,
    );
    let rod = (qb - qa).length();
    assert!(
        (rod - 2.0).abs() < 0.3,
        "migrated rod must hold its length, got {rod}"
    );
    // Avbd -> Builtin: roundtrip keeps stepping on the same handles.
    engine.set_solver_kind(SolverKind::Builtin, Vec3::new(0.0, -9.81, 0.0));
    assert_eq!(engine.kind(), SolverKind::Builtin);
    for _ in 0..30 {
        engine.step(DT);
    }
    let (ra, rb) = (
        engine.get_body(a).unwrap().position,
        engine.get_body(b).unwrap().position,
    );
    let rod2 = (rb - ra).length();
    assert!(
        (rod2 - 2.0).abs() < 0.5,
        "roundtripped rod must hold its length, got {rod2}"
    );
}

#[test]
fn engine_kinds_agree_on_drop_settle() {
    // Engine-level equivalence pin (per-solver formulation options stay
    // deferred work): both orchestrated solvers settle the same drop
    // scene to the same rest poses. Behavioral tolerance, never bit
    // identity — the implementations differ by design, the rest state
    // must not.
    for kind in [SolverKind::Builtin, SolverKind::Avbd] {
        let mut engine = Engine::new(kind, Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        let hb = engine.add_body(box_at(2.0));
        let hs = engine.add_body(RigidBody::new_sphere(Vec3::new(2.0, 4.0, 0.0), 0.5, 1.0));
        for _ in 0..300 {
            engine.step(DT);
        }
        let b = engine.get_body(hb).unwrap();
        let s = engine.get_body(hs).unwrap();
        assert!(
            (b.position.y - 0.5).abs() < 0.05 && b.velocity.length() < 0.1,
            "{kind:?} box must rest on the floor"
        );
        assert!(
            (s.position.y - 0.5).abs() < 0.05 && s.velocity.length() < 0.1,
            "{kind:?} sphere must rest on the floor"
        );
    }
}
