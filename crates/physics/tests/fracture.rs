//! `Engine` fracture gate: Hit-driven box splitting, uniform across solvers.

use glam::Vec3;
use ornis_physics::{Engine, JointKind, PhysicsEngine, RigidBody, Shape, SolverKind};

const DT: f32 = 1.0 / 60.0;

#[test]
fn fracture_splits_box_on_hard_hit_both_solvers() {
    for kind in [SolverKind::Builtin, SolverKind::Avbd] {
        // Victim hangs on a Ball joint; a projectile knocks it at 8 m/s.
        let mut engine = Engine::new(kind, Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        let anchor = engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, 4.0, 0.0),
            Vec3::splat(0.5),
            0.0,
        ));
        let mut victim = RigidBody::new_box(Vec3::new(0.0, 3.0, 0.0), Vec3::splat(0.5), 1.0);
        victim.fracture_impact_speed = 3.0;
        let victim_h = engine.add_body(victim);
        engine
            .add_joint(
                anchor,
                victim_h,
                JointKind::Ball {
                    local_anchor_a: Vec3::new(0.0, -0.5, 0.0),
                    local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
                },
            )
            .expect("valid joint");
        let mut shot = RigidBody::new_box(Vec3::new(-2.5, 3.0, 0.0), Vec3::splat(0.3), 1.0);
        // Fast and close: 20 m/s over 2 m drops 5 cm in flight, still a
        // clean hit on the victim's side (an 8 m/s shot from 6 m falls
        // ~2.8 m under gravity and passes underneath).
        shot.velocity = Vec3::new(20.0, 0.0, 0.0);
        shot.restitution = 0.0;
        engine.add_body(shot);
        let mut events = Vec::new();
        for _ in 0..240 {
            engine.step(DT);
            events.extend(engine.drain_fracture_events());
            if !events.is_empty() {
                break;
            }
        }
        assert_eq!(events.len(), 1, "{kind:?}: one fracture expected");
        let ev = events[0];
        assert_eq!(ev.parent, victim_h, "{kind:?}: parent must be the victim");
        let pa = engine.get_body(ev.pieces[0]).expect("half A live");
        let pb = engine.get_body(ev.pieces[1]).expect("half B live");
        // Conserved mass, halved longest (here X, tie-break) axis.
        assert!((pa.mass - 0.5).abs() < 1e-6, "{kind:?}: half mass {pa:?}");
        assert!((pb.mass - 0.5).abs() < 1e-6, "{kind:?}: half mass {pb:?}");
        for half in [pa, pb] {
            match half.shape {
                Shape::Box { half_extents: h } => assert!(
                    (h.x - 0.25).abs() < 1e-4 && (h.y - 0.5).abs() < 1e-4,
                    "{kind:?}: half extents {h:?}"
                ),
                ref s => panic!("{kind:?}: halves stay boxes, got {s:?}"),
            }
            assert_eq!(
                half.fracture_impact_speed, 3.0,
                "{kind:?}: threshold inherits"
            );
        }
        // Split along X by half a meter, preserving the rigid velocity field.
        let span = pb.position - pa.position;
        assert!(
            (span.length() - 0.5).abs() < 0.05 && span.x.abs() > 0.4,
            "{kind:?}: halves straddle X, span {span:?}"
        );
        let center = (pa.position + pb.position) * 0.5;
        let va = pa.velocity + pa.angular_velocity.cross(center - pa.position);
        let vb = pb.velocity + pb.angular_velocity.cross(center - pb.position);
        assert!(
            (va - vb).length() < 1e-5,
            "{kind:?}: rigid velocity continuity {va:?} vs {vb:?}"
        );
        // Joint died with the parent: freeze fracture, drop 120 steps —
        // free halves fall away from the anchor (a live Ball would hold
        // them within ~0.5 m of y=3).
        for h in ev.pieces {
            engine.get_body_mut(h).unwrap().fracture_impact_speed = f32::INFINITY;
        }
        for _ in 0..120 {
            engine.step(DT);
        }
        for h in ev.pieces {
            let y = engine.get_body(h).unwrap().position.y;
            assert!(y < 1.5, "{kind:?}: joint must be gone, half at y={y}");
        }
    }
}

#[test]
fn fracture_ignores_soft_hits_and_non_boxes() {
    for kind in [SolverKind::Builtin, SolverKind::Avbd] {
        let mut engine = Engine::new(kind, Vec3::new(0.0, -9.81, 0.0));
        engine.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        // Firm drop: impact ~3 m/s fires a Hit but stays below the 8.0
        // threshold — intact.
        let mut soft = RigidBody::new_box(Vec3::new(-2.0, 0.9, 0.0), Vec3::splat(0.4), 1.0);
        soft.fracture_impact_speed = 8.0;
        soft.restitution = 0.0;
        let soft_h = engine.add_body(soft);
        // Sphere with a threshold: hard impact, but spheres never split.
        let mut ball = RigidBody::new_sphere(Vec3::new(2.0, 3.0, 0.0), 0.4, 1.0);
        ball.fracture_impact_speed = 1.5;
        ball.restitution = 0.0;
        let ball_h = engine.add_body(ball);
        for _ in 0..240 {
            engine.step(DT);
        }
        assert!(
            engine.drain_fracture_events().is_empty(),
            "{kind:?}: no fracture expected"
        );
        assert!(
            engine.get_body(soft_h).is_some(),
            "{kind:?}: soft box intact"
        );
        assert!(engine.get_body(ball_h).is_some(), "{kind:?}: sphere intact");
    }
}
