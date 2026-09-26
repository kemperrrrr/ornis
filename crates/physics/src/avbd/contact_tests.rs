//! Contact-row regression tests for the AVBD residual and its Jacobian.

use super::*;

#[test]
fn rolling_sphere_keeps_support_without_gaining_speed() {
    for reversed in [false, true] {
        let mut engine = AvbdEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let floor = RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(50.0, 1.0, 50.0), 0.0);
        let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.5, 0.0), 0.5, 1.0);
        ball.velocity.x = 1.0;
        ball.angular_velocity.z = -2.0;
        let h = if reversed {
            let h = engine.add_body(ball);
            engine.add_body(floor);
            h
        } else {
            engine.add_body(floor);
            engine.add_body(ball)
        };
        for step in 0..1500 {
            engine.step(DT_STEP);
            let b = &engine.bodies[h.index()];
            assert!(
                (b.position.y - 0.5).abs() < 0.03,
                "rolling support lost: reversed={reversed} step={step} y={}",
                b.position.y
            );
            assert!(
                (b.velocity.x - 1.0).abs() < 0.02 && (b.angular_velocity.z + 2.0).abs() < 0.04,
                "normal force accelerated roll: step={step} v={:?} w={:?}",
                b.velocity,
                b.angular_velocity
            );
        }
    }
}

#[test]
fn refreshed_face_contacts_preserve_penetration() {
    let mut engine = AvbdEngine::new(Vec3::ZERO);
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    engine.add_body(RigidBody::new_box(
        Vec3::new(-3.0, 0.3, 0.0),
        Vec3::splat(0.4),
        1.0,
    ));
    engine.ensure_scratch();
    engine.pos0 = engine.bodies.iter().map(|b| b.position).collect();
    engine.rot0 = engine.bodies.iter().map(|b| b.orientation).collect();
    engine.generate_pairs();
    assert!(engine.pairs[0].points.len() >= 3);
    for qi in 0..engine.pairs[0].points.len() {
        let c0 = engine.gap_c0(0, qi);
        assert!(
            c0 < -0.08,
            "buried face point {qi} lost penetration: C0={c0}"
        );
    }
}

#[test]
fn face_anchors_lie_on_their_own_surface() {
    let mut engine = AvbdEngine::new(Vec3::ZERO);
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    engine.add_body(RigidBody::new_box(
        Vec3::new(-3.0, 0.3, 0.0),
        Vec3::splat(0.4),
        1.0,
    ));
    engine.ensure_scratch();
    engine.generate_pairs();
    for pt in &engine.pairs[0].points {
        assert!((pt.ra.y - 1.0).abs() < 1e-6, "floor anchor={:?}", pt.ra);
        assert!((pt.rb.y + 0.4).abs() < 1e-6, "box anchor={:?}", pt.rb);
        assert!(
            pt.rb.x.abs() <= 0.401 && pt.rb.z.abs() <= 0.401,
            "contact outside the small body's footprint: {:?}",
            pt.rb
        );
    }
}

#[test]
fn sphere_contact_normal_is_independent_of_spin() {
    let mut engine = AvbdEngine::new(Vec3::ZERO);
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    engine.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.49, 0.0), 0.5, 1.0));
    engine.ensure_scratch();
    engine.pos0 = engine.bodies.iter().map(|b| b.position).collect();
    engine.rot0 = engine.bodies.iter().map(|b| b.orientation).collect();
    engine.generate_pairs();
    // A persisted material point has orbited away from the geometric pole.
    engine.pairs[0].points[0].rb = Quat::from_rotation_z(0.3) * Vec3::NEG_Y * 0.5;
    let p = &engine.pairs[0];
    let pt = &p.points[0];
    let c0 = engine.gap_c0(0, 0);
    let before = engine.row_c(p, pt, p.n, c0).0;
    engine.bodies[1].orientation = Quat::from_rotation_z(0.1);
    let (after, _, r) = engine.row_c(p, pt, p.n, c0);
    assert!(
        r.cross(p.n).length() < 1e-7,
        "sphere normal torque lever={r:?}"
    );
    assert!(
        (after - before).abs() < 1e-7,
        "spin changed normal C: {before}->{after}"
    );
    assert!(
        c0.abs() < 1e-6,
        "orbited material point changed geometric C0={c0}"
    );
}

#[test]
fn contact_rotation_enters_taylor_residual_once() {
    let mut engine = AvbdEngine::new(Vec3::ZERO);
    engine.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    engine.add_body(RigidBody::new_box(Vec3::Y, Vec3::splat(0.5), 1.0));
    engine.ensure_scratch();
    engine.pos0 = vec![Vec3::ZERO, Vec3::Y];
    engine.rot0 = vec![Quat::IDENTITY; 2];
    let point = AvbdPoint {
        ra: Vec3::new(0.3, -0.5, 0.2),
        rb: Vec3::new(-0.2, 0.5, -0.1),
        lam: [0.0; 3],
        pen: [1.0; 3],
        stuck: true,
        roll_lam: [0.0; 3],
    };
    let pair = AvbdPair {
        a: 0,
        b: 1,
        n: Vec3::Y,
        mu: [0.5; 2],
        gap: 0.0,
        points: vec![],
    };
    // Pure rotation isolates A1: the old anchor displacement plus J*dtheta
    // differentiates to twice the angular Jacobian at the step-start pose.
    for side in 0..2 {
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let delta = Vec3::new(0.0001, -0.0002, 0.0003);
            engine.bodies[side].orientation = quat_integrate(Quat::IDENTITY, delta);
            let (c, ra, rb) = engine.row_c(&pair, &point, axis, 0.0);
            let expected = if side == 0 {
                ra.cross(axis).dot(delta)
            } else {
                -rb.cross(axis).dot(delta)
            };
            assert!(
                (c - expected).abs() < 1e-7,
                "side={side} axis={axis:?}: residual={c} single rotation={expected}"
            );
            engine.bodies[side].orientation = Quat::IDENTITY;
        }
    }
}

#[test]
fn tangent_basis_is_orthonormal() {
    for n in [
        Vec3::X,
        Vec3::Y,
        Vec3::Z,
        Vec3::new(1.0, 2.0, 3.0).normalize(),
    ] {
        let (t1, t2) = tangent_basis(n);
        assert!((t1.dot(n)).abs() < 1e-6);
        assert!((t2.dot(n)).abs() < 1e-6);
        assert!((t1.dot(t2)).abs() < 1e-6);
        assert!((t1.length() - 1.0).abs() < 1e-6);
    }
}
