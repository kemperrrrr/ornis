//! Cross-solver joints gate: joints whose ends live in different solvers
//! stay coupled by the post-step pass (ball/distance rows in canonical
//! order, mass-restore on both sides), rerun bit-identical under any rayon
//! thread count, survive hysteresis migrations, and refuse non-structural
//! kinds explicitly instead of silently dropping them.

use glam::Vec3;
use ornis_physics::{
    BodyHandle, CrossJointStatus, CrossRowKind, Engine, JointKind, PhysicsEngine, RigidBody,
    RoutingKind, SolverKind, cross_row_kind,
};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn heavy_anchor(pos: Vec3) -> RigidBody {
    RigidBody::new_box(pos, Vec3::splat(0.5), 100.0)
}

fn bob(pos: Vec3) -> RigidBody {
    RigidBody::new_sphere(pos, 0.3, 1.0)
}

/// World-space anchor separation of a ball/distance joint.
fn anchor_separation(e: &Engine, a: BodyHandle, b: BodyHandle) -> f32 {
    let (pa, pb) = (
        e.get_body(a).unwrap().position,
        e.get_body(b).unwrap().position,
    );
    (pb - pa).length()
}

/// Pendulum pivot in the anchor frame (box bottom) and in the bob frame
/// (2 m above its centre): the world anchors coincide at assembly while
/// the bodies rest 3 m apart, overlap-free.
const PIVOT_A: Vec3 = Vec3::new(0.0, -1.0, 0.0);
const PIVOT_B: Vec3 = Vec3::new(0.0, 2.0, 0.0);

/// Build a cross-pendulum: heavy anchor above, light bob below, ball
/// joint assembled exactly at the shared pivot. Returns the joint handle.
fn pendulum_scene(e: &mut Engine) -> (BodyHandle, BodyHandle, ornis_physics::JointHandle) {
    let anchor = e.add_body(heavy_anchor(Vec3::new(0.0, 1.0, 0.0)));
    let ball = e.add_body(bob(Vec3::new(0.0, -2.0, 0.0)));
    let j = e
        .add_joint(
            anchor,
            ball,
            JointKind::Ball {
                local_anchor_a: PIVOT_A,
                local_anchor_b: PIVOT_B,
            },
        )
        .expect("valid ball joint");
    (anchor, ball, j)
}

/// World-anchor gap of a ball joint with the given local anchors.
fn anchor_gap(e: &Engine, a: BodyHandle, b: BodyHandle, la: Vec3, lb: Vec3) -> f32 {
    let (ba, bb) = (e.get_body(a).unwrap(), e.get_body(b).unwrap());
    ((bb.position + bb.orientation * lb) - (ba.position + ba.orientation * la)).length()
}

/// World-anchor gap of the pendulum ball joint (≈ 0 while coupled).
fn pivot_error(e: &Engine, a: BodyHandle, b: BodyHandle) -> f32 {
    anchor_gap(e, a, b, PIVOT_A, PIVOT_B)
}

/// Bit-exact body snapshot for determinism comparisons.
type BitState = Vec<([u32; 3], [u32; 4], [u32; 3], [u32; 3])>;

fn bit_state(e: &Engine) -> BitState {
    (0..e.body_count())
        .map(|h| {
            let b = e.get_body(BodyHandle::from(h)).unwrap();
            (
                b.position.to_array().map(f32::to_bits),
                b.orientation.to_array().map(f32::to_bits),
                b.velocity.to_array().map(f32::to_bits),
                b.angular_velocity.to_array().map(f32::to_bits),
            )
        })
        .collect()
}

#[test]
fn cross_ball_pendulum_holds_anchor_length_si_avbd() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let (anchor, ball, j) = pendulum_scene(&mut e);
    e.set_routing(RoutingKind::Islands);
    e.pin_body_solver(anchor, Some(SolverKind::SequentialImpulse));
    e.pin_body_solver(ball, Some(SolverKind::Avbd));
    for _ in 0..240 {
        e.step(DT);
    }
    assert_eq!(e.body_solver(anchor), Some(SolverKind::SequentialImpulse));
    assert_eq!(e.body_solver(ball), Some(SolverKind::Avbd));
    assert_eq!(
        e.cross_joint_status(j),
        Some(CrossJointStatus::Coupled(CrossRowKind::Ball))
    );
    let err = pivot_error(&e, anchor, ball);
    assert!(err < 0.15, "cross ball pendulum pivot split: {err}");
    let (pa, pb) = (
        e.get_body(anchor).unwrap().position,
        e.get_body(ball).unwrap().position,
    );
    assert!(
        pb.y < pa.y,
        "pendulum bob must hang below the anchor: {pa:?} vs {pb:?}"
    );
}

#[test]
fn cross_distance_holds_rest_length_si_xpbd() {
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let a = e.add_body(heavy_anchor(Vec3::new(-1.5, 2.0, 0.0)));
    let b = e.add_body(bob(Vec3::new(1.5, 2.0, 0.0)));
    let j = e
        .add_joint(
            a,
            b,
            JointKind::Distance {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid distance joint");
    e.set_routing(RoutingKind::Islands);
    e.pin_body_solver(a, Some(SolverKind::SequentialImpulse));
    e.pin_body_solver(b, Some(SolverKind::Xpbd));
    for _ in 0..240 {
        e.step(DT);
    }
    assert_eq!(e.body_solver(a), Some(SolverKind::SequentialImpulse));
    assert_eq!(e.body_solver(b), Some(SolverKind::Xpbd));
    assert_eq!(
        e.cross_joint_status(j),
        Some(CrossJointStatus::Coupled(CrossRowKind::Distance))
    );
    let m = e.split_metrics().expect("metrics under Islands");
    assert_eq!(m.xpbd_bodies, 1, "XPBD must own the pinned body");
    let len = anchor_separation(&e, a, b);
    assert!((len - 3.0).abs() < 0.3, "cross distance rod changed: {len}");
}

#[test]
fn cross_coupling_deterministic_one_vs_32_threads() {
    let run = |workers: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
            let (anchor, ball, _) = pendulum_scene(&mut e);
            e.set_routing(RoutingKind::Islands);
            e.pin_body_solver(anchor, Some(SolverKind::SequentialImpulse));
            e.pin_body_solver(ball, Some(SolverKind::Avbd));
            for _ in 0..120 {
                e.step(DT);
            }
            (bit_state(&e), e.drain_contact_events().len())
        })
    };
    assert_eq!(run(1), run(32));
}

#[test]
fn migration_does_not_tear_cross_joint() {
    // Hanging chain through the solver boundary: static bar → A (SI,
    // native ball) → B (AVBD, cross ball). The chain hangs clear of the
    // floor; the migrant on the floor drives hysteresis around it.
    let mut e = Engine::new(SolverKind::Avbd, GRAVITY);
    e.add_body(RigidBody::new_box(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::new(2.0, 0.25, 0.25),
        0.0,
    ));
    e.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(50.0, 0.5, 50.0),
        0.0,
    ));
    let bar = BodyHandle::from(0usize);
    let a = e.add_body(RigidBody::new_box(
        Vec3::new(0.0, 4.25, 0.0),
        Vec3::splat(0.5),
        5.0,
    ));
    let b = e.add_body(bob(Vec3::new(0.0, 2.55, 0.0)));
    let j1 = e
        .add_joint(
            bar,
            a,
            JointKind::Ball {
                local_anchor_a: Vec3::new(0.0, -0.25, 0.0),
                local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
            },
        )
        .expect("valid native joint");
    const LA2: Vec3 = Vec3::new(0.0, -0.5, 0.0);
    const LB2: Vec3 = Vec3::new(0.0, 1.2, 0.0);
    let j2 = e
        .add_joint(
            a,
            b,
            JointKind::Ball {
                local_anchor_a: LA2,
                local_anchor_b: LB2,
            },
        )
        .expect("valid cross joint");
    // Free third body, kicked awake periodically to drive hysteresis.
    let migrant = e.add_body(bob(Vec3::new(5.0, 3.0, 0.0)));
    e.set_routing(RoutingKind::Islands);
    e.pin_body_solver(a, Some(SolverKind::SequentialImpulse));
    e.pin_body_solver(b, Some(SolverKind::Avbd));
    for step in 0..600 {
        if step % 90 == 0
            && step > 0
            && let Some(body) = e.get_body_mut(migrant)
        {
            body.velocity = Vec3::new(0.0, 4.0, 0.0);
        }
        e.step(DT);
    }
    // Neither joint was dropped; the native one stayed native, the cross
    // one stayed coupled through every migration.
    assert_eq!(e.joint_count(), 2);
    assert_eq!(e.cross_joint_status(j1), Some(CrossJointStatus::Native));
    assert_eq!(
        e.cross_joint_status(j2),
        Some(CrossJointStatus::Coupled(CrossRowKind::Ball))
    );
    assert_eq!(e.body_solver(a), Some(SolverKind::SequentialImpulse));
    assert_eq!(e.body_solver(b), Some(SolverKind::Avbd));
    let err = anchor_gap(&e, a, b, LA2, LB2);
    assert!(err < 0.2, "migration tore the joint: {err}");
    let m = e.split_metrics().expect("metrics under Islands");
    assert!(
        m.migrations > 0,
        "expected routing activity around the pair"
    );
    // Release both ends: the calm island reunites in one solver, natively.
    e.pin_body_solver(a, None);
    e.pin_body_solver(b, None);
    for _ in 0..400 {
        e.step(DT);
    }
    assert_eq!(e.body_solver(a), e.body_solver(b));
    assert_eq!(e.cross_joint_status(j2), Some(CrossJointStatus::Native));
    let err = anchor_gap(&e, a, b, LA2, LB2);
    assert!(err < 0.2, "reunited joint drifted: {err}");
}

#[test]
fn unsupported_cross_joint_refuses_explicitly() {
    assert_eq!(
        cross_row_kind(&JointKind::Ball {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
        }),
        Some(CrossRowKind::Ball)
    );
    assert_eq!(
        cross_row_kind(&JointKind::Distance {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
        }),
        Some(CrossRowKind::Distance)
    );
    for kind in [
        JointKind::Revolute {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_axis_a: Vec3::Y,
            local_axis_b: Vec3::Y,
            limit: None,
            motor: None,
        },
        JointKind::Prismatic {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            local_axis_a: Vec3::Y,
            local_axis_b: Vec3::Y,
            limit: None,
            motor: None,
        },
        JointKind::Fixed {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
        },
        JointKind::SixDof {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::ZERO,
            linear: [ornis_physics::joint::AxisConfig::Free; 3],
            angular: [ornis_physics::joint::AxisConfig::Free; 3],
        },
    ] {
        assert_eq!(cross_row_kind(&kind), None);
    }
    // A cross revolute is reported, never silently dropped.
    let mut e = Engine::new(SolverKind::SequentialImpulse, GRAVITY);
    let a = e.add_body(heavy_anchor(Vec3::ZERO));
    let b = e.add_body(bob(Vec3::new(0.0, -2.0, 0.0)));
    let j = e
        .add_joint(
            a,
            b,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::Y,
                local_axis_b: Vec3::Y,
                limit: None,
                motor: None,
            },
        )
        .expect("valid revolute joint");
    e.set_routing(RoutingKind::Islands);
    e.pin_body_solver(a, Some(SolverKind::SequentialImpulse));
    e.pin_body_solver(b, Some(SolverKind::Avbd));
    for _ in 0..60 {
        e.step(DT);
    }
    match e.cross_joint_status(j) {
        Some(CrossJointStatus::Unsupported { detail }) => assert!(!detail.is_empty()),
        other => panic!("cross revolute must refuse explicitly, got {other:?}"),
    }
    // Pins are Islands-only and ignore statics/invalid handles quietly.
    e.pin_body_solver(BodyHandle::from(9999usize), Some(SolverKind::Avbd));
    let mut single = Engine::new(SolverKind::Avbd, GRAVITY);
    let h = single.add_body(bob(Vec3::ZERO));
    single.pin_body_solver(h, Some(SolverKind::SequentialImpulse));
    assert_eq!(single.body_solver(h), Some(SolverKind::Avbd));
}
