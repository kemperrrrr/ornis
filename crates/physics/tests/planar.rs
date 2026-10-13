//! K1 gate tests: planar (2D) physics as degenerate 3D.
//!
//! (a) drop-settle in the plane on SI, (a') tilt lock across a half turn,
//! (a'') snap and unsupported paths, (b) Strong-Confluence 1 vs 32 threads
//! (nightly, `#[ignore]`), (d) the AVBD known limit (`known_limit_*`,
//! expected to fail). Canonical snapshots are exercised by the existing
//! `si_step` / `avbd_engine` suites and stay untouched here.

use std::f32::consts::PI;

use glam::{Quat, Vec2, Vec3};
use ornis_physics::{
    AvbdEngine, BodyHandle, Engine, JointError, MotionPlane, PLANAR_THICKNESS, PhysicsEngine,
    PlanarError, PlanarPose, PlanarShape, RigidBody, SequentialImpulseEngine, SolverKind, UnitVec3,
    XpbdEngine, add_plane_anchor, constrain_to_plane, planar_body,
};

const DT: f32 = 1.0 / 60.0;
const G: Vec3 = Vec3::new(0.0, -9.81, 0.0);
const DROP_STEPS: usize = 600;
const HALF_TURN_STEPS: usize = 600;
const CUSTOM_PLANE_STEPS: usize = 300;
const CONFLUENCE_STEPS: usize = 180;
const BIN_COUNT: usize = 8;
const BODIES_PER_BIN: usize = 6;
const MAX_DEVIATION: f32 = 5e-3;
const MAX_OFF_NORMAL_SPIN: f32 = 1e-3;
const MAX_TILT: f32 = 0.1;
const MAX_BIN_SPREAD: f32 = 3.5;
const REST_DEVIATION: f32 = 1e-3;
const REST_TILT: f32 = 1e-3;
const REST_SPEED: f32 = 0.05;
const REST_SPIN: f32 = 0.2;
const TILT_LOCK_TOLERANCE: f32 = 1e-4;
const SNAP_TOLERANCE: f32 = 1e-5;
const SNAP_TILT: f32 = 1e-4;
const TWIST_TARGET: f32 = 10.0;
const TWIST_TOLERANCE: f32 = 0.05;

/// Absolute distance from the body center to the plane (m).
fn plane_deviation(plane: &MotionPlane, body: &RigidBody) -> f32 {
    plane.signed_distance(body.position).abs()
}

/// Length of the angular-velocity component off the plane normal (rad/s).
fn off_normal_spin(normal: Vec3, body: &RigidBody) -> f32 {
    (body.angular_velocity - normal * body.angular_velocity.dot(normal)).length()
}

/// Sine of the tilt angle between the body normal and the reference pose
/// (never `acos`: f32 cannot resolve small tilts near 1).
fn tilt(normal: Vec3, q0: Quat, body: &RigidBody) -> f32 {
    let m = (body.orientation * q0.conjugate()) * normal;
    (m - normal * m.dot(normal)).length()
}

/// Regular `sides`-gon with circumradius `radius`, first vertex at angle 0,
/// counter-clockwise.
fn regular_polygon(sides: usize, radius: f32) -> Vec<Vec2> {
    (0..sides)
        .map(|i| {
            let a = std::f32::consts::TAU * i as f32 / sides as f32;
            Vec2::new(radius * a.cos(), radius * a.sin())
        })
        .collect()
}

/// The (a) drop-settle scene: floor, two bin walls, plane anchor and 12
/// planar bodies. Returns `(handle, orientation after constrain)` per body.
fn drop_settle_scene<E: PhysicsEngine>(engine: &mut E) -> Vec<(BodyHandle, Quat)> {
    let plane = MotionPlane::XY;
    engine.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(30.0, 1.0, 2.0),
        0.0,
    ));
    engine.add_body(RigidBody::new_box(
        Vec3::new(-4.0, 2.0, 0.0),
        Vec3::new(0.5, 3.0, 2.0),
        0.0,
    ));
    engine.add_body(RigidBody::new_box(
        Vec3::new(4.0, 2.0, 0.0),
        Vec3::new(0.5, 3.0, 2.0),
        0.0,
    ));
    let anchor = add_plane_anchor(engine, plane);
    let mut bodies = Vec::new();
    for k in 0..12 {
        let shape = match k % 3 {
            0 => PlanarShape::Rect {
                half_extents: Vec2::new(0.4, 0.25),
            },
            1 => PlanarShape::Circle { radius: 0.3 },
            _ => PlanarShape::Polygon {
                vertices: regular_polygon(3 + k % 4, 0.35),
            },
        };
        let pose = PlanarPose {
            position: Vec2::new(-2.4 + (k % 4) as f32 * 1.6, 0.6 + (k / 4) as f32 * 1.0),
            angle: 0.37 * k as f32,
        };
        let body =
            planar_body(&plane, &shape, pose, PLANAR_THICKNESS, 1.0).expect("valid planar body");
        let handle = engine.add_body(body);
        constrain_to_plane(engine, anchor, handle).expect("plane lock");
        let q0 = engine.get_body(handle).expect("body live").orientation;
        bodies.push((handle, q0));
    }
    bodies
}

/// Per-step planar gate of the drop-settle scene: in-plane, calm normals,
/// upright, above the floor and inside the bin.
fn assert_drop_settle_step<E: PhysicsEngine>(
    engine: &E,
    bodies: &[(BodyHandle, Quat)],
    plane: &MotionPlane,
    step: usize,
) {
    let normal = plane.normal();
    for (handle, q0) in bodies {
        let b = engine.get_body(*handle).expect("body live");
        let dev = plane_deviation(plane, b);
        assert!(
            dev < MAX_DEVIATION,
            "step {step}: body {handle:?} left the plane, dev {dev}"
        );
        let spin = off_normal_spin(normal, b);
        assert!(
            spin < MAX_OFF_NORMAL_SPIN,
            "step {step}: body {handle:?} off-normal spin {spin}"
        );
        let t = tilt(normal, *q0, b);
        assert!(t < MAX_TILT, "step {step}: body {handle:?} tilt {t}");
        assert!(
            b.position.y > 0.0,
            "step {step}: body {handle:?} fell through the floor, y {}",
            b.position.y
        );
        let u = plane.to_plane(b.position).x;
        assert!(
            u.abs() < MAX_BIN_SPREAD,
            "step {step}: body {handle:?} left the bin, u {u}"
        );
    }
}

/// Final gate of a drop-settle scene: in plane, upright and settled. The
/// `asleep` probe carries the solver-specific sleep clause (SI reports it;
/// AVBD has none and passes `|_| false`).
fn assert_drop_settle_rest<E: PhysicsEngine>(
    engine: &E,
    bodies: &[(BodyHandle, Quat)],
    plane: &MotionPlane,
    asleep: impl Fn(BodyHandle) -> bool,
) {
    let normal = plane.normal();
    for (handle, q0) in bodies {
        let b = engine.get_body(*handle).expect("body live");
        let dev = plane_deviation(plane, b);
        assert!(dev < REST_DEVIATION, "body {handle:?} rest deviation {dev}");
        let t = tilt(normal, *q0, b);
        assert!(t < REST_TILT, "body {handle:?} rest tilt {t}");
        let calm = b.velocity.length() < REST_SPEED && b.angular_velocity.length() < REST_SPIN;
        assert!(
            asleep(*handle) || calm,
            "body {handle:?} must settle: |v| {} |w| {}",
            b.velocity.length(),
            b.angular_velocity.length()
        );
    }
}

/// Step a drop-settle scene `steps` times with the per-step planar gate.
fn run_drop_settle<E: PhysicsEngine>(
    engine: &mut E,
    bodies: &[(BodyHandle, Quat)],
    plane: &MotionPlane,
    steps: usize,
) {
    for step in 0..steps {
        engine.step(DT);
        assert_drop_settle_step(engine, bodies, plane, step);
    }
}

/// (a) Drop-settle in the plane on SI: 12 planar bodies land, stay in the
/// plane and settle asleep.
#[test]
fn planar_si_drop_settle_stays_in_plane() {
    let mut e = SequentialImpulseEngine::new(G);
    let plane = MotionPlane::XY;
    let bodies = drop_settle_scene(&mut e);
    run_drop_settle(&mut e, &bodies, &plane, DROP_STEPS);
    assert_drop_settle_rest(&e, &bodies, &plane, |h| e.is_asleep(h));
}

/// (a') Tilt-lock regression of the SixDof swing fix: the lock survives a
/// half turn (the pre-fix position pass diverged there) and a fast free
/// yaw.
#[test]
fn planar_si_tilt_lock_survives_half_turn() {
    let plane = MotionPlane::XY;
    let normal = plane.normal();
    // Scenario 1: yaw = π plus a small tilt; the tilt must not grow.
    let mut e = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = add_plane_anchor(&mut e, plane);
    let body = planar_body(
        &plane,
        &PlanarShape::Rect {
            half_extents: Vec2::new(0.5, 0.3),
        },
        PlanarPose {
            position: Vec2::new(1.0, 2.0),
            angle: 0.0,
        },
        PLANAR_THICKNESS,
        1.0,
    )
    .expect("rect builds");
    let h = e.add_body(body);
    constrain_to_plane(&mut e, anchor, h).expect("plane lock");
    let q0 = e.get_body(h).expect("body live").orientation;
    e.get_body_mut(h).expect("body live").orientation =
        Quat::from_rotation_z(PI) * Quat::from_rotation_x(SNAP_TILT);
    for step in 0..HALF_TURN_STEPS {
        e.step(DT);
        let b = e.get_body(h).expect("body live");
        let t = tilt(normal, q0, b);
        assert!(
            t < TILT_LOCK_TOLERANCE,
            "step {step}: tilt drifted near a half turn, {t}"
        );
    }
    // Scenario 2: free fast yaw; twist stays free, the tilt still holds.
    let mut e = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = add_plane_anchor(&mut e, plane);
    let body = planar_body(
        &plane,
        &PlanarShape::Rect {
            half_extents: Vec2::new(0.5, 0.3),
        },
        PlanarPose {
            position: Vec2::new(1.0, 2.0),
            angle: 0.0,
        },
        PLANAR_THICKNESS,
        1.0,
    )
    .expect("rect builds");
    let h = e.add_body(body);
    constrain_to_plane(&mut e, anchor, h).expect("plane lock");
    let q0 = e.get_body(h).expect("body live").orientation;
    {
        let b = e.get_body_mut(h).expect("body live");
        b.orientation = Quat::from_rotation_x(SNAP_TILT);
        b.angular_velocity = Vec3::new(0.0, 0.0, TWIST_TARGET);
    }
    for _ in 0..HALF_TURN_STEPS {
        e.step(DT);
    }
    let b = e.get_body(h).expect("body live");
    let t = tilt(normal, q0, b);
    assert!(t < TILT_LOCK_TOLERANCE, "tilt drifted under fast yaw: {t}");
    assert!(
        (b.angular_velocity.z - TWIST_TARGET).abs() < TWIST_TOLERANCE,
        "twist about the normal must stay free, w_z {}",
        b.angular_velocity.z
    );
}

/// (a'') First snap: the position lands on the plane, the normal velocity
/// component is dropped and only the normal spin survives; the lock then
/// holds the plane without gravity.
#[test]
fn planar_constrain_snaps_body_onto_plane() {
    let mut e = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = add_plane_anchor(&mut e, MotionPlane::XY);
    assert_eq!(anchor.plane(), MotionPlane::XY);
    assert!(e.get_body(anchor.body()).is_some());
    let mut body = RigidBody::new_box(Vec3::new(0.0, 1.0, 0.3), Vec3::new(0.3, 0.3, 0.05), 1.0);
    body.velocity = Vec3::new(1.0, 0.0, 2.0);
    body.angular_velocity = Vec3::new(0.5, 0.0, 1.0);
    let h = e.add_body(body);
    constrain_to_plane(&mut e, anchor, h).expect("plane lock");
    let b = e.get_body(h).expect("body live");
    assert_eq!(b.position.z, 0.0);
    assert_eq!(b.velocity.z, 0.0);
    assert_eq!(b.angular_velocity, Vec3::new(0.0, 0.0, 1.0));
    for step in 0..60 {
        e.step(DT);
        let b = e.get_body(h).expect("body live");
        assert!(
            b.position.z.abs() < SNAP_TOLERANCE,
            "step {step}: z drifted, {}",
            b.position.z
        );
    }
}

/// (a'') Only dynamic bodies take the plane lock, and bad handles are
/// rejected before any mutation.
#[test]
fn planar_constrain_rejects_static_and_invalid_handles() {
    let mut e = SequentialImpulseEngine::new(G);
    let anchor = add_plane_anchor(&mut e, MotionPlane::XY);
    let floor = e.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let err = constrain_to_plane(&mut e, anchor, floor).expect_err("static body refuses the lock");
    assert!(matches!(err, PlanarError::NotDynamic { .. }), "{err:?}");
    let missing = BodyHandle::from_raw(999);
    let err = constrain_to_plane(&mut e, anchor, missing).expect_err("handle out of range");
    assert!(matches!(err, PlanarError::InvalidBody { .. }), "{err:?}");
}

/// (a'') XPBD rejects the plane lock outright (no SixDof there).
#[test]
fn planar_constraint_is_unsupported_on_xpbd() {
    let plane = MotionPlane::XY;
    let mut e = XpbdEngine::new(G);
    let anchor = add_plane_anchor(&mut e, plane);
    let body = planar_body(
        &plane,
        &PlanarShape::Circle { radius: 0.3 },
        PlanarPose {
            position: Vec2::new(0.0, 1.0),
            angle: 0.0,
        },
        PLANAR_THICKNESS,
        1.0,
    )
    .expect("circle builds");
    let h = e.add_body(body);
    let err = constrain_to_plane(&mut e, anchor, h).expect_err("XPBD has no SixDof");
    assert!(
        matches!(err, PlanarError::Joint(JointError::Unsupported { .. })),
        "{err:?}"
    );
}

/// (a'') The same drop-settle physics on a custom plane (YZ, gravity inside
/// the plane): the scene is the (a) setup rotated 90°.
#[test]
fn planar_si_custom_plane_keeps_bodies_in_plane() {
    let plane = MotionPlane::new(
        Vec3::ZERO,
        UnitVec3::try_from(Vec3::X).expect("unit normal"),
    );
    let mut e = SequentialImpulseEngine::new(G);
    e.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(2.0, 1.0, 30.0),
        0.0,
    ));
    let anchor = add_plane_anchor(&mut e, plane);
    let shapes = [
        PlanarShape::Rect {
            half_extents: Vec2::new(0.4, 0.25),
        },
        PlanarShape::Circle { radius: 0.3 },
        PlanarShape::Polygon {
            vertices: regular_polygon(5, 0.35),
        },
        PlanarShape::Rect {
            half_extents: Vec2::new(0.4, 0.25),
        },
    ];
    let mut bodies = Vec::new();
    for (k, shape) in shapes.iter().enumerate() {
        let pose = PlanarPose {
            position: Vec2::new(-1.5 + k as f32, 0.6),
            angle: 0.37 * k as f32,
        };
        let body =
            planar_body(&plane, shape, pose, PLANAR_THICKNESS, 1.0).expect("valid planar body");
        let h = e.add_body(body);
        constrain_to_plane(&mut e, anchor, h).expect("plane lock");
        let q0 = e.get_body(h).expect("body live").orientation;
        bodies.push((h, q0));
    }
    run_drop_settle(&mut e, &bodies, &plane, CUSTOM_PLANE_STEPS);
    // Thresholds as in (a) minus the sleep clause: the probe is always
    // "asleep", so only the planar rest bounds remain.
    assert_drop_settle_rest(&e, &bodies, &plane, |_| true);
}

// Slow (~29s: 48 planar bodies, 180 SI steps under 1- and 32-thread pools).
// Skipped by the default suite; full profile: cargo test -p ornis-physics -- --ignored
#[test]
#[ignore]
fn planar_si_confluence_one_vs_many_threads() {
    let run = |workers: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            let plane = MotionPlane::XY;
            let mut e = Engine::new(SolverKind::SequentialImpulse, G);
            e.add_body(RigidBody::new_box(
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(25.0, 1.0, 2.0),
                0.0,
            ));
            for i in 0..BIN_COUNT {
                let cx = (i as f32 - 3.5) * 5.0;
                e.add_body(RigidBody::new_box(
                    Vec3::new(cx - 2.0, 2.0, 0.0),
                    Vec3::new(0.25, 2.5, 2.0),
                    0.0,
                ));
                e.add_body(RigidBody::new_box(
                    Vec3::new(cx + 2.0, 2.0, 0.0),
                    Vec3::new(0.25, 2.5, 2.0),
                    0.0,
                ));
            }
            let anchor = add_plane_anchor(&mut e, plane);
            for i in 0..BIN_COUNT {
                let cx = (i as f32 - 3.5) * 5.0;
                for k in 0..BODIES_PER_BIN {
                    let shape = match k % 3 {
                        0 => PlanarShape::Rect {
                            half_extents: Vec2::new(0.35, 0.2),
                        },
                        1 => PlanarShape::Circle { radius: 0.25 },
                        _ => PlanarShape::Polygon {
                            vertices: regular_polygon(3 + k % 4, 0.3),
                        },
                    };
                    let pose = PlanarPose {
                        position: Vec2::new(cx - 1.0 + (k % 3) as f32, 0.6 + (k / 3) as f32),
                        angle: 0.37 * (i * BODIES_PER_BIN + k) as f32,
                    };
                    let body = planar_body(&plane, &shape, pose, PLANAR_THICKNESS, 1.0)
                        .expect("valid planar body");
                    let h = e.add_body(body);
                    constrain_to_plane(&mut e, anchor, h).expect("plane lock");
                }
            }
            let mut events = Vec::new();
            for _ in 0..CONFLUENCE_STEPS {
                e.step(DT);
                events.push(format!(
                    "{:?}|{:?}",
                    e.drain_contact_events(),
                    e.drain_trigger_events()
                ));
            }
            let states = (0..e.body_count())
                .map(|h| {
                    let b = e.get_body(BodyHandle::from(h)).expect("body live");
                    (
                        b.position.to_array().map(f32::to_bits),
                        b.orientation.to_array().map(f32::to_bits),
                        b.velocity.to_array().map(f32::to_bits),
                        b.angular_velocity.to_array().map(f32::to_bits),
                    )
                })
                .collect::<Vec<_>>();
            (states, events)
        })
    };
    assert_eq!(run(1), run(32));
}

// EXPECTED TO FAIL is named `known_limit_*` and stays `#[ignore]`d. Determinism
// Nightly runs `--ignored --skip regenerate --skip known_limit_`, so a plain
// `#[ignore]` means "slow but must pass" (gates), never "known broken".
#[test]
#[ignore = "K1, AVBD — известный лимит (в этом срезе не чинится): \
    1) SixDof-строки AVBD (avbd/assembly.rs, avbd/joints_ext.rs) считают \
    полный вектор поворота quat_diff_vec(qrel, j.q_ref).dot(*e) — тот же \
    полуугловой изъян, что был в SI: одиночный вращающийся планарный бокс \
    взрывается (|w| около 188 рад/с, наклон 2.6–3.1 рад); \
    2) даже со swing-фиксом в AVBD (проверено на скретче) куча из 12 \
    планарных прямоугольников в бине взрывается (|v| около 460 м/с); \
    3) тонкие цилиндр и hull проваливаются в пол-бокс (y около -1 через 600 \
    шагов), свободный тонкий пятиугольный hull без джойнта тоже проваливается \
    — отдельный лимит AVBD для тонких GJK-шейпов. Чинить AVBD — отдельный \
    будущий срез."]
fn known_limit_avbd_planar_drop_settle_stays_in_plane() {
    let mut e = AvbdEngine::new(G);
    let plane = MotionPlane::XY;
    let bodies = drop_settle_scene(&mut e);
    run_drop_settle(&mut e, &bodies, &plane, DROP_STEPS);
    assert_drop_settle_rest(&e, &bodies, &plane, |_| false);
}
