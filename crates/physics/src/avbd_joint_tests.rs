//! Joint-only primal/dual regressions; intentionally no contact pairs.
use super::*;

fn row_scene(kind: AvbdJointKind) -> AvbdEngine {
    let mut engine = AvbdEngine::new(Vec3::ZERO);
    let a = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.25, 0.0));
    let b = engine.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.25, 1.0));
    engine
        .add_joint(
            a,
            b,
            JointKind::Ball {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .unwrap();
    engine.joints[0].kind = kind;
    engine.joints[0].ax_a = Vec3::Y;
    engine.joints[0].ax_b = Vec3::Y;
    engine.joints[0].bx_a = Vec3::Y;
    engine.joints[0].bx_b = Vec3::Y;
    engine.pos0 = vec![Vec3::ZERO; 2];
    engine.rot0 = vec![Quat::IDENTITY; 2];
    engine.inertial = engine.pos0.clone();
    engine.inertial_rot = engine.rot0.clone();
    engine
}

#[test]
fn sixdof_angular_rows_commit_duals_in_local_frame() {
    for locked in [false, true] {
        let mut e = row_scene(AvbdJointKind::SixDof);
        e.bodies[0].orientation = Quat::from_rotation_y(0.7);
        e.bodies[1].orientation = e.bodies[0].orientation * Quat::from_rotation_z(0.3);
        e.joints[0].six_ang[2] = if locked {
            AxisConfig::Locked
        } else {
            AxisConfig::Limited {
                min: -0.2,
                max: 0.2,
            }
        };
        e.dual_update();
        let force = if locked {
            e.joints[0].lam_a[2]
        } else {
            e.joints[0].sacc[5]
        };
        let expected = if locked {
            // Locked rows see the violation projected onto the
            // assembly-frame axis (A is pitched 0.7 about Y, so the z-row
            // reads 0.3*cos(0.7)); limited rows read travel directly.
            0.3 * 0.7f32.cos()
        } else {
            0.1
        };
        assert!(
            (force - expected).abs() < 1e-6,
            "angular dual missing: {force} vs {expected}"
        );
        assert!(e.joints[0].pen_a[2] > JOINT_PENALTY_INIT);
        e.bodies[1].orientation = e.bodies[0].orientation;
        e.dual_update();
        if !locked {
            assert_eq!(e.joints[0].sacc[5], 0.0);
        }
    }
}

#[test]
fn weld_and_sixdof_do_not_fight_their_own_contacts() {
    // No-collide is narrowed by design (`discover_pair`): pin-like joints
    // (Ball/Revolute/Prismatic/Distance/Wheel) skip contact between their
    // endpoints, while weld-like assemblies (Fixed/SixDof) keep the
    // contact as structural (SixDof spin-clamp overshoots 4x without it).
    for kind in [
        AvbdJointKind::Ball,
        AvbdJointKind::Revolute,
        AvbdJointKind::Prismatic,
        AvbdJointKind::Distance,
        AvbdJointKind::Wheel,
    ] {
        let mut e = row_scene(kind);
        e.ensure_scratch();
        e.generate_pairs();
        assert!(
            e.pairs.is_empty(),
            "pin joint must not self-collide: {kind:?}"
        );
    }
    for kind in [AvbdJointKind::Fixed, AvbdJointKind::SixDof] {
        let mut e = row_scene(kind);
        e.ensure_scratch();
        e.generate_pairs();
        assert!(
            !e.pairs.is_empty(),
            "weld assembly keeps structural contact by design: {kind:?}"
        );
    }
}

#[test]
fn a4_limit_zero_violation_keeps_warm_reaction() {
    for kind in [
        AvbdJointKind::Prismatic,
        AvbdJointKind::Revolute,
        AvbdJointKind::SixDof,
    ] {
        for angular in [false, true] {
            if (matches!(kind, AvbdJointKind::Prismatic) && angular)
                || (matches!(kind, AvbdJointKind::Revolute) && !angular)
            {
                continue;
            }
            let mut e = row_scene(kind);
            e.joints[0].lim = Some([0.0, 1.0]);
            e.joints[0].lim_dual = -10.0;
            e.joints[0].acc_lim = -10.0;
            if matches!(kind, AvbdJointKind::SixDof) {
                if angular {
                    e.joints[0].six_ang[0] = AxisConfig::Limited { min: 0.0, max: 1.0 };
                    e.joints[0].sacc[3] = -10.0;
                } else {
                    e.joints[0].six_lin[0] = AxisConfig::Limited { min: 0.0, max: 1.0 };
                    e.joints[0].sacc[0] = -10.0;
                }
            }
            e.solve_body(1);
            let movement = if angular {
                e.bodies[1].orientation.to_array()[..3]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
            } else {
                e.bodies[1].position.length()
            };
            assert!(
                movement > 1e-4,
                "warm limit reaction dropped: {kind:?} angular={angular}"
            );
        }
    }
}

#[test]
fn a5_prismatic_high_load_release_both_signs_no_catapult() {
    for sign in [-1.0, 1.0] {
        for load in [9.81, 98.1, 981.0] {
            let mut e = AvbdEngine::new(sign * load * Vec3::Y);
            let a = e.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.25), 0.0));
            let b = e.add_body(RigidBody::new_sphere(sign * 2.9 * Vec3::Y, 0.25, 1.0));
            e.add_joint(
                a,
                b,
                JointKind::Prismatic {
                    local_anchor_a: Vec3::ZERO,
                    local_anchor_b: -sign * Vec3::Y,
                    local_axis_a: Vec3::Y,
                    local_axis_b: Vec3::Y,
                    limit: Some(crate::joint::PrismaticLimit {
                        min: -2.0,
                        max: 2.0,
                    }),
                    motor: None,
                },
            )
            .unwrap();
            let rest = sign * 4.9;
            let mut peak_speed = 0.0f32;
            for step in 0..3000 {
                // Exercise the solver, not a sleeping pose: a frozen body
                // can otherwise conceal a stale multiplier indefinitely.
                e.wake_body(b);
                if !e.sleep_timer.is_empty() {
                    e.sleep_timer[b] = 0.0;
                }
                e.step(DT_STEP);
                if step >= 200 {
                    peak_speed = peak_speed.max(e.bodies[b].velocity.length());
                    assert!(
                        (e.bodies[b].position.y - rest).abs() < 0.02,
                        "load={load} sign={sign} step={step}: {:?}",
                        e.bodies[b]
                    );
                    assert!(
                        e.bodies[b].velocity.length() < 1.0,
                        "catapult load={load} sign={sign} step={step}: {:?}",
                        e.bodies[b]
                    );
                }
            }
            let force = e.joints[0].lim_dual;
            assert!(
                (force - sign * load).abs() < load * 0.02,
                "multiplier must hold the load at C~=0: {force} vs {}",
                sign * load
            );
            eprintln!(
                "load={load} sign={sign}: peak settled speed={peak_speed}, lambda={force}, y={}",
                e.bodies[b].position.y
            );
            e.gravity = Vec3::ZERO;
            e.bodies[b].velocity = -sign * Vec3::Y;
            for step in 0..120 {
                e.wake_body(b);
                e.sleep_timer[b] = 0.0;
                e.step(DT_STEP);
                assert!(
                    e.bodies[b].velocity.length() < 2.0,
                    "stale reaction after release load={load} step={step}"
                );
            }
            assert!(
                sign * (rest - e.bodies[b].position.y) > 0.5,
                "limit retained a phantom holding force after release"
            );
            assert_eq!(e.joints[0].lim_dual, 0.0);
        }
    }
}

#[test]
fn a5_prismatic_dual_carries_force_between_iterations() {
    for sign in [-1.0, 1.0] {
        let mut e = row_scene(AvbdJointKind::Prismatic);
        e.joints[0].lim = Some([-1.0, 1.0]);
        e.joints[0].pen_l[2] = 100.0;
        e.joints[0].lim_dual = sign * 10.0;
        e.bodies[1].position.y = sign * 1.01;
        let c = e.bodies[1].position.y - sign;
        let expected = sign * 10.0 + 100.0 * c;
        e.dual_update();
        assert!(
            (e.joints[0].lim_dual - expected).abs() < 1e-5,
            "dual must carry old force: {} vs {expected}",
            e.joints[0].lim_dual
        );
        let carried = e.joints[0].lim_dual;
        e.bodies[1].position.y = sign; // zero C, nonzero reaction
        e.dual_update();
        assert_eq!(e.joints[0].lim_dual, carried);
        // Inside the active slop band: the signed residual must unwind,
        // not accumulate its absolute value or stick at a hard force cap.
        e.bodies[1].position.y = sign * (1.0 - LIMIT_SLOP_LIN * 0.5);
        e.dual_update();
        assert!(e.joints[0].lim_dual.abs() < carried.abs());
        e.bodies[1].position.y = 0.0;
        e.dual_update();
        assert_eq!(e.joints[0].lim_dual, 0.0);
    }
}

#[test]
fn a4_linear_zero_violation_keeps_warm_reaction() {
    let mut e = row_scene(AvbdJointKind::Ball);
    e.joints[0].lam_l[0] = 10.0;
    e.joints[0].pen_l[0] = 100.0;
    e.dual_update(); // Zero C must neither integrate lambda nor ramp K.
    assert_eq!(e.joints[0].lam_l[0], 10.0);
    assert_eq!(e.joints[0].pen_l[0], 100.0);
    e.solve_body(1);
    let expected = 10.0 / (1.0 / (DT_STEP * DT_STEP) + 100.0);
    assert!(
        (e.bodies[1].position.x - expected).abs() < 1e-8,
        "warm reaction dropped: {:?}, expected {expected}",
        e.bodies[1].position
    );
}

#[test]
fn a4_angular_zero_violation_keeps_warm_reaction() {
    let mut e = row_scene(AvbdJointKind::Fixed);
    e.joints[0].lam_a[0] = 10.0;
    e.joints[0].pen_a[0] = 100.0;
    e.dual_update();
    assert_eq!(e.joints[0].lam_a[0], 10.0);
    assert_eq!(e.joints[0].pen_a[0], 100.0);
    e.solve_body(1);
    assert!(
        e.bodies[1].orientation.x < -1e-4,
        "warm torque dropped: {:?}",
        e.bodies[1].orientation
    );
}
