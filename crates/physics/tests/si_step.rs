//! SI step/island pins: integration, substeps, stacks, joints and solver math.

use glam::{Mat3, Mat4, Quat, Vec3};
use ornis_physics::engine::{
    apply_impulse, effective_mass, inv_inertia_axis, mul_inv_inertia, point_velocity,
    solve_normal_block, solve_small,
};
use ornis_physics::{
    AxisConfig, BodyHandle, BroadPhaseKind, JointHandle, JointKind, PhysicsEngine, PrismaticLimit,
    PrismaticMotor, RevoluteMotor, RigidBody, SequentialImpulseEngine, StepBudget, WheelSuspension,
};

fn dense_shedding_scene() -> SequentialImpulseEngine {
    // 24 overlapping dynamic boxes (276 candidate pairs) plus one fast
    // body forcing the 12-substep speed request.
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    for i in 0..24 {
        physics.add_body(RigidBody::new_box(
            Vec3::new((i % 5) as f32 * 0.1, (i / 5) as f32 * 0.1, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    let mut fast = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    fast.velocity = Vec3::new(0.0, -40.0, 0.0);
    physics.add_body(fast);
    physics
}

fn angular_momentum(body: &RigidBody) -> Vec3 {
    let w_body = body.orientation.conjugate() * body.angular_velocity;
    body.orientation * (body.inertia * w_body)
}

fn rotational_energy(body: &RigidBody) -> f32 {
    let w = body.orientation.conjugate() * body.angular_velocity;
    0.5 * body.inertia.dot(w * w)
}

/// Canonical cross-platform determinism snapshot (Box3D-level claim):
/// a fixed heterogeneous scene (stack, sphere, fast drop, jointed
/// pendulum) stepped 120 times, compared bit-for-bit against the
/// checked-in `tests/data/determinism_snapshot.hex` generated on ARM.
/// CI runs this on x86_64: any float-contraction or codegen drift
/// (including LLVM fusing mul+add into fma, which stable rustc cannot
/// disable) fails loudly here instead of silently diverging.
/// Re-baseline ONLY for intentional solver changes: run
/// `determinism_snapshot_regenerate` (ignored), inspect the diff, commit.
fn determinism_snapshot_scene() -> SequentialImpulseEngine {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    for i in 0..4 {
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5 + i as f32 * 1.02, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
    }
    physics.add_body(RigidBody::new_sphere(Vec3::new(3.0, 6.0, 0.0), 0.5, 1.0));
    let mut fast = RigidBody::new_box(Vec3::new(-3.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    fast.velocity = Vec3::new(0.0, -30.0, 0.0);
    physics.add_body(fast);
    let anchor = physics.add_body(RigidBody::new_box(
        Vec3::new(6.0, 3.0, 0.0),
        Vec3::splat(0.5),
        0.0,
    ));
    let arm = physics.add_body(RigidBody::new_box(
        Vec3::new(6.0, 1.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            arm,
            JointKind::Ball {
                local_anchor_a: Vec3::new(0.0, -1.0, 0.0),
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
            },
        )
        .expect("valid joint");
    physics
}

fn determinism_snapshot_render(physics: &SequentialImpulseEngine) -> String {
    let mut out = format!(
        "ornis-determinism-snapshot v1 bodies={} steps=120 dt=0.0166667\n",
        physics.bodies.len()
    );
    for b in &physics.bodies {
        let mut first = true;
        for x in b
            .position
            .to_array()
            .into_iter()
            .chain(b.orientation.to_array())
            .chain(b.velocity.to_array())
            .chain(b.angular_velocity.to_array())
        {
            if !first {
                out.push(' ');
            }
            first = false;
            out.push_str(&format!("{:08x}", x.to_bits()));
        }
        out.push('\n');
    }
    out
}

/// World-space distance between the two anchor points of a joint.
fn joint_anchor_error(
    physics: &SequentialImpulseEngine,
    ja: BodyHandle,
    jb: BodyHandle,
    la: Vec3,
    lb: Vec3,
) -> f32 {
    let (a, b) = (physics.get_body(ja).unwrap(), physics.get_body(jb).unwrap());
    let pa = a.position + a.orientation * la;
    let pb = b.position + b.orientation * lb;
    (pa - pb).length()
}

/// Orientation as a glam rotation matrix (independent oracle for
/// `mul_inv_inertia`, which feeds `k_entry`/`effective_mass`).
fn rot_mat(q: Quat) -> Mat3 {
    Mat3::from_quat(q)
}

#[test]
fn sphere_falls() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let sphere = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 10.0, 0.0), 1.0, 1.0));
    physics.step(1.0 / 60.0);
    let body = physics.get_body(sphere).unwrap();
    assert!(body.position.y < 10.0);
}

#[test]
fn static_body_does_not_fall() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let ground = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(10.0, 1.0, 10.0),
        0.0,
    ));
    physics.step(1.0 / 60.0);
    let body = physics.get_body(ground).unwrap();
    assert_eq!(body.position.y, -1.0);
}

#[test]
fn broadphase_backend_can_be_selected_explicitly() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    assert_eq!(physics.broadphase_kind(), BroadPhaseKind::UniformGrid);
    physics.set_broadphase(BroadPhaseKind::UniformGrid);
    assert_eq!(physics.broadphase_kind(), BroadPhaseKind::UniformGrid);
    physics.set_uniform_grid_cell_size(1.0);
    assert_eq!(physics.broadphase_kind(), BroadPhaseKind::UniformGrid);
    physics.set_broadphase(BroadPhaseKind::SweepAndPrune);
    assert_eq!(physics.broadphase_kind(), BroadPhaseKind::SweepAndPrune);
}

#[test]
fn auto_broadphase_routes_small_scene_to_sweep_and_steps() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.set_broadphase(BroadPhaseKind::Auto);
    assert_eq!(physics.broadphase_kind(), BroadPhaseKind::Auto);
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::splat(0.4),
        1.0,
    ));
    physics.step(1.0 / 60.0);
    assert_eq!(
        physics.auto_active_broadphase(),
        Some(BroadPhaseKind::SweepAndPrune)
    );
    let body = physics.get_body(BodyHandle::from_raw(1)).unwrap();
    assert!(body.position.y < 5.0, "dynamic body still falls under Auto");

    // Explicit selections report no auto-active backend.
    physics.set_broadphase(BroadPhaseKind::UniformGrid);
    assert_eq!(physics.auto_active_broadphase(), None);
}

#[test]
fn step_budget_shed_arithmetic_is_deterministic() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.set_step_budget(Some(StepBudget {
        max_pair_substeps: 200_000,
        min_substeps: 4,
    }));
    assert_eq!(physics.apply_step_budget(12, 0), (12, 0));
    // Tiled-10k-cold scale stays untouched under the default budget.
    assert_eq!(physics.apply_step_budget(12, 14_161), (12, 0));
    assert_eq!(physics.apply_step_budget(12, 1_000_000), (4, 8));
    // 100k-tiled operating point (100k candidates × 12 requested):
    // the budget sheds to the floor instead of running away.
    assert_eq!(physics.apply_step_budget(12, 100_000), (4, 8));
    // The floor is never shed, even under extreme load.
    assert_eq!(physics.apply_step_budget(4, 1_000_000), (4, 0));
    physics.set_step_budget(None);
    assert_eq!(physics.apply_step_budget(12, 1_000_000), (12, 0));
}

#[test]
fn step_budget_leaves_typical_scenes_untouched() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let mut fast = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    fast.velocity = Vec3::new(0.0, -40.0, 0.0);
    physics.add_body(fast);
    physics.step(1.0 / 60.0);
    assert_eq!(physics.step_timing().substeps, 12);
    assert_eq!(physics.last_substep_shed(), 0);
}

#[test]
fn step_budget_sheds_substeps_but_never_pairs() {
    let mut budgeted = dense_shedding_scene();
    budgeted.set_step_budget(Some(StepBudget {
        max_pair_substeps: 100,
        min_substeps: 4,
    }));
    budgeted.step(1.0 / 60.0);
    assert_eq!(budgeted.step_timing().substeps, 4);
    assert_eq!(budgeted.last_substep_shed(), 8);

    // Same initial state, not the same final trajectory. The backend
    // stats are overwritten by end-of-step trigger/CCD queries; compare
    // the complete frame candidate input retained by the bucket builder.
    let mut unbudgeted = dense_shedding_scene();
    unbudgeted.set_step_budget(None);
    unbudgeted.step(1.0 / 60.0);
    assert_eq!(unbudgeted.step_timing().substeps, 12);
    assert_eq!(unbudgeted.last_substep_shed(), 0);
    let mut budget_pairs = budgeted.scratch_pairs.clone();
    let mut full_pairs = unbudgeted.scratch_pairs.clone();
    budget_pairs.sort_unstable();
    full_pairs.sort_unstable();
    assert!(!budget_pairs.is_empty());
    assert_eq!(
        budget_pairs, full_pairs,
        "budget must retain every candidate pair"
    );
}

/// Dzhanibekov discriminant for the gyroscopic correction: half extents
/// (0.2, 0.6, 0.4) give Ix > Iz > Iy, so body Z is the intermediate
/// axis — spin about it must tumble end over end. Without the correction
/// the spin axis stays world-fixed and the body Z rides a ~1.7deg cone
/// (min dot ≈ 0.998), so a deep flip is unreachable by construction.
#[test]
fn gyroscopic_intermediate_axis_spin_tumbles() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::new(0.2, 0.6, 0.4), 1.0);
    body.angular_velocity = Vec3::new(0.3, 0.0, 10.0);
    physics.add_body(body);

    let l0 = angular_momentum(&physics.bodies[0]);
    let e0 = rotational_energy(&physics.bodies[0]);
    let mut min_dot = 1.0f32;
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
        let z_now = physics.bodies[0].orientation * Vec3::Z;
        min_dot = min_dot.min(z_now.dot(Vec3::Z));
    }
    let b = &physics.bodies[0];
    let dl = (angular_momentum(b) - l0).length() / l0.length();
    let de = ((rotational_energy(b) - e0) / e0).abs();
    eprintln!("dzhanibekov: min_dot={min_dot:.3} dL/L={dl:.4} dE/E={de:.4}");
    assert!(min_dot < -0.5, "no Dzhanibekov flip, min_dot={min_dot}");
    // Free motion has no torques, so these only drift by discretization
    // (measured 0.055/0.108 at 300 steps; bounds carry ~1.6x margin).
    assert!(dl < 0.09, "angular momentum drifted, dL/L={dl}");
    assert!(de < 0.18, "energy drifted, dE/E={de}");
}

/// Guard against overcorrection: spin about the major axis (body X here)
/// is genuinely stable and must stay aligned.
#[test]
fn gyroscopic_major_axis_spin_stays_stable() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::new(0.2, 0.6, 0.4), 1.0);
    body.angular_velocity = Vec3::new(10.0, 0.3, 0.0);
    physics.add_body(body);

    let mut min_dot = 1.0f32;
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
        let x_now = physics.bodies[0].orientation * Vec3::X;
        min_dot = min_dot.min(x_now.dot(Vec3::X));
    }
    assert!(min_dot > 0.9, "major-axis spin wandered, min_dot={min_dot}");
}

/// Isotropic fast path: a spinning cube must come back bit-identical —
/// the gyroscopic term is exactly zero there, so the skip gate fires.
#[test]
fn gyroscopic_isotropic_spin_is_untouched() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let mut body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
    body.angular_velocity = Vec3::new(1.0, 2.0, 3.0);
    physics.add_body(body);

    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }
    assert_eq!(
        physics.bodies[0].angular_velocity.to_array(),
        [1.0, 2.0, 3.0],
        "isotropic skip gate leaked"
    );
}

#[test]
fn adaptive_substeps_scale_with_body_speed() {
    // A fast body needs the full substep cap; a resting scene drops to the
    // minimum so it can sleep cheaply.
    let mut fast = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    fast.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let mut ball = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    ball.velocity = Vec3::new(0.0, -40.0, 0.0);
    fast.add_body(ball);
    fast.step(1.0 / 60.0);
    assert_eq!(fast.step_timing().substeps, 12, "fast body uses full cap");

    // Resting grid: after settling, velocities are ~0 -> minimum substeps.
    let mut rest = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    rest.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(100.0, 0.5, 100.0),
        0.0,
    ));
    for i in 0..4 {
        rest.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.4 + i as f32 * 0.82, 0.0),
            Vec3::splat(0.4),
            1.0,
        ));
    }
    for _ in 0..240 {
        rest.step(1.0 / 60.0);
    }
    assert!(
        rest.step_timing().substeps < 12,
        "resting scene adapts below the 12 cap (got {})",
        rest.step_timing().substeps
    );
}

#[test]
fn per_island_iters_scale_with_speed() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let dt = 1.0 / 60.0;
    // slow island → minimal iters (3 vel from 4/12*8), fast → full cap
    assert_eq!(physics.adaptive_iters_for_island(0.0, dt, 8), 3);
    assert_eq!(physics.adaptive_iters_for_island(0.1, dt, 8), 3);
    assert!(physics.adaptive_iters_for_island(2.0, dt, 8) > 3);
    assert!(physics.adaptive_iters_for_island(2.0, dt, 8) < 8);
    assert_eq!(physics.adaptive_iters_for_island(40.0, dt, 8), 8);
    assert_eq!(physics.adaptive_iters_for_island(40.0, dt, 4), 4);
    // penetration drives iters even when speed is zero
    assert_eq!(
        physics.adaptive_iters_for_island_with_pen(0.0, 0.12, dt, 8),
        8
    );
    assert!(physics.adaptive_iters_for_island_with_pen(0.0, 0.06, dt, 8) > 3);
    // respects substeps cap — still returns scaled within base
    physics.set_substeps(1);
    assert_eq!(physics.adaptive_iters_for_island(40.0, dt, 8), 8);
}

/// Tall-stack stability guard for settled-cost work: a 5-box tower
/// stands 5 seconds without toppling or drifting (per-island minimum
/// iterations must still correct residual penetration). Taller towers
/// (6+ with this geometry) topple from micro-asymmetry amplification —
/// a pre-existing solver limit, not a settled-cost regression (measured
/// via a scratch probe: 4–5 stand, 6+ scatter; see perf_probe).
#[test]
fn tall_stack_stands_still() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let mut handles = Vec::new();
    for level in 0..5 {
        handles.push(physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.4 + level as f32 * 0.82, 0.0),
            Vec3::splat(0.4),
            1.0,
        )));
    }
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    for (level, &h) in handles.iter().enumerate() {
        let b = physics.get_body(h).unwrap();
        let expected_y = 0.4 + level as f32 * 0.82;
        assert!(
            (b.position.y - expected_y).abs() < 0.3,
            "box {level} rest height ≈{expected_y}, got {}",
            b.position.y
        );
        assert!(
            b.position.x.abs() < 0.3 && b.position.z.abs() < 0.3,
            "box {level} drifted: {:?}",
            b.position
        );
    }
}

/// Anisotropic floor (ODE fdir1/mu/mu2 parity): slick along X, grippy
/// along Z. A box kicked diagonally must keep its X slide while the Z
/// component dies — separate per-axis Coulomb caps, one basis.
#[test]
fn aniso_floor_channels_sliding() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let mut floor = RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(5.0, 1.0, 5.0), 0.0);
    floor.friction = 0.0;
    floor.friction_transverse = 1.0;
    floor.friction_dir = Some(Vec3::X);
    physics.add_body(floor);
    let mut b = RigidBody::new_box(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(0.5), 1.0);
    b.friction = 0.0;
    b.velocity = Vec3::new(3.0, 0.0, 3.0);
    physics.add_body(b);
    // 60 steps ≈ 1 s: lands at ~0.55 s, then ~0.45 s of channeled
    // slide — short enough to stay on the 10 m floor (x ≈ 3 < 5).
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }
    let v = physics.bodies[1].velocity;
    let y = physics.bodies[1].position.y;
    eprintln!("ANISO v={v:?} y={y}");
    assert!(v.x > 2.5, "slick axis must preserve slide, got {v:?}");
    assert!(v.z.abs() < 0.4, "grippy axis must kill slide, got {v:?}");
    assert!(
        v.y.abs() < 1.0 && (y - 0.5).abs() < 0.1,
        "box must rest ON the floor, not fall through, got {v:?} y={y}"
    );
}

/// Rolling resistance (MuJoCo parity): a ball with rolling friction
/// must stop; the zero-coefficient control keeps rolling.
#[test]
fn rolling_resistance_stops_ball() {
    fn run(rolling: f32) -> (Vec3, f32) {
        let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
        // ±30 m floor: 120 steps at ~5 m/s stay on the slab, so both
        // balls are measured in rolling contact, never in free fall.
        let mut floor =
            RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(30.0, 1.0, 30.0), 0.0);
        floor.rolling_friction = rolling;
        physics.add_body(floor);
        let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.6, 0.0), 0.5, 1.0);
        ball.rolling_friction = rolling;
        // 240 steps: the damped ball stops (~190 steps at μr = 0.1);
        // the control is in pure rolling (zero slip) and coasts at
        // 8·5/7 ≈ 5.71 indefinitely — x ≈ 23 < 30 stays on the slab.
        ball.velocity = Vec3::new(8.0, 0.0, 0.0);
        physics.add_body(ball);
        for _ in 0..240 {
            physics.step(1.0 / 60.0);
        }
        (physics.bodies[1].velocity, physics.bodies[1].position.y)
    }
    let (stopped, ys) = run(0.1);
    let (rolling, yr) = run(0.0);
    eprintln!("ROLL stopped={stopped:?} y={ys} control={rolling:?} y={yr}");
    assert!(
        stopped.length() < 1.0 && (ys - 0.5).abs() < 0.1,
        "rolling friction must stop the ball ON the floor, got {stopped:?} y={ys}"
    );
    assert!(
        rolling.x > 3.0 && rolling.y.abs() < 1.0 && (yr - 0.5).abs() < 0.1,
        "zero rolling friction must keep it rolling on the floor, got {rolling:?} y={yr}"
    );
}

/// Torsional friction (MuJoCo parity): a sphere spinning about the
/// contact normal (no slip, so slide friction is blind to it) must
/// lose its spin; the control keeps spinning.
#[test]
fn torsion_friction_kills_spin() {
    fn run(torsion: f32) -> Vec3 {
        let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
        let mut floor =
            RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(5.0, 1.0, 5.0), 0.0);
        floor.torsion_friction = torsion;
        physics.add_body(floor);
        let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.6, 0.0), 0.5, 1.0);
        ball.torsion_friction = torsion;
        ball.angular_velocity = Vec3::new(0.0, 10.0, 0.0);
        physics.add_body(ball);
        for _ in 0..600 {
            physics.step(1.0 / 60.0);
        }
        physics.bodies[1].angular_velocity
    }
    let damped = run(0.05);
    let spinning = run(0.0);
    eprintln!("TORSION damped={damped:?} control={spinning:?}");
    assert!(
        damped.y.abs() < 2.0,
        "torsion friction must kill spin, got {damped:?}"
    );
    assert!(
        spinning.y.abs() > 5.0,
        "zero torsion must preserve spin, got {spinning:?}"
    );
}

/// Sphere-vs-box contact point (regression for the half-depth bug):
/// with slide friction and NO rolling resistance, a rolling ball must
/// converge to TRUE rolling (contact slip → 0). The old midpoint
/// contact sat at half the radius depth, so the solver saw zero slip
/// at a phantom "half-rolling" v = ω·r/2 and held it forever.
#[test]
fn rolling_converges_to_true_rolling_not_half() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(30.0, 1.0, 30.0),
        0.0,
    ));
    let mut ball = RigidBody::new_sphere(Vec3::new(0.0, 0.6, 0.0), 0.5, 1.0);
    ball.velocity = Vec3::new(8.0, 0.0, 0.0);
    physics.add_body(ball);
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = &physics.bodies[1];
    // Still rolling (above the sleep threshold), so the slip below is
    // solver-converged, not frozen.
    assert!(
        b.velocity.x > 3.0,
        "ball must still be rolling, got {:?}",
        b.velocity
    );
    let slip = b.velocity.x + b.angular_velocity.z * 0.5;
    eprintln!(
        "ROLLTRUE v={:?} w={:?} slip={slip}",
        b.velocity, b.angular_velocity
    );
    assert!(
        slip.abs() < 0.2,
        "true rolling means zero contact slip, got {slip} (half-rolling phantom if ~v/2)"
    );
}

#[test]
fn angular_velocity_rotates_body() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let handle = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
    physics
        .get_body_mut(handle)
        .unwrap()
        .set_angular_velocity(Vec3::new(0.0, 2.0, 0.0));
    physics.step(1.0 / 60.0);
    let body = physics.get_body(handle).unwrap();
    // Orientation must have changed and remain a unit quaternion.
    assert!(
        body.orientation.to_axis_angle().1.abs() > 1e-4,
        "should have rotated about Y"
    );
    assert!(
        (body.orientation.length() - 1.0).abs() < 1e-4,
        "unit quaternion preserved"
    );
}

#[test]
fn torque_turns_body() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let sphere = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0));
    // Apply torque around Z -> angular velocity must appear.
    let w_after = {
        physics
            .get_body_mut(sphere)
            .unwrap()
            .apply_torque(Vec3::new(0.0, 0.0, 1.0));
        physics.step(1.0 / 60.0);
        physics.get_body(sphere).unwrap().angular_velocity
    };
    assert!(
        w_after.z.abs() > 1e-5,
        "torque must produce angular velocity, got {w_after:?}"
    );
}

#[test]
fn two_box_stack_stays_stable() {
    // G2 gate: a 2-box stack stands for 5 seconds without drift or toppling.
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let lower = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.5, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        1.0,
    ));
    let upper = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.55, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        1.0,
    ));
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    let lo = physics.get_body(lower).unwrap();
    let hi = physics.get_body(upper).unwrap();
    assert!(
        (lo.position.y - 0.5).abs() < 0.05,
        "lower box rest height, got {}",
        lo.position.y
    );
    assert!(
        (hi.position.y - 1.5).abs() < 0.08,
        "upper box rest height, got {}",
        hi.position.y
    );
    // No horizontal drift: the stack must stay centred.
    assert!(
        lo.position.x.abs() < 0.05 && lo.position.z.abs() < 0.05,
        "lower box drifted: {:?}",
        lo.position
    );
    assert!(
        hi.position.x.abs() < 0.08 && hi.position.z.abs() < 0.08,
        "upper box drifted: {:?}",
        hi.position
    );
    assert!(
        lo.velocity.length() < 0.05 && hi.velocity.length() < 0.05,
        "stack not settled: {:?} / {:?}",
        lo.velocity,
        hi.velocity
    );
    assert!(
        lo.angular_velocity.length() < 0.05 && hi.angular_velocity.length() < 0.05,
        "stack spinning: {:?} / {:?}",
        lo.angular_velocity,
        hi.angular_velocity
    );
}

#[test]
fn four_box_stack_stays_stable() {
    // G3 gate: a 4-box stack stands for 5 seconds without drift or topple.
    // Taller stacks need the iterated cross-manifold position solve —
    // per-manifold nested correction cannot balance the chain.
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let mut handles = Vec::new();
    for level in 0..4 {
        handles.push(physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
            Vec3::new(0.5, 0.5, 0.5),
            1.0,
        )));
    }
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    for (level, &h) in handles.iter().enumerate() {
        let b = physics.get_body(h).unwrap();
        let expected_y = 0.5 + level as f32;
        assert!(
            (b.position.y - expected_y).abs() < 0.1,
            "box {level} rest height ≈{expected_y}, got {}",
            b.position.y
        );
        assert!(
            b.position.x.abs() < 0.15 && b.position.z.abs() < 0.15,
            "box {level} drifted: {:?}",
            b.position
        );
        assert!(
            b.velocity.length() < 0.08,
            "box {level} not settled: {:?}",
            b.velocity
        );
        assert!(
            b.angular_velocity.length() < 0.08,
            "box {level} spinning: {:?}",
            b.angular_velocity
        );
    }
}

#[test]
fn solver_is_deterministic_across_thread_counts() {
    // G7 gate: per-island parallel dispatch must be bit-identical to the
    // sequential run. Islands are disjoint over dynamic bodies and the
    // warm cache is merged by disjoint keys, so any difference here is a
    // data race, not float noise. The scene (9 separate 4-box stacks on
    // a floor) is wide enough to engage the rayon path: ≥2 islands,
    // ≥24 manifolds.
    fn build_scene() -> SequentialImpulseEngine {
        let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(8.0, 1.0, 8.0),
            0.0,
        ));
        for gx in 0..3 {
            for gz in 0..3 {
                let base = Vec3::new(gx as f32 * 2.5 - 2.5, 0.0, gz as f32 * 2.5 - 2.5);
                for level in 0..4 {
                    physics.add_body(RigidBody::new_box(
                        base + Vec3::new(0.0, 0.5 + level as f32 * 1.02, 0.0),
                        Vec3::new(0.5, 0.5, 0.5),
                        1.0,
                    ));
                }
            }
        }
        physics
    }
    /// (position, orientation, velocity, angular velocity) as f32 bits.
    type Snapshot = ([u32; 3], [u32; 4], [u32; 3], [u32; 3]);
    fn run(threads: usize) -> Vec<Snapshot> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let mut physics = build_scene();
            for _ in 0..120 {
                physics.step(1.0 / 60.0);
            }
            (0..physics.bodies.len())
                .map(|i| {
                    let b = &physics.bodies[i];
                    let (p, o) = (b.position.to_array(), b.orientation.to_array());
                    let (v, w) = (b.velocity.to_array(), b.angular_velocity.to_array());
                    (
                        p.map(f32::to_bits),
                        o.map(f32::to_bits),
                        v.map(f32::to_bits),
                        w.map(f32::to_bits),
                    )
                })
                .collect()
        })
    }
    let single = run(1);
    let multi = run(4);
    assert_eq!(single.len(), multi.len(), "body count differs between runs");
    for (i, (a, b)) in single.iter().zip(multi.iter()).enumerate() {
        assert_eq!(a, b, "body {i} diverged between 1-thread and 4-thread runs");
    }
}

#[test]
fn solver_is_deterministic_across_runs() {
    // Same binary, two fresh engines: every hash map gets a fresh hasher
    // state per instance, so bit-identical snapshots prove iteration
    // order cannot leak into float state (not merely same-seed luck).
    // Uses a small heterogeneous scene (stacked boxes, a resting
    // sphere and a fast drop), so caches, islands and CCD all engage.
    fn build() -> SequentialImpulseEngine {
        let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
        physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(10.0, 1.0, 10.0),
            0.0,
        ));
        for i in 0..4 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(0.0, 0.5 + i as f32 * 1.02, 0.0),
                Vec3::splat(0.5),
                1.0,
            ));
        }
        physics.add_body(RigidBody::new_sphere(Vec3::new(3.0, 6.0, 0.0), 0.5, 1.0));
        let mut fast = RigidBody::new_box(Vec3::new(-3.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
        fast.velocity = Vec3::new(0.0, -30.0, 0.0);
        physics.add_body(fast);
        physics
    }
    #[allow(clippy::type_complexity)]
    fn snapshot(
        physics: &SequentialImpulseEngine,
    ) -> Vec<([u32; 3], [u32; 4], [u32; 3], [u32; 3])> {
        physics
            .bodies
            .iter()
            .map(|b| {
                (
                    b.position.to_array().map(f32::to_bits),
                    b.orientation.to_array().map(f32::to_bits),
                    b.velocity.to_array().map(f32::to_bits),
                    b.angular_velocity.to_array().map(f32::to_bits),
                )
            })
            .collect()
    }
    let mut first = build();
    let mut second = build();
    for _ in 0..120 {
        first.step(1.0 / 60.0);
        second.step(1.0 / 60.0);
    }
    assert_eq!(snapshot(&first), snapshot(&second));
}

#[test]
fn determinism_snapshot_matches_canonical() {
    let mut physics = determinism_snapshot_scene();
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let expected = include_str!("data/determinism_snapshot.hex");
    assert_eq!(
        determinism_snapshot_render(&physics),
        expected,
        "simulation bits drifted: intentional solver change? re-baseline via \
         determinism_snapshot_regenerate, else float/codegen drift"
    );
}

#[test]
#[ignore]
fn determinism_snapshot_regenerate() {
    let mut physics = determinism_snapshot_scene();
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let path = format!(
        "{}/tests/data/determinism_snapshot.hex",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(path, determinism_snapshot_render(&physics)).unwrap();
}

#[test]
fn tilted_box_falls_flat() {
    // G3 gate: a box dropped at a 20° tilt lands on an edge, tips over,
    // and comes to rest flat on the floor (4-point face manifold).
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let tilt = Quat::from_rotation_z(20.0f32.to_radians());
    let top = physics.add_body(
        RigidBody::new_box(Vec3::new(0.0, 1.2, 0.0), Vec3::new(0.5, 0.5, 0.5), 1.0)
            .with_orientation(tilt),
    );
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(top).unwrap();
    // Resting flat: the box's local +Y axis must align with world ±Y.
    let up = b.orientation * Vec3::Y;
    assert!(
        up.dot(Vec3::Y).abs() > 0.99,
        "box should lie flat, up={up:?}"
    );
    assert!(
        (b.position.y - 0.5).abs() < 0.08,
        "flat rest height ≈0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.05 && b.angular_velocity.length() < 0.05,
        "not settled: {:?} / {:?}",
        b.velocity,
        b.angular_velocity
    );
}

#[test]
fn ball_joint_pendulum_holds_anchor() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_sphere(Vec3::ZERO, 0.1, 0.0));
    // Pendulum bob released off to the side: it must swing, not fall.
    let bob = physics.add_body(RigidBody::new_sphere(Vec3::new(1.0, -1.0, 0.0), 0.25, 1.0));
    let lb = Vec3::new(-1.0, 1.0, 0.0); // world anchor = origin
    physics
        .add_joint(
            anchor,
            bob,
            JointKind::Ball {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: lb,
            },
        )
        .expect("valid joint");
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
        let err = joint_anchor_error(&physics, anchor, bob, Vec3::ZERO, lb);
        assert!(err < 0.05, "anchor drifted apart: {err}");
    }
    let b = physics.get_body(bob).unwrap();
    // Still hanging from the anchor: distance to the pivot stays ≈ √2.
    let dist = b.position.length();
    assert!(
        (dist - std::f32::consts::SQRT_2).abs() < 0.15,
        "pendulum length drifted: {dist}"
    );
    // And it did swing at some point (started at x=1, must reach x<0).
    // (Checked implicitly: a falling bob would have y << -1.5.)
    assert!(
        b.position.y > -1.6,
        "bob fell off the joint: {:?}",
        b.position
    );
}

#[test]
fn ball_joint_chain_hangs() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let mut prev = anchor;
    let mut links = Vec::new();
    for k in 1..=3 {
        let link = physics.add_body(RigidBody::new_box(
            Vec3::new(0.0, -(k as f32), 0.0),
            Vec3::splat(0.1),
            0.5,
        ));
        physics
            .add_joint(
                prev,
                link,
                JointKind::Ball {
                    local_anchor_a: if prev == anchor {
                        Vec3::ZERO
                    } else {
                        Vec3::new(0.0, -0.5, 0.0)
                    },
                    local_anchor_b: Vec3::new(0.0, 0.5, 0.0),
                },
            )
            .expect("valid joint");
        links.push(link);
        prev = link;
    }
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    // Every link still connected: anchor pairs coincide.
    let mut prev = anchor;
    let mut prev_anchor = Vec3::ZERO;
    for (k, &link) in links.iter().enumerate() {
        let lb = Vec3::new(0.0, 0.5, 0.0);
        let err = joint_anchor_error(&physics, prev, link, prev_anchor, lb);
        assert!(err < 0.1, "chain link {k} detached: err={err}");
        let b = physics.get_body(link).unwrap();
        assert!(
            b.position.y > -(k as f32) - 1.5,
            "link {k} fell too far: {:?}",
            b.position
        );
        prev = link;
        prev_anchor = Vec3::new(0.0, -0.5, 0.0);
    }
}

#[test]
fn revolute_hinge_rotates_about_axis_only() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    // Arm hangs with its top at the origin: center one meter below. The
    // jointed pair does not collide (a hinge pin passes through the arm),
    // so the test measures the JOINT, not contact friction.
    let arm = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.1, 1.0, 0.1),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            arm,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0), // arm top at origin
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: None,
            },
        )
        .expect("valid joint");
    // Kick sideways so the pendulum arm swings about the Z hinge.
    physics.get_body_mut(arm).unwrap().velocity = Vec3::new(1.5, 0.0, 0.0);
    // The pendulum oscillates; the swing EXTREMES are what must show pure
    // Z rotation, so track the maxima rather than the final frame's phase.
    let mut max_z_rot = 0.0f32;
    let mut max_tilt = 0.0f32;
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
        let q = physics.get_body(arm).unwrap().orientation;
        max_z_rot = max_z_rot.max(q.z.abs());
        max_tilt = max_tilt.max(q.x.abs()).max(q.y.abs());
    }
    // The arm swung about Z (the 1.5 m/s kick lifts it well past 5°)...
    assert!(max_z_rot > 0.05, "hinge barely rotated: {max_z_rot}");
    // ...but tilt about X and Y stays locked throughout the swing.
    assert!(max_tilt < 0.02, "hinge tilted off its axis: {max_tilt}");
    // Anchor stays coincident.
    let err = joint_anchor_error(&physics, anchor, arm, Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
    assert!(err < 0.05, "hinge anchor drifted: {err}");
}

/// Prismatic slider: block kicked along X on a horizontal rail under
/// gravity must travel freely along the axis while the point-to-line
/// constraints carry its weight (no sag) and lock the spin.
#[test]
fn prismatic_slider_travels_on_axis_without_sag() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let block = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(0.2, 0.2, 0.2),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            block,
            JointKind::Prismatic {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::X,
                local_axis_b: Vec3::X,
                limit: None,
                motor: None,
            },
        )
        .expect("valid joint");
    physics.get_body_mut(block).unwrap().velocity = Vec3::new(3.0, 0.0, 0.0);
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(block).unwrap();
    assert!(b.position.x > 1.0, "slider must travel, x={}", b.position.x);
    assert!(
        b.position.y.abs() < 0.05 && b.position.z.abs() < 0.05,
        "slider must not sag off axis: {:?}",
        b.position
    );
    let wx = b.orientation * Vec3::X;
    assert!(
        wx.dot(Vec3::X) > 0.995,
        "slider must not twist off axis: {wx:?}"
    );
}

/// Prismatic limit: a fast block stops at the window end and stays.
#[test]
fn prismatic_limit_blocks_travel_past_bounds() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let block = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(0.2, 0.2, 0.2),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            block,
            JointKind::Prismatic {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::X,
                local_axis_b: Vec3::X,
                limit: Some(PrismaticLimit {
                    min: -1.0,
                    max: 1.0,
                }),
                motor: None,
            },
        )
        .expect("valid joint");
    physics.get_body_mut(block).unwrap().velocity = Vec3::new(10.0, 0.0, 0.0);
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(block).unwrap();
    assert!(
        (b.position.x - 1.0).abs() < 0.3,
        "slider must stop at the upper bound, x={}",
        b.position.x
    );
    assert!(
        b.velocity.x.abs() < 1.0,
        "limit must kill the slide speed: {:?}",
        b.velocity
    );
}

/// Prismatic motor: a resting block spins up to the target slide speed.
#[test]
fn prismatic_motor_drives_to_target_speed() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let block = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(0.2, 0.2, 0.2),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            block,
            JointKind::Prismatic {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::X,
                local_axis_b: Vec3::X,
                limit: None,
                motor: Some(PrismaticMotor {
                    target_speed: 2.0,
                    max_force: 100.0,
                }),
            },
        )
        .expect("valid joint");
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(block).unwrap();
    assert!(
        (b.velocity.x - 2.0).abs() < 0.3,
        "motor must reach target speed: {:?}",
        b.velocity
    );
}

/// Prismatic assembly with a twisted block: the axis-alignment pass
/// pulls the slide axis back parallel.
#[test]
fn prismatic_misaligned_axes_realign() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let mut block = RigidBody::new_box(Vec3::ZERO, Vec3::new(0.2, 0.2, 0.2), 1.0);
    block.orientation = Quat::from_rotation_y(10.0f32.to_radians());
    let block = physics.add_body(block);
    physics
        .add_joint(
            anchor,
            block,
            JointKind::Prismatic {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::X,
                local_axis_b: Vec3::X,
                limit: None,
                motor: None,
            },
        )
        .expect("valid joint");
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(block).unwrap();
    let wx = b.orientation * Vec3::X;
    assert!(wx.dot(Vec3::X) > 0.998, "slide axes must realign: {wx:?}");
}

#[test]
fn revolute_limit_blocks_travel_past_bounds() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let arm = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.1, 1.0, 0.1),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            arm,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: Some(ornis_physics::joint::RevoluteLimit {
                    min: -0.2,
                    max: 0.2,
                }),
                motor: None,
            },
        )
        .expect("valid joint");
    // Hard kick: a free hinge would swing to ~1.4 rad.
    physics.get_body_mut(arm).unwrap().velocity = Vec3::new(4.0, 0.0, 0.0);
    let mut max_travel = 0.0f32;
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
        let a = physics.get_body(anchor).unwrap();
        let b = physics.get_body(arm).unwrap();
        let travel =
            ornis_physics::engine::joints::hinge_twist(a.orientation, b.orientation, Vec3::Z).abs();
        max_travel = max_travel.max(travel);
    }
    assert!(max_travel > 0.05, "arm never moved: {max_travel}");
    assert!(max_travel < 0.45, "limit failed to hold: {max_travel}");
}

#[test]
fn revolute_motor_spins_up_to_target_speed() {
    // Zero gravity: only the motor drives the hinge.
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let arm = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.1, 1.0, 0.1),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            arm,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: Some(ornis_physics::joint::RevoluteMotor {
                    target_speed: 3.0,
                    max_torque: 50.0,
                }),
            },
        )
        .expect("valid joint");
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let a = physics.get_body(anchor).unwrap();
    let b = physics.get_body(arm).unwrap();
    let w = (b.angular_velocity - a.angular_velocity).dot(Vec3::Z);
    assert!((w - 3.0).abs() < 0.3, "motor missed target speed: {w}");
    // A starved torque budget must NOT reach the target (clamp binds).
    let mut weak = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = weak.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let arm = weak.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.1, 1.0, 0.1),
        1.0,
    ));
    weak.add_joint(
        anchor,
        arm,
        JointKind::Revolute {
            local_anchor_a: Vec3::ZERO,
            local_anchor_b: Vec3::new(0.0, 1.0, 0.0),
            local_axis_a: Vec3::Z,
            local_axis_b: Vec3::Z,
            limit: None,
            motor: Some(ornis_physics::joint::RevoluteMotor {
                target_speed: 3.0,
                max_torque: 0.05,
            }),
        },
    )
    .expect("valid joint");
    for _ in 0..120 {
        weak.step(1.0 / 60.0);
    }
    let a = weak.get_body(anchor).unwrap();
    let b = weak.get_body(arm).unwrap();
    let w = (b.angular_velocity - a.angular_velocity).dot(Vec3::Z);
    assert!(w < 1.5, "torque clamp did not bind: {w}");
}

// ---- T13 regression: intermediate-value soundness of the 5 solver ----
// ---- primitives. These lock the *algebra*, not just finiteness, so ----
// ---- they catch op/sign mutants (e.g. `*`->`+`, `+=`->`-=`) that a ----
// ---- finite-only debug_assert cannot. ----

#[test]
fn mul_inv_inertia_matches_quat_application() {
    // I_world⁻¹ · v = R · I_body⁻¹ · Rᵀ · v, where I_body⁻¹ is diagonal.
    // We test the rotational part by checking the transformation is an
    // orientation application that the quaternion gives consistently.
    let inertia = Vec3::new(2.0, 4.0, 8.0);
    let ori = Quat::from_rotation_y(0.7);
    let v = Vec3::new(1.0, 2.0, -3.0);

    let got = mul_inv_inertia(inertia, ori, v);

    // Oracle: R · diag(1/inertia) · Rᵀ · v, built from glam matrices
    // (never touches mul_inv_inertia, so a mutant cannot pass it).
    let r = rot_mat(ori);
    let inv_diag = Vec3::new(
        inv_inertia_axis(inertia.x),
        inv_inertia_axis(inertia.y),
        inv_inertia_axis(inertia.z),
    );
    let body = r.transpose() * v;
    let scaled = Vec3::new(
        inv_diag.x * body.x,
        inv_diag.y * body.y,
        inv_diag.z * body.z,
    );
    let oracle = r * scaled;

    assert!(
        (got - oracle).length() < 1e-5,
        "mul_inv_inertia diverged from quaternion oracle: got {got:?}, oracle {oracle:?}"
    );
    // Sanity: a permutation of axes — same vector, different orientation,
    // must not all collapse to the input (catches `* -> +` on every axis).
    let ori2 = Quat::from_rotation_x(1.1);
    let got2 = mul_inv_inertia(inertia, ori2, v);
    assert!(
        (got - got2).length() > 1e-4,
        "orientation must change the result, got {got:?} vs {got2:?}"
    );
}

#[test]
fn effective_mass_matches_assembled_inverse_inertia() {
    // effective_mass(dir, ra) = 1/m + (ra×dir)·I_world⁻¹·(ra×dir).
    // Build two distinct bodies and check the assembled scalar matches the
    // matrix form: m⁻¹ + (ra×d)ᵀ · R·I⁻¹·Rᵀ · (ra×d).
    let mut a = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
    a.orientation = Quat::from_rotation_z(0.6);
    a.inertia = Vec3::new(3.0, 5.0, 7.0);
    a.inv_mass = 1.0 / a.mass;
    let mut b = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 2.0);
    b.orientation = Quat::from_rotation_x(-0.4);
    b.inertia = Vec3::new(2.0, 6.0, 4.0);
    b.inv_mass = 1.0 / b.mass;

    let bodies = [a, b];
    let dir = Vec3::new(0.0, 1.0, 0.0).normalize();
    let ra = Vec3::new(0.5, 0.0, 0.0);
    let rb = Vec3::new(-0.5, 0.0, 0.0);

    let em = effective_mass(&bodies, 0, 1, dir, ra, rb);

    // Oracle: 1/m_i + 1/m_j + (ra×d)ᵀ Iᵢ⁻¹ (ra×d) + (rb×d)ᵀ Iⱼ⁻¹ (rb×d).
    fn rot_inertia(ori: Quat, inv: Vec3) -> Mat3 {
        let r = rot_mat(ori);
        let diag = Mat3::from_diagonal(inv);
        r * diag * r.transpose()
    }
    let ra_d = ra.cross(dir);
    let rb_d = rb.cross(dir);
    let iw_i = rot_inertia(
        bodies[0].orientation,
        Vec3::new(
            inv_inertia_axis(bodies[0].inertia.x),
            inv_inertia_axis(bodies[0].inertia.y),
            inv_inertia_axis(bodies[0].inertia.z),
        ),
    );
    let iw_j = rot_inertia(
        bodies[1].orientation,
        Vec3::new(
            inv_inertia_axis(bodies[1].inertia.x),
            inv_inertia_axis(bodies[1].inertia.y),
            inv_inertia_axis(bodies[1].inertia.z),
        ),
    );
    let oracle =
        bodies[0].inv_mass + bodies[1].inv_mass + ra_d.dot(iw_i * ra_d) + rb_d.dot(iw_j * rb_d);

    assert!(
        (em - oracle).abs() < 1e-5,
        "effective_mass diverged from matrix oracle: got {em}, oracle {oracle}"
    );
    assert!(em > 0.0, "effective mass must be positive, got {em}");
}

#[test]
fn solve_small_matches_glam_lu() {
    // Independent oracle: solve A x = b with glam's matrix inverse and
    // check we recover `b` (A·x ≈ b) plus match glam's x. Any op/sign
    // mutant in the Gaussian elimination changes the recovered residual.
    let a = [
        [4.0, 1.0, 0.0, 0.0],
        [1.0, 3.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 5.0],
    ];
    let b = [1.0, 2.0, 3.0, 4.0];
    let n = 4;

    let x = solve_small(&a, &b, n).expect("well-conditioned system");

    // Reconstruct A·x via glam and confirm we recover b.
    let am = Mat4::from_cols_array(&[
        a[0][0], a[1][0], a[2][0], a[3][0], a[0][1], a[1][1], a[2][1], a[3][1], a[0][2], a[1][2],
        a[2][2], a[3][2], a[0][3], a[1][3], a[2][3], a[3][3],
    ]);
    let xv = glam::vec4(x[0], x[1], x[2], x[3]);
    let ax = am * xv;
    let residual = glam::Vec4::new(b[0], b[1], b[2], b[3]) - ax;
    assert!(
        residual.length() < 1e-3,
        "solve_small does not satisfy A x = b: residual {residual:?}"
    );

    // Cross-check against glam's own inverse solution.
    let inv = am.inverse();
    let oracle = inv * glam::vec4(b[0], b[1], b[2], b[3]);
    let diff = (glam::vec4(x[0], x[1], x[2], x[3]) - oracle).length();
    assert!(
        diff < 1e-3,
        "solve_small diverged from glam inverse: got {x:?}, oracle {oracle:?}"
    );
}

#[test]
fn solve_small_singular_returns_none() {
    // A singular (rank-deficient) matrix must be rejected, not produce a
    // finite-but-wrong answer or loop forever (the `* -> %`, `* -> /`
    // and `-= -> +=` mutants are caught here).
    let a = [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0], // row 2 = 2 * row 0 -> singular
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ];
    let b = [1.0, 2.0, 3.0, 4.0];
    let x = solve_small(&a, &b, 4);
    assert!(x.is_none(), "singular system must return None, got {x:?}");
}

#[test]
fn apply_impulse_is_symmetric_and_linear() {
    // Impulse j at contact point p between i and j must:
    //  - change v_i by -j/m_i and v_j by +j/m_j (linear term),
    //  - be antisymmetric: swapping (i,j) flips the velocity deltas,
    //  - preserve total (linear) momentum: m_i Δv_i + m_j Δv_j = 0.
    // Any `+= -> -=` / `-= -> +=` mutant breaks momentum/antisymmetry.
    let mut bodies = vec![
        RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0),
        RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0),
    ];
    bodies[0].velocity = Vec3::new(0.5, 0.0, 0.0);
    bodies[1].velocity = Vec3::new(-0.2, 0.0, 0.0);

    let imp = Vec3::new(0.0, 3.0, 0.0);
    let ra = Vec3::new(0.0, 1.0, 0.0);
    let rb = Vec3::new(0.0, -1.0, 0.0);

    let v0_i = bodies[0].velocity;
    let v0_j = bodies[1].velocity;
    apply_impulse(&mut bodies, 0, 1, imp, ra, rb);
    let dv_i = bodies[0].velocity - v0_i;
    let dv_j = bodies[1].velocity - v0_j;

    let expected_i = -imp * bodies[0].inv_mass;
    let expected_j = imp * bodies[1].inv_mass;
    assert!(
        (dv_i - expected_i).length() < 1e-5,
        "v_i delta wrong: got {dv_i:?}, expected {expected_i:?}"
    );
    assert!(
        (dv_j - expected_j).length() < 1e-5,
        "v_j delta wrong: got {dv_j:?}, expected {expected_j:?}"
    );

    // Momentum conservation (angular contributes via ang. momentum, but
    // the linear part alone must cancel exactly).
    let p_delta = bodies[0].mass * dv_i + bodies[1].mass * dv_j;
    assert!(
        p_delta.length() < 1e-5,
        "linear momentum not conserved: {p_delta:?}"
    );

    // Antisymmetry: for the SAME physical body, the velocity delta when it
    // plays role `i` must be the exact negative of its delta when it plays
    // role `j` (the impulse is antisymmetric under i<->j swap). This is
    // independent of the array slot, so it catches `+= <-> -=` mutants.
    let mut bodies2 = vec![
        RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0),
        RigidBody::new_sphere(Vec3::ZERO, 1.0, 2.0),
    ];
    bodies2[0].velocity = Vec3::new(0.5, 0.0, 0.0);
    bodies2[1].velocity = Vec3::new(-0.2, 0.0, 0.0);
    let v0b_0 = bodies2[0].velocity;
    let v0b_1 = bodies2[1].velocity;
    // body 0 is now role `j`, body 1 is role `i`.
    apply_impulse(&mut bodies2, 1, 0, imp, rb, ra);
    let dv_0_as_j = bodies2[0].velocity - v0b_0;
    let dv_1_as_i = bodies2[1].velocity - v0b_1;

    // body 0 as i (first call, dv_i) should oppose body 0 as j (dv_0_as_j).
    assert!(
        (dv_i + dv_0_as_j).length() < 1e-5,
        "body 0 i/j antisymmetry broken: as_i {dv_i:?} vs as_j {dv_0_as_j:?}"
    );
    // body 1 as j (first call, dv_j) should oppose body 1 as i (dv_1_as_i).
    assert!(
        (dv_j + dv_1_as_i).length() < 1e-5,
        "body 1 i/j antisymmetry broken: as_j {dv_j:?} vs as_i {dv_1_as_i:?}"
    );
}

#[test]
fn solve_normal_block_reduces_normal_velocity() {
    // Drive solve_normal_block on a 2-point manifold and assert the
    // complementarity result: the normal relative velocity at the active
    // points moves toward the target floor, and the committed state is
    // self-consistent (acc impulses ≥ 0 on the active set). The `- -> +`
    // / `* -> /` / `== -> !=` mutants in the block solver change this
    // outcome detectably.
    let mut bodies = vec![
        RigidBody::new_sphere(Vec3::ZERO, 1.0, 1.0),
        RigidBody::new_sphere(Vec3::new(0.0, -2.0, 0.0), 1.0, 1.0),
    ];
    // Body j approaches body i (moves up, +Y, into i which is above): its
    // normal relative velocity is negative, so solve_normal_block must
    // commit a positive separating impulse (acc > 0).
    bodies[1].velocity = Vec3::new(0.0, 5.0, 0.0);
    let n = Vec3::new(0.0, -1.0, 0.0); // i->j normal
    let pts = [
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.3, -1.0, 0.0),
        Vec3::ZERO,
        Vec3::ZERO,
    ];
    let mut acc = [0.0f32; 4];
    let target = [0.0f32, 0.0, 0.0, 0.0];
    let count = 2;

    // Normal relative velocity of body j minus body i at each point,
    // measured before solving.
    let vn_before: Vec<f32> = (0..count)
        .map(|k| {
            (point_velocity(&bodies[1], pts[k] - bodies[1].position)
                - point_velocity(&bodies[0], pts[k] - bodies[0].position))
            .dot(n)
        })
        .collect();

    solve_normal_block(&mut bodies, 0, 1, n, &pts, &mut acc, &target, count);

    let vn_after: Vec<f32> = (0..count)
        .map(|k| {
            (point_velocity(&bodies[1], pts[k] - bodies[1].position)
                - point_velocity(&bodies[0], pts[k] - bodies[0].position))
            .dot(n)
        })
        .collect();

    // Active-set impulses must be non-negative.
    for (k, impulse) in acc.iter().enumerate().take(count) {
        assert!(
            *impulse >= -1e-6,
            "accumulated impulse {} negative: {}",
            k,
            impulse
        );
    }
    // Each point's post-solve normal velocity must be at/above target (0),
    // i.e. separation or resting contact, not interpenetration growth.
    for k in 0..count {
        assert!(
            vn_after[k] >= target[k] - 1e-4,
            "point {k} normal velocity regressed below target: before {} after {}",
            vn_before[k],
            vn_after[k]
        );
        // The block solve must have done *something* (it found an active set).
        assert!(
            acc.iter().take(count).cloned().fold(0.0f32, f32::max) > 0.0,
            "solve_normal_block committed no impulse"
        );
    }
}

/// Bucket-sort regression: jointed pairs are class 0 (never visited) —
/// their slots must not leak into substep 0 as (0, 0) self-pairs (a
/// hard island-solver crash whenever body 0 is dynamic). Two jointed
/// dynamics in free fall step cleanly and report no contacts.
#[test]
fn jointed_pair_buckets_emit_no_self_pairs() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(1.2, 0.3, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    physics
        .add_joint(
            a,
            b,
            JointKind::Ball {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid joint");
    for _ in 0..10 {
        physics.step(1.0 / 60.0);
    }
    assert_eq!(physics.debug_contact_count(a), 0, "no self-manifolds");
    assert_eq!(physics.debug_contact_count(b), 0, "no self-manifolds");
}

/// Fixed weld: two boxes keep their assembly transform under gravity —
/// anchor coincidence and relative orientation both hold.
#[test]
fn fixed_weld_holds_assembly_pose() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let mut bb = RigidBody::new_box(Vec3::new(1.2, 0.3, 0.0), Vec3::splat(0.5), 1.0);
    bb.orientation = Quat::from_rotation_z(0.4);
    let b = physics.add_body(bb);
    // Coincident world anchors at assembly: the midpoint.
    let p = Vec3::new(0.6, 0.15, 0.0);
    let la = p - Vec3::ZERO;
    let lb = Quat::from_rotation_z(-0.4) * (p - Vec3::new(1.2, 0.3, 0.0));
    physics
        .add_joint(
            a,
            b,
            JointKind::Fixed {
                local_anchor_a: la,
                local_anchor_b: lb,
            },
        )
        .expect("valid joint");
    for _ in 0..180 {
        physics.step(1.0 / 60.0);
    }
    let (pa, pb) = (physics.get_body(a).unwrap(), physics.get_body(b).unwrap());
    let rel_pos = pb.position - pa.position;
    assert!(
        (rel_pos - Vec3::new(1.2, 0.3, 0.0)).length() < 0.05,
        "weld must hold anchor offset, got {rel_pos:?}"
    );
    let rel = pa.orientation.conjugate() * pb.orientation;
    let angle = 2.0 * rel.w.clamp(-1.0, 1.0).acos();
    assert!(
        (angle - 0.4).abs() < 0.05,
        "weld must hold relative rotation, got {angle}"
    );
}

/// Distance rod: a pendulum keeps its anchor separation under gravity
/// and swings through the bottom instead of stretching or freezing.
#[test]
fn distance_rod_keeps_anchor_separation() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.1), 0.0));
    let ball = physics.add_body(RigidBody::new_sphere(Vec3::new(2.0, 0.0, 0.0), 0.3, 1.0));
    physics
        .add_joint(
            anchor,
            ball,
            JointKind::Distance {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid joint");
    // Undamped 2 m pendulum, T ≈ 2.84 s: after 0.75 s it hangs near the
    // bottom; the rod length must hold the whole way.
    let mut worst = 0.0f32;
    for s in 0..45 {
        physics.step(1.0 / 60.0);
        let (pa, pb) = (
            physics.get_body(anchor).unwrap(),
            physics.get_body(ball).unwrap(),
        );
        worst = worst.max(((pb.position - pa.position).length() - 2.0).abs());
        let _ = s;
    }
    let pb = physics.get_body(ball).unwrap();
    assert!(worst < 0.08, "rod must keep 2 m separation, drift {worst}");
    assert!(
        pb.position.y < -1.0,
        "pendulum must swing down, got {:?}",
        pb.position
    );
}

/// Wheel suspension: a sprung chassis settles near its rest length
/// under gravity instead of collapsing onto the wheel.
#[test]
fn wheel_suspension_holds_chassis_height() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let chassis = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.5, 0.2, 0.3),
        2.0,
    ));
    let wheel = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.25, 0.25, 0.15),
        1.0,
    ));
    physics
        .add_joint(
            chassis,
            wheel,
            JointKind::Wheel {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_suspension_a: Vec3::Y,
                local_suspension_b: Vec3::Y,
                local_axle_a: Vec3::Z,
                local_axle_b: Vec3::Z,
                suspension: WheelSuspension {
                    frequency_hz: 3.0,
                    damping_ratio: 0.7,
                },
                motor: None,
            },
        )
        .expect("valid joint");
    for _ in 0..300 {
        physics.step(1.0 / 60.0);
    }
    let (pc, pw) = (
        physics.get_body(chassis).unwrap(),
        physics.get_body(wheel).unwrap(),
    );
    let sep = pc.position.y - pw.position.y;
    assert!(
        (0.5..=1.1).contains(&sep),
        "spring must hold the chassis near rest (1.0 m), got {sep}"
    );
}

/// Wheel motor: a free wheel spins up about its axle toward the target
/// speed in zero gravity.
#[test]
fn wheel_motor_spins_axle_to_target() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let anchor = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.2), 0.0));
    let wheel = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(0.2, 0.2, 0.1),
        1.0,
    ));
    physics
        .add_joint(
            anchor,
            wheel,
            JointKind::Wheel {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_suspension_a: Vec3::Y,
                local_suspension_b: Vec3::Y,
                local_axle_a: Vec3::Z,
                local_axle_b: Vec3::Z,
                suspension: WheelSuspension {
                    frequency_hz: 2.0,
                    damping_ratio: 0.5,
                },
                motor: Some(RevoluteMotor {
                    target_speed: 6.0,
                    max_torque: 50.0,
                }),
            },
        )
        .expect("valid joint");
    for _ in 0..240 {
        physics.step(1.0 / 60.0);
    }
    let w = physics.get_body(wheel).unwrap().angular_velocity;
    assert!(
        (w.z - 6.0).abs() < 1.5,
        "axle must spin up toward 6 rad/s, got {w:?}"
    );
}

/// Gear ratio: a motor on hinge A drives hinge B at -1/ratio speed
/// (`coord_a + ratio * coord_b = const`).
#[test]
fn gear_ratio_couples_hinges() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let ground = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.2), 0.0));
    let arm_a = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.6, 0.0),
        Vec3::new(0.1, 0.6, 0.1),
        1.0,
    ));
    let arm_b = physics.add_body(RigidBody::new_box(
        Vec3::new(1.0, 0.6, 0.0),
        Vec3::new(0.1, 0.6, 0.1),
        1.0,
    ));
    let ja = physics
        .add_joint(
            ground,
            arm_a,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::new(0.0, -0.6, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: Some(RevoluteMotor {
                    target_speed: 1.5,
                    max_torque: 20.0,
                }),
            },
        )
        .expect("valid joint");
    let jb = physics
        .add_joint(
            ground,
            arm_b,
            JointKind::Revolute {
                local_anchor_a: Vec3::new(1.0, 0.0, 0.0),
                local_anchor_b: Vec3::new(0.0, -0.6, 0.0),
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: None,
            },
        )
        .expect("valid joint");
    physics
        .add_joint(
            arm_a,
            arm_b,
            JointKind::Gear {
                joint_a: ja,
                joint_b: jb,
                ratio: 2.0,
            },
        )
        .expect("valid gear");
    let (mut ta, mut tb) = (0.0, 0.0);
    let (mut old_a, mut old_b) = (0.0, 0.0);
    for _ in 0..180 {
        physics.step(1.0 / 60.0);
        let qa = physics.get_body(arm_a).unwrap().orientation;
        let qb = physics.get_body(arm_b).unwrap().orientation;
        let raw_a = ornis_physics::engine::joints::hinge_twist(Quat::IDENTITY, qa, Vec3::Z);
        let raw_b = ornis_physics::engine::joints::hinge_twist(Quat::IDENTITY, qb, Vec3::Z);
        ta = ornis_physics::migration::gear_coordinate(
            raw_a,
            ornis_physics::CoordKind::Angular,
            Some((old_a, ta)),
        );
        tb = ornis_physics::migration::gear_coordinate(
            raw_b,
            ornis_physics::CoordKind::Angular,
            Some((old_b, tb)),
        );
        old_a = raw_a;
        old_b = raw_b;
    }
    assert!(ta.abs() > 0.5, "motor must turn hinge A, got twist {ta}");
    // coord_a + 2 * coord_b = 0 (assembly constant) within solver drift.
    let c = (ta + 2.0 * tb).abs();
    assert!(
        c < 0.35 * ta.abs().max(1.0),
        "gear must hold a + 2b = 0, got a={ta} b={tb}"
    );
}

/// Gear validation: dangling references and non-hinge joints are
/// rejected, and removing a referenced joint drops the gear silently.
#[test]
fn gear_validation_and_cleanup() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b = physics.add_body(RigidBody::new_box(Vec3::X, Vec3::splat(0.5), 1.0));
    assert!(
        physics
            .add_joint(
                a,
                b,
                JointKind::Gear {
                    joint_a: JointHandle::from_raw(7),
                    joint_b: JointHandle::from_raw(9),
                    ratio: 1.0,
                },
            )
            .is_err(),
        "dangling gear refs must be rejected"
    );
    let ball = physics
        .add_joint(
            a,
            b,
            JointKind::Ball {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
            },
        )
        .expect("valid joint");
    assert!(
        physics
            .add_joint(
                a,
                b,
                JointKind::Gear {
                    joint_a: ball,
                    joint_b: ball,
                    ratio: 1.0,
                },
            )
            .is_err(),
        "gear over non-hinge joints must be rejected"
    );
    let hinge = physics
        .add_joint(
            a,
            b,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: None,
            },
        )
        .expect("valid joint");
    assert!(
        physics
            .add_joint(
                a,
                b,
                JointKind::Gear {
                    joint_a: ball,
                    joint_b: hinge,
                    ratio: 1.0,
                },
            )
            .is_err(),
        "gear over a non-hinge joint must be rejected"
    );
    let hinge2 = physics
        .add_joint(
            a,
            b,
            JointKind::Revolute {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_axis_a: Vec3::Z,
                local_axis_b: Vec3::Z,
                limit: None,
                motor: None,
            },
        )
        .expect("valid joint");
    let before = physics.joint_count();
    let gear = physics
        .add_joint(
            a,
            b,
            JointKind::Gear {
                joint_a: hinge,
                joint_b: hinge2,
                ratio: 1.0,
            },
        )
        .expect("valid gear");
    assert_eq!(physics.joint_count(), before + 1);
    physics.remove_joint(hinge);
    // The hinge plus its dependent gear are gone; the other joints stay.
    assert_eq!(physics.joint_count(), before - 1);
    let _ = gear;
    for _ in 0..10 {
        physics.step(1.0 / 60.0);
    }
}

/// Six-DOF all-locked degenerates to a weld: assembly pose holds.
#[test]
fn sixdof_all_locked_behaves_like_fixed() {
    let locked = [AxisConfig::Locked; 3];
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.4, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    physics
        .add_joint(
            a,
            b,
            JointKind::SixDof {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                linear: locked,
                angular: locked,
            },
        )
        .expect("valid joint");
    for _ in 0..180 {
        physics.step(1.0 / 60.0);
    }
    let (pa, pb) = (physics.get_body(a).unwrap(), physics.get_body(b).unwrap());
    let gap = (pb.position - pa.position).y;
    assert!(
        (gap - 1.4).abs() < 0.05,
        "all-locked six-DOF must weld, got gap {gap}"
    );
}

/// Six-DOF free axis: X slides under impulse while locked Y holds.
#[test]
fn sixdof_free_axis_slides_but_locked_holds() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b = physics.add_body(RigidBody::new_box(Vec3::X, Vec3::splat(0.5), 1.0));
    physics
        .add_joint(
            a,
            b,
            JointKind::SixDof {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                linear: [AxisConfig::Free, AxisConfig::Locked, AxisConfig::Locked],
                angular: [AxisConfig::Locked; 3],
            },
        )
        .expect("valid joint");
    physics.get_body_mut(b).unwrap().velocity = Vec3::new(3.0, 1.0, 0.0);
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }
    let (pa, pb) = (physics.get_body(a).unwrap(), physics.get_body(b).unwrap());
    let d = pb.position - pa.position;
    assert!(d.x > 1.5, "free X must slide, got {d:?}");
    assert!(d.y.abs() < 0.08, "locked Y must hold, got {d:?}");
}

/// Six-DOF angular limit: a fast spin about Z clamps at the window.
#[test]
fn sixdof_angular_limit_blocks_spin() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0));
    let b = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.4), 1.0));
    physics
        .add_joint(
            a,
            b,
            JointKind::SixDof {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                linear: [AxisConfig::Locked; 3],
                angular: [
                    AxisConfig::Locked,
                    AxisConfig::Locked,
                    AxisConfig::Limited {
                        min: -0.2,
                        max: 0.2,
                    },
                ],
            },
        )
        .expect("valid joint");
    physics.get_body_mut(b).unwrap().angular_velocity = Vec3::new(0.0, 0.0, 8.0);
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let qb = physics.get_body(b).unwrap().orientation;
    let tw = ornis_physics::engine::joints::hinge_twist(Quat::IDENTITY, qb, Vec3::Z);
    assert!(
        tw.abs() < 0.4,
        "Z twist must clamp near the 0.2 window, got {tw}"
    );
}

/// Wheel with a parallel axle still assembles (deterministic fallback)
/// and steps without NaN.
#[test]
fn wheel_degenerate_axle_falls_back() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    physics
        .add_joint(
            a,
            b,
            JointKind::Wheel {
                local_anchor_a: Vec3::ZERO,
                local_anchor_b: Vec3::ZERO,
                local_suspension_a: Vec3::Y,
                local_suspension_b: Vec3::Y,
                // Parallel to the suspension: must fall back, not NaN.
                local_axle_a: Vec3::Y,
                local_axle_b: Vec3::Y,
                suspension: WheelSuspension {
                    frequency_hz: 2.0,
                    damping_ratio: 0.5,
                },
                motor: None,
            },
        )
        .expect("valid joint");
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }
    for h in [a, b] {
        let body = physics.get_body(h).unwrap();
        assert!(
            body.position.is_finite() && body.velocity.is_finite(),
            "no NaN after degenerate assembly"
        );
    }
}
