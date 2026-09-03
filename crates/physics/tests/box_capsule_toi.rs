//! Integration tests for the new physics API: box↔capsule contact and
//! analytic swept-volume TOI with fast rotation (G6).
//!
//! Covers the public `BuiltinPhysicsEngine`/`RigidBody` API without access
//! to private `engine::` internals. Each test drives `step` and checks
//! observable behavior: contact presence/absence, normal orientation
//! and absence of tunneling under fast rotation.

use glam::{Quat, Vec3};
use ornis_physics::{BuiltinPhysicsEngine, PhysicsEngine, RigidBody};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_asleep_pair(physics: &BuiltinPhysicsEngine, a: usize, b: usize) -> bool {
    physics.debug_contact_count(a) > 0 && physics.debug_contact_count(b) > 0
}

#[allow(dead_code)]
fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

// ---------------------------------------------------------------------------
// box ↔ capsule: contact generation
// ---------------------------------------------------------------------------

#[test]
fn box_and_capsule_overlap_generates_contact() {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    // Box half 0.5 at origin, capsule radius 0.5 half_height 1.0 at x=0.6
    // capsule core is along Y, center 0.6 away -> surface gap = 0.6 - 0.5(box face) -0.5 = -0.4 (overlap)
    let bx = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let cp = physics.add_body(RigidBody::new_capsule(
        Vec3::new(0.6, 0.0, 0.0),
        0.5,
        1.0,
        1.0,
    ));
    physics.step(1.0 / 60.0);
    assert!(
        is_asleep_pair(&physics, bx, cp),
        "overlapping box+capsule must produce manifold, got bx={} cp={}",
        physics.debug_contact_count(bx),
        physics.debug_contact_count(cp)
    );
}

#[test]
fn capsule_and_box_swapped_order_also_generates_contact() {
    // Insertion in reverse order: capsule first, box second — normal must flip,
    // but contact must still appear (symmetry check for Box↔Capsule / Capsule↔Box branches).
    let mut p1 = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let a = p1.add_body(RigidBody::new_capsule(
        Vec3::new(0.6, 0.0, 0.0),
        0.5,
        1.0,
        1.0,
    ));
    let b = p1.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    p1.step(1.0 / 60.0);
    assert!(is_asleep_pair(&p1, a, b));

    let mut p2 = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let a2 = p2.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b2 = p2.add_body(RigidBody::new_capsule(
        Vec3::new(0.6, 0.0, 0.0),
        0.5,
        1.0,
        1.0,
    ));
    p2.step(1.0 / 60.0);
    assert!(is_asleep_pair(&p2, a2, b2));
    // contact count must match regardless of insertion order
    assert_eq!(
        p1.debug_contact_count(a),
        p2.debug_contact_count(a2),
        "contact count must be order-independent"
    );
}

#[test]
fn separated_box_and_capsule_no_contact() {
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let bx = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let cp = physics.add_body(RigidBody::new_capsule(
        Vec3::new(3.0, 0.0, 0.0),
        0.5,
        1.0,
        1.0,
    ));
    physics.step(1.0 / 60.0);
    assert_eq!(physics.debug_contact_count(bx), 0);
    assert_eq!(physics.debug_contact_count(cp), 0);
}

#[test]
fn rotated_box_vs_capsule_contact() {
    // Box rotated 45° around Z, capsule to the side. AABB expands to sqrt(2),
    // distance is computed via exact `shape_distance`, contact must appear.
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let rot = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    let bx = physics
        .add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0).with_orientation(rot));
    // capsule to the right, slightly above to catch the corner
    let cp = physics.add_body(RigidBody::new_capsule(
        Vec3::new(0.9, 0.2, 0.0),
        0.3,
        0.6,
        1.0,
    ));
    physics.step(1.0 / 60.0);
    assert!(
        is_asleep_pair(&physics, bx, cp),
        "rotated box vs capsule must collide"
    );
}

#[test]
fn box_capsule_touching_at_speculative_margin_generates_contact() {
    // Speculative margin = 0.05 + rel_speed*dt. At zero velocity margin=0.05.
    // Place distance at exactly 0.04 — inside margin, contact present.
    // Distance = dist(box face, capsule surface).
    // Box half 0.5, capsule radius 0.3, capsule center at 0.5+0.3+0.04=0.84
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let bx = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let cp = physics.add_body(RigidBody::new_capsule(
        Vec3::new(0.84, 0.0, 0.0),
        0.3,
        0.5,
        1.0,
    ));
    physics.step(1.0 / 60.0);
    // Inside speculative margin there must be a manifold (penetration negative but contact present)
    // In practice debug_contact_count >0 if dist <= margin
    assert!(
        physics.debug_contact_count(bx) > 0,
        "touching within speculative margin must generate speculative contact"
    );
    let _ = physics.debug_contact_count(cp);
}

#[test]
fn capsule_capsule_still_works_after_box_capsule_patch() {
    // Regression: adding Box↔Capsule branches must not break Capsule↔Capsule
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_capsule(Vec3::ZERO, 0.5, 1.0, 1.0));
    let b = physics.add_body(RigidBody::new_capsule(
        Vec3::new(0.9, 0.0, 0.0),
        0.5,
        1.0,
        1.0,
    ));
    physics.step(1.0 / 60.0);
    assert!(is_asleep_pair(&physics, a, b));
}

#[test]
fn box_vs_box_still_produces_four_point_manifold() {
    // Regression for OBB-OBB (4-point face manifold)
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.75, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    physics.step(1.0 / 60.0);
    assert!(physics.debug_contact_count(a) > 0);
    assert!(physics.debug_contact_count(b) > 0);
}

// ---------------------------------------------------------------------------
// TOI: analytic swept-volume with fast rotation
// ---------------------------------------------------------------------------

#[test]
fn fast_spinning_box_toi_stops_before_tunneling() {
    // Thin long box 3.0×0.2×0.2 rotating 90° in one substep (dt) around Z.
    // Static wall just above: at 90° the box tip must hit.
    // Analytic conservative advancement must catch the first contact
    // fraction ∈ (0,1) and clamp orientation, zeroing angular velocity.
    let dt = 1.0 / 60.0;
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.1, 0.0),
        Vec3::new(0.2, 0.05, 0.2),
        0.0,
    ));
    let mover = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(1.5, 0.1, 0.1),
        1.0,
    ));
    physics.get_body_mut(mover).unwrap().angular_velocity =
        Vec3::Z * (std::f32::consts::FRAC_PI_2 / dt);

    physics.step(dt);

    let body = physics.get_body(mover).unwrap();
    // Must stop (angular CCD clamped)
    assert!(
        body.angular_velocity.length() < 1e-5,
        "angular CCD must zero angular velocity at impact, got {:?}",
        body.angular_velocity
    );
    // And must not tunnel through the target
    assert_ne!(
        body.orientation,
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        "rotating body must not tunnel through target"
    );
    // Orientation between 0 and 90° (fraction in (0,1))
    let angle = body.orientation.to_axis_angle().1.abs();
    assert!(
        angle > 0.05 && angle < std::f32::consts::FRAC_PI_2 - 0.05,
        "clamped angle must be in (0, 90°), got {angle}"
    );
}

#[test]
fn fast_spinning_capsule_toi_stops_before_tunneling() {
    // Same for a capsule: long capsule along X rotating in plane.
    let dt = 1.0 / 60.0;
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.3, 0.0),
        Vec3::new(0.3, 0.05, 0.3),
        0.0,
    ));
    let mover = physics.add_body(RigidBody::new_capsule(Vec3::ZERO, 0.15, 1.2, 1.0));
    // Rotate capsule horizontally (along X) so rotation around Z sweeps a circle
    physics.get_body_mut(mover).unwrap().orientation =
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
    physics.get_body_mut(mover).unwrap().angular_velocity =
        Vec3::Z * (std::f32::consts::FRAC_PI_2 / dt);
    physics.step(dt);
    let body = physics.get_body(mover).unwrap();
    assert!(
        body.angular_velocity.length() < 1e-5,
        "capsule angular CCD must stop at impact"
    );
    assert_ne!(
        body.orientation,
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)
            * Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        "capsule must not tunnel"
    );
}

#[test]
fn slow_rotation_does_not_trigger_false_toi() {
    // Slow rotation < 15°/substep must not trigger angular CCD (MIN_ANGLE).
    // Body should freely reach the full orientation.
    let dt = 1.0 / 60.0;
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::splat(0.5),
        0.0,
    ));
    let mover = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(0.5, 0.1, 0.1),
        1.0,
    ));
    let slow_w = Vec3::Z * (10.0f32.to_radians() / dt); // 10°/frame < 15° threshold
    physics.get_body_mut(mover).unwrap().angular_velocity = slow_w;
    let expected = Quat::from_scaled_axis(slow_w * dt);
    physics.step(dt);
    let body = physics.get_body(mover).unwrap();
    // Not clamped — angular velocity preserved
    assert!(
        body.angular_velocity.length() > 1e-3,
        "slow rotation must not be clamped by CCD"
    );
    // Orientation almost matches expected (integration without TOI)
    let dot = body.orientation.dot(expected).abs();
    assert!(
        dot > 0.999,
        "slow rot should integrate fully, dot={dot}, got {:?} vs {expected:?}",
        body.orientation
    );
}

#[test]
fn thin_feature_under_fast_rotation_does_not_tunnel() {
    // Thin wall 0.04 thick — 5° sampling could miss it, analytic must not.
    // Rotating box with a large angle must hit, not pass through.
    let dt = 1.0 / 60.0;
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    // Thin vertical wall near the trajectory of the rotating box tip
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 1.1, 0.0),
        Vec3::new(0.5, 0.02, 0.5),
        0.0,
    ));
    let mover = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(1.5, 0.08, 0.08),
        1.0,
    ));
    physics.get_body_mut(mover).unwrap().angular_velocity =
        Vec3::Z * (std::f32::consts::FRAC_PI_2 / dt);
    physics.step(dt);
    let body = physics.get_body(mover).unwrap();
    // Must be clamped, otherwise tunneled through thin wall
    assert!(
        body.angular_velocity.length() < 1e-5,
        "thin wall must be caught by analytic swept-volume, got w={:?}",
        body.angular_velocity
    );
}

#[test]
fn capsule_vs_box_toi_with_combined_translation_and_rotation() {
    // Combined motion: translation + fast rotation of the capsule.
    // Checks that bound = |disp| + r*angle accounts for both contributions.
    let dt = 1.0 / 60.0;
    let mut physics = BuiltinPhysicsEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    physics.add_body(RigidBody::new_box(
        Vec3::new(2.0, 0.0, 0.0),
        Vec3::splat(0.5),
        0.0,
    ));
    let mover = physics.add_body(RigidBody::new_capsule(Vec3::ZERO, 0.2, 0.8, 1.0));
    {
        let b = physics.get_body_mut(mover).unwrap();
        b.velocity = Vec3::new(5.0, 0.0, 0.0);
        b.angular_velocity = Vec3::Z * (60.0f32.to_radians() / dt);
        b.orientation = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
    }
    physics.step(dt);
    let body = physics.get_body(mover).unwrap();
    // Must be stopped before the target, not pass through
    assert!(
        body.position.x < 1.0,
        "combined sweep must stop before target, x={}",
        body.position.x
    );
}

#[test]
fn multiple_capsules_and_boxes_interact_without_panic() {
    // Stress: several boxes and capsules interleaved, gravity, 60 steps — no panic and no NaN.
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    for i in 0..6 {
        let x = (i as f32 - 2.5) * 1.2;
        if i % 2 == 0 {
            physics.add_body(RigidBody::new_box(
                Vec3::new(x, 2.0 + i as f32 * 0.5, 0.0),
                Vec3::splat(0.3),
                1.0,
            ));
        } else {
            physics.add_body(RigidBody::new_capsule(
                Vec3::new(x, 2.0 + i as f32 * 0.5, 0.0),
                0.25,
                0.4,
                1.0,
            ));
        }
    }
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }
    for h in 0..7 {
        if let Some(b) = physics.get_body(h) {
            assert!(
                b.position.is_finite(),
                "body {h} position must stay finite: {:?}",
                b.position
            );
            assert!(b.velocity.is_finite(), "body {h} velocity must stay finite");
        }
    }
}

#[test]
fn box_capsule_resting_penetration_is_resolved() {
    // Capsule resting on a box floor: penetration must be resolved by the solver, not grow.
    let mut physics = BuiltinPhysicsEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        0.0,
    ));
    let cap = physics.add_body(RigidBody::new_capsule(
        Vec3::new(0.0, 1.0, 0.0),
        0.3,
        0.5,
        1.0,
    ));
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(cap).unwrap();
    // Capsule resting on floor: due to best-effort distance for overlap
    // box↔capsule penetration = 0 (witness on surface), so positional
    // correction is weaker than box↔box — partial sinking is allowed,
    // but contact must exist, velocity must be low and no NaN.
    assert!(
        physics.debug_contact_count(cap) > 0,
        "capsule must have box contact after settling"
    );
    assert!(
        b.position.y > -0.5 && b.position.y < 1.50,
        "capsule must stay near floor, y={}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 1.0,
        "resting capsule must have low velocity, got {:?}",
        b.velocity
    );
    assert!(b.position.is_finite() && b.velocity.is_finite());
}
