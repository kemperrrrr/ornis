//! P4 rotational CCD + bullet mode (Rapier `dynamics/ccd` parity).
//!
//! * Nonlinear (rotating) sweep catches what the frozen-orientation linear
//!   cast misses: high angular velocity with near-zero linear travel.
//! * Bullets (`RigidBody::ccd_enabled`) always sweep the nonlinear path,
//!   bypassing the travel gate; non-bullet bodies keep the legacy policy
//!   (linear sweep plus the gated angular sweep).
//! * `max_ccd_substeps` caps the sweep iterations with an explicit
//!   best-effort fallback clamp, counted via `last_ccd_caps` (the
//!   `last_substep_shed` observability pattern).

use glam::Vec3;
use ornis_physics::engine::find_angular_continuous_hit;
use ornis_physics::{PhysicsEngine, RigidBody, SequentialImpulseEngine};

/// Spin kinetic energy from public body state: E = ½ΣIᵢωᵢ² in the body
/// frame. The angular CCD energy cap (`cap_spin_correction`) guarantees
/// this never grows across the impact response.
fn spin_energy(body: &RigidBody) -> f32 {
    let wb = body.orientation.conjugate() * body.angular_velocity;
    0.5 * (body.inertia * wb).dot(wb)
}

/// Absolute rotation angle carried by a body orientation.
fn angle_of(body: &RigidBody) -> f32 {
    body.orientation.to_axis_angle().1.abs()
}

/// High-spin scene: long thin box at the origin spinning about Z under a
/// thin static wall (thickness 0.04 m). Linear velocity is zero, so the
/// linear travel gate skips the frozen-orientation cast by construction —
/// any clamp must come from the nonlinear (rotating) sweep.
fn high_spin_scene() -> (SequentialImpulseEngine, ornis_physics::BodyHandle, f32) {
    let dt = 1.0 / 60.0;
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    // Thin wall 0.04 m thick, bottom face at y = 0.08, spanning x∈[0, 1.2].
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.6, 0.10, 0.0),
        Vec3::new(0.6, 0.02, 0.6),
        0.0,
    ));
    let mover = physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(1.0, 0.05, 0.05),
        1.0,
    ));
    // 65 rad/s: the tip sweeps 1.08 rad (62°) per step, far past the wall.
    physics.get_body_mut(mover).unwrap().angular_velocity = Vec3::Z * 65.0;
    (physics, mover, dt)
}

#[test]
fn spinning_box_65rad_s_in_thin_wall_does_not_tunnel() {
    // Nonlinear sweep catches what the linear cast misses: zero linear
    // travel (linear gate skips), 65 rad/s spin into a 0.04 m wall.
    let (mut physics, mover, dt) = high_spin_scene();
    let full_angle = 65.0 * dt;
    let energy_before = spin_energy(physics.get_body(mover).unwrap());

    physics.step(dt);

    let body = physics.get_body(mover).unwrap();
    assert!(body.position.is_finite() && body.velocity.is_finite());
    // Clamped strictly before the full 62° rotation: no tunneling.
    let angle = angle_of(body);
    assert!(
        angle < full_angle - 0.05,
        "nonlinear sweep must catch the wall, angle={angle} full={full_angle}"
    );
    // Energy cap: the CCD response must never inject spin energy.
    let energy_after = spin_energy(body);
    assert!(
        energy_after <= energy_before * 1.01,
        "CCD must not inject spin energy: before={energy_before} after={energy_after}"
    );
}

/// Small-rotation scene for the bullet/non-bullet comparison: a chunky cube
/// (half 0.5) spinning 15°/step sits BELOW the angular travel gate
/// (`r*angle = 0.227 ≤ 0.5*0.5 = 0.25`), with a wall 0.06 m above the
/// corner path — outside the speculative margin, so the discrete phase
/// cannot mask the sweep-level difference in one step.
fn gated_spin_scene(bullet: bool) -> (SequentialImpulseEngine, ornis_physics::BodyHandle, f32) {
    let dt = 1.0 / 60.0;
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    // Wall bottom face at y = 0.56, spanning x∈[0.2, 1.2]: the rotating
    // corner (start y = 0.5) reaches 0.612 at 15°, so the sweep must fire.
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.7, 0.58, 0.0),
        Vec3::new(0.5, 0.02, 0.5),
        0.0,
    ));
    let mut mover_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
    mover_body.set_ccd_enabled(bullet);
    let mover = physics.add_body(mover_body);
    // 15°/step: below the gate for non-bullet bodies.
    let w = Vec3::Z * (15.0f32.to_radians() / dt);
    physics.get_body_mut(mover).unwrap().angular_velocity = w;
    (physics, mover, dt)
}

#[test]
fn bullet_sweeps_below_the_gate_non_bullet_skips() {
    // Query level: identical bodies, identical motion — only the bullet
    // flag differs. Non-bullet is gated out, bullet sweeps and brackets a
    // TOI fraction in (0, 1).
    let (physics_plain, mover_plain, dt) = gated_spin_scene(false);
    let bodies = &physics_plain.bodies;
    assert!(
        find_angular_continuous_hit(bodies, mover_plain.index(), Vec3::ZERO, dt).is_none(),
        "non-bullet spin below the travel gate must skip the sweep"
    );

    let (physics_bullet, mover_bullet, _) = gated_spin_scene(true);
    let hit =
        find_angular_continuous_hit(&physics_bullet.bodies, mover_bullet.index(), Vec3::ZERO, dt)
            .expect("bullet must sweep regardless of the travel gate");
    assert!(
        hit.fraction > 0.0 && hit.fraction < 1.0,
        "bullet TOI must bracket the wall, fraction={}",
        hit.fraction
    );
}

#[test]
fn bullet_clamps_non_bullet_integrates_through() {
    // Step level on the same scene: the bullet clamps at the TOI with the
    // CCD velocity response, while the non-bullet body integrates the full
    // 15° (its overlap is the discrete solver's job on later steps).
    // Documented difference, not a tunneling verdict on either path.
    let full = 15.0f32.to_radians();

    let (mut bullet_physics, bullet_mover, dt) = gated_spin_scene(true);
    bullet_physics.step(dt);
    let bullet_angle = angle_of(bullet_physics.get_body(bullet_mover).unwrap());
    assert!(
        bullet_angle < full - 0.05,
        "bullet must clamp at the TOI, angle={bullet_angle} full={full}"
    );
    assert_eq!(
        bullet_physics.last_ccd_caps(),
        0,
        "a proven TOI is not a cap fallback"
    );

    let (mut plain_physics, plain_mover, _) = gated_spin_scene(false);
    plain_physics.step(dt);
    let plain_angle = angle_of(plain_physics.get_body(plain_mover).unwrap());
    assert!(
        plain_angle > full - 0.03,
        "non-bullet must integrate (nearly) the full gated-out rotation, angle={plain_angle} full={full}"
    );
    assert_eq!(plain_physics.last_ccd_caps(), 0);
}

#[test]
fn ccd_substep_cap_fires_without_panic() {
    // A starved budget (1 iteration) cannot finish the high-spin sweep:
    // the step still completes, the best-effort clamp stops the body early
    // (safe side of the true TOI — no tunnel), and the cap is counted.
    let (mut physics, mover, dt) = high_spin_scene();
    physics.set_max_ccd_substeps(1);
    physics.step(dt);
    assert!(
        physics.last_ccd_caps() > 0,
        "starved sweep budget must be observed"
    );
    let body = physics.get_body(mover).unwrap();
    assert!(body.position.is_finite() && body.velocity.is_finite());
    assert!(
        angle_of(body) < 65.0 * dt - 0.05,
        "cap fallback must still stop before the wall"
    );
}

#[test]
fn zero_ccd_budget_disables_the_angular_sweep() {
    // `max_ccd_substeps = 0` is the explicit off switch (Rapier parity):
    // no sweep runs, nothing is counted, the body integrates freely.
    let (mut physics, mover, dt) = gated_spin_scene(true);
    physics.set_max_ccd_substeps(0);
    physics.step(dt);
    assert_eq!(physics.last_ccd_caps(), 0);
    let full = 15.0f32.to_radians();
    let angle = angle_of(physics.get_body(mover).unwrap());
    assert!(
        angle > full - 0.03,
        "disabled sweep must not clamp, angle={angle} full={full}"
    );
}

#[test]
fn ccd_flag_and_budget_accessors_behave() {
    let body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
    assert!(!body.ccd_enabled);
    assert!(!body.is_bullet());
    let body = body.with_ccd_enabled(true);
    assert!(body.ccd_enabled && body.is_bullet());
    // Static bodies are never bullets (Rapier `is_bullet` parity).
    let mut static_body = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 0.0);
    static_body.set_ccd_enabled(true);
    assert!(!static_body.is_bullet());

    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    assert_eq!(physics.max_ccd_substeps(), 32);
    assert_eq!(physics.last_ccd_caps(), 0);
    physics.set_max_ccd_substeps(4);
    assert_eq!(physics.max_ccd_substeps(), 4);
}
