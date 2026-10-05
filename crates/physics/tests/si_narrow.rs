//! SI narrowphase pins: manifolds, CCD motion and shape contact.

use glam::{Quat, Vec3};
use ornis_physics::engine::{
    Manifold, NarrowShardPool, SatCache, box_manifold, ccd_impact_velocity, detect_collisions_into,
    find_angular_continuous_hit, kinematic_cast, obb_sat, remove_angular_approach, sweep_gap,
};
use ornis_physics::{PhysicsEngine, RigidBody, SequentialImpulseEngine, Shape, Triangle};

/// Rotational kinetic energy oracle for the CCD energy-cap tests.
fn rotational_energy(body: &RigidBody) -> f32 {
    let w = body.orientation.conjugate() * body.angular_velocity;
    0.5 * body.inertia.dot(w * w)
}

/// Unit tetrahedron corners for hull rest tests (edge length 2,
/// centered near the origin).
fn tetra_vertices() -> Vec<Vec3> {
    vec![
        Vec3::new(1.0, 0.0, -1.0 / 2.0f32.sqrt()),
        Vec3::new(-1.0, 0.0, -1.0 / 2.0f32.sqrt()),
        Vec3::new(0.0, 1.0, 1.0 / 2.0f32.sqrt()),
        Vec3::new(0.0, -1.0, 1.0 / 2.0f32.sqrt()),
    ]
}

/// Fast-drop guard for settled-cost work: a box thrown at the floor at
/// -40 m/s rests on it instead of tunneling (TOI clamp + discrete
/// solve, perf_probe `fast_drop` scenario as a unit test).
#[test]
fn fast_box_drop_does_not_tunnel() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(10.0, 0.5, 10.0),
        0.0,
    ));
    let mut ball = RigidBody::new_box(Vec3::new(0.0, 8.0, 0.0), Vec3::splat(0.4), 1.0);
    ball.velocity = Vec3::new(0.0, -40.0, 0.0);
    // Inelastic: the pin is "no tunnel, then rest", not the bounce.
    // Default e = 0.3 still leaves this drop in the air after 120 steps.
    ball.restitution = 0.0;
    let h = physics.add_body(ball);
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(h).unwrap();
    assert!(
        b.position.y > -0.4,
        "fast box tunneled through the floor: {:?}",
        b.position
    );
    assert!(
        (b.position.y - 0.4).abs() < 0.3,
        "fast box must rest on the floor, got {}",
        b.position.y
    );
}

#[test]
fn sat_cache_is_shared_by_parallel_narrowphase() {
    // 299 overlapping slow box pairs force the rayon path (>256) and the
    // SAT-eligible branch (near-zero speeds, substep 0). The lock-free
    // cache must fill on the first pass and replay identically.
    let mut bodies = Vec::new();
    for i in 0..300 {
        bodies.push(RigidBody::new_box(
            Vec3::new(i as f32 * 0.9, 0.0, 0.0),
            Vec3::splat(0.5),
            1.0,
        ));
    }
    let pairs: Vec<(usize, usize)> = (0..299).map(|i| (i, i + 1)).collect();
    let asleep = vec![false; bodies.len()];
    let cache = SatCache::default();
    let mut first: Vec<Manifold> = Vec::new();
    let mut pool = NarrowShardPool::default();
    detect_collisions_into(
        &bodies,
        &pairs,
        &asleep,
        1.0 / 240.0,
        &mut first,
        None,
        0,
        Some(&cache),
        &mut pool,
    );
    assert_eq!(
        cache.len(),
        pairs.len(),
        "every slow pair seeds the SAT cache"
    );
    assert!(!first.is_empty());
    let mut second: Vec<Manifold> = Vec::new();
    detect_collisions_into(
        &bodies,
        &pairs,
        &asleep,
        1.0 / 240.0,
        &mut second,
        None,
        0,
        Some(&cache),
        &mut pool,
    );
    assert_eq!(second.len(), first.len());
    for (a, b) in first.iter().zip(second.iter()) {
        assert_eq!((a.body_a, a.body_b), (b.body_a, b.body_b));
        assert!((a.normal - b.normal).length() < 1e-6);
        assert_eq!(a.point_count, b.point_count);
    }
}

#[test]
fn sphere_vs_sphere_collision() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_sphere(Vec3::new(-0.4, 0.0, 0.0), 0.5, 1.0));
    let b = physics.add_body(RigidBody::new_sphere(Vec3::new(0.4, 0.0, 0.0), 0.5, 1.0));
    physics.step(1.0 / 60.0);
    let body_a = physics.get_body(a).unwrap();
    let body_b = physics.get_body(b).unwrap();
    let dist = (body_a.position - body_b.position).length();
    assert!(dist < 1.1);
}

#[test]
fn box_vs_box_collision() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let a = physics.add_body(RigidBody::new_box(
        Vec3::new(-0.4, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        1.0,
    ));
    let b = physics.add_body(RigidBody::new_box(
        Vec3::new(0.4, 0.0, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        1.0,
    ));
    physics.step(1.0 / 60.0);
    let body_a = physics.get_body(a).unwrap();
    let body_b = physics.get_body(b).unwrap();
    let dist = (body_a.position - body_b.position).length();
    assert!(dist < 1.1);
}

// ---- G1: orientation + angular dynamics ----

#[test]
fn oriented_boxes_collide() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    // Same center, rotated 45° about Y, box-ish units: OBB-OBB should separate.
    let half = Vec3::new(0.5, 0.5, 0.5);
    let a = physics.add_body(
        RigidBody::new_box(Vec3::new(0.0, 0.0, 0.0), half, 1.0)
            .with_orientation(Quat::from_rotation_z(0.0)),
    );
    let b = physics.add_body(
        RigidBody::new_box(Vec3::new(0.4, 0.0, 0.0), half, 1.0)
            .with_orientation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
    );
    physics.step(1.0 / 60.0);
    let body_a = physics.get_body(a).unwrap();
    let body_b = physics.get_body(b).unwrap();
    // Resting separation for two half-0.5 cubes is exactly 1.0 (touching).
    let dist = (body_a.position - body_b.position).length();
    assert!(dist <= 1.05, "oriented boxes should resolve, dist={dist}");
}

#[test]
fn obb_aabb_respects_rotation() {
    let half = Vec3::new(1.0, 1.0, 1.0);
    let q = glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
    let aabb = Shape::Box { half_extents: half }.aabb(Vec3::ZERO, q);
    // ALL EIGHT rotated corners must lie inside the AABB — not just one.
    // The previous version checked only `q * Vec3::splat(1.0)`, which
    // for a unit-cube half-extent is numerically identical to the
    // (buggy) `orientation.mul_vec3(half).abs()` formula the
    // production code used to compute, so the assertion was
    // tautological and passed even with an under-sized AABB (night
    // gate, 2026-08-24: fixed real OBB->AABB bug in `Shape::aabb`,
    // see its comment for the derivation).
    for sx in [-1.0f32, 1.0] {
        for sy in [-1.0f32, 1.0] {
            for sz in [-1.0f32, 1.0] {
                let corner = q * (half * Vec3::new(sx, sy, sz));
                assert!(
                    aabb.contains_point(corner),
                    "corner {corner:?} not inside {aabb:?}"
                );
            }
        }
    }
    // Both X and Y half-extents grow to sqrt(2) after a 45° Z rotation
    // of a unit cube (Z is the rotation axis, so its extent is
    // unchanged). The buggy formula zeroed the X extent here.
    let half_x = (aabb.max.x - aabb.min.x) * 0.5;
    let half_y = (aabb.max.y - aabb.min.y) * 0.5;
    let half_z = (aabb.max.z - aabb.min.z) * 0.5;
    assert!(
        (half_x - 2f32.sqrt()).abs() < 1e-3,
        "OBB->AABB x-extent, got {half_x}"
    );
    assert!(
        (half_y - 2f32.sqrt()).abs() < 1e-3,
        "OBB->AABB y-extent, got {half_y}"
    );
    assert!(
        (half_z - 1.0).abs() < 1e-3,
        "OBB->AABB z-extent, got {half_z}"
    );
}

#[test]
fn sphere_capsule_collision() {
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    let sphere = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 0.0, 0.0), 0.5, 1.0));
    let capsule = physics.add_body(
        RigidBody::new_capsule(Vec3::new(0.6, 0.0, 0.0), 0.5, 1.0, 1.0)
            .with_orientation(glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
    );
    physics.step(1.0 / 60.0);
    let d = (physics.get_body(sphere).unwrap().position
        - physics.get_body(capsule).unwrap().position)
        .length();
    // Sphere radius 0.5 + capsule radius 0.5 -> resting center distance ~1.0.
    assert!(d <= 1.05, "sphere/capsule should resolve on contact, d={d}");
}

#[test]
fn fast_sphere_does_not_tunnel() {
    // Bullet vs thin floor (G6): at -80 m/s the sphere moves 0.111 m per
    // substep (12 substeps at 60 Hz) — more than the 0.1 m floor slab.
    // Without speculative contacts + the TOI pass it would sail through;
    // here it must end up resting on top.
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::ZERO,
        Vec3::new(10.0, 0.05, 10.0),
        0.0,
    ));
    let bullet = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 3.0, 0.0), 0.1, 1.0));
    {
        let b = physics.get_body_mut(bullet).unwrap();
        b.velocity = Vec3::new(0.0, -80.0, 0.0);
        b.restitution = 0.0; // we test tunneling, not bouncing
    }
    for _ in 0..120 {
        physics.step(1.0 / 60.0);
    }
    let y = physics.get_body(bullet).unwrap().position.y;
    assert!(y > 0.0, "bullet tunneled through the floor: y={y}");
    // And it settled near the contact plane (center = slab top + radius),
    // not hovering or buried.
    assert!(
        (y - 0.15).abs() < 0.05,
        "bullet did not settle on the floor: y={y}"
    );
}

/// Travel gate, thin body: 10°/substep is below the old flat 15° gate, but
/// R·angle = 1.51·0.175 = 0.26 > 0.5·0.2 = 0.1 arms CCD — blades tunnel
/// easily, so they get the sweep early. Pre-rotated to 70° so the bar
/// enters the target box (0, 1.1) mid-substep (separated at 70°,
/// centerline inside at 80°).
#[test]
fn angular_gate_fires_below_15deg_for_thin_bodies() {
    let dt = 1.0 / 60.0;
    let mut mover = RigidBody::new_box(Vec3::ZERO, Vec3::new(1.5, 0.1, 0.1), 1.0);
    mover.orientation = Quat::from_rotation_z(70.0f32.to_radians());
    mover.angular_velocity = Vec3::Z * (10.0f32.to_radians() / dt);
    let target = RigidBody::new_box(Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.2, 0.05, 0.2), 0.0);
    let bodies = [mover, target];

    let hit = find_angular_continuous_hit(&bodies, 0, Vec3::ZERO, dt)
        .expect("thin fast spinner must arm angular CCD below 15°/substep");
    assert!(hit.angular());
    assert!(hit.fraction > 0.0 && hit.fraction < 1.0);
    assert!(hit.contact.is_some(), "angular hit must carry its contact");
}

/// Travel gate, chunky body: 20°/substep exceeds the old flat 15° gate,
/// but R·angle = 0.87·0.35 = 0.30 < 0.5·1.0 = 0.5 exempts the cube —
/// the discrete phase resolves that travel, CCD would be pure overhead.
#[test]
fn angular_gate_spares_slow_chunky_spinners() {
    let dt = 1.0 / 60.0;
    let mut mover = RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0);
    mover.angular_velocity = Vec3::Z * (20.0f32.to_radians() / dt);
    let target = RigidBody::new_box(Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.2, 0.05, 0.2), 0.0);
    let bodies = [mover, target];

    assert!(
        find_angular_continuous_hit(&bodies, 0, Vec3::ZERO, dt).is_none(),
        "chunky slow spinner must skip angular CCD"
    );
}

/// Frictionless response, true 3D: ω=(10,0,10), lever +Y, normal −Z.
/// The X-spin drives the approach, the Z-spin is tangential and must
/// survive exactly; the old full stop would return zero here.
#[test]
fn angular_response_keeps_tangential_spin_in_3d() {
    let out = remove_angular_approach(
        Vec3::new(10.0, 0.0, 10.0),
        Vec3::splat(2.0),
        Quat::IDENTITY,
        Vec3::Y,
        Vec3::NEG_Z,
    );
    assert!(
        (out - Vec3::new(0.0, 0.0, 10.0)).length() < 1e-6,
        "tangential spin must survive, got {out:?}"
    );
}

/// Frictionless response, planar head-on: exactly the old full stop
/// (ω ∥ lever×normal always in-plane, so the constraint eats all spin).
#[test]
fn angular_response_stops_planar_head_on() {
    let out = remove_angular_approach(
        Vec3::Z * 10.0,
        Vec3::splat(2.0),
        Quat::IDENTITY,
        Vec3::X * 1.5,
        Vec3::NEG_Y,
    );
    assert!(out.length() < 1e-5, "planar head-on must stop, got {out:?}");
}

/// Stiff-lever cap: thin-bar inertia (huge I⁻¹ on X) with an off-axis
/// contact. The exact projection would demand a wall impulse that
/// injects spin energy (the blender); the cap holds energy neutral and
/// leaves the rest to the discrete solver.
#[test]
fn angular_response_caps_energy_on_stiff_levers() {
    let w = Vec3::Z * 90.0;
    let inertia = Vec3::new(0.0067, 0.753, 0.753);
    let energy = |v: Vec3| 0.5 * inertia.dot(v * v);
    let out = remove_angular_approach(
        w,
        inertia,
        Quat::IDENTITY,
        Vec3::new(1.5, 0.05, 0.05),
        Vec3::NEG_Y,
    );
    assert!(out.is_finite(), "cap must never produce NaN, got {out:?}");
    assert!(
        energy(out) <= energy(w) + 1e-3,
        "cap must not inject energy: {} -> {}",
        energy(w),
        energy(out)
    );
    // ...while still reducing the approach (cap walks back along the
    // correction, never reverses it).
    let approach = |v: Vec3| v.cross(Vec3::new(1.5, 0.05, 0.05)).dot(Vec3::NEG_Y);
    assert!(
        approach(out) >= approach(w),
        "cap must not worsen the approach: {} -> {}",
        approach(w),
        approach(out)
    );
}

/// Unified impact, restitution: v=0, ω=(10,0,10), lever +Y, normal −Z,
/// e=0.5. The tip approaches at −10 m/s, bounces to +5; the impulse
/// couples into translation (v'=−10ẑ) and trims the driving X-spin
/// (ω'=(5,0,10)) while the tangential Z-spin survives. Old code: v
/// untouched, ω zeroed — a spinning bounce was impossible.
#[test]
fn ccd_impact_couples_spin_and_bounce() {
    let (v, w) = ccd_impact_velocity(
        Vec3::ZERO,
        Vec3::new(10.0, 0.0, 10.0),
        1.0,
        Vec3::splat(2.0),
        Quat::IDENTITY,
        Vec3::Y,
        Vec3::NEG_Z,
        0.5,
    );
    assert!(
        (v - Vec3::new(0.0, 0.0, -10.0)).length() < 1e-5,
        "bounce couples into translation, got {v:?}"
    );
    assert!(
        (w - Vec3::new(5.0, 0.0, 10.0)).length() < 1e-5,
        "driving spin trimmed, tangential kept, got {w:?}"
    );
    // Restitution identity on the contact point: vn' = −e·vn.
    let vn_after = (v + w.cross(Vec3::Y)).dot(Vec3::NEG_Z);
    assert!(
        (vn_after - 5.0).abs() < 1e-4,
        "vn' must be +5, got {vn_after}"
    );
}

/// Unified impact, inelastic limit: same setup with e=0 kills the
/// contact approach exactly (vn' ≈ 0), dissipating total energy.
#[test]
fn ccd_impact_inelastic_kills_contact_approach() {
    let (v, w) = ccd_impact_velocity(
        Vec3::ZERO,
        Vec3::new(10.0, 0.0, 10.0),
        1.0,
        Vec3::splat(2.0),
        Quat::IDENTITY,
        Vec3::Y,
        Vec3::NEG_Z,
        0.0,
    );
    let vn_after = (v + w.cross(Vec3::Y)).dot(Vec3::NEG_Z);
    assert!(
        vn_after.abs() < 1e-4,
        "inelastic must stop the approach, got {vn_after}"
    );
    assert!(
        (v - Vec3::new(0.0, 0.0, -20.0 / 3.0)).length() < 1e-5,
        "got {v:?}"
    );
}

/// Unified impact, stiff lever: thin-bar inertia, e=0. The inv_mass
/// floor in the denominator regularizes the impulse (no blender);
/// total energy must drop, approach must vanish.
#[test]
fn ccd_impact_stiff_lever_dissipates() {
    let w0 = Vec3::Z * 90.0;
    let inertia = Vec3::new(0.0067, 0.753, 0.753);
    let lever = Vec3::new(1.5, 0.05, 0.05);
    let total = |v: Vec3, w: Vec3| 0.5 * v.dot(v) + 0.5 * inertia.dot(w * w);
    let (v, w) = ccd_impact_velocity(
        Vec3::ZERO,
        w0,
        1.0,
        inertia,
        Quat::IDENTITY,
        lever,
        Vec3::NEG_Y,
        0.0,
    );
    assert!(v.is_finite() && w.is_finite(), "got {v:?} {w:?}");
    assert!(
        total(v, w) <= total(Vec3::ZERO, w0) + 1e-2,
        "inelastic impact must dissipate: {} -> {}",
        total(Vec3::ZERO, w0),
        total(v, w)
    );
    let vn_after = (v + w.cross(lever)).dot(Vec3::NEG_Y);
    assert!(
        vn_after.abs() < 1e-2,
        "approach must vanish, got {vn_after}"
    );
}

/// Engine-level spin bounce: cube corner 10 cm above the floor (outside
/// the 5 cm speculative margin, so no discrete phantom contact
/// pre-empts the sweep), pure (40,0,40) spin, zero gravity, one
/// substep. The corner outruns the discrete phase; CCD must clamp (no
/// penetration) AND convert spin into an upward pop (old code:
/// velocity stays zero, spin dies).
#[test]
fn ccd_spin_bounce_pops_upward() {
    let dt = 1.0 / 60.0;
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    let floor_pos = Vec3::new(0.0, -1.0, 0.0);
    let floor_half = Vec3::new(5.0, 1.0, 5.0);
    physics.add_body(RigidBody::new_box(floor_pos, floor_half, 0.0));
    let mover = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.6, 0.0),
        Vec3::splat(0.5),
        1.0,
    ));
    physics.get_body_mut(mover).unwrap().angular_velocity = Vec3::new(40.0, 0.0, 40.0);

    physics.step(dt);

    let body = physics.get_body(mover).expect("mover remains alive");
    assert!(
        obb_sat(
            body.position,
            Vec3::splat(0.5),
            body.orientation,
            floor_pos,
            floor_half,
            Quat::IDENTITY,
            1e-5
        )
        .is_none(),
        "CCD must not let the corner through the floor"
    );
    assert!(
        body.velocity.y > 2.0,
        "spin bounce must pop the body up, got {:?}",
        body.velocity
    );
}

/// Frictionless response, separating scrape: bit-identical passthrough.
#[test]
fn angular_response_ignores_separating_scrape() {
    let w = Vec3::Z * 10.0;
    let out = remove_angular_approach(w, Vec3::splat(2.0), Quat::IDENTITY, Vec3::X * 1.5, Vec3::Y);
    assert_eq!(out.to_array(), w.to_array());
}

/// Engine-level 3D graze: a cube corner outruns the discrete phase
/// (0.82 m/substep vs a 1 cm gap) with a combined (40,0,40) spin. CCD
/// must clamp before the wall face AND keep tangential spin (the old
/// response returns exactly zero here).
#[test]
fn angular_graze_keeps_tangential_spin() {
    let dt = 1.0 / 60.0;
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
    physics.set_substeps(1);
    let wall_half = Vec3::new(0.49, 2.0, 2.0);
    let wall_pos = Vec3::new(1.0, 0.0, 0.0);
    physics.add_body(RigidBody::new_box(wall_pos, wall_half, 0.0));
    let mover = physics.add_body(RigidBody::new_box(Vec3::ZERO, Vec3::splat(0.5), 1.0));
    physics.get_body_mut(mover).unwrap().angular_velocity = Vec3::new(40.0, 0.0, 40.0);

    physics.step(dt);

    let body = physics.get_body(mover).expect("mover remains alive");
    assert!(
        obb_sat(
            body.position,
            Vec3::splat(0.5),
            body.orientation,
            wall_pos,
            wall_half,
            Quat::IDENTITY,
            1e-5
        )
        .is_none(),
        "CCD must not let the corner through the wall"
    );
    assert!(
        body.angular_velocity.length() > 5.0,
        "tangential spin must survive the graze, got {:?}",
        body.angular_velocity
    );
}

#[test]
fn angular_sweep_finds_rotating_box_impact() {
    let dt = 1.0 / 60.0;
    let mut mover = RigidBody::new_box(Vec3::ZERO, Vec3::new(1.5, 0.1, 0.1), 1.0);
    mover.angular_velocity = Vec3::Z * (std::f32::consts::FRAC_PI_2 / dt);
    let target = RigidBody::new_box(Vec3::new(0.0, 1.1, 0.0), Vec3::new(0.2, 0.05, 0.2), 0.0);
    let bodies = [mover, target];

    let hit = find_angular_continuous_hit(&bodies, 0, Vec3::ZERO, dt)
        .expect("angular sweep must find the rotating box impact");
    assert_eq!(hit.handle, ornis_physics::BodyHandle::from_raw(1));
    assert!(hit.angular());
    assert!(hit.fraction > 0.0 && hit.fraction < 1.0);
}

#[test]
fn angular_continuous_motion_stops_at_first_impact() {
    let dt = 1.0 / 60.0;
    let mut physics = SequentialImpulseEngine::new(Vec3::ZERO);
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

    let e_before = rotational_energy(physics.get_body(mover).expect("mover remains alive"));

    physics.step(dt);

    let body = physics.get_body(mover).expect("mover remains alive");
    // New contract (frictionless response + energy cap): the sweep still
    // clamps before tunneling, but a stiff out-of-plane lever no longer
    // dies to zero — the correction is capped at energy-neutral and the
    // discrete solver owns the remainder next substep.
    assert!(
        rotational_energy(body) <= e_before + 1e-3,
        "CCD response must never inject spin energy, before={e_before} after={} {:?}",
        rotational_energy(body),
        body.angular_velocity
    );
    assert_ne!(
        body.orientation,
        Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
        "rotating body must not jump through the target"
    );
}

#[test]
fn box_manifold_produces_four_points() {
    // Two equal half-0.5 boxes, overlapping by 0.25 along +Y: the resting
    // face yields 4 manifold points (vertex-face contact), not one.
    let half = Vec3::new(0.5, 0.5, 0.5);
    let m = box_manifold(
        Vec3::new(0.0, 0.0, 0.0),
        half,
        Quat::IDENTITY,
        Vec3::new(0.0, 0.75, 0.0),
        half,
        Quat::IDENTITY,
        0.05,
    )
    .expect("boxes overlap");
    assert_eq!(m.point_count, 4, "expected a 4-point manifold");
    assert!((m.normal - Vec3::Y).length() < 1e-3, "normal should be +Y");
    for k in 0..m.point_count {
        assert!(
            m.points[k].penetration > 0.0,
            "point {k} has positive penetration"
        );
    }
}

#[test]
fn box_rests_on_static_floor() {
    // A box in free fall must settle on a static floor (G2b target).
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let top = physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, 0.8, 0.0),
        Vec3::new(0.5, 0.5, 0.5),
        1.0,
    ));
    for _ in 0..240 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(top).unwrap();
    assert!(
        b.position.y > 0.40 && b.position.y < 0.55,
        "box should rest at y≈0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.05,
        "settled velocity: {:?}",
        b.velocity
    );
    assert!(
        b.angular_velocity.length() < 0.05,
        "no jitter: {:?}",
        b.angular_velocity
    );
}

#[test]
fn sphere_rests_on_static_floor() {
    // G2 gate: a sphere dropped on a static floor settles and stays.
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(5.0, 1.0, 5.0),
        0.0,
    ));
    let ball = physics.add_body(RigidBody::new_sphere(Vec3::new(0.0, 2.0, 0.0), 0.5, 1.0));
    for _ in 0..240 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(ball).unwrap();
    assert!(
        b.position.y > 0.40 && b.position.y < 0.55,
        "sphere should rest at y≈0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.05,
        "settled velocity: {:?}",
        b.velocity
    );
}

/// `kinematic_cast` unit gate: a box wall sweeping 1 m across a thin
/// box victim reports the crossing the shared `cast_shape` cannot see
/// (the OBB distance oracle is unsigned). Pins hit distance and normal.
#[test]
fn kinematic_cast_finds_box_box_crossing() {
    use ornis_physics::distance::ShapeRef;
    let wall = Shape::Box {
        half_extents: Vec3::new(0.5, 1.0, 1.0),
    };
    let victim = Shape::Box {
        half_extents: Vec3::new(0.02, 0.5, 0.5),
    };
    let target = ShapeRef {
        shape: &victim,
        pos: Vec3::ZERO,
        rot: Quat::IDENTITY,
    };
    // Start gap is the face-to-face 0.48 m (signed SAT separation).
    let gap = sweep_gap(&wall, Vec3::new(-1.0, 0.0, 0.0), Quat::IDENTITY, target);
    assert!((gap - 0.48).abs() < 1e-4, "signed start gap, got {gap}");
    let (t, n) = kinematic_cast(
        &wall,
        Quat::IDENTITY,
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        target,
    )
    .expect("wall front must reach the victim mid-segment");
    assert!((t - 0.48).abs() < 0.01, "hit travel, got {t}");
    assert!(
        (n - Vec3::NEG_X).length() < 1e-3,
        "normal back toward the mover, got {n}"
    );
    // Touching at t=0 is resting contact, not a sweep hit.
    assert!(
        kinematic_cast(
            &wall,
            Quat::IDENTITY,
            Vec3::new(-0.52, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            target,
        )
        .is_none(),
        "starting contact must not report"
    );
}

/// Cylinder (GJK/EPA path) falls onto a static floor and rests on its
/// flat cap: position near the rest pose AND velocity near zero (a
/// velocity-only assert would also pass for a body that rolled off).
#[test]
fn cylinder_rests_on_static_floor() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        0.0,
    ));
    let body = physics.add_body(RigidBody::new_cylinder(
        Vec3::new(0.0, 3.0, 0.0),
        0.5,
        0.5,
        1.0,
    ));
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(body).unwrap();
    assert!(
        (b.position.y - 0.5).abs() < 0.05,
        "cylinder must rest on its cap at y=0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.15,
        "resting velocity near zero, got {}",
        b.velocity
    );
}

/// Cone dropped base-down rests on its base disk (EPA penetration
/// path on first contact, GJK separation after).
#[test]
fn cone_rests_on_base() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        0.0,
    ));
    let body = physics.add_body(RigidBody::new_cone(Vec3::new(0.0, 3.0, 0.0), 0.5, 0.5, 1.0));
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(body).unwrap();
    assert!(
        (b.position.y - 0.5).abs() < 0.08,
        "cone must rest on its base at y=0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.2,
        "resting velocity near zero, got {}",
        b.velocity
    );
}

/// Cube hull (GJK path) falls flat and rests exactly like an analytic
/// box: the hull of the unit-cube corners must agree with the box
/// oracle everywhere, including the settled state.
#[test]
fn hull_cube_rests_on_floor() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        0.0,
    ));
    let body = physics.add_body(RigidBody::new_convex_hull(
        Vec3::new(0.0, 2.0, 0.0),
        vec![
            Vec3::new(-0.5, -0.5, -0.5),
            Vec3::new(0.5, -0.5, -0.5),
            Vec3::new(-0.5, 0.5, -0.5),
            Vec3::new(0.5, 0.5, -0.5),
            Vec3::new(-0.5, -0.5, 0.5),
            Vec3::new(0.5, -0.5, 0.5),
            Vec3::new(-0.5, 0.5, 0.5),
            Vec3::new(0.5, 0.5, 0.5),
        ],
        1.0,
    ));
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(body).unwrap();
    assert!(
        (b.position.y - 0.5).abs() < 0.05,
        "cube hull must rest at y=0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.15,
        "resting velocity near zero, got {}",
        b.velocity
    );
}

/// Flat quad mesh (2 triangles, +Y wound) as a static floor: a ball
/// dropped on it rests at terrain + radius through the BVH triangle
/// loop — the concave-mesh narrow path end to end.
#[test]
fn ball_rests_on_trimesh_floor() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let floor_tris = [Triangle::from_raw([0, 1, 2]), Triangle::from_raw([0, 3, 1])];
    physics.add_body(RigidBody::new_trimesh(
        Vec3::ZERO,
        &[
            Vec3::new(-5.0, 0.0, -5.0),
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(5.0, 0.0, -5.0),
            Vec3::new(-5.0, 0.0, 5.0),
        ],
        &floor_tris,
        0.0,
    ));
    let ball = physics.add_body(RigidBody::new_sphere(Vec3::new(0.4, 4.0, -0.3), 0.5, 1.0));
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(ball).unwrap();
    assert!(
        (b.position.y - 0.5).abs() < 0.08,
        "ball must rest at mesh top (0.0)+radius(0.5), got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.2,
        "resting velocity near zero, got {}",
        b.velocity
    );
}

/// Dynamic cube mesh (12 triangles, Mirtich inertia) dropped flat on a
/// static box floor rests like an analytic box.
#[test]
fn mesh_cube_rests_on_floor() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        0.0,
    ));
    let v = [
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
    ];
    let raw = [
        [0, 3, 1],
        [0, 2, 3],
        [4, 5, 7],
        [4, 7, 6],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
        [0, 1, 5],
        [0, 5, 4],
        [2, 7, 3],
        [2, 6, 7],
    ];
    let idx: Vec<Triangle> = raw.iter().map(|t| Triangle::from_raw(*t)).collect();
    let body = physics.add_body(RigidBody::new_trimesh(
        Vec3::new(0.0, 2.0, 0.0),
        &v,
        &idx,
        1.0,
    ));
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(body).unwrap();
    assert!(
        (b.position.y - 0.5).abs() < 0.08,
        "mesh cube must rest at y=0.5, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.2,
        "resting velocity near zero, got {}",
        b.velocity
    );
}

/// Mesh-vs-mesh is undefined (concave-concave): overlapping meshes
/// produce no contact and the dynamic one keeps falling.
#[test]
fn mesh_vs_mesh_reports_no_contact() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    let v = [
        Vec3::new(-0.5, -0.5, -0.5),
        Vec3::new(0.5, -0.5, -0.5),
        Vec3::new(-0.5, 0.5, -0.5),
        Vec3::new(0.5, 0.5, -0.5),
        Vec3::new(-0.5, -0.5, 0.5),
        Vec3::new(0.5, -0.5, 0.5),
        Vec3::new(-0.5, 0.5, 0.5),
        Vec3::new(0.5, 0.5, 0.5),
    ];
    let raw = [
        [0, 3, 1],
        [0, 2, 3],
        [4, 5, 7],
        [4, 7, 6],
        [0, 4, 6],
        [0, 6, 2],
        [1, 3, 7],
        [1, 7, 5],
        [0, 1, 5],
        [0, 5, 4],
        [2, 7, 3],
        [2, 6, 7],
    ];
    let idx: Vec<Triangle> = raw.iter().map(|t| Triangle::from_raw(*t)).collect();
    physics.add_body(RigidBody::new_trimesh(Vec3::ZERO, &v, &idx, 0.0));
    let top = physics.add_body(RigidBody::new_trimesh(
        Vec3::new(0.0, 0.4, 0.0),
        &v,
        &idx,
        1.0,
    ));
    for _ in 0..60 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(top).unwrap();
    assert_eq!(
        physics.debug_contact_count(top),
        0,
        "mesh-vs-mesh must not contact"
    );
    assert!(
        b.position.y < 0.4,
        "unsupported pair keeps falling, got y={}",
        b.position.y
    );
}

/// Tetrahedron dropped face-down settles on its face (hull-vs-box
/// through the generic distance-contact builder). Settles thanks to
/// the hull rolling/torsion defaults (0.2/0.05): undamped, a tetra
/// rocks on vertices/edges indefinitely. Vertex-first drops are still
/// not asserted to rest (unstable equilibrium by geometry).
#[test]
fn hull_tetrahedron_rests_on_face() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_box(
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::new(5.0, 0.5, 5.0),
        0.0,
    ));
    // Face (v1,v2,v3) has outward normal ~(-0.816,0,+0.577) (away from
    // the apex v0): lay it flat down. Rest height = face distance
    // 0.408 below the center.
    let n = Vec3::new(-2.828427, 0.0, 2.0).normalize();
    let mut hull = RigidBody::new_convex_hull(Vec3::new(0.0, 1.2, 0.0), tetra_vertices(), 1.0)
        .with_orientation(Quat::from_rotation_arc(n, Vec3::NEG_Y));
    // Face rest is the rolling/torsion pin. Default e = 0.3 keeps the
    // drop bouncing past the rest height for the whole run.
    hull.restitution = 0.0;
    let body = physics.add_body(hull);
    for _ in 0..900 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(body).unwrap();
    assert!(
        (b.position.y - 0.408).abs() < 0.1,
        "tetrahedron must rest on its face at y~0.408, got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.25,
        "resting velocity near zero, got {}",
        b.velocity
    );
    assert!(
        b.angular_velocity.length() < 0.25,
        "resting spin near zero, got {}",
        b.angular_velocity
    );
}

/// Ball dropped onto a flat heightfield rests on the terrain surface
/// (column-oracle path): center height == sample height + radius.
#[test]
fn ball_rests_on_flat_heightfield() {
    let mut physics = SequentialImpulseEngine::new(Vec3::new(0.0, -9.81, 0.0));
    physics.add_body(RigidBody::new_heightfield(
        Vec3::ZERO,
        vec![1.0f32; 16],
        4,
        4,
        1.0,
        0.0,
    ));
    let ball = physics.add_body(RigidBody::new_sphere(Vec3::new(0.4, 5.0, -0.3), 0.5, 1.0));
    for _ in 0..600 {
        physics.step(1.0 / 60.0);
    }
    let b = physics.get_body(ball).unwrap();
    assert!(
        (b.position.y - 1.5).abs() < 0.08,
        "ball must rest at terrain(1.0)+radius(0.5), got {}",
        b.position.y
    );
    assert!(
        b.velocity.length() < 0.2,
        "resting velocity near zero, got {}",
        b.velocity
    );
}
