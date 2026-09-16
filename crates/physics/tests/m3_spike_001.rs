//! THROWAWAY M3 SPIKE 001 — cross-solver contact via kinematic mirrors.
//! Not production code: hardcodes everything, no quality gates apply.
//! Delete with `spikes/001-cross-solver-contact/` after the verdict.

use glam::Vec3;
use ornis_physics::{
    AvbdEngine, BodyHandle, BodyType, BuiltinPhysicsEngine, PhysicsEngine, RigidBody,
};

const DT: f32 = 1.0 / 60.0;
const GRAV: Vec3 = Vec3::new(0.0, -9.81, 0.0);

fn floor() -> RigidBody {
    RigidBody::new_box(Vec3::new(0.0, -1.0, 0.0), Vec3::new(10.0, 1.0, 10.0), 0.0)
}

fn dyn_box(y: f32) -> RigidBody {
    RigidBody::new_box(Vec3::new(0.0, y, 0.0), Vec3::splat(0.5), 1.0)
}

/// Kinematic mirror: driven pose, infinite mass.
fn mirror(y: f32) -> RigidBody {
    let mut b = RigidBody::new_box(Vec3::new(0.0, y, 0.0), Vec3::splat(0.5), 1.0);
    b.body_type = BodyType::Kinematic;
    b.inv_mass = 0.0;
    b
}

struct Rig {
    avbd: AvbdEngine,
    builtin: BuiltinPhysicsEngine,
    top: BodyHandle,
    bottom: BodyHandle,
    mirror_top: BodyHandle,
    mirror_bottom: BodyHandle,
}

fn build() -> Rig {
    let mut avbd = AvbdEngine::new(GRAV);
    let mut builtin = BuiltinPhysicsEngine::new(GRAV);
    avbd.add_body(floor());
    builtin.add_body(floor());
    // AVBD owns the TOP box, Builtin the BOTTOM box.
    let top = avbd.add_body(dyn_box(1.6));
    let mirror_bottom = avbd.add_body(mirror(0.5));
    let bottom = builtin.add_body(dyn_box(0.5));
    let mirror_top = builtin.add_body(mirror(1.6));
    Rig {
        avbd,
        builtin,
        top,
        bottom,
        mirror_top,
        mirror_bottom,
    }
}

fn sync_mirrors(r: &mut Rig) {
    let t = r.avbd.get_body(r.top).unwrap();
    let (tp, tv) = (t.position, t.velocity);
    let m = r.builtin.get_body_mut(r.mirror_top).unwrap();
    m.position = tp;
    m.velocity = tv;
    let b = r.builtin.get_body(r.bottom).unwrap();
    let (bp, bv) = (b.position, b.velocity);
    let m = r.avbd.get_body_mut(r.mirror_bottom).unwrap();
    m.position = bp;
    m.velocity = bv;
}

fn step_coupled(r: &mut Rig) {
    sync_mirrors(r);
    r.avbd.step(DT);
    r.builtin.step(DT);
}

fn poses(r: &Rig) -> (Vec3, Vec3, Vec3, Vec3) {
    let t = r.avbd.get_body(r.top).unwrap();
    let b = r.builtin.get_body(r.bottom).unwrap();
    (t.position, t.velocity, b.position, b.velocity)
}

#[test]
#[ignore = "by-design failure: symmetric teleported mirrors bulldoze the stack (see spikes/001 verdict); kept runnable as the finding's proof"]
fn spike_cross_stack_rests() {
    let mut r = build();
    for _ in 0..300 {
        step_coupled(&mut r);
    }
    let (tp, tv, bp, bv) = poses(&r);
    assert!(
        (bp.y - 0.5).abs() < 0.1 && bv.length() < 0.3,
        "bottom must rest on floor: pos={tp:?}/{bp:?} vel={tv:?}/{bv:?}"
    );
    assert!(
        (tp.y - 1.5).abs() < 0.15 && tv.length() < 0.3,
        "top must rest on bottom: pos={tp:?}/{bp:?} vel={tv:?}/{bv:?}"
    );
}

#[test]
fn spike_cross_stack_deterministic() {
    let run = || {
        let mut r = build();
        for _ in 0..300 {
            step_coupled(&mut r);
        }
        poses(&r)
    };
    let (a, b) = (run(), run());
    assert_eq!(a, b, "coupled stepping must be rerun-identical");
}

#[test]
fn spike_coupled_step_overhead() {
    let mut r = build();
    for _ in 0..10 {
        step_coupled(&mut r);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..60 {
        step_coupled(&mut r);
    }
    let coupled = t0.elapsed();
    // Single-solver baseline: same two boxes, AVBD only.
    let mut e = AvbdEngine::new(GRAV);
    e.add_body(floor());
    let b1 = e.add_body(dyn_box(0.5));
    let b2 = e.add_body(dyn_box(1.6));
    for _ in 0..10 {
        e.step(DT);
    }
    let t0 = std::time::Instant::now();
    for _ in 0..60 {
        e.step(DT);
    }
    let single = t0.elapsed();
    let (tp, _, bp, _) = poses(&r);
    let s1 = e.get_body(b1).unwrap().position;
    let s2 = e.get_body(b2).unwrap().position;
    println!("coupled 60 steps: {coupled:?}, single 60 steps: {single:?}");
    println!("coupled bottom={bp:?} top={tp:?} | single bottom={s1:?} top={s2:?}");
}

/// One-directional variant: cross pair solved ONLY in AVBD (mirror of the
/// Builtin bottom, no mirror of the AVBD top in Builtin). If this rests
/// while the symmetric scheme bulldozes, the failure is the teleported
/// mirror plowing through the other body — diagnosis confirmed.
#[test]
fn spike_one_directional_rests() {
    let mut avbd = AvbdEngine::new(GRAV);
    let mut builtin = BuiltinPhysicsEngine::new(GRAV);
    avbd.add_body(floor());
    builtin.add_body(floor());
    let top = avbd.add_body(dyn_box(1.6));
    let mb = avbd.add_body(mirror(0.5));
    let bottom = builtin.add_body(dyn_box(0.5));
    for _ in 0..300 {
        let b = builtin.get_body(bottom).unwrap();
        let m = avbd.get_body_mut(mb).unwrap();
        m.position = b.position;
        m.velocity = b.velocity;
        avbd.step(DT);
        builtin.step(DT);
    }
    let t = avbd.get_body(top).unwrap();
    let b = builtin.get_body(bottom).unwrap();
    assert!(
        (b.position.y - 0.5).abs() < 0.1,
        "bottom must rest: {:?}",
        b.position
    );
    assert!(
        (t.position.y - 1.5).abs() < 0.15,
        "top must rest on mirrored bottom: {:?}",
        t.position
    );
}
