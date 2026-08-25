//! Integration test for `#[derive(Pack)]` (Component Packing, SoA).
//!
//! Proves `Pack` is wired into the ECS, not dead code: a packed struct is
//! registered/inserted via the generated `Pack` methods, traversed with the
//! generated `for_each_packed` (the packed analogue of `for_each_entity!`),
//! and its lanes are directly usable by `for_each_entity!` through the
//! generated wrapper lane types.

use ornis_core::{Entity, Pack, SmartStore};
use ornis_macros::{for_each_entity, Pack};

#[derive(Clone, Pack)]
struct Transform {
    x: f32,
    y: f32,
    z: f32,
}

#[test]
fn pack_register_insert_and_roundtrip() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    let t = Transform {
        x: 1.0,
        y: 2.0,
        z: 3.0,
    };
    t.pack_insert(&mut store, e);

    let got = Transform::pack_get(&store, e).expect("entity has the packed component");
    assert_eq!(got.x, 1.0);
    assert_eq!(got.y, 2.0);
    assert_eq!(got.z, 3.0);
}

#[test]
fn pack_for_each_packed_mutates_fields() {
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let a = store.create_entity();
    let b = store.create_entity();
    Transform { x: 0.0, y: 0.0, z: 0.0 }.pack_insert(&mut store, a);
    Transform { x: 0.0, y: 0.0, z: 0.0 }.pack_insert(&mut store, b);

    // `for_each_packed` walks every entity owning `Transform` and hands the
    // closure `&mut PackMut` — field accessors per lane (SoA).
    Transform::for_each_packed(&mut store, |c| {
        let x = *c.x();
        let y = *c.y();
        let z = *c.z();
        *c.x() = x + 10.0;
        *c.y() = y + 20.0;
        *c.z() = z + 30.0;
    });

    let got_a = Transform::pack_get(&store, a).unwrap();
    let got_b = Transform::pack_get(&store, b).unwrap();
    assert_eq!((got_a.x, got_a.y, got_a.z), (10.0, 20.0, 30.0));
    assert_eq!((got_b.x, got_b.y, got_b.z), (10.0, 20.0, 30.0));
}

#[test]
fn pack_lanes_compatible_with_for_each_entity() {
    // The `#[derive(Pack)]` macro generates one wrapper lane type per field.
    // Those wrappers are ordinary `register`-ed components, so `for_each_entity!`
    // can iterate them directly — proving Pack integrates with the existing
    // parallel traversal without a bespoke code path.
    let mut store = SmartStore::new();
    Transform::pack_register(&mut store);

    let e = store.create_entity();
    Transform { x: 5.0, y: 6.0, z: 7.0 }.pack_insert(&mut store, e);

    // `Transform__x__PackLane__0` is the generated wrapper for field `x`.
    let mut seen = Vec::new();
    for_each_entity!(store, |lane: &Transform__x__PackLane__0| {
        seen.push(lane.0);
    });
    assert_eq!(seen, vec![5.0]);
}
