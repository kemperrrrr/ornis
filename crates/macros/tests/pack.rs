//! Functional smoke test: does `#[derive(Pack)]` actually work end-to-end?
//! Decides whether Pack is a usable feature or dead code to prune.

#![allow(dead_code)]
use ornis_core::{Entity, Pack, SmartStore};
use ornis_macros::Pack;

#[derive(Debug, Clone, PartialEq, Pack)]
struct Health {
    current: f32,
    max: f32,
}

#[test]
fn pack_roundtrip_insert_and_get() {
    let mut store = SmartStore::new();
    let e: Entity = store.create_entity();

    let hp = Health {
        current: 75.0,
        max: 100.0,
    };
    hp.pack_insert(&mut store, e);

    let back = Health::pack_get(&store, e).expect("roundtrip");
    assert_eq!(back, hp);
}

#[test]
fn pack_register_creates_lanes_and_get_missing_is_none() {
    let mut store = SmartStore::new();
    Health::pack_register(&mut store);

    let e: Entity = store.create_entity();
    assert!(Health::pack_get(&store, e).is_none(), "no data yet");
}

#[test]
fn pack_get_mut_writes_through_lanes() {
    let mut store = SmartStore::new();
    Health::pack_register(&mut store);
    let e: Entity = store.create_entity();

    let hp = Health {
        current: 10.0,
        max: 10.0,
    };
    hp.pack_insert(&mut store, e);

    {
        let mut packed = Health::pack_get_mut(&mut store, e).expect("packed mut");
        // PackMut exposes accessor METHODS (not fields): packed.current() -> &mut f32.
        *packed.current() = 5.0;
    }

    let back = Health::pack_get(&store, e).expect("still there");
    assert_eq!(back.current, 5.0);
    assert_eq!(back.max, 10.0);
}
