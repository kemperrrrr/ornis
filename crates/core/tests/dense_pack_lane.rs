//! Dense lane field slices: `repr(transparent)` wrappers alias `[Field]`.

use ornis_core::{DensePackLane, Pack, Resources, SmartStore};

/// One-lane bundle so the slice view can write through the wrapper column.
#[derive(Clone, Debug, PartialEq, ornis_macros::Pack)]
struct Dot {
    x: f32,
}

/// A write through [`DensePackLane::as_mut_slice`] is visible as the packed field.
#[test]
fn dense_lane_field_slices_alias_column() {
    let mut resources = Resources::new();
    let mut store = SmartStore::new();
    Dot::pack_register(&mut store);
    let a = store.create_entity();
    let b = store.create_entity();
    Dot { x: 1.0 }.pack_insert(&mut store, a);
    Dot { x: 3.0 }.pack_insert(&mut store, b);
    resources.insert(store);

    let store = resources.get::<SmartStore>().expect("store");
    {
        let mut xs = store.write_lane::<Dot__x__PackLane>().expect("x lane");
        let slice = DensePackLane::as_mut_slice(&mut xs.data);
        assert_eq!(slice, &[1.0, 3.0]);
        slice[0] = 9.0;
        slice[1] = 8.0;
    }
    {
        let xs = store.read_lane::<Dot__x__PackLane>().expect("x lane");
        assert_eq!(DensePackLane::as_slice(&xs.data), &[9.0, 8.0]);
        assert_eq!(xs.get(a), Some(&Dot__x__PackLane(9.0)));
        assert_eq!(xs.get(b), Some(&Dot__x__PackLane(8.0)));
    }
    assert_eq!(Dot::pack_get(store, a).unwrap().x, 9.0);
    assert_eq!(Dot::pack_get(store, b).unwrap().x, 8.0);
}
