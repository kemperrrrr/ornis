//! `#[pack(gpu)]` acceptance: lane descriptors, aligned upload and the
//! shared-index-space layout contract. Compile-error paths live in
//! `tests/ui` (trybuild).

#![allow(dead_code)]

use ornis_core::{Entity, Pack, PackGpu, SmartStore};
use ornis_macros::Pack as PackDerive;

// ── fixtures ──────────────────────────────────────────────────────────────

/// Dense two-field pack with an f32 and a u32 column.
#[derive(Debug, Clone, PartialEq, PackDerive)]
#[pack(gpu)]
struct Dense {
    x: f32,
    id: u32,
}

/// Sparse pack: `Option<T>` field → sparse lane, zero-fill when absent.
#[derive(Debug, Clone, PartialEq, PackDerive)]
#[pack(gpu)]
struct Sparse {
    pos: f32,
    tag: Option<u32>,
}

/// Sparse-*first* pack: the shared index space must come from the
/// dense field, not from field order — a `tag: None` entity must keep
/// its slot in the `pos` column.
#[derive(Debug, Clone, PartialEq, PackDerive)]
#[pack(gpu)]
struct SparseFirst {
    tag: Option<u32>,
    pos: f32,
}

/// All-sparse pack: no dense anchor exists — the index space is the
/// sorted union of every lane.
#[derive(Debug, Clone, PartialEq, PackDerive)]
#[pack(gpu)]
struct AllSparse {
    a: Option<u32>,
    b: Option<f32>,
}

/// Plain (non-GPU) pack: must *not* implement PackGpu (opt-in only).
#[derive(Debug, Clone, PartialEq, PackDerive)]
struct CpuOnly {
    v: f32,
}

// ── lane descriptors ──────────────────────────────────────────────────────

#[test]
fn dense_lanes_follow_field_order_and_spelling() {
    let lanes = Dense::pack_gpu_lanes();
    assert_eq!(lanes.len(), 2, "one lane per field");
    assert_eq!(lanes[0].field, "x");
    assert_eq!(lanes[0].wgsl_elem, "f32");
    assert!(!lanes[0].sparse);
    assert_eq!(lanes[1].field, "id");
    assert_eq!(lanes[1].wgsl_elem, "u32");
    assert!(!lanes[1].sparse);
}

#[test]
fn option_field_is_marked_sparse() {
    let lanes = Sparse::pack_gpu_lanes();
    assert_eq!(lanes.len(), 2);
    assert!(!lanes[0].sparse, "dense field stays dense");
    assert!(lanes[1].sparse, "Option field is a sparse lane");
}

// ── upload alignment ──────────────────────────────────────────────────────

fn bytes_of<T: bytemuck::Pod>(v: &[T]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// A registered `Dense` store with three packed entities (x = 1/2/3,
/// id = 10/20/30) — the shared setup of the alignment tests.
fn dense_store_with_three() -> (SmartStore, Entity, Entity, Entity) {
    let mut store = SmartStore::new();
    Dense::pack_register(&mut store);
    let a = store.create_entity();
    let b = store.create_entity();
    let c = store.create_entity();
    Dense { x: 1.0, id: 10 }.pack_insert(&mut store, a);
    Dense { x: 2.0, id: 20 }.pack_insert(&mut store, b);
    Dense { x: 3.0, id: 30 }.pack_insert(&mut store, c);
    (store, a, b, c)
}

#[test]
fn upload_columns_align_on_shared_ascending_entity_order() {
    let (store, _, _, _) = dense_store_with_three();

    let upload = Dense::pack_gpu_upload(&store).expect("upload");
    assert_eq!(upload.len, 3);
    assert_eq!(upload.columns.len(), 2);
    assert_eq!(upload.columns[0], bytes_of(&[1.0f32, 2.0, 3.0]));
    assert_eq!(upload.columns[1], bytes_of(&[10u32, 20, 30]));
}

/// After a swap-remove churns the dense order, columns stay aligned on
/// the shared entity list (the whole point of the per-entity lookup).
#[test]
fn upload_survives_dense_order_churn() {
    let (store, _a, b, _c) = dense_store_with_three();

    // Remove the middle entity → dense order churns via swap-remove.
    store
        .write_lane::<Dense__x__PackLane>()
        .expect("x lane")
        .remove(b);
    store
        .write_lane::<Dense__id__PackLane>()
        .expect("id lane")
        .remove(b);

    let upload = Dense::pack_gpu_upload(&store).expect("upload");
    assert_eq!(upload.len, 2);
    // Remaining entities in ascending id order: a then c.
    assert_eq!(upload.columns[0], bytes_of(&[1.0f32, 3.0]));
    assert_eq!(upload.columns[1], bytes_of(&[10u32, 30]));
}

#[test]
fn sparse_lane_absent_entity_uploads_all_zero_element() {
    let mut store = SmartStore::new();
    Sparse::pack_register(&mut store);
    let a = store.create_entity();
    let b = store.create_entity();
    // Only `a` gets a bundle; `b` has no entry in the `tag` lane.
    Sparse {
        pos: 5.0,
        tag: Some(7),
    }
    .pack_insert(&mut store, a);

    // Insert `b`'s dense `pos` lane only, to desync the two lanes.
    store
        .write_lane::<Sparse__pos__PackLane>()
        .expect("pos")
        .insert(b, Sparse__pos__PackLane(9.0));

    let upload = Sparse::pack_gpu_upload(&store).expect("upload");
    assert_eq!(upload.len, 2);
    assert_eq!(upload.columns[0], bytes_of(&[5.0f32, 9.0]));
    // `b` has no `tag` entry → zeroed u32 placeholder.
    assert_eq!(upload.columns[1], bytes_of(&[7u32, 0]));
}

/// A leading `Option<T>` field must not shrink the shared index space:
/// the anchor is the first *non-sparse* lane, so the `tag: None`
/// entity keeps its slot in the dense column instead of vanishing
/// from every column.
#[test]
fn sparse_first_field_keeps_dense_entities_in_every_column() {
    let mut store = SmartStore::new();
    SparseFirst::pack_register(&mut store);
    let a = store.create_entity();
    let b = store.create_entity();
    SparseFirst {
        tag: Some(7),
        pos: 1.0,
    }
    .pack_insert(&mut store, a);
    SparseFirst {
        tag: None,
        pos: 9.0,
    }
    .pack_insert(&mut store, b);

    let upload = SparseFirst::pack_gpu_upload(&store).expect("upload");
    assert_eq!(upload.len, 2, "`tag: None` must not drop the entity");
    assert_eq!(upload.columns[0], bytes_of(&[7u32, 0]), "tag zero-fills b");
    assert_eq!(
        upload.columns[1],
        bytes_of(&[1.0f32, 9.0]),
        "pos keeps b's row"
    );
}

/// An all-sparse bundle has no dense anchor: the index space is the
/// sorted union of every lane. Lanes overlap (entity `c` is in both)
/// and insertions run outside id order, so dropping `sort_by_key` or
/// `dedup_by_key` in `index_space_tokens` fails this test.
#[test]
fn all_sparse_bundle_uses_sorted_union_index_space() {
    let mut store = SmartStore::new();
    AllSparse::pack_register(&mut store);
    let a = store.create_entity();
    let b = store.create_entity();
    let c = store.create_entity();
    // Insert out of id order; `c` lands in both lanes.
    AllSparse {
        a: Some(3),
        b: Some(3.5),
    }
    .pack_insert(&mut store, c);
    AllSparse {
        a: Some(1),
        b: None,
    }
    .pack_insert(&mut store, a);
    AllSparse {
        a: None,
        b: Some(2.5),
    }
    .pack_insert(&mut store, b);

    let upload = AllSparse::pack_gpu_upload(&store).expect("upload");
    assert_eq!(upload.len, 3, "sorted union of both lanes, deduped");
    assert_eq!(upload.columns[0], bytes_of(&[1u32, 0, 3]));
    assert_eq!(upload.columns[1], bytes_of(&[0.0f32, 2.5, 3.5]));
}
