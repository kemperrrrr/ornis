//! `#[pack(gpu)]` / `#[gpu_pipeline(pack(T))]` acceptance: lane
//! descriptors, aligned upload, WGSL stitching, layout contract, and
//! a device-gated dispatch readback parity check. Compile-error paths
//! live in `tests/ui` (trybuild).

#![allow(dead_code)]

use ornis_core::{Entity, Pack, PackGpu, SmartStore};
use ornis_macros::{Pack as PackDerive, gpu_pipeline};

mod common;
use common::{create_test_device, read_staging};

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

// ── gpu_pipeline(pack(T)) stitching ───────────────────────────────────────

/// Explicit read-write output at binding 0; pack lanes follow. One
/// element per workgroup (`workgroup_size = 1`, `gid.x` = group index).
#[gpu_pipeline(
    workgroup_size = 1,
    storage(out: [f32; 64], read_write),
    pack(Dense),
    builtin(gid: workgroup_id),
)]
fn scale_dense() {
    let i = gid.x;
    out[i] = x[i] * 2.0;
}

#[gpu_pipeline(
    workgroup_size = 4,
    storage(counter: [u32; 64], read_write),
    pack(Dense),
    pack(Sparse),
    builtin(gid: workgroup_id),
)]
fn multi_pack() {
    let i = gid.x;
    counter[i] = id[0] + tag[0];
}

/// Pack-only pipeline (no explicit bindings): lanes start at 0.
#[gpu_pipeline(workgroup_size = 1, pack(Dense))]
fn lanes_only() {}

#[test]
fn pack_pipeline_stitches_lanes_after_base_binding() {
    let src = scale_dense::wgsl_source();
    // Explicit binding count is 1 → lanes start at 1.
    assert!(
        src.contains("@group(0) @binding(0) var<storage, read_write> out: array<f32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(1) var<storage, read> x: array<f32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(2) var<storage, read> id: array<u32>;"),
        "{src}"
    );
    assert!(src.contains("@compute @workgroup_size(1)"), "{src}");
    validate_module("scale_dense", &src);
}

#[test]
fn multi_pack_continues_binding_numbers_across_bundles() {
    let src = multi_pack::wgsl_source();
    assert!(
        src.contains("@group(0) @binding(0) var<storage, read_write> counter: array<u32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(1) var<storage, read> x: array<f32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(2) var<storage, read> id: array<u32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(3) var<storage, read> pos: array<f32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(4) var<storage, read> tag: array<u32>;"),
        "{src}"
    );
    validate_module("multi_pack", &src);
}

#[test]
fn lanes_only_pipeline_starts_at_binding_zero_and_validates() {
    let src = lanes_only::wgsl_source();
    assert!(
        src.contains("@group(0) @binding(0) var<storage, read> x: array<f32>;"),
        "{src}"
    );
    assert!(
        src.contains("@group(0) @binding(1) var<storage, read> id: array<u32>;"),
        "{src}"
    );
    validate_module("lanes_only", &src);
}

/// Drift pin: the generated WGSL declaration must match
/// `PackLaneDesc::decl` exactly — the two descriptor types
/// (`ornis_core::PackGpuLane` vs `ornis_shader_lang::PackLaneDesc`)
/// are pinned together so they cannot diverge in spelling.
#[test]
fn pack_lane_decl_matches_shader_lang_descriptor() {
    let src = lanes_only::wgsl_source();
    let base = 0u32;
    for (i, lane) in Dense::pack_gpu_lanes().iter().enumerate() {
        let desc = ornis_shader_lang::PackLaneDesc {
            field: lane.field,
            wgsl_elem: lane.wgsl_elem,
            sparse: lane.sparse,
        };
        let expected = desc.decl(0, base + i as u32);
        assert!(
            src.contains(&expected),
            "generated pipeline must contain `{expected}`\nsource:\n{src}"
        );
    }
}

// ── duplicate-name panic ──────────────────────────────────────────────────

/// Explicit `id` storage at binding 0 collides with Dense's `id` lane —
/// undetectable at macro time (the derive runs in another expansion),
/// so the collision panics at `wgsl_source()` stitch.
#[gpu_pipeline(
    storage(id: [u32; 64], read_write),
    pack(Dense),
)]
fn collides_with_explicit() {}

#[test]
#[should_panic(expected = "duplicate WGSL storage name `id`")]
fn duplicate_lane_name_panics_at_stitch() {
    let _ = collides_with_explicit::wgsl_source();
}

// ── helpers ───────────────────────────────────────────────────────────────

fn validate_module(name: &str, source: &str) {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{name} must parse: {e}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name} must validate: {e:?}"));
}

fn whole_buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }),
    }
}

// ── device-gated: layout / pipeline / dispatch parity ─────────────────────

/// Depth-cube-array texture alongside pack lanes: `texture_depth_cube_array`
/// binds as `TextureViewDimension::CubeArray`, mirroring the WGSL type
/// (see `ornis_render::ResourceKind::bgl_ty`).
#[gpu_pipeline(
    workgroup_size = 1,
    texture(shadow_map: texture_depth_cube_array),
    pack(Dense),
)]
fn cube_array_with_lane() {}

/// Layout mismatch → runtime error, not UB: a bind group that omits
/// the pack-lane entries is rejected by wgpu validation instead of
/// silently reading garbage.
#[test]
fn missing_lane_bindings_are_a_validation_error_not_ub() {
    let Some((device, _queue)) = create_test_device("ornis-macros pack gpu test") else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };

    let mut store = SmartStore::new();
    Dense::pack_register(&mut store);
    let a = store.create_entity();
    Dense { x: 1.5, id: 10 }.pack_insert(&mut store, a);
    let pack_buffers = scale_dense::create_pack_buffers(&device, &store).expect("pack buffers");

    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: 64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let bgl = scale_dense::create_bind_group_layout(&device);

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Bindings 1 and 2 (the two pack lanes) are left out on purpose.
    let _missing = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[whole_buffer_entry(0, &out_buf)],
    });
    let err = pollster::block_on(scope.pop());
    assert!(
        err.is_some(),
        "a bind group without the pack-lane entries must fail wgpu validation"
    );
    drop(pack_buffers);
}

/// A cube-array view against the generated texture slot: the layout
/// entry must ask for `CubeArray` (the `texture_depth_cube_array`
/// spelling), or wgpu rejects the view dimension at bind-group
/// creation — a validation error, not UB.
#[test]
fn cube_array_texture_slot_accepts_a_cube_view() {
    let Some((device, _queue)) = create_test_device("ornis-macros pack gpu test") else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };

    let mut store = SmartStore::new();
    Dense::pack_register(&mut store);
    let a = store.create_entity();
    Dense { x: 1.5, id: 10 }.pack_insert(&mut store, a);
    let pack_buffers =
        cube_array_with_lane::create_pack_buffers(&device, &store).expect("pack buffers");

    // Cube maps are 6-layer 2D-array textures viewed as `Cube`.
    let shadow = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow cube array"),
        size: wgpu::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_view = shadow.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::CubeArray),
        base_array_layer: 0,
        array_layer_count: Some(6),
        ..Default::default()
    });

    let bgl = cube_array_with_lane::create_bind_group_layout(&device);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&shadow_view),
            },
            whole_buffer_entry(1, &pack_buffers[0]),
            whole_buffer_entry(2, &pack_buffers[1]),
        ],
    });
    let err = pollster::block_on(scope.pop());
    assert!(
        err.is_none(),
        "the cube-array slot must accept a CubeArray view: {err:?}"
    );
}

/// Dispatch the generated pipeline against real pack buffers and
/// compare the GPU readback of the explicit output to a CPU
/// evaluation of the same store.
#[test]
fn pack_pipeline_dispatch_matches_cpu_evaluation() {
    let Some((device, queue)) = create_test_device("ornis-macros pack gpu test") else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };

    let mut store = SmartStore::new();
    Dense::pack_register(&mut store);
    let a = store.create_entity();
    let b = store.create_entity();
    Dense { x: 1.5, id: 10 }.pack_insert(&mut store, a);
    Dense { x: 2.5, id: 20 }.pack_insert(&mut store, b);

    // Lane buffers straight from the generated serializer.
    let pack_buffers = scale_dense::create_pack_buffers(&device, &store).expect("pack buffers");
    assert_eq!(pack_buffers.len(), 2, "one buffer per lane");

    // Explicit output (binding 0) is not a pack lane — created here.
    let out_size = 64 * 4;
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: out_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = scale_dense::create_bind_group_layout(&device);
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            whole_buffer_entry(0, &out_buf),
            whole_buffer_entry(1, &pack_buffers[0]),
            whole_buffer_entry(2, &pack_buffers[1]),
        ],
    });

    // CPU reference: same store, same doubling, same ascending order.
    let cpu: Vec<f32> = Dense::pack_entities(&store)
        .iter()
        .map(|e| Dense::pack_get(&store, *e).expect("pack get").x * 2.0)
        .collect();

    let pipeline = scale_dense::create_pipeline(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pack gpu test"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        // `workgroup_size = 1`: one element per workgroup, `gid.x` =
        // group index — one group per packed entity.
        pass.dispatch_workgroups(cpu.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, out_size);
    queue.submit([encoder.finish()]);

    let data = read_staging(&device, &staging);

    assert_eq!(cpu, vec![3.0, 5.0], "CPU reference values");
    assert_eq!(data[0..cpu.len()], cpu[..], "GPU readback vs CPU");
}

/// An empty store (registered, zero entities) must still bind: every
/// column carries one zeroed element, because wgpu rejects zero-size
/// buffer bindings — an empty scene start is the normal case, not an
/// error.
#[test]
fn empty_store_lane_buffers_bind_without_validation_error() {
    let Some((device, _queue)) = create_test_device("ornis-macros pack gpu test") else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };

    let mut store = SmartStore::new();
    Dense::pack_register(&mut store);

    let upload = Dense::pack_gpu_upload(&store).expect("upload of registered empty store");
    assert_eq!(upload.len, 0, "no entities in the index space");
    assert_eq!(upload.columns[0].len(), 4, "one zeroed f32 element");
    assert_eq!(upload.columns[1].len(), 4, "one zeroed u32 element");

    let pack_buffers = lanes_only::create_pack_buffers(&device, &store).expect("pack buffers");
    assert_eq!(pack_buffers.len(), 2, "one buffer per lane");
    assert_eq!(pack_buffers[0].size(), 4, "no zero-size buffer created");

    let bgl = lanes_only::create_bind_group_layout(&device);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            whole_buffer_entry(0, &pack_buffers[0]),
            whole_buffer_entry(1, &pack_buffers[1]),
        ],
    });
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "empty-store lane buffers must bind: {err:?}");
}
