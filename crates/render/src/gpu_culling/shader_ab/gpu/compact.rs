// qual:allow(srp, max_independent_clusters=4) reason: "The benchmark separates pipeline setup, dispatch verification, buffer packing, and readback; keep this focused A/B harness together."
//! Timings for cull_compact against production WGSL. The duplicated float/word
//! source views are part of measured shader traffic.

use super::super::scene::MATERIAL_AT;
use super::super::scene::OBJECT_BYTES;
use super::super::scene::Scene;
use super::dispatch;
use super::pipeline;
use super::readback;
use super::session::Session;
use super::upload::make;
use super::upload::resource;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const UNIFORM_STRIDE: usize = 16;
const COUNT_LANES: usize = 3;
const SOURCE_VIEW_COUNT: usize = 2;
const ARGS_BYTES: u64 = super::super::super::INDIRECT_ARGS_WORD_COUNT as u64
    * super::super::super::INDIRECT_ARGS_WORD_BYTES;
const MAX_EXACT_F32_COUNT: usize = 16_777_216;
const PLANE_BINDING: u32 = 0;
const SOURCE_FLOAT_BINDING: u32 = 1;
const DESTINATION_BINDING: u32 = 2;
const ARGS_BINDING: u32 = 3;
const SOURCE_WORDS_BINDING: u32 = 4;
const COUNTS_BINDING: u32 = 5;

struct CompactSet {
    bind: wgpu::BindGroup,
    args: wgpu::Buffer,
    destination: wgpu::Buffer,
    _source_float: wgpu::Buffer,
    _source_words: wgpu::Buffer,
    _planes: wgpu::Buffer,
    _counts: wgpu::Buffer,
}

/// Time production and compact-guest dispatches on every shared scene.
pub(super) fn time_scenes(session: &Session, lowered: &str) -> String {
    let production = pipeline::build(
        &session.device,
        pipeline::prod(),
        pipeline::PROD_ENTRY,
        true,
    );
    let compact_layout = compact_layout(session);
    let compact = pipeline::compile_with(
        &session.device,
        lowered,
        pipeline::FLAG_ENTRY,
        compact_layout,
    );
    let mut report =
        String::from("\ncompact ABI A/B — same 52-word records; duplicated source view included\n");
    super::super::scene::scenes().iter().for_each(|scene| {
        report.push_str(&time_scene(session, scene, &production, &compact, lowered));
        report.push('\n');
    });
    report
}

fn time_scene(
    session: &Session,
    scene: &Scene,
    production: &pipeline::Pipe,
    compact: &pipeline::Pipe,
    lowered: &str,
) -> String {
    let production_set = super::upload::upload_prod(session, scene, production);
    let compact_set = upload_compact(session, scene, compact);
    verify_outputs(
        session,
        scene,
        production,
        compact,
        &production_set,
        &compact_set,
        lowered,
    );
    let production_groups = crate::gpu_culling::cull_workgroup_count(scene.spheres.len() as u32);
    let compact_groups = pipeline::groups(scene.spheres.len(), lowered);
    let production_samples = super::clock::measure(|| {
        dispatch::submit(
            session,
            production,
            dispatch::prod_bind(&production_set),
            production_groups,
            Some(dispatch::prod_args(&production_set)),
        )
    });
    let compact_samples = super::clock::measure(|| {
        dispatch::submit(
            session,
            compact,
            &compact_set.bind,
            compact_groups,
            Some(&compact_set.args),
        )
    });
    super::clock::compact_scene_report(
        scene,
        &production_samples,
        &compact_samples,
        SOURCE_VIEW_COUNT * scene.spheres.len() * OBJECT_BYTES,
    )
}

// qual:allow(srp, max_parameters=7) reason: "The parity gate compares both independently configured pipelines and their live resource sets before any timings are trusted."
fn verify_outputs(
    session: &Session,
    scene: &Scene,
    production: &pipeline::Pipe,
    compact: &pipeline::Pipe,
    production_set: &super::dispatch::ProdSet,
    compact_set: &CompactSet,
    lowered: &str,
) {
    let production_groups = crate::gpu_culling::cull_workgroup_count(scene.spheres.len() as u32);
    let compact_groups = pipeline::groups(scene.spheres.len(), lowered);
    let _ = dispatch::submit(
        session,
        production,
        dispatch::prod_bind(production_set),
        production_groups,
        Some(dispatch::prod_args(production_set)),
    );
    let production_ids = readback::prod_ids(session, production_set, scene.spheres.len());
    let _ = dispatch::submit(
        session,
        compact,
        &compact_set.bind,
        compact_groups,
        Some(&compact_set.args),
    );
    let compact_ids = compact_ids(session, compact_set, scene.spheres.len());
    let expected = super::super::scene::visible_ids(scene);
    if production_ids != expected || compact_ids != expected {
        eprintln!(
            "cull_compact A/B visible-set mismatch on {}: cpu {} production {} compact {}",
            super::super::scene::kind_name(scene.kind),
            expected.len(),
            production_ids.len(),
            compact_ids.len(),
        );
        std::process::exit(1);
    }
}

fn upload_compact(session: &Session, scene: &Scene, pipe: &pipeline::Pipe) -> CompactSet {
    let object_bytes = super::super::scene::pack_objects(&scene.spheres);
    let source_size = object_bytes.len() as u64;
    let storage_src = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let source_float = make(
        session,
        "compact source float view",
        source_size,
        storage_src,
    );
    let source_words = make(
        session,
        "compact source word view",
        source_size,
        storage_src,
    );
    let destination = make(
        session,
        "compact destination",
        source_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let args = make(
        session,
        "compact indirect args",
        ARGS_BYTES,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    let planes_bytes = super::super::scene::pack_flag_planes(&scene.planes);
    let planes = make(
        session,
        "compact planes",
        planes_bytes.len() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let counts_bytes = pack_counts(scene.spheres.len());
    let counts = make(
        session,
        "compact counts",
        counts_bytes.len() as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    session.queue.write_buffer(&source_float, 0, &object_bytes);
    session.queue.write_buffer(&source_words, 0, &object_bytes);
    session.queue.write_buffer(&planes, 0, &planes_bytes);
    session.queue.write_buffer(&counts, 0, &counts_bytes);
    let initial_args = [0u8; ARGS_BYTES as usize];
    session.queue.write_buffer(&args, 0, &initial_args);
    let bind = session
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("compact cull bindings"),
            layout: &pipe.layout,
            entries: &[
                resource(PLANE_BINDING, &planes),
                resource(SOURCE_FLOAT_BINDING, &source_float),
                resource(DESTINATION_BINDING, &destination),
                resource(ARGS_BINDING, &args),
                resource(SOURCE_WORDS_BINDING, &source_words),
                resource(COUNTS_BINDING, &counts),
            ],
        });
    CompactSet {
        bind,
        args,
        destination,
        _source_float: source_float,
        _source_words: source_words,
        _planes: planes,
        _counts: counts,
    }
}

fn compact_layout(session: &Session) -> wgpu::BindGroupLayout {
    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    session
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compact cull layout"),
            entries: &[
                pipeline::entry(PLANE_BINDING, wgpu::BufferBindingType::Uniform),
                pipeline::entry(SOURCE_FLOAT_BINDING, storage),
                pipeline::entry(DESTINATION_BINDING, storage),
                pipeline::entry(ARGS_BINDING, storage),
                pipeline::entry(SOURCE_WORDS_BINDING, storage),
                pipeline::entry(COUNTS_BINDING, wgpu::BufferBindingType::Uniform),
            ],
        })
}

fn pack_counts(count: usize) -> Vec<u8> {
    assert!(
        count <= MAX_EXACT_F32_COUNT,
        "active/source/destination counts must be exactly representable as f32"
    );
    let mut bytes = vec![0u8; COUNT_LANES * UNIFORM_STRIDE];
    for index in 0..COUNT_LANES {
        let value = count as f32;
        let offset = index * UNIFORM_STRIDE;
        bytes[offset..offset + F32_BYTES].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn compact_ids(session: &Session, set: &CompactSet, capacity: usize) -> Vec<u32> {
    let args = readback::read_exact(session, &set.args, ARGS_BYTES);
    let count = readback::load_u32(&args, super::dispatch::instance_at());
    let take = (count as usize).min(capacity);
    if take == 0 {
        return Vec::new();
    }
    let bytes = readback::read_exact(session, &set.destination, (take * OBJECT_BYTES) as u64);
    let mut ids = bytes
        .chunks_exact(OBJECT_BYTES)
        .map(|record| readback::load_u32(record, MATERIAL_AT))
        .collect::<Vec<_>>();
    ids.sort_unstable();
    ids
}
