//! RAT-34 regression tests for the `#[kernel]` compute path: `as`-cast
//! lowering, the tail-dispatch bounds guard, and the pipeline bind group
//! layout. String assertions pin the generated WGSL; the device test runs
//! a 100-element dispatch (not a multiple of 64) and compares every lane
//! against CPU `eval`, skipping gracefully without a wgpu adapter.

use ornis_macros::{gpu_pipeline, kernel};

/// int↔float casts: truncation on the CPU must survive into WGSL as
/// conversion constructors.
#[kernel]
fn cast_add(a: f32, b: f32) -> f32 {
    let x = a as u32;
    let y = b as i32;
    (x as f32) + (y as f32)
}

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

#[test]
fn casts_lower_to_wgsl_constructors() {
    let helper = cast_add::wgsl_source();
    assert!(helper.contains("u32(a)"), "{helper}");
    assert!(helper.contains("i32(b)"), "{helper}");
    assert!(helper.contains("f32(x)"), "{helper}");
    assert!(helper.contains("f32(y)"), "{helper}");
    validate_module("cast_add helper", helper);
}

#[test]
fn cast_eval_matches_truncation_semantics() {
    // CPU `as` truncates toward zero; the WGSL constructors must compute
    // the same values — live-checked on-device in `cast_roundtrip_device`.
    assert_eq!(cast_add::eval(1.7, 2.3), 3.0);
    assert_eq!(cast_add::eval(0.2, 0.9), 0.0);
    // Negative float → i32 truncates toward zero on both sides (in range).
    assert_eq!(cast_add::eval(1.7, -2.3), -1.0);
}

/// Elementwise cast executed on-device: the full-shader compute path is
/// the only kernel-DSL body shape that can run casts per lane (legacy
/// compute bodies name whole buffers as scalars — see above), so the
/// truncation-constructor parity gets its live GPU check here.
#[gpu_pipeline(
    workgroup_size = 64,
    storage(cast_in: [f32; 64]),
    storage(cast_out: [f32; 64], read_write),
    builtin(cast_id: global_invocation_id),
)]
fn cast_roundtrip() {
    let i = cast_id.x;
    cast_out[i] = (cast_in[i] as u32) as f32;
}

#[test]
fn cast_roundtrip_source_uses_constructors() {
    let src = cast_roundtrip::wgsl_source();
    // Source parens are preserved verbatim by the writer.
    assert!(src.contains("f32((u32(cast_in[i])))"), "{src}");
    validate_module("cast_roundtrip compute", src);
}

/// Every lane must equal the CPU `(x as u32) as f32`. Skips without a
/// wgpu adapter.
#[test]
fn cast_roundtrip_device() {
    const N: usize = 64;
    let input: Vec<f32> = (0..N).map(|i| i as f32 + 0.7).collect();
    let want: Vec<f32> = input.iter().map(|&x| (x as u32) as f32).collect();

    let Some((device, queue)) = create_test_device() else {
        eprintln!("cast_roundtrip_device: no wgpu device — skipped");
        return;
    };
    let got = run_cast_roundtrip(&device, &queue, &input);

    assert_eq!(got.len(), N);
    for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "lane {i}: gpu {got} != cpu {want}"
        );
    }
}

/// Constant body: the only legacy-compute shape whose tail expression is
/// valid WGSL as-is (params lower to storage arrays while the tail names
/// them as scalars — a pre-existing path limitation, out of scope here).
/// Unused params keep their bindings so the layout stays multi-entry.
#[kernel]
fn const_two(_a: f32, _b: f32) -> f32 {
    2.0
}

#[test]
fn compute_source_has_bounds_guard_and_bindings() {
    // Guard shape is identical for every kernel; assert it on the cast
    // kernel too, but validate naga only on the valid module.
    let cast_src = cast_add::wgsl_compute_source();
    assert!(
        cast_src.contains("if (i >= arrayLength(&output)) { return; }"),
        "tail dispatch must be guarded, got: {cast_src}"
    );
    let src = const_two::wgsl_compute_source();
    assert!(
        src.contains("if (i >= arrayLength(&output)) { return; }"),
        "tail dispatch must be guarded, got: {src}"
    );
    // Two inputs plus the output buffer.
    for binding in 0..3 {
        assert!(
            src.contains(&format!("@binding({binding})")),
            "missing binding {binding}, got: {src}"
        );
    }
    validate_module("const_two compute", src);
}

fn f32_list(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn read_f32_list(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Full round trip on a length that is not a multiple of 64: the tail
/// workgroup (lanes 100..128) hits the bounds guard, and every live lane
/// must match CPU `eval`. Skips without a wgpu adapter.
#[test]
fn compute_round_trip_on_non_multiple_of_64() {
    const N: usize = 100;
    let a: Vec<f32> = (0..N).map(|i| i as f32 + 0.7).collect();
    let b: Vec<f32> = (0..N).map(|i| i as f32 * 0.5 + 0.2).collect();
    let want: Vec<f32> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| const_two::eval(x, y))
        .collect();

    let Some((device, queue)) = create_test_device() else {
        eprintln!("compute_round_trip_on_non_multiple_of_64: no wgpu device — skipped");
        return;
    };
    let got = run_const_two(&device, &queue, &a, &b);

    assert_eq!(got.len(), N);
    for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "lane {i}: gpu {got} != cpu {want}"
        );
    }
}

/// Test device, or `None` without an adapter (machines without GPU
/// drivers skip instead of failing) — unless `ORNIS_REQUIRE_GPU` is set
/// (CI `test` shard), where a missing adapter panics so silently-skipped
/// GPU coverage cannot pass unnoticed.
fn create_test_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let require_gpu = std::env::var("ORNIS_REQUIRE_GPU").is_ok();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let Some(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()
    else {
        if require_gpu {
            panic!("ORNIS_REQUIRE_GPU is set but no wgpu adapter is available");
        }
        return None;
    };
    let Some((device, queue)) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ornis-macros kernel test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .ok()
    else {
        if require_gpu {
            panic!("ORNIS_REQUIRE_GPU is set but no wgpu device could be created");
        }
        return None;
    };
    Some((device, queue))
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
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

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One compute dispatch plus output readback: shared by the
/// `const_two` and `cast_roundtrip` round trips so the two call sites
/// cannot drift apart.
struct RoundTrip<'a> {
    label: &'static str,
    pipeline: &'a wgpu::ComputePipeline,
    bind_group: &'a wgpu::BindGroup,
    out_buf: &'a wgpu::Buffer,
    staging: &'a wgpu::Buffer,
    size: u64,
    workgroups: u32,
}

fn dispatch_and_readback(device: &wgpu::Device, queue: &wgpu::Queue, job: &RoundTrip) -> Vec<f32> {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(job.label),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(job.label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(job.pipeline);
        cpass.set_bind_group(0, job.bind_group, &[]);
        cpass.dispatch_workgroups(job.workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(job.out_buf, 0, job.staging, 0, job.size);
    queue.submit([encoder.finish()]);
    read_staging(device, job.staging)
}

/// Dispatch `const_two` over `a`/`b` and read back the output lanes.
fn run_const_two(device: &wgpu::Device, queue: &wgpu::Queue, a: &[f32], b: &[f32]) -> Vec<f32> {
    let size = (a.len() * 4) as u64;
    let a_buf = storage_buffer(
        device,
        "a",
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let b_buf = storage_buffer(
        device,
        "b",
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let out_buf = storage_buffer(
        device,
        "out",
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let staging = storage_buffer(
        device,
        "staging",
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&a_buf, 0, &f32_list(a));
    queue.write_buffer(&b_buf, 0, &f32_list(b));

    // `create_pipeline` must succeed with its storage declarations: an
    // empty bind group layout would fail pipeline creation here.
    let bgl = const_two::create_bind_group_layout(device);
    let pipeline = const_two::create_pipeline(device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("const_two"),
        layout: &bgl,
        entries: &[
            whole_buffer_entry(0, &a_buf),
            whole_buffer_entry(1, &b_buf),
            whole_buffer_entry(2, &out_buf),
        ],
    });

    dispatch_and_readback(
        device,
        queue,
        &RoundTrip {
            label: "const_two",
            pipeline: &pipeline,
            bind_group: &bind_group,
            out_buf: &out_buf,
            staging: &staging,
            size,
            workgroups: a.len().div_ceil(64) as u32,
        },
    )
}

/// Block on a staging-buffer readback and decode it as `f32` lanes.
fn read_staging(device: &wgpu::Device, staging: &wgpu::Buffer) -> Vec<f32> {
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .ok();
    rx.recv()
        .expect("map callback must fire")
        .expect("staging map must succeed");
    let got = {
        let view = slice
            .get_mapped_range()
            .expect("staging range must be available");
        read_f32_list(&view)
    };
    staging.unmap();
    got
}

/// Dispatch `cast_roundtrip` over `input` and read back the output lanes.
/// The full-shader mode exposes no `create_pipeline` helper, so the
/// two-entry layout is built here against the declared bindings.
fn run_cast_roundtrip(device: &wgpu::Device, queue: &wgpu::Queue, input: &[f32]) -> Vec<f32> {
    let size = (input.len() * 4) as u64;
    let in_buf = storage_buffer(
        device,
        "cast_in",
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let out_buf = storage_buffer(
        device,
        "cast_out",
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let staging = storage_buffer(
        device,
        "staging",
        size,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    queue.write_buffer(&in_buf, 0, &f32_list(input));

    let shader = cast_roundtrip::create_shader_module(device);
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("cast_roundtrip"),
        entries: &[
            storage_layout_entry(0, true),
            storage_layout_entry(1, false),
        ],
    });
    // Read-only input, read-write output — mirrors the attribute.
    assert!(cast_roundtrip::wgsl_source().contains("var<storage> cast_in"));
    assert!(cast_roundtrip::wgsl_source().contains("var<storage, read_write> cast_out"));
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cast_roundtrip"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cast_roundtrip"),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("cast_roundtrip"),
        layout: &bgl,
        entries: &[
            whole_buffer_entry(0, &in_buf),
            whole_buffer_entry(1, &out_buf),
        ],
    });

    dispatch_and_readback(
        device,
        queue,
        &RoundTrip {
            label: "cast_roundtrip",
            pipeline: &pipeline,
            bind_group: &bind_group,
            out_buf: &out_buf,
            staging: &staging,
            size,
            workgroups: input.len().div_ceil(64) as u32,
        },
    )
}
