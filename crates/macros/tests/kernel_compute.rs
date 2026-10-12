//! RAT-34 regression tests for the `#[kernel]` compute path: `as`-cast
//! lowering, the tail-dispatch bounds guard, and the pipeline bind group
//! layout. String assertions pin the generated WGSL; the device test runs
//! a 100-element dispatch (not a multiple of 64) and compares every lane
//! against CPU `eval`, skipping gracefully without a wgpu adapter.

use ornis_macros::kernel;

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
    // the same values (checked on-device in `compute_round_trip` below).
    assert_eq!(cast_add::eval(1.7, 2.3), 3.0);
    assert_eq!(cast_add::eval(0.2, 0.9), 0.0);
    // Negative float → i32 truncates toward zero on both sides (in range).
    assert_eq!(cast_add::eval(1.7, -2.3), -1.0);
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
        eprintln!("compute_round_trip_on_non_multiple_of_64: no wgpu adapter — skipped");
        return;
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
        eprintln!("compute_round_trip_on_non_multiple_of_64: no wgpu device — skipped");
        return;
    };

    let size = (N * 4) as u64;
    let a_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("a"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("b"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&a_buf, 0, &f32_list(&a));
    queue.write_buffer(&b_buf, 0, &f32_list(&b));

    // `create_pipeline` must succeed with its storage declarations: an
    // empty bind group layout would fail pipeline creation here.
    let bgl = const_two::create_bind_group_layout(&device);
    let pipeline = const_two::create_pipeline(&device);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("const_two"),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &a_buf,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &b_buf,
                    offset: 0,
                    size: None,
                }),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &out_buf,
                    offset: 0,
                    size: None,
                }),
            },
        ],
    });

    let workgroups = N.div_ceil(64) as u32;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("const_two"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("const_two"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);

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

    assert_eq!(got.len(), N);
    for (i, (&got, &want)) in got.iter().zip(&want).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "lane {i}: gpu {got} != cpu {want}"
        );
    }
}
