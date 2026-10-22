//! Shared GPU-test helpers for the macros acceptance suites
//! (`kernel_compute`, `pack_gpu_lanes`): one copy each so the two
//! files cannot drift into duplicate fragments.

/// Test device, or `None` without an adapter (machines without GPU
/// drivers skip instead of failing) — unless `ORNIS_REQUIRE_GPU` is set
/// (CI `test` shard), where a missing adapter panics so silently-skipped
/// GPU coverage cannot pass unnoticed.
pub(crate) fn create_test_device(label: &'static str) -> Option<(wgpu::Device, wgpu::Queue)> {
    // `= "1"`, not `is_ok()`: an explicit `ORNIS_REQUIRE_GPU=0` must
    // keep the skip path.
    let require_gpu = matches!(std::env::var("ORNIS_REQUIRE_GPU").as_deref(), Ok("1"));
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
            label: Some(label),
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

/// Map-read a staging buffer back to its f32 payload: install the
/// callback, poll until it fires, then decode the mapped range.
pub(crate) fn read_staging(device: &wgpu::Device, staging: &wgpu::Buffer) -> Vec<f32> {
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

fn read_f32_list(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
