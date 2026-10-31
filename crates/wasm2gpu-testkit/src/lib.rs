//! Shared E2E harness for the wasm2gpu lifts: build the guest wasm,
//! bootstrap a wgpu device, plumb buffers, and pin the lowered kernels.
//!
//! The kit was split out of `wasm2gpu-shader`'s test tree so `render`'s
//! A/B tests (`cull_prod`, `fog_vs`) can share it through a layer-legal
//! edge: the kit lives in `foundation`, `render` (domain) may depend on
//! it, and the `foundation` tests keep their harness without reaching
//! upward into `render`.

use std::process::Command;

/// The guest's import module and entry symbols — re-exported so
/// downstream tests need only this crate for symbol lookups.
pub use ornis_gpu_abi::symbols;

pub mod cull;
pub mod compute_lift;

/// Build the guest crate (`ornis-wasm2gpu-shader`) for
/// `wasm32-unknown-unknown` — release, size-optimized, no unwind — and
/// return the `.wasm` bytes.
pub fn compile_wasm() -> Vec<u8> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|path| path.parent())
        .expect("workspace");
    let target_dir = workspace.join("target").join("wasm2gpu-e2e");
    let mut cmd = Command::new(cargo);
    cmd.current_dir(workspace)
        .args([
            "build",
            "-p",
            "ornis-wasm2gpu-shader",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--target-dir",
        ])
        .arg(&target_dir)
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .env(
            "RUSTFLAGS",
            "-C opt-level=z -C codegen-units=1 -C panic=abort",
        );
    let built = cmd.output().expect("spawn cargo");
    if !built.status.success() {
        panic!(
            "wasm32 build failed\n{}",
            String::from_utf8_lossy(&built.stderr)
        );
    }
    let path = target_dir.join("wasm32-unknown-unknown/release/ornis_wasm2gpu_shader.wasm");
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {path:?}: {err}"))
}

/// The (device, queue) pair of the preferred adapter, or `None` on
/// hosts where no adapter answers (tests skip unless the CI gate forces
/// an adapter via `ORNIS_REQUIRE_GPU`).
pub fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("wasm2gpu e2e"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .ok()
}

/// A buffer with exactly `usage` and no mapping at creation.
pub fn gpu_buffer(
    device: &wgpu::Device,
    label: &str,
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

/// A storage buffer that passes can write and copies can read back.
pub fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    gpu_buffer(
        device,
        label,
        size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

/// One bind-group entry for `buffer` at `binding`.
pub fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Map a buffer for reading and return its bytes (staging copy, unmap).
pub fn map_bytes(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: None,
    });
    rx.recv().expect("map callback").expect("map buffer");
    let bytes = slice.get_mapped_range().expect("mapped range").to_vec();
    buffer.unmap();
    bytes
}

/// Pack a little-endian `u32` at `offset`.
pub fn put_u32(bytes: &mut [u8], offset: u32, value: u32) {
    let start = offset as usize;
    bytes[start..start + 4].copy_from_slice(&value.to_le_bytes());
}

/// Pack a little-endian `f32` at `offset`.
pub fn put_f32(bytes: &mut [u8], offset: u32, value: f32) {
    put_u32(bytes, offset, value.to_bits());
}
