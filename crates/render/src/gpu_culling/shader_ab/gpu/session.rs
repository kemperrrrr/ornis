//! Headless device for the A/B. Software adapters are labeled in the header
//! so a lavapipe run cannot be read as a discrete-GPU result.

use super::clock::REPEATS;
use super::clock::WARMUP;

/// Timestamp query slots: pass begin and pass end.
pub(super) const QUERY_SLOTS: u32 = 2;
/// Query index written at the start of the compute pass.
pub(super) const QUERY_BEGIN: u32 = 0;
/// Query index written at the end of the compute pass.
pub(super) const QUERY_END: u32 = 1;

pub(super) struct Session {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    pub(super) header: String,
    pub(super) queries: Option<wgpu::QuerySet>,
    pub(super) resolve: wgpu::Buffer,
    pub(super) period_ns: f32,
}

pub(super) fn open() -> Option<Session> {
    match pollster::block_on(request()) {
        Some(session) => Some(session),
        None => absent(),
    }
}

fn absent() -> Option<Session> {
    if crate::gpu_required() {
        eprintln!("ORNIS_REQUIRE_GPU is set and no wgpu adapter was found");
        std::process::exit(1);
    }
    eprintln!("wasm2gpu cull A/B: no wgpu adapter — GPU timings skipped");
    None
}

async fn request() -> Option<Session> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok()?;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("wasm2gpu cull ab"),
            required_features: timestamps(&adapter),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .ok()?;
    Some(session(adapter, device, queue))
}

fn timestamps(adapter: &wgpu::Adapter) -> wgpu::Features {
    wgpu::Features::TIMESTAMP_QUERY & adapter.features()
}

fn session(adapter: wgpu::Adapter, device: wgpu::Device, queue: wgpu::Queue) -> Session {
    let info = adapter.get_info();
    let queries = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| query_set(&device));
    Session {
        header: header(&info, queries.is_some(), queue.get_timestamp_period()),
        period_ns: queue.get_timestamp_period(),
        resolve: resolve_buffer(&device),
        queries,
        device,
        queue,
    }
}

fn query_set(device: &wgpu::Device) -> wgpu::QuerySet {
    device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("cull ab stamps"),
        ty: wgpu::QueryType::Timestamp,
        count: QUERY_SLOTS,
    })
}

fn resolve_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cull ab stamps"),
        size: stamp_span(),
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn stamp_span() -> u64 {
    u64::from(QUERY_SLOTS) * std::mem::size_of::<u64>() as u64
}

fn header(info: &wgpu::AdapterInfo, timestamps: bool, period: f32) -> String {
    format!(
        "adapter: {name}\n\
         backend: {backend:?}\n\
         device_type: {kind:?}\n\
         driver: {driver} {driver_info}\n\
         software: {software}\n\
         timestamps: {timestamps} (period {period} ns)\n\
         warmup: {WARMUP}  repeats: {REPEATS}\n\
         GPU column: pass timestamps, median [q1–q3] iqr, microseconds. The production instance-count reset is outside this window.\n\
         wall column: queue write of that reset (production only) plus submit and the blocking poll, same hinges. Stamp readback is outside the window.\n\
         software=yes means a CPU Vulkan device (lavapipe/llvmpipe class). Those times are not discrete-GPU costs.",
        name = info.name,
        backend = info.backend,
        kind = info.device_type,
        driver = info.driver,
        driver_info = info.driver_info,
        software = software_label(is_software(info)),
    )
}

fn is_software(info: &wgpu::AdapterInfo) -> bool {
    info.device_type == wgpu::DeviceType::Cpu || software_name(&info.name)
}

fn software_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("llvmpipe") || lower.contains("lavapipe") || lower.contains("swiftshader")
}

fn software_label(software: bool) -> &'static str {
    if software {
        "yes — CPU adapter, not a discrete GPU"
    } else {
        "no"
    }
}
