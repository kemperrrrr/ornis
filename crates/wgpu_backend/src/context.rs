//! `wgpu` instance/adapter/device initialization.
//!
//! [`WgpuContext`] picks a high-performance adapter across all backends and
//! requests a device with downlevel-compatible limits, so the same context
//! works on native and web targets. Construction panics when no compatible
//! adapter exists.

/// Fully initialized `wgpu` handles for a high-performance adapter.
///
/// Construction picks the best available backend and requests a device with
/// downlevel-compatible limits, so the same context works on native and web
/// targets.
pub struct WgpuContext {
    /// Entry point owning the underlying graphics backend connections.
    pub instance: wgpu::Instance,
    /// Handle to the selected physical GPU.
    pub adapter: wgpu::Adapter,
    /// Logical device used to create all GPU resources.
    pub device: wgpu::Device,
    /// Submission queue paired with [`device`](Self::device).
    pub queue: wgpu::Queue,
}

impl WgpuContext {
    /// Blocking wrapper around [`new`](Self::new) for synchronous call sites.
    ///
    /// Panics if no compatible GPU adapter is found.
    pub fn new_blocking() -> Self {
        pollster::block_on(Self::new())
    }

    /// Requests a high-performance adapter and a device with downlevel limits.
    ///
    /// Panics if no compatible GPU adapter is found or device creation fails.
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                apply_limit_buckets: false,
            })
            .await
            .expect("no compatible GPU adapter found");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ornis device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create wgpu device");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
}
