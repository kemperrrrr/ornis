//! Shared GPU-test helpers for the renderer split: NDC math, WGSL validation, adapter acquisition, and light probes.
use super::*;

pub(super) fn clip_of(vp: [[f32; 4]; 4], p: [f32; 3]) -> glam::Vec4 {
    glam::Mat4::from_cols_array_2d(&vp) * glam::Vec4::new(p[0], p[1], p[2], 1.0)
}

pub(super) fn ndc_of(vp: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
    let c = clip_of(vp, p);
    [c.x / c.w, c.y / c.w, c.z / c.w]
}

pub(super) fn assert_valid_wgsl(name: &str, source: &str) {
    let module =
        naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{name} must parse: {e}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
}

pub(super) fn dir_probe(direction: [f32; 3], shadow: ornis_assets::scene::ShadowCast) -> LightDesc {
    LightDesc::Directional {
        direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::from_array(direction))
            .expect("non-zero direction"),
        intensity: 1.0,
        color: [1.0, 1.0, 1.0],
        shadow,
    }
}

/// Adapter handle, or `None` on headless CI without a GPU.
pub(super) fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
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
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    })
}
