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
///
/// `ORNIS_REQUIRE_GPU=1` (the CI `test` shard) panics instead of skipping, so
/// a missing adapter can never turn a GPU gate into a pass; an explicit
/// `ORNIS_REQUIRE_GPU=0` keeps the skip.
pub(super) fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let pair = pollster::block_on(async {
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
    });
    if pair.is_none() && require_gpu() {
        panic!("ORNIS_REQUIRE_GPU=1 but no wgpu adapter is available");
    }
    pair
}

/// Whether the harness must fail (rather than skip) when no adapter exists.
/// `= "1"`; an explicit `ORNIS_REQUIRE_GPU=0` keeps the skip.
fn require_gpu() -> bool {
    matches!(std::env::var("ORNIS_REQUIRE_GPU").as_deref(), Ok("1"))
}

/// Offscreen `Rgba8UnormSrgb` render target shared by the headless
/// frame gates (single-sample, readable back).
pub(super) fn offscreen_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test offscreen target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Reads an `Rgba8UnormSrgb` target back to tightly-packed rows.
pub(super) fn readback_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    const BPP: u32 = 4;
    const PAD_ROW: u32 = 256;
    let unpadded = width * BPP;
    let padded = unpadded.div_ceil(PAD_ROW) * PAD_ROW;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test readback"),
        size: u64::from(padded * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test readback copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv()
        .expect("readback callback")
        .expect("readback maps");
    let view = slice.get_mapped_range().expect("mapped range");
    let mut pixels = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        pixels.extend_from_slice(&view[start..start + unpadded as usize]);
    }
    drop(view);
    buffer.unmap();
    pixels
}
