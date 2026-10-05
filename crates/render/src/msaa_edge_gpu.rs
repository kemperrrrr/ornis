//! Headless gate: a 4x silhouette must not light the octahedral clear.
//!
//! The quad faces the camera with shading normal +X and a light toward +X,
//! so a covered sample is bright. A cleared neighbor decodes to +Z (N·L ≈ 0).
//! The old resolver box-filtered those together and the edge went dim; the
//! edge mask shades only covered samples, so the last lit pixel stays near
//! the interior.

use super::Renderer3D;
use crate::mesh::create_facing_quad;
use crate::renderer::{InstanceData, MSAA_SAMPLE_COUNT, MaterialIdx, negotiate_sample_count};
use ornis_core::material::OpenPBRMaterial;

const FRAME_W: u32 = 320;
const FRAME_H: u32 = 180;
const BPP: u32 = 4;
const ROW: u32 = 90;
const INTERIOR_X: u32 = 40;
const EDGE_PIXEL: f32 = 160.0;
const SHIFTS: u32 = 8;
const SHIFTS_F: f32 = 8.0;
const PIXEL_NDC: f32 = 2.0 / 320.0;
const LIT_SUM: u32 = 12;
const MIN_INTERIOR: u32 = 40;
const EDGE_RATIO_NUM: u32 = 4;
const EDGE_RATIO_DEN: u32 = 5;

fn edge_translation(shift: u32) -> f32 {
    let frac = shift as f32 / SHIFTS_F;
    // Left edge of pixel 160 is NDC 0; `frac` walks the standard sample grid.
    let ndc = EDGE_PIXEL / (FRAME_W as f32) * 2.0 - 1.0 + frac * PIXEL_NDC;
    ndc - 1.0
}

fn channel_sum(pixels: &[u8], x: u32) -> u32 {
    let stride = FRAME_W * BPP;
    let index = (ROW * stride + x * BPP) as usize;
    pixels[index] as u32 + pixels[index + 1] as u32 + pixels[index + 2] as u32
}

/// Last pixel on the probe row that is still lit, and the interior sum.
fn transition(pixels: &[u8]) -> (u32, u32) {
    let interior = channel_sum(pixels, INTERIOR_X);
    let mut last = 0u32;
    let mut x = 0u32;
    while x < FRAME_W {
        if channel_sum(pixels, x) > LIT_SUM {
            last = x;
        }
        x += 1;
    }
    (last, interior)
}

fn bright_enough(edge: u32, interior: u32) -> bool {
    edge * EDGE_RATIO_DEN >= interior * EDGE_RATIO_NUM
}

fn edge_holds(pixels: &[u8]) -> bool {
    let (last, interior) = transition(pixels);
    interior > MIN_INTERIOR && bright_enough(channel_sum(pixels, last), interior)
}

fn open_adapter() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;
        Some((adapter, device, queue))
    })
}

fn render_shift(device: &wgpu::Device, queue: &wgpu::Queue, shift: u32) -> Vec<u8> {
    let renderer = Renderer3D::new(device, &surface_config(), MSAA_SAMPLE_COUNT);
    let mesh = create_facing_quad(device, [2.0, 2.0], [1.0, 0.0, 0.0]);
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([0.55, 0.55, 0.55]);
    material.specular.roughness(1.0);
    renderer.upload_materials(device, queue, &[material]);
    let tx = edge_translation(shift);
    let model = glam::Mat4::from_translation(glam::Vec3::new(tx, 0.0, 0.0));
    renderer.upload_instances(
        device,
        queue,
        &[InstanceData {
            model_matrix: model,
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        }],
    );
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 0.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 10.0);
    renderer.set_camera(queue, &(proj * view).to_cols_array_2d(), [0.0, 0.0, 3.0]);
    renderer.set_lights(
        queue,
        [0.02, 0.02, 0.02],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::X).expect("axis"),
            intensity: 0.45,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::Disabled,
        }],
    );
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("msaa edge target"),
        size: wgpu::Extent3d {
            width: FRAME_W,
            height: FRAME_H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("msaa edge"),
    });
    renderer.render_deferred_frame(device, queue, &mut encoder, &view, &mesh);
    queue.submit([encoder.finish()]);
    readback(device, queue, &target)
}

fn surface_config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: FRAME_W,
        height: FRAME_H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let unpadded = FRAME_W * BPP;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("msaa edge readback"),
        size: (padded * FRAME_H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("msaa edge readback"),
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
                rows_per_image: Some(FRAME_H),
            },
        },
        wgpu::Extent3d {
            width: FRAME_W,
            height: FRAME_H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    let data = slice.get_mapped_range().expect("mapped readback");
    let mut pixels = vec![0u8; (unpadded * FRAME_H) as usize];
    let mut y = 0u32;
    while y < FRAME_H {
        let src = (y * padded) as usize;
        let dst = (y * unpadded) as usize;
        pixels[dst..dst + unpadded as usize].copy_from_slice(&data[src..src + unpadded as usize]);
        y += 1;
    }
    drop(data);
    buffer.unmap();
    pixels
}

fn shift_ok(device: &wgpu::Device, queue: &wgpu::Queue, shift: u32) -> bool {
    let pixels = render_shift(device, queue, shift);
    let (last, interior) = transition(&pixels);
    let edge = channel_sum(&pixels, last);
    eprintln!("msaa edge shift {shift}: interior {interior} last_x {last} edge {edge}");
    edge_holds(&pixels)
}

fn all_shifts(device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    let mut shift = 0u32;
    let mut ok = true;
    while shift < SHIFTS {
        ok = ok && shift_ok(device, queue, shift);
        shift += 1;
    }
    ok
}

fn is_four(count: u32) -> bool {
    count == MSAA_SAMPLE_COUNT
}

fn run_probe(is_msaa: bool, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
    match is_msaa {
        true => all_shifts(device, queue),
        false => true,
    }
}

#[test]
fn silhouette_edge_stays_as_bright_as_the_interior() {
    match open_adapter() {
        Some((adapter, device, queue)) => {
            let count = negotiate_sample_count(&adapter, MSAA_SAMPLE_COUNT);
            assert!(
                run_probe(is_four(count), &device, &queue),
                "silhouette fringe: edge pixel darker than the +X interior"
            );
        }
        None => eprintln!("SKIP: no wgpu adapter"),
    }
}
