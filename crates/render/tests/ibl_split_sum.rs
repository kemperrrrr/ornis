//! Headless gate for split-sum IBL: the default environment is a no-op
//! (black frame with no analytic lights), a white environment lights the
//! sphere, and a +Z-only red environment tints the camera-facing normal.

use ornis_render::{
    CubeFace, EnvironmentCube, InstanceData, MaterialIdx, OpenPBRMaterial, Renderer3D,
};

const W: u32 = 160;
const H: u32 = 90;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn try_adapter() -> Option<(wgpu::Device, wgpu::Queue)> {
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
        Some((device, queue))
    })
}

fn surface_config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: FORMAT,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

fn render_sphere(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env: Option<&EnvironmentCube>,
) -> Vec<u8> {
    let mut renderer = Renderer3D::new(device, &surface_config(), 1);
    if let Some(env) = env {
        renderer.set_image_based_light(device, queue, Some(env));
    }
    let mesh = ornis_render::create_sphere(device, 1.0, 16, 12);
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([0.8, 0.8, 0.8]);
    material.specular.roughness(0.5);
    renderer.upload_materials(device, queue, &[material]);
    let model = glam::Mat4::IDENTITY;
    renderer.upload_instances(
        device,
        queue,
        &[InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: MaterialIdx::from_raw(0),
        }],
    );
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 0.2, 3.2),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        50.0,
    );
    renderer.set_camera(queue, &(proj * view).to_cols_array_2d(), [0.0, 0.2, 3.2]);
    renderer.set_lights(queue, [0.0, 0.0, 0.0], &[]);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ibl probe"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ibl probe"),
    });
    renderer.render_scene(device, queue, &mut encoder, &view, &mesh, 1);
    queue.submit([encoder.finish()]);
    readback(device, queue, &target)
}

fn readback(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let unpadded = W * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ibl readback"),
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("ibl readback"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    let data = slice.get_mapped_range().unwrap();
    let mut pixels = vec![0u8; (unpadded * H) as usize];
    for y in 0..H as usize {
        pixels[y * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
    }
    drop(data);
    readback.unmap();
    pixels
}

fn center_luma(pixels: &[u8]) -> u32 {
    let x = (W / 2) as usize;
    let y = (H / 2) as usize;
    let i = (y * W as usize + x) * 4;
    pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32
}

fn center_red(pixels: &[u8]) -> (u8, u8, u8) {
    let x = (W / 2) as usize;
    let y = (H / 2) as usize;
    let i = (y * W as usize + x) * 4;
    (pixels[i], pixels[i + 1], pixels[i + 2])
}

#[test]
fn missing_env_is_black_and_white_env_lights() {
    let Some((device, queue)) = try_adapter() else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let dark = render_sphere(&device, &queue, None);
    assert!(
        center_luma(&dark) < 12,
        "default IBL must stay a no-op, luma {}",
        center_luma(&dark)
    );
    let white = EnvironmentCube::solid(ornis_core::units::Color::WHITE, 4);
    let lit = render_sphere(&device, &queue, Some(&white));
    assert!(
        center_luma(&lit) > 40,
        "white environment should light the sphere, luma {}",
        center_luma(&lit)
    );
    let red = EnvironmentCube::solid(ornis_core::units::Color::BLACK, 4).with_solid_face(
        CubeFace::PositiveZ,
        ornis_core::units::Color::linear_rgb(1.0, 0.0, 0.0),
    );
    let tinted = render_sphere(&device, &queue, Some(&red));
    let (r, g, b) = center_red(&tinted);
    assert!(
        r > g.saturating_add(8) && r > b.saturating_add(8),
        "camera-facing +Z should sample the red face, rgb=({r},{g},{b})"
    );
}
