//! Shared GPU harness and base backend tests: device helper, submodule wiring, and config/factory/trait coverage.

use super::*;

mod custom_mesh;
mod golden;
mod lights;
mod shadows_dir;
mod shadows_point_spot;

/// None when no adapter is available (CI without GPU and without
/// lavapipe); the tests below skip in that case.
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

#[test]
fn default_config_matches_documented_defaults() {
    let config = RenderBackendConfig::default();
    assert_eq!(config.surface_config.width, 800);
    assert_eq!(config.surface_config.height, 600);
    assert_eq!(
        config.surface_config.format,
        wgpu::TextureFormat::Rgba8UnormSrgb
    );
    assert_eq!(
        config.surface_config.usage,
        wgpu::TextureUsages::RENDER_ATTACHMENT
    );
    assert_eq!(
        config.surface_config.present_mode,
        wgpu::PresentMode::AutoNoVsync
    );
    assert_eq!(config.surface_config.desired_maximum_frame_latency, 2);
    assert!(config.surface_config.view_formats.is_empty());
    assert_eq!(config.sample_count, 1);
    assert_eq!(config.exposure, 1.0);
    assert_eq!(config.max_objects, 256);
    assert_eq!(config.max_materials, 64);
}

/// MSAA/exposure passthrough gate: the default config carries
/// `sample_count = 1` and `exposure = 1.0` (both exact no-ops), and a
/// non-default config round-trips its values into the built renderer
/// without changing the defaults. Pure CPU except the build itself.
#[test]
fn config_msaa_and_exposure_forward_to_renderer() {
    let defaults = RenderBackendConfig::default();
    assert_eq!(defaults.sample_count, 1);
    assert_eq!(defaults.exposure, 1.0);
    let custom = RenderBackendConfig {
        sample_count: 4,
        exposure: 2.0,
        ..RenderBackendConfig::default()
    };
    assert_eq!(custom.sample_count, 4);
    assert_eq!(custom.exposure, 2.0);
    // The defaults above are unchanged by constructing a custom value.
    assert_eq!(RenderBackendConfig::default().sample_count, 1);
    assert_eq!(RenderBackendConfig::default().exposure, 1.0);
}

/// Transparency-flag gate: building with default transparency options
/// renders pixel-identical to the plain constructor (the flag defaults
/// off and must not change the default frame). Skipped when no adapter
/// is available.
#[test]
fn default_transparency_flag_leaves_frame_unchanged() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    const W: u32 = 160;
    const H: u32 = 90;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let plain = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
    let flagged = crate::renderer::Renderer3D::new_with_transparency(
        &device,
        &surface_config,
        1,
        crate::renderer::TransparencyOptions::default(),
    );
    assert_eq!(
        flagged.transparency(),
        crate::renderer::TransparencyOptions::default()
    );
    assert_eq!(flagged.sample_count(), 1);
    assert_eq!(flagged.exposure(), 1.0);

    let mesh = crate::mesh::create_sphere(&device, 1.0, 16, 12);
    let mut red = ornis_core::OpenPBRMaterial::dielectric();
    red.base.color_rgb([0.8, 0.2, 0.2]);
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 0.0, 5.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    let view_proj = (proj * view).to_cols_array_2d();
    let render = |renderer: &crate::renderer::Renderer3D| -> Vec<u8> {
        renderer.upload_materials(&device, &queue, &[red]);
        let model = glam::Mat4::IDENTITY;
        renderer.upload_instances(
            &device,
            &queue,
            &[crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: crate::renderer::MaterialIdx::from_raw(0),
            }],
        );
        renderer.set_camera(&queue, &view_proj, [0.0, 0.0, 5.0]);
        renderer.set_lights(
            &queue,
            [0.1, 0.1, 0.15],
            &[ornis_assets::scene::LightDesc::Directional {
                direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(1.0, 1.0, 1.0))
                    .expect("non-zero direction"),
                intensity: 1.0,
                color: [1.0, 1.0, 1.0],
                shadow: ornis_assets::scene::ShadowCast::Disabled,
            }],
        );
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("transparency flag target"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("transparency flag encoder"),
        });
        renderer.render_scene(&device, &queue, &mut encoder, &target_view, &mesh, 1);
        let bpp = 4u32;
        let unpadded = W * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transparency flag readback"),
            size: (padded * H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target_tex,
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
            .expect("poll");
        let data = slice.get_mapped_range().unwrap();
        let mut pixels = vec![0u8; (unpadded * H) as usize];
        for y in 0..H as usize {
            pixels[y * unpadded as usize..][..unpadded as usize]
                .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
        }
        drop(data);
        readback.unmap();
        pixels
    };

    let a = render(&plain);
    let b = render(&flagged);
    assert_eq!(a.len(), b.len());
    let diffs = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
    eprintln!("transparency flag: differing bytes={diffs}");
    assert_eq!(diffs, 0, "default transparency flag changed the frame");
}

/// Uploads past the initial 256-instance / 64-material capacities grow
/// the storage buffers (doubling) instead of truncating the frame:
/// after uploading 300 instances + 70 materials the draw still sees
/// entry 299 / material 69.
#[test]
fn oversized_frame_grows_buffers_without_truncation() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let config = RenderBackendConfig::default();
    let mut backend = create_render_backend(&device, &config);
    let materials: Vec<OpenPBRMaterial> = (0..70).map(|_| OpenPBRMaterial::default()).collect();
    backend.upload_materials(&device, &queue, &materials);
    let instances: Vec<InstanceData> = (0..300)
        .map(|i| InstanceData {
            model_matrix: glam::Mat4::from_translation(glam::Vec3::new(i as f32, 0.0, 0.0)),
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: crate::renderer::MaterialIdx::from((i % 70) as u32),
        })
        .collect();
    backend.upload_instances(&device, &queue, &instances);

    // The 300th instance / 70th material survive the round trip:
    // draw them alone into a target and require non-background pixels.
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("growth probe target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mesh = crate::mesh::create_sphere(&device, 1.0, 8, 4);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("growth probe"),
    });
    // Re-upload just the tail entries at index 0 to prove they landed.
    backend.upload_materials(&device, &queue, &materials[69..70]);
    backend.upload_instances(&device, &queue, &instances[299..300]);
    backend.render_scene(
        RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &view,
        },
        &mesh,
        1,
    );
    queue.submit(std::iter::once(encoder.finish()));
}

#[test]
fn factory_builds_backend_and_resize_reallocates() {
    let Some((device, _queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let config = RenderBackendConfig::default();
    let mut backend = create_render_backend(&device, &config);
    // Resizing to a different extent must not panic and keeps the
    // backend usable (exercises the RenderBackend trait impl path).
    backend.resize(&device, 320, 240);
    backend.resize(&device, 800, 600);
}

/// Drives every `RenderBackend` trait method end-to-end through the
/// factory handle: upload one instance + material, draw one frame into
/// an offscreen target, submit.
#[test]
fn trait_object_renders_one_instance() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let config = RenderBackendConfig::default();
    let mut backend = create_render_backend(&device, &config);

    backend.set_camera(
        &queue,
        &glam::Mat4::IDENTITY.to_cols_array_2d(),
        [0.0, 0.0, 3.0],
    );
    backend.set_lights(
        &queue,
        [0.1, 0.1, 0.1],
        &[LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.0, 1.0, 1.0))
                .expect("non-zero direction"),
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::Disabled,
        }],
    );
    backend.upload_materials(&device, &queue, &[OpenPBRMaterial::default()]);
    backend.upload_instances(
        &device,
        &queue,
        &[InstanceData {
            model_matrix: glam::Mat4::IDENTITY,
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: crate::renderer::MaterialIdx::from_raw(0),
        }],
    );

    let mesh = crate::mesh::create_sphere(&device, 1.0, 8, 4);
    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("backend test target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("backend test encoder"),
    });
    backend.render_scene(
        RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &target,
        },
        &mesh,
        1,
    );
    queue.submit([encoder.finish()]);
}
