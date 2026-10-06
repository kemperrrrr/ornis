//! Point shadow-mapping probes: cube-path acne guard and the all-techniques darkening regression.

use super::super::*;
use super::*;
use crate::flags::ShadowCast;

/// Acne guard for the cube path: a lone receiver under a shadowed
/// point light must render (nearly) identically with the shadow on
/// versus off. The cube faces share the depth-bias pair and the
/// reference bias with the 2D layers, so both need the pin.
/// Skipped when no adapter is available.
#[test]
fn shadowed_point_without_occluder_has_no_acne() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping shadow probe");
        return;
    };
    const W: u32 = 320;
    const H: u32 = 180;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("point acne probe target"),
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
    let mut backend = create_render_backend(
        &device,
        &RenderBackendConfig {
            surface_config,
            sample_count: 1,
            exposure: 1.0,
            max_objects: 256,
            max_materials: 64,
        },
    );
    let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
    let mut gray = ornis_core::OpenPBRMaterial::dielectric();
    gray.base.color_rgb([0.8, 0.8, 0.8]);
    gray.specular.roughness(0.5);
    backend.upload_materials(&device, &queue, &[gray]);
    let model = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::splat(2.0),
        glam::Quat::IDENTITY,
        glam::Vec3::ZERO,
    );
    backend.upload_instances(
        &device,
        &queue,
        &[crate::renderer::InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: crate::renderer::MaterialIdx::from_raw(0),
        }],
    );
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 3.0, 10.0),
        glam::Vec3::new(0.0, 1.0, 0.0),
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);
    let mut render = |shadow: ShadowCast| -> Vec<u8> {
        backend.set_lights(
            &queue,
            [0.05, 0.05, 0.08],
            &[ornis_assets::scene::LightDesc::Point {
                position: glam::Vec3::new(5.0, 4.0, 6.0),
                intensity: 200.0,
                color: [1.0, 1.0, 1.0],
                range: ornis_core::units::Meters::new(30.0),
                shadow: ornis_assets::scene::ShadowCast::from(shadow.is_enabled()),
            }],
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("point acne probe encoder"),
        });
        backend.render_scene(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &target_view,
            },
            &mesh,
            1,
        );
        let bpp = 4u32;
        let unpadded = W * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point acne probe readback"),
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
    let on = render(ShadowCast::Enabled);
    let off = render(ShadowCast::Disabled);
    let mut diff_px = 0usize;
    for (a, b) in on.chunks_exact(4).zip(off.chunks_exact(4)) {
        let d = (a[0] as i16 - b[0] as i16).abs()
            + (a[1] as i16 - b[1] as i16).abs()
            + (a[2] as i16 - b[2] as i16).abs();
        if d > 12 {
            diff_px += 1;
        }
    }
    eprintln!("point acne probe: diff_px={diff_px}");
    assert!(
        diff_px < 100,
        "cube self-shadowing acne without occluder: {diff_px} pixels differ"
    );
}

/// Point-shadow regression: a shadowed point light must darken the
/// receiver behind the occluder versus `shadow: ShadowCast::Disabled`, in every
/// technique (deferred, forward, hybrid). Exercises the full chain
/// (cube-slot assignment → 6 face renders with the hardware
/// sampler frame → analytic major-axis compare), including the
/// forward-only plan path, which owns its shadow pre-pass via
/// `Forward<OwnsDepth>`. Skipped when no adapter is available.
#[test]
fn shadowed_point_darkens_occluded_receiver_in_all_techniques() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping shadow probe");
        return;
    };
    const W: u32 = 320;
    const H: u32 = 180;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("point shadow probe target"),
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
    let backend = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
    // Unit sphere; sizes come from instance scales (true geometry,
    // unlike the probe examples that reuse the first entity's mesh).
    // (Concrete `Renderer3D` — not the boxed backend — because the
    // plan path below takes `&Renderer3D` directly.)
    let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
    let mut gray = ornis_core::OpenPBRMaterial::dielectric();
    gray.base.color_rgb([0.8, 0.8, 0.8]);
    gray.specular.roughness(0.5);
    backend.upload_materials(&device, &queue, &[gray, gray]);
    let place = |translation: [f32; 3], scale: f32| {
        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(scale),
            glam::Quat::IDENTITY,
            glam::Vec3::from(translation),
        );
        crate::renderer::InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: crate::renderer::MaterialIdx::from_raw(0),
        }
    };
    // Receiver r=2 at origin; occluder r=0.6 on the light axis.
    // The light sits OFF the cube seams (|x|≠|y|≠|z| along the
    // axis): axis-aligned light→receiver directions ride a cube
    // edge, where the 2x2 PCF footprint straddles two faces and
    // small centered occluders vanish. Light (5,4,6), occluder at
    // t=0.35 along light→origin: (3.25,2.6,3.9). Point range
    // comfortably covers the scene.
    backend.upload_instances(
        &device,
        &queue,
        &[place([0.0, 0.0, 0.0], 2.0), place([3.25, 2.6, 3.9], 0.6)],
    );
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 3.0, 10.0),
        glam::Vec3::new(0.0, 1.0, 0.0),
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);

    let render = |technique: crate::frame_exec::Technique, shadow: ShadowCast| -> Vec<u8> {
        backend.set_lights(
            &queue,
            [0.05, 0.05, 0.08],
            &[ornis_assets::scene::LightDesc::Point {
                position: glam::Vec3::new(5.0, 4.0, 6.0),
                intensity: 200.0,
                color: [1.0, 1.0, 1.0],
                range: ornis_core::units::Meters::new(30.0),
                shadow: ornis_assets::scene::ShadowCast::from(shadow.is_enabled()),
            }],
        );
        let mut plan = crate::frame_exec::RenderFrame3D::new_with(
            format,
            (W, H),
            technique,
            crate::flags::Bloom::Off,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("point shadow probe encoder"),
        });
        plan.render(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &target_view,
            },
            &backend,
            &mesh,
            2,
        );
        let bpp = 4u32;
        let unpadded = W * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("point shadow probe readback"),
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

    for technique in [
        crate::frame_exec::Technique::Deferred,
        crate::frame_exec::Technique::Forward,
        crate::frame_exec::Technique::Hybrid,
    ] {
        let on = render(technique, ShadowCast::Enabled);
        let off = render(technique, ShadowCast::Disabled);
        assert_eq!(on.len(), off.len());
        let lum = |px: &[u8]| -> f64 {
            px.chunks_exact(4)
                .map(|c| (c[0] as f64 + c[1] as f64 + c[2] as f64) / 3.0)
                .sum::<f64>()
                / (px.len() / 4) as f64
        };
        let (mean_on, mean_off) = (lum(&on), lum(&off));
        let mut diff_px = 0usize;
        for (a, b) in on.chunks_exact(4).zip(off.chunks_exact(4)) {
            let d = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            if d > 12 {
                diff_px += 1;
            }
        }
        eprintln!(
            "point shadow probe ({technique:?}): mean_on={mean_on:.2} mean_off={mean_off:.2} diff_px={diff_px}"
        );
        assert!(
            diff_px > 500,
            "point shadow had no visible effect ({technique:?}): {diff_px} pixels differ"
        );
        assert!(
            mean_on + 1.0 < mean_off,
            "shadowed frame is not darker ({technique:?}): on={mean_on:.2} off={mean_off:.2}"
        );
    }
}
