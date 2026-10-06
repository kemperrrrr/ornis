//! Custom-mesh smoke test proving the Custom upload path draws real geometry.

use super::super::*;
use super::*;

/// Custom smoke: a quad `MeshDesc::Custom` uploads through
/// `renderer::upload_custom_mesh` and draws non-empty pixels that
/// differ from the empty frame — the Custom path renders real
/// geometry instead of panicking (cf. the golden probe's Custom
/// arm). Skipped when no adapter is available.
#[test]
fn custom_quad_renders_nonempty_pixels_distinct_from_empty() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping custom probe");
        return;
    };
    const W: u32 = 320;
    const H: u32 = 180;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("custom probe target"),
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
    // Planar quad in y=0, CCW seen from +Y (the same soup as the
    // extraction `custom_quad_*` tests) — the upload must not panic.
    let positions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ];
    let indices = [0u32, 1, 2, 0, 2, 3];
    let mesh = crate::renderer::upload_custom_mesh(&device, &positions, &indices)
        .expect("quad soup valid");
    let mut red = ornis_core::OpenPBRMaterial::dielectric();
    red.base.color_rgb([0.8, 0.2, 0.2]);
    red.specular.roughness(0.5);
    backend.upload_materials(&device, &queue, &[red]);
    // The soup spans x/z in [0, 1]: enlarge 2x and center at the
    // origin so the camera frames it.
    let model = glam::Mat4::from_scale_rotation_translation(
        glam::Vec3::new(2.0, 1.0, 2.0),
        glam::Quat::IDENTITY,
        glam::Vec3::new(-1.0, 0.0, -1.0),
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
    // Camera above-front looking at the quad center; the light comes
    // from above so the +Y face is lit.
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 3.0, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 3.0]);
    backend.set_lights(
        &queue,
        [0.05, 0.05, 0.08],
        &[ornis_assets::scene::LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.2, 1.0, 0.3))
                .expect("non-zero direction"),
            intensity: 1.2,
            color: [1.0, 1.0, 1.0],
            shadow: ornis_assets::scene::ShadowCast::Disabled,
        }],
    );
    let render = |count: u32| -> Vec<u8> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("custom probe encoder"),
        });
        backend.render_scene(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &target_view,
            },
            &mesh,
            count,
        );
        let bpp = 4u32;
        let unpadded = W * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("custom probe readback"),
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

    let custom = render(1);
    let empty = render(0);
    assert_eq!(custom.len(), empty.len());
    let lit_px = custom
        .chunks_exact(4)
        .filter(|c| (c[0] as u16 + c[1] as u16 + c[2] as u16) > 30)
        .count();
    let mut diff_px = 0usize;
    for (a, b) in custom.chunks_exact(4).zip(empty.chunks_exact(4)) {
        let d = (a[0] as i16 - b[0] as i16).abs()
            + (a[1] as i16 - b[1] as i16).abs()
            + (a[2] as i16 - b[2] as i16).abs();
        if d > 12 {
            diff_px += 1;
        }
    }
    eprintln!("custom probe: lit_px={lit_px} diff_px={diff_px}");
    assert!(
        lit_px > 200,
        "custom quad drew nothing: {lit_px} lit pixels"
    );
    assert!(
        diff_px > 200,
        "custom frame matches the empty frame: {diff_px} pixels differ"
    );
}
