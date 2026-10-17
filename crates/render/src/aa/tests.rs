//! Reference and shader-contract regressions for AA.
use super::*;

#[test]
fn geometry_samples_and_shader_modes() {
    for (mode, samples, shader) in [
        (AntiAliasing::Off, 1, 0),
        (AntiAliasing::Msaa, 4, 1),
        (AntiAliasing::Fxaa, 1, 2),
        (AntiAliasing::Smaa, 1, 3),
        (AntiAliasing::Taa, 1, 4),
    ] {
        assert_eq!(mode.sample_count(), samples);
        assert_eq!(mode.shader_mode(), shader);
    }
}

#[test]
fn jitter_is_bounded_periodic_and_not_the_zero_index_corner() {
    assert!((temporal_jitter(0) - Vec2::new(0.0, -1.0 / 6.0)).length() < 1e-6);
    for frame in 0..64 {
        let jitter = temporal_jitter(frame);
        assert!(jitter.abs().max_element() <= 0.5);
        assert_eq!(jitter, temporal_jitter(frame + 8));
    }
    let projection = Mat4::perspective_rh(1.0, 1.5, 0.1, 100.0);
    let point = Vec4::new(1.0, 2.0, -10.0, 1.0);
    let original = projection * point;
    let shifted = jitter_projection(projection, Vec2::new(0.25, -0.25), [100, 100]) * point;
    assert!((shifted.x / shifted.w - original.x / original.w - 0.005).abs() < 1e-6);
    assert!((shifted.y / shifted.w - original.y / original.w - 0.005).abs() < 1e-6);
    assert_eq!(shifted.z, original.z);
    assert_eq!(shifted.w, original.w);
    assert_eq!(jitter_projection(projection, Vec2::ONE, [0, 1]), projection);
}

#[test]
fn motion_includes_skin_deformation_and_uv_y_flip() {
    let current = Vec4::new(0.4, 0.2, 0.5, 1.0);
    let previous = Vec4::new(0.0, -0.2, 0.5, 1.0);
    let motion = motion_vector(current, previous).unwrap();
    assert!((motion - Vec2::new(-0.2, 0.2)).length() < 1e-6);
    assert_eq!(motion_vector(current, current), Some(Vec2::ZERO));
    assert_eq!(motion_vector(current, Vec4::ZERO), None);
    assert_eq!(motion_vector(Vec4::splat(f32::NAN), previous), None);
}

#[test]
fn static_history_converges_after_64_frames() {
    let target = Vec3::new(0.2, 0.4, 0.8);
    let mut history = Vec3::ZERO;
    for _ in 0..64 {
        history = temporal_blend(target, history, &[Vec3::ZERO, Vec3::ONE], 0.9, true);
    }
    assert!((history - target).abs().max_element() < 0.001);
}

#[test]
fn moving_foreground_does_not_leave_a_bright_trail() {
    let background = Vec3::splat(0.02);
    let foreground = Vec3::splat(16.0);
    // Even an erroneously accepted reprojection cannot escape a flat neighbourhood.
    assert_eq!(
        temporal_blend(background, foreground, &[background; 9], 0.9, true),
        background
    );
    // Correct disocclusion rejection bypasses accumulation entirely.
    assert_eq!(
        temporal_blend(background, foreground, &[foreground; 9], 0.9, false),
        background
    );
    assert_eq!(
        temporal_blend(background, Vec3::NAN, &[], 0.9, true),
        background
    );
}

#[test]
fn shader_validates_with_webgpu_capabilities() {
    let source = include_str!("resolve.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 4);
}

#[test]
fn gpu_modes_preserve_a_flat_field_across_history_and_resize() {
    let Some((device, queue)) = pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    }) else {
        eprintln!("AA GPU test skipped: no adapter");
        return;
    };
    let extent = [8, 8];
    let input = Target::new(&device, extent);
    let motion = Target::new(&device, extent);
    let output = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("AA test output"),
        size: wgpu::Extent3d {
            width: 8,
            height: 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let mut aa = AaResolve::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        extent,
        AntiAliasing::Off,
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    for (view, color) in [
        (
            &input.view,
            wgpu::Color {
                r: 0.2,
                g: 0.4,
                b: 0.6,
                a: 1.0,
            },
        ),
        (
            &motion.view,
            wgpu::Color {
                r: 0.0,
                g: 0.0,
                b: 1.0,
                a: 1.0,
            },
        ),
    ] {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("AA test clear"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(color),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
    }
    let mut readbacks = Vec::new();
    for mode in [
        AntiAliasing::Off,
        AntiAliasing::Msaa,
        AntiAliasing::Fxaa,
        AntiAliasing::Smaa,
        AntiAliasing::Taa,
    ] {
        aa.set_mode(mode);
        assert!(!aa.valid);
        for _ in 0..32 {
            aa.render(
                &device,
                &mut encoder,
                AaInputs {
                    color: &input.view,
                    motion: &motion.view,
                    target: &output_view,
                },
            );
        }
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AA test readback"),
            size: 256 * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(8),
                },
            },
            wgpu::Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1,
            },
        );
        readbacks.push((mode, readback));
    }
    queue.submit([encoder.finish()]);
    for (mode, buffer) in readbacks {
        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = slice.get_mapped_range().unwrap();
        for y in 0..8 {
            for x in 0..8 {
                let offset = y * 256 + x * 4;
                for (channel, expected) in [51u8, 102, 153, 255].into_iter().enumerate() {
                    assert!(
                        bytes[offset + channel].abs_diff(expected) <= 1,
                        "{mode:?} at {x},{y}"
                    );
                }
            }
        }
        drop(bytes);
        buffer.unmap();
    }
    aa.resize(&device, [16, 8]);
    assert!(!aa.valid);
    assert_eq!(aa.frame, 0);
    aa.reset_history();
    assert_eq!(aa.jitter(), temporal_jitter(0));
}
