//! Headless GPU resolve regressions.
use super::*;
const SIDE: u32 = 8;
const ROW_BYTES: u32 = 256;
const FRAME_COUNT: usize = 32;
const RESIZED_WIDTH: u32 = 16;
const EXPECTED_PIXEL: [u8; 4] = [51, 102, 153, 255];
const INPUT_COLOR: wgpu::Color = wgpu::Color {
    r: 0.2,
    g: 0.4,
    b: 0.6,
    a: 1.0,
};
const MOTION_COLOR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 0.0,
    b: 1.0,
    a: 1.0,
};

#[test]
fn gpu_modes_preserve_a_flat_field_across_history_and_resize() {
    let Some((device, queue)) = try_device() else {
        eprintln!("AA GPU test skipped: no adapter");
        return;
    };
    let extent = [SIDE, SIDE];
    let input = Target::new(&device, extent);
    let motion = Target::new(&device, extent);
    let output = output_texture(&device);
    let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());
    let mut aa = AaResolve::new(
        &device,
        wgpu::TextureFormat::Rgba8Unorm,
        extent,
        AntiAliasing::Off,
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    clear(&mut encoder, &input.view, INPUT_COLOR);
    clear(&mut encoder, &motion.view, MOTION_COLOR);
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
        for _ in 0..FRAME_COUNT {
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
        let readback = copy_output(&device, &mut encoder, &output);
        readbacks.push((mode, readback));
    }
    queue.submit([encoder.finish()]);
    for (mode, buffer) in readbacks {
        assert_output(&device, mode, &buffer);
    }
    aa.resize(&device, [RESIZED_WIDTH, SIDE]);
    assert!(!aa.valid);
    assert_eq!(aa.frame, 0);
    aa.reset_history();
    assert_eq!(aa.jitter(), temporal_jitter(0));
}

fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
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

fn clear(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, color: wgpu::Color) {
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

fn output_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("AA test output"),
        size: wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn copy_output(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    output: &wgpu::Texture,
) -> wgpu::Buffer {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("AA test readback"),
        size: u64::from(ROW_BYTES * SIDE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW_BYTES),
                rows_per_image: Some(SIDE),
            },
        },
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
    );
    readback
}

fn assert_output(device: &wgpu::Device, mode: AntiAliasing, buffer: &wgpu::Buffer) {
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
    for y in 0..SIDE as usize {
        for x in 0..SIDE as usize {
            let offset = y * ROW_BYTES as usize + x * EXPECTED_PIXEL.len();
            for (channel, expected) in EXPECTED_PIXEL.into_iter().enumerate() {
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
