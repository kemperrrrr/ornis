//! Pixel parity for the stats-window blit: the translated module
//! (`crates/gpu-blit`, checked in as `stats/blit_generated.wgsl`) against the
//! hand-written WGSL it replaced, frozen in `data/blit_handwritten.wgsl`.
//!
//! The blit is a 1:1 texel copy, so the interesting behaviour is the tail: the
//! stats surface and the panel are resized independently, and a pixel past the
//! panel must read the last texel rather than sample outside it. The gate
//! therefore draws a target strictly larger than the panel in both axes, gives
//! every panel texel a distinct value in every channel, and checks that the
//! tail repeats the last column and row — otherwise "identical pixels" could
//! hold simply because neither shader ever left the panel.

use std::sync::mpsc;

/// The hand-written shader this pass used before the migration.
const REFERENCE_WGSL: &str = include_str!("data/blit_handwritten.wgsl");

/// The checked-in artifact of `ornis-gpu-blit`, as the stats window loads it.
const TRANSLATED_WGSL: &str = include_str!("../src/stats/blit_generated.wgsl");

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Bytes per `RGBA8` texel.
const TEXEL_BYTES: u32 = 4;
/// Panel extent: non-square, so an axis mix-up cannot pass.
const PANEL: (u32, u32) = (3, 2);
/// Target extent: larger than the panel in both axes, so the tail clamps.
const TARGET: (u32, u32) = (7, 5);
/// `copy_texture_to_buffer` row alignment.
const ROW_ALIGN: u32 = 256;

#[test]
fn translated_blit_copies_the_panel_like_the_hand_written_shader() {
    let Some((device, queue)) = adapter_or_skip() else {
        return;
    };
    // The frozen copy keeps the names it was written with; the translated
    // module names its exports through `#[gpu_entry]`.
    let reference = render(&device, &queue, REFERENCE_WGSL, ("vs", "fs"));
    let entries = (ornis_gpu_blit::VERTEX_ENTRY, ornis_gpu_blit::FRAGMENT_ENTRY);
    for entry in [entries.0, entries.1] {
        assert!(
            TRANSLATED_WGSL.contains(&format!("fn {entry}(")),
            "the artifact does not declare {entry}, the entry the crate exports"
        );
    }
    let translated = render(&device, &queue, TRANSLATED_WGSL, entries);
    assert_eq!(translated.len(), reference.len(), "read-back sizes differ");
    assert_eq!(
        translated, reference,
        "the translated blit differs from the hand-written one"
    );

    // Not vacuous: every panel texel reaches the target, and the tail is the
    // clamped last column and row rather than a copy of the panel corner.
    for y in 0..PANEL.1 {
        for x in 0..PANEL.0 {
            let texel = panel_texel(x, y);
            assert!(
                translated.chunks_exact(4).any(|pixel| pixel == texel),
                "panel texel {texel:?} at ({x},{y}) is missing from the blit"
            );
        }
    }
    assert_eq!(
        pixel(&translated, TARGET.0 - 1, 0),
        panel_texel(PANEL.0 - 1, 0),
        "the column past the panel is not the last texel"
    );
    assert_eq!(
        pixel(&translated, 0, TARGET.1 - 1),
        panel_texel(0, PANEL.1 - 1),
        "the row past the panel is not the last texel"
    );
    assert!(
        panel_texel(0, 0)[3] != panel_texel(1, 1)[3],
        "alpha must vary, or the fourth channel is untested"
    );
}

/// One panel texel, distinct in every channel.
fn panel_texel(x: u32, y: u32) -> [u8; 4] {
    [
        16 + (x * 40 + y * 7) as u8,
        200 - (x * 30 + y * 11) as u8,
        (x * 50 + y * 90) as u8,
        255 - (x + y) as u8 * 20,
    ]
}

/// Draw the fullscreen triangle through `wgsl` and read the target back.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    wgsl: &str,
    (vertex, fragment): (&str, &str),
) -> Vec<u8> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("blit parity"),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("blit parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(vertex),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(fragment),
            targets: &[Some(FORMAT.into())],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let panel = panel_texture(device, queue);
    // The panel bind group comes from the pipeline's own layout: the pass
    // declares its binding in WGSL, so an explicit layout could only disagree
    // with it.
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blit parity panel"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(
                &panel.create_view(&wgpu::TextureViewDescriptor::default()),
            ),
        }],
    });
    let target = texture(
        device,
        "blit parity target",
        TARGET,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("blit parity"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("blit parity"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // No vertex buffer: the triangle comes from `vertex_index`.
        pass.draw(0..3, 0..1);
    }
    read_back(device, queue, &target, encoder)
}

/// An `RGBA8` texture of `size` with `usage`, one mip, one sample.
fn texture(
    device: &wgpu::Device,
    label: &str,
    size: (u32, u32),
    usage: wgpu::TextureUsages,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage,
        view_formats: &[],
    })
}

/// `size` as the single-layer extent every texture and copy in this gate uses.
fn extent(size: (u32, u32)) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size.0,
        height: size.1,
        depth_or_array_layers: 1,
    }
}

/// Mip 0 of `texture`, the origin of every copy in this gate.
fn copy_source(texture: &wgpu::Texture) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    }
}

/// The panel texture with its texels uploaded.
fn panel_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let texture = texture(
        device,
        "blit parity panel",
        PANEL,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    );
    let mut texels = Vec::with_capacity((PANEL.0 * PANEL.1 * TEXEL_BYTES) as usize);
    for y in 0..PANEL.1 {
        for x in 0..PANEL.0 {
            texels.extend_from_slice(&panel_texel(x, y));
        }
    }
    queue.write_texture(
        copy_source(&texture),
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(PANEL.0 * TEXEL_BYTES),
            rows_per_image: Some(PANEL.1),
        },
        extent(PANEL),
    );
    texture
}

/// Submit `encoder`, read the target back and strip the row padding
/// `copy_texture_to_buffer` requires.
///
/// The padded-row arithmetic and the row-strip are the repo's one read-back
/// idiom (`crates/render/tests/common/sized.rs` spells it the same way); a
/// shared helper would have to live in a production crate for two test
/// modules.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target: &wgpu::Texture,
    mut encoder: wgpu::CommandEncoder,
) -> Vec<u8> {
    // qual:allow(dry, fragment) reason: "the padded-row arithmetic of a texture read-back, spelled as the render harness spells it; a shared helper would have to ship in a production crate for two test modules"
    let (width, height) = TARGET;
    let unpadded = width * TEXEL_BYTES;
    let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blit parity read-back"),
        size: u64::from(padded) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        copy_source(target),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        extent(TARGET),
    );
    queue.submit([encoder.finish()]);
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    receiver
        .recv()
        .expect("read-back callback")
        .expect("map the read-back buffer");
    // qual:allow(dry, fragment) reason: "the row-strip of a padded texture read-back, spelled as the render harness spells it; see the note above"
    let mapped = slice.get_mapped_range().expect("map the read-back buffer");
    let mut out = Vec::with_capacity((unpadded * height) as usize);
    for row in 0..height {
        let start = (row * padded) as usize;
        out.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    buffer.unmap();
    out
}

/// One pixel of a tight `RGBA8` read-back.
fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * TARGET.0 + x) * TEXEL_BYTES) as usize;
    pixels[offset..offset + 4].try_into().expect("four bytes")
}

/// Headless adapter and device, or `None` on a host without one.
fn adapter_or_skip() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let opened = pollster::block_on(async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()
    });
    if let Some(pair) = opened {
        return Some(pair);
    }
    assert!(
        !gpu_required(),
        "ORNIS_REQUIRE_GPU is set but no wgpu adapter is available"
    );
    eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
    None
}

/// Whether a missing adapter is a failure rather than a skip.
fn gpu_required() -> bool {
    std::env::var("ORNIS_REQUIRE_GPU").is_ok_and(|value| !value.is_empty() && value != "0")
}
