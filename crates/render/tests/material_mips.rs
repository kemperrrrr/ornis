//! Headless material-texture gates: analytic checker golden and KTX2 mip sampling.
//! CI uses lavapipe; a compression-capable adapter also exercises its native targets.

#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_render::texture_upload::{
    preferred_texture_encoding, texture_compression_features, upload_image,
};
use ornis_render::textures::upload_texture;
use ornis_render::{CpuImage, GpuTexture, Ktx2Image, TextureEncoding, TextureImage, TextureRole};

const SHADER: &str = r#"
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@vertex fn plane(@builtin(vertex_index) i: u32) -> Vertex {
    let positions = array<vec4f, 4>(vec4f(-0.875, -0.875, 0.5, 1.0),
        vec4f(0.875, -0.875, 0.5, 1.0), vec4f(-2.0, 6.0, 4.0, 8.0), vec4f(2.0, 6.0, 4.0, 8.0));
    let uvs = array<vec2f, 4>(vec2f(0.0, 0.0), vec2f(32.0, 0.0), vec2f(0.0, 32.0), vec2f(32.0, 32.0));
    return Vertex(positions[i], uvs[i]);
}
@vertex fn quad(@builtin(vertex_index) i: u32) -> Vertex {
    let positions = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    let p = positions[i];
    return Vertex(vec4f(p, 0.5, 1.0), vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
}
@fragment fn filtered(v: Vertex) -> @location(0) vec4f {
    return textureSample(image, image_sampler, v.uv);
}
"#;

fn sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &GpuTexture,
    view: &wgpu::TextureView,
    size: u32,
    level: Option<u32>,
) -> Vec<u8> {
    let source = match level {
        Some(mip) => format!(
            "{SHADER}\n@fragment fn selected(v: Vertex) -> @location(0) vec4f {{ return textureSampleLevel(image, image_sampler, v.uv, {mip}.0); }}"
        ),
        None => SHADER.to_owned(),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("material texture pixel gate"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("material texture pixel gate"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(if level.is_some() { "quad" } else { "plane" }),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(if level.is_some() {
                "selected"
            } else {
                "filtered"
            }),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&image.sampler),
            },
        ],
    });
    render_and_readback(
        device,
        queue,
        &FrameSpec {
            size: (size, size),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        },
        |encoder, target| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..4, 0..1);
        },
    )
}

#[test]
fn tilted_checker_matches_linear_light_golden_not_moire() {
    let Some((_, device, queue)) = pollster::block_on(common::request_device()) else {
        assert!(
            std::env::var_os("CI").is_none(),
            "CI requires a GPU adapter (lavapipe)"
        );
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let pixels = (0..256 * 256)
        .flat_map(|i| {
            let c = if (i % 256 + i / 256) % 2 == 0 { 0 } else { 255 };
            [c, c, c, 255]
        })
        .collect();
    let image = CpuImage::from_rgba8(256, 256, pixels).unwrap();
    let uploaded = upload_texture(&device, &queue, &image, TextureRole::BaseColor).unwrap();
    assert_eq!(uploaded.texture.mip_level_count(), 9);
    let pixels = sample(&device, &queue, &uploaded, &uploaded.view, 128, None);
    let golden = include_bytes!("data/tilted_checker.rgba");
    assert_eq!(pixels.len(), golden.len());
    let mismatches = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(&a, &b)| a.abs_diff(b) > 2))
        .count();
    // At most a one-pixel rasterization boundary; interior must remain flat.
    assert!(
        mismatches < 128,
        "{mismatches} pixels differ from the analytic golden"
    );
    let base_only = uploaded.texture.create_view(&wgpu::TextureViewDescriptor {
        mip_level_count: Some(1),
        ..Default::default()
    });
    let control = sample(&device, &queue, &uploaded, &base_only, 128, None);
    let unstable = control
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden.as_chunks::<4>().0.iter())
        .filter(|(a, b)| b[0] == 188 && a[0].abs_diff(188) > 20)
        .count();
    assert!(
        unstable > 500,
        "base-only control must demonstrate aliasing: {unstable}"
    );
}

#[test]
fn ktx2_authored_mips_sample_on_fallback_and_native_devices() {
    let Some((adapter, device, queue)) = pollster::block_on(common::request_device()) else {
        assert!(std::env::var_os("CI").is_none(), "CI requires lavapipe");
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    for bytes in [
        include_bytes!("../../texture/tests/data/uastc.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/etc1s.ktx2").as_slice(),
    ] {
        let image = TextureImage::Ktx2(Ktx2Image::from_bytes(bytes).unwrap());
        // No compression bits were requested even if lavapipe advertises BC.
        verify_ktx_mips(&device, &queue, &image, TextureEncoding::Rgba8);
        let features = texture_compression_features(adapter.features());
        if !features.is_empty() {
            let (native_device, native_queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_features: features,
                    ..Default::default()
                }))
                .unwrap();
            verify_ktx_mips(
                &native_device,
                &native_queue,
                &image,
                preferred_texture_encoding(features),
            );
        }
    }
}

fn verify_ktx_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &TextureImage,
    expected: TextureEncoding,
) {
    let uploaded = upload_image(device, queue, image, TextureRole::BaseColor).unwrap();
    assert_eq!(uploaded.texture.mip_level_count(), 6);
    assert_eq!(
        uploaded.texture.format(),
        ornis_render::texture_upload::gpu_texture_format(expected, TextureRole::BaseColor)
    );
    let reference = image
        .prepare(TextureRole::BaseColor, TextureEncoding::Rgba8)
        .unwrap();
    for (mip, rgba) in reference.levels().iter().enumerate() {
        let pixels = sample(
            device,
            queue,
            &uploaded,
            &uploaded.view,
            (40 >> mip).max(1),
            Some(mip as u32),
        );
        let error: u64 = pixels
            .iter()
            .zip(rgba)
            .map(|(&a, &b)| a.abs_diff(b) as u64)
            .sum();
        let mean = error as f64 / rgba.len() as f64;
        let tolerance = if expected == TextureEncoding::Rgba8 {
            1.0
        } else {
            12.0
        };
        assert!(
            mean <= tolerance,
            "{expected:?} mip {mip}: mean channel error {mean}"
        );
    }
    eprintln!(
        "KTX2 {:?}: {} uploaded payload bytes",
        uploaded.texture.format(),
        uploaded.payload_bytes
    );
    let mut cache = ornis_render::TextureCache::new();
    let a = cache
        .upload_image_cached(device, queue, image, TextureRole::BaseColor)
        .unwrap();
    let b = cache
        .upload_image_cached(device, queue, image, TextureRole::BaseColor)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(cache.len(), 1);
}
