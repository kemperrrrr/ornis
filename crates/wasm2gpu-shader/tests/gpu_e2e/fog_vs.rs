//! Second live shader, fog vertex stage, through wasm2gpu.
//!
//! The production fog `vs_main` — fullscreen-quad corner/uv fetch by
//! `vertex_index`, `ornis_render::shaders::fog_generated::
//! wgsl_vertex_source()` — runs VERBATIM in pipeline A. The wasm guest
//! carries the same const tables (inputs shared by both sides), is
//! lifted to a compute kernel and writes `{ pos, uv }` per vertex into
//! storage. Pipeline B re-emits those outputs through a thin adapter
//! vertex stage; one shared fragment stage stores the rasterized
//! `position.xy` + `uv` per pixel. The readbacks must be bitwise
//! identical, so the interpolated field is the comparison channel: any
//! divergence between the lifted kernel and the current shader lands
//! in the bits.

use std::borrow::Cow;

use ornis_gpu_abi::symbols;
use ornis_wasm2gpu::translate;

use super::compute_lift::lift_wat;
use super::{compile_wasm, device, map_bytes, storage_buffer};

/// Side of the drawn quad in pixels: one fragment per texel of capture.
const SIZE: u32 = 8;

/// Vertices in the fullscreen strip.
const VERTICES: u32 = 4;

/// Bytes of one kernel vertex output (32-byte WGSL struct stride).
const STRIDE: u64 = 32;

/// Shared fragment stage: park what the rasterizer hands it in storage.
const CAPTURE: &str = "
struct QuadOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
@group(0) @binding(0) var<storage, read_write> pixels: array<vec4<f32>>;
@fragment
fn fs_main(input: QuadOut) {
    let idx = u32(input.clip_position.y) * 8u + u32(input.clip_position.x);
    pixels[idx] = vec4<f32>(
        input.clip_position.x,
        input.clip_position.y,
        input.uv.x,
        input.uv.y,
    );
}
";

/// Pipeline B's vertex stage: re-emit the kernel's per-vertex outputs.
const ADAPTER: &str = "
struct QuadOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}
struct KernelVert {
    pos: vec4<f32>,
    uv: vec2<f32>,
}
@group(1) @binding(0) var<storage, read> kernel_out: array<KernelVert>;
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadOut {
    let k = kernel_out[vertex_index];
    return QuadOut { clip_position: k.pos, uv: k.uv };
}
";

/// Lift the `vs_main` guest export and pin what the lowering must keep.
fn kernel_wgsl() -> String {
    let wasm = compile_wasm();
    let wat = lift_wat(&wasm, "vs_main");
    let module = wat::parse_str(&wat).unwrap_or_else(|err| panic!("vs_main wat: {err}\n{wat}"));
    let lowered = match translate(&module) {
        Ok(lowered) => lowered,
        Err(err) => panic!("vs_main translate {err:?}\nWAT:\n{wat}"),
    };
    let wgsl = &lowered.wgsl;
    assert!(wgsl.contains("@compute"), "{wgsl}");
    assert!(wgsl.contains("@builtin(global_invocation_id)"), "{wgsl}");
    assert!(wgsl.contains("storage_0_0_"), "{wgsl}");
    assert!(!lowered.compute_spirv.is_empty());
    wgsl.clone()
}

/// Render pipeline over two vertex modules (production or adapter) and
/// the shared storage-capturing fragment stage; no color outputs.
fn capture_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vs: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let fs = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs capture"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(CAPTURE)),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("fog-vs"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vs,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs,
            entry_point: Some("fs_main"),
            targets: &[],
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
    })
}

/// Copy `words` u32 words out of a GPU buffer and return them.
fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    words: u32,
) -> Vec<u32> {
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fog-vs read"),
        size: u64::from(words) * 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(src, 0, &read, 0, u64::from(words) * 4);
    queue.submit([encoder.finish()]);
    let bytes = map_bytes(device, &read);
    let mut out = Vec::with_capacity(words as usize);
    let mut index = 0usize;
    while index < words as usize {
        let mut chunk = [0u8; 4];
        chunk.copy_from_slice(&bytes[index * 4..index * 4 + 4]);
        out.push(u32::from_le_bytes(chunk));
        index += 1;
    }
    out
}

/// One storage layout entry at the given visibility.
fn layout_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[test]
fn fog_vertex_kernel_matches_the_production_stage() {
    let Some((device, queue)) = device() else {
        if std::env::var_os("ORNIS_REQUIRE_GPU").is_none() {
            eprintln!("no adapter; skipping (set ORNIS_REQUIRE_GPU to fail instead)");
            return;
        }
        panic!("ORNIS_REQUIRE_GPU is set but no adapter is available");
    };

    let wgsl = kernel_wgsl();
    let kernel = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs kernel"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl)),
    });
    let production = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs production"),
        source: wgpu::ShaderSource::Wgsl(Cow::Owned(
            ornis_render::shaders::fog_generated::wgsl_vertex_source(),
        )),
    });
    let adapter = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fog-vs adapter"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADAPTER)),
    });

    let storage = wgpu::BufferBindingType::Storage { read_only: false };
    let pixels_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog-vs pixels"),
        entries: &[layout_entry(0, storage, wgpu::ShaderStages::FRAGMENT)],
    });
    let out_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog-vs kernel out"),
        entries: &[layout_entry(
            0,
            wgpu::BufferBindingType::Storage { read_only: true },
            wgpu::ShaderStages::VERTEX,
        )],
    });
    let kernel_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fog-vs kernel"),
        entries: &[layout_entry(0, storage, wgpu::ShaderStages::COMPUTE)],
    });

    let layout_a = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fog-vs a"),
        bind_group_layouts: &[&pixels_layout],
        push_constant_ranges: &[],
    });
    let layout_b = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fog-vs b"),
        bind_group_layouts: &[&pixels_layout, &out_layout],
        push_constant_ranges: &[],
    });
    let pipeline_a = capture_pipeline(&device, &layout_a, &production);
    let pipeline_b = capture_pipeline(&device, &layout_b, &adapter);
    let kernel_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fog-vs kernel"),
        layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fog-vs kernel"),
            bind_group_layouts: &[&kernel_layout],
            push_constant_ranges: &[],
        })),
        module: &kernel,
        entry_point: Some(symbols::CS_MAIN),
        compilation_options: Default::default(),
        cache: None,
    });

    let out_bytes = STRIDE * u64::from(VERTICES);
    let pixels_bytes = u64::from(SIZE * SIZE) * 16;
    let kernel_out = storage_buffer(&device, "fog-vs kernel out", out_bytes);
    let pixels_a = storage_buffer(&device, "fog-vs pixels a", pixels_bytes);
    let pixels_b = storage_buffer(&device, "fog-vs pixels b", pixels_bytes);
    queue.write_buffer(&kernel_out, 0, &vec![0u8; out_bytes as usize]);
    queue.write_buffer(&pixels_a, 0, &vec![0u8; pixels_bytes as usize]);
    queue.write_buffer(&pixels_b, 0, &vec![0u8; pixels_bytes as usize]);

    let group_kernel = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog-vs kernel"),
        layout: &kernel_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: kernel_out.as_entire_binding(),
        }],
    });
    let group_pixels_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog-vs pixels a"),
        layout: &pixels_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pixels_a.as_entire_binding(),
        }],
    });
    let group_pixels_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog-vs pixels b"),
        layout: &pixels_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: pixels_b.as_entire_binding(),
        }],
    });
    let group_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog-vs adapter out"),
        layout: &out_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: kernel_out.as_entire_binding(),
        }],
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog-vs target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fog-vs kernel"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel_pipeline);
        pass.set_bind_group(0, &group_kernel, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    let draw = wgpu::RenderPassDescriptor {
        label: Some("fog-vs draw"),
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
    };
    {
        let mut pass = encoder.begin_render_pass(&draw);
        pass.set_pipeline(&pipeline_a);
        pass.set_bind_group(0, &group_pixels_a, &[]);
        pass.draw(0..VERTICES, 0..1);
    }
    {
        let mut pass = encoder.begin_render_pass(&draw);
        pass.set_pipeline(&pipeline_b);
        pass.set_bind_group(0, &group_pixels_b, &[]);
        pass.set_bind_group(1, &group_out, &[]);
        pass.draw(0..VERTICES, 0..1);
    }
    queue.submit([encoder.finish()]);

    let words = SIZE * SIZE * 4;
    let production_pixels = read_words(&device, &queue, &pixels_a, words);
    let kernel_pixels = read_words(&device, &queue, &pixels_b, words);
    assert!(
        kernel_pixels.iter().any(|word| *word != 0),
        "the lifted kernel wrote nothing"
    );
    let mut index = 0usize;
    while index < production_pixels.len() {
        assert!(
            production_pixels[index] == kernel_pixels[index],
            "pixel word {index} differs: production {:08x} vs kernel {:08x}",
            production_pixels[index],
            kernel_pixels[index],
        );
        index += 1;
    }
}
