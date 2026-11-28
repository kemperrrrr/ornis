//! Production skinning benchmark: the skinned g-buffer vertex stage.
//!
//! The translated GPU skinning (`crates/gpu-skinning`, artifact
//! `skinning_generated.wgsl`) is the vertex stage of every skinned draw in
//! production, and until now no benchmark measured it — only the research
//! wasm2gpu guest had one. This draws the stage over a grid of skinned
//! instances into an off-screen target and reports wall-clock and, when the
//! adapter exposes them, GPU timestamp medians.
//!
//! The instance palette carries one animated pose (a swing on the shoulder
//! joint and the hip), so the vertex work includes the palette blend, not just
//! the transform. Run (ignored by default):
//!
//! ```text
//! cargo test -p ornis-render --test skinning_bench -- --ignored --nocapture
//! ```

use std::time::Duration;

use glam::{Mat4, Quat, Vec3};
use ornis_render::mesh::SkinnedVertex;
use ornis_render::skinning::joint_palette_bytes;
use ornis_render::skinning::wgsl_vertex_source_skinned;
use ornis_render::{MaterialIdx, PerObjectGpu};

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// Instances drawn; the `anim` mannequin is ~17k vertices each.
const INSTANCES: u32 = 64;
const VERTICES_PER_INSTANCE: u32 = 1024;
const JOINTS: usize = 24;
const WARMUP: u32 = 10;
const SAMPLES: u32 = 30;
const QUERY_BYTES: u64 = 2 * std::mem::size_of::<u64>() as u64;

#[ignore = "manual M1 benchmark; run with --ignored --nocapture"]
#[test]
fn bench_production_gpu_skinning() {
    let Some((device, queue, timed)) = open() else {
        eprintln!("SKIP: no wgpu adapter");
        return;
    };
    let (vertex_bytes, index_count) = (skinned_mesh(), skinned_indices().len() as u32);
    let vertex_buffer = upload(&device, &queue, "skinning vertices", &vertex_bytes);
    let index_bytes: Vec<u8> = skinned_indices()
        .iter()
        .flat_map(|index| index.to_le_bytes())
        .collect();
    let index_buffer = upload(&device, &queue, "skinning indices", &index_bytes);
    let camera = upload(&device, &queue, "skinning camera", &camera_bytes());
    let mut objects = Vec::new();
    for instance in 0..INSTANCES {
        let x = (instance % 8) as f32 * 2.5 - 7.0;
        let y = (instance / 8) as f32 * 2.5 - 7.0;
        let model = Mat4::from_translation(Vec3::new(x, y, 0.0));
        let mut record = PerObjectGpu::default();
        record.model = model.to_cols_array_2d();
        record.previous_model = model.to_cols_array_2d();
        record.material_index = MaterialIdx::from_raw(0);
        objects.extend_from_slice(bytemuck::bytes_of(&record));
    }
    let objects = upload(&device, &queue, "skinning objects", &objects);
    let mut palettes = Vec::new();
    let mut previous = Vec::new();
    for instance in 0..INSTANCES {
        let palette = pose(instance as f32 * 0.05);
        palettes.extend_from_slice(&joint_palette_bytes(&palette).expect("palette"));
        let previous_pose = pose(-0.05);
        previous.extend_from_slice(&joint_palette_bytes(&previous_pose).expect("palette"));
    }
    let palette = upload(&device, &queue, "skinning palette", &palettes);
    let previous_palette = upload(&device, &queue, "skinning previous", &previous);

    let target = target(&device);
    let queries = timed.then(|| {
        device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("skinning bench queries"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        })
    });
    let resolve = queries.as_ref().map(|_| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning bench resolve"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let readback = resolve.as_ref().map(|_resolve| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skinning bench readback"),
            size: QUERY_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });

    let period_ns = queue.get_timestamp_period();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinning bench vertex"),
        source: wgpu::ShaderSource::Wgsl(wgsl_vertex_source_skinned().into()),
    });
    // The production skinned stage shares the classic fragment; for a
    // vertex-stage bench the fragment is a trivial passthrough, so the numbers
    // are the skinning, not the shading.
    let fragment = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("skinning bench fragment"),
        source: wgpu::ShaderSource::Wgsl(FRAGMENT.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skinning bench layout"),
        entries: &[
            uniform(0),
            storage(1, true),
            storage(3, true),
            storage(4, true),
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skinning bench"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: objects.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: palette.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: previous_palette.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("skinning bench"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("skinning bench"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some(ornis_render::skinning::SKINNED_ENTRY),
            buffers: &[Some(SkinnedVertex::desc())],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment,
            entry_point: Some("fs_bench"),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let mut wall = Vec::new();
    let mut gpu_ns = Vec::new();
    for run in 0..WARMUP + SAMPLES {
        let started = std::time::Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("skinning bench"),
        });
        let pass_writes = queries
            .as_ref()
            .map(|queries| wgpu::RenderPassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("skinning bench"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: pass_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..index_count, 0, 0..INSTANCES);
        }
        if let (Some(queries), Some(resolve), Some(readback)) = (&queries, &resolve, &readback) {
            encoder.resolve_query_set(queries, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, QUERY_BYTES);
        }
        queue.submit(std::iter::once(encoder.finish()));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if run >= WARMUP {
            wall.push(started.elapsed());
        }
        if let Some(readback) = &readback {
            let slice = readback.slice(..QUERY_BYTES);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let view = slice.get_mapped_range().expect("mapped");
            let stamps: &[u64] = bytemuck::cast_slice(&view);
            let Some(delta) = stamps.get(1).zip(stamps.first()).map(|(a, b)| a - b) else {
                continue;
            };
            drop(view);
            readback.unmap();
            if run >= WARMUP {
                gpu_ns.push(delta as f64 * f64::from(period_ns));
            }
        }
    }
    println!(
        "production GPU skinning: {INSTANCES} instances × {VERTICES_PER_INSTANCE} vertices, {JOINTS} joints"
    );
    println!("wall median: {:.2} µs", median_us(&wall));
    if !gpu_ns.is_empty() {
        let total = f64::from(INSTANCES) * f64::from(VERTICES_PER_INSTANCE);
        println!(
            "gpu median: {:.0} ns ({:.1} Mverts/s)",
            median(&gpu_ns),
            total / (median(&gpu_ns) / 1.0e9) / 1.0e6
        );
    }
}

/// The trivial passthrough the bench pairs with the skinned vertex stage.
const FRAGMENT: &str = "
@fragment
fn fs_bench(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(position.z, position.z, position.z, 1.0);
}
";

/// One layout entry per binding kind.
fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// One read-only storage binding.
fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A headless device; `timed` says whether timestamps are available.
fn open() -> Option<(wgpu::Device, wgpu::Queue, bool)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(),
        backend_options: Default::default(),
        display: None,
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .ok()?;
    let wanted = wgpu::Features::TIMESTAMP_QUERY;
    let features = wanted & adapter.features();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("skinning bench"),
        required_features: features,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .ok()?;
    let timed = features.contains(wgpu::Features::TIMESTAMP_QUERY);
    Some((device, queue, timed))
}

/// An RGBA8 target of the bench size.
fn target(device: &wgpu::Device) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("skinning bench target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Upload `bytes` and return the buffer.
fn upload(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::INDEX
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// The camera uniform: one view-projection over the grid.
///
/// The record is four matrices ([`CameraUniform`]); slots 2..4 stay zero.
fn camera_bytes() -> Vec<u8> {
    let projection =
        glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
    let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 4.0, 26.0), Vec3::ZERO, Vec3::Y);
    let view_proj = projection * view;
    let mut bytes = Vec::with_capacity(208);
    bytes.extend_from_slice(bytemuck::bytes_of(&view_proj.to_cols_array()));
    bytes.extend_from_slice(bytemuck::bytes_of(&view_proj.inverse().to_cols_array()));
    bytes.extend_from_slice(bytemuck::bytes_of(&[0.0f32, 4.0, 26.0, 1.0]));
    bytes.extend_from_slice(bytemuck::bytes_of(&view_proj.to_cols_array()));
    bytes
}

/// One pose of the palette: a swing on the shoulder and hip joints.
fn pose(phase: f32) -> Vec<Mat4> {
    let mut palette = vec![Mat4::IDENTITY; JOINTS];
    for (joint, slot) in palette.iter_mut().enumerate().skip(1) {
        let angle = phase * joint as f32;
        *slot =
            Mat4::from_rotation_translation(Quat::from_rotation_z(angle), Vec3::new(0.1, 0.0, 0.0));
    }
    palette
}

/// A grid of skinned triangles sharing one palette: bind pose plus weights
/// pinned to four joints, matching the staging contract. Returns the
/// interleaved vertex bytes and the index count.
fn skinned_mesh() -> Vec<u8> {
    let mut vertices = Vec::with_capacity(VERTICES_PER_INSTANCE as usize);
    for index in 0..VERTICES_PER_INSTANCE {
        let row = index / 32;
        let column = index % 32;
        let position = Vec3::new(column as f32 * 0.05, row as f32 * 0.05, 0.0);
        let joints = [
            u32::from(index % 3 == 0),
            u32::from(index % 3 == 1),
            u32::from(index % 3 == 2),
            0,
        ];
        let weights = [0.4f32, 0.3, 0.3, 0.0];
        vertices.push(SkinnedVertex {
            position: position.to_array(),
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            joints,
            weights,
        });
    }
    vertices
        .iter()
        .flat_map(|vertex| bytemuck::bytes_of(vertex).to_vec())
        .collect()
}

/// The grid indices: two triangles per cell of the 31×31 mesh.
fn skinned_indices() -> Vec<u32> {
    let mut indices = Vec::new();
    for row in 0..31 {
        for column in 0..31 {
            let base = row * 32 + column;
            indices.extend_from_slice(&[base, base + 1, base + 32, base + 1, base + 33, base + 32]);
        }
    }
    indices
}

/// Median of a float list.
fn median(values: &[f64]) -> f64 {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    sorted[sorted.len() / 2]
}

/// Median wall time in microseconds.
fn median_us(values: &[std::time::Duration]) -> f64 {
    median(&values.iter().map(Duration::as_secs_f64).collect::<Vec<_>>()) * 1.0e6
}
