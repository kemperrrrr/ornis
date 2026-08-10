//! Phase 1 verification: renders assets/scene.ron through the legacy
//! `Renderer3D::render_scene` path AND through the render-graph path
//! (`RenderGraph3D`), reads both back and asserts byte-identical pixels.
//!
//! Run from the workspace root:
//!   cargo run -p ornis-render --example render_graph_probe -- [scene.ron]
//!
//! Writes target/render_graph_probe_{legacy,graph}.png and the graph
//! layout dump (transient windows + pool slots) to stdout. Prints PASS and
//! exits 0 when the two paths match pixel-for-pixel.

use glam::{Mat4, Quat, Vec3};
use ornis_core::OpenPBRMaterial;
use ornis_render::scene::{LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_render::{InstanceData, RenderGraph3D, Renderer3D};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const BYTES_PER_PIXEL: u32 = 4;

fn build_material(entity_material: &MaterialDesc) -> OpenPBRMaterial {
    match entity_material {
        MaterialDesc::Dielectric {
            base_color,
            roughness,
        } => OpenPBRMaterial::dielectric()
            .base_color_rgb(*base_color)
            .specular_roughness(*roughness),
        MaterialDesc::Metal {
            base_color,
            roughness,
        } => OpenPBRMaterial::metal()
            .base_color_rgb(*base_color)
            .specular_roughness(*roughness),
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
        } => OpenPBRMaterial::coat()
            .base_color_rgb(*base_color)
            .coat_weight(*coat_weight)
            .coat_roughness(*coat_roughness),
    }
}

fn main() {
    let scene_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "assets/scene.ron".to_string());
    let ron_text = std::fs::read_to_string(&scene_path).expect("read scene.ron");
    let scene = Scene::from_ron(&ron_text).expect("parse scene.ron");
    println!(
        "scene '{}': {} entities, {} lights",
        scene.name,
        scene.entities.len(),
        scene.lights.len()
    );
    pollster::block_on(run(&scene));
}

async fn run(scene: &Scene) {
    // ── Headless device ───────────────────────────────────────────────
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("adapter");
    println!("adapter: {:?}", adapter.get_info().name);

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("render_graph_probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("device");

    // ── Two identical offscreen targets ───────────────────────────────
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let make_target = |label: &str| {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    };
    let (legacy_tex, legacy_view) = make_target("probe legacy target");
    let (graph_tex, graph_view) = make_target("probe graph target");

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };

    let renderer = Renderer3D::new(&device, &surface_config, 1);

    // ── Scene → GPU data ──────────────────────────────────────────────
    let first = scene.entities.first().expect("scene has no entities");
    let mesh = match &first.mesh {
        MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => ornis_render::create_sphere(&device, *radius, *segments, *rings),
    };

    let mut materials = Vec::new();
    let mut instances = Vec::new();
    for (i, entity) in scene.entities.iter().enumerate() {
        materials.push(build_material(&entity.material));
        let t = &entity.transform;
        let model = Mat4::from_scale_rotation_translation(
            Vec3::from(t.scale),
            Quat::from_xyzw(t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]).normalize(),
            Vec3::from(t.translation),
        );
        let normal_matrix = model.inverse().transpose();
        instances.push(InstanceData {
            model_matrix: model,
            normal_matrix,
            material_index: i as u32,
        });
    }
    let lights: Vec<([f32; 3], f32, [f32; 3])> = scene
        .lights
        .iter()
        .map(|l| match l {
            LightDesc::Directional {
                direction,
                intensity,
                color,
            } => (*direction, *intensity, *color),
        })
        .collect();

    renderer.upload_materials(&queue, &materials);
    renderer.upload_instances(&queue, &instances);
    renderer.set_lights(&queue, scene.ambient, &lights);

    let cam = &scene.camera;
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view = Mat4::look_at_rh(
        Vec3::from(cam.position),
        Vec3::from(cam.target),
        Vec3::from(cam.up),
    );
    let proj = Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.near, cam.far);
    let view_proj = proj * view;
    renderer.set_camera(&queue, &view_proj.to_cols_array_2d(), cam.position);

    // ── Render: legacy path, then graph path ──────────────────────────
    let mut graph3d = RenderGraph3D::new(format, (WIDTH, HEIGHT));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe encoder"),
    });
    renderer.render_scene(
        &device,
        &mut encoder,
        &legacy_view,
        &mesh,
        instances.len() as u32,
    );
    graph3d.render(
        &device,
        &mut encoder,
        &graph_view,
        &renderer,
        &mesh,
        instances.len() as u32,
    );
    let command_buffer = encoder.finish();
    queue.submit(std::iter::once(command_buffer));

    // ── Readback both targets ─────────────────────────────────────────
    let read_target = |texture: &wgpu::Texture, label: &str| {
        let unpadded = WIDTH * BYTES_PER_PIXEL;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (padded * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(HEIGHT),
                },
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map readback"));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range();
        let mut pixels = vec![0u8; (unpadded * HEIGHT) as usize];
        for y in 0..HEIGHT as usize {
            let src = &data[y * padded as usize..][..unpadded as usize];
            pixels[y * unpadded as usize..][..unpadded as usize].copy_from_slice(src);
        }
        drop(data);
        readback.unmap();
        (pixels, unpadded)
    };

    let (legacy_pixels, unpadded) = read_target(&legacy_tex, "legacy readback");
    let (graph_pixels, _) = read_target(&graph_tex, "graph readback");

    // ── Compare ───────────────────────────────────────────────────────
    let diff_count = legacy_pixels
        .iter()
        .zip(&graph_pixels)
        .filter(|(a, b)| a != b)
        .count();
    let max_diff = legacy_pixels
        .iter()
        .zip(&graph_pixels)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    let same = diff_count == 0;

    save_png(
        "target/render_graph_probe_legacy.png",
        &legacy_pixels,
        unpadded,
    );
    save_png(
        "target/render_graph_probe_graph.png",
        &graph_pixels,
        unpadded,
    );

    println!("--- graph layout ---");
    println!("{}", graph3d.layout_dump());
    println!(
        "pool slots: {} (aliasing win over {} declared resources)",
        graph3d.pool_slots(),
        9
    );

    if same {
        println!("PASS: legacy and render-graph paths are pixel-identical");
        println!("PNGs: target/render_graph_probe_{{legacy,graph}}.png ({WIDTH}x{HEIGHT})");
    } else {
        println!(
            "FAIL: {diff_count} mismatching pixels, max byte diff {max_diff} (of {})",
            legacy_pixels.len()
        );
        println!("PNGs: target/render_graph_probe_{{legacy,graph}}.png");
        std::process::exit(1);
    }
}

fn save_png(path: &str, pixels: &[u8], unpadded: u32) {
    let file = std::fs::File::create(path).expect("create png");
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), WIDTH, HEIGHT);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer
        .write_image_data(&pixels[..(unpadded * HEIGHT) as usize])
        .expect("png data");
    println!("saved {path}");
}
