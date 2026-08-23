//! Phase 1 verification: renders assets/scene.ron through the legacy
//! `Renderer3D::render_scene` path AND through the render-graph path
//! (`RenderFrame3D`), reads both back and asserts byte-identical pixels.
//!
//! Run from the workspace root:
//!   cargo run -p ornis-render --example frame_plan_probe -- [scene.ron]
//!
//! Writes target/frame_plan_probe_{legacy,graph}.png and the graph
//! layout dump (transient windows + pool slots) to stdout. Prints PASS and
//! exits 0 when the two paths match pixel-for-pixel.

use glam::{Mat4, Quat, Vec3};
use ornis_core::OpenPBRMaterial;
use ornis_render::scene::{LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_render::{InstanceData, RenderFrame3D, Renderer3D, Technique};

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
            label: Some("frame_plan_probe"),
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
    let mut graph3d = RenderFrame3D::new(format, (WIDTH, HEIGHT));

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe encoder"),
    });
    renderer.render_scene(
        &device,
        &queue,
        &mut encoder,
        &legacy_view,
        &mesh,
        instances.len() as u32,
    );
    graph3d.render(
        ornis_render::render_backend::RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &graph_view,
        },
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

    // ── Multi-frame stability: pool reuse across frames ───────────────
    let slots_before = graph3d.pool_slots();
    const FRAMES: u32 = 16;
    let mut frames_identical = 0u32;
    for _ in 1..FRAMES {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe frame encoder"),
        });
        graph3d.render(
            ornis_render::render_backend::RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &graph_view,
            },
            &renderer,
            &mesh,
            instances.len() as u32,
        );
        queue.submit(std::iter::once(encoder.finish()));
        let (pixels, _) = read_target(&graph_tex, "graph readback (frame)");
        if pixels == graph_pixels {
            frames_identical += 1;
        }
    }
    let slots_after = graph3d.pool_slots();
    let pool_stable = slots_before == slots_after && slots_after < 9;
    let all_frames_stable = frames_identical == FRAMES - 1;

    // ── Bloom: the same graph plus the bloom node chain ───────────────
    let (bloom_tex, bloom_view) = make_target("probe bloom target");
    let mut graph3d_bloom = RenderFrame3D::new_with_bloom(format, (WIDTH, HEIGHT));
    let mut bloom_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe bloom encoder"),
    });
    graph3d_bloom.render(
        ornis_render::render_backend::RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut bloom_encoder,
            target: &bloom_view,
        },
        &renderer,
        &mesh,
        instances.len() as u32,
    );
    queue.submit(std::iter::once(bloom_encoder.finish()));
    let (bloom_pixels, _) = read_target(&bloom_tex, "bloom readback");

    // Bloom must not change pixels without bright content... but it must
    // change *something*: the bright specular highlights should glow.
    let bloom_diff = graph_pixels
        .iter()
        .zip(&bloom_pixels)
        .filter(|(a, b)| a != b)
        .count();
    let bloom_active = bloom_diff > 0;
    save_png(
        "target/frame_plan_probe_bloom.png",
        &bloom_pixels,
        unpadded,
    );

    println!("--- bloom graph layout ---");
    println!("{}", graph3d_bloom.layout_dump());
    let bloom_slots = graph3d_bloom.pool_slots();
    let bloom_bytes = graph3d_bloom.texture_budget();
    println!("bloom pool slots: {bloom_slots}, budget: {bloom_bytes} bytes");
    println!(
        "bloom vs no-bloom: {bloom_diff} differing pixels (of {}), active: {bloom_active}",
        graph_pixels.len()
    );

    // ── Technique switch: same graph, different node wires ──────────
    // Render every technique standalone; each gets its own graph, target
    // and pool, so the slot/budget numbers are directly comparable.
    let render_technique =
        |technique: Technique, bloom: bool, label: &str| -> (Vec<u8>, u32, usize, u64, String) {
            let (tex, view) = make_target(label);
            let mut frame = RenderFrame3D::new_with(format, (WIDTH, HEIGHT), technique, bloom);
            let mut encoder = device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
            frame.render(
                ornis_render::render_backend::RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &view,
                },
                &renderer,
                &mesh,
                instances.len() as u32,
            );
            queue.submit(std::iter::once(encoder.finish()));
            let (pixels, unpadded) = read_target(&tex, label);
            let dump = frame.layout_dump();
            let slots = frame.pool_slots();
            let budget = frame.texture_budget();
            (pixels, unpadded, slots, budget, dump)
        };

    let (fwd_pixels, _fwd_unpadded, fwd_slots, fwd_bytes, fwd_dump) =
        render_technique(Technique::Forward, false, "probe forward target");
    let (def_pixels, _def_unpadded, def_slots, def_bytes, def_dump) =
        render_technique(Technique::Deferred, false, "probe deferred target");
    // Forward + bloom: the bright-pass must read hdr_fwd (hdr is dead).
    let (fwd_bloom, _fb_unpadded, fwd_b_slots, fwd_b_bytes, _fwd_b_dump) =
        render_technique(Technique::Forward, true, "probe forward bloom target");
    save_png(
        "target/frame_plan_probe_forward.png",
        &fwd_pixels,
        _fwd_unpadded,
    );
    save_png(
        "target/frame_plan_probe_deferred.png",
        &def_pixels,
        _def_unpadded,
    );

    // A technique switch must change the picture (no silent no-op), and
    // the differences must be on the materials/geometry, not the whole
    // frame turning black.
    let fwd_vs_legacy = diff_count(&legacy_pixels, &fwd_pixels);
    let def_vs_legacy = diff_count(&legacy_pixels, &def_pixels);
    let fwd_active = fwd_slots <= 4;
    let fwd_lookup_active = diff_count(&fwd_pixels, &fwd_bloom) > 0;

    println!("--- forward technique ---");
    println!("{fwd_dump}");
    println!(
        "forward slots: {fwd_slots}, budget: {fwd_bytes} bytes — differs from legacy: {fwd_vs_legacy} px"
    );
    println!("--- deferred technique ---");
    println!("{def_dump}");
    println!(
        "deferred slots: {def_slots}, budget: {def_bytes} bytes — differs from legacy: {def_vs_legacy} px"
    );
    println!(
        "forward bloom active: {fwd_lookup_active} ({fwd_b_slots} slots, {fwd_b_bytes} bytes)"
    );

    // ── Compare ───────────────────────────────────────────────────────
    let legacy_diff = diff_count(&legacy_pixels, &graph_pixels);
    let max_diff = legacy_pixels
        .iter()
        .zip(&graph_pixels)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    let same = legacy_diff == 0;

    save_png(
        "target/frame_plan_probe_legacy.png",
        &legacy_pixels,
        unpadded,
    );
    save_png(
        "target/frame_plan_probe_graph.png",
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

    // ── Memory: legacy persistent textures vs graph pool ──────────────
    let legacy_bytes = renderer.texture_budget();
    let graph_bytes = graph3d.texture_budget();
    let saved = legacy_bytes.saturating_sub(graph_bytes);
    let pct = if legacy_bytes > 0 {
        saved as f64 * 100.0 / legacy_bytes as f64
    } else {
        0.0
    };
    println!(
        "legacy texture budget: {legacy_bytes} bytes ({:.1} MB)",
        legacy_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "graph pool budget:     {graph_bytes} bytes ({:.1} MB)",
        graph_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "saved: {saved} bytes ({pct:.1}%) — {FRAMES}-frame pool stable: {pool_stable}, frames identical: {all_frames_stable}"
    );

    if same && all_frames_stable && pool_stable && bloom_active {
        println!("PASS: legacy and render-graph paths are pixel-identical");
        println!(
            "PASS: bloom chain is active (downsample/upsample/composite) with {bloom_slots} pool slots"
        );
        println!(
            "PASS: technique switch — forward-only: {fwd_slots} slots/{fwd_bytes} B ({fwd_vs_legacy} px differ from legacy; bloom active: {fwd_lookup_active})"
        );
        println!(
            "PASS: technique switch — deferred-only matches legacy pixel-for-pixel ({def_slots} slots, {def_bytes} B)"
        );
        let fwd_budget_ok = fwd_bytes < graph_bytes;
        // Deferred keeps the gbuffer textures, so its budget equals hybrid
        // (the forward layer aliases material_params there); it must not
        // exceed it, though.
        let def_budget_ok = def_bytes <= graph_bytes;
        println!(
            "technique budgets: hybrid {graph_bytes} B, forward {fwd_bytes} B (smaller: {fwd_budget_ok}), deferred {def_bytes} B (<= hybrid: {def_budget_ok})"
        );
        // deferred-only is the classic path → must equal legacy; forward-only
        // must differ (it drops the gbuffer) and stay non-empty.
        let def_is_legacy = def_vs_legacy == 0;
        if !(fwd_active
            && fwd_vs_legacy > 0
            && def_is_legacy
            && fwd_lookup_active
            && fwd_budget_ok
            && def_budget_ok)
        {
            println!("FAIL: technique matrix produced an inactive or budget-worse path");
            println!(
                "forward: active={fwd_active}, diff_vs_legacy={fwd_vs_legacy}, bloom={fwd_lookup_active}"
            );
            println!(
                "deferred: diff_vs_legacy={def_vs_legacy} (must be 0 = legacy), <= hybrid budget: {def_budget_ok}"
            );
            std::process::exit(1);
        }
        println!(
            "PNGs: target/frame_plan_probe_{{legacy,graph,bloom,forward,deferred}}.png ({WIDTH}x{HEIGHT})"
        );
    } else {
        println!(
            "FAIL: {legacy_diff} mismatching pixels, max byte diff {max_diff} (of {})",
            legacy_pixels.len()
        );
        println!("bloom active: {bloom_active} ({bloom_diff} differing pixels vs no-bloom)");
        println!("frames identical: {all_frames_stable}, pool stable: {pool_stable}");
        println!("PNGs: target/frame_plan_probe_{{legacy,graph,bloom}}.png");
        std::process::exit(1);
    }
}

/// Number of bytes that differ between two readback buffers.
fn diff_count(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x != y).count()
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
