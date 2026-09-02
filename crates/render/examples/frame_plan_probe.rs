#![allow(deprecated)]
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
use ornis_render::scene::{CameraDesc, LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_render::{InstanceData, RenderFrame3D, Renderer3D, Technique};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const BYTES_PER_PIXEL: u32 = 4;

fn build_material(entity_material: &MaterialDesc) -> OpenPBRMaterial {
    match entity_material {
        MaterialDesc::Dielectric {
            base_color,
            roughness,
        } => {
            let mut mat = OpenPBRMaterial::dielectric();
            mat.base.color_rgb(*base_color);
            mat.specular.roughness(*roughness);
            mat
        }
        MaterialDesc::Metal {
            base_color,
            roughness,
        } => {
            let mut mat = OpenPBRMaterial::metal();
            mat.base.color_rgb(*base_color);
            mat.specular.roughness(*roughness);
            mat
        }
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
        } => {
            let mut mat = OpenPBRMaterial::coat();
            mat.base.color_rgb(*base_color);
            mat.coat.weight(*coat_weight);
            mat.coat.roughness(*coat_roughness);
            mat
        }
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

/// Headless adapter + device for offscreen probing.
async fn create_headless_device(label: &str) -> (wgpu::Device, wgpu::Queue) {
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
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("device");
    (device, queue)
}

fn surface_config_for(format: wgpu::TextureFormat) -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

fn make_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
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
}

/// Shared mesh from the first entity plus per-entity material/instance data.
fn build_scene_data(
    device: &wgpu::Device,
    scene: &Scene,
) -> (ornis_render::Mesh, Vec<OpenPBRMaterial>, Vec<InstanceData>) {
    let first = scene.entities.first().expect("scene has no entities");
    let mesh = match &first.mesh {
        MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => ornis_render::create_sphere(device, *radius, *segments, *rings),
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
    (mesh, materials, instances)
}

fn camera_view_proj(cam: &CameraDesc) -> [[f32; 4]; 4] {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view = Mat4::look_at_rh(
        Vec3::from(cam.position),
        Vec3::from(cam.target),
        Vec3::from(cam.up),
    );
    let proj = Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.near, cam.far);
    (proj * view).to_cols_array_2d()
}

async fn read_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    label: &str,
) -> (Vec<u8>, u32) {
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
    let data = slice.get_mapped_range().unwrap();
    let mut pixels = vec![0u8; (unpadded * HEIGHT) as usize];
    for y in 0..HEIGHT as usize {
        let src = &data[y * padded as usize..][..unpadded as usize];
        pixels[y * unpadded as usize..][..unpadded as usize].copy_from_slice(src);
    }
    drop(data);
    readback.unmap();
    (pixels, unpadded)
}

/// Shared probe state: device/queue plus the uploaded scene.
struct Probe<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    renderer: &'a Renderer3D,
    mesh: &'a ornis_render::Mesh,
    instance_count: u32,
}

impl<'a> Probe<'a> {
    /// Render one frame of `frame` into its own target and read it back.
    async fn render_and_read(
        &self,
        frame: &mut RenderFrame3D,
        target: &wgpu::TextureView,
        tex: &wgpu::Texture,
        label: &str,
    ) -> Vec<u8> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        frame.render(
            ornis_render::render_backend::RenderContext {
                device: self.device,
                queue: self.queue,
                encoder: &mut encoder,
                target,
            },
            self.renderer,
            self.mesh,
            self.instance_count,
        );
        self.queue.submit(std::iter::once(encoder.finish()));
        read_target(self.device, self.queue, tex, label).await.0
    }

    /// Multi-frame stability: pool reuse across frames must keep pixels fixed.
    async fn stability_frames(
        &self,
        graph3d: &mut RenderFrame3D,
        graph_view: &wgpu::TextureView,
        graph_tex: &wgpu::Texture,
        reference: &[u8],
    ) -> bool {
        const FRAMES: u32 = 16;
        let mut frames_identical = 0u32;
        for _ in 1..FRAMES {
            let pixels = self
                .render_and_read(graph3d, graph_view, graph_tex, "graph readback (frame)")
                .await;
            if pixels == reference {
                frames_identical += 1;
            }
        }
        frames_identical == FRAMES - 1
    }

    /// Standalone render of one technique; each gets its own graph, target
    /// and pool, so the slot/budget numbers are directly comparable.
    async fn render_technique(
        &self,
        format: wgpu::TextureFormat,
        technique: Technique,
        bloom: bool,
        label: &str,
    ) -> TechniqueStats {
        let (tex, view) = make_target(self.device, format, label);
        let mut frame = RenderFrame3D::new_with(format, (WIDTH, HEIGHT), technique, bloom);
        let pixels = self.render_and_read(&mut frame, &view, &tex, label).await;
        TechniqueStats {
            diff_vs_legacy: 0,
            slots: frame.pool_slots(),
            bytes: frame.texture_budget(),
            dump: frame.layout_dump(),
            pixels,
        }
    }

    /// Legacy (non-graph) render of the whole scene into `target`.
    fn render_legacy(&self, target: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        self.renderer.render_scene(
            self.device,
            self.queue,
            encoder,
            target,
            self.mesh,
            self.instance_count,
        );
    }

    /// Render one frame both ways (legacy + graph), submit together.
    fn render_both_ways(
        &self,
        graph3d: &mut RenderFrame3D,
        legacy_view: &wgpu::TextureView,
        graph_view: &wgpu::TextureView,
    ) -> wgpu::CommandEncoder {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("probe encoder"),
            });
        self.render_legacy(legacy_view, &mut encoder);
        graph3d.render(
            ornis_render::render_backend::RenderContext {
                device: self.device,
                queue: self.queue,
                encoder: &mut encoder,
                target: graph_view,
            },
            self.renderer,
            self.mesh,
            self.instance_count,
        );
        encoder
    }

    /// The three standalone technique variants (forward / deferred /
    /// forward+bloom); each gets its own graph, target and pool.
    async fn render_all_techniques(
        &self,
        format: wgpu::TextureFormat,
        legacy_pixels: &[u8],
    ) -> (TechniqueStats, TechniqueStats, TechniqueStats) {
        let mut forward = self
            .render_technique(format, Technique::Forward, false, "probe forward target")
            .await;
        let mut deferred = self
            .render_technique(format, Technique::Deferred, false, "probe deferred target")
            .await;
        // Forward + bloom: the bright-pass must read hdr_fwd (hdr is dead).
        let fwd_bloom = self
            .render_technique(
                format,
                Technique::Forward,
                true,
                "probe forward bloom target",
            )
            .await;

        // A technique switch must change the picture (no silent no-op).
        forward.diff_vs_legacy = diff_count(legacy_pixels, &forward.pixels);
        deferred.diff_vs_legacy = diff_count(legacy_pixels, &deferred.pixels);
        (forward, deferred, fwd_bloom)
    }
}

#[derive(Default)]
struct TechniqueStats {
    pixels: Vec<u8>,
    diff_vs_legacy: usize,
    slots: usize,
    bytes: u64,
    dump: String,
}

impl TechniqueStats {
    fn log(&self, name: &str) {
        println!("--- {name} technique ---");
        println!("{}", self.dump);
        println!(
            "{name} slots: {}, budget: {} bytes — differs from legacy: {} px",
            self.slots, self.bytes, self.diff_vs_legacy
        );
    }
}

async fn run(scene: &Scene) {
    // ── Headless device ───────────────────────────────────────────────
    let (device, queue) = create_headless_device("frame_plan_probe").await;

    // ── Two identical offscreen targets ───────────────────────────────
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (legacy_tex, legacy_view) = make_target(&device, format, "probe legacy target");
    let (graph_tex, graph_view) = make_target(&device, format, "probe graph target");

    let surface_config = surface_config_for(format);
    let renderer = Renderer3D::new(&device, &surface_config, 1);

    // ── Scene → GPU data ──────────────────────────────────────────────
    let (mesh, materials, instances) = build_scene_data(&device, scene);
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
    renderer.set_camera(&queue, &camera_view_proj(cam), cam.position);

    let probe = Probe {
        device: &device,
        queue: &queue,
        renderer: &renderer,
        mesh: &mesh,
        instance_count: instances.len() as u32,
    };

    // ── Render: legacy path, then graph path; read back both ──────────
    let mut graph3d = RenderFrame3D::new(format, (WIDTH, HEIGHT));
    let encoder = probe.render_both_ways(&mut graph3d, &legacy_view, &graph_view);
    queue.submit(std::iter::once(encoder.finish()));

    let unpadded = WIDTH * BYTES_PER_PIXEL;
    let (legacy_pixels, _) = read_target(&device, &queue, &legacy_tex, "legacy readback").await;
    let (graph_pixels, _) = read_target(&device, &queue, &graph_tex, "graph readback").await;

    // ── Multi-frame stability: pool reuse across frames ───────────────
    const FRAMES: u32 = 16;
    let slots_before = graph3d.pool_slots();
    let all_frames_stable = probe
        .stability_frames(&mut graph3d, &graph_view, &graph_tex, &graph_pixels)
        .await;
    let pool_stable = slots_before == graph3d.pool_slots() && graph3d.pool_slots() < 9;
    log_stability(graph3d.pool_slots(), all_frames_stable, pool_stable, FRAMES);

    // ── Bloom: the same graph plus the bloom node chain ───────────────
    let bloom = bloom_phase(&probe, format, &graph_pixels).await;

    // ── Technique switch: same graph, different node wires ────────────
    let (forward, deferred, fwd_bloom) = probe.render_all_techniques(format, &legacy_pixels).await;
    let fwd_lookup_active = diff_count(&forward.pixels, &fwd_bloom.pixels) > 0;
    forward.log("forward");
    deferred.log("deferred");
    println!(
        "forward bloom active: {fwd_lookup_active} ({} slots, {} bytes)",
        fwd_bloom.slots, fwd_bloom.bytes
    );

    // ── Compare legacy vs graph ───────────────────────────────────────
    let verdict = compare_legacy_graph(&legacy_pixels, &graph_pixels);
    save_png(
        "target/frame_plan_probe_legacy.png",
        &legacy_pixels,
        unpadded,
    );
    save_png("target/frame_plan_probe_graph.png", &graph_pixels, unpadded);
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
    memory_report(legacy_bytes, graph_bytes);

    report_pass_or_fail(
        &verdict,
        &bloom,
        all_frames_stable,
        pool_stable,
        &forward,
        &deferred,
        fwd_lookup_active,
        graph_bytes,
    );
}

/// Bloom sanity phase: render with the bloom node chain and diff it against
/// the no-bloom pixels. Bloom must change *something* on a lit scene.
async fn bloom_phase(
    probe: &Probe<'_>,
    format: wgpu::TextureFormat,
    graph_pixels: &[u8],
) -> PhaseBloom {
    let (bloom_tex, bloom_view) = make_target(probe.device, format, "probe bloom target");
    let mut graph3d_bloom = RenderFrame3D::new_with_bloom(format, (WIDTH, HEIGHT));
    let bloom_pixels = probe
        .render_and_read(
            &mut graph3d_bloom,
            &bloom_view,
            &bloom_tex,
            "probe bloom encoder",
        )
        .await;

    // Bloom must not change pixels without bright content... but it must
    // change *something*: the bright specular highlights should glow.
    let bloom_diff = diff_count(graph_pixels, &bloom_pixels);
    save_png(
        "target/frame_plan_probe_bloom.png",
        &bloom_pixels,
        WIDTH * BYTES_PER_PIXEL,
    );
    println!("--- bloom graph layout ---");
    println!("{}", graph3d_bloom.layout_dump());
    println!(
        "bloom pool slots: {}, budget: {} bytes",
        graph3d_bloom.pool_slots(),
        graph3d_bloom.texture_budget()
    );
    println!(
        "bloom vs no-bloom: {bloom_diff} differing pixels (of {}), active: {}",
        graph_pixels.len(),
        bloom_diff > 0
    );
    PhaseBloom {
        diff: bloom_diff,
        slots: graph3d_bloom.pool_slots(),
    }
}

struct PhaseBloom {
    diff: usize,
    slots: usize,
}

fn log_stability(slots: usize, all_frames_stable: bool, pool_stable: bool, frames: u32) {
    println!("{frames}-frame pool stable: {pool_stable}, frames identical: {all_frames_stable}");
    let _ = slots;
}

struct Verdict {
    same: bool,
    legacy_diff: usize,
    max_diff: u8,
    pixel_total: usize,
}

fn compare_legacy_graph(legacy: &[u8], graph: &[u8]) -> Verdict {
    let max_diff = legacy
        .iter()
        .zip(graph)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    Verdict {
        same: legacy == graph,
        legacy_diff: diff_count(legacy, graph),
        max_diff,
        pixel_total: legacy.len(),
    }
}

/// Final PASS/FAIL matrix for the probe (exits the process on failure).
#[allow(clippy::too_many_arguments)]
fn report_pass_or_fail(
    v: &Verdict,
    bloom: &PhaseBloom,
    all_frames_stable: bool,
    pool_stable: bool,
    forward: &TechniqueStats,
    deferred: &TechniqueStats,
    fwd_lookup_active: bool,
    graph_bytes: u64,
) {
    if !(v.same && all_frames_stable && pool_stable && bloom.diff > 0) {
        println!(
            "FAIL: {} mismatching pixels, max byte diff {} (of {})",
            v.legacy_diff, v.max_diff, v.pixel_total
        );
        println!(
            "bloom active: false ({} differing pixels vs no-bloom)",
            bloom.diff
        );
        println!("frames identical: {all_frames_stable}, pool stable: {pool_stable}");
        println!("PNGs: target/frame_plan_probe_{{legacy,graph,bloom}}.png");
        std::process::exit(1);
    }

    println!("PASS: legacy and render-graph paths are pixel-identical");
    println!(
        "PASS: bloom chain is active (downsample/upsample/composite) with {} pool slots",
        bloom.slots
    );
    println!(
        "PASS: technique switch — forward-only: {} slots/{} B ({} px differ from legacy; bloom active: {fwd_lookup_active})",
        forward.slots, forward.bytes, forward.diff_vs_legacy
    );
    println!(
        "PASS: technique switch — deferred-only matches legacy pixel-for-pixel ({} slots, {} B)",
        deferred.slots, deferred.bytes
    );
    check_technique_budgets(forward, deferred, fwd_lookup_active, graph_bytes);
    println!(
        "PNGs: target/frame_plan_probe_{{legacy,graph,bloom,forward,deferred}}.png ({WIDTH}x{HEIGHT})"
    );
}

/// Budget checks for the technique matrix; exits on an inactive or
/// budget-worse path.
fn check_technique_budgets(
    forward: &TechniqueStats,
    deferred: &TechniqueStats,
    fwd_lookup_active: bool,
    graph_bytes: u64,
) {
    let fwd_active = forward.slots <= 4;
    let fwd_vs_legacy = forward.diff_vs_legacy;
    let def_vs_legacy = deferred.diff_vs_legacy;
    let fwd_bytes = forward.bytes;
    let def_bytes = deferred.bytes;
    // Deferred keeps the gbuffer textures, so its budget equals hybrid
    // (the forward layer aliases material_params there); it must not
    // exceed it, though.
    let fwd_budget_ok = fwd_bytes < graph_bytes;
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
}
fn memory_report(legacy_bytes: u64, graph_bytes: u64) {
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
    println!("saved: {saved} bytes ({pct:.1}%)");
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
