#![allow(deprecated)]
//! Offscreen probe: renders assets/scene.ron through Renderer3D (RenderBackend
//! trait) into a headless wgpu texture and saves the frame as PNG.
//!
//! Run from the workspace root:
//!   cargo run -p ornis-render --example render_probe -- [scene.ron] [out.png]
//!
//! Prints the final view/proj matrices, the first two instance transforms and
//! buffer expectations so the browser (WASM) path can be compared against it.

use glam::{Mat4, Quat, Vec3};
use ornis_core::OpenPBRMaterial;
use ornis_render::scene::{CameraDesc, LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_render::{
    InstanceData, RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

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
    let out_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "target/render_probe.png".to_string());

    let ron_text = std::fs::read_to_string(&scene_path).expect("read scene.ron");
    let scene = Scene::from_ron(&ron_text).expect("parse scene.ron");
    println!(
        "scene '{}': {} entities, {} lights, ambient {:?}",
        scene.name,
        scene.entities.len(),
        scene.lights.len(),
        scene.ambient
    );

    pollster::block_on(run(&scene, &out_path));
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

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("device")
}

fn make_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe target"),
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
    println!(
        "mesh: {} vertices, {} indices",
        mesh.vertex_count, mesh.num_indices
    );

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

fn lights_of(scene: &Scene) -> Vec<([f32; 3], f32, [f32; 3])> {
    scene
        .lights
        .iter()
        .map(|l| match l {
            LightDesc::Directional {
                direction,
                intensity,
                color,
            } => (*direction, *intensity, *color),
        })
        .collect()
}

fn camera_view_proj(cam: &CameraDesc) -> (Mat4, Mat4, [[f32; 4]; 4]) {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view = Mat4::look_at_rh(
        Vec3::from(cam.position),
        Vec3::from(cam.target),
        Vec3::from(cam.up),
    );
    let proj = Mat4::perspective_rh(cam.fov.to_radians(), aspect, cam.near, cam.far);
    (view, proj, (proj * view).to_cols_array_2d())
}

/// Copy the rendered texture into a tightly-packed RGBA byte buffer.
fn read_back_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    bytes_per_pixel: u32,
) -> Vec<u8> {
    let unpadded_bytes_per_row = WIDTH * bytes_per_pixel;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe readback"),
        size: (padded_bytes_per_row * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
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
                bytes_per_row: Some(padded_bytes_per_row),
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

    let mut pixels = vec![0u8; (unpadded_bytes_per_row * HEIGHT) as usize];
    for y in 0..HEIGHT as usize {
        let src = &data[y * padded_bytes_per_row as usize..][..unpadded_bytes_per_row as usize];
        pixels[y * unpadded_bytes_per_row as usize..][..unpadded_bytes_per_row as usize]
            .copy_from_slice(src);
    }
    drop(data);
    readback.unmap();
    pixels
}

fn save_png(path: &str, pixels: &[u8]) {
    let file = std::fs::File::create(path).expect("create png");
    let mut encoder_png = png::Encoder::new(std::io::BufWriter::new(file), WIDTH, HEIGHT);
    encoder_png.set_color(png::ColorType::Rgba);
    encoder_png.set_depth(png::BitDepth::Eight);
    let mut writer = encoder_png.write_header().expect("png header");
    writer.write_image_data(pixels).expect("png data");
    println!("saved {path} ({WIDTH}x{HEIGHT})");
}

/// Quick pixel sanity: sample the center and the horizontal strip where the
/// 5 spheres should be (y ≈ 55% of height).
fn log_pixel_samples(pixels: &[u8], bytes_per_pixel: u32) {
    let sample = |x: u32, y: u32| {
        let off = ((y * WIDTH + x) * bytes_per_pixel) as usize;
        [
            pixels[off],
            pixels[off + 1],
            pixels[off + 2],
            pixels[off + 3],
        ]
    };
    let mid_y = (HEIGHT as f32 * 0.55) as u32;
    for frac in [0.1f32, 0.3, 0.5, 0.7, 0.9] {
        let x = (WIDTH as f32 * frac) as u32;
        println!("pixel({x},{mid_y}) = {:?}", sample(x, mid_y));
    }
    println!("pixel(center) = {:?}", sample(WIDTH / 2, HEIGHT / 2));
}

fn print_instance_dump(instances: &[InstanceData]) {
    for (i, inst) in instances.iter().take(2).enumerate() {
        let m = inst.model_matrix.to_cols_array_2d();
        println!(
            "instance[{i}] translation = {:.3?}, scale_col0_len = {:.3}",
            [m[3][0], m[3][1], m[3][2]],
            (m[0][0] * m[0][0] + m[0][1] * m[0][1] + m[0][2] * m[0][2]).sqrt()
        );
    }
}

async fn run(scene: &Scene, out_path: &str) {
    // ── Headless device ───────────────────────────────────────────────
    let (device, queue) = create_headless_device("render_probe").await;

    // ── Offscreen target (same format the browser surface uses) ───────
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let (target_texture, target_view) = make_target(&device, format);
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        color_space: wgpu::SurfaceColorSpace::Auto,
        desired_maximum_frame_latency: 2,
    };

    let backend_config = RenderBackendConfig {
        surface_config: surface_config.clone(),
        sample_count: 1,
        max_objects: 256,
        max_materials: 64,
    };
    let mut renderer: Box<dyn RenderBackend> = create_render_backend(&device, &backend_config);

    // ── Scene → GPU data ──────────────────────────────────────────────
    let (mesh, materials, instances) = build_scene_data(&device, scene);
    renderer.upload_materials(&queue, &materials);
    renderer.upload_instances(&queue, &instances);
    renderer.set_lights(&queue, scene.ambient, &lights_of(scene));

    // ── Camera ────────────────────────────────────────────────────────
    let cam = &scene.camera;
    let (view, proj, view_proj) = camera_view_proj(cam);
    renderer.set_camera(&queue, &view_proj, cam.position);

    // ── Validation dump ───────────────────────────────────────────────
    println!("fov_deg={} fov_rad={}", cam.fov, cam.fov.to_radians());
    println!("view = {:.3?}", view.to_cols_array_2d());
    println!("proj = {:.3?}", proj.to_cols_array_2d());
    print_instance_dump(&instances);

    // ── Render ────────────────────────────────────────────────────────
    let bytes_per_pixel = 4u32;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe encoder"),
    });
    renderer.render_scene(
        RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &target_view,
        },
        &mesh,
        instances.len() as u32,
    );
    let pixels = read_back_pixels(&device, &queue, encoder, &target_texture, bytes_per_pixel);

    // ── Save PNG ──────────────────────────────────────────────────────
    save_png(out_path, &pixels);

    // Quick pixel sanity samples.
    log_pixel_samples(&pixels, bytes_per_pixel);
}
