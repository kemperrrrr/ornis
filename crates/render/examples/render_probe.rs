//! Offscreen probe: renders assets/scene.ron through Renderer3D (RenderBackend
//! trait) into a headless wgpu texture and saves the frame as PNG.
//!
//! Run from the workspace root:
//!   cargo run -p ornis-render --example render_probe -- [scene.ron] [out.png]
//!
//! Prints the final view/proj matrices, the first two instance transforms and
//! buffer expectations so the browser (WASM) path can be compared against it.

use glam::Mat4;
use ornis_assets::scene::{CameraDesc, LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_core::OpenPBRMaterial;
use ornis_core::material::ShadingMode;
use ornis_core::units::PositiveF32;
use ornis_render::{
    InstanceData, MaterialIdx, RenderBackend, RenderBackendConfig, RenderContext,
    create_render_backend,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
/// wgpu copy buffer row alignment (bytes).
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
/// Sphere-strip sample height as a fraction of the frame.
const SAMPLE_STRIP_Y: f32 = 0.55;
/// Horizontal sample fractions across the sphere strip.
const SAMPLE_X_FRACS: [f32; 5] = [0.1, 0.3, 0.5, 0.7, 0.9];
/// Default GPU object capacity for the probe backend.
const MAX_OBJECTS: u32 = 256;
/// Default GPU material capacity for the probe backend.
const MAX_MATERIALS: u32 = 64;
/// RGBA8 bytes per pixel.
const BYTES_PER_PIXEL: u32 = 4;

/// Peak-luminance emission mapping, mirroring `extraction::apply_emission`.
fn apply_emission(mat: &mut OpenPBRMaterial, emission: [f32; 3]) {
    let peak = emission[0].max(emission[1]).max(emission[2]).max(0.0);
    if peak > 0.0 {
        mat.emission.luminance(peak);
        mat.emission
            .color_rgb([emission[0] / peak, emission[1] / peak, emission[2] / peak]);
    }
}

fn build_material(entity_material: &MaterialDesc) -> OpenPBRMaterial {
    match entity_material {
        MaterialDesc::Dielectric {
            base_color,
            roughness,
            emission,
            ..
        } => {
            let mut mat = OpenPBRMaterial::dielectric();
            mat.base.color_rgb(*base_color);
            mat.specular.roughness(roughness.get());
            apply_emission(&mut mat, *emission);
            mat
        }
        MaterialDesc::Metal {
            base_color,
            roughness,
            emission,
            ..
        } => {
            let mut mat = OpenPBRMaterial::metal();
            mat.base.color_rgb(*base_color);
            mat.specular.roughness(roughness.get());
            apply_emission(&mut mat, *emission);
            mat
        }
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
            emission,
            ..
        } => {
            let mut mat = OpenPBRMaterial::coat();
            mat.base.color_rgb(*base_color);
            mat.coat.weight(coat_weight.get());
            mat.coat.roughness(coat_roughness.get());
            apply_emission(&mut mat, *emission);
            mat
        }
        MaterialDesc::Matte {
            base_color,
            roughness,
            ..
        } => {
            let mut mat = OpenPBRMaterial::dielectric();
            mat.base.color_rgb(*base_color);
            mat.base.diffuse_roughness(roughness.get());
            // Matte is diffuse-only: no specular lobe.
            mat.specular.weight(0.0);
            mat
        }
        MaterialDesc::Glass {
            base_color,
            roughness,
            ior,
            ..
        } => {
            let mut mat = OpenPBRMaterial::glass();
            mat.transmission.color_rgb(*base_color);
            mat.specular.roughness(roughness.get());
            mat.specular.ior(ior.get());
            mat
        }
        MaterialDesc::Unlit { color } => {
            // Mirrors the extraction mapping: no BRDF lobe, sprite color
            // as emission, unlit shading-mode flag.
            let mut mat = OpenPBRMaterial::dielectric();
            mat.base.color_rgb([0.0, 0.0, 0.0]);
            mat.base.weight(0.0);
            mat.specular.weight(0.0);
            mat.base.metalness(0.0);
            apply_emission(&mut mat, *color);
            mat.geometry.set_shading(ShadingMode::Unlit);
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

    let Ok(ron_text) = std::fs::read_to_string(&scene_path) else {
        eprintln!("failed to read {scene_path}");
        return;
    };
    let Ok(scene) = Scene::from_ron(&ron_text) else {
        eprintln!("failed to parse {scene_path}");
        return;
    };
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
async fn create_headless_device(label: &str) -> Option<(wgpu::Device, wgpu::Queue)> {
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
        .ok()?;
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
        .ok()
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
) -> Option<(ornis_render::Mesh, Vec<OpenPBRMaterial>, Vec<InstanceData>)> {
    let first = scene.entities.first()?;
    let mesh = match &first.mesh {
        MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => ornis_render::create_sphere(device, radius.get(), *segments, *rings),
        MeshDesc::Box { size } => {
            ornis_render::mesh::create_box(device, size.map(PositiveF32::get))
        }
        MeshDesc::Plane { size } => {
            ornis_render::mesh::create_plane(device, size.map(PositiveF32::get))
        }
        MeshDesc::Quad { size } => {
            ornis_render::mesh::create_quad(device, size.map(PositiveF32::get))
        }
        MeshDesc::Cylinder {
            radius,
            height,
            radial_segments,
        } => ornis_render::mesh::create_cylinder(
            device,
            radius.get(),
            height.get(),
            *radial_segments,
        ),
        // This probe renders procedural scenes; Custom soups have no
        // upload path here yet.
        MeshDesc::Custom { .. } => return None,
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
        let model = Mat4::from_scale_rotation_translation(t.scale, t.rotation.get(), t.translation);
        let normal_matrix = model.inverse().transpose();
        instances.push(InstanceData {
            model_matrix: model,
            normal_matrix,
            material_index: MaterialIdx::from(i as u32),
        });
    }
    Some((mesh, materials, instances))
}

fn lights_of(scene: &Scene) -> Vec<LightDesc> {
    scene.lights.clone()
}

fn camera_view_proj(cam: &CameraDesc) -> (Mat4, Mat4, [[f32; 4]; 4]) {
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view = glam::camera::rh::view::look_at_mat4(cam.position, cam.target, cam.up.get());
    let proj = glam::camera::rh::proj::directx::perspective(
        cam.fov.to_radians().get(),
        aspect,
        cam.near.get(),
        cam.far.get(),
    );
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
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT)
        * COPY_BYTES_PER_ROW_ALIGNMENT;
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
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        eprintln!("GPU poll failed during readback");
        return Vec::new();
    }
    let Ok(Ok(())) = rx.recv() else {
        eprintln!("map readback failed");
        return Vec::new();
    };
    let Ok(data) = slice.get_mapped_range() else {
        eprintln!("get_mapped_range failed");
        return Vec::new();
    };

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
    let Ok(file) = std::fs::File::create(path) else {
        eprintln!("failed to create {path}");
        return;
    };
    let mut encoder_png = png::Encoder::new(std::io::BufWriter::new(file), WIDTH, HEIGHT);
    encoder_png.set_color(png::ColorType::Rgba);
    encoder_png.set_depth(png::BitDepth::Eight);
    let Ok(mut writer) = encoder_png.write_header() else {
        eprintln!("failed to write png header for {path}");
        return;
    };
    if writer.write_image_data(pixels).is_err() {
        eprintln!("failed to write png data for {path}");
        return;
    }
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
    let mid_y = (HEIGHT as f32 * SAMPLE_STRIP_Y) as u32;
    for frac in SAMPLE_X_FRACS {
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
    let Some((device, queue)) = create_headless_device("render_probe").await else {
        eprintln!("no GPU adapter/device for render_probe");
        return;
    };

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
        exposure: 1.0,
        max_objects: MAX_OBJECTS,
        max_materials: MAX_MATERIALS,
    };
    let mut renderer: Box<dyn RenderBackend> = create_render_backend(&device, &backend_config);

    // ── Scene → GPU data ──────────────────────────────────────────────
    let Some((mesh, materials, instances)) = build_scene_data(&device, scene) else {
        eprintln!("scene has no renderable entities");
        return;
    };
    renderer.upload_materials(&device, &queue, &materials);
    renderer.upload_instances(&device, &queue, &instances);
    renderer.set_lights(&queue, scene.ambient, &lights_of(scene));

    // ── Camera ────────────────────────────────────────────────────────
    let cam = &scene.camera;
    let (view, proj, view_proj) = camera_view_proj(cam);
    renderer.set_camera(&queue, &view_proj, cam.position.to_array());

    // ── Validation dump ───────────────────────────────────────────────
    println!(
        "fov_deg={} fov_rad={}",
        cam.fov.get(),
        cam.fov.to_radians().get()
    );
    println!("view = {:.3?}", view.to_cols_array_2d());
    println!("proj = {:.3?}", proj.to_cols_array_2d());
    print_instance_dump(&instances);

    // ── Render ────────────────────────────────────────────────────────
    let bytes_per_pixel = BYTES_PER_PIXEL;
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
