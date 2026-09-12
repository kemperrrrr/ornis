//! Single-technique probe: renders a scene RON through one `RenderFrame3D`
//! technique (forward/deferred/hybrid) headless and saves the frame as PNG.
//!
//! Run from the workspace root:
//!   cargo run -p ornis-render --example technique_probe -- [forward|deferred|hybrid] [scene.ron] [out.png]
//!
//! Additive diagnostic used to bisect deferred-only vs shared shading
//! artifacts; touches no library code.

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
    let technique = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "forward".to_string());
    let scene_path = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "assets/scene.ron".to_string());
    let out_path = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "target/technique_probe.png".to_string());
    let technique = match technique.as_str() {
        "forward" => Technique::Forward,
        "deferred" => Technique::Deferred,
        "hybrid" => Technique::Hybrid,
        other => panic!("unknown technique `{other}` (forward|deferred|hybrid)"),
    };

    let ron_text = std::fs::read_to_string(&scene_path).expect("read scene.ron");
    let scene = Scene::from_ron(&ron_text).expect("parse scene.ron");
    pollster::block_on(run(&scene, technique, &out_path));
}

async fn run(scene: &Scene, technique: Technique, out_path: &str) {
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
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("technique_probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .expect("device");

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: WIDTH,
        height: HEIGHT,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let renderer = Renderer3D::new(&device, &surface_config, 1);

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
        instances.push(InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: i as u32,
        });
    }
    renderer.upload_materials(&queue, &materials);
    renderer.upload_instances(&queue, &instances);
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
    renderer.set_lights(&queue, scene.ambient, &lights);
    let cam = &scene.camera;
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view = glam::camera::rh::view::look_at_mat4(
        Vec3::from(cam.position),
        Vec3::from(cam.target),
        Vec3::from(cam.up),
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        cam.fov.to_radians(),
        aspect,
        cam.near,
        cam.far,
    );
    renderer.set_camera(&queue, &(proj * view).to_cols_array_2d(), cam.position);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("technique target"),
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
    let view_tex = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut frame = RenderFrame3D::new_with(format, (WIDTH, HEIGHT), technique, false);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("technique probe"),
    });
    frame.render(
        ornis_render::render_backend::RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &view_tex,
        },
        &renderer,
        &mesh,
        instances.len() as u32,
    );
    queue.submit(std::iter::once(encoder.finish()));

    let unpadded = WIDTH * BYTES_PER_PIXEL;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("technique readback"),
        size: (padded * HEIGHT) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("technique copy"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
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
        pixels[y * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
    }
    drop(data);
    readback.unmap();

    let file = std::fs::File::create(out_path).expect("create png");
    let mut encoder_png = png::Encoder::new(std::io::BufWriter::new(file), WIDTH, HEIGHT);
    encoder_png.set_color(png::ColorType::Rgba);
    encoder_png.set_depth(png::BitDepth::Eight);
    encoder_png
        .write_header()
        .expect("png header")
        .write_image_data(&pixels)
        .expect("png data");
    println!("saved {out_path}");
}
