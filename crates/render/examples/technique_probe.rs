//! Single-technique probe: renders a scene RON through one `RenderFrame3D`
//! technique (forward/deferred/hybrid) headless and saves the frame as PNG.
//!
//! Run from the workspace root:
//!   cargo run -p ornis-render --example technique_probe -- [forward|deferred|hybrid] [scene.ron] [out.png]
//!
//! Additive diagnostic used to bisect deferred-only vs shared shading
//! artifacts; touches no library code.

use glam::Mat4;
use ornis_assets::scene::{LightDesc, MaterialDesc, MeshDesc, Scene};
use ornis_core::OpenPBRMaterial;
use ornis_core::units::PositiveF32;
use ornis_render::{InstanceData, MaterialIdx, RenderFrame3D, Renderer3D, Technique};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const BYTES_PER_PIXEL: u32 = 4;
/// wgpu copy buffer row alignment (bytes).
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;
/// CLI argv index for the optional output path.
const ARG_OUT_PATH: usize = 3;

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
        } => {
            let mut mat = OpenPBRMaterial::glass();
            mat.transmission.color_rgb(*base_color);
            mat.specular.roughness(roughness.get());
            mat.specular.ior(ior.get());
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
        .nth(ARG_OUT_PATH)
        .unwrap_or_else(|| "target/technique_probe.png".to_string());
    let technique = match technique.as_str() {
        "forward" => Technique::Forward,
        "deferred" => Technique::Deferred,
        "hybrid" => Technique::Hybrid,
        other => {
            eprintln!("unknown technique `{other}` (forward|deferred|hybrid)");
            return;
        }
    };

    let Ok(ron_text) = std::fs::read_to_string(&scene_path) else {
        eprintln!("failed to read {scene_path}");
        return;
    };
    let Ok(scene) = Scene::from_ron(&ron_text) else {
        eprintln!("failed to parse {scene_path}");
        return;
    };
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
    let Ok(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
    else {
        eprintln!("no suitable GPU adapter");
        return;
    };
    let Ok((device, queue)) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("technique_probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
    else {
        eprintln!("failed to create GPU device");
        return;
    };

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

    let Some(first) = scene.entities.first() else {
        eprintln!("scene has no entities");
        return;
    };
    let mesh = match &first.mesh {
        MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => ornis_render::create_sphere(&device, radius.get(), *segments, *rings),
        MeshDesc::Box { size } => {
            ornis_render::mesh::create_box(&device, size.map(PositiveF32::get))
        }
        MeshDesc::Plane { size } => {
            ornis_render::mesh::create_plane(&device, size.map(PositiveF32::get))
        }
        MeshDesc::Cylinder {
            radius,
            height,
            radial_segments,
        } => ornis_render::mesh::create_cylinder(
            &device,
            radius.get(),
            height.get(),
            *radial_segments,
        ),
        // This probe renders procedural scenes; Custom soups have no
        // upload path here yet.
        MeshDesc::Custom { .. } => {
            eprintln!("Custom mesh not supported by this probe");
            return;
        }
    };
    let mut materials = Vec::new();
    let mut instances = Vec::new();
    for (i, entity) in scene.entities.iter().enumerate() {
        materials.push(build_material(&entity.material));
        let t = &entity.transform;
        let model = Mat4::from_scale_rotation_translation(t.scale, t.rotation.get(), t.translation);
        instances.push(InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: MaterialIdx::from(i as u32),
        });
    }
    renderer.upload_materials(&device, &queue, &materials);
    renderer.upload_instances(&device, &queue, &instances);
    let lights: Vec<LightDesc> = scene.lights.clone();
    renderer.set_lights(&queue, scene.ambient, &lights);
    let cam = &scene.camera;
    let aspect = WIDTH as f32 / HEIGHT as f32;
    let view = glam::camera::rh::view::look_at_mat4(cam.position, cam.target, cam.up.get());
    let proj = glam::camera::rh::proj::directx::perspective(
        cam.fov.to_radians().get(),
        aspect,
        cam.near.get(),
        cam.far.get(),
    );
    renderer.set_camera(
        &queue,
        &(proj * view).to_cols_array_2d(),
        cam.position.to_array(),
    );

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

    let mut frame =
        RenderFrame3D::new_with(format, (WIDTH, HEIGHT), technique, ornis_render::Bloom::Off);
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
    let padded = unpadded.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
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
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    if device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
        eprintln!("GPU poll failed during readback");
        return;
    }
    let Ok(Ok(())) = rx.recv() else {
        eprintln!("map readback failed");
        return;
    };
    let Ok(data) = slice.get_mapped_range() else {
        eprintln!("get_mapped_range failed");
        return;
    };
    let mut pixels = vec![0u8; (unpadded * HEIGHT) as usize];
    for y in 0..HEIGHT as usize {
        pixels[y * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
    }
    drop(data);
    readback.unmap();

    let Ok(file) = std::fs::File::create(out_path) else {
        eprintln!("failed to create {out_path}");
        return;
    };
    let mut encoder_png = png::Encoder::new(std::io::BufWriter::new(file), WIDTH, HEIGHT);
    encoder_png.set_color(png::ColorType::Rgba);
    encoder_png.set_depth(png::BitDepth::Eight);
    let Ok(mut writer) = encoder_png.write_header() else {
        eprintln!("failed to write png header for {out_path}");
        return;
    };
    if writer.write_image_data(&pixels).is_err() {
        eprintln!("failed to write png data for {out_path}");
        return;
    }
    println!("saved {out_path}");
}
