//! Backend-neutral rendering interface.
//!
//! [`RenderBackend`] abstracts the deferred renderer behind a small trait so
//! callers (and tests) can drive a full frame — camera, lights, uploads, one
//! draw — without touching [`crate::renderer::Renderer3D`] directly. The
//! factory [`create_render_backend`] returns the production implementation.
use crate::mesh::Mesh;
use crate::renderer::InstanceData;
use ornis_core::material::OpenPBRMaterial;

use wgpu;

/// Sizing and capacity knobs for backend construction.
#[derive(Debug, Clone)]
pub struct RenderBackendConfig {
    /// Surface format/size/present parameters; must be compatible with the
    /// target surface (or an offscreen texture in tests).
    pub surface_config: wgpu::SurfaceConfiguration,
    /// MSAA sample count for the gbuffer and lighting passes.
    pub sample_count: u32,
    /// Upper bound on instances per frame (sized into GPU buffers).
    pub max_objects: u32,
    /// Upper bound on materials per frame.
    pub max_materials: u32,
}

impl Default for RenderBackendConfig {
    fn default() -> Self {
        Self {
            surface_config: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                width: 800,
                height: 600,
                present_mode: wgpu::PresentMode::AutoNoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
            sample_count: 1,
            max_objects: 256,
            max_materials: 64,
        }
    }
}

/// Per-frame resources a [`RenderBackend::render_scene`] call needs.
#[derive(Debug)]
pub struct RenderContext<'a> {
    /// Logical device owning the pipeline/buffers.
    pub device: &'a wgpu::Device,
    /// Upload queue for uniform data.
    pub queue: &'a wgpu::Queue,
    /// Encoder the render pass is recorded onto.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// View of the frame's final output target.
    pub target: &'a wgpu::TextureView,
}

/// Backend-neutral interface over one deferred renderer instance.
pub trait RenderBackend {
    /// Reallocate size-dependent targets after the output extent changed.
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);

    /// Upload view-projection matrix (column-major `[[f32;4];4]`) and world-space
    /// eye position used by lighting.
    fn set_camera(&mut self, queue: &wgpu::Queue, view_proj: &[[f32; 4]; 4], camera_pos: [f32; 3]);

    /// Upload ambient RGB and directional lights as
    /// `(direction, intensity, color)` triples.
    fn set_lights(
        &mut self,
        queue: &wgpu::Queue,
        ambient: [f32; 3],
        lights: &[([f32; 3], f32, [f32; 3])],
    );

    /// Replace the material table; instance data references entries by index.
    fn upload_materials(&mut self, queue: &wgpu::Queue, materials: &[OpenPBRMaterial]);

    /// Replace per-object instance transforms + material indices for the next draw.
    fn upload_instances(&mut self, queue: &wgpu::Queue, instances: &[InstanceData]);

    /// Record the full deferred frame (gbuffer -> lighting -> composite) into
    /// `context`, drawing the first `instance_count` uploaded instances with `mesh`.
    fn render_scene(&self, context: RenderContext<'_>, mesh: &Mesh, instance_count: u32);
}

/// Build the production [`RenderBackend`] (the deferred [`crate::renderer::Renderer3D`])
/// from `config`.
pub fn create_render_backend(
    device: &wgpu::Device,
    config: &RenderBackendConfig,
) -> Box<dyn RenderBackend> {
    Box::new(crate::renderer::Renderer3D::new(
        device,
        &config.surface_config,
        config.sample_count,
    ))
}

/// Adapter implementing [`RenderBackend`] by delegating to the concrete
/// [`crate::renderer::Renderer3D`]; this is what [`create_render_backend`] hands out.
pub mod renderer3d_backend {
    use super::*;
    use crate::renderer::Renderer3D;

    impl RenderBackend for Renderer3D {
        fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
            Renderer3D::resize(self, device, width, height);
        }

        fn set_camera(
            &mut self,
            queue: &wgpu::Queue,
            view_proj: &[[f32; 4]; 4],
            camera_pos: [f32; 3],
        ) {
            Renderer3D::set_camera(self, queue, view_proj, camera_pos);
        }

        fn set_lights(
            &mut self,
            queue: &wgpu::Queue,
            ambient: [f32; 3],
            lights: &[([f32; 3], f32, [f32; 3])],
        ) {
            Renderer3D::set_lights(self, queue, ambient, lights);
        }

        fn upload_materials(&mut self, queue: &wgpu::Queue, materials: &[OpenPBRMaterial]) {
            Renderer3D::upload_materials(self, queue, materials);
        }

        fn upload_instances(&mut self, queue: &wgpu::Queue, instances: &[InstanceData]) {
            Renderer3D::upload_instances(self, queue, instances);
        }

        fn render_scene(&self, context: RenderContext<'_>, mesh: &Mesh, instance_count: u32) {
            Renderer3D::render_scene(
                self,
                context.device,
                context.queue,
                context.encoder,
                context.target,
                mesh,
                instance_count,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_documented_defaults() {
        let config = RenderBackendConfig::default();
        assert_eq!(config.surface_config.width, 800);
        assert_eq!(config.surface_config.height, 600);
        assert_eq!(
            config.surface_config.format,
            wgpu::TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            config.surface_config.usage,
            wgpu::TextureUsages::RENDER_ATTACHMENT
        );
        assert_eq!(
            config.surface_config.present_mode,
            wgpu::PresentMode::AutoNoVsync
        );
        assert_eq!(config.surface_config.desired_maximum_frame_latency, 2);
        assert!(config.surface_config.view_formats.is_empty());
        assert_eq!(config.sample_count, 1);
        assert_eq!(config.max_objects, 256);
        assert_eq!(config.max_materials, 64);
    }

    /// None when no adapter is available (CI without GPU and without
    /// lavapipe); the tests below skip in that case.
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::empty(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .ok()
        })
    }

    #[test]
    fn factory_builds_backend_and_resize_reallocates() {
        let Some((device, _queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let config = RenderBackendConfig::default();
        let mut backend = create_render_backend(&device, &config);
        // Resizing to a different extent must not panic and keeps the
        // backend usable (exercises the RenderBackend trait impl path).
        backend.resize(&device, 320, 240);
        backend.resize(&device, 800, 600);
    }

    /// Drives every `RenderBackend` trait method end-to-end through the
    /// factory handle: upload one instance + material, draw one frame into
    /// an offscreen target, submit.
    #[test]
    fn trait_object_renders_one_instance() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let config = RenderBackendConfig::default();
        let mut backend = create_render_backend(&device, &config);

        backend.set_camera(
            &queue,
            &glam::Mat4::IDENTITY.to_cols_array_2d(),
            [0.0, 0.0, 3.0],
        );
        backend.set_lights(
            &queue,
            [0.1, 0.1, 0.1],
            &[([0.0, 1.0, 1.0], 1.0, [1.0, 1.0, 1.0])],
        );
        backend.upload_materials(&queue, &[OpenPBRMaterial::default()]);
        backend.upload_instances(
            &queue,
            &[InstanceData {
                model_matrix: glam::Mat4::IDENTITY,
                normal_matrix: glam::Mat4::IDENTITY,
                material_index: 0,
            }],
        );

        let mesh = crate::mesh::create_sphere(&device, 1.0, 8, 4);
        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("backend test target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("backend test encoder"),
        });
        backend.render_scene(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &target,
            },
            &mesh,
            1,
        );
        queue.submit([encoder.finish()]);
    }

    /// Golden-frame: offscreen 1280×720 render of `assets/scene.ron` (5 entities)
    /// через `RenderBackend` — пининг регрессов реального GPU пайплайна.
    ///
    /// Сравнивает текущий кадр попиксельно с чекином
    /// `crates/render/tests/data/golden_probe_1280x720.png` (снятым на Apple M1
    /// via `cargo run -p ornis-render --example render_probe`), допускает
    /// канальный дрейф ≤2 (округление sRGB/тоновая компрессия между драйверами).
    /// Если адаптера нет (CI без GPU) — пропускается.
    #[test]
    fn golden_full_scene_probe_matches_snapshot() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping golden probe");
            return;
        };

        // ── Scene RON ───────────────────────────────────────────────
        let ron_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/scene.ron");
        let ron = std::fs::read_to_string(&ron_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", ron_path.display()));
        let scene = crate::scene::Scene::from_ron(&ron).expect("parse assets/scene.ron");

        // ── Gold PNG → bytes ───────────────────────────────────────
        let gold_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/golden_probe_1280x720.png");
        let gold_bytes = std::fs::read(&gold_path)
            .unwrap_or_else(|e| panic!("read {}: {e}", gold_path.display()));
        let mut decoder = png::Decoder::new(std::io::Cursor::new(&gold_bytes));
        decoder.set_transformations(png::Transformations::normalize_to_color8());
        let mut reader = decoder.read_info().expect("png header");
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).expect("png frame");
        assert_eq!(info.width, 1280);
        assert_eq!(info.height, 720);
        let gold_pixels = buf[..info.buffer_size()].to_vec();

        // ── Render current frame 1280×720 offscreen ─────────────────
        const W: u32 = 1280;
        const H: u32 = 720;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("golden probe target"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: W,
            height: H,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let backend_config = RenderBackendConfig {
            surface_config: surface_config.clone(),
            sample_count: 1,
            max_objects: 256,
            max_materials: 64,
        };
        let mut backend = create_render_backend(&device, &backend_config);

        // Build mesh/materials/instances как в render_probe::build_scene_data.
        let first = scene.entities.first().expect("scene has entities");
        let mesh = match &first.mesh {
            crate::scene::MeshDesc::Sphere {
                radius,
                segments,
                rings,
            } => crate::mesh::create_sphere(&device, *radius, *segments, *rings),
        };
        let mut materials = Vec::new();
        let mut instances = Vec::new();
        for (i, ent) in scene.entities.iter().enumerate() {
            let mat = match &ent.material {
                crate::scene::MaterialDesc::Dielectric {
                    base_color,
                    roughness,
                } => {
                    let mut m = ornis_core::OpenPBRMaterial::dielectric();
                    m.base.color_rgb(*base_color);
                    m.specular.roughness(*roughness);
                    m
                }
                crate::scene::MaterialDesc::Metal {
                    base_color,
                    roughness,
                } => {
                    let mut m = ornis_core::OpenPBRMaterial::metal();
                    m.base.color_rgb(*base_color);
                    m.specular.roughness(*roughness);
                    m
                }
                crate::scene::MaterialDesc::Coat {
                    base_color,
                    coat_weight,
                    coat_roughness,
                } => {
                    let mut m = ornis_core::OpenPBRMaterial::coat();
                    m.base.color_rgb(*base_color);
                    m.coat.weight(*coat_weight);
                    m.coat.roughness(*coat_roughness);
                    m
                }
            };
            materials.push(mat);
            let t = &ent.transform;
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from(t.scale),
                glam::Quat::from_xyzw(t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3])
                    .normalize(),
                glam::Vec3::from(t.translation),
            );
            instances.push(crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: i as u32,
            });
        }
        backend.upload_materials(&queue, &materials);
        backend.upload_instances(&queue, &instances);
        let lights: Vec<([f32; 3], f32, [f32; 3])> = scene
            .lights
            .iter()
            .map(|l| match l {
                crate::scene::LightDesc::Directional {
                    direction,
                    intensity,
                    color,
                } => (*direction, *intensity, *color),
            })
            .collect();
        backend.set_lights(&queue, scene.ambient, &lights);
        let (view, proj) = {
            let cam = &scene.camera;
            let aspect = W as f32 / H as f32;
            let view = glam::camera::rh::view::look_at_mat4(
                glam::Vec3::from(cam.position),
                glam::Vec3::from(cam.target),
                glam::Vec3::from(cam.up),
            );
            let proj = glam::camera::rh::proj::directx::perspective(
                cam.fov.to_radians(),
                aspect,
                cam.near,
                cam.far,
            );
            (view, proj)
        };
        let view_proj = (proj * view).to_cols_array_2d();
        backend.set_camera(&queue, &view_proj, scene.camera.position);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("golden probe encoder"),
        });
        backend.render_scene(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &target_view,
            },
            &mesh,
            instances.len() as u32,
        );

        // Read-back (та же логика что в render_probe::read_back_pixels).
        let bpp = 4u32;
        let unpadded = W * bpp;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("golden readback"),
            size: (padded * H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().unwrap();
        let mut pixels = vec![0u8; (unpadded * H) as usize];
        for y in 0..H as usize {
            pixels[y * unpadded as usize..][..unpadded as usize]
                .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
        }
        drop(data);
        readback.unmap();

        // ── Compare ────────────────────────────────────────────────
        assert_eq!(pixels.len(), gold_pixels.len());
        let mut max_diff: u8 = 0;
        let mut bad: usize = 0;
        for (a, b) in pixels.iter().zip(gold_pixels.iter()) {
            let d = a.abs_diff(*b);
            max_diff = max_diff.max(d);
            if d > 2 {
                bad += 1;
            }
        }
        let bad_pct = bad as f64 / pixels.len() as f64 * 100.0;
        eprintln!("golden probe: max_diff={max_diff} bad>2={bad} ({bad_pct:.4}%)");
        assert!(
            bad_pct < 0.01,
            "golden frame drifted: {bad} bytes diff >2 ({bad_pct:.4}%); max_diff={max_diff} — update tests/data/golden_probe_1280x720.png via render_probe if change is intentional"
        );
        // Санитарка: центр не чёрный, как в probe логе [52,52,186].
        let center_off = ((H / 2 * W + W / 2) * bpp) as usize;
        let center = &pixels[center_off..center_off + 4];
        assert!(center[2] > 80, "center blue must dominate: {center:?}");
    }
}
