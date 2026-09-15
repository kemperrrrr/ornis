//! Backend-neutral rendering interface.
//!
//! [`RenderBackend`] abstracts the deferred renderer behind a small trait so
//! callers (and tests) can drive a full frame — camera, lights, uploads, one
//! draw — without touching [`crate::renderer::Renderer3D`] directly. The
//! factory [`create_render_backend`] returns the production implementation.
use crate::mesh::Mesh;
use crate::renderer::InstanceData;
use crate::scene::LightDesc;
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

    /// Upload ambient RGB and scene lights ([`LightDesc`]); the renderer
    /// uploads the first four of any kind.
    fn set_lights(&mut self, queue: &wgpu::Queue, ambient: [f32; 3], lights: &[LightDesc]);

    /// Replace the material table; instance data references entries by index.
    /// `device` is needed because oversized frames regrow the buffer.
    fn upload_materials(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[OpenPBRMaterial],
    );

    /// Replace per-object instance transforms + material indices for the next draw.
    /// `device` is needed because oversized frames regrow the buffer.
    fn upload_instances(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceData],
    );

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

        fn set_lights(&mut self, queue: &wgpu::Queue, ambient: [f32; 3], lights: &[LightDesc]) {
            Renderer3D::set_lights(self, queue, ambient, lights);
        }

        fn upload_materials(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            materials: &[OpenPBRMaterial],
        ) {
            Renderer3D::upload_materials(self, device, queue, materials);
        }

        fn upload_instances(
            &mut self,
            device: &wgpu::Device,
            queue: &wgpu::Queue,
            instances: &[InstanceData],
        ) {
            Renderer3D::upload_instances(self, device, queue, instances);
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

    /// Uploads past the initial 256-instance / 64-material capacities grow
    /// the storage buffers (doubling) instead of truncating the frame:
    /// after uploading 300 instances + 70 materials the draw still sees
    /// entry 299 / material 69.
    #[test]
    fn oversized_frame_grows_buffers_without_truncation() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let config = RenderBackendConfig::default();
        let mut backend = create_render_backend(&device, &config);
        let materials: Vec<OpenPBRMaterial> = (0..70).map(|_| OpenPBRMaterial::default()).collect();
        backend.upload_materials(&device, &queue, &materials);
        let instances: Vec<InstanceData> = (0..300)
            .map(|i| InstanceData {
                model_matrix: glam::Mat4::from_translation(glam::Vec3::new(i as f32, 0.0, 0.0)),
                normal_matrix: glam::Mat4::IDENTITY,
                material_index: (i % 70) as u32,
            })
            .collect();
        backend.upload_instances(&device, &queue, &instances);

        // The 300th instance / 70th material survive the round trip:
        // draw them alone into a target and require non-background pixels.
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("growth probe target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let mesh = crate::mesh::create_sphere(&device, 1.0, 8, 4);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("growth probe"),
        });
        // Re-upload just the tail entries at index 0 to prove they landed.
        backend.upload_materials(&device, &queue, &materials[69..70]);
        backend.upload_instances(&device, &queue, &instances[299..300]);
        backend.render_scene(
            RenderContext {
                device: &device,
                queue: &queue,
                encoder: &mut encoder,
                target: &view,
            },
            &mesh,
            1,
        );
        queue.submit(std::iter::once(encoder.finish()));
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
            &[LightDesc::Directional {
                direction: [0.0, 1.0, 1.0],
                intensity: 1.0,
                color: [1.0, 1.0, 1.0],
                shadow: false,
            }],
        );
        backend.upload_materials(&device, &queue, &[OpenPBRMaterial::default()]);
        backend.upload_instances(
            &device,
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
    /// via `RenderBackend` — pinning regressions of the real GPU pipeline.
    ///
    /// Compares the current frame pixel-by-pixel against the checked-in
    /// `crates/render/tests/data/golden_probe_1280x720.png` (captured on Apple M1
    /// via `cargo run -p ornis-render --example render_probe`), allowing
    /// per-channel drift ≤2 (sRGB rounding / tone-compression across drivers).
    /// Skipped when no adapter is available (CI without GPU).
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
        let mut buf = vec![0u8; reader.output_buffer_size().expect("png output size")];
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

        // Build mesh/materials/instances as in render_probe::build_scene_data.
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
        backend.upload_materials(&device, &queue, &materials);
        backend.upload_instances(&device, &queue, &instances);
        backend.set_lights(&queue, scene.ambient, &scene.lights);
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

        // Read-back (same logic as in render_probe::read_back_pixels).
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
        // Sanity check: center is not black, as in probe log [52,52,186].
        let center_off = ((H / 2 * W + W / 2) * bpp) as usize;
        let center = &pixels[center_off..center_off + 4];
        assert!(center[2] > 80, "center blue must dominate: {center:?}");
    }

    /// Shadow regression: a directional light with `shadow: true` must
    /// darken the receiver behind the occluder versus `shadow: false`.
    /// Exercises the full chain (layer assignment → depth pre-pass →
    /// PCF compare). Skipped when no adapter is available.
    #[test]
    fn shadowed_directional_darkens_occluded_receiver() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping shadow probe");
            return;
        };
        const W: u32 = 320;
        const H: u32 = 180;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow probe target"),
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
        let mut backend = create_render_backend(
            &device,
            &RenderBackendConfig {
                surface_config,
                sample_count: 1,
                max_objects: 256,
                max_materials: 64,
            },
        );
        // Unit sphere; sizes come from instance scales (true geometry,
        // unlike the probe examples that reuse the first entity's mesh).
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        backend.upload_materials(&device, &queue, &[gray, gray]);
        let place = |translation: [f32; 3], scale: f32| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(scale),
                glam::Quat::IDENTITY,
                glam::Vec3::from(translation),
            );
            crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
            }
        };
        // Receiver r=2 at origin; occluder r=0.9 on the light axis
        // (to-light ≈ (0.43, 0.85, 0.30)), casting onto the receiver.
        backend.upload_instances(
            &device,
            &queue,
            &[place([0.0, 0.0, 0.0], 2.0), place([1.5, 2.9, 1.05], 0.9)],
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0.0, 3.0, 10.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            W as f32 / H as f32,
            0.1,
            100.0,
        );
        backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);

        let mut render = |shadow: bool| -> Vec<u8> {
            backend.set_lights(
                &queue,
                [0.05, 0.05, 0.08],
                &[crate::scene::LightDesc::Directional {
                    direction: [0.42, 0.84, 0.3],
                    intensity: 1.2,
                    color: [1.0, 1.0, 1.0],
                    shadow,
                }],
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("shadow probe encoder"),
            });
            backend.render_scene(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &target_view,
                },
                &mesh,
                2,
            );
            let bpp = 4u32;
            let unpadded = W * bpp;
            let padded = unpadded.div_ceil(256) * 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow probe readback"),
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
            pixels
        };

        let on = render(true);
        let off = render(false);
        assert_eq!(on.len(), off.len());
        let lum = |px: &[u8]| -> f64 {
            px.chunks_exact(4)
                .map(|c| (c[0] as f64 + c[1] as f64 + c[2] as f64) / 3.0)
                .sum::<f64>()
                / (px.len() / 4) as f64
        };
        let (mean_on, mean_off) = (lum(&on), lum(&off));
        let mut diff_px = 0usize;
        for (a, b) in on.chunks_exact(4).zip(off.chunks_exact(4)) {
            let d = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            if d > 12 {
                diff_px += 1;
            }
        }
        eprintln!("shadow probe: mean_on={mean_on:.2} mean_off={mean_off:.2} diff_px={diff_px}");
        // The r=0.9 occluder covers a visible patch of the receiver:
        // hundreds of pixels must change, and the frame must get darker.
        assert!(
            diff_px > 200,
            "shadow had no visible effect: {diff_px} pixels differ"
        );
        // Mean margin calibrated on the true (small) blob: ~340 px
        // swing against a 320×180 frame moves the mean by ≈ 0.9.
        assert!(
            mean_on + 0.5 < mean_off,
            "shadowed frame is not darker: on={mean_on:.2} off={mean_off:.2}"
        );
    }

    /// Acne guard: a lone receiver (no occluder) must render (nearly)
    /// identically with a shadowed directional light on versus off —
    /// depth bias plus the evaluator reference bias must not invent
    /// self-shadowing. Skipped when no adapter is available.
    #[test]
    fn shadowed_directional_without_occluder_has_no_acne() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping shadow probe");
            return;
        };
        const W: u32 = 320;
        const H: u32 = 180;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("acne probe target"),
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
        let mut backend = create_render_backend(
            &device,
            &RenderBackendConfig {
                surface_config,
                sample_count: 1,
                max_objects: 256,
                max_materials: 64,
            },
        );
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        backend.upload_materials(&device, &queue, &[gray]);
        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::IDENTITY,
            glam::Vec3::ZERO,
        );
        backend.upload_instances(
            &device,
            &queue,
            &[crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
            }],
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0.0, 3.0, 10.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            W as f32 / H as f32,
            0.1,
            100.0,
        );
        backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);
        let mut render = |shadow: bool| -> Vec<u8> {
            backend.set_lights(
                &queue,
                [0.05, 0.05, 0.08],
                &[crate::scene::LightDesc::Directional {
                    direction: [0.42, 0.84, 0.3],
                    intensity: 1.2,
                    color: [1.0, 1.0, 1.0],
                    shadow,
                }],
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("acne probe encoder"),
            });
            backend.render_scene(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &target_view,
                },
                &mesh,
                1,
            );
            let bpp = 4u32;
            let unpadded = W * bpp;
            let padded = unpadded.div_ceil(256) * 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("acne probe readback"),
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
            pixels
        };
        let on = render(true);
        let off = render(false);
        let mut diff_px = 0usize;
        for (a, b) in on.chunks_exact(4).zip(off.chunks_exact(4)) {
            let d = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            if d > 12 {
                diff_px += 1;
            }
        }
        eprintln!("acne probe: diff_px={diff_px}");
        assert!(
            diff_px < 100,
            "self-shadowing acne without occluder: {diff_px} pixels differ"
        );
    }

    /// Acne guard for the cube path: a lone receiver under a shadowed
    /// point light must render (nearly) identically with the shadow on
    /// versus off. The cube faces share the depth-bias pair and the
    /// reference bias with the 2D layers, so both need the pin.
    /// Skipped when no adapter is available.
    #[test]
    fn shadowed_point_without_occluder_has_no_acne() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping shadow probe");
            return;
        };
        const W: u32 = 320;
        const H: u32 = 180;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("point acne probe target"),
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
        let mut backend = create_render_backend(
            &device,
            &RenderBackendConfig {
                surface_config,
                sample_count: 1,
                max_objects: 256,
                max_materials: 64,
            },
        );
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        backend.upload_materials(&device, &queue, &[gray]);
        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::IDENTITY,
            glam::Vec3::ZERO,
        );
        backend.upload_instances(
            &device,
            &queue,
            &[crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
            }],
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0.0, 3.0, 10.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            W as f32 / H as f32,
            0.1,
            100.0,
        );
        backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);
        let mut render = |shadow: bool| -> Vec<u8> {
            backend.set_lights(
                &queue,
                [0.05, 0.05, 0.08],
                &[crate::scene::LightDesc::Point {
                    position: [5.0, 4.0, 6.0],
                    intensity: 200.0,
                    color: [1.0, 1.0, 1.0],
                    range: 30.0,
                    shadow,
                }],
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("point acne probe encoder"),
            });
            backend.render_scene(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &target_view,
                },
                &mesh,
                1,
            );
            let bpp = 4u32;
            let unpadded = W * bpp;
            let padded = unpadded.div_ceil(256) * 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point acne probe readback"),
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
            pixels
        };
        let on = render(true);
        let off = render(false);
        let mut diff_px = 0usize;
        for (a, b) in on.chunks_exact(4).zip(off.chunks_exact(4)) {
            let d = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            if d > 12 {
                diff_px += 1;
            }
        }
        eprintln!("point acne probe: diff_px={diff_px}");
        assert!(
            diff_px < 100,
            "cube self-shadowing acne without occluder: {diff_px} pixels differ"
        );
    }

    /// Spot-shadow position: with the light straight down and the
    /// occluder at +X, the shadow must fall on the +X (right) half —
    /// darkness-only tests are blind to a V-mirrored lookup, which
    /// would throw it onto the −X half instead. Skipped without GPU.
    #[test]
    fn spot_shadow_falls_on_occluder_side() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        const W: u32 = 320;
        const H: u32 = 180;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("spot side target"),
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
        let mut backend = create_render_backend(
            &device,
            &RenderBackendConfig {
                surface_config,
                sample_count: 1,
                max_objects: 256,
                max_materials: 64,
            },
        );
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        backend.upload_materials(&device, &queue, &[gray, gray]);
        let place = |translation: [f32; 3], scale: f32| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(scale),
                glam::Quat::IDENTITY,
                glam::Vec3::from(translation),
            );
            crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
            }
        };
        // Receiver r=2 at origin; occluder r=0.7 at +X under a
        // straight-down spot: the shadow blob lands on the +X limb
        // (centroid ≈ x178), clear of the mirrored −X candidate
        // (≈ x142).
        backend.upload_instances(
            &device,
            &queue,
            &[place([0.0, 0.0, 0.0], 2.0), place([0.8, 2.9, 0.0], 0.7)],
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0.0, 3.0, 10.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            W as f32 / H as f32,
            0.1,
            100.0,
        );
        backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);
        let mut render = |shadow: bool| -> Vec<u8> {
            backend.set_lights(
                &queue,
                [0.05, 0.05, 0.08],
                &[crate::scene::LightDesc::Spot {
                    position: [0.0, 8.0, 0.0],
                    direction: [0.0, -1.0, 0.0],
                    intensity: 2000.0,
                    color: [1.0, 1.0, 1.0],
                    range: 30.0,
                    inner_angle: 25.0,
                    outer_angle: 35.0,
                    shadow,
                }],
            );
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("spot side encoder"),
            });
            backend.render_scene(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &target_view,
                },
                &mesh,
                2,
            );
            let bpp = 4u32;
            let unpadded = W * bpp;
            let padded = unpadded.div_ceil(256) * 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("spot side readback"),
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
            pixels
        };
        let on = render(true);
        let off = render(false);
        // Changed pixels (shadowed in `on`, lit in `off`): their
        // centroid must sit on the +X half (true blob ≈ x178; the
        // mirrored lookup would center it at ≈ x142 instead).
        let mut sx = 0u64;
        let mut sn = 0u64;
        for (i, (a, b)) in on.chunks_exact(4).zip(off.chunks_exact(4)).enumerate() {
            let d = (a[0] as i16 - b[0] as i16).abs()
                + (a[1] as i16 - b[1] as i16).abs()
                + (a[2] as i16 - b[2] as i16).abs();
            if d > 12 {
                sx += (i % W as usize) as u64;
                sn += 1;
            }
        }
        assert!(
            sn > 100,
            "spot shadow had no visible effect: {sn} pixels differ"
        );
        let cx = sx as f64 / sn as f64;
        eprintln!("spot side: changed={sn} centroid_x={cx:.1}");
        assert!(
            cx > 168.0,
            "spot shadow on the wrong half (mirrored lookup?): centroid_x={cx:.1}"
        );
    }

    /// Spot-layer texel contract: the off-axis occluder (light-space
    /// NDC y ≈ +0.25) must rasterize at row (1−ndc)/2·H, the row the
    /// sampler's mirrored `shadow_uv` reads — not at (ndc+1)/2·H.
    /// Pins the raster half of the V convention directly on texels.
    #[test]
    fn spot_shadow_layer_unmirrored_texel() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: 320,
            height: 180,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let renderer = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        renderer.upload_materials(&device, &queue, &[gray, gray]);
        let place = |translation: [f32; 3], scale: f32| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(scale),
                glam::Quat::IDENTITY,
                glam::Vec3::from(translation),
            );
            crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
            }
        };
        renderer.upload_instances(
            &device,
            &queue,
            &[place([0.0, 0.0, 0.0], 2.0), place([0.8, 3.5, 0.0], 0.7)],
        );
        renderer.set_lights(
            &queue,
            [0.05, 0.05, 0.08],
            &[crate::scene::LightDesc::Spot {
                position: [0.0, 8.0, 0.0],
                direction: [0.0, -1.0, 0.0],
                intensity: 2000.0,
                color: [1.0, 1.0, 1.0],
                range: 30.0,
                inner_angle: 25.0,
                outer_angle: 35.0,
                shadow: true,
            }],
        );
        // CPU prediction with the same glam calls as `spot_shadow_vp`.
        let eye = glam::Vec3::new(0.0, 8.0, 0.0);
        let axis = glam::Vec3::new(0.0, -1.0, 0.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, eye + axis, glam::Vec3::X);
        let proj = glam::camera::rh::proj::directx::perspective(
            35.0f32.to_radians() * 2.0,
            1.0,
            0.5,
            30.0,
        );
        let vp = proj * view;
        for p in [[0.8, 3.5, 0.0], [0.0, 0.0, 0.0], [0.8, 4.2, 0.0]] {
            let c = vp * glam::Vec4::new(p[0], p[1], p[2], 1.0);
            eprintln!(
                "spot layer: world={p:?} ndc=({:.3},{:.3},{:.4})",
                c.x / c.w,
                c.y / c.w,
                c.z / c.w
            );
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("spot layer shadows"),
        });
        renderer.render_shadows(&device, &mut encoder, &mesh, 2);
        queue.submit(std::iter::once(encoder.finish()));
        let depths = renderer.read_shadow_layer_for_tests(&device, &queue, 0);
        let size = crate::renderer::SHADOW_SIZE as usize;
        assert_eq!(depths.len(), size * size);
        // Near depths isolate the occluder (its NDC z ≈ 0.88–0.90)
        // from the farther receiver (≈ 0.93+).
        let mut sx = 0u64;
        let mut sy = 0u64;
        let mut n = 0u64;
        for (i, d) in depths.iter().enumerate() {
            if *d < 0.92 {
                sx += (i % size) as u64;
                sy += (i / size) as u64;
                n += 1;
            }
        }
        assert!(n > 5000, "occluder blob missing from the layer: {n} texels");
        let (cx, cy) = (sx as f64 / n as f64, sy as f64 / n as f64);
        eprintln!("spot layer: occluder centroid=({cx:.0},{cy:.0}) n={n}");
        // NDC (0, +0.254) → col 512, row (1−0.254)/2·1024 ≈ 382.
        // The mirrored row would be ≈ 642.
        assert!(
            (cx - 512.0).abs() < 60.0 && (cy - 382.0).abs() < 60.0,
            "occluder at the mirrored texel: ({cx:.0},{cy:.0})"
        );
    }

    /// Point-shadow regression: a shadowed point light must darken the
    /// receiver behind the occluder versus `shadow: false`, in every
    /// technique (deferred, forward, hybrid). Exercises the full chain
    /// (cube-slot assignment → 6 face renders with the hardware
    /// sampler frame → analytic major-axis compare), including the
    /// forward-only plan path, which owns its shadow pre-pass via
    /// `Forward<OwnsDepth>`. Skipped when no adapter is available.
    #[test]
    fn shadowed_point_darkens_occluded_receiver_in_all_techniques() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping shadow probe");
            return;
        };
        const W: u32 = 320;
        const H: u32 = 180;
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let target_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("point shadow probe target"),
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
        let backend = crate::renderer::Renderer3D::new(&device, &surface_config, 1);
        // Unit sphere; sizes come from instance scales (true geometry,
        // unlike the probe examples that reuse the first entity's mesh).
        // (Concrete `Renderer3D` — not the boxed backend — because the
        // plan path below takes `&Renderer3D` directly.)
        let mesh = crate::mesh::create_sphere(&device, 1.0, 24, 16);
        let mut gray = ornis_core::OpenPBRMaterial::dielectric();
        gray.base.color_rgb([0.8, 0.8, 0.8]);
        gray.specular.roughness(0.5);
        backend.upload_materials(&device, &queue, &[gray, gray]);
        let place = |translation: [f32; 3], scale: f32| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(scale),
                glam::Quat::IDENTITY,
                glam::Vec3::from(translation),
            );
            crate::renderer::InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
            }
        };
        // Receiver r=2 at origin; occluder r=0.6 on the light axis.
        // The light sits OFF the cube seams (|x|≠|y|≠|z| along the
        // axis): axis-aligned light→receiver directions ride a cube
        // edge, where the 2x2 PCF footprint straddles two faces and
        // small centered occluders vanish. Light (5,4,6), occluder at
        // t=0.35 along light→origin: (3.25,2.6,3.9). Point range
        // comfortably covers the scene.
        backend.upload_instances(
            &device,
            &queue,
            &[place([0.0, 0.0, 0.0], 2.0), place([3.25, 2.6, 3.9], 0.6)],
        );
        let view = glam::camera::rh::view::look_at_mat4(
            glam::Vec3::new(0.0, 3.0, 10.0),
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Y,
        );
        let proj = glam::camera::rh::proj::directx::perspective(
            55.0f32.to_radians(),
            W as f32 / H as f32,
            0.1,
            100.0,
        );
        backend.set_camera(&queue, &(proj * view).to_cols_array_2d(), [0.0, 3.0, 10.0]);

        let render = |technique: crate::frame_exec::Technique, shadow: bool| -> Vec<u8> {
            backend.set_lights(
                &queue,
                [0.05, 0.05, 0.08],
                &[crate::scene::LightDesc::Point {
                    position: [5.0, 4.0, 6.0],
                    intensity: 200.0,
                    color: [1.0, 1.0, 1.0],
                    range: 30.0,
                    shadow,
                }],
            );
            let mut plan =
                crate::frame_exec::RenderFrame3D::new_with(format, (W, H), technique, false);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("point shadow probe encoder"),
            });
            plan.render(
                RenderContext {
                    device: &device,
                    queue: &queue,
                    encoder: &mut encoder,
                    target: &target_view,
                },
                &backend,
                &mesh,
                2,
            );
            let bpp = 4u32;
            let unpadded = W * bpp;
            let padded = unpadded.div_ceil(256) * 256;
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("point shadow probe readback"),
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
            pixels
        };

        for technique in [
            crate::frame_exec::Technique::Deferred,
            crate::frame_exec::Technique::Forward,
            crate::frame_exec::Technique::Hybrid,
        ] {
            let on = render(technique, true);
            let off = render(technique, false);
            assert_eq!(on.len(), off.len());
            let lum = |px: &[u8]| -> f64 {
                px.chunks_exact(4)
                    .map(|c| (c[0] as f64 + c[1] as f64 + c[2] as f64) / 3.0)
                    .sum::<f64>()
                    / (px.len() / 4) as f64
            };
            let (mean_on, mean_off) = (lum(&on), lum(&off));
            let mut diff_px = 0usize;
            for (a, b) in on.chunks_exact(4).zip(off.chunks_exact(4)) {
                let d = (a[0] as i16 - b[0] as i16).abs()
                    + (a[1] as i16 - b[1] as i16).abs()
                    + (a[2] as i16 - b[2] as i16).abs();
                if d > 12 {
                    diff_px += 1;
                }
            }
            eprintln!(
                "point shadow probe ({technique:?}): mean_on={mean_on:.2} mean_off={mean_off:.2} diff_px={diff_px}"
            );
            assert!(
                diff_px > 500,
                "point shadow had no visible effect ({technique:?}): {diff_px} pixels differ"
            );
            assert!(
                mean_on + 1.0 < mean_off,
                "shadowed frame is not darker ({technique:?}): on={mean_on:.2} off={mean_off:.2}"
            );
        }
    }
}
