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
}
