use ornis_core::material::OpenPBRMaterial;
use crate::mesh::Mesh;
use crate::renderer::InstanceData;
use wgpu;

#[derive(Debug, Clone)]
pub struct RenderBackendConfig {
    pub surface_config: wgpu::SurfaceConfiguration,
    pub sample_count: u32,
    pub max_objects: u32,
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

pub struct RenderContext<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub target: &'a wgpu::TextureView,
}

pub trait RenderBackend {
    fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32);

    fn set_camera(&mut self, queue: &wgpu::Queue, view_proj: &[[f32; 4]; 4], camera_pos: [f32; 3]);

    fn set_lights(
        &mut self,
        queue: &wgpu::Queue,
        ambient: [f32; 3],
        lights: &[([f32; 3], f32, [f32; 3])],
    );

    fn upload_materials(&mut self, queue: &wgpu::Queue, materials: &[OpenPBRMaterial]);

    fn upload_instances(&mut self, queue: &wgpu::Queue, instances: &[InstanceData]);

    fn render_scene(
        &self,
        context: RenderContext<'_>,
        mesh: &Mesh,
        instance_count: u32,
    );
}

/// Factory function to create a render backend implementation
pub fn create_render_backend(
    device: &wgpu::Device,
    config: &RenderBackendConfig,
) -> Box<dyn RenderBackend> {
    Box::new(crate::renderer::Renderer3D::new(device, &config.surface_config, config.sample_count))
}

pub mod renderer3d_backend {
    use super::*;
    use crate::renderer::Renderer3D;

    impl RenderBackend for Renderer3D {
        fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
            Renderer3D::resize(self, device, width, height);
        }

        fn set_camera(&mut self, queue: &wgpu::Queue, view_proj: &[[f32; 4]; 4], camera_pos: [f32; 3]) {
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

        fn render_scene(
            &self,
            context: RenderContext<'_>,
            mesh: &Mesh,
            instance_count: u32,
        ) {
            Renderer3D::render_scene(self, context.device, context.encoder, context.target, mesh, instance_count);
        }
    }
}