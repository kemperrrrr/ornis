//! Backend-neutral rendering interface.
//!
//! [`RenderBackend`] abstracts the deferred renderer behind a small trait so
//! callers (and tests) can drive a full frame — camera, lights, uploads, one
//! draw — without touching [`crate::renderer::Renderer3D`] directly. The
//! factory [`create_render_backend`] returns the production implementation.
use crate::flags::AnisotropyLevel;
use crate::mesh::Mesh;
use crate::renderer::InstanceData;
use crate::textures::{MaterialTextureSet, TextureCache};
use ornis_assets::scene::LightDesc;
use ornis_core::material::OpenPBRMaterial;

use wgpu;

/// Sizing and capacity knobs for backend construction.
#[derive(Debug, Clone)]
pub struct RenderBackendConfig {
    /// Surface format/size/present parameters; must be compatible with the
    /// target surface (or an offscreen texture in tests).
    pub surface_config: wgpu::SurfaceConfiguration,
    /// MSAA sample count for the gbuffer and lighting passes.
    ///
    /// `1` (the default) is single-sample; pass
    /// [`crate::renderer::MSAA_SAMPLE_COUNT`] (4) for the native MSAA path
    /// after gating it through
    /// [`crate::renderer::negotiate_sample_count`] against the adapter
    /// (software adapters fall back to 1 — never a panic). Fullscreen
    /// passes and shadow maps stay single-sample in all modes; the
    /// frame-plan pool follows this count for its geometry layers (see
    /// [`crate::frame_exec::RenderFrame3D::new_with_samples`]).
    pub sample_count: u32,
    /// Exposure multiplier baked into every light color by
    /// `Renderer3D::set_lights_full` (via [`create_render_backend`]);
    /// `1.0` is the exact no-op.
    pub exposure: f32,
    /// Upper bound on instances per frame (sized into GPU buffers).
    pub max_objects: u32,
    /// Upper bound on materials per frame.
    pub max_materials: u32,
    /// Opt-in indirect submission of the shared mesh batch (slice 2c):
    /// the frame compacts visible instances on the GPU and draws them
    /// with one `draw_indexed_indirect`; `false` (default) keeps the
    /// direct `draw_indexed` path. Shadows always draw directly.
    pub indirect_draw: bool,
    /// Anisotropic level of the textured-forward material sampler.
    ///
    /// [`AnisotropyLevel::DEFAULT`] (16) is the project default. The value
    /// reaches [`crate::renderer::Renderer3D::ensure_textured_forward`]; it
    /// is not part of the texture-cache key, so one image stays one GPU
    /// texture whatever level a caller asks for at upload. Shadow and
    /// post-process samplers stay isotropic.
    pub material_anisotropy: AnisotropyLevel,
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
            exposure: 1.0,
            max_objects: 256,
            max_materials: 64,
            indirect_draw: false,
            material_anisotropy: AnisotropyLevel::DEFAULT,
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
    /// uploads the first eight of any kind (see
    /// `crate::renderer::MAX_LIGHTS`) and reports drops via
    /// `crate::renderer::Renderer3D::light_upload_stats`.
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

    /// Anisotropic level currently stored for the material sampler.
    fn material_anisotropy(&self) -> AnisotropyLevel;

    /// Replace the material-sampler level.
    ///
    /// A raw settings value saturates in [`AnisotropyLevel::new`] before it
    /// is passed here. A textured-forward pass that already exists gets a
    /// new sampler; one that does not is built later at this level. `device`
    /// is the sampler's owner.
    fn set_material_anisotropy(&mut self, device: &wgpu::Device, level: AnisotropyLevel);

    /// Record the textured-forward pass: the material sampler is the one
    /// configured by [`RenderBackendConfig::material_anisotropy`], and
    /// `textures` resolve through `cache` (unbound roles stay neutral white).
    ///
    /// `context.target` must be a single-sample `Rgba16Float` view matching
    /// the renderer's extent; `depth` must be `Depth32Float` of that extent.
    /// Builds the pass on first use.
    fn render_textured_forward(
        &mut self,
        context: RenderContext<'_>,
        depth: &wgpu::TextureView,
        mesh: &Mesh,
        instance_count: u32,
        textures: &MaterialTextureSet,
        cache: &TextureCache,
    );
}

/// Build the production [`RenderBackend`] (the deferred [`crate::renderer::Renderer3D`])
/// from `config`.
pub fn create_render_backend(
    device: &wgpu::Device,
    config: &RenderBackendConfig,
) -> Box<dyn RenderBackend> {
    let mut renderer =
        crate::renderer::Renderer3D::new(device, &config.surface_config, config.sample_count);
    renderer.set_exposure(config.exposure);
    renderer.set_indirect_draw(config.indirect_draw);
    renderer.set_material_anisotropy(device, config.material_anisotropy);
    Box::new(renderer)
}

/// Adapter implementing [`RenderBackend`] by delegating to the concrete
/// [`crate::renderer::Renderer3D`]; this is what [`create_render_backend`] hands out.
pub mod renderer3d_backend {
    use super::*;
    use crate::flags::AnisotropyLevel;
    use crate::renderer::Renderer3D;
    use crate::textures::{MaterialTextureSet, TextureCache};

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

        fn material_anisotropy(&self) -> AnisotropyLevel {
            Renderer3D::material_anisotropy(self)
        }

        fn set_material_anisotropy(&mut self, device: &wgpu::Device, level: AnisotropyLevel) {
            Renderer3D::set_material_anisotropy(self, device, level);
        }

        fn render_textured_forward(
            &mut self,
            context: RenderContext<'_>,
            depth: &wgpu::TextureView,
            mesh: &Mesh,
            instance_count: u32,
            textures: &MaterialTextureSet,
            cache: &TextureCache,
        ) {
            self.ensure_textured_forward(context.device, context.queue);
            self.render_forward_textured(
                context.device,
                context.encoder,
                depth,
                context.target,
                mesh,
                instance_count,
                true,
                textures,
                cache,
            );
        }
    }
}

#[cfg(test)]
mod tests;
