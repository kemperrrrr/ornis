//! Per-camera GPU targets, resolve recording and history lifecycle.
use super::{AntiAliasing, temporal_jitter};
use glam::Vec2;
use wgpu::util::DeviceExt;
mod pipeline;
const FULLSCREEN_VERTICES: u32 = 3;

struct Target {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Target {
    fn new(device: &wgpu::Device, extent: [u32; 2]) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("AA persistent target"),
            size: wgpu::Extent3d {
                width: extent[0],
                height: extent[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

/// GPU inputs: all textures must be single-sampled and match the output extent.
pub struct AaInputs<'a> {
    /// Linear color, without UI. Must not alias the output.
    pub color: &'a wgpu::TextureView,
    /// RGBA16Float: previous minus current UV, current linear depth,
    /// expected previous linear depth. Negative previous depth rejects history.
    /// For spatial modes bind any initialized, matching-size float texture.
    pub motion: &'a wgpu::TextureView,
    /// Destination matching the format passed to [`AaResolve::new`].
    pub target: &'a wgpu::TextureView,
}

/// Per-camera GPU resolve; history is invalidated by resize/mode switches.
/// Camera cuts, teleports and scene changes must call [`Self::reset_history`].
/// Submitted frames must execute in order; advance only for rendered frames.
pub struct AaResolve {
    mode: AntiAliasing,
    extent: [u32; 2],
    frame: u64,
    valid: bool,
    history: [Target; 2],
    edges: Target,
    weights: Target,
    sampler: wgpu::Sampler,
    layout: wgpu::BindGroupLayout,
    resolve: wgpu::RenderPipeline,
    edge_pipeline: wgpu::RenderPipeline,
    weight_pipeline: wgpu::RenderPipeline,
}

impl AaResolve {
    /// Allocates pipelines and four RGBA16Float targets (32 bytes/pixel).
    /// Zero dimensions are clamped to one; callers should skip minimized frames.
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        extent: [u32; 2],
        mode: AntiAliasing,
    ) -> Self {
        let extent = extent.map(|v| v.max(1));
        let (layout, resolve, edge_pipeline, weight_pipeline) = pipeline::create(device, format);
        Self {
            mode,
            extent,
            frame: 0,
            valid: false,
            history: [Target::new(device, extent), Target::new(device, extent)],
            edges: Target::new(device, extent),
            weights: Target::new(device, extent),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("AA clamp sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            resolve,
            edge_pipeline,
            weight_pipeline,
            layout,
        }
    }

    /// Switches the post mode; geometry sample count is managed by the caller.
    pub fn set_mode(&mut self, mode: AntiAliasing) {
        if self.mode != mode {
            self.mode = mode;
            self.reset_history();
        }
    }

    /// Invalidates temporal accumulation and restarts the jitter sequence.
    pub fn reset_history(&mut self) {
        self.valid = false;
        self.frame = 0;
    }

    /// Recreates size-dependent targets only when the extent changes.
    pub fn resize(&mut self, device: &wgpu::Device, extent: [u32; 2]) {
        let extent = extent.map(|v| v.max(1));
        if self.extent == extent {
            return;
        }
        self.extent = extent;
        self.history = [Target::new(device, extent), Target::new(device, extent)];
        self.edges = Target::new(device, extent);
        self.weights = Target::new(device, extent);
        self.reset_history();
    }

    /// Current geometry jitter in pixels (zero outside TAA).
    pub fn jitter(&self) -> Vec2 {
        if self.mode == AntiAliasing::Taa {
            temporal_jitter(self.frame)
        } else {
            Vec2::ZERO
        }
    }

    /// Records one resolve (three draws for SMAA), then advances history.
    /// No texture copies, readbacks or queue writes; uniforms are frame-local.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inputs: AaInputs<'_>,
    ) {
        let previous = (self.frame % 2) as usize;
        let next = 1 - previous;
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AA params"),
            contents: bytemuck::cast_slice(&[self.mode.shader_mode(), u32::from(self.valid), 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bindings = |color, auxiliary| {
            pipeline::bindings(
                device,
                &self.layout,
                &self.sampler,
                &uniform,
                [
                    color,
                    &self.history[previous].view,
                    inputs.motion,
                    auxiliary,
                ],
            )
        };
        if self.mode == AntiAliasing::Smaa {
            let group = bindings(inputs.color, inputs.color);
            draw(encoder, &self.edge_pipeline, &group, &[&self.edges.view]);
            let group = bindings(&self.edges.view, inputs.color);
            draw(
                encoder,
                &self.weight_pipeline,
                &group,
                &[&self.weights.view],
            );
        }
        let auxiliary = if self.mode == AntiAliasing::Smaa {
            &self.weights.view
        } else {
            inputs.color
        };
        let group = bindings(inputs.color, auxiliary);
        draw(
            encoder,
            &self.resolve,
            &group,
            &[inputs.target, &self.history[next].view],
        );
        self.frame = self.frame.wrapping_add(1);
        self.valid = self.mode == AntiAliasing::Taa;
    }
}

fn draw(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    group: &wgpu::BindGroup,
    targets: &[&wgpu::TextureView],
) {
    let attachments: Vec<_> = targets
        .iter()
        .map(|view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        })
        .collect();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("AA"),
        color_attachments: &attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[]);
    pass.draw(0..FULLSCREEN_VERTICES, 0..1);
}

#[cfg(test)]
#[path = "gpu_tests.rs"]
mod tests;
