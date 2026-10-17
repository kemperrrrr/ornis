//! Opt-in spatial/temporal AA resolve, with persistent ping-pong history.
//!
//! Run on single-sampled, linear color before tone mapping/UI. The caller
//! supplies motion/depth and applies [`jitter_projection`] to geometry only.
//! Temporal state belongs to a camera, never to the transient texture pool.
use glam::{Mat4, Vec2, Vec3, Vec4};
use wgpu::util::DeviceExt;

#[cfg(not(target_arch = "wasm32"))]
mod graph;
#[cfg(not(target_arch = "wasm32"))]
pub use graph::{AaColor, AaMotion, AaPass};

/// Selects geometry sampling and the screen-space resolve independently.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AntiAliasing {
    /// Single-sample rendering without post AA.
    #[default]
    Off,
    /// Geometry-only four-sample MSAA (adapter negotiation still required).
    Msaa,
    /// Single-pass fast approximate AA, suitable for WebGPU.
    Fxaa,
    /// Three-pass morphological AA with analytic area weights.
    ///
    /// This lookup-free SMAA variant omits diagonal/corner refinement.
    Smaa,
    /// Temporal reprojection, depth rejection and neighbourhood clamp.
    Taa,
}

impl AntiAliasing {
    /// Requested geometry sample count; negotiate against the adapter.
    pub const fn sample_count(self) -> u32 {
        match self {
            Self::Msaa => 4,
            _ => 1,
        }
    }

    fn shader_mode(self) -> u32 {
        match self {
            Self::Off => 0,
            Self::Msaa => 1,
            Self::Fxaa => 2,
            Self::Smaa => 3,
            Self::Taa => 4,
        }
    }
}

fn halton(mut index: u32, base: u32) -> f32 {
    let mut value = 0.0;
    let mut fraction = 1.0;
    while index != 0 {
        fraction /= base as f32;
        value += fraction * (index % base) as f32;
        index /= base;
    }
    value
}

/// Eight-frame Halton(2,3) sequence, in pixel units centred around zero.
/// Frame zero is the first nonzero Halton sample, not the corner (-0.5,-0.5).
pub fn temporal_jitter(frame: u64) -> Vec2 {
    let index = (frame % 8) as u32 + 1;
    Vec2::new(halton(index, 2), halton(index, 3)) - Vec2::splat(0.5)
}

/// Offsets clip x/y by clip w; works for perspective and orthographic cameras.
/// Returns the original matrix for an empty viewport. UV y points down.
pub fn jitter_projection(view_projection: Mat4, jitter: Vec2, extent: [u32; 2]) -> Mat4 {
    if extent.contains(&0) {
        return view_projection;
    }
    let offset = Vec2::new(
        2.0 * jitter.x / extent[0] as f32,
        -2.0 * jitter.y / extent[1] as f32,
    );
    let mut shift = Mat4::IDENTITY;
    shift.w_axis.x = offset.x;
    shift.w_axis.y = offset.y;
    shift * view_projection
}

/// Builds previous-UV minus current-UV motion from *jittered* clip positions.
///
/// Use the previous skinned world position for `previous_clip`, not today's
/// position transformed by yesterday's camera. Returns `None` behind either
/// camera or for non-finite input; mark those pixels' previous depth negative.
pub fn motion_vector(current_clip: Vec4, previous_clip: Vec4) -> Option<Vec2> {
    if !current_clip.is_finite()
        || !previous_clip.is_finite()
        || current_clip.w <= 0.0
        || previous_clip.w <= 0.0
    {
        return None;
    }
    let delta = previous_clip.truncate().truncate() / previous_clip.w
        - current_clip.truncate().truncate() / current_clip.w;
    Some(delta * Vec2::new(0.5, -0.5))
}

/// CPU reference for neighbourhood clipping and temporal accumulation.
/// `feedback` is saturated; a rejected history is exactly the current sample.
pub fn temporal_blend(
    current: Vec3,
    history: Vec3,
    neighbourhood: &[Vec3],
    feedback: f32,
    accepted: bool,
) -> Vec3 {
    if !accepted || !history.is_finite() || !feedback.is_finite() {
        return current;
    }
    let (lo, hi) = neighbourhood
        .iter()
        .filter(|sample| sample.is_finite())
        .fold((current, current), |(lo, hi), sample| {
            (lo.min(*sample), hi.max(*sample))
        });
    current.lerp(history.clamp(lo, hi), feedback.clamp(0.0, 1.0))
}

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
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("AA resolve"),
            source: wgpu::ShaderSource::Wgsl(include_str!("resolve.wgsl").into()),
        });
        let mut entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 4,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 5,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AA bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("AA layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry, formats: &[wgpu::TextureFormat]| {
            let targets: Vec<_> = formats
                .iter()
                .map(|&format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect();
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
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
            resolve: pipeline("resolve", &[format, wgpu::TextureFormat::Rgba16Float]),
            edge_pipeline: pipeline("edges", &[wgpu::TextureFormat::Rgba16Float]),
            weight_pipeline: pipeline("weights", &[wgpu::TextureFormat::Rgba16Float]),
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
            let views = [
                color,
                &self.history[previous].view,
                inputs.motion,
                auxiliary,
            ];
            let mut entries: Vec<_> = views
                .iter()
                .enumerate()
                .map(|(binding, view)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: wgpu::BindingResource::TextureView(view),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: 5,
                resource: uniform.as_entire_binding(),
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("AA inputs"),
                layout: &self.layout,
                entries: &entries,
            })
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
    pass.draw(0..3, 0..1);
}

#[cfg(test)]
mod tests;
