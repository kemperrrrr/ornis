//! The stats OS window: its own small wgpu stack (instance, device,
//! surface) independent of the render window, so the panel never touches
//! the engine's GPU resources. Each refresh uploads the CPU [`Canvas`]
//! into a texture and blits it with a fullscreen triangle.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ornis_core::{ScheduleTimings, Stage};
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes, WindowId};

use super::StatsCollector;
use super::panel::{Canvas, PIXEL_STRIDE, draw_panel};

/// Vertices of the fullscreen blit triangle.
const FULLSCREEN_TRIANGLE: std::ops::Range<u32> = 0..3;
/// Logical stats window size.
const SIZE: (u32, u32) = (420, 380);

const BLIT_WGSL: &str = "
@group(0) @binding(0) var panel: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(panel));
    return textureLoad(panel, min(vec2<i32>(p.xy), size - 1), 0);
}
";

/// Second window showing [`StatsCollector`] readouts.
pub(crate) struct StatsWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    /// Panel texture and its bind group, re-created on resize.
    target: Option<(wgpu::Texture, wgpu::BindGroup)>,
    canvas: Canvas,
    collector: StatsCollector,
}

impl StatsWindow {
    /// Opens the window and its GPU stack.
    pub fn open(event_loop: &ActiveEventLoop) -> Result<Self, String> {
        let attrs = WindowAttributes::default()
            .with_title("ornis stats")
            .with_inner_size(LogicalSize::new(SIZE.0, SIZE.1));
        let window = Arc::new(event_loop.create_window(attrs).map_err(|e| e.to_string())?);
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| e.to_string())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .map_err(|e| e.to_string())?;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|e| e.to_string())?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or("no surface format")?;
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stats blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stats blit"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(format.into())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let mut stats = Self {
            canvas: Canvas::new(1, 1, 1),
            window,
            surface,
            config,
            device,
            queue,
            pipeline,
            target: None,
            collector: StatsCollector::default(),
        };
        stats.resize(size);
        Ok(stats)
    }

    /// The stats window id (event routing).
    pub fn id(&self) -> WindowId {
        self.window.id()
    }

    /// Reconfigures the surface and the panel for a new physical size.
    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(&self.device, &self.config);
        let scale = self.window.scale_factor().round().max(1.0) as u32;
        self.canvas = Canvas::new(self.config.width, self.config.height, scale);
        self.target = None;
    }

    /// Records one frame and redraws the panel when a refresh is due.
    pub fn on_frame(
        &mut self,
        interval: Duration,
        work: Duration,
        stages: &[(Stage, ScheduleTimings)],
    ) {
        self.collector.record(interval, work, stages);
        self.redraw_if_due(Instant::now());
    }

    /// Redraws and presents the panel when the collector refreshed.
    fn redraw_if_due(&mut self, now: Instant) -> Option<()> {
        self.collector.refresh(now).then_some(())?;
        draw_panel(&mut self.canvas, &self.collector);
        let frame = self.acquire()?;
        let bind_group = self.upload();
        self.blit(frame, &bind_group);
        Some(())
    }

    /// Next swapchain image; reconfigures on `Outdated`/`Lost` and skips
    /// this refresh (the panel is redrawn on the next one).
    fn acquire(&mut self) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Some(frame),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                None
            }
            _ => None,
        }
    }

    /// Uploads the canvas into the panel texture (created on first use
    /// after a resize) and returns its bind group.
    fn upload(&mut self) -> wgpu::BindGroup {
        let extent = wgpu::Extent3d {
            width: self.canvas.width,
            height: self.canvas.height,
            depth_or_array_layers: 1,
        };
        let format = self.config.format;
        let (device, pipeline) = (&self.device, &self.pipeline);
        let (texture, bind_group) = self
            .target
            .get_or_insert_with(|| panel_target(device, pipeline, extent, format));
        self.queue.write_texture(
            texture.as_image_copy(),
            &self.canvas.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.canvas.width * PIXEL_STRIDE),
                rows_per_image: Some(self.canvas.height),
            },
            extent,
        );
        bind_group.clone()
    }

    /// Draws the panel over the whole swapchain image and presents it.
    fn blit(&self, frame: wgpu::SurfaceTexture, bind_group: &wgpu::BindGroup) {
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("stats blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(FULLSCREEN_TRIANGLE, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

/// Panel texture matching the surface encoding (sRGB surface → sRGB
/// texture, so the RGBA bytes reach the screen unchanged) and its bind
/// group.
fn panel_target(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    size: wgpu::Extent3d,
    surface_format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let format = if surface_format.is_srgb() {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("stats panel"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("stats panel"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&view),
        }],
    });
    (texture, bind_group)
}
