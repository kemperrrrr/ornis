use std::num::NonZeroUsize;

use vello::peniko::{self, Color, Fill, FontData};
use vello::peniko::kurbo::{Affine, Circle, Rect, RoundedRect, Stroke};
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};
use wgpu::{util::TextureBlitter, Device, Queue, Surface, SurfaceConfiguration, Texture};

pub struct UIRenderer {
    renderer: Renderer,
    scene: Scene,
    blitter: TextureBlitter,
    output: Option<Texture>,
    brush: peniko::Brush,
    width: u32,
    height: u32,
}

impl UIRenderer {
    pub fn new(
        device: &Device,
        surface_config: &SurfaceConfiguration,
        surface_format: wgpu::TextureFormat,
    ) -> Result<Self, vello::Error> {
        let renderer = Renderer::new(
            device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )?;

        let blitter = TextureBlitter::new(device, surface_format);
        let width = surface_config.width;
        let height = surface_config.height;

        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello output"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        Ok(UIRenderer {
            renderer,
            scene: Scene::new(),
            blitter,
            output: Some(output),
            brush: peniko::Brush::from(Color::new([1.0, 1.0, 1.0, 1.0])),
            width,
            height,
        })
    }

    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.output = Some(device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello output"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
    }

    pub fn begin_frame(&mut self) {
        self.scene.reset();
    }

    pub fn fill_rect(&mut self, x: f64, y: f64, w: f64, h: f64, color: Color) {
        let rect = Rect::new(x, y, x + w, y + h);
        self.brush = peniko::Brush::from(color);
        self.scene.fill(
            peniko::Fill::NonZero,
            Affine::IDENTITY,
            &self.brush,
            None,
            &rect,
        );
    }

    pub fn fill_rounded_rect(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64, color: Color) {
        let rect = RoundedRect::new(x, y, x + w, y + h, r);
        self.brush = peniko::Brush::from(color);
        self.scene.fill(
            peniko::Fill::NonZero,
            Affine::IDENTITY,
            &self.brush,
            None,
            &rect,
        );
    }

    pub fn stroke_rect(&mut self, x: f64, y: f64, w: f64, h: f64, width: f64, color: Color) {
        let rect = Rect::new(x, y, x + w, y + h);
        self.brush = peniko::Brush::from(color);
        self.scene.stroke(
            &Stroke::new(width),
            Affine::IDENTITY,
            &self.brush,
            None,
            &rect,
        );
    }

    pub fn fill_circle(&mut self, cx: f64, cy: f64, r: f64, color: Color) {
        self.brush = peniko::Brush::from(color);
        self.scene.fill(
            peniko::Fill::NonZero,
            Affine::IDENTITY,
            &self.brush,
            None,
            &Circle::new((cx, cy), r),
        );
    }

    pub fn fill_text(&mut self, x: f64, y: f64, text: &str, font_size: f32, color: Color, font: &FontData) {
        let glyphs = crate::text::layout_text(font, text, font_size);
        if glyphs.is_empty() {
            return;
        }
        self.brush = peniko::Brush::from(color);
        self.scene.draw_glyphs(font)
            .transform(Affine::translate((x, y)))
            .font_size(font_size)
            .brush(color)
            .draw(Fill::NonZero, glyphs.into_iter());
    }

    pub fn end_frame(
        &mut self,
        device: &Device,
        queue: &Queue,
        surface: &Surface,
    ) -> Result<(), vello::Error> {
        let output = self.output.take().unwrap();
        let view = output.create_view(&wgpu::TextureViewDescriptor::default());

        self.renderer.render_to_texture(
            device,
            queue,
            &self.scene,
            &view,
            &RenderParams {
                base_color: Color::new([0.0, 0.0, 0.0, 1.0]),
                width: self.width,
                height: self.height,
                antialiasing_method: AaConfig::Area,
            },
        )?;

        let frame = surface.get_current_texture()
            .map_err(|_| vello::Error::UnsupportedSurfaceFormat)?;
        let frame_view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ornis ui blit"),
        });
        self.blitter.copy(device, &mut encoder, &view, &frame_view);
        queue.submit(Some(encoder.finish()));
        frame.present();

        self.output = Some(output);
        self.scene.reset();

        Ok(())
    }
}
