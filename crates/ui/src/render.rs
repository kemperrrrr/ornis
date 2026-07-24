use std::num::NonZeroUsize;

use vello::peniko::kurbo::{Affine, Circle, Rect, RoundedRect, Stroke};
use vello::peniko::{self, Color, Fill, FontData};
use vello::{AaConfig, AaSupport, RenderParams, Renderer, RendererOptions, Scene};
use wgpu::{Device, Queue, Surface, SurfaceConfiguration, Texture, util::TextureBlitter};

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
        let use_cpu = cfg!(target_arch = "wasm32");
        let renderer = Renderer::new(
            device,
            RendererOptions {
                use_cpu,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )?;

        let blitter = TextureBlitter::new(device, surface_format);
        let width = surface_config.width;
        let height = surface_config.height;

        // Vello renders into a storage-capable, linearly-encoded target.
        // srgb surface formats (e.g. Bgra8UnormSrgb) do NOT support
        // STORAGE_BINDING, so we use Rgba8Unorm for the offscreen target and
        // let the composite pass sample it as a plain float texture.
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
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
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
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

    pub fn fill_text(
        &mut self,
        x: f64,
        y: f64,
        text: &str,
        font_size: f32,
        color: Color,
        font: &FontData,
        bold: bool,
        font_weight: Option<f32>,
    ) {
        self.fill_text_with_spacing(x, y, text, font_size, color, font, bold, font_weight, 0.0);
    }

    pub fn fill_text_with_spacing(
        &mut self,
        x: f64,
        y: f64,
        text: &str,
        font_size: f32,
        color: Color,
        font: &FontData,
        bold: bool,
        font_weight: Option<f32>,
        letter_spacing: f32,
    ) {
        let mut glyphs = crate::text::layout_text(font, text, font_size);
        if glyphs.is_empty() {
            return;
        }

        // Apply CSS letter-spacing: shift each glyph's x by cumulative spacing
        if letter_spacing != 0.0 {
            let mut x_offset = 0.0;
            for glyph in &mut glyphs {
                glyph.x += x_offset;
                x_offset += letter_spacing;
            }
        }

        self.brush = peniko::Brush::from(color);
        self.scene
            .draw_glyphs(font)
            .transform(Affine::translate((x, y)))
            .font_size(font_size)
            .brush(color)
            .draw(Fill::NonZero, glyphs.iter().cloned());
        if bold {
            // Gradient fake-bold based on font-weight:
            // 500 (medium): 0.025 — thin thickening
            // 600 (semibold): 0.04 — medium
            // 700+ (bold): 0.06 — strong
            let weight = font_weight.unwrap_or(400.0);
            let weight_factor = if weight >= 700.0 {
                0.06
            } else if weight >= 600.0 {
                0.04
            } else {
                0.025
            };
            let stroke = Stroke {
                width: (font_size as f64 * weight_factor).max(0.3),
                ..Default::default()
            };
            self.scene
                .draw_glyphs(font)
                .transform(Affine::translate((x, y)))
                .font_size(font_size)
                .brush(color)
                .hint(true)
                .draw(&stroke, glyphs.iter().cloned());
        }
    }

    pub fn fill_bez_path(
        &mut self,
        path: &vello::peniko::kurbo::BezPath,
        transform: Affine,
        color: Color,
    ) {
        self.brush = peniko::Brush::from(color);
        self.scene
            .fill(peniko::Fill::NonZero, transform, &self.brush, None, path);
    }

    /// Draws a raster image (`peniko::ImageData`, e.g. a decoded `<img>`) into
    /// the scene, scaling its intrinsic pixel size to `transform`.
    ///
    /// `quality` controls the sampler hint: `High` gives a smooth (linear)
    /// up/downscale so bitmaps such as the editor logo stay anti-aliased
    /// instead of showing nearest-neighbor pixel stairs.
    pub fn draw_image(
        &mut self,
        image: &vello::peniko::ImageData,
        transform: Affine,
        quality: vello::peniko::ImageQuality,
    ) {
        let brush = vello::peniko::ImageBrush::new(image.clone()).with_quality(quality);
        self.scene.draw_image(brush.as_ref(), transform);
    }

    /// Clips all subsequent drawing commands to the given rounded rectangle
    /// (matching a CSS `border-radius`), until [`Self::pop_clip`] is called.
    /// Used to keep a `background-image` inside the element's rounded box.
    pub fn push_rounded_clip(&mut self, x: f64, y: f64, w: f64, h: f64, r: f64) {
        let rect = RoundedRect::new(x, y, x + w, y + h, r);
        self.scene.push_clip_layer(vello::peniko::Fill::NonZero, Affine::IDENTITY, &rect);
    }

    /// Pops the clip layer opened by [`Self::push_rounded_clip`].
    pub fn pop_clip(&mut self) {
        self.scene.pop_layer();
    }

    pub fn get_internal_texture_view(&self) -> wgpu::TextureView {
        self.output
            .as_ref()
            .unwrap()
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub fn render_scene(&mut self, device: &Device, queue: &Queue) -> Result<(), vello::Error> {
        let output = self.output.take().unwrap();
        let view = output.create_view(&wgpu::TextureViewDescriptor::default());

        let result = self.renderer.render_to_texture(
            device,
            queue,
            &self.scene,
            &view,
            &RenderParams {
                base_color: Color::new([0.0, 0.0, 0.0, 0.0]),
                width: self.width,
                height: self.height,
                antialiasing_method: AaConfig::Area,
            },
        );

        self.output = Some(output);
        result
    }

    pub fn blit_to_surface(
        &mut self,
        device: &Device,
        encoder: &mut wgpu::CommandEncoder,
        surface_texture: &wgpu::TextureView,
    ) {
        let output = self.output.as_ref().unwrap();
        let view = output.create_view(&wgpu::TextureViewDescriptor::default());
        self.blitter.copy(device, encoder, &view, surface_texture);
    }

    pub fn end_frame(
        &mut self,
        device: &Device,
        queue: &Queue,
        surface: &Surface,
    ) -> Result<(), vello::Error> {
        self.render_scene(device, queue)?;

        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            _ => return Err(vello::Error::UnsupportedSurfaceFormat),
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ornis ui blit"),
        });
        self.blit_to_surface(device, &mut encoder, &frame_view);
        queue.submit(Some(encoder.finish()));
        frame.present();

        Ok(())
    }

    /// Renders the current scene to the offscreen `output` texture and copies it
    /// back to the CPU, writing an 8-bit RGBA PNG to `path`. Used for offline
    /// visual verification (the engine otherwise only renders to a live surface).
    pub fn save_png(
        &mut self,
        device: &Device,
        queue: &Queue,
        path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.render_scene(device, queue)?;

        let output = self.output.as_ref().unwrap();
        let (w, h) = (self.width, self.height);
        let bytes_per_pixel = 4u32;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded = w * bytes_per_pixel;
        let padded = (unpadded + align - 1) & !(align - 1);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("png readback"),
            size: (padded as u64) * (h as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("png copy"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: output,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let mapped = slice.get_mapped_range();
        let mut img_data = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let src = &mapped[(y * padded) as usize..(y * padded + unpadded) as usize];
            img_data.extend_from_slice(src);
        }
        drop(mapped);
        buffer.unmap();

        let file = std::fs::File::create(path)?;
        let mut enc = png::Encoder::new(file, w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&img_data)?;
        Ok(())
    }
}
