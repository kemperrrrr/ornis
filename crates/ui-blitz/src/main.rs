use anyrender::render_to_buffer;
use anyrender_vello::VelloImageRenderer;
use blitz_dom::DocumentConfig;
use blitz_html::HtmlDocument;
use blitz_paint::paint_scene;
use blitz_traits::shell::{ColorScheme, Viewport};
use std::num::NonZeroUsize;
use std::sync::Arc;
use vello::peniko::Color;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::WindowAttributes;

struct Ctx {
    #[allow(dead_code)]
    window: winit::window::Window,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    sc: wgpu::SurfaceConfiguration,
    vello: vello::Renderer,
    blitter: wgpu::util::TextureBlitter,
    doc: HtmlDocument,
}

struct App {
    ctx: Option<Ctx>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.ctx.is_some() {
            return;
        }
        let rt = tokio::runtime::Runtime::new().unwrap();
        match rt.block_on(init(el)) {
            Ok(c) => {
                eprintln!("blitz: ready");
                self.ctx = Some(c);
            }
            Err(e) => {
                eprintln!("blitz: {e}");
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: winit::window::WindowId, ev: WindowEvent) {
        let c = match &mut self.ctx {
            Some(c) => c,
            None => return,
        };
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => {
                let (w, h) = (c.sc.width, c.sc.height);
                let rgba = render_to_buffer::<VelloImageRenderer, _>(|scene| {
                    use anyrender::PaintScene;
                    scene.fill(vello::peniko::Fill::NonZero, Default::default(), Color::BLACK, Default::default(),
                        &vello::peniko::kurbo::Rect::new(0.0, 0.0, w as f64, h as f64));
                    paint_scene(scene, c.doc.as_mut(), 1.0, w, h, 0, 0);
                }, w, h);

                // Upload RGBA → Rgba8Unorm texture
                let tex = c.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("blitz_src"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                c.queue.write_texture(
                    wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                    &rgba,
                    wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                    wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                );

                // Create Rgba8Unorm render target for Vello
                let rt = c.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("blitz_rt"),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                });
                let rt_view = rt.create_view(&wgpu::TextureViewDescriptor::default());

                // Draw Blitz texture as image in a Vello scene
                let mut scene = vello::Scene::new();
                let handle = c.vello.register_texture(tex);
                scene.draw_image(&handle, vello::peniko::kurbo::Affine::IDENTITY);

                // Render Vello scene → Rgba8Unorm render target
                c.vello.render_to_texture(&c.device, &c.queue, &scene, &rt_view,
                    &vello::RenderParams {
                        base_color: Color::BLACK,
                        width: w, height: h,
                        antialiasing_method: vello::AaConfig::Area,
                    }).ok();

                // Blit Rgba8Unorm → surface frame
                let frame = match c.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(f) => f,
                    _ => return,
                };
                let fv = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = c.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("blitz blit") });
                c.blitter.copy(&c.device, &mut encoder, &rt_view, &fv);
                c.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            WindowEvent::Resized(sz) => {
                c.sc.width = sz.width.max(1);
                c.sc.height = sz.height.max(1);
                c.surface.configure(&c.device, &c.sc);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(c) = &mut self.ctx {
            c.window.request_redraw();
        }
    }
}

async fn init(el: &ActiveEventLoop) -> Result<Ctx, String> {
    let window = el
        .create_window(WindowAttributes::default().with_title("Ornis — Blitz").with_inner_size(PhysicalSize::new(1200, 800)))
        .map_err(|e| format!("{e}"))?;
    let sz = window.inner_size();
    let inst = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(), flags: wgpu::InstanceFlags::empty(),
        memory_budget_thresholds: Default::default(), backend_options: Default::default(), display: None,
    });
    let st = unsafe { wgpu::SurfaceTargetUnsafe::from_display_and_window(el, &window) }.unwrap();
    let surface = unsafe { inst.create_surface_unsafe(st) }.unwrap();
    let adapter = match pollster::block_on(inst.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false, compatible_surface: Some(&surface),
    })) {
        Ok(a) => a, _ => return Err("no adapter".into()),
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("blitz"), required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        },
    )).map_err(|e| format!("{e}"))?;

    let caps = surface.get_capabilities(&adapter);
    let fmt = wgpu::TextureFormat::Bgra8Unorm;
    let sc = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: fmt,
        width: sz.width.max(1), height: sz.height.max(1),
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![], desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &sc);

    let vello = vello::Renderer::new(&device, vello::RendererOptions {
        use_cpu: false, antialiasing_support: vello::AaSupport::area_only(),
        num_init_threads: NonZeroUsize::new(1), pipeline_cache: None,
    }).map_err(|e| format!("{e}"))?;

    let blitter = wgpu::util::TextureBlitter::new(&device, fmt);

    let html_raw = include_str!("../../ui/assets/editor/index.html");
    let body = &html_raw[html_raw.find("<body").unwrap_or(0)..html_raw.find("</body>").unwrap_or(html_raw.len())];
    let html = format!("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><link rel=\"stylesheet\" href=\"index.css\"></head>{body}</body></html>");
    let net = Arc::new(blitz_net::Provider::new(None));
    let mut doc = HtmlDocument::from_html(&html, DocumentConfig {
        base_url: Some("file:///Users/a0000/AI-Projects/ornis/crates/ui/assets/editor/".to_string()),
        net_provider: Some(Arc::clone(&net) as _),
        viewport: Some(Viewport::new(sz.width, sz.height, 1.0, ColorScheme::Dark)),
        ..Default::default()
    });
    for _ in 0..500 { doc.resolve(0.0); if net.is_empty() { break; } }
    doc.as_mut().resolve(0.0);
    doc.as_mut().resolve_layout();

    Ok(Ctx { window, device, queue, surface, sc, vello, blitter, doc })
}

fn main() {
    EventLoop::new().unwrap().run_app(&mut App { ctx: None }).unwrap();
}
