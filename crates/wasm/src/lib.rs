use wasm_bindgen::prelude::*;
use web_sys::console;

// ═══════════════════════════════════════════════════════════════════════════
// Ornis WASM — WebGPU entry point for browser editor
// Build: wasm-pack build crates/wasm --target web
// ═══════════════════════════════════════════════════════════════════════════

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    console::log_1(&"[ornis-wasm] module loaded".into());
}

/// Wrapper to give HtmlCanvasElement a raw window handle for wgpu.
struct CanvasWindow(web_sys::HtmlCanvasElement);

impl raw_window_handle::HasWindowHandle for CanvasWindow {
    fn window_handle(
        &self,
    ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
        use core::ptr::NonNull;
        use raw_window_handle::{RawWindowHandle, WebCanvasWindowHandle, WindowHandle};

        let js_value: &wasm_bindgen::JsValue = &self.0;
        let obj = NonNull::from(js_value).cast();
        let web_handle = WebCanvasWindowHandle::new(obj);
        let raw = RawWindowHandle::WebCanvas(web_handle);
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

/// Initialize WebGPU on a canvas and start render loop.
#[wasm_bindgen]
pub async fn start_renderer(canvas_id: String) -> Result<(), JsValue> {
    console::log_1(&format!("[ornis-wasm] init canvas={}", canvas_id).into());

    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let canvas: web_sys::HtmlCanvasElement = document
        .get_element_by_id(&canvas_id)
        .ok_or("canvas not found")?
        .dyn_into()?;

    // Resize canvas to match viewport
    let resize = |c: &web_sys::HtmlCanvasElement| {
        let parent = c.parent_element().unwrap();
        c.set_width(parent.client_width() as u32);
        c.set_height(parent.client_height() as u32);
    };
    resize(&canvas);

    // ── wgpu WebGPU setup ─────────────────────────────────────────────
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        flags: wgpu::InstanceFlags::empty(),
        backend_options: wgpu::BackendOptions::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        display: None,
    });

    let surface = unsafe {
        instance
            .create_surface_unsafe(
                wgpu::SurfaceTargetUnsafe::from_window(&CanvasWindow(canvas.clone()))
                    .map_err(|e| format!("surface target: {:?}", e))?,
            )
            .map_err(|e| format!("surface: {:?}", e))?
    };

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .map_err(|_| "adapter not found")?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("ornis-wasm"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        })
        .await
        .map_err(|e| format!("device: {:?}", e))?;

    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats[0];

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: canvas.width(),
        height: canvas.height(),
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    console::log_1(
        &format!("[ornis-wasm] WebGPU ready, format={:?}", surface_format).into(),
    );

    // ── Simple triangle shader ────────────────────────────────────────
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("triangle"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        cache: None,
        multiview_mask: None,
    });

    // ── Render loop ───────────────────────────────────────────────────
    let f = std::rc::Rc::new(std::cell::RefCell::new(
        None as Option<Closure<dyn FnMut()>>,
    ));
    let f_clone = f.clone();

    let window_for_loop = window.clone();
    let canvas_for_loop = canvas.clone();

    let f_inner = f.clone();
    *f_clone.borrow_mut() = Some(Closure::new(move || {
        // Handle resize
        let pw = canvas_for_loop
            .parent_element()
            .map(|p| p.client_width() as u32)
            .unwrap_or(canvas_for_loop.width());
        let ph = canvas_for_loop
            .parent_element()
            .map(|p| p.client_height() as u32)
            .unwrap_or(canvas_for_loop.height());
        if canvas_for_loop.width() != pw || canvas_for_loop.height() != ph {
            canvas_for_loop.set_width(pw);
            canvas_for_loop.set_height(ph);
            config.width = pw;
            config.height = ph;
            surface.configure(&device, &config);
        }

        match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render_encoder"),
                });

                {
                    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("render_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.05,
                                    g: 0.05,
                                    b: 0.07,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                            depth_slice: None,
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });

                    render_pass.set_pipeline(&render_pipeline);
                    render_pass.draw(0..3, 0..1);
                }

                queue.submit(std::iter::once(encoder.finish()));
                frame.present();
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Outdated
            | wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip frame, will retry next animation frame
            }
        }

        // Schedule next frame
        window_for_loop
            .request_animation_frame(f_inner.borrow().as_ref().unwrap().as_ref().unchecked_ref())
            .unwrap();
    }));

    window
        .request_animation_frame(f_clone.borrow().as_ref().unwrap().as_ref().unchecked_ref())?;

    // Prevent dropping the closure so the JS callback stays valid
    std::mem::forget(f);
    std::mem::forget(f_clone);

    Ok(())
}
