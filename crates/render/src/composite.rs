//! Final full-screen pass: blends the lit PBR image with the 2D UI texture.
//!
//! Draws one triangle-strip quad over the output target and mixes
//! `ui.rgb` into `pbr.rgb` weighted by `ui.a`, decoding the sRGB-encoded
//! UI bytes to linear light first (see the WGSL comment) so the hardware's
//! single OETF re-encode lands on the correct value.

const COMPOSITE_SHADER: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var pbr_tex: texture_2d<f32>;
@group(0) @binding(1) var pbr_sampler: sampler;
@group(0) @binding(2) var ui_tex: texture_2d<f32>;
@group(0) @binding(3) var ui_sampler: sampler;

const QUAD: array<vec4<f32>, 4> = array<vec4<f32>, 4>(
    vec4<f32>(-1.0, -1.0, 0.0, 1.0),
    vec4<f32>(-1.0,  1.0, 0.0, 1.0),
    vec4<f32>( 1.0, -1.0, 0.0, 1.0),
    vec4<f32>( 1.0,  1.0, 0.0, 1.0),
);
const UVS: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
);

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    out.position = QUAD[idx];
    out.uv = UVS[idx];
    return out;
}

// Converts an sRGB-encoded channel to linear light. Vello renders into a
// non-srgb Rgba8Unorm target and applies the sRGB OETF itself, so the UI
// texture holds sRGB-encoded bytes. The surface is an *_Srgb format, so the
// hardware re-applies the OETF on write. Without decoding here the UI would be
// sRGB-encoded twice -> visibly washed out / too light. Decoding to linear
// before the mix lets the hardware's single re-encode land on the right value.
fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let cutoff = c <= vec3<f32>(0.04045);
    let low = c / 12.92;
    let high = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(high, low, cutoff);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
    // pbr_tex is an *_Srgb texture -> the sampler already returns linear light.
    let bg = textureSampleLevel(pbr_tex, pbr_sampler, input.uv, 0.0);
    // ui_tex is Rgba8Unorm holding sRGB-encoded bytes -> decode to linear.
    let ui = textureSampleLevel(ui_tex, ui_sampler, input.uv, 0.0);
    let ui_linear = srgb_to_linear(ui.rgb);
    return vec4<f32>(mix(bg.rgb, ui_linear, ui.a), 1.0);
}
"#;

/// Final full-screen pass blending the lit PBR image with the 2D UI layer.
pub struct CompositePass {
    /// Full-screen triangle-strip pipeline targeting the surface format.
    pipeline: wgpu::RenderPipeline,
    /// Layout binding pbr/ui textures + shared sampler (bindings 0..3).
    bind_group_layout: wgpu::BindGroupLayout,
    /// Linear-filtering, clamp-to-edge sampler used by both textures.
    sampler: wgpu::Sampler,
}

impl CompositePass {
    /// Build the pipeline against `surface_format` (the final output format).
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(COMPOSITE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("composite pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("composite pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    /// Record the blend pass: `target = mix(pbr, srgb_to_linear(ui), ui.a)`.
    ///
    /// `pbr_texture` must be an `*_Srgb` view (sampler returns linear);
    /// `ui_texture` is plain `Rgba8Unorm` holding sRGB-encoded bytes.
    pub fn compose(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        pbr_texture: &wgpu::TextureView,
        ui_texture: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(pbr_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(ui_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("composite render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.draw(0..4, 0..1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 2;
    const BPP: u32 = 4;
    const ROW: u32 = WIDTH * BPP; // 256, satisfies the copy alignment

    /// None when no adapter is available (headless CI without lavapipe).
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

    fn solid_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
        format: wgpu::TextureFormat,
        rgba: [u8; 4],
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mut data = vec![0u8; (ROW * HEIGHT) as usize];
        for px in data.chunks_exact_mut(4) {
            px.copy_from_slice(&rgba);
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ROW),
                rows_per_image: Some(HEIGHT),
            },
            texture.size(),
        );
        texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn compose_pixel(ui_rgba: [u8; 4]) -> Option<[u8; 4]> {
        let (device, queue) = try_device()?;
        let pass = CompositePass::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        // Opaque red PBR background (sRGB target format, as in production).
        let pbr = solid_texture(
            &device,
            &queue,
            "pbr",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            [255, 0, 0, 255],
        );
        let ui = solid_texture(
            &device,
            &queue,
            "ui",
            wgpu::TextureFormat::Rgba8Unorm,
            ui_rgba,
        );
        let target_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("composite target"),
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let target = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite test encoder"),
        });
        pass.compose(&device, &mut encoder, &target, &pbr, &ui);

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite readback"),
            size: (ROW * HEIGHT) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: Some(HEIGHT),
                },
            },
            target_texture.size(),
        );
        queue.submit([encoder.finish()]);

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
        rx.recv().ok()?.ok()?;
        let view = slice.get_mapped_range();
        let pixel = [view[0], view[1], view[2], view[3]];
        Some(pixel)
    }

    #[test]
    fn opaque_ui_replaces_pbr_background() {
        let Some(px) = compose_pixel([0, 255, 0, 255]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // ui.a = 1 -> pure UI green after the sRGB round trip.
        assert!(px[0] < 10 && px[1] > 240 && px[2] < 10, "got {px:?}");
    }

    #[test]
    fn transparent_ui_keeps_pbr_background() {
        let Some(px) = compose_pixel([0, 255, 0, 0]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        // ui.a = 0 -> untouched PBR red.
        assert!(px[0] > 240 && px[1] < 10 && px[2] < 10, "got {px:?}");
    }
}
