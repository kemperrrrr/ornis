//! Shared AA pipeline and binding construction.
const TEXTURE_BINDINGS: u32 = 4;
const SAMPLER_BINDING: u32 = 4;
const PARAMS_BINDING: u32 = 5;

pub(super) fn create(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> (
    wgpu::BindGroupLayout,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
    wgpu::RenderPipeline,
) {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("AA resolve"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../resolve.wgsl").into()),
    });
    let layout = bindings_layout(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("AA layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let resolve = pipeline(
        device,
        &module,
        &pipeline_layout,
        "resolve",
        &[format, wgpu::TextureFormat::Rgba16Float],
    );
    let edges = pipeline(
        device,
        &module,
        &pipeline_layout,
        "edges",
        &[wgpu::TextureFormat::Rgba16Float],
    );
    let weights = pipeline(
        device,
        &module,
        &pipeline_layout,
        "weights",
        &[wgpu::TextureFormat::Rgba16Float],
    );
    (layout, resolve, edges, weights)
}

fn bindings_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries: Vec<_> = (0..TEXTURE_BINDINGS)
        .map(|binding| {
            layout_entry(
                binding,
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
            )
        })
        .collect();
    entries.push(layout_entry(
        SAMPLER_BINDING,
        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
    ));
    entries.push(layout_entry(
        PARAMS_BINDING,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    ));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("AA bindings"),
        entries: &entries,
    })
}

fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry: &str,
    formats: &[wgpu::TextureFormat],
) -> wgpu::RenderPipeline {
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
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
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
}

pub(super) fn bindings(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
    views: [&wgpu::TextureView; TEXTURE_BINDINGS as usize],
) -> wgpu::BindGroup {
    let mut entries: Vec<_> = views
        .iter()
        .enumerate()
        .map(|(binding, view)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: wgpu::BindingResource::TextureView(view),
        })
        .collect();
    entries.push(wgpu::BindGroupEntry {
        binding: SAMPLER_BINDING,
        resource: wgpu::BindingResource::Sampler(sampler),
    });
    entries.push(wgpu::BindGroupEntry {
        binding: PARAMS_BINDING,
        resource: uniform.as_entire_binding(),
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("AA inputs"),
        layout,
        entries: &entries,
    })
}

fn layout_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}
