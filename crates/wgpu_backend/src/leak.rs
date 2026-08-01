/// Build a LEAK-style WGSL compute shader string.
/// The shader loads a block into shared memory, processes it, and writes dirty flags.
///
/// `block_size`: workgroup size (64–256).
/// `body`: the per-element computation in WGSL (e.g. `data[i] * 2.0`).
pub fn leak_wgsl(block_size: u32, body: &str) -> String {
    format!(
        "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read_write> dirty: array<u32>;

var<workgroup> sdata: array<f32, {block_size}>;

@compute @workgroup_size({block_size})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {{
    let i = id.x;
    sdata[lid] = input[i];
    workgroupBarrier();
    output[i] = {body};
    if output[i] != sdata[lid] {{
        dirty[i] = 1u;
    }}
}}
",
        block_size = block_size,
        body = body,
    )
}

/// Build a LEAK-style WGSL shader for generic element types.
/// The user provides the type, block size, and the per-element expression.
pub fn leak_wgsl_typed(
    block_size: u32,
    elem_type: &str,
    input_binding: u32,
    output_binding: u32,
    dirty_binding: u32,
    body: &str,
) -> String {
    format!(
        "\
@group(0) @binding({input_binding}) var<storage, read> input: array<{elem_type}>;
@group(0) @binding({output_binding}) var<storage, read_write> output: array<{elem_type}>;
@group(0) @binding({dirty_binding}) var<storage, read_write> dirty: array<u32>;

var<workgroup> sdata: array<{elem_type}, {block_size}>;

@compute @workgroup_size({block_size})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {{
    let i = id.x;
    sdata[lid] = input[i];
    workgroupBarrier();
    output[i] = {body};
    if output[i] != sdata[lid] {{
        dirty[i] = 1u;
    }}
}}
",
        block_size = block_size,
        elem_type = elem_type,
        input_binding = input_binding,
        output_binding = output_binding,
        dirty_binding = dirty_binding,
        body = body,
    )
}

pub struct LeakDispatch {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group: wgpu::BindGroup,
    pub workgroup_count: (u32, u32, u32),
}

impl LeakDispatch {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        block_size: u32,
        elem_type: &str,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        dirty_buffer: &wgpu::Buffer,
        body_expr: &str,
        element_count: u32,
    ) -> Self {
        let wgsl = leak_wgsl_typed(block_size, elem_type, 0, 1, 2, body_expr);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("leak_shader"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("leak_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("leak_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dirty_buffer.as_entire_binding(),
                },
            ],
        });

        let wgc = element_count.div_ceil(block_size);

        Self {
            pipeline,
            bind_group,
            workgroup_count: (wgc, 1, 1),
        }
    }

    pub fn record(&self, device: &wgpu::Device) -> wgpu::CommandBuffer {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("leak_encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("leak_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(
                self.workgroup_count.0,
                self.workgroup_count.1,
                self.workgroup_count.2,
            );
        }
        encoder.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WgpuContext;
    use wgpu::util::DeviceExt;

    #[test]
    fn leak_wgsl_generates_valid_code() {
        let wgsl = leak_wgsl(64, "input[i] * 2.0");
        assert!(wgsl.contains("workgroup_size(64)"));
        assert!(wgsl.contains("var<workgroup> sdata"));
        assert!(wgsl.contains("dirty[i] = 1u"));
    }

    #[test]
    fn leak_dispatch_compiles() {
        let ctx = WgpuContext::new_blocking();

        let input = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("input"),
                contents: bytemuck::cast_slice(&[1.0f32; 256]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let output = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("output"),
                contents: bytemuck::cast_slice(&[0.0f32; 256]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let dirty = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("dirty"),
                contents: bytemuck::cast_slice(&[0u32; 256]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let dispatch = LeakDispatch::new(
            &ctx.device,
            64,
            "f32",
            &input,
            &output,
            &dirty,
            "input[i] * 2.0",
            256,
        );

        let buf = dispatch.record(&ctx.device);
        ctx.queue.submit([buf]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .ok();
    }
}
