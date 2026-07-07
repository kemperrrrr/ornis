use crate::dispatcher::{choose_platform, DispatchConfig, Platform};

type CpuCommand = Box<dyn FnOnce() + Send>;

enum RecordedCommand {
    Gpu(wgpu::CommandBuffer),
    Cpu(CpuCommand),
}

/// Records GPU compute commands and CPU closures, then flushes them together.
/// CPU sends commands to where data already lives — no eager PCIe transfers.
pub struct CommandSync {
    device: std::sync::Arc<wgpu::Device>,
    queue: std::sync::Arc<wgpu::Queue>,
    commands: Vec<RecordedCommand>,
}

impl CommandSync {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        let device = std::sync::Arc::new(device);
        let queue = std::sync::Arc::new(queue);
        Self {
            device,
            queue,
            commands: Vec::new(),
        }
    }

    /// Record a GPU compute dispatch.
    pub fn dispatch_gpu(
        &mut self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroup_count: (u32, u32, u32),
        label: &str,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bind_group, &[]);
            cpass.dispatch_workgroups(workgroup_count.0, workgroup_count.1, workgroup_count.2);
        }
        self.commands
            .push(RecordedCommand::Gpu(encoder.finish()));
    }

    /// Record a CPU-side closure for parallel execution.
    pub fn dispatch_cpu<F: FnOnce() + Send + 'static>(&mut self, f: F) {
        self.commands.push(RecordedCommand::Cpu(Box::new(f)));
    }

    /// Dispatch based on element count: GPU above threshold, CPU otherwise.
    pub fn dispatch_auto(
        &mut self,
        config: &DispatchConfig,
        element_count: usize,
        pipeline: Option<&wgpu::ComputePipeline>,
        bind_group: Option<&wgpu::BindGroup>,
        cpu_fn: CpuCommand,
    ) {
        match choose_platform(config, element_count) {
            Platform::Gpu => {
                if let (Some(pipeline), Some(bg)) = (pipeline, bind_group) {
                    let wgc =
                        (element_count as u32).div_ceil(config.workgroup_size);
                    self.dispatch_gpu(pipeline, bg, (wgc, 1, 1), &config.label);
                } else {
                    self.commands.push(RecordedCommand::Cpu(cpu_fn));
                }
            }
            Platform::Cpu => {
                self.commands.push(RecordedCommand::Cpu(cpu_fn));
            }
        }
    }

    /// Submit queued GPU command buffers and execute CPU closures.
    pub fn flush(&mut self) {
        let mut gpu_buffers: Vec<wgpu::CommandBuffer> = Vec::new();

        for cmd in self.commands.drain(..) {
            match cmd {
                RecordedCommand::Gpu(buf) => gpu_buffers.push(buf),
                RecordedCommand::Cpu(f) => f(),
            }
        }

        if !gpu_buffers.is_empty() {
            self.queue.submit(gpu_buffers);
            self.device.poll(wgpu::Maintain::Wait);
        }
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WgpuContext;
    use wgpu::util::DeviceExt;

    #[test]
    fn gpu_dispatch_records_and_flushes() {
        let ctx = WgpuContext::new_blocking();

        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("test"),
                source: wgpu::ShaderSource::Wgsl(
                    "@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = data[id.x] * 2.0;
}"
                    .into(),
                ),
            });

        let pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("test_pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_buf"),
            contents: bytemuck::cast_slice(&[1.0f32, 2.0, 3.0]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_bg"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });

        let mut sync = CommandSync::new(ctx.device.clone(), ctx.queue.clone());
        sync.dispatch_gpu(&pipeline, &bind_group, (1, 1, 1), "test_dispatch");
        assert_eq!(sync.len(), 1);
        sync.flush();
        assert!(sync.is_empty());
    }
}
