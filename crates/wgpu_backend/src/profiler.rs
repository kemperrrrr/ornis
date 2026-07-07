use std::path::PathBuf;
use std::time::Instant;
use serde::{Deserialize, Serialize};
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Element count threshold above which GPU is preferred.
    pub gpu_threshold: usize,
    /// Round-trip time for GPU dispatch at threshold size (ns).
    pub gpu_dispatch_ns: f64,
    /// CPU time for same workload at threshold size (ns).
    pub cpu_dispatch_ns: f64,
    /// Timestamp of last calibration.
    pub calibrated_at: String,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            gpu_threshold: 10_000,
            gpu_dispatch_ns: 0.0,
            cpu_dispatch_ns: 0.0,
            calibrated_at: String::new(),
        }
    }
}

const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];

pub struct AutoProfiler;

impl AutoProfiler {
    pub fn config_path() -> PathBuf {
        let mut p = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        p.push("ornis");
        let _ = std::fs::create_dir_all(&p);
        p.push("profiler_config.json");
        p
    }

    pub fn load_or_calibrate(device: &wgpu::Device, queue: &wgpu::Queue) -> ProfilerConfig {
        let path = Self::config_path();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(cfg) = serde_json::from_str::<ProfilerConfig>(&content) {
                    return cfg;
                }
            }
        }

        let cfg = Self::calibrate(device, queue);
        if let Ok(json) = serde_json::to_string_pretty(&cfg) {
            let _ = std::fs::write(&path, json);
        }
        cfg
    }

    pub fn calibrate(device: &wgpu::Device, queue: &wgpu::Queue) -> ProfilerConfig {
        let mut gpu_times: Vec<(usize, f64)> = Vec::new();
        let mut cpu_times: Vec<(usize, f64)> = Vec::new();

        for &size in SIZES {
            let gpu_ns = Self::bench_gpu(device, queue, size);
            let cpu_ns = Self::bench_cpu(size);
            gpu_times.push((size, gpu_ns));
            cpu_times.push((size, cpu_ns));
        }

        let threshold = Self::find_crossover(&gpu_times, &cpu_times);

        let gpu_at_th = gpu_times
            .iter()
            .find(|(s, _)| *s == threshold)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);
        let cpu_at_th = cpu_times
            .iter()
            .find(|(s, _)| *s == threshold)
            .map(|(_, t)| *t)
            .unwrap_or(0.0);

        ProfilerConfig {
            gpu_threshold: threshold,
            gpu_dispatch_ns: gpu_at_th,
            cpu_dispatch_ns: cpu_at_th,
            calibrated_at: format!("{:?}", std::time::SystemTime::now()),
        }
    }

    fn bench_gpu(device: &wgpu::Device, queue: &wgpu::Queue, count: usize) -> f64 {
        let data: Vec<f32> = vec![1.0f32; count];
        let out: Vec<f32> = vec![0.0f32; count];

        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("profiler_staging"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let output = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("profiler_output"),
            contents: bytemuck::cast_slice(&out),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("profiler_shader"),
            source: wgpu::ShaderSource::Wgsl(
                "\
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    output[i] = input[i] * 2.0 + 1.0;
}"
                .into(),
            ),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("profiler_pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("profiler_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: staging.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        // Warmup
        let mut cbe = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("profiler_warmup"),
        });
        {
            let mut cpass = cbe.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((count as u32 + 63) / 64, 1, 1);
        }
        queue.submit([cbe.finish()]);
        device.poll(wgpu::Maintain::Wait);

        // Timed run
        let start = Instant::now();
        const ITERS: u32 = 10;
        for _ in 0..ITERS {
            let mut cbe = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("profiler_timed"),
            });
            {
                let mut cpass = cbe.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                cpass.set_pipeline(&pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups((count as u32 + 63) / 64, 1, 1);
            }
            queue.submit([cbe.finish()]);
        }
        device.poll(wgpu::Maintain::Wait);

        start.elapsed().as_nanos() as f64 / ITERS as f64
    }

    fn bench_cpu(count: usize) -> f64 {
        let data: Vec<f32> = vec![1.0f32; count];
        let mut output: Vec<f32> = vec![0.0f32; count];

        let start = Instant::now();
        const ITERS: u32 = 100;
        for _ in 0..ITERS {
            for i in 0..count {
                output[i] = data[i] * 2.0 + 1.0;
            }
        }
        start.elapsed().as_nanos() as f64 / ITERS as f64
    }

    fn find_crossover(gpu: &[(usize, f64)], cpu: &[(usize, f64)]) -> usize {
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            if g.1 < c.1 {
                return g.0;
            }
        }
        SIZES[SIZES.len() - 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WgpuContext;

    #[test]
    fn profiler_calibrates() {
        let ctx = WgpuContext::new_blocking();
        let cfg = AutoProfiler::calibrate(&ctx.device, &ctx.queue);
        assert!(cfg.gpu_threshold > 0);
        assert!(cfg.gpu_dispatch_ns > 0.0 || cfg.cpu_dispatch_ns > 0.0);
    }

    #[test]
    fn profiler_find_crossover() {
        let gpu = vec![(100, 500.0), (1_000, 300.0), (10_000, 100.0)];
        let cpu = vec![(100, 50.0), (1_000, 100.0), (10_000, 500.0)];
        assert_eq!(AutoProfiler::find_crossover(&gpu, &cpu), 10_000);
    }
}
