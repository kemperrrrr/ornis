use std::any::TypeId;
use std::collections::HashMap;
use std::path::PathBuf;

struct CachedPipeline {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    label: String,
}

pub struct PsoCache {
    device: wgpu::Device,
    pipelines: HashMap<(TypeId, TypeId), CachedPipeline>,
    disk_path: PathBuf,
}

impl PsoCache {
    pub fn new(device: wgpu::Device) -> Self {
        let mut path = dirs::cache_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push("ornis");
        path.push("pso_cache");
        let _ = std::fs::create_dir_all(&path);

        Self {
            device,
            pipelines: HashMap::new(),
            disk_path: path,
        }
    }

    fn shader_path(&self, kernel_id: TypeId, component_id: TypeId) -> PathBuf {
        let key = format!("{:?}_{:?}.wgsl", kernel_id, component_id)
            .replace('"', "")
            .replace(' ', "_");
        self.disk_path.join(&key)
    }

    /// Get or create a compute pipeline from WGSL source.
    #[allow(clippy::map_entry)]
    pub fn get_or_create(
        &mut self,
        kernel_id: TypeId,
        component_id: TypeId,
        wgsl_source: &str,
        label: &str,
    ) -> &wgpu::ComputePipeline {
        let key = (kernel_id, component_id);

        if !self.pipelines.contains_key(&key) {
            let path = self.shader_path(kernel_id, component_id);
            let pipeline = self.compile_pipeline(wgsl_source, label);
            let _ = std::fs::write(path, wgsl_source);
            self.pipelines.insert(
                key,
                CachedPipeline {
                    pipeline,
                    label: label.to_string(),
                },
            );
        }

        &self.pipelines[&key].pipeline
    }

    fn compile_pipeline(&self, wgsl_source: &str, label: &str) -> wgpu::ComputePipeline {
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });

        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// Number of cached pipelines.
    pub fn len(&self) -> usize {
        self.pipelines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pipelines.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WgpuContext;

    #[test]
    fn cache_creates_and_reuses() {
        let ctx = WgpuContext::new_blocking();
        let mut cache = PsoCache::new(ctx.device);

        let kernel_id = TypeId::of::<fn(f32) -> f32>();
        let component_id = TypeId::of::<f32>();

        let wgsl = "\
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    output[id.x] = a[id.x] * 2.0;
}";

        let _p1 = cache.get_or_create(kernel_id, component_id, wgsl, "test");
        assert_eq!(cache.len(), 1);

        // Second call should reuse
        let _p2 = cache.get_or_create(kernel_id, component_id, wgsl, "test");
        assert_eq!(cache.len(), 1);
    }
}
