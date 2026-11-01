//! Buffer binding layouts and storage/uniform allocation helpers.

use ornis_wasm2gpu_testkit::gpu_buffer;

pub(super) fn binding(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

pub(super) fn bind_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn storage_buffer(device: &wgpu::Device, label: &str, elements: usize) -> wgpu::Buffer {
    gpu_buffer(
        device,
        label,
        (elements * std::mem::size_of::<u32>()) as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

pub(super) fn uniform_buffer(device: &wgpu::Device, label: &str, elements: usize) -> wgpu::Buffer {
    gpu_buffer(
        device,
        label,
        (elements * 16) as u64,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    )
}
