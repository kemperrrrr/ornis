use ornis_core::ComponentStore;
use wgpu::util::DeviceExt;

/// Creates a GPU buffer from a ComponentStore's dense data array.
pub fn create_buffer_from_store<T: bytemuck::Pod>(
    store: &ComponentStore<T>,
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    label: Option<&str>,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label,
        contents: bytemuck::cast_slice(store.data.as_slice()),
        usage,
    })
}

/// Creates a GPU buffer from any Pod slice.
pub fn create_buffer_from_slice<T: bytemuck::Pod>(
    data: &[T],
    device: &wgpu::Device,
    usage: wgpu::BufferUsages,
    label: Option<&str>,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label,
        contents: bytemuck::cast_slice(data),
        usage,
    })
}
