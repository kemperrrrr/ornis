use bitflags::bitflags;
use wgpu::util::DeviceExt;

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ResidencyFlags: u8 {
        const DIRTY_CPU = 0b01;
        const DIRTY_GPU = 0b10;
    }
}

pub struct SmartBuffer<T: bytemuck::Pod> {
    cpu_data: Vec<T>,
    gpu_buffer: Option<wgpu::Buffer>,
    flags: ResidencyFlags,
    _size: usize,
    label: String,
}

impl<T: bytemuck::Pod> SmartBuffer<T> {
    pub fn new(cpu_data: Vec<T>, device: &wgpu::Device, usage: wgpu::BufferUsages, label: &str) -> Self {
        let size = cpu_data.len() * std::mem::size_of::<T>();
        let raw: &[u8] = bytemuck::cast_slice(&cpu_data);
        let gpu_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: raw,
            usage,
        });

        Self {
            cpu_data,
            gpu_buffer: Some(gpu_buffer),
            flags: ResidencyFlags::empty(),
            _size: size,
            label: label.to_string(),
        }
    }

    pub fn gpu_buffer(&self) -> Option<&wgpu::Buffer> {
        self.gpu_buffer.as_ref()
    }

    pub fn cpu_data(&self) -> &[T] {
        &self.cpu_data
    }

    pub fn cpu_data_mut(&mut self) -> &mut [T] {
        self.flags.insert(ResidencyFlags::DIRTY_CPU);
        &mut self.cpu_data
    }

    pub fn mark_gpu_dirty(&mut self) {
        self.flags.insert(ResidencyFlags::DIRTY_GPU);
    }

    pub fn flags(&self) -> ResidencyFlags {
        self.flags
    }

    /// Upload CPU data to GPU if DIRTY_CPU is set.
    pub fn sync_to_gpu(&mut self, queue: &wgpu::Queue) {
        if !self.flags.contains(ResidencyFlags::DIRTY_CPU) {
            return;
        }

        if let Some(ref buffer) = self.gpu_buffer {
            let raw: &[u8] = bytemuck::cast_slice(&self.cpu_data);
            queue.write_buffer(buffer, 0, raw);
        }

        self.flags.remove(ResidencyFlags::DIRTY_CPU);
    }

    /// Download GPU data to CPU if DIRTY_GPU is set (blocking, uses pollster).
    /// Requires COPY_DST usage on the buffer.
    pub fn sync_to_cpu_blocking(&mut self, device: &wgpu::Device) {
        if !self.flags.contains(ResidencyFlags::DIRTY_GPU) {
            return;
        }

        if let Some(ref buffer) = self.gpu_buffer {
            let buffer_slice = buffer.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device.poll(wgpu::Maintain::Wait);
            if let Ok(Ok(())) = receiver.recv() {
                let view = buffer_slice.get_mapped_range();
                let downloaded: &[T] = bytemuck::cast_slice(&view);
                self.cpu_data.copy_from_slice(downloaded);
                drop(view);
                buffer.unmap();
            }
        }

        self.flags.remove(ResidencyFlags::DIRTY_GPU);
    }

    /// Ensure GPU buffer exists (recreate if dropped).
    pub fn ensure_gpu_buffer(&mut self, device: &wgpu::Device, usage: wgpu::BufferUsages) {
        if self.gpu_buffer.is_none() {
            let raw: &[u8] = bytemuck::cast_slice(&self.cpu_data);
            self.gpu_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&self.label),
                contents: raw,
                usage,
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::WgpuContext;

    #[test]
    fn smart_buffer_tracks_dirty() {
        let ctx = WgpuContext::new_blocking();

        let data = vec![1.0f32, 2.0, 3.0];
        let mut buf = SmartBuffer::new(
            data,
            &ctx.device,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "test",
        );

        assert_eq!(buf.flags(), ResidencyFlags::empty());

        buf.cpu_data_mut()[0] = 42.0;
        assert!(buf.flags().contains(ResidencyFlags::DIRTY_CPU));

        buf.sync_to_gpu(&ctx.queue);
        assert!(!buf.flags().contains(ResidencyFlags::DIRTY_CPU));
    }
}
