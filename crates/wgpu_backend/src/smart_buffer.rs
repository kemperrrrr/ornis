use bitflags::bitflags;
use wgpu::util::DeviceExt;

bitflags! {
    /// Tracks which side of a [`SmartBuffer`] holds the newer copy of the
    /// data, so sync operations transfer bytes only when needed.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ResidencyFlags: u8 {
        /// The CPU copy was mutated; the GPU buffer is stale.
        const DIRTY_CPU = 0b01;
        /// The GPU copy was mutated by a dispatch; the CPU slice is stale.
        const DIRTY_GPU = 0b10;
    }
}

/// A buffer resident on both CPU and GPU with dirty-flag synchronization.
///
/// Mutating through [`cpu_data_mut`](Self::cpu_data_mut) or a GPU dispatch
/// (flagged via [`mark_gpu_dirty`](Self::mark_gpu_dirty)) marks the opposite
/// side stale; [`sync_to_gpu`](Self::sync_to_gpu) and
/// [`sync_to_cpu_blocking`](Self::sync_to_cpu_blocking) then move the bytes
/// on demand instead of eagerly.
pub struct SmartBuffer<T: bytemuck::Pod> {
    cpu_data: Vec<T>,
    gpu_buffer: Option<wgpu::Buffer>,
    flags: ResidencyFlags,
    _size: usize,
    label: String,
}

impl<T: bytemuck::Pod> SmartBuffer<T> {
    /// Creates a buffer with `cpu_data` as the initial contents, uploaded to
    /// a fresh GPU buffer with the given usage flags. Both sides start clean.
    pub fn new(
        cpu_data: Vec<T>,
        device: &wgpu::Device,
        usage: wgpu::BufferUsages,
        label: &str,
    ) -> Self {
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

    /// The GPU-side buffer, or `None` if it was dropped and not yet
    /// recreated via [`ensure_gpu_buffer`](Self::ensure_gpu_buffer).
    pub fn gpu_buffer(&self) -> Option<&wgpu::Buffer> {
        self.gpu_buffer.as_ref()
    }

    /// Read-only view of the CPU copy. Does not raise any dirty flag.
    pub fn cpu_data(&self) -> &[T] {
        &self.cpu_data
    }

    /// Mutable view of the CPU copy; marks the GPU side stale
    /// ([`ResidencyFlags::DIRTY_CPU`]).
    pub fn cpu_data_mut(&mut self) -> &mut [T] {
        self.flags.insert(ResidencyFlags::DIRTY_CPU);
        &mut self.cpu_data
    }

    /// Records that a GPU dispatch mutated the buffer, marking the CPU side
    /// stale ([`ResidencyFlags::DIRTY_GPU`]).
    pub fn mark_gpu_dirty(&mut self) {
        self.flags.insert(ResidencyFlags::DIRTY_GPU);
    }

    /// Current dirty flags for both residency sides.
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

    /// Download GPU data to CPU if DIRTY_GPU is set (blocking).
    /// Requires COPY_SRC usage on the buffer; the read goes through a
    /// MAP_READ staging buffer (a storage buffer cannot be mapped).
    pub fn sync_to_cpu_blocking(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if !self.flags.contains(ResidencyFlags::DIRTY_GPU) {
            return;
        }

        if let Some(ref buffer) = self.gpu_buffer {
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("smart_buffer staging"),
                size: self._size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("smart_buffer sync_to_cpu"),
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, self._size as u64);
            queue.submit([encoder.finish()]);

            let buffer_slice = staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .ok();
            if let Ok(Ok(())) = receiver.recv() {
                let view = buffer_slice.get_mapped_range();
                let downloaded: &[T] = bytemuck::cast_slice(&view);
                self.cpu_data.copy_from_slice(downloaded);
                drop(view);
                staging.unmap();
            }
        }

        self.flags.remove(ResidencyFlags::DIRTY_GPU);
    }

    /// Ensure GPU buffer exists (recreate if dropped).
    pub fn ensure_gpu_buffer(&mut self, device: &wgpu::Device, usage: wgpu::BufferUsages) {
        if self.gpu_buffer.is_none() {
            let raw: &[u8] = bytemuck::cast_slice(&self.cpu_data);
            self.gpu_buffer = Some(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&self.label),
                    contents: raw,
                    usage,
                }),
            );
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
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            "test",
        );

        assert_eq!(buf.flags(), ResidencyFlags::empty());

        buf.cpu_data_mut()[0] = 42.0;
        assert!(buf.flags().contains(ResidencyFlags::DIRTY_CPU));

        buf.sync_to_gpu(&ctx.queue);
        assert!(!buf.flags().contains(ResidencyFlags::DIRTY_CPU));
    }

    fn test_buffer<T: bytemuck::Pod>(data: Vec<T>) -> (WgpuContext, SmartBuffer<T>) {
        let ctx = WgpuContext::new_blocking();
        let buf = SmartBuffer::new(
            data,
            &ctx.device,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            "test",
        );
        (ctx, buf)
    }

    #[test]
    fn sync_to_gpu_is_noop_when_clean() {
        let (ctx, mut buf) = test_buffer(vec![1.0f32, 2.0]);
        assert_eq!(buf.flags(), ResidencyFlags::empty());
        buf.sync_to_gpu(&ctx.queue);
        assert_eq!(buf.flags(), ResidencyFlags::empty());
        assert_eq!(buf.cpu_data(), &[1.0, 2.0]);
    }

    #[test]
    fn sync_to_gpu_clears_only_cpu_flag() {
        let (ctx, mut buf) = test_buffer(vec![1.0f32]);
        buf.cpu_data_mut()[0] = 5.0;
        buf.mark_gpu_dirty();
        assert_eq!(buf.flags(), ResidencyFlags::all());

        buf.sync_to_gpu(&ctx.queue);
        assert_eq!(buf.flags(), ResidencyFlags::DIRTY_GPU);
    }

    #[test]
    fn sync_to_cpu_is_noop_when_clean() {
        let (ctx, mut buf) = test_buffer(vec![1.0f32, 2.0, 3.0]);
        buf.sync_to_cpu_blocking(&ctx.device, &ctx.queue);
        assert_eq!(buf.flags(), ResidencyFlags::empty());
        assert_eq!(buf.cpu_data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn gpu_to_cpu_roundtrip_downloads_uploaded_data() {
        let (ctx, mut buf) = test_buffer(vec![1u32, 2, 3, 4]);

        // Upload new data, then read it back through the GPU side.
        buf.cpu_data_mut().copy_from_slice(&[10, 20, 30, 40]);
        buf.sync_to_gpu(&ctx.queue);
        buf.mark_gpu_dirty();
        buf.sync_to_cpu_blocking(&ctx.device, &ctx.queue);

        assert_eq!(buf.cpu_data(), &[10, 20, 30, 40]);
        assert_eq!(buf.flags(), ResidencyFlags::empty());
    }

    #[test]
    fn gpu_to_cpu_roundtrip_preserves_initial_data() {
        let (ctx, mut buf) = test_buffer(vec![7.5f32, -1.25, 0.0]);
        // The initial contents were uploaded at construction; downloading
        // them must reproduce the same bytes.
        buf.mark_gpu_dirty();
        buf.sync_to_cpu_blocking(&ctx.device, &ctx.queue);
        assert_eq!(buf.cpu_data(), &[7.5, -1.25, 0.0]);
        assert_eq!(buf.flags(), ResidencyFlags::empty());
    }

    #[test]
    fn ensure_gpu_buffer_keeps_existing_buffer() {
        let (ctx, mut buf) = test_buffer(vec![1.0f32, 2.0]);
        assert!(buf.gpu_buffer().is_some());
        let before = buf.gpu_buffer().unwrap() as *const _;
        buf.ensure_gpu_buffer(&ctx.device, wgpu::BufferUsages::STORAGE);
        // Not recreated: same allocation, data untouched, no flags raised.
        let after = buf.gpu_buffer().unwrap() as *const _;
        assert_eq!(before, after);
        assert_eq!(buf.cpu_data(), &[1.0, 2.0]);
        assert_eq!(buf.flags(), ResidencyFlags::empty());
    }

    #[test]
    fn single_element_buffer_syncs() {
        let (ctx, mut buf) = test_buffer(vec![42u64]);
        buf.cpu_data_mut()[0] = 43;
        buf.sync_to_gpu(&ctx.queue);
        buf.mark_gpu_dirty();
        buf.sync_to_cpu_blocking(&ctx.device, &ctx.queue);
        assert_eq!(buf.cpu_data(), &[43]);
    }
}
