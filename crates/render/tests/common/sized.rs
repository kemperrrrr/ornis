//! Sized headless-render helpers shared by the pixel gates that outgrow
//! the square `SIZE` harness in `super` (the 320×180 MSAA/ortho gates):
//! one render-into-fresh-target-and-read-back call, so the gate binaries
//! keep zero duplicated harness code (the rustqual ratchet flags exact
//! `DUPLICATE` pairs otherwise).

/// Offscreen frame spec for [`render_and_readback`].
pub struct FrameSpec {
    /// Frame extent in pixels.
    pub size: (u32, u32),
    /// Read-back format (matches the composite output).
    pub format: wgpu::TextureFormat,
}

impl FrameSpec {
    /// Swap-chain-less surface configuration for [`Renderer3D`] setup at
    /// this spec (same usage/present parameters as the square harness).
    pub fn surface_config(&self) -> wgpu::SurfaceConfiguration {
        wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.format,
            width: self.size.0,
            height: self.size.1,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        }
    }
}

/// Renders one frame into a fresh `COPY_SRC` target through `record` and
/// reads it back (blocking), stripping the 256-byte row padding. The gates
/// differ only in the recording closure (legacy `render_scene` vs the plan
/// path); allocation, submit and readback stay in this one place.
pub fn render_and_readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    spec: &FrameSpec,
    record: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
) -> Vec<u8> {
    /// Bytes per read-back pixel.
    const BPP: u32 = 4;
    /// `copy_texture_to_buffer` row alignment.
    const ROW_ALIGN: u32 = 256;
    let (w, h) = spec.size;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pixel gate target"),
        size: wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: spec.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel gate encoder"),
    });
    record(&mut encoder, &target_view);
    queue.submit([encoder.finish()]);
    let unpadded = w * BPP;
    let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel gate readback"),
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel gate readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll readback");
    let data = slice.get_mapped_range().unwrap();
    let mut pixels = vec![0u8; (unpadded * h) as usize];
    for y in 0..h as usize {
        pixels[y * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
    }
    drop(data);
    readback.unmap();
    pixels
}
