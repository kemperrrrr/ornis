//! Shared CSM probe pieces: one yaw, one pixel compare, and one color readback.
//!
//! The GPU gates and the CPU snap tests used to each carry a copy. rustqual
//! counted those copies as duplicate groups and as matching statement windows.

use super::*;

/// Channel sum above which two pixels count as different.
const PIXEL_DIFF: i16 = 12;

/// Offscreen RGBA target plus the surface config a renderer needs to match it.
pub(super) fn probe_color_target(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::SurfaceConfiguration) {
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let surface = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    (target, view, surface)
}

/// Copy an RGBA frame out of `texture`, dropping the 256-byte row padding.
pub(super) fn read_color_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    size: (u32, u32),
) -> Vec<u8> {
    let (width, height) = size;
    {
        let unpadded = width * 4;
        let padded = unpadded.div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("csm probe readback"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().unwrap();
        let mut pixels = vec![0u8; (unpadded * height) as usize];
        for y in 0..height as usize {
            pixels[y * unpadded as usize..][..unpadded as usize]
                .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
        }
        drop(data);
        readback.unmap();
        pixels
    }
}

/// Pixels whose RGB channel sum differs by more than [`PIXEL_DIFF`].
pub(super) fn pixel_diff(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| {
            (x[0] as i16 - y[0] as i16).abs()
                + (x[1] as i16 - y[1] as i16).abs()
                + (x[2] as i16 - y[2] as i16).abs()
                > PIXEL_DIFF
        })
        .count()
}

/// Rotate `eye` around `target` by `yaw` radians about world up.
pub(crate) fn yawed_eye(eye: glam::Vec3, target: glam::Vec3, yaw: f32) -> glam::Vec3 {
    let offset = eye - target;
    let (s, c) = yaw.sin_cos();
    target
        + glam::Vec3::new(
            offset.x * c + offset.z * s,
            offset.y,
            -offset.x * s + offset.z * c,
        )
}
