//! Shared setup for the translated-shader pixel-parity gates.
//!
//! Both gates render the frozen DSL shader and the kernel-crate shader into one
//! target with identical inputs, so the setup lives here once: one image
//! generator, one uploader. The gates differ only in which textures, uniforms
//! and modes they feed.

#![allow(dead_code)]

/// One `width`×`height` RGBA8 image from a per-texel function.
pub fn gradient(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&fill(x, y));
        }
    }
    pixels
}

/// Upload one RGBA8 image and return its sampled view.
pub fn upload_rgba8(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: (u32, u32),
    pixels: &[u8],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        pixels,
        wgpu::TexelCopyBufferLayout {
            bytes_per_row: Some(size.0 * 4),
            ..Default::default()
        },
        wgpu::Extent3d {
            width: size.0,
            height: size.1,
            ..Default::default()
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Count the pixels of two read-backs that differ by more than one 8-bit step.
pub fn mismatched_pixels(reference: &[u8], candidate: &[u8]) -> usize {
    assert_eq!(reference.len(), candidate.len(), "read-back size");
    reference
        .as_chunks::<4>()
        .0
        .iter()
        .zip(candidate.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(&a, &b)| a.abs_diff(b) > 1))
        .count()
}
