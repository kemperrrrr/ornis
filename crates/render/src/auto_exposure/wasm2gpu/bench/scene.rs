//! Deterministic HDR scenes and texture uploads shared by the A/B benchmark.

use bytemuck::cast_slice;

pub(super) fn bench_texels(width: u32, height: u32) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let levels = [0.001, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0, 32.0, 128.0];
    let count = (width * height) as usize;
    let mut deferred = Vec::with_capacity(count);
    let mut forward = Vec::with_capacity(count);
    for index in 0..count {
        let level = levels[index % levels.len()];
        let forward_level = levels[(index * 5 + 3) % levels.len()];
        deferred.push([level, level, level, 1.0]);
        forward.push([forward_level, forward_level * 0.75, forward_level * 0.5, 0.35]);
    }
    (deferred, forward)
}

pub(super) fn texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    width: u32,
    height: u32,
    texels: &[[f32; 4]],
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
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
        cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 16),
            rows_per_image: Some(height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
