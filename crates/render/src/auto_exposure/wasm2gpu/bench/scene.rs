//! Deterministic HDR scenes and texture uploads shared by the A/B benchmark.

use bytemuck::cast_slice;

const BENCH_LUMINANCE_LEVELS: [f32; 9] = [0.001, 0.01, 0.05, 0.18, 0.5, 1.0, 4.0, 32.0, 128.0];
const FORWARD_LEVEL_INDEX_SCALE: usize = 5;
const FORWARD_LEVEL_INDEX_OFFSET: usize = 3;
const FORWARD_GREEN_SCALE: f32 = 0.75;
const FORWARD_BLUE_SCALE: f32 = 0.5;
const FORWARD_ALPHA: f32 = 0.35;
const RGBA32_BYTES_PER_PIXEL: u32 = 16;

pub(super) fn bench_texels(width: u32, height: u32) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let count = (width * height) as usize;
    let mut deferred = Vec::with_capacity(count);
    let mut forward = Vec::with_capacity(count);
    for index in 0..count {
        let level = BENCH_LUMINANCE_LEVELS[index % BENCH_LUMINANCE_LEVELS.len()];
        let forward_index =
            (index * FORWARD_LEVEL_INDEX_SCALE + FORWARD_LEVEL_INDEX_OFFSET)
                % BENCH_LUMINANCE_LEVELS.len();
        let forward_level = BENCH_LUMINANCE_LEVELS[forward_index];
        deferred.push([level, level, level, 1.0]);
        forward.push([
            forward_level,
            forward_level * FORWARD_GREEN_SCALE,
            forward_level * FORWARD_BLUE_SCALE,
            FORWARD_ALPHA,
        ]);
    }
    (deferred, forward)
}

pub(super) struct TextureUpload<'a> {
    pub(super) label: &'a str,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) texels: &'a [[f32; 4]],
}

pub(super) fn texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    upload: TextureUpload<'_>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let size = wgpu::Extent3d {
        width: upload.width,
        height: upload.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(upload.label),
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
        cast_slice(upload.texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(upload.width * RGBA32_BYTES_PER_PIXEL),
            rows_per_image: Some(upload.height),
        },
        size,
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
