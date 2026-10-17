//! Device capability selection and block-aware uploads for material mip chains.

use ornis_texture::{TextureEncoding, TextureImage, TextureLevels};

use crate::flags::AnisotropyLevel;
use crate::textures::{
    GpuTexture, TextureRole, TextureUploadError, sampler_descriptor_for_role,
    texture_format_for_role,
};

/// Optional compression features to enable when creating a render device.
/// Only supported adapter bits are requested; upload checks the enabled device bits.
pub fn texture_compression_features(available: wgpu::Features) -> wgpu::Features {
    available
        & (wgpu::Features::TEXTURE_COMPRESSION_BC
            | wgpu::Features::TEXTURE_COMPRESSION_ASTC
            | wgpu::Features::TEXTURE_COMPRESSION_ETC2)
}

/// Preferred Basis target: BC7 on desktop, ASTC 4×4, ETC2 RGBA8, then RGBA8.
pub fn preferred_texture_encoding(enabled: wgpu::Features) -> TextureEncoding {
    if enabled.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        TextureEncoding::Bc7
    } else if enabled.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        TextureEncoding::Astc4x4
    } else if enabled.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        TextureEncoding::Etc2
    } else {
        TextureEncoding::Rgba8
    }
}

fn supports(encoding: TextureEncoding, enabled: wgpu::Features) -> bool {
    match encoding {
        TextureEncoding::Rgba8 => true,
        TextureEncoding::Bc1 | TextureEncoding::Bc3 | TextureEncoding::Bc7 => {
            enabled.contains(wgpu::Features::TEXTURE_COMPRESSION_BC)
        }
        TextureEncoding::Astc4x4 => enabled.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
        TextureEncoding::Etc2 => enabled.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
    }
}

/// GPU format with role-specific color space, including compressed color/data pairs.
pub fn gpu_texture_format(encoding: TextureEncoding, role: TextureRole) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as F;
    let srgb = matches!(role, TextureRole::BaseColor | TextureRole::Emissive);
    match (encoding, srgb) {
        (TextureEncoding::Rgba8, _) => texture_format_for_role(role),
        (TextureEncoding::Bc1, false) => F::Bc1RgbaUnorm,
        (TextureEncoding::Bc1, true) => F::Bc1RgbaUnormSrgb,
        (TextureEncoding::Bc3, false) => F::Bc3RgbaUnorm,
        (TextureEncoding::Bc3, true) => F::Bc3RgbaUnormSrgb,
        (TextureEncoding::Bc7, false) => F::Bc7RgbaUnorm,
        (TextureEncoding::Bc7, true) => F::Bc7RgbaUnormSrgb,
        (TextureEncoding::Etc2, false) => F::Etc2Rgba8Unorm,
        (TextureEncoding::Etc2, true) => F::Etc2Rgba8UnormSrgb,
        (TextureEncoding::Astc4x4, srgb) => F::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: if srgb {
                wgpu::AstcChannel::UnormSrgb
            } else {
                wgpu::AstcChannel::Unorm
            },
        },
    }
}

/// Uploads imported RGBA8 or KTX2 directly, selecting compression from enabled features.
/// Native KTX2 blocks are retained when supported; otherwise all authored levels
/// decode to RGBA8. Non-block-aligned base extents also use RGBA8, as WebGPU
/// requires compressed base dimensions to be multiples of their block extent.
/// The stored sampler is [`AnisotropyLevel::DEFAULT`]. A frame that needs
/// another level builds its own sampler; this one is not configurable here.
///
/// # Errors
/// Rejects invalid images, failed transcodes and dimensions exceeding device limits.
pub fn upload_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &TextureImage,
    role: TextureRole,
) -> Result<GpuTexture, TextureUploadError> {
    let (width, height) = image.dimensions();
    validate_device_size(device, width, height)?;
    let mut target = preferred_texture_encoding(device.features());
    if let TextureImage::Ktx2(ktx) = image
        && let Some(native) = ktx.encoding().filter(|&e| supports(e, device.features()))
    {
        target = native;
    }
    let block = TextureEncoding::Bc7.block_extent();
    if !width.is_multiple_of(block) || !height.is_multiple_of(block) {
        target = TextureEncoding::Rgba8;
    }
    let levels = image.prepare(role, target)?;
    upload_levels(device, queue, &levels, role)
}

fn validate_device_size(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> Result<(), TextureUploadError> {
    let limit = device.limits().max_texture_dimension_2d;
    if width > limit || height > limit {
        return Err(TextureUploadError::ImageTooLarge { width, height });
    }
    Ok(())
}

pub(crate) fn upload_levels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    levels: &TextureLevels,
    role: TextureRole,
) -> Result<GpuTexture, TextureUploadError> {
    let (width, height) = levels.dimensions();
    validate_device_size(device, width, height)?;
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("material mip chain"),
        size,
        mip_level_count: levels.levels().len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu_texture_format(levels.encoding(), role),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let block = levels.encoding().block_extent();
    for (mip, bytes) in levels.levels().iter().enumerate() {
        // Copy extent is the physical block-rounded mip size, even for the
        // 2×2 and 1×1 tail. Sampling still sees the logical mip dimensions.
        let w = (width >> mip).max(1).next_multiple_of(block);
        let h = (height >> mip).max(1).next_multiple_of(block);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                // Queue::write_texture has no 256-byte row alignment requirement.
                bytes_per_row: Some(w / block * levels.encoding().block_bytes()),
                rows_per_image: Some(h / block),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                ..size
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler =
        device.create_sampler(&sampler_descriptor_for_role(role, AnisotropyLevel::DEFAULT));
    Ok(GpuTexture {
        texture,
        view,
        sampler,
        role,
        width,
        height,
        payload_bytes: levels.byte_len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preference_and_color_space_pairs() {
        let cases = [
            (wgpu::Features::empty(), TextureEncoding::Rgba8),
            (
                wgpu::Features::TEXTURE_COMPRESSION_ETC2,
                TextureEncoding::Etc2,
            ),
            (
                wgpu::Features::TEXTURE_COMPRESSION_ASTC | wgpu::Features::TEXTURE_COMPRESSION_ETC2,
                TextureEncoding::Astc4x4,
            ),
            (
                wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TEXTURE_COMPRESSION_ASTC,
                TextureEncoding::Bc7,
            ),
        ];
        for (features, expected) in cases {
            assert_eq!(preferred_texture_encoding(features), expected);
            assert_eq!(texture_compression_features(features), features);
            assert!(gpu_texture_format(expected, TextureRole::BaseColor).is_srgb());
            assert!(!gpu_texture_format(expected, TextureRole::Normal).is_srgb());
        }
    }
}
