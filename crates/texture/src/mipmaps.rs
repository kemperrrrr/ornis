//! Area-filtered RGBA8 mip chains: linear-light colors, linear data, unit normals.

use crate::{CpuImage, TextureRole, TextureUploadError};

/// Builds every level down to 1×1, preserving the original base level.
///
/// sRGB RGB is decoded before averaging and encoded afterwards; alpha is
/// always linear. Normal RGB is decoded to [-1, 1] and renormalized (a
/// cancelling vector becomes +Z). Exact area weights include odd edge texels.
/// Authored KTX2 levels bypass this function entirely.
///
/// # Errors
/// Returns an error if the base image fails validation.
pub fn generate_mipmaps(
    image: &CpuImage,
    role: TextureRole,
) -> Result<Vec<CpuImage>, TextureUploadError> {
    image.validate()?;
    let mut levels = vec![image.clone()];
    while let Some(previous) = levels.last() {
        if previous.width == 1 && previous.height == 1 {
            break;
        }
        levels.push(downsample(previous, role));
    }
    Ok(levels)
}

fn downsample(src: &CpuImage, role: TextureRole) -> CpuImage {
    let width = (src.width / 2).max(1);
    let height = (src.height / 2).max(1);
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let average = area_average(src, x, y, width, height, role);
            pixels.extend_from_slice(&encode(average, role));
        }
    }
    CpuImage {
        width,
        height,
        pixels,
    }
}

fn area_average(
    src: &CpuImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    role: TextureRole,
) -> [f32; 4] {
    let sx = src.width as f32 / width as f32;
    let sy = src.height as f32 / height as f32;
    let (left, right) = (x as f32 * sx, (x + 1) as f32 * sx);
    let (top, bottom) = (y as f32 * sy, (y + 1) as f32 * sy);
    let mut sum = [0.0; 4];
    for iy in top.floor() as u32..(bottom.ceil() as u32).min(src.height) {
        for ix in left.floor() as u32..(right.ceil() as u32).min(src.width) {
            let weight = (right.min((ix + 1) as f32) - left.max(ix as f32))
                * (bottom.min((iy + 1) as f32) - top.max(iy as f32));
            let offset = ((iy * src.width + ix) * 4) as usize;
            for (c, value) in sum.iter_mut().enumerate() {
                *value += decode(src.pixels[offset + c], c, role) * weight;
            }
        }
    }
    sum.map(|v| v / (sx * sy))
}

fn decode(byte: u8, channel: usize, role: TextureRole) -> f32 {
    let value = byte as f32 / 255.0;
    if channel == 3 {
        return value;
    }
    match role {
        TextureRole::BaseColor | TextureRole::Emissive => {
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        }
        TextureRole::Normal => value * 2.0 - 1.0,
        TextureRole::MetallicRoughness => value,
    }
}

fn encode(mut value: [f32; 4], role: TextureRole) -> [u8; 4] {
    match role {
        TextureRole::BaseColor | TextureRole::Emissive => {
            for c in &mut value[..3] {
                *c = if *c <= 0.0031308 {
                    *c * 12.92
                } else {
                    1.055 * c.powf(1.0 / 2.4) - 0.055
                };
            }
        }
        TextureRole::Normal => {
            let length = value[..3].iter().map(|c| c * c).sum::<f32>().sqrt();
            let normal = if length > 1e-6 {
                [value[0] / length, value[1] / length, value[2] / length]
            } else {
                [0.0, 0.0, 1.0]
            };
            for (c, n) in value[..3].iter_mut().zip(normal) {
                *c = n * 0.5 + 0.5;
            }
        }
        TextureRole::MetallicRoughness => {}
    }
    value.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_chain_including_non_square_and_single_axis() {
        for (w, h) in [(1, 1), (1, 17), (32, 1), (7, 13), (16, 16)] {
            let image = CpuImage::from_rgba8(w, h, vec![127; (w * h * 4) as usize]).unwrap();
            let levels = generate_mipmaps(&image, TextureRole::MetallicRoughness).unwrap();
            assert_eq!(levels.len() as u32, w.max(h).ilog2() + 1);
            assert_eq!(levels[0], image);
            for (i, level) in levels.iter().enumerate() {
                assert_eq!(
                    (level.width, level.height),
                    ((w >> i).max(1), (h >> i).max(1))
                );
                assert!(level.pixels.iter().all(|&v| v == 127));
                level.validate().unwrap();
            }
        }
    }

    #[test]
    fn srgb_is_filtered_in_linear_light_alpha_is_not() {
        let image = CpuImage::from_rgba8(2, 1, vec![0, 0, 0, 0, 255, 255, 255, 255]).unwrap();
        for role in [TextureRole::BaseColor, TextureRole::Emissive] {
            let levels = generate_mipmaps(&image, role).unwrap();
            assert_eq!(levels[1].pixels, [188, 188, 188, 128]);
        }
        assert_eq!(
            generate_mipmaps(&image, TextureRole::MetallicRoughness).unwrap()[1].pixels,
            [128; 4]
        );
    }

    #[test]
    fn odd_edge_contributes_and_normals_are_unit_length() {
        let image =
            CpuImage::from_rgba8(3, 1, vec![0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255])
                .unwrap();
        assert_eq!(
            generate_mipmaps(&image, TextureRole::MetallicRoughness).unwrap()[1].pixels,
            [85, 85, 85, 255]
        );
        let normals =
            CpuImage::from_rgba8(2, 1, vec![255, 128, 128, 255, 128, 255, 128, 255]).unwrap();
        let result = &generate_mipmaps(&normals, TextureRole::Normal).unwrap()[1].pixels;
        let length = result[..3]
            .iter()
            .map(|&c| (c as f32 / 127.5 - 1.0).powi(2))
            .sum::<f32>()
            .sqrt();
        assert!((length - 1.0).abs() < 0.01);
        assert!(result[0] > 210 && result[1] > 210);
        let cancel = CpuImage::from_rgba8(2, 1, vec![255, 255, 255, 0, 0, 0, 0, 255]).unwrap();
        assert_eq!(
            generate_mipmaps(&cancel, TextureRole::Normal).unwrap()[1].pixels,
            [128, 128, 255, 128]
        );
    }
}
