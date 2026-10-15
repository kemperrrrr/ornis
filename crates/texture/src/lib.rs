//! GPU-independent material images, role-aware mip generation and KTX2 transcoding.

#![warn(missing_docs)]

mod basis;
mod container;
mod encode;
mod encoding;
mod image;
mod mipmaps;

pub use container::Ktx2Image;
pub use encode::{encode_rgba8_ktx2, ktx2_container};
pub use encoding::TextureEncoding;
pub use image::{TextureImage, TextureLevels};
pub use mipmaps::generate_mipmaps;

/// Maximum texture edge in pixels accepted by [`CpuImage::validate`].
///
/// Matches the guaranteed WebGPU limit (`maxTextureDimension2D` floor), so an
/// image passing validation is uploadable on every backend.
pub const MAX_TEXTURE_EDGE: u32 = 8192;
/// Bytes per RGBA8 texel.
const RGBA8_BYTES: usize = 4;
/// Which material slot an uploaded image feeds.
///
/// Mirrors `ornis_gltf::TextureRole` without depending on that crate; the
/// upload matches on this to pick the GPU format and the material binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureRole {
    /// `baseColorTexture`: albedo multiplier (sRGB color data).
    BaseColor,
    /// `metallicRoughnessTexture`: green holds roughness, blue holds
    /// metallic (linear data, never sRGB-decoded).
    MetallicRoughness,
    /// `emissiveTexture`: emission multiplier (sRGB color data).
    Emissive,
    /// Tangent-space normal map (linear, renormalized when downsampling).
    Normal,
}

/// CPU-side texture image: always RGBA8, row-major, top row first.
///
/// PNG and JPEG decode to this representation. `pixels` holds exactly
/// `width * height * 4` bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CpuImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 bytes, row-major from the top row.
    pub pixels: Vec<u8>,
}

impl CpuImage {
    /// Builds an image, validating dimensions and pixel length.
    ///
    /// # Errors
    ///
    /// Returns [`TextureUploadError`] when either dimension is zero, either
    /// edge exceeds [`MAX_TEXTURE_EDGE`], or `pixels` is not exactly
    /// `width * height * 4` bytes.
    pub fn from_rgba8(
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    ) -> Result<Self, TextureUploadError> {
        let image = Self {
            width,
            height,
            pixels,
        };
        image.validate()?;
        Ok(image)
    }

    /// Checks dimensions and pixel length (see [`Self::from_rgba8`]).
    ///
    /// # Errors
    ///
    /// Same as [`Self::from_rgba8`].
    pub fn validate(&self) -> Result<(), TextureUploadError> {
        if self.width == 0 || self.height == 0 {
            return Err(TextureUploadError::EmptyDimensions {
                width: self.width,
                height: self.height,
            });
        }
        if self.width > MAX_TEXTURE_EDGE || self.height > MAX_TEXTURE_EDGE {
            return Err(TextureUploadError::ImageTooLarge {
                width: self.width,
                height: self.height,
            });
        }
        let expected = self.width as u64 * self.height as u64 * RGBA8_BYTES as u64;
        if self.pixels.len() as u64 != expected {
            return Err(TextureUploadError::PixelLengthMismatch {
                expected,
                actual: self.pixels.len() as u64,
            });
        }
        Ok(())
    }

    /// RGBA bytes of the texel at `(x, y)` (origin: top-left).
    ///
    /// Returns `None` when the coordinates are out of bounds. Pure helper
    /// for the CPU sampling mirrors below.
    pub fn texel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = (y as usize * self.width as usize + x as usize) * RGBA8_BYTES;
        let slice = self.pixels.get(offset..offset + RGBA8_BYTES)?;
        Some([slice[0], slice[1], slice[2], slice[3]])
    }
}

/// Invalid image data or unsupported texture container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TextureUploadError {
    /// Either dimension is zero — there is no honest GPU texture for it.
    #[error("texture has empty dimensions ({width}x{height})")]
    EmptyDimensions {
        /// Offending width.
        width: u32,
        /// Offending height.
        height: u32,
    },
    /// An edge exceeds [`MAX_TEXTURE_EDGE`] (not uploadable everywhere).
    #[error("texture too large ({width}x{height})")]
    ImageTooLarge {
        /// Offending width.
        width: u32,
        /// Offending height.
        height: u32,
    },
    /// `pixels` is not exactly `width * height * 4` bytes.
    #[error("texture pixel length mismatch: expected {expected}, got {actual}")]
    PixelLengthMismatch {
        /// Required byte count.
        expected: u64,
        /// Actual byte count.
        actual: u64,
    },
    /// Malformed KTX2 container or mip payload.
    #[error("invalid KTX2 texture: {0}")]
    InvalidKtx2(&'static str),
    /// Unsupported container shape, format or supercompression.
    #[error("unsupported KTX2 texture: {0}")]
    UnsupportedKtx2(&'static str),
    /// Basis transcoding or native block decoding failed.
    #[error("texture transcoding failed")]
    Transcode,
}
