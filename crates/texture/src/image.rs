//! CPU material image variants and validated upload-ready mip payloads.

use crate::{
    CpuImage, Ktx2Image, TextureEncoding, TextureRole, TextureUploadError, generate_mipmaps,
};

/// Decoded PNG/JPEG pixels or an authored KTX2 chain, never ambiguous bytes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TextureImage {
    /// RGBA8 base level; mipmaps are generated for the material role at upload.
    Rgba8(CpuImage),
    /// Retains the encoded file and its authored mips until device selection.
    Ktx2(Ktx2Image),
}

impl TextureImage {
    /// Base dimensions in texels.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            Self::Rgba8(image) => (image.width, image.height),
            Self::Ktx2(image) => image.dimensions(),
        }
    }

    /// Decoded pixels, or `None` for a deferred KTX2 payload.
    pub fn rgba8(&self) -> Option<&CpuImage> {
        match self {
            Self::Rgba8(image) => Some(image),
            Self::Ktx2(_) => None,
        }
    }

    /// Produces upload-ready levels. KTX2 mip count and authored filtering are preserved.
    ///
    /// `target` selects Basis transcoding; native blocks pass through only if
    /// it matches their encoding, otherwise they decode to RGBA8.
    ///
    /// # Errors
    /// Invalid RGBA8 data or a failed transcode returns a typed error.
    pub fn prepare(
        &self,
        role: TextureRole,
        target: TextureEncoding,
    ) -> Result<TextureLevels, TextureUploadError> {
        match self {
            Self::Rgba8(image) => TextureLevels::from_rgba8(image, role),
            Self::Ktx2(image) => image.transcode(target),
        }
    }
}

/// Validated, tightly packed mips in largest-to-smallest order.
#[derive(Debug)]
pub struct TextureLevels {
    width: u32,
    height: u32,
    encoding: TextureEncoding,
    levels: Vec<Vec<u8>>,
}

impl TextureLevels {
    /// Generates a complete RGBA8 chain using the material role's filtering.
    ///
    /// # Errors
    /// Rejects invalid base dimensions or pixel length.
    pub fn from_rgba8(image: &CpuImage, role: TextureRole) -> Result<Self, TextureUploadError> {
        let levels = generate_mipmaps(image, role)?
            .into_iter()
            .map(|l| l.pixels)
            .collect();
        Self::checked(image.width, image.height, TextureEncoding::Rgba8, levels)
    }

    pub(crate) fn checked(
        width: u32,
        height: u32,
        encoding: TextureEncoding,
        levels: Vec<Vec<u8>>,
    ) -> Result<Self, TextureUploadError> {
        for (i, level) in levels.iter().enumerate() {
            if level.len() as u64 != encoding.byte_len((width >> i).max(1), (height >> i).max(1)) {
                return Err(TextureUploadError::InvalidKtx2("transcoded mip length"));
            }
        }
        Ok(Self {
            width,
            height,
            encoding,
            levels,
        })
    }

    /// Base size in texels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    /// GPU storage encoding.
    pub fn encoding(&self) -> TextureEncoding {
        self.encoding
    }
    /// All tightly packed mip payloads, largest first.
    pub fn levels(&self) -> &[Vec<u8>] {
        &self.levels
    }
    /// Exact texture payload bytes, including block-rounded small mips.
    /// Driver allocation granularity and metadata are not included.
    pub fn byte_len(&self) -> u64 {
        self.levels.iter().map(|l| l.len() as u64).sum()
    }
}
