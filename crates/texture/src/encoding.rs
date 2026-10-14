//! Portable LDR material storage formats and their physical block geometry.

const RGBA_BYTES: u32 = 4;
const BLOCK_EDGE: u32 = 4;
const BC1_BYTES: u32 = 8;
const BLOCK_BYTES: u32 = 16;

/// Storage encoding independent of color space (the material role selects sRGB).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureEncoding {
    /// Four uncompressed 8-bit channels.
    Rgba8,
    /// BC1 RGBA, eight bytes per 4×4 block.
    Bc1,
    /// BC3 RGBA, sixteen bytes per 4×4 block.
    Bc3,
    /// BC7 RGBA, sixteen bytes per 4×4 block.
    Bc7,
    /// ASTC LDR RGBA, sixteen bytes per 4×4 block.
    Astc4x4,
    /// ETC2 RGBA8, sixteen bytes per 4×4 block.
    Etc2,
}

impl TextureEncoding {
    /// Texel dimensions of one storage block.
    pub const fn block_extent(self) -> u32 {
        if matches!(self, Self::Rgba8) {
            1
        } else {
            BLOCK_EDGE
        }
    }

    /// Bytes in a storage block (or one uncompressed texel).
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Rgba8 => RGBA_BYTES,
            Self::Bc1 => BC1_BYTES,
            _ => BLOCK_BYTES,
        }
    }

    /// Exact payload size, including small mip levels' minimum full block.
    pub fn byte_len(self, width: u32, height: u32) -> u64 {
        u64::from(width.div_ceil(self.block_extent()))
            * u64::from(height.div_ceil(self.block_extent()))
            * u64::from(self.block_bytes())
    }
}
