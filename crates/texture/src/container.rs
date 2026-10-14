//! Validated two-dimensional KTX2 containers and portable mip payloads.

use std::sync::Arc;

use basisu::{DecodeFlags, SourceFormat, TargetFormat, Transcoder};

use crate::{CpuImage, MAX_TEXTURE_EDGE, TextureRole, TextureUploadError, generate_mipmaps};

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
        if matches!(self, Self::Rgba8) { 1 } else { 4 }
    }

    /// Bytes in a storage block (or one uncompressed texel).
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Rgba8 => 4,
            Self::Bc1 => 8,
            _ => 16,
        }
    }

    /// Exact payload size, including small mip levels' minimum full block.
    pub fn byte_len(self, width: u32, height: u32) -> u64 {
        u64::from(width.div_ceil(self.block_extent()))
            * u64::from(height.div_ceil(self.block_extent()))
            * u64::from(self.block_bytes())
    }

    fn basis_target(self) -> TargetFormat {
        match self {
            Self::Rgba8 => TargetFormat::Rgba32,
            Self::Bc1 => TargetFormat::Bc1Rgb,
            Self::Bc3 => TargetFormat::Bc3Rgba,
            Self::Bc7 => TargetFormat::Bc7Rgba,
            Self::Astc4x4 => TargetFormat::Astc4x4Rgba,
            Self::Etc2 => TargetFormat::Etc2Rgba,
        }
    }
}

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

/// Validated material KTX2 (single 2D image, no arrays, cubemaps or HDR).
/// Bytes are shared on clone; untrusted dimensions and level lengths are checked
/// before the transcoder can allocate output or inflate Zstd data.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Ktx2Image {
    bytes: Arc<[u8]>,
    width: u32,
    height: u32,
    encoding: Option<TextureEncoding>,
}

impl Ktx2Image {
    /// Parses a KTX2 file without discarding authored mips.
    ///
    /// Accepts ETC1S/BasisLZ, UASTC LDR (plain or Zstd), RGBA8 and native
    /// BC1 RGBA/BC3/BC7/ASTC 4×4/ETC2 RGBA8 blocks. Other encodings are rejected.
    /// Native GPU blocks must not be supercompressed.
    ///
    /// # Errors
    /// Rejects malformed containers, unsupported shapes/codecs and oversized data.
    pub fn from_bytes(bytes: impl Into<Arc<[u8]>>) -> Result<Self, TextureUploadError> {
        let bytes = bytes.into();
        // Bound level-index/DFD work before Reader walks attacker-controlled tables.
        let header_bytes = bytes
            .get(..ktx2::Header::LENGTH)
            .ok_or(TextureUploadError::InvalidKtx2("truncated header"))?;
        let header = ktx2::Header::from_bytes(
            header_bytes
                .try_into()
                .map_err(|_| TextureUploadError::InvalidKtx2("header"))?,
        )
        .map_err(|_| TextureUploadError::InvalidKtx2("header"))?;
        validate_shape(&header)?;
        if bytes.len() > 512 * 1024 * 1024 {
            return Err(TextureUploadError::InvalidKtx2("file exceeds 512 MiB"));
        }
        let reader = ktx2::Reader::new(bytes.as_ref())
            .map_err(|_| TextureUploadError::InvalidKtx2("container bounds or DFD"))?;
        let encoding = header.format.map(native_encoding).transpose()?;
        validate_levels(&reader, encoding)?;
        if encoding.is_none() {
            validate_basis(&bytes)?;
        }
        Ok(Self {
            bytes,
            width: header.pixel_width,
            height: header.pixel_height,
            encoding,
        })
    }

    /// Original bytes used for deduplication and deferred transcoding.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Base dimensions in texels.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Native block format, or `None` when Basis needs a target format.
    pub fn encoding(&self) -> Option<TextureEncoding> {
        self.encoding
    }

    /// Transcodes Basis mips, passes through native blocks, or decodes RGBA8 fallback.
    ///
    /// # Errors
    /// Returns a typed error if decoding fails; never regenerates authored levels.
    pub fn transcode(&self, target: TextureEncoding) -> Result<TextureLevels, TextureUploadError> {
        let reader = ktx2::Reader::new(self.bytes())
            .map_err(|_| TextureUploadError::InvalidKtx2("container"))?;
        let (encoding, levels) = if let Some(native) = self.encoding {
            native_levels(&reader, native, target)?
        } else {
            let transcoder =
                Transcoder::new(self.bytes()).map_err(|_| TextureUploadError::Transcode)?;
            let target = if transcoder.supports(target.basis_target()) {
                target
            } else {
                TextureEncoding::Rgba8
            };
            let levels = (0..transcoder.level_count())
                .map(|level| {
                    transcoder
                        .transcode(level, target.basis_target(), DecodeFlags::HIGH_QUALITY)
                        .map_err(|_| TextureUploadError::Transcode)
                })
                .collect::<Result<Vec<_>, _>>()?;
            (target, levels)
        };
        TextureLevels::checked(self.width, self.height, encoding, levels)
    }
}

fn validate_shape(header: &ktx2::Header) -> Result<(), TextureUploadError> {
    let (w, h) = (header.pixel_width, header.pixel_height);
    if w == 0 || h == 0 || w > MAX_TEXTURE_EDGE || h > MAX_TEXTURE_EDGE {
        return Err(TextureUploadError::InvalidKtx2("dimensions out of range"));
    }
    if header.type_size != 1 {
        return Err(TextureUploadError::UnsupportedKtx2(
            "only byte-based material formats",
        ));
    }
    if header.pixel_depth != 0 || header.layer_count != 0 || header.face_count != 1 {
        return Err(TextureUploadError::UnsupportedKtx2(
            "only non-array 2D materials are supported",
        ));
    }
    if header.level_count == 0 || header.level_count > w.max(h).ilog2() + 1 {
        return Err(TextureUploadError::InvalidKtx2("invalid mip count"));
    }
    Ok(())
}

fn native_encoding(format: ktx2::Format) -> Result<TextureEncoding, TextureUploadError> {
    use ktx2::Format as F;
    match format {
        F::R8G8B8A8_UNORM | F::R8G8B8A8_SRGB => Ok(TextureEncoding::Rgba8),
        F::BC1_RGBA_UNORM_BLOCK | F::BC1_RGBA_SRGB_BLOCK => Ok(TextureEncoding::Bc1),
        F::BC3_UNORM_BLOCK | F::BC3_SRGB_BLOCK => Ok(TextureEncoding::Bc3),
        F::BC7_UNORM_BLOCK | F::BC7_SRGB_BLOCK => Ok(TextureEncoding::Bc7),
        F::ASTC_4x4_UNORM_BLOCK | F::ASTC_4x4_SRGB_BLOCK => Ok(TextureEncoding::Astc4x4),
        F::ETC2_R8G8B8A8_UNORM_BLOCK | F::ETC2_R8G8B8A8_SRGB_BLOCK => Ok(TextureEncoding::Etc2),
        _ => Err(TextureUploadError::UnsupportedKtx2("native GPU format")),
    }
}

fn validate_levels(
    reader: &ktx2::Reader<&[u8]>,
    encoding: Option<TextureEncoding>,
) -> Result<(), TextureUploadError> {
    let header = reader.header();
    if encoding.is_some() && header.supercompression_scheme.is_some() {
        return Err(TextureUploadError::UnsupportedKtx2(
            "supercompressed native blocks",
        ));
    }
    for (index, level) in reader.levels().enumerate() {
        let (w, h) = (
            (header.pixel_width >> index).max(1),
            (header.pixel_height >> index).max(1),
        );
        let expected = encoding.unwrap_or(TextureEncoding::Astc4x4).byte_len(w, h);
        if level.data.is_empty() || level.uncompressed_byte_length > expected {
            return Err(TextureUploadError::InvalidKtx2(
                "mip length or decompression budget",
            ));
        }
        if encoding.is_some()
            && (level.data.len() as u64 != expected || level.uncompressed_byte_length != expected)
        {
            return Err(TextureUploadError::InvalidKtx2("native mip length"));
        }
    }
    Ok(())
}

fn validate_basis(bytes: &[u8]) -> Result<(), TextureUploadError> {
    let transcoder =
        Transcoder::new(bytes).map_err(|_| TextureUploadError::InvalidKtx2("Basis payload"))?;
    if !matches!(
        transcoder.source_format(),
        SourceFormat::Etc1s | SourceFormat::UastcLdr
    ) || transcoder.is_video()
    {
        return Err(TextureUploadError::UnsupportedKtx2(
            "only ETC1S and UASTC LDR Basis materials",
        ));
    }
    Ok(())
}

fn native_levels(
    reader: &ktx2::Reader<&[u8]>,
    native: TextureEncoding,
    target: TextureEncoding,
) -> Result<(TextureEncoding, Vec<Vec<u8>>), TextureUploadError> {
    if target == native || native == TextureEncoding::Rgba8 {
        return Ok((native, reader.levels().map(|l| l.data.to_vec()).collect()));
    }
    let header = reader.header();
    let levels = reader
        .levels()
        .enumerate()
        .map(|(i, l)| {
            decode_blocks(
                l.data,
                (header.pixel_width >> i).max(1),
                (header.pixel_height >> i).max(1),
                native,
            )
        })
        .collect::<Result<_, _>>()?;
    Ok((TextureEncoding::Rgba8, levels))
}

fn decode_blocks(
    bytes: &[u8],
    w: u32,
    h: u32,
    encoding: TextureEncoding,
) -> Result<Vec<u8>, TextureUploadError> {
    let mut output = vec![0; (w * h) as usize];
    let (w, h) = (w as usize, h as usize);
    use texture2ddecoder as d;
    let result = match encoding {
        TextureEncoding::Bc1 => d::decode_bc1a(bytes, w, h, &mut output),
        TextureEncoding::Bc3 => d::decode_bc3(bytes, w, h, &mut output),
        TextureEncoding::Bc7 => d::decode_bc7(bytes, w, h, &mut output),
        TextureEncoding::Astc4x4 => d::decode_astc(bytes, w, h, 4, 4, &mut output),
        TextureEncoding::Etc2 => d::decode_etc2_rgba8(bytes, w, h, &mut output),
        TextureEncoding::Rgba8 => return Ok(bytes.to_vec()),
    };
    result.map_err(|_| TextureUploadError::Transcode)?;
    // Decoder's packed words are BGRA in little-endian order, not RGBA.
    Ok(output
        .into_iter()
        .flat_map(|p| {
            let [b, g, r, a] = p.to_le_bytes();
            [r, g, b, a]
        })
        .collect())
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

    fn checked(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_fallback_has_rgba_order_and_preserves_alpha() {
        // BC1: solid red, then transparent index 3 with endpoint0 <= endpoint1.
        let red = decode_blocks(
            &[0x00, 0xf8, 0x00, 0x00, 0, 0, 0, 0],
            4,
            4,
            TextureEncoding::Bc1,
        )
        .unwrap();
        assert!(
            red.as_chunks::<4>()
                .0
                .iter()
                .all(|p| *p == [255, 0, 0, 255])
        );
        let transparent = decode_blocks(
            &[0, 0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
            1,
            1,
            TextureEncoding::Bc1,
        )
        .unwrap();
        assert_eq!(transparent[3], 0);
    }
}
