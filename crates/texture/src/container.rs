//! Validated two-dimensional KTX2 containers and portable mip payloads.

use std::sync::Arc;

use crate::{MAX_TEXTURE_EDGE, TextureEncoding, TextureLevels, TextureUploadError};

const MAX_FILE_BYTES: usize = 512 * 1024 * 1024;

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
        if bytes.len() > MAX_FILE_BYTES {
            return Err(TextureUploadError::InvalidKtx2("file exceeds 512 MiB"));
        }
        let reader = ktx2::Reader::new(bytes.as_ref())
            .map_err(|_| TextureUploadError::InvalidKtx2("container bounds or DFD"))?;
        let encoding = header.format.map(native_encoding).transpose()?;
        validate_levels(&reader, encoding)?;
        if encoding.is_none() {
            crate::basis::validate(&reader)?;
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
            crate::basis::transcode(&reader, target)?
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
