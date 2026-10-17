//! KTX2 container writing: one 2D, non-array, single-face image per file.
//!
//! The write side of [`Ktx2Image::from_bytes`](crate::Ktx2Image::from_bytes).
//! [`encode_rgba8_ktx2`] is the offline bake step: it takes the levels
//! [`generate_mipmaps`] produced for a material role and packs them into a
//! container the runtime loader accepts, so the GPU sees an authored chain and
//! never generates a material mip. [`ktx2_container`] is the level-agnostic
//! core, which also writes the native-block fixtures the KTX2 tests compare
//! against. Output is a pure function of the inputs, so a bake is
//! byte-reproducible and cache-friendly.
//!
//! Basis (UASTC/ETC1S) payloads are not written here: there is no pure-Rust
//! Basis *encoder*, so compressed storage stays with the reference `basisu`
//! command line driven by `ornis-assets` (see its `bake` module). This writer
//! needs no codec and always works, at whole-texel RGBA8 size.

use crate::{CpuImage, TextureRole, TextureUploadError, generate_mipmaps};

/// Bytes of the DFD section's own length prefix: the data-format descriptor
/// starts with a `u32` that counts itself.
const DFD_LENGTH_PREFIX: usize = 4;
/// KTX2 requires every level's byte offset to be a multiple of this.
const LEVEL_ALIGNMENT: usize = 16;

/// Packs `image` and its role-filtered mip chain into an uncompressed KTX2.
///
/// Levels come from [`generate_mipmaps`]: the base level plus
/// `floor(log2(max(width, height)))` smaller ones, averaged in linear light
/// for color slots and renormalized for normals. Their payload is
/// byte-identical to the RGBA8 levels the renderer would have generated; only
/// the container's transfer function separates a color slot (`sRGB`) from a
/// data slot (linear), matching the GPU format the upload picks per role.
///
/// # Errors
///
/// [`TextureUploadError`] when the base image is invalid (empty or oversized
/// dimensions, pixel length mismatch) or its format has no data-format
/// descriptor.
pub fn encode_rgba8_ktx2(
    image: &CpuImage,
    role: TextureRole,
) -> Result<Vec<u8>, TextureUploadError> {
    let levels = generate_mipmaps(image, role)?;
    let payloads: Vec<Vec<u8>> = levels.into_iter().map(|level| level.pixels).collect();
    ktx2_container(format_for(role), image.width, image.height, &payloads)
}

/// Builds one KTX2 container from authored level payloads.
///
/// `levels` is largest first, as mips are stored logically: each payload is
/// written verbatim and 16-byte aligned, with the level index pointing at it.
/// Nothing is validated against `format` here — payloads are the caller's
/// blocks of texels — but a container whose level lengths do not match
/// `format` and the dimensions is rejected by
/// [`Ktx2Image::from_bytes`](crate::Ktx2Image::from_bytes), so an invalid
/// combination cannot reach the renderer.
///
/// # Errors
///
/// [`TextureUploadError::UnsupportedKtx2`] when `format` has no basic
/// data-format descriptor to describe it.
pub fn ktx2_container(
    format: ktx2::Format,
    width: u32,
    height: u32,
    levels: &[Vec<u8>],
) -> Result<Vec<u8>, TextureUploadError> {
    let (basic, type_size) = dfd_basic(format)?;
    let dfd = ktx2::dfd::Block::Basic(basic).to_vec();
    let dfd_offset = ktx2::Header::LENGTH + levels.len() * ktx2::LevelIndex::LENGTH;
    let header = ktx2::Header {
        format: Some(format),
        type_size,
        pixel_width: width,
        pixel_height: height,
        pixel_depth: 0,
        layer_count: 0,
        face_count: 1,
        level_count: levels.len() as u32,
        supercompression_scheme: None,
        index: ktx2::Index {
            dfd_byte_offset: dfd_offset as u32,
            dfd_byte_length: (DFD_LENGTH_PREFIX + dfd.len()) as u32,
            kvd_byte_offset: 0,
            kvd_byte_length: 0,
            sgd_byte_offset: 0,
            sgd_byte_length: 0,
        },
    };
    let mut bytes = header.as_bytes().to_vec();
    bytes.resize(dfd_offset, 0);
    bytes.extend_from_slice(&header.index.dfd_byte_length.to_le_bytes());
    bytes.extend_from_slice(&dfd);
    append_levels(&mut bytes, levels);
    Ok(bytes)
}

/// Basic data-format descriptor and upload type size for one storage format.
fn dfd_basic(format: ktx2::Format) -> Result<(ktx2::dfd::Basic, u32), TextureUploadError> {
    ktx2::dfd::Basic::from_format(format)
        .map_err(|_| TextureUploadError::UnsupportedKtx2("no data format descriptor"))
}

/// Storage format carrying the role's color space: four uncompressed 8-bit
/// channels either way, so the transfer function is the whole difference.
fn format_for(role: TextureRole) -> ktx2::Format {
    match role {
        TextureRole::BaseColor | TextureRole::Emissive => ktx2::Format::R8G8B8A8_SRGB,
        TextureRole::MetallicRoughness | TextureRole::Normal => ktx2::Format::R8G8B8A8_UNORM,
    }
}

/// Appends every level tail first, each aligned and indexed where it lands.
fn append_levels(bytes: &mut Vec<u8>, levels: &[Vec<u8>]) {
    levels
        .iter()
        .enumerate()
        .rev()
        .for_each(|(index, level)| append_level(bytes, index, level));
}

/// Writes one level index entry, then the payload it points at.
fn append_level(bytes: &mut Vec<u8>, index: usize, level: &[u8]) {
    bytes.resize(bytes.len().next_multiple_of(LEVEL_ALIGNMENT), 0);
    let entry = ktx2::LevelIndex {
        byte_offset: bytes.len() as u64,
        byte_length: level.len() as u64,
        uncompressed_byte_length: level.len() as u64,
    };
    let at = ktx2::Header::LENGTH + index * ktx2::LevelIndex::LENGTH;
    bytes[at..at + ktx2::LevelIndex::LENGTH].copy_from_slice(&entry.as_bytes());
    bytes.extend_from_slice(level);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Ktx2Image, TextureEncoding, TextureImage};

    /// A raw `VkFormat` value no `ktx2` variant and no DFD builder covers.
    const UNKNOWN_FORMAT_VALUE: u32 = 0x7fff_ffff;

    /// 4×4 RGBA image with one distinct texel per position.
    fn image() -> CpuImage {
        let pixels = (0..16u8)
            .flat_map(|texel| [texel * 16, 255 - texel * 16, texel, 255])
            .collect();
        CpuImage::from_rgba8(4, 4, pixels).expect("valid 4x4 image")
    }

    #[test]
    fn round_trip_keeps_the_authored_chain_untouched() {
        let source = image();
        for role in [
            TextureRole::BaseColor,
            TextureRole::MetallicRoughness,
            TextureRole::Emissive,
            TextureRole::Normal,
        ] {
            let bytes = encode_rgba8_ktx2(&source, role).expect("encodes");
            let image = Ktx2Image::from_bytes(bytes).expect("reader accepts the container");
            assert_eq!(image.dimensions(), (4, 4));
            assert_eq!(image.encoding(), Some(TextureEncoding::Rgba8));
            let levels = TextureImage::Ktx2(image)
                .prepare(role, TextureEncoding::Bc7)
                .expect("prepares");
            assert_eq!(levels.encoding(), TextureEncoding::Rgba8);
            assert_eq!(levels.levels().len(), 3, "4x4 keeps three levels");
            assert_eq!(levels.levels()[0], source.pixels);
            assert_eq!(levels.levels()[1].len(), 2 * 2 * 4);
            assert_eq!(levels.levels()[2].len(), 4, "1x1 tail is one texel");
        }
    }

    #[test]
    fn encodes_deterministically_and_declares_the_role_transfer_function() {
        let first = encode_rgba8_ktx2(&image(), TextureRole::BaseColor).expect("encodes");
        let second = encode_rgba8_ktx2(&image(), TextureRole::BaseColor).expect("encodes");
        assert_eq!(first, second, "same pixels and role give the same bytes");
        let header = ktx2::Reader::new(first.as_slice())
            .expect("container parses")
            .header();
        assert_eq!(header.format, Some(ktx2::Format::R8G8B8A8_SRGB));
        assert_eq!(header.type_size, 1);
        assert_eq!(header.level_count, 3);
        assert_eq!(header.face_count, 1);
        assert_eq!(header.index.kvd_byte_length, 0);
        assert_eq!(header.index.sgd_byte_length, 0);
        let plain = encode_rgba8_ktx2(&image(), TextureRole::Normal).expect("encodes");
        let header = ktx2::Reader::new(plain.as_slice())
            .expect("container parses")
            .header();
        assert_eq!(header.format, Some(ktx2::Format::R8G8B8A8_UNORM));
    }

    #[test]
    fn rejects_invalid_base_images() {
        let empty = CpuImage {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
        assert!(encode_rgba8_ktx2(&empty, TextureRole::BaseColor).is_err());
        let short = CpuImage {
            width: 2,
            height: 2,
            pixels: vec![0; 3],
        };
        assert!(encode_rgba8_ktx2(&short, TextureRole::BaseColor).is_err());
    }

    #[test]
    fn non_square_rectangles_keep_every_level() {
        let image = CpuImage::from_rgba8(3, 1, vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2])
            .expect("valid 3x1 image");
        let bytes = encode_rgba8_ktx2(&image, TextureRole::MetallicRoughness).expect("encodes");
        let levels = Ktx2Image::from_bytes(bytes)
            .expect("container parses")
            .transcode(TextureEncoding::Bc7)
            .expect("prepares");
        assert_eq!(levels.dimensions(), (3, 1));
        assert_eq!(levels.levels().len(), 2, "3x1 has two levels");
        assert_eq!(levels.levels()[0], image.pixels);
    }

    #[test]
    fn unknown_formats_are_rejected_instead_of_guessed() {
        let unknown = ktx2::Format::new(UNKNOWN_FORMAT_VALUE).expect("non-zero raw format");
        let result = ktx2_container(unknown, 4, 4, &[vec![0; 64]]);
        assert!(
            matches!(result, Err(TextureUploadError::UnsupportedKtx2(_))),
            "unexpected {result:?}"
        );
    }

    #[test]
    fn a_wrong_level_length_is_caught_by_the_reader_not_the_writer() {
        // The writer copies payloads verbatim, so this container is built and
        // then rejected where it matters: the loader the renderer uses.
        let short = [vec![0; 8]];
        let bytes = ktx2_container(ktx2::Format::R8G8B8A8_UNORM, 4, 4, &short).expect("writes");
        assert!(matches!(
            Ktx2Image::from_bytes(bytes),
            Err(TextureUploadError::InvalidKtx2(_))
        ));
    }
}
