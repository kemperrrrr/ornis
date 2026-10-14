//! Basis transcoding and bounded Zstd inflation using a patched, independent decoder.

use std::{borrow::Cow, io::Read};

use basisu::{DecodeFlags, TargetFormat, Transcoder};

use crate::{TextureEncoding, TextureUploadError};

/// UASTC LDR block alignment and Zstd's smallest non-single-segment window.
const UASTC_ALIGNMENT: usize = 16;
const MIN_ZSTD_WINDOW: u64 = 1024;
const TARGETS: [TargetFormat; 6] = [
    TargetFormat::Rgba32,
    TargetFormat::Bc1Rgb,
    TargetFormat::Bc3Rgba,
    TargetFormat::Bc7Rgba,
    TargetFormat::Astc4x4Rgba,
    TargetFormat::Etc2Rgba,
];

pub(crate) fn validate(reader: &ktx2::Reader<&[u8]>) -> Result<(), TextureUploadError> {
    use ktx2::{ColorModel as M, SupercompressionScheme as S};
    match (
        reader.color_model(),
        reader.header().supercompression_scheme,
    ) {
        (Some(M::ETC1S), Some(S::BasisLZ)) => Ok(()),
        (Some(M::UASTC), None | Some(S::Zstandard)) => Ok(()),
        _ => Err(TextureUploadError::UnsupportedKtx2(
            "only ETC1S/BasisLZ and UASTC LDR/plain or Zstd",
        )),
    }
}

pub(crate) fn transcode(
    reader: &ktx2::Reader<&[u8]>,
    target: TextureEncoding,
) -> Result<(TextureEncoding, Vec<Vec<u8>>), TextureUploadError> {
    let bytes = plain_basis(reader)?;
    let transcoder = Transcoder::new(&bytes).map_err(|_| TextureUploadError::Transcode)?;
    let target = if transcoder.supports(TARGETS[target as usize])
        && !(target == TextureEncoding::Bc1 && transcoder.has_alpha())
    {
        target
    } else {
        TextureEncoding::Rgba8
    };
    let levels = (0..transcoder.level_count())
        .map(|level| {
            transcoder
                .transcode(level, TARGETS[target as usize], DecodeFlags::HIGH_QUALITY)
                .map_err(|_| TextureUploadError::Transcode)
        })
        .collect::<Result<_, _>>()?;
    Ok((target, levels))
}

fn plain_basis<'a>(reader: &'a ktx2::Reader<&[u8]>) -> Result<Cow<'a, [u8]>, TextureUploadError> {
    let mut header = reader.header();
    if header.supercompression_scheme != Some(ktx2::SupercompressionScheme::Zstandard) {
        return Ok(Cow::Borrowed(reader.data()));
    }
    // Keep metadata at its authored offsets, but not the compressed level data.
    let index = header.index;
    let metadata_end = (ktx2::Header::LENGTH
        + header.level_count as usize * ktx2::LevelIndex::LENGTH)
        .max((index.dfd_byte_offset + index.dfd_byte_length) as usize)
        .max((index.kvd_byte_offset + index.kvd_byte_length) as usize)
        .max((index.sgd_byte_offset + index.sgd_byte_length) as usize);
    let mut bytes = reader.data()[..metadata_end].to_vec();
    header.supercompression_scheme = None;
    bytes[..ktx2::Header::LENGTH].copy_from_slice(&header.as_bytes());
    for (i, level) in reader
        .levels()
        .collect::<Vec<_>>()
        .into_iter()
        .enumerate()
        .rev()
    {
        let decoded = inflate(level)?;
        bytes.resize(bytes.len().next_multiple_of(UASTC_ALIGNMENT), 0);
        let entry = ktx2::LevelIndex {
            byte_offset: bytes.len() as u64,
            byte_length: decoded.len() as u64,
            uncompressed_byte_length: decoded.len() as u64,
        };
        let offset = ktx2::Header::LENGTH + i * ktx2::LevelIndex::LENGTH;
        bytes[offset..offset + ktx2::LevelIndex::LENGTH].copy_from_slice(&entry.as_bytes());
        bytes.extend_from_slice(&decoded);
    }
    Ok(Cow::Owned(bytes))
}

fn inflate(level: ktx2::Level<'_>) -> Result<Vec<u8>, TextureUploadError> {
    // Container validation has already bounded this length by the logical
    // mip's UASTC footprint. Bound the frame window too, before allocating it.
    let mut decoder = ruzstd::decoding::StreamingDecoder::new_with_max_window_size(
        level.data,
        level.uncompressed_byte_length.max(MIN_ZSTD_WINDOW),
    )
    .map_err(|_| TextureUploadError::InvalidKtx2("Zstd frame or window"))?;
    let mut output = vec![0; level.uncompressed_byte_length as usize];
    decoder
        .read_exact(&mut output)
        .map_err(|_| TextureUploadError::InvalidKtx2("Zstd mip length"))?;
    let mut extra = [0];
    if decoder
        .read(&mut extra)
        .map_err(|_| TextureUploadError::Transcode)?
        != 0
        || !decoder.get_ref().is_empty()
    {
        return Err(TextureUploadError::InvalidKtx2("Zstd trailing data"));
    }
    Ok(output)
}
