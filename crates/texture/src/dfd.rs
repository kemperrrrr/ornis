//! Rejects KTX2 data-format descriptors that `ktx2` 0.5.0 panics on.
//!
//! `ktx2` 0.5.0 computes `NonZeroU8::new(dim + 1).unwrap()` for each
//! `texelBlockDimension` byte. A value of 255 overflows in debug and unwraps
//! `None` in release, so a corrupt container must be refused before
//! `ktx2::Reader::new`.

use crate::TextureUploadError;

/// Bytes of the DFD section's leading `dfdTotalSize`, skipped before blocks.
const DFD_TOTAL_SIZE_LEN: usize = 4;
/// Length of a DFD block header (vendor/type word, version, descriptor size).
const DFD_BLOCK_HEADER_LEN: usize = 8;
/// Offset of `texelBlockDimension[0..4]` inside a Basic DFD block.
const DFD_TEXEL_BLOCK_DIMS: std::ops::Range<usize> = 12..16;
/// Packed vendor id and descriptor type of `dfd::BlockHeader::BASIC` (both 0).
const BASIC_VENDOR_TYPE: u32 = 0;
/// Version number of `dfd::BlockHeader::BASIC`.
const BASIC_DFD_VERSION: u16 = 2;

/// Walks DFD blocks the way `ktx2::Reader` does and rejects a Basic block
/// whose `texelBlockDimension` byte is `0xFF`.
///
/// Indexes use checked arithmetic. A short or truncated block is left for
/// `ktx2::Reader::new`, which reports it and never evaluates `dim + 1`.
///
/// # Errors
///
/// Returns [`TextureUploadError::InvalidKtx2`] when the DFD section is out of
/// range or a Basic block carries an overflowing texel dimension.
pub(crate) fn validate_dfd(bytes: &[u8], header: &ktx2::Header) -> Result<(), TextureUploadError> {
    const BAD: TextureUploadError = TextureUploadError::InvalidKtx2("DFD bounds");
    let start = usize::try_from(header.index.dfd_byte_offset).map_err(|_| BAD)?;
    let len = usize::try_from(header.index.dfd_byte_length).map_err(|_| BAD)?;
    let end = start.checked_add(len).ok_or(BAD)?;
    let mut data = bytes
        .get(start.checked_add(DFD_TOTAL_SIZE_LEN).ok_or(BAD)?..end)
        .ok_or(BAD)?;
    while data.len() >= DFD_BLOCK_HEADER_LEN {
        let word0 = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let version = u16::from_le_bytes([data[4], data[5]]);
        let size = usize::from(u16::from_le_bytes([data[6], data[7]]));
        if size < DFD_BLOCK_HEADER_LEN {
            return Ok(());
        }
        let Some(block) = data.get(..size) else {
            return Ok(());
        };
        let basic = word0 == BASIC_VENDOR_TYPE && version == BASIC_DFD_VERSION;
        let overflowing = block
            .get(DFD_TEXEL_BLOCK_DIMS)
            .is_some_and(|dims| dims.contains(&u8::MAX));
        if basic && overflowing {
            return Err(TextureUploadError::InvalidKtx2("DFD texel block dimension"));
        }
        data = &data[size..];
    }
    Ok(())
}
