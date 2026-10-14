//! Real ETC1S/UASTC fixtures exercise every target, mip preservation and rejection paths.

use ornis_texture::{Ktx2Image, TextureEncoding as E};

const ETC1S: &[u8] = include_bytes!("data/etc1s.ktx2");
const UASTC: &[u8] = include_bytes!("data/uastc.ktx2");

#[test]
fn real_basis_files_transcode_all_mips_to_every_target() {
    for bytes in [
        ETC1S,
        UASTC,
        include_bytes!("data/uastc-zstd.ktx2").as_slice(),
    ] {
        let image = Ktx2Image::from_bytes(bytes).unwrap();
        assert_eq!(image.dimensions(), (40, 40));
        assert_eq!(image.bytes(), bytes);
        for target in [E::Rgba8, E::Bc7, E::Astc4x4, E::Etc2] {
            let levels = image.transcode(target).unwrap();
            assert_eq!(levels.encoding(), target);
            assert_eq!(levels.levels().len(), 6, "authored chain must survive");
            for (i, level) in levels.levels().iter().enumerate() {
                assert_eq!(
                    level.len() as u64,
                    target.byte_len((40 >> i).max(1), (40 >> i).max(1))
                );
            }
        }
    }
}

#[test]
fn native_blocks_are_preserved_and_fallback_decodes_each_level() {
    let image = Ktx2Image::from_bytes(UASTC).unwrap();
    // These encodings have the same 16-byte footprint as the source UASTC,
    // so its level index and DFD sample layout can be reused for the test.
    for (encoding, vk_format, model) in [
        (E::Bc7, 145_u32, 134_u8),
        (E::Astc4x4, 157, 162),
        (E::Etc2, 151, 161),
    ] {
        let compressed = image.transcode(encoding).unwrap();
        let mut native = UASTC.to_vec();
        native[12..16].copy_from_slice(&vk_format.to_le_bytes());
        let reader = ktx2::Reader::new(UASTC).unwrap();
        native[reader.header().index.dfd_byte_offset as usize + 12] = model;
        for (i, level) in compressed.levels().iter().enumerate() {
            let offset =
                u64::from_le_bytes(UASTC[80 + 24 * i..88 + 24 * i].try_into().unwrap()) as usize;
            native[offset..offset + level.len()].copy_from_slice(level);
        }
        let parsed = Ktx2Image::from_bytes(native).unwrap();
        assert_eq!(parsed.encoding(), Some(encoding));
        assert_eq!(
            parsed.transcode(encoding).unwrap().levels(),
            compressed.levels()
        );
        let fallback = parsed.transcode(E::Rgba8).unwrap();
        assert_eq!(fallback.levels().len(), 6);
        assert_eq!(fallback.encoding(), E::Rgba8);
        let reference = image.transcode(E::Rgba8).unwrap();
        let total_error: u64 = reference
            .levels()
            .iter()
            .flatten()
            .zip(fallback.levels().iter().flatten())
            .map(|(&a, &b)| a.abs_diff(b) as u64)
            .sum();
        let mean_error = total_error as f64 / reference.byte_len() as f64;
        assert!(
            mean_error < 12.0,
            "{encoding:?}: RGBA fallback error {mean_error}"
        );
    }
}

#[test]
fn rejects_invalid_shape_truncation_and_decompression_bombs() {
    for length in [0, 12, 79, 90, 300, UASTC.len() - 1] {
        assert!(Ktx2Image::from_bytes(&UASTC[..length]).is_err());
    }
    for (offset, value) in [
        (20, 0_u32),
        (20, 8193),
        (28, 1),
        (32, 1),
        (36, 6),
        (40, 0),
        (40, 32),
    ] {
        let mut bytes = UASTC.to_vec();
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        assert!(
            Ktx2Image::from_bytes(bytes).is_err(),
            "header offset {offset}"
        );
    }
    let mut bomb = UASTC.to_vec();
    bomb[96..104].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Ktx2Image::from_bytes(bomb).is_err());
}

#[test]
fn texture_payload_accounting_includes_small_block_mips() {
    let image = Ktx2Image::from_bytes(UASTC).unwrap();
    assert_eq!(image.transcode(E::Rgba8).unwrap().byte_len(), 8520);
    for format in [E::Bc7, E::Astc4x4, E::Etc2] {
        assert_eq!(image.transcode(format).unwrap().byte_len(), 2240);
    }
    // 2048² full chain, useful for the memory-budget comparison in the PR.
    let chain_bytes = |format: E| -> u64 {
        (0..12)
            .map(|i| format.byte_len((2048 >> i).max(1), (2048 >> i).max(1)))
            .sum()
    };
    assert_eq!(chain_bytes(E::Rgba8), 22_369_620);
    assert_eq!(chain_bytes(E::Bc7), 5_592_432);
    assert_eq!(chain_bytes(E::Bc1), 2_796_216);
}

#[test]
fn authored_rgba_levels_are_not_regenerated_or_recolored() {
    let mut header = ktx2::Reader::new(UASTC).unwrap().header();
    header.format = Some(ktx2::Format::R8G8B8A8_SRGB);
    header.pixel_width = 2;
    header.pixel_height = 2;
    header.level_count = 2;
    header.supercompression_scheme = None;
    let (basic, _) = ktx2::dfd::Basic::from_format(header.format.unwrap()).unwrap();
    let dfd = ktx2::dfd::Block::Basic(basic).to_vec();
    header.index = ktx2::Index {
        dfd_byte_offset: 128,
        dfd_byte_length: (4 + dfd.len()) as u32,
        kvd_byte_offset: 0,
        kvd_byte_length: 0,
        sgd_byte_offset: 0,
        sgd_byte_length: 0,
    };
    let mut bytes = header.as_bytes().to_vec();
    bytes.resize(128, 0);
    bytes.extend_from_slice(&header.index.dfd_byte_length.to_le_bytes());
    bytes.extend_from_slice(&dfd);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let tail = ktx2::LevelIndex {
        byte_offset: bytes.len() as u64,
        byte_length: 4,
        uncompressed_byte_length: 4,
    };
    bytes.extend_from_slice(&[0, 255, 0, 255]);
    let base = ktx2::LevelIndex {
        byte_offset: bytes.len() as u64,
        byte_length: 16,
        uncompressed_byte_length: 16,
    };
    bytes.extend_from_slice(&[255, 0, 0, 255].repeat(4));
    bytes[80..104].copy_from_slice(&base.as_bytes());
    bytes[104..128].copy_from_slice(&tail.as_bytes());
    let image = ornis_texture::TextureImage::Ktx2(Ktx2Image::from_bytes(bytes).unwrap());
    let result = image
        .prepare(ornis_texture::TextureRole::BaseColor, E::Bc7)
        .unwrap();
    assert_eq!(result.encoding(), E::Rgba8);
    assert_eq!(result.levels()[0], [255, 0, 0, 255].repeat(4));
    assert_eq!(result.levels()[1], [0, 255, 0, 255]);
}
