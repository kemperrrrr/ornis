//! Real ETC1S/UASTC fixtures exercise every target, mip preservation and rejection paths.

use ornis_texture::{Ktx2Image, TextureEncoding as E, TextureUploadError};

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
    for (encoding, format) in [
        (E::Bc7, ktx2::Format::BC7_UNORM_BLOCK),
        (E::Astc4x4, ktx2::Format::ASTC_4x4_UNORM_BLOCK),
        (E::Etc2, ktx2::Format::ETC2_R8G8B8A8_UNORM_BLOCK),
    ] {
        let compressed = image.transcode(encoding).unwrap();
        let native = native_container(format, 40, compressed.levels());
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
fn a_texel_block_dimension_of_255_is_an_error_not_a_panic() {
    // `ktx2` 0.5.0 adds 1 to each texelBlockDimension byte and unwraps.
    // 255 overflows in debug and unwraps None in release. Both Basis paths.
    for bytes in [UASTC, ETC1S] {
        assert!(Ktx2Image::from_bytes(bytes).is_ok());
        let header = ktx2::Header::from_bytes(bytes[..ktx2::Header::LENGTH].try_into().unwrap())
            .expect("fixture header");
        // Basic block follows the 4-byte dfdTotalSize; dimensions sit at +12.
        let dims = header.index.dfd_byte_offset as usize + 4 + 12;
        // 254 + 1 = 255, a legal `NonZeroU8`. Only 255 overflows.
        let mut legal = bytes.to_vec();
        legal[dims] = u8::MAX - 1;
        assert!(
            Ktx2Image::from_bytes(legal.as_slice()).is_ok(),
            "dimension byte 254 must still parse"
        );
        for offset in dims..dims + 4 {
            let mut mutated = bytes.to_vec();
            assert!(offset < mutated.len(), "DFD dimension offset {offset}");
            assert_ne!(
                mutated[offset],
                u8::MAX,
                "fixture byte {offset} is already 255"
            );
            mutated[offset] = u8::MAX;
            assert!(
                matches!(
                    Ktx2Image::from_bytes(mutated.as_slice()),
                    Err(TextureUploadError::InvalidKtx2("DFD texel block dimension"))
                ),
                "offset {offset}"
            );
        }
    }
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
    let bytes = native_container(
        ktx2::Format::R8G8B8A8_SRGB,
        2,
        &[[255, 0, 0, 255].repeat(4), vec![0, 255, 0, 255]],
    );
    let image = ornis_texture::TextureImage::Ktx2(Ktx2Image::from_bytes(bytes).unwrap());
    let result = image
        .prepare(ornis_texture::TextureRole::BaseColor, E::Bc7)
        .unwrap();
    assert_eq!(result.encoding(), E::Rgba8);
    assert_eq!(result.levels()[0], [255, 0, 0, 255].repeat(4));
    assert_eq!(result.levels()[1], [0, 255, 0, 255]);
}

// Independent of `ornis_texture::ktx2_container`: a header taken from a real
// Basis file, then a per-format DFD from `ktx2::dfd`. The reader tests must
// not be fed by the same writer they are checking.
fn native_container(format: ktx2::Format, edge: u32, levels: &[Vec<u8>]) -> Vec<u8> {
    const DFD_LENGTH_PREFIX: usize = 4;
    const LEVEL_ALIGNMENT: usize = 16;
    let mut header = ktx2::Reader::new(UASTC).unwrap().header();
    header.format = Some(format);
    header.pixel_width = edge;
    header.pixel_height = edge;
    header.level_count = levels.len() as u32;
    header.supercompression_scheme = None;
    let (basic, _) = ktx2::dfd::Basic::from_format(format).unwrap();
    let dfd = ktx2::dfd::Block::Basic(basic).to_vec();
    let dfd_offset = ktx2::Header::LENGTH + levels.len() * ktx2::LevelIndex::LENGTH;
    header.index = ktx2::Index {
        dfd_byte_offset: dfd_offset as u32,
        dfd_byte_length: (DFD_LENGTH_PREFIX + dfd.len()) as u32,
        kvd_byte_offset: 0,
        kvd_byte_length: 0,
        sgd_byte_offset: 0,
        sgd_byte_length: 0,
    };
    let mut bytes = header.as_bytes().to_vec();
    bytes.resize(dfd_offset, 0);
    bytes.extend_from_slice(&header.index.dfd_byte_length.to_le_bytes());
    bytes.extend_from_slice(&dfd);
    for (i, level) in levels.iter().enumerate().rev() {
        bytes.resize(bytes.len().next_multiple_of(LEVEL_ALIGNMENT), 0);
        let entry = ktx2::LevelIndex {
            byte_offset: bytes.len() as u64,
            byte_length: level.len() as u64,
            uncompressed_byte_length: level.len() as u64,
        };
        let at = ktx2::Header::LENGTH + i * ktx2::LevelIndex::LENGTH;
        bytes[at..at + ktx2::LevelIndex::LENGTH].copy_from_slice(&entry.as_bytes());
        bytes.extend_from_slice(level);
    }
    bytes
}

#[test]
fn the_independent_container_agrees_with_the_crate_writer_on_levels() {
    let levels = [vec![7u8; 16], vec![9, 8, 7, 6]];
    let format = ktx2::Format::R8G8B8A8_SRGB;
    let independent = native_container(format, 2, &levels);
    let written = ornis_texture::ktx2_container(format, 2, 2, &levels).expect("writer");
    let oracle = Ktx2Image::from_bytes(independent).unwrap();
    let crate_writer = Ktx2Image::from_bytes(written).unwrap();
    assert_eq!(oracle.encoding(), Some(E::Rgba8));
    assert_eq!(
        oracle.transcode(E::Rgba8).unwrap().levels(),
        crate_writer.transcode(E::Rgba8).unwrap().levels()
    );
    assert_eq!(
        ktx2::Reader::new(oracle.bytes())
            .unwrap()
            .transfer_function(),
        Some(ktx2::TransferFunction::SRGB)
    );
}

#[test]
fn bake_fixtures_declare_srgb_for_albedo_and_linear_for_normals() {
    let srgb = ktx2::Reader::new(include_bytes!("data/bake-uastc.ktx2")).unwrap();
    let linear = ktx2::Reader::new(include_bytes!("data/bake-normal-uastc.ktx2")).unwrap();
    assert_eq!(srgb.transfer_function(), Some(ktx2::TransferFunction::SRGB));
    assert_eq!(srgb.color_model(), Some(ktx2::ColorModel::UASTC));
    assert_eq!(
        linear.transfer_function(),
        Some(ktx2::TransferFunction::Linear)
    );
    assert_eq!(linear.color_model(), Some(ktx2::ColorModel::UASTC));
    let image =
        Ktx2Image::from_bytes(include_bytes!("data/bake-normal-uastc.ktx2").as_slice()).unwrap();
    assert_eq!(image.encoding(), None);
    assert_eq!(image.dimensions(), (16, 16));
    let levels = image.transcode(E::Rgba8).unwrap();
    assert_eq!(levels.levels().len(), 5, "16x16 keeps five authored levels");
}

#[test]
fn zstd_preserves_pixels_and_rejects_a_false_decoded_length() {
    let compressed = include_bytes!("data/uastc-zstd.ktx2");
    let expected = Ktx2Image::from_bytes(UASTC)
        .unwrap()
        .transcode(E::Rgba8)
        .unwrap();
    let actual = Ktx2Image::from_bytes(compressed.as_slice())
        .unwrap()
        .transcode(E::Rgba8)
        .unwrap();
    assert_eq!(actual.levels(), expected.levels());
    let mut bad = compressed.to_vec();
    // This fits the allocation budget but is not the stream's actual length.
    bad[96..104].copy_from_slice(&16_u64.to_le_bytes());
    let result = Ktx2Image::from_bytes(bad).and_then(|image| image.transcode(E::Rgba8));
    assert!(result.is_err());
}

#[test]
fn basis_with_alpha_refuses_bc1_and_preserves_rgba_levels() {
    let image = Ktx2Image::from_bytes(include_bytes!("data/uastc-alpha.ktx2").as_slice()).unwrap();
    let fallback = image.transcode(E::Bc1).unwrap();
    let rgba = image.transcode(E::Rgba8).unwrap();
    assert_eq!(fallback.encoding(), E::Rgba8);
    assert_eq!(fallback.levels(), rgba.levels());
}
