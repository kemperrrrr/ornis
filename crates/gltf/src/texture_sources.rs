//! Resolve texture source choices lazily, with optional Basis-to-core fallback.

use std::{collections::HashMap, path::Path};

use crate::basis_extension::{ImageIndex, Parsed, TextureSource};
use crate::{ImportError, LoadedImage};

/// Selected images in texture-index order, sharing decoded source work.
/// Unsupported/corrupt optional Basis containers fall back to PNG/JPEG;
/// missing optional sources also fall back; required Basis and unsafe paths
/// still propagate errors unchanged.
pub(crate) fn resolve_textures(
    parsed: &Parsed,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
) -> Result<Vec<LoadedImage>, ImportError> {
    let images: Vec<_> = parsed.gltf.document.images().collect();
    let mut cache = HashMap::new();
    let mut load = |index: ImageIndex| {
        if let Some(image) = cache.get(&index.value()) {
            return Ok(LoadedImage::clone(image));
        }
        let image = crate::textures::resolve_image(&images[index.value()], buffers, base_dir)?;
        cache.insert(index.value(), image.clone());
        Ok(image)
    };
    parsed
        .sources
        .iter()
        .map(|source| select(*source, &mut load))
        .collect()
}

fn select(
    source: TextureSource,
    load: &mut impl FnMut(ImageIndex) -> Result<LoadedImage, ImportError>,
) -> Result<LoadedImage, ImportError> {
    match source {
        TextureSource::Core(index) => core_image(load(index)?),
        TextureSource::Basis { primary, fallback } => {
            let result = load(primary).and_then(basis_image);
            match (result, fallback) {
                (
                    Err(
                        ImportError::InvalidImage { .. }
                        | ImportError::UnsupportedImage { .. }
                        | ImportError::Io(_)
                        | ImportError::ExternalBuffer(_),
                    ),
                    Some(core),
                ) => core_image(load(core)?),
                (result, _) => result,
            }
        }
    }
}

fn core_image(image: LoadedImage) -> Result<LoadedImage, ImportError> {
    if matches!(image, LoadedImage::Ktx2(_)) {
        return Err(ImportError::UnsupportedImage {
            context: "KTX2 requires KHR_texture_basisu, core source must be PNG/JPEG".to_owned(),
        });
    }
    Ok(image)
}

fn basis_image(image: LoadedImage) -> Result<LoadedImage, ImportError> {
    if !matches!(image, LoadedImage::Ktx2(_)) {
        return Err(ImportError::InvalidImage {
            context: "KHR_texture_basisu source must contain KTX2".to_owned(),
        });
    }
    Ok(image)
}
