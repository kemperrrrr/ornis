//! Normalize KHR_texture_basisu before gltf-rs core validation, retaining all other checks.

use gltf::{Document, Gltf};

use crate::ImportError;

const EXTENSION: &str = "KHR_texture_basisu";

/// Image references remain typed across source selection and document validation.
pub(crate) type ImageIndex = gltf::json::Index<gltf::json::image::Image>;

/// Validated source choices for a texture; required Basis has no fallback.
#[derive(Clone, Copy)]
pub(crate) enum TextureSource {
    Core(ImageIndex),
    Basis {
        primary: ImageIndex,
        fallback: Option<ImageIndex>,
    },
}

/// Core-validated document plus the original source choices in texture order.
pub(crate) struct Parsed {
    pub gltf: Gltf,
    pub sources: Vec<TextureSource>,
}

/// Normalize Basis for gltf-rs validation without discarding the core fallback.
pub(crate) fn parse(bytes: &[u8]) -> Result<Parsed, ImportError> {
    let parsed = Gltf::from_slice_without_validation(bytes).map_err(parse_error)?;
    let mut root = parsed.document.into_json();
    let required = root
        .extensions_required
        .iter()
        .any(|name| name == EXTENSION);
    let declared = root.extensions_used.iter().any(|name| name == EXTENSION);
    let image_count = root.images.len();
    let sources = root
        .textures
        .iter_mut()
        .map(|texture| source_choice(texture, image_count, required, declared))
        .collect::<Result<_, _>>()?;
    root.extensions_required.retain(|name| name != EXTENSION);
    validate_attribute_indices(&root)?;
    let document = Document::from_json(root).map_err(parse_error)?;
    Ok(Parsed {
        gltf: Gltf {
            document,
            blob: parsed.blob,
        },
        sources,
    })
}

fn source_choice(
    texture: &mut gltf::json::texture::Texture,
    image_count: usize,
    required: bool,
    declared: bool,
) -> Result<TextureSource, ImportError> {
    let Some(basis) = texture
        .extensions
        .as_ref()
        .and_then(|e| e.others.get(EXTENSION))
    else {
        return Ok(TextureSource::Core(texture.source));
    };
    if !declared {
        return Err(ImportError::Parse {
            message: format!("{EXTENSION} must appear in extensionsUsed"),
        });
    }
    let core = texture.source;
    if core.value() != u32::MAX as usize && core.value() >= image_count {
        return Err(invalid("core fallback image index out of bounds"));
    }
    let source = basis
        .get("source")
        .and_then(|v| v.as_u64())
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| invalid("source must be an unsigned image index"))?;
    if source as usize >= image_count {
        return Err(invalid("source image index out of bounds"));
    }
    let primary = ImageIndex::new(source);
    texture.source = primary;
    let fallback = (!required && core.value() != u32::MAX as usize).then_some(core);
    Ok(TextureSource::Basis { primary, fallback })
}

// gltf-json 1.4.1 reports an invalid attribute index, then its POSITION
// min/max hook indexes the same accessor unchecked before returning errors.
// TODO: remove after a gltf-json release includes the bounds-checked hook:
// https://github.com/gltf-rs/gltf/blob/master/gltf-json/src/mesh.rs#L140-L160
fn validate_attribute_indices(root: &gltf::json::Root) -> Result<(), ImportError> {
    let invalid_index = root
        .meshes
        .iter()
        .flat_map(|mesh| &mesh.primitives)
        .flat_map(|primitive| primitive.attributes.values())
        .any(|index| index.value() >= root.accessors.len());
    if invalid_index {
        return Err(ImportError::Parse {
            message: "mesh attribute accessor index out of bounds".to_owned(),
        });
    }
    Ok(())
}

fn invalid(reason: &str) -> ImportError {
    ImportError::InvalidImage {
        context: format!("{EXTENSION}: {reason}"),
    }
}

fn parse_error(error: gltf::Error) -> ImportError {
    ImportError::Parse {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests;
