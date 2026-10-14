//! Normalize KHR_texture_basisu before gltf-rs core validation, retaining all other checks.

use gltf::{Document, Gltf};

use crate::{ImportError, LoadedImage};

const EXTENSION: &str = "KHR_texture_basisu";

/// gltf-rs has no typed Basis extension. Promote its image source into the
/// core texture slot and remove only this supported required-extension name,
/// then run the normal validator (never import an unvalidated document).
pub(crate) fn parse(bytes: &[u8]) -> Result<Gltf, ImportError> {
    let parsed = Gltf::from_slice_without_validation(bytes).map_err(parse_error)?;
    let mut root = parsed.document.into_json();
    let image_count = root.images.len();
    for texture in &mut root.textures {
        let Some(basis) = texture
            .extensions
            .as_ref()
            .and_then(|e| e.others.get(EXTENSION))
        else {
            continue;
        };
        // An explicit invalid core fallback is still an invalid document.
        let fallback = texture.source.value();
        if fallback != u32::MAX as usize && fallback >= image_count {
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
        texture.source = gltf::json::Index::new(source);
    }
    root.extensions_required.retain(|name| name != EXTENSION);
    validate_attribute_indices(&root)?;
    let document = Document::from_json(root).map_err(parse_error)?;
    Ok(Gltf {
        document,
        blob: parsed.blob,
    })
}

// gltf-json 1.4.1 reports an invalid attribute index, then its POSITION
// min/max hook indexes the same accessor unchecked before returning errors.
// Guard that path until upstream publishes the bounds-checked validator.
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

/// The extension specifically requires KTX2, not just any decodable image.
pub(crate) fn validate_images(gltf: &Gltf, images: &[LoadedImage]) -> Result<(), ImportError> {
    for texture in gltf.document.textures() {
        if texture.extension_value(EXTENSION).is_some()
            && !matches!(
                images.get(texture.source().index()),
                Some(LoadedImage::Ktx2(_))
            )
        {
            return Err(invalid("source must contain a KTX2 image"));
        }
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
mod tests {
    use super::*;
    use crate::fixtures::{build_gltf, encode, triangle};
    use serde_json::{Value, json};

    const IMAGE: &[u8] = include_bytes!("../../texture/tests/data/uastc.ktx2");

    fn document() -> Value {
        let mut doc: Value = serde_json::from_str(&build_gltf(&triangle())).unwrap();
        doc["extensionsUsed"] = json!([EXTENSION]);
        doc["extensionsRequired"] = json!([EXTENSION]);
        doc["images"] = json!([{"uri": format!("data:image/ktx2;base64,{}", encode(IMAGE))}]);
        doc["textures"] = json!([{"extensions": {EXTENSION: {"source": 0}}}]);
        // The bare triangle fixture has no material table. Add the material
        // and its primitive binding explicitly before assigning texture slots.
        doc["materials"] = json!([{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]);
        doc["meshes"][0]["primitives"][0]["material"] = json!(0);
        doc
    }

    #[test]
    fn required_basis_without_core_source_imports_and_preserves_mips() {
        let doc = document();
        let model = crate::load_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        let image = model.primitives[0]
            .material
            .textures
            .base_color_texture
            .as_ref()
            .unwrap();
        let LoadedImage::Ktx2(image) = image else {
            panic!("KTX2 was expanded prematurely");
        };
        assert_eq!(image.bytes(), IMAGE);
        assert_eq!(
            image
                .transcode(ornis_texture::TextureEncoding::Rgba8)
                .unwrap()
                .levels()
                .len(),
            6
        );
    }

    #[test]
    fn optional_basis_overrides_core_fallback_and_external_files_resolve() {
        let mut doc = document();
        doc.as_object_mut().unwrap().remove("extensionsRequired");
        let fallback = doc["images"][0].clone();
        doc["images"].as_array_mut().unwrap().push(fallback);
        doc["textures"][0]["source"] = json!(1);
        doc["images"][0] = json!({"uri": "texture.ktx2", "mimeType": "image/ktx2"});
        let bytes = serde_json::to_vec(&doc).unwrap();
        assert_eq!(
            parse(&bytes)
                .unwrap()
                .document
                .textures()
                .next()
                .unwrap()
                .source()
                .index(),
            0
        );
        assert!(matches!(
            crate::load_slice(&bytes),
            Err(ImportError::ExternalBuffer(_))
        ));
        let dir = std::env::temp_dir().join(format!(
            "ornis-basis-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("texture.ktx2"), IMAGE).unwrap();
        std::fs::write(dir.join("scene.gltf"), bytes).unwrap();
        let model = crate::load_path(&dir.join("scene.gltf")).unwrap();
        assert!(matches!(
            model.primitives[0].material.textures.base_color_texture,
            Some(LoadedImage::Ktx2(_))
        ));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn basis_buffer_view_and_normal_slot_import() {
        let mut doc = document();
        doc["buffers"].as_array_mut().unwrap().push(json!({"byteLength": IMAGE.len(), "uri": format!("data:application/octet-stream;base64,{}", encode(IMAGE))}));
        let view = doc["bufferViews"].as_array().unwrap().len();
        doc["bufferViews"]
            .as_array_mut()
            .unwrap()
            .push(json!({"buffer": 1, "byteOffset": 0, "byteLength": IMAGE.len()}));
        doc["images"][0] = json!({"bufferView": view, "mimeType": "image/ktx2"});
        doc["materials"][0]["normalTexture"] = json!({"index": 0});
        let model = crate::load_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
        assert!(matches!(
            model.primitives[0].material.textures.normal_texture,
            Some(LoadedImage::Ktx2(_))
        ));
    }

    #[test]
    fn malformed_extension_and_unrelated_required_extensions_still_fail() {
        for value in [
            json!({}),
            json!({"source": -1}),
            json!({"source": "0"}),
            json!({"source": 99}),
            json!({"source": 4294967296_u64}),
        ] {
            let mut doc = document();
            doc["textures"][0]["extensions"][EXTENSION] = value;
            assert!(crate::load_slice(&serde_json::to_vec(&doc).unwrap()).is_err());
        }
        let mut doc = document();
        doc["extensionsRequired"] = json!([EXTENSION, "UNKNOWN_extension"]);
        assert!(matches!(
            crate::load_slice(&serde_json::to_vec(&doc).unwrap()),
            Err(ImportError::Parse { .. })
        ));
        let mut doc = document();
        doc["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = json!(999);
        assert!(matches!(
            crate::load_slice(&serde_json::to_vec(&doc).unwrap()),
            Err(ImportError::Parse { .. })
        ));
        let mut doc = document();
        // A valid PNG is still invalid as the Basis extension's source.
        doc["images"][0]["uri"] = json!(
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg=="
        );
        assert!(crate::load_slice(&serde_json::to_vec(&doc).unwrap()).is_err());
    }
}
