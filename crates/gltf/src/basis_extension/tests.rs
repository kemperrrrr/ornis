//! Source-selection, validation and storage regression tests for Basis glTF import.

use super::*;
use crate::LoadedImage;
use crate::fixtures::{build_gltf, encode, triangle};
use serde_json::{Value, json};

const IMAGE: &[u8] = include_bytes!("../../../texture/tests/data/uastc.ktx2");

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
    let fallback = png_source();
    doc["images"].as_array_mut().unwrap().push(fallback);
    doc["textures"][0]["source"] = json!(1);
    doc["images"][0] = json!({"uri": "texture.ktx2", "mimeType": "image/ktx2"});
    let bytes = serde_json::to_vec(&doc).unwrap();
    assert_eq!(
        parse(&bytes)
            .unwrap()
            .gltf
            .document
            .textures()
            .next()
            .unwrap()
            .source()
            .index(),
        0
    );
    assert!(matches!(selected(&doc).unwrap(), LoadedImage::Rgba8(_)));
    let mut required = doc.clone();
    required["extensionsRequired"] = json!([EXTENSION]);
    assert!(matches!(
        selected(&required),
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
    // A missing optional file also selects the embedded PNG under load_path.
    std::fs::write(dir.join("scene.gltf"), &bytes).unwrap();
    let fallback_model = crate::load_path(&dir.join("scene.gltf")).unwrap();
    assert!(
        fallback_model.primitives[0]
            .material
            .texture(crate::TextureRole::BaseColor)
            .unwrap()
            .rgba8()
            .is_some()
    );
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

fn png_source() -> Value {
    let mut output = std::io::Cursor::new(Vec::new());
    let image = image::RgbaImage::from_pixel(1, 1, image::Rgba([17, 83, 191, 255]));
    image
        .write_to(&mut output, image::ImageFormat::Png)
        .unwrap();
    json!({"uri": format!("data:image/png;base64,{}", encode(output.get_ref()))})
}

fn with_png_fallback() -> Value {
    let mut doc = document();
    doc["images"].as_array_mut().unwrap().push(png_source());
    doc["textures"][0]["source"] = json!(1);
    doc
}

fn selected(doc: &Value) -> Result<LoadedImage, ImportError> {
    let model = crate::load_slice(&serde_json::to_vec(doc).unwrap())?;
    Ok(model.primitives[0]
        .material
        .texture(crate::TextureRole::BaseColor)
        .unwrap()
        .clone())
}

#[test]
fn optional_valid_basis_beats_png_and_never_decodes_unused_images() {
    let mut doc = with_png_fallback();
    doc.as_object_mut().unwrap().remove("extensionsRequired");
    assert!(matches!(selected(&doc).unwrap(), LoadedImage::Ktx2(_)));
    doc["images"][1] = json!({"uri": "missing.png"});
    doc["images"]
        .as_array_mut()
        .unwrap()
        .push(json!({"uri": "data:image/png;base64,AAAA"}));
    assert!(matches!(selected(&doc).unwrap(), LoadedImage::Ktx2(_)));
}

#[test]
fn optional_invalid_or_unsupported_basis_uses_png_but_required_returns_error() {
    let mut array = IMAGE.to_vec();
    array[32..36].copy_from_slice(&1_u32.to_le_bytes());
    let mut hdr = IMAGE.to_vec();
    hdr[12..16].copy_from_slice(&97_u32.to_le_bytes()); // VK_FORMAT_R16G16B16A16_SFLOAT
    let mut native_zstd = IMAGE.to_vec();
    native_zstd[12..16].copy_from_slice(&145_u32.to_le_bytes()); // BC7
    native_zstd[44..48].copy_from_slice(&2_u32.to_le_bytes());
    for bytes in [
        array,
        hdr,
        native_zstd,
        IMAGE[..90].to_vec(),
        b"broken KTX2".to_vec(),
    ] {
        let mut doc = with_png_fallback();
        doc["images"][0] = json!({"uri": format!("data:image/ktx2;base64,{}", encode(&bytes))});
        assert!(matches!(
            selected(&doc),
            Err(ImportError::InvalidImage { .. })
        ));
        doc.as_object_mut().unwrap().remove("extensionsRequired");
        let image = selected(&doc).unwrap();
        assert_eq!(image.rgba8().unwrap().pixels, [17, 83, 191, 255]);
        doc["textures"][0].as_object_mut().unwrap().remove("source");
        assert!(
            selected(&doc).is_err(),
            "without fallback the error must survive"
        );
    }
}

#[test]
fn source_selection_is_per_texture_not_per_image() {
    let mut doc = with_png_fallback();
    doc.as_object_mut().unwrap().remove("extensionsRequired");
    doc["images"][0] = json!({"uri": "data:image/ktx2;base64,AAAA"});
    // Core and Basis texture slots deliberately reference the same fallback.
    doc["textures"]
        .as_array_mut()
        .unwrap()
        .push(json!({"source": 1}));
    doc["materials"][0]["normalTexture"] = json!({"index": 1});
    let model = crate::load_slice(&serde_json::to_vec(&doc).unwrap()).unwrap();
    let material = &model.primitives[0].material;
    for role in [crate::TextureRole::BaseColor, crate::TextureRole::Normal] {
        assert_eq!(
            material.texture(role).unwrap().rgba8().unwrap().pixels,
            [17, 83, 191, 255]
        );
    }
}

#[test]
fn extensions_used_and_mime_are_enforced_and_core_ktx_is_rejected() {
    let mut doc = document();
    doc.as_object_mut().unwrap().remove("extensionsUsed");
    assert!(matches!(selected(&doc), Err(ImportError::Parse { .. })));
    let mut doc = document();
    doc["images"][0]["mimeType"] = json!("image/png");
    assert!(matches!(
        selected(&doc),
        Err(ImportError::InvalidImage { .. })
    ));
    let mut doc = document();
    doc["textures"][0] = json!({"source": 0});
    assert!(matches!(
        selected(&doc),
        Err(ImportError::UnsupportedImage { .. })
    ));
}

#[test]
fn unsafe_ktx_paths_do_not_trigger_optional_fallback() {
    let dir = std::env::temp_dir().join(format!("ornis-basis-paths-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for uri in ["../x.ktx2", "/x.ktx2"] {
        let mut doc = with_png_fallback();
        doc.as_object_mut().unwrap().remove("extensionsRequired");
        doc["images"][0] = json!({"uri": uri, "mimeType": "image/ktx2"});
        let path = dir.join("scene.gltf");
        std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).unwrap();
        assert!(matches!(
            crate::load_path(&path),
            Err(ImportError::UnsafeUri { .. })
        ));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn other_core_indices_still_go_through_gltf_validation() {
    let changes = [
        ("/meshes/0/primitives/0/indices", json!(999)),
        ("/meshes/0/primitives/0/material", json!(999)),
        ("/bufferViews/0/buffer", json!(999)),
        ("/textures/0/source", json!(999)),
        ("/textures/0/sampler", json!(999)),
        ("/nodes/0/skin", json!(999)),
        ("/nodes/0/children", json!([999])),
        ("/accessors/0/bufferView", json!(999)),
        (
            "/accessors/0/sparse",
            json!({"count":1,"indices":{"bufferView":999,"componentType":5123},"values":{"bufferView":999}}),
        ),
    ];
    for (pointer, value) in changes {
        let mut doc = with_png_fallback();
        let (parent, field) = pointer.rsplit_once('/').unwrap();
        doc.pointer_mut(parent).unwrap()[field] = value;
        assert!(
            crate::load_slice(&serde_json::to_vec(&doc).unwrap()).is_err(),
            "{pointer}"
        );
    }
    let mut doc = document();
    doc["animations"] = json!([{"channels":[{"sampler":999,"target":{"node":0,"path":"translation"}}],"samplers":[{"input":0,"output":0}]}]);
    assert!(crate::load_slice(&serde_json::to_vec(&doc).unwrap()).is_err());
}
