//! Texture image resolution: glTF images to RGBA8 pixels.
//!
//! Covers the three core image storages (buffer view, `data:` URI, external
//! file — the last only via [`load_path`](crate::load_path), mirroring the
//! buffer mechanics) and the two core mimes (`image/png`, `image/jpeg`).
//! Decoding is the `image` crate with the `png`/`jpeg` features only; output
//! is always RGBA8, so the GPU-upload step needs no format switch.

use std::path::Path;

use gltf::Gltf;
use image::GenericImageView as _;

use crate::base64;
use crate::import::{is_data_uri, reject_remote_uri};
use crate::{ImportError, LoadedImage};

/// Resolves every document image to RGBA8 pixels, in index order.
///
/// Material slots clone their image's pixels (see `texture_image` in
/// `import.rs`); unreferenced images decode too — one straight pass, no
/// use-tracking. `base_dir` gates external URIs exactly like buffers: `None`
/// (i.e. [`load_slice`](crate::load_slice)) rejects them with
/// [`ImportError::ExternalBuffer`]; `Some` (i.e.
/// [`load_path`](crate::load_path)) joins relative URIs under it.
pub(crate) fn resolve_images(
    gltf: &Gltf,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
) -> Result<Vec<LoadedImage>, ImportError> {
    gltf.document
        .images()
        .map(|image| {
            let context = format!("image {}", image.index());
            let bytes = image_bytes(&image, buffers, base_dir, &context)?;
            decode_image(&bytes, &context)
        })
        .collect()
}

/// Raw (still encoded) bytes of one image plus mime gating.
///
/// Buffer views slice the resolved buffers; `data:` URIs decode inline;
/// anything else is an external file under `base_dir`.
fn image_bytes(
    image: &gltf::image::Image<'_>,
    buffers: &[Vec<u8>],
    base_dir: Option<&Path>,
    context: &str,
) -> Result<Vec<u8>, ImportError> {
    match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            check_mime(Some(mime_type), None, context)?;
            let buffer =
                buffers
                    .get(view.buffer().index())
                    .ok_or_else(|| ImportError::InvalidImage {
                        context: format!(
                            "{context}: references missing buffer {}",
                            view.buffer().index()
                        ),
                    })?;
            let end = view.offset().checked_add(view.length()).ok_or_else(|| {
                ImportError::InvalidImage {
                    context: format!("{context}: buffer view range overflows"),
                }
            })?;
            buffer
                .get(view.offset()..end)
                .map(<[u8]>::to_vec)
                .ok_or_else(|| ImportError::InvalidImage {
                    context: format!(
                        "{context}: buffer view [{}..{end}) out of {} bytes",
                        view.offset(),
                        buffer.len()
                    ),
                })
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            if is_data_uri(uri) {
                let mime = mime_type.or_else(|| data_uri_mime(uri));
                check_mime(mime, None, context)?;
                let payload = uri.split_once(',').map_or("", |(_, after)| after);
                base64::decode(payload).map_err(|_| ImportError::InvalidImage {
                    context: format!("{context}: undecodable image data URI"),
                })
            } else {
                let base = base_dir
                    .ok_or_else(|| ImportError::ExternalBuffer(std::path::PathBuf::from(uri)))?;
                reject_remote_uri(uri)?;
                let extension = Path::new(uri).extension().and_then(|stem| stem.to_str());
                check_mime(mime_type, extension, context)?;
                Ok(std::fs::read(base.join(uri))?)
            }
        }
    }
}

/// Mime from a `data:<mime>;base64,...` header (`None` for `data:;base64,...`).
fn data_uri_mime(uri: &str) -> Option<&str> {
    let head = uri.split_once(',')?.0;
    let mime = head.strip_prefix("data:")?.split(';').next()?;
    (!mime.is_empty()).then_some(mime)
}

/// Accepts the two core glTF mimes; anything else fails before the decoder.
///
/// A missing mime falls back to the file extension (typeless external files);
/// a missing extension passes through and lets the decoder sniff — its
/// rejection surfaces as [`ImportError::InvalidImage`].
fn check_mime(
    mime: Option<&str>,
    extension: Option<&str>,
    context: &str,
) -> Result<(), ImportError> {
    match mime {
        Some("image/png") | Some("image/jpeg") => Ok(()),
        Some(other) => Err(ImportError::UnsupportedImage {
            context: format!("{context}: mime '{other}'"),
        }),
        None => match extension {
            None => Ok(()),
            Some("png" | "jpg" | "jpeg") => Ok(()),
            Some(other) => Err(ImportError::UnsupportedImage {
                context: format!("{context}: extension '.{other}'"),
            }),
        },
    }
}

/// Sniffs PNG/JPEG bytes into RGBA8 (`to_rgba8` gives JPEG opaque alpha).
fn decode_image(bytes: &[u8], context: &str) -> Result<LoadedImage, ImportError> {
    if bytes.is_empty() {
        return Err(ImportError::InvalidImage {
            context: format!("{context}: empty"),
        });
    }
    let decoded = image::load_from_memory(bytes).map_err(|_| ImportError::InvalidImage {
        context: format!("{context}: undecodable ({} bytes)", bytes.len()),
    })?;
    let (width, height) = decoded.dimensions();
    Ok(LoadedImage {
        width,
        height,
        pixels: decoded.to_rgba8().into_raw(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        FixtureEncoding, FixtureImage, FixtureImageStorage, build_glb, build_glb_with_files,
        load_triangle, triangle,
    };
    use crate::{TextureRole, load_path, load_slice};

    #[test]
    fn mime_gate_accepts_core_pair_and_extension_fallback() {
        assert!(check_mime(Some("image/png"), None, "ctx").is_ok());
        assert!(check_mime(Some("image/jpeg"), None, "ctx").is_ok());
        assert!(matches!(
            check_mime(Some("image/webp"), None, "ctx"),
            Err(ImportError::UnsupportedImage { .. })
        ));
        assert!(check_mime(None, Some("png"), "ctx").is_ok());
        assert!(check_mime(None, Some("jpg"), "ctx").is_ok());
        assert!(check_mime(None, Some("jpeg"), "ctx").is_ok());
        assert!(matches!(
            check_mime(None, Some("webp"), "ctx"),
            Err(ImportError::UnsupportedImage { .. })
        ));
        // Unknown entirely: the decoder sniffs, so the gate passes.
        assert!(check_mime(None, None, "ctx").is_ok());
    }

    #[test]
    fn data_uri_mime_parses_header() {
        assert_eq!(
            data_uri_mime("data:image/png;base64,AAAA"),
            Some("image/png")
        );
        assert_eq!(
            data_uri_mime("data:image/jpeg;base64,AAAA"),
            Some("image/jpeg")
        );
        assert_eq!(data_uri_mime("data:;base64,AAAA"), None);
        assert_eq!(data_uri_mime("not-a-uri"), None);
    }

    #[test]
    fn decode_rejects_garbage_and_empty() {
        assert!(matches!(
            decode_image(b"definitely not png or jpeg", "image 0"),
            Err(ImportError::InvalidImage { .. })
        ));
        assert!(matches!(
            decode_image(&[], "image 0"),
            Err(ImportError::InvalidImage { .. })
        ));
        // Truncated PNG signature: sniffs as PNG, then fails.
        assert!(matches!(
            decode_image(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A], "image 0"),
            Err(ImportError::InvalidImage { .. })
        ));
    }

    /// 2×2 RGBA pixels pushed through a fixture image builder.
    fn fixture_image(
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        encoding: FixtureEncoding,
        storage: FixtureImageStorage,
    ) -> FixtureImage {
        FixtureImage {
            width,
            height,
            rgba,
            encoding,
            storage,
            mime_override: None,
        }
    }

    #[test]
    fn base_color_png_data_uri_decodes_exact() {
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255, //
            255, 255, 255, 128, // alpha survives PNG, unlike JPEG
        ];
        let mut fixture = triangle();
        fixture.images.push(fixture_image(
            2,
            2,
            rgba.clone(),
            FixtureEncoding::Png,
            FixtureImageStorage::DataUri,
        ));
        fixture.base_color_texture = Some(0);
        let scene = load_slice(&build_glb(&fixture)).expect("textured parses");
        let material = &scene.entities[0].material;
        let image = material
            .base_color_texture
            .as_ref()
            .expect("base color bound");
        assert_eq!(image.width, 2);
        assert_eq!(image.height, 2);
        assert_eq!(image.pixels, rgba);
        assert!(material.metallic_roughness_texture.is_none());
        assert!(material.emissive_texture.is_none());
        assert_eq!(
            material.texture(TextureRole::BaseColor).map(|i| i.width),
            Some(2)
        );
        assert!(material.texture(TextureRole::Emissive).is_none());
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn emissive_png_buffer_view_decodes_exact() {
        // 3×1 strip: exercises multi-pixel rows plus buffer-view slicing.
        let rgba: Vec<u8> = vec![
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255,
        ];
        let mut fixture = triangle();
        fixture.images.push(fixture_image(
            3,
            1,
            rgba.clone(),
            FixtureEncoding::Png,
            FixtureImageStorage::BufferView,
        ));
        fixture.emissive_texture = Some(0);
        let scene = load_slice(&build_glb(&fixture)).expect("textured parses");
        let material = &scene.entities[0].material;
        let image = material.emissive_texture.as_ref().expect("emissive bound");
        assert_eq!((image.width, image.height), (3, 1));
        assert_eq!(image.pixels, rgba);
        assert_eq!(
            material.texture(TextureRole::Emissive).map(|i| i.height),
            Some(1)
        );
        assert!(material.base_color_texture.is_none());
    }

    #[test]
    fn metallic_roughness_jpeg_decodes_with_opaque_alpha() {
        // 4×4 solid green; JPEG is lossy, so channels get tolerance, not
        // equality — alpha must still decode fully opaque.
        let rgba = [[0u8, 255, 0, 255]; 16].concat();
        let mut fixture = triangle();
        fixture.images.push(fixture_image(
            4,
            4,
            rgba,
            FixtureEncoding::Jpeg,
            FixtureImageStorage::BufferView,
        ));
        fixture.metallic_roughness_texture = Some(0);
        let scene = load_slice(&build_glb(&fixture)).expect("textured parses");
        let material = &scene.entities[0].material;
        let image = material
            .metallic_roughness_texture
            .as_ref()
            .expect("metallic-roughness bound");
        assert_eq!((image.width, image.height), (4, 4));
        assert_eq!(image.pixels.len(), 4 * 4 * 4);
        for pixel in image.pixels.chunks_exact(4) {
            assert_eq!(pixel[3], 255, "jpeg alpha is opaque");
            assert!(pixel[1] > 200, "green stays dominant: {pixel:?}");
            assert!(
                pixel[0] < 80 && pixel[2] < 80,
                "red/blue stay near zero: {pixel:?}"
            );
        }
        assert_eq!(
            material
                .texture(TextureRole::MetallicRoughness)
                .map(|i| i.pixels.len()),
            Some(64)
        );
    }

    #[test]
    fn shared_image_across_roles_clones_pixels() {
        let rgba: Vec<u8> = vec![10, 20, 30, 40];
        let mut fixture = triangle();
        fixture.images.push(fixture_image(
            1,
            1,
            rgba.clone(),
            FixtureEncoding::Png,
            FixtureImageStorage::DataUri,
        ));
        fixture.base_color_texture = Some(0);
        fixture.emissive_texture = Some(0);
        let scene = load_slice(&build_glb(&fixture)).expect("textured parses");
        let material = &scene.entities[0].material;
        assert_eq!(
            material.base_color_texture.as_ref().map(|i| &i.pixels),
            Some(&rgba)
        );
        assert_eq!(
            material.emissive_texture.as_ref().map(|i| &i.pixels),
            Some(&rgba)
        );
    }

    #[test]
    fn external_png_resolves_via_load_path() {
        let rgba: Vec<u8> = vec![
            1, 2, 3, 255, //
            4, 5, 6, 255, //
            7, 8, 9, 255, //
            10, 11, 12, 255,
        ];
        let mut fixture = triangle();
        fixture.images.push(fixture_image(
            2,
            2,
            rgba.clone(),
            FixtureEncoding::Png,
            FixtureImageStorage::External("tex.png".to_string()),
        ));
        fixture.base_color_texture = Some(0);
        let (glb, files) = build_glb_with_files(&fixture);
        // No filesystem on this path: external image URIs fail like buffers.
        assert!(matches!(
            load_slice(&glb),
            Err(ImportError::ExternalBuffer(_))
        ));
        let dir = std::env::temp_dir().join(format!(
            "ornis-gltf-tex-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        std::fs::write(dir.join("tri.glb"), &glb).expect("write glb");
        for (name, bytes) in &files {
            std::fs::write(dir.join(name), bytes).expect("write sibling image");
        }
        let scene = load_path(&dir.join("tri.glb")).expect("external image loads");
        let image = scene.entities[0]
            .material
            .base_color_texture
            .as_ref()
            .expect("base color bound");
        assert_eq!((image.width, image.height), (2, 2));
        assert_eq!(image.pixels, rgba);
        std::fs::remove_dir_all(&dir).expect("temp cleanup");
    }

    #[test]
    fn unsupported_mime_errors_before_decode() {
        let mut fixture = triangle();
        let mut image = fixture_image(
            1,
            1,
            vec![9, 9, 9, 255],
            FixtureEncoding::Png,
            FixtureImageStorage::BufferView,
        );
        image.mime_override = Some("image/webp".to_string());
        fixture.images.push(image);
        fixture.base_color_texture = Some(0);
        assert!(matches!(
            load_slice(&build_glb(&fixture)),
            Err(ImportError::UnsupportedImage { .. })
        ));
    }

    #[test]
    fn untextured_material_has_no_images() {
        let scene = load_triangle();
        let material = &scene.entities[0].material;
        assert!(material.base_color_texture.is_none());
        assert!(material.metallic_roughness_texture.is_none());
        assert!(material.emissive_texture.is_none());
        assert!(material.texture(TextureRole::BaseColor).is_none());
        assert!(material.texture(TextureRole::MetallicRoughness).is_none());
        assert!(material.texture(TextureRole::Emissive).is_none());
    }
}
