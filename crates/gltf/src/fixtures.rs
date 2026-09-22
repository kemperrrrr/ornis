//! In-code fixtures: minimal single-mesh `.glb` / `.gltf` documents.
//!
//! Everything is assembled from parts here — no binary blobs in the repo.
//! Images included: fixture pixels are generated in code and encoded with
//! the same `image` crate the loader decodes with.
//!
//! Test-only: declared under `#[cfg(test)]` in the crate root.

use super::LoadedScene;
use super::import::load_slice;

/// Built document: JSON, raw buffer bytes, plus sibling files for
/// [`FixtureImageStorage::External`] images.
type DocumentParts = (String, Vec<u8>, Vec<(String, Vec<u8>)>);

/// Index storage of the fixture primitive (`None` = unindexed soup).
#[derive(Debug, Clone)]
pub(crate) enum FixtureIndices {
    /// `u16` index list.
    U16(Vec<u16>),
    /// `u32` index list.
    U32(Vec<u32>),
    /// `u8` index list.
    U8(Vec<u8>),
    /// No index accessor; loader must sequence `0..n`.
    Sequential,
}

/// Transform + attachment of one fixture node.
#[derive(Debug, Clone)]
pub(crate) struct FixtureNode {
    /// `name` field (`None` = unnamed).
    pub(crate) name: Option<String>,
    /// Whether mesh `0` attaches here.
    pub(crate) mesh: bool,
    /// TRS fields (`None` = glTF default).
    pub(crate) translation: Option<[f32; 3]>,
    /// Unit quaternion `(x, y, z, w)` (`None` = identity).
    pub(crate) rotation: Option<[f32; 4]>,
    /// Per-axis scale (`None` = one).
    pub(crate) scale: Option<[f32; 3]>,
    /// Flat column-major matrix; exclusive with the TRS fields above.
    pub(crate) matrix: Option<[f32; 16]>,
    /// Child node indices.
    pub(crate) children: Vec<usize>,
}

/// Scalar material factors of the fixture primitive.
#[derive(Debug, Clone)]
pub(crate) struct FixtureMaterial {
    /// `baseColorFactor` RGBA.
    pub(crate) base_color: [f32; 4],
    /// `metallicFactor`.
    pub(crate) metallic: f32,
    /// `roughnessFactor`.
    pub(crate) roughness: f32,
    /// `emissiveFactor` RGB.
    pub(crate) emission: [f32; 3],
}

/// Pixel encoding of one fixture image.
#[derive(Debug, Clone)]
pub(crate) enum FixtureEncoding {
    /// Lossless RGBA; decodes bit-exact.
    Png,
    /// Lossy RGB (alpha decodes opaque); assert with tolerance.
    Jpeg,
}

/// Where the encoded bytes of one fixture image live.
#[derive(Debug, Clone)]
pub(crate) enum FixtureImageStorage {
    /// Appended to the asset buffer, referenced by `bufferView` (+`mimeType`).
    BufferView,
    /// Inline `data:<mime>;base64,...` URI (no `mimeType` field — the header
    /// carries it).
    DataUri,
    /// Sibling file for [`load_path`](super::import::load_path) tests; the
    /// `(filename, bytes)` pairs come out of [`build_glb_with_files`], and
    /// [`load_slice`](super::import::load_slice) rejects these documents.
    External(String),
}

/// One image exercised by the texture tests: generated pixels, no blobs.
#[derive(Debug, Clone)]
pub(crate) struct FixtureImage {
    /// Width in pixels.
    pub(crate) width: u32,
    /// Height in pixels.
    pub(crate) height: u32,
    /// RGBA source pixels, row-major (JPEG encoding keeps RGB only).
    pub(crate) rgba: Vec<u8>,
    /// Lossless or lossy encoding.
    pub(crate) encoding: FixtureEncoding,
    /// Buffer view, data URI, or sibling file.
    pub(crate) storage: FixtureImageStorage,
    /// Overrides the emitted `mimeType`/`data:`-URI header (error-path tests).
    pub(crate) mime_override: Option<String>,
}

/// One primitive + node tree assembled to bytes by the builders below.
#[derive(Debug, Clone)]
pub(crate) struct Fixture {
    /// `POSITION` data (may be empty for the empty-soup case).
    pub(crate) positions: Vec<[f32; 3]>,
    /// Index storage.
    pub(crate) indices: FixtureIndices,
    /// `NORMAL` data (`None` = attribute absent).
    pub(crate) normals: Option<Vec<[f32; 3]>>,
    /// `TEXCOORD_0` data (`None` = attribute absent).
    pub(crate) uvs: Option<Vec<[f32; 2]>>,
    /// glTF primitive `mode` number (`None` = default triangles).
    pub(crate) mode: Option<u32>,
    /// Adds `JOINTS_0` + `WEIGHTS_0` (skin-skip case).
    pub(crate) skinned: bool,
    /// Node tree; scene roots at node `0`.
    pub(crate) nodes: Vec<FixtureNode>,
    /// Mesh `name` (`None` = unnamed).
    pub(crate) mesh_name: Option<String>,
    /// Scene `name` (`None` = unnamed).
    pub(crate) scene_name: Option<String>,
    /// Scene root node indices (default `vec![0]`).
    pub(crate) roots: Vec<usize>,
    /// Material (`None` = default material; auto-created when any texture
    /// slot below is set).
    pub(crate) material: Option<FixtureMaterial>,
    /// Fixture images (empty = untextured document).
    pub(crate) images: Vec<FixtureImage>,
    /// `baseColorTexture` source image (`None` = slot unbound).
    pub(crate) base_color_texture: Option<usize>,
    /// `metallicRoughnessTexture` source image (`None` = slot unbound).
    pub(crate) metallic_roughness_texture: Option<usize>,
    /// `emissiveTexture` source image (`None` = slot unbound).
    pub(crate) emissive_texture: Option<usize>,
}

/// Default single triangle in the XY plane (`+Z` face normal).
pub(crate) fn triangle() -> Fixture {
    Fixture {
        positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        indices: FixtureIndices::U16(vec![0, 1, 2]),
        normals: Some(vec![[0.0, 0.0, 1.0]; 3]),
        uvs: Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        mode: None,
        skinned: false,
        nodes: vec![FixtureNode {
            name: Some("tri-node".to_string()),
            mesh: true,
            translation: Some([1.0, 2.0, 3.0]),
            rotation: None,
            scale: None,
            matrix: None,
            children: Vec::new(),
        }],
        mesh_name: Some("tri-mesh".to_string()),
        scene_name: Some("tri-scene".to_string()),
        roots: vec![0],
        material: None,
        images: Vec::new(),
        base_color_texture: None,
        metallic_roughness_texture: None,
        emissive_texture: None,
    }
}

/// Parses the default triangle fixture straight from generated `.glb` bytes.
pub(crate) fn load_triangle() -> LoadedScene {
    load_slice(&build_glb(&triangle())).expect("fixture parses")
}

/// Assembles a `.glb` container (JSON + BIN chunks, 4-byte aligned).
pub(crate) fn build_glb(fixture: &Fixture) -> Vec<u8> {
    let (json, bin) = build_parts(fixture, None);
    assemble_glb(&json, &bin)
}

/// Assembles a `.glb` plus sibling files for external-image tests.
///
/// The second element holds `(filename, bytes)` pairs to write next to the
/// `.glb` before [`load_path`](super::import::load_path).
pub(crate) fn build_glb_with_files(fixture: &Fixture) -> (Vec<u8>, Vec<(String, Vec<u8>)>) {
    let (json, bin, files) = build_document(fixture, None);
    (assemble_glb(&json, &bin), files)
}

/// Packs one JSON/BIN pair into a `.glb` container (4-byte aligned chunks).
fn assemble_glb(json: &str, bin: &[u8]) -> Vec<u8> {
    let json_pad = json.len().next_multiple_of(4) - json.len();
    let bin_pad = bin.len().next_multiple_of(4) - bin.len();
    let total = 12
        + 8
        + json.len()
        + json_pad
        + if bin.is_empty() {
            0
        } else {
            8 + bin.len() + bin_pad
        };
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&((json.len() + json_pad) as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(json.as_bytes());
    out.extend(std::iter::repeat_n(b' ', json_pad));
    if !bin.is_empty() {
        out.extend_from_slice(&((bin.len() + bin_pad) as u32).to_le_bytes());
        out.extend_from_slice(b"BIN\x00");
        out.extend_from_slice(bin);
        out.extend(std::iter::repeat_n(0u8, bin_pad));
    }
    out
}

/// Assembles a `.gltf` JSON document with the buffer as a `data:` URI.
pub(crate) fn build_gltf(fixture: &Fixture) -> String {
    let (_, bin) = build_parts(fixture, None);
    let uri = format!("data:application/octet-stream;base64,{}", encode(&bin));
    let (json, _) = build_parts(fixture, Some(uri));
    json
}

/// Builds the default triangle as `.gltf` JSON referencing an external
/// `mesh.bin` file, plus those buffer bytes (for [`load_slice`] rejection
/// and [`load_path`](super::import::load_path) round-trip tests).
pub(crate) fn build_external_parts() -> (String, Vec<u8>) {
    build_parts(&triangle(), Some("mesh.bin".to_string()))
}

/// Builds the document JSON plus the raw buffer bytes.
///
/// `buffer_uri`: `None` → `BIN`-chunk buffer (`.glb`); `Some(uri)` → that URI
/// (`.gltf` with `data:` or external reference).
fn build_parts(fixture: &Fixture, buffer_uri: Option<String>) -> (String, Vec<u8>) {
    let (json, bin, _) = build_document(fixture, buffer_uri);
    (json, bin)
}

/// Builds the document JSON, the raw buffer bytes, plus sibling files.
///
/// The third element holds `(filename, bytes)` pairs for
/// [`FixtureImageStorage::External`] images — empty unless the fixture uses
/// external image storage.
fn build_document(fixture: &Fixture, buffer_uri: Option<String>) -> DocumentParts {
    let mut bin: Vec<u8> = Vec::new();
    // (offset, length) per buffer view, in attribute order.
    let mut views: Vec<(usize, usize)> = Vec::new();
    let push = |bytes: &[u8], bin: &mut Vec<u8>, views: &mut Vec<(usize, usize)>| {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        views.push((offset, bytes.len()));
    };

    let mut raw = Vec::new();
    for position in &fixture.positions {
        for component in position {
            raw.extend_from_slice(&component.to_le_bytes());
        }
    }
    push(&raw, &mut bin, &mut views);
    let positions_view = 0;

    let mut index_view = None;
    let mut index_component = 5123u32;
    let mut index_count = 0;
    match &fixture.indices {
        FixtureIndices::U16(values) => {
            let mut raw = Vec::new();
            for value in values {
                raw.extend_from_slice(&value.to_le_bytes());
            }
            push(&raw, &mut bin, &mut views);
            index_view = Some(views.len() - 1);
            index_component = 5123;
            index_count = values.len();
        }
        FixtureIndices::U32(values) => {
            let mut raw = Vec::new();
            for value in values {
                raw.extend_from_slice(&value.to_le_bytes());
            }
            push(&raw, &mut bin, &mut views);
            index_view = Some(views.len() - 1);
            index_component = 5125;
            index_count = values.len();
        }
        FixtureIndices::U8(values) => {
            push(values, &mut bin, &mut views);
            index_view = Some(views.len() - 1);
            index_component = 5121;
            index_count = values.len();
        }
        FixtureIndices::Sequential => {}
    }

    let mut normal_accessor = None;
    if let Some(normals) = &fixture.normals {
        let mut raw = Vec::new();
        for normal in normals {
            for component in normal {
                raw.extend_from_slice(&component.to_le_bytes());
            }
        }
        push(&raw, &mut bin, &mut views);
        normal_accessor = Some((views.len() - 1, normals.len()));
    }

    let mut uv_accessor = None;
    if let Some(uvs) = &fixture.uvs {
        let mut raw = Vec::new();
        for uv in uvs {
            for component in uv {
                raw.extend_from_slice(&component.to_le_bytes());
            }
        }
        push(&raw, &mut bin, &mut views);
        uv_accessor = Some((views.len() - 1, uvs.len()));
    }

    let mut joint_accessor = None;
    let mut weight_accessor = None;
    if fixture.skinned {
        let joints = vec![[0u8, 0, 0, 0]; fixture.positions.len().max(1)];
        let mut raw = Vec::new();
        for joint in &joints {
            raw.extend_from_slice(joint);
        }
        push(&raw, &mut bin, &mut views);
        joint_accessor = Some((views.len() - 1, joints.len()));
        let mut raw = Vec::new();
        for _ in &joints {
            raw.extend_from_slice(&[1.0f32, 0.0, 0.0, 0.0].map(f32::to_le_bytes).concat());
        }
        push(&raw, &mut bin, &mut views);
        weight_accessor = Some((views.len() - 1, joints.len()));
    }

    // Fixture images: encoded with the same `image` crate the loader uses.
    // Buffer-view images append to the asset buffer; data-URI and external
    // images only surface in the JSON (plus sibling files below).
    let mut images_json_parts: Vec<String> = Vec::new();
    let mut external_files: Vec<(String, Vec<u8>)> = Vec::new();
    for image in &fixture.images {
        let encoded = encode_fixture_image(image);
        let mime = image.mime_override.clone().unwrap_or_else(|| {
            match image.encoding {
                FixtureEncoding::Png => "image/png",
                FixtureEncoding::Jpeg => "image/jpeg",
            }
            .to_string()
        });
        match &image.storage {
            FixtureImageStorage::BufferView => {
                push(&encoded, &mut bin, &mut views);
                images_json_parts.push(format!(
                    "{{\"bufferView\":{},\"mimeType\":\"{mime}\"}}",
                    views.len() - 1
                ));
            }
            FixtureImageStorage::DataUri => {
                let uri = format!("data:{mime};base64,{}", encode(&encoded));
                images_json_parts.push(format!("{{\"uri\":\"{uri}\"}}"));
            }
            FixtureImageStorage::External(name) => {
                external_files.push((name.clone(), encoded));
                images_json_parts.push(format!("{{\"uri\":\"{name}\"}}"));
            }
        }
    }

    // Accessor table; attribute accessors reference entries by index.
    let mut accessors: Vec<String> = Vec::new();
    let positions_accessor = accessors.len();
    accessors.push(accessor_json(
        positions_view,
        5126,
        fixture.positions.len(),
        "VEC3",
    ));
    // Validators require POSITION bounds: tracked while pushing above is
    // overkill, so recompute from the source array (empty → zeros).
    let (mut min, mut max) = ([0.0f32; 3], [0.0f32; 3]);
    for position in &fixture.positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    accessors[positions_accessor] =
        accessor_json_bounds(positions_view, fixture.positions.len(), &min, &max);
    let indices_accessor = index_view.map(|view| {
        let index = accessors.len();
        accessors.push(accessor_json(view, index_component, index_count, "SCALAR"));
        index
    });
    let mut attributes = format!("\"POSITION\":{positions_accessor}");
    if let Some((view, count)) = normal_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, 5126, count, "VEC3"));
        attributes += &format!(",\"NORMAL\":{index}");
    }
    if let Some((view, count)) = uv_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, 5126, count, "VEC2"));
        attributes += &format!(",\"TEXCOORD_0\":{index}");
    }
    if let Some((view, count)) = joint_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, 5121, count, "VEC4"));
        attributes += &format!(",\"JOINTS_0\":{index}");
    }
    if let Some((view, count)) = weight_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, 5126, count, "VEC4"));
        attributes += &format!(",\"WEIGHTS_0\":{index}");
    }

    let views_json = views
        .iter()
        .map(|(offset, length)| {
            format!("{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{length}}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let accessors_json = accessors.join(",");
    let indices_json =
        indices_accessor.map_or(String::new(), |index| format!(",\"indices\":{index}"));
    let mode_json = fixture
        .mode
        .map_or(String::new(), |mode| format!(",\"mode\":{mode}"));
    // Texture slots share one texture table; several slots may point at the
    // same image (each gets its own texture entry, as real exporters emit).
    let mut texture_images: Vec<usize> = Vec::new();
    let slot_texture = |slot: Option<usize>, textures: &mut Vec<usize>| -> Option<usize> {
        slot.map(|image| {
            let index = textures.len();
            textures.push(image);
            index
        })
    };
    let base_texture = slot_texture(fixture.base_color_texture, &mut texture_images);
    let mr_texture = slot_texture(fixture.metallic_roughness_texture, &mut texture_images);
    let emissive_texture = slot_texture(fixture.emissive_texture, &mut texture_images);
    // A textured primitive needs a material even when the fixture sets none.
    let material = fixture.material.clone().or(if texture_images.is_empty() {
        None
    } else {
        Some(FixtureMaterial {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 1.0,
            roughness: 1.0,
            emission: [0.0, 0.0, 0.0],
        })
    });
    let material_ref_json = material
        .as_ref()
        .map_or(String::new(), |_| ",\"material\":0".to_string());
    let primitive_json =
        format!("{{\"attributes\":{{{attributes}}}{indices_json}{mode_json}{material_ref_json}}}");
    let mesh_name_json = fixture
        .mesh_name
        .as_ref()
        .map_or(String::new(), |name| format!(",\"name\":\"{name}\""));
    let materials_json = match &material {
        None => String::new(),
        Some(material) => {
            let base = material
                .base_color
                .map(|component| format!("{component:?}"))
                .join(",");
            let emission = material
                .emission
                .map(|component| format!("{component:?}"))
                .join(",");
            let base_texture_json = base_texture.map_or(String::new(), |index| {
                format!(",\"baseColorTexture\":{{\"index\":{index}}}")
            });
            let mr_texture_json = mr_texture.map_or(String::new(), |index| {
                format!(",\"metallicRoughnessTexture\":{{\"index\":{index}}}")
            });
            let emissive_texture_json = emissive_texture.map_or(String::new(), |index| {
                format!(",\"emissiveTexture\":{{\"index\":{index}}}")
            });
            format!(
                "\"materials\":[{{\"pbrMetallicRoughness\":{{\"baseColorFactor\":[{base}],\
                \"metallicFactor\":{:?},\"roughnessFactor\":{:?}{base_texture_json}\
                {mr_texture_json}}},\"emissiveFactor\":[{emission}]{emissive_texture_json}}}],",
                material.metallic, material.roughness
            )
        }
    };
    let textures_json = if texture_images.is_empty() {
        String::new()
    } else {
        let parts = texture_images
            .iter()
            .map(|image| format!("{{\"source\":{image}}}"))
            .collect::<Vec<_>>()
            .join(",");
        format!("\"textures\":[{parts}],")
    };
    let images_json = if images_json_parts.is_empty() {
        String::new()
    } else {
        format!("\"images\":[{}],", images_json_parts.join(","))
    };
    let nodes_json = fixture
        .nodes
        .iter()
        .map(|node| {
            let mut fields: Vec<String> = Vec::new();
            if let Some(name) = &node.name {
                fields.push(format!("\"name\":\"{name}\""));
            }
            if node.mesh {
                fields.push("\"mesh\":0".to_string());
            }
            if let Some(translation) = node.translation {
                fields.push(format!(
                    "\"translation\":[{:?},{:?},{:?}]",
                    translation[0], translation[1], translation[2]
                ));
            }
            if let Some(rotation) = node.rotation {
                fields.push(format!(
                    "\"rotation\":[{:?},{:?},{:?},{:?}]",
                    rotation[0], rotation[1], rotation[2], rotation[3]
                ));
            }
            if let Some(scale) = node.scale {
                fields.push(format!(
                    "\"scale\":[{:?},{:?},{:?}]",
                    scale[0], scale[1], scale[2]
                ));
            }
            if let Some(matrix) = node.matrix {
                let flat = matrix.map(|component| format!("{component:?}")).join(",");
                fields.push(format!("\"matrix\":[{flat}]"));
            }
            if !node.children.is_empty() {
                let children = node
                    .children
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                fields.push(format!("\"children\":[{children}]"));
            }
            format!("{{{}}}", fields.join(","))
        })
        .collect::<Vec<_>>()
        .join(",");
    let scene_name_json = fixture
        .scene_name
        .as_ref()
        .map_or(String::new(), |name| format!("\"name\":\"{name}\","));
    let roots_json = fixture
        .roots
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let buffer_json = match buffer_uri {
        None => format!("{{\"byteLength\":{}}}", bin.len()),
        Some(uri) => format!("{{\"byteLength\":{},\"uri\":\"{uri}\"}}", bin.len()),
    };
    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"ornis-gltf-fixture\"}},\
        \"scene\":0,\"scenes\":[{{{scene_name_json}\"nodes\":[{roots_json}]}}],\
        \"nodes\":[{nodes_json}],\
        \"meshes\":[{{\"primitives\":[{primitive_json}]{mesh_name_json}}}],\
        {materials_json}{textures_json}{images_json}\"buffers\":[{buffer_json}],\
        \"bufferViews\":[{views_json}],\"accessors\":[{accessors_json}]}}"
    );
    (json, bin, external_files)
}

/// Encodes one fixture image with the same `image` decoders the loader uses.
fn encode_fixture_image(image: &FixtureImage) -> Vec<u8> {
    use image::ImageEncoder as _;
    let mut out = Vec::new();
    match image.encoding {
        FixtureEncoding::Png => image::codecs::png::PngEncoder::new(&mut out)
            .write_image(
                &image.rgba,
                image.width,
                image.height,
                image::ExtendedColorType::Rgba8,
            )
            .expect("fixture png encodes"),
        FixtureEncoding::Jpeg => {
            let rgb: Vec<u8> = image
                .rgba
                .chunks_exact(4)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .collect();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 95)
                .write_image(
                    &rgb,
                    image.width,
                    image.height,
                    image::ExtendedColorType::Rgb8,
                )
                .expect("fixture jpeg encodes")
        }
    }
    out
}

/// One accessor entry: view, component type, count, data type.
fn accessor_json(view: usize, component: u32, count: usize, kind: &str) -> String {
    format!(
        "{{\"bufferView\":{view},\"componentType\":{component},\"count\":{count},\"type\":\"{kind}\"}}"
    )
}

/// POSITION accessor with validator-required `min`/`max` bounds.
fn accessor_json_bounds(view: usize, count: usize, min: &[f32; 3], max: &[f32; 3]) -> String {
    format!(
        "{{\"bufferView\":{view},\"componentType\":5126,\"count\":{count},\"type\":\"VEC3\",\
        \"min\":[{:?},{:?},{:?}],\"max\":[{:?},{:?},{:?}]}}",
        min[0], min[1], min[2], max[0], max[1], max[2]
    )
}

/// Test-only standard base64 encoder (mirrors `crate::base64` alphabet).
pub(crate) fn encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len() / 3 * 4 + 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .fold(0u32, |acc, &byte| (acc << 8) | u32::from(byte));
        let shift = (3 - chunk.len()) * 8;
        let n = n << shift;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[((n >> 6) & 63) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(n & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
