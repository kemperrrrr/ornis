//! In-code fixtures: minimal single-mesh `.glb` / `.gltf` documents.
//!
//! Everything is assembled from parts here — no binary blobs in the repo.
//! Images included: fixture pixels are generated in code and encoded with
//! the same `image` crate the loader decodes with.
//!
//! Test-only: declared under `#[cfg(test)]` in the crate root.

use super::Model;
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
    /// `skin` index (`None` = unskinned node).
    pub(crate) skin: Option<usize>,
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

/// Storage width of the `WEIGHTS_0` fixture attribute.
#[derive(Debug, Clone, Copy)]
pub(crate) enum FixtureWeightsKind {
    /// `f32` weights, verbatim.
    F32,
    /// `u8` weights, scaled by `255` (decoder normalizes back).
    U8,
    /// `u16` weights, scaled by `65535` (decoder normalizes back).
    U16,
}

/// Per-vertex skin influences of the fixture primitive.
#[derive(Debug, Clone)]
pub(crate) struct FixtureInfluence {
    /// `JOINTS_0` values, one entry per vertex.
    pub(crate) joints: Vec<[u16; 4]>,
    /// Emit `JOINTS_0` as `u16` (`false` = `u8`).
    pub(crate) joints_u16: bool,
    /// `WEIGHTS_0` values, one entry per vertex.
    pub(crate) weights: Vec<[f32; 4]>,
    /// Storage width of `WEIGHTS_0`.
    pub(crate) weights_kind: FixtureWeightsKind,
    /// Optional extra set (`JOINTS_1`/`WEIGHTS_1`, always `u16`/`f32`) for
    /// the top-4 truncation case.
    pub(crate) extra: Option<FixtureInfluenceSet>,
}

/// One dense skin-influence set: per-vertex joints + weights.
pub(crate) type FixtureInfluenceSet = (Vec<[u16; 4]>, Vec<[f32; 4]>);

/// One entry of the fixture `skins` array.
#[derive(Debug, Clone)]
pub(crate) struct FixtureSkin {
    /// `joints` node indices, in skin order.
    pub(crate) joints: Vec<usize>,
    /// `inverseBindMatrices` row data (`None` = accessor absent → identity).
    pub(crate) inverse_bind: Option<Vec<[f32; 16]>>,
    /// Skin `name` (`None` = unnamed).
    pub(crate) name: Option<String>,
    /// Skeleton root node index (`None` = field absent).
    pub(crate) skeleton: Option<usize>,
}
/// Sampler interpolation of the dummy animation clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixtureInterp {
    /// `LINEAR` (assembles to a linear track).
    Linear,
    /// `STEP` (assembles to a stepped track).
    Step,
    /// `CUBICSPLINE` (skips honestly with a counter).
    CubicSpline,
}

impl FixtureInterp {
    /// glTF sampler `interpolation` name.
    fn as_str(self) -> &'static str {
        match self {
            Self::Linear => "LINEAR",
            Self::Step => "STEP",
            Self::CubicSpline => "CUBICSPLINE",
        }
    }
}

/// Target path of the dummy animation clips.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FixtureAnimPath {
    /// `translation` (`VEC3` outputs).
    Translation,
    /// `rotation` (`VEC4` unit-quaternion outputs).
    Rotation,
    /// `scale` (`VEC3` outputs).
    Scale,
}

impl FixtureAnimPath {
    /// glTF channel `target.path` name.
    fn as_str(self) -> &'static str {
        match self {
            Self::Translation => "translation",
            Self::Rotation => "rotation",
            Self::Scale => "scale",
        }
    }
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
    /// Skin influences (`JOINTS_0`/`WEIGHTS_0`, `None` = unskinned primitive).
    pub(crate) influence: Option<FixtureInfluence>,
    /// `skins` array; mesh nodes link by [`FixtureNode::skin`].
    pub(crate) skins: Vec<FixtureSkin>,
    /// Dummy animations targeting node 0 (clip-assembly cases).
    pub(crate) animations: usize,
    /// Sampler interpolation of the dummy animation clips.
    pub(crate) anim_interp: FixtureInterp,
    /// Overrides the emitted output accessor `count` (error-path tests: a
    /// short count reads fewer entries than pushed, so the loader must skip
    /// honestly on the count mismatch).
    pub(crate) anim_output_len_override: Option<usize>,
    /// Target path of the dummy animation clips.
    pub(crate) anim_path: FixtureAnimPath,
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
        influence: None,
        skins: Vec::new(),
        animations: 0,
        anim_interp: FixtureInterp::Linear,
        anim_output_len_override: None,
        anim_path: FixtureAnimPath::Translation,
        nodes: vec![FixtureNode {
            name: Some("tri-node".to_string()),
            mesh: true,
            skin: None,
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
pub(crate) fn load_triangle() -> Model {
    load_slice(&build_glb(&triangle())).expect("fixture parses")
}

/// Single-joint skinned triangle: mesh node doubles as the only joint
/// (`parents == [-1]`, identity bind), full weight on joint 0.
pub(crate) fn skinned_triangle() -> Fixture {
    let mut fixture = triangle();
    fixture.nodes[0].skin = Some(0);
    fixture.influence = Some(FixtureInfluence {
        joints: vec![[0, 0, 0, 0]; 3],
        joints_u16: false,
        weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
        weights_kind: FixtureWeightsKind::F32,
        extra: None,
    });
    fixture.skins = vec![FixtureSkin {
        joints: vec![0],
        inverse_bind: None,
        name: Some("skin".to_string()),
        skeleton: None,
    }];
    fixture
}

/// One channel of a multi-channel fixture animation (mixed-interpolation
/// clip case): target path, sampler interpolation, and input times.
#[derive(Debug, Clone)]
pub(crate) struct FixtureChannel {
    /// Target node index.
    pub(crate) node: usize,
    /// Channel target path.
    pub(crate) path: FixtureAnimPath,
    /// Sampler interpolation.
    pub(crate) interp: FixtureInterp,
    /// Input times in seconds (one per key).
    pub(crate) times: Vec<f32>,
}

/// Output values for one animation channel with `keys` input times: one
/// entry per key for `LINEAR`/`STEP`, tripled (in-tangent, value,
/// out-tangent) per key for `CUBICSPLINE`.
///
/// Values step `+X` per key (translation), grow uniformly (scale), or hold
/// identity (rotation); cubic tangents are distinctive per key and slot
/// (`10 + key` in, `20 + key` out) so tests can prove slot placement —
/// every entry of a channel is pairwise distinct.
fn anim_output_values(path: FixtureAnimPath, interp: FixtureInterp, keys: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for key in 0..keys {
        let value: Vec<f32> = match path {
            FixtureAnimPath::Translation => vec![key as f32, 0.0, 0.0],
            FixtureAnimPath::Scale => vec![1.0 + key as f32, 1.0 + key as f32, 1.0 + key as f32],
            FixtureAnimPath::Rotation => vec![0.0, 0.0, 0.0, 1.0],
        };
        if interp == FixtureInterp::CubicSpline {
            let (in_tangent, out_tangent) = match path {
                FixtureAnimPath::Rotation => (
                    vec![10.0 + key as f32, 0.0, 0.0, 0.0],
                    vec![20.0 + key as f32, 0.0, 0.0, 0.0],
                ),
                FixtureAnimPath::Translation | FixtureAnimPath::Scale => (
                    vec![10.0 + key as f32, 0.0, 0.0],
                    vec![20.0 + key as f32, 0.0, 0.0],
                ),
            };
            out.push(in_tangent);
            out.push(value);
            out.push(out_tangent);
        } else {
            out.push(value);
        }
    }
    out
}

/// Builds a `.glb` whose single animation holds one channel per entry in
/// `channels` (mixed-interpolation clip case): the mesh is the default
/// triangle on node 0, each channel gets its own sampler with patterned
/// outputs from [`anim_output_values`].
pub(crate) fn build_mixed_clip_glb(channels: &[FixtureChannel]) -> Vec<u8> {
    let mut bin: Vec<u8> = Vec::new();
    // (offset, length) per buffer view, in push order.
    let mut views: Vec<(usize, usize)> = Vec::new();
    let push = |bytes: &[u8], bin: &mut Vec<u8>, views: &mut Vec<(usize, usize)>| {
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        views.push((offset, bytes.len()));
        views.len() - 1
    };

    // Default triangle geometry (positions plus `u16` indices).
    let mut raw = Vec::new();
    for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        for component in position {
            raw.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_view = push(&raw, &mut bin, &mut views);
    let mut raw = Vec::new();
    for index in [0u16, 1, 2] {
        raw.extend_from_slice(&index.to_le_bytes());
    }
    let indices_view = push(&raw, &mut bin, &mut views);

    let mut accessors: Vec<String> = Vec::new();
    accessors.push(accessor_json_bounds(
        positions_view,
        3,
        &[0.0, 0.0, 0.0],
        &[1.0, 1.0, 0.0],
    ));
    accessors.push(accessor_json(indices_view, 5123, 3, "SCALAR"));

    let mut sampler_json: Vec<String> = Vec::new();
    let mut channel_json: Vec<String> = Vec::new();
    for channel in channels {
        let mut raw = Vec::new();
        for time in &channel.times {
            raw.extend_from_slice(&time.to_le_bytes());
        }
        let input_view = push(&raw, &mut bin, &mut views);
        let input = accessors.len();
        accessors.push(accessor_json(
            input_view,
            5126,
            channel.times.len(),
            "SCALAR",
        ));
        let values = anim_output_values(channel.path, channel.interp, channel.times.len());
        let kind = match channel.path {
            FixtureAnimPath::Rotation => "VEC4",
            FixtureAnimPath::Translation | FixtureAnimPath::Scale => "VEC3",
        };
        let output_len = values.len();
        let mut raw = Vec::new();
        for value in &values {
            for component in value {
                raw.extend_from_slice(&component.to_le_bytes());
            }
        }
        let output_view = push(&raw, &mut bin, &mut views);
        let output = accessors.len();
        accessors.push(accessor_json(output_view, 5126, output_len, kind));
        let sampler = sampler_json.len();
        sampler_json.push(format!(
            "{{\"input\":{input},\"interpolation\":\"{}\",\"output\":{output}}}",
            channel.interp.as_str()
        ));
        channel_json.push(format!(
            "{{\"sampler\":{sampler},\"target\":{{\"node\":{},\"path\":\"{}\"}}}}",
            channel.node,
            channel.path.as_str()
        ));
    }
    let views_json = views
        .iter()
        .map(|(offset, length)| {
            format!("{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{length}}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let accessors_json = accessors.join(",");
    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"ornis-gltf-fixture\"}},\
        \"scene\":0,\"scenes\":[{{\"nodes\":[0]}}],\
        \"nodes\":[{{\"mesh\":0}}],\
        \"meshes\":[{{\"primitives\":[{{\"attributes\":{{\"POSITION\":0}},\"indices\":1}}]}}],\
        \"animations\":[{{\"name\":\"mixed\",\"channels\":[{}],\
        \"samplers\":[{}]}}],\
        \"buffers\":[{{\"byteLength\":{}}}],\
        \"bufferViews\":[{views_json}],\"accessors\":[{accessors_json}]}}",
        channel_json.join(","),
        sampler_json.join(","),
        bin.len()
    );
    assemble_glb(&json, &bin)
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

/// Same triangle as [`build_external_parts`] with a caller-chosen buffer
/// URI (path-confinement and percent-decoding tests).
pub(crate) fn build_external_parts_with_uri(uri: &str) -> (String, Vec<u8>) {
    build_parts(&triangle(), Some(uri.to_string()))
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
    let mut joint1_accessor = None;
    let mut weight1_accessor = None;
    if let Some(influence) = &fixture.influence {
        let (component, raw) = if influence.joints_u16 {
            let mut raw = Vec::new();
            for joint in &influence.joints {
                for slot in joint {
                    raw.extend_from_slice(&slot.to_le_bytes());
                }
            }
            (5123u32, raw)
        } else {
            let mut raw = Vec::new();
            for joint in &influence.joints {
                for slot in joint {
                    raw.push(*slot as u8);
                }
            }
            (5121u32, raw)
        };
        push(&raw, &mut bin, &mut views);
        joint_accessor = Some((views.len() - 1, influence.joints.len(), component));
        let (component, raw) = encode_weights(&influence.weights, influence.weights_kind);
        push(&raw, &mut bin, &mut views);
        weight_accessor = Some((views.len() - 1, influence.weights.len(), component));
        if let Some((extra_joints, extra_weights)) = &influence.extra {
            let mut raw = Vec::new();
            for joint in extra_joints {
                for slot in joint {
                    raw.extend_from_slice(&slot.to_le_bytes());
                }
            }
            push(&raw, &mut bin, &mut views);
            joint1_accessor = Some((views.len() - 1, extra_joints.len()));
            let (_, raw) = encode_weights(extra_weights, FixtureWeightsKind::F32);
            push(&raw, &mut bin, &mut views);
            weight1_accessor = Some((views.len() - 1, extra_weights.len()));
        }
    }

    // Animation clip data: one input/output pair per dummy clip (input
    // times + TRS outputs), referenced by the `animations` JSON below.
    // `CUBICSPLINE` needs triple outputs per input (in-tangent, vertex,
    // out-tangent) with distinctive tangents per key so tests can prove
    // slot placement; see [`anim_output_values`].
    let mut anim_parts: Vec<(usize, usize, usize)> = Vec::new();
    for _ in 0..fixture.animations {
        let mut raw = Vec::new();
        for time in [0.0f32, 1.0] {
            raw.extend_from_slice(&time.to_le_bytes());
        }
        push(&raw, &mut bin, &mut views);
        let input_view = views.len() - 1;
        let keys: Vec<Vec<f32>> = anim_output_values(fixture.anim_path, fixture.anim_interp, 2);
        let output_len = keys.len();
        let mut raw = Vec::new();
        for value in &keys {
            for component in value {
                raw.extend_from_slice(&component.to_le_bytes());
            }
        }
        push(&raw, &mut bin, &mut views);
        anim_parts.push((input_view, views.len() - 1, output_len));
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
    if let Some((view, count, component)) = joint_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, component, count, "VEC4"));
        attributes += &format!(",\"JOINTS_0\":{index}");
    }
    if let Some((view, count, component)) = weight_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, component, count, "VEC4"));
        attributes += &format!(",\"WEIGHTS_0\":{index}");
    }
    if let Some((view, count)) = joint1_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, 5123, count, "VEC4"));
        attributes += &format!(",\"JOINTS_1\":{index}");
    }
    if let Some((view, count)) = weight1_accessor {
        let index = accessors.len();
        accessors.push(accessor_json(view, 5126, count, "VEC4"));
        attributes += &format!(",\"WEIGHTS_1\":{index}");
    }

    // NOTE: `views_json`/`accessors_json` render at the end of the
    // builder (skin and animation accessors are pushed below), so the
    // table strings below only cover the attribute accessors so far.
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
            if let Some(skin) = node.skin {
                fields.push(format!("\"skin\":{skin}"));
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
    // Skin inverse-bind matrices: one MAT4 view per skin with an explicit
    // bind, referenced by the `skins` JSON below (absent entry → identity).
    let mut ibm_accessors: Vec<Option<usize>> = Vec::new();
    for skin in &fixture.skins {
        if let Some(matrices) = &skin.inverse_bind {
            let mut raw = Vec::new();
            for matrix in matrices {
                for component in matrix {
                    raw.extend_from_slice(&component.to_le_bytes());
                }
            }
            push(&raw, &mut bin, &mut views);
            let index = accessors.len();
            accessors.push(accessor_json(views.len() - 1, 5126, matrices.len(), "MAT4"));
            ibm_accessors.push(Some(index));
        } else {
            ibm_accessors.push(None);
        }
    }
    // Dummy animation clips: one channel on node 0 per entry
    // (input times + outputs from the `anim_parts` views above).
    let mut anim_json_parts: Vec<String> = Vec::new();
    for (clip, (input_view, output_view, output_len)) in anim_parts.iter().enumerate() {
        let input = accessors.len();
        accessors.push(accessor_json(*input_view, 5126, 2, "SCALAR"));
        let output = accessors.len();
        let (kind, count) = match fixture.anim_path {
            FixtureAnimPath::Rotation => ("VEC4", *output_len),
            FixtureAnimPath::Translation | FixtureAnimPath::Scale => ("VEC3", *output_len),
        };
        let count = fixture.anim_output_len_override.unwrap_or(count);
        accessors.push(accessor_json(*output_view, 5126, count, kind));
        let path = fixture.anim_path.as_str();
        let interp = fixture.anim_interp.as_str();
        anim_json_parts.push(format!(
            "{{\"name\":\"clip_{clip}\",\"channels\":[{{\"sampler\":0,\
            \"target\":{{\"node\":0,\"path\":\"{path}\"}}}}],\
            \"samplers\":[{{\"input\":{input},\"interpolation\":\"{interp}\",\
            \"output\":{output}}}]}}"
        ));
    }
    let animations_json = if anim_json_parts.is_empty() {
        String::new()
    } else {
        format!("\"animations\":[{}],", anim_json_parts.join(","))
    };
    let skins_json = if fixture.skins.is_empty() {
        String::new()
    } else {
        let parts = fixture
            .skins
            .iter()
            .zip(ibm_accessors)
            .map(|(skin, ibm)| {
                let joints = skin
                    .joints
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(",");
                let ibm_json = ibm.map_or(String::new(), |index| {
                    format!(",\"inverseBindMatrices\":{index}")
                });
                let name_json = skin
                    .name
                    .as_ref()
                    .map_or(String::new(), |name| format!(",\"name\":\"{name}\""));
                let skeleton_json = skin
                    .skeleton
                    .map_or(String::new(), |node| format!(",\"skeleton\":{node}"));
                format!("{{\"joints\":[{joints}]{ibm_json}{name_json}{skeleton_json}}}")
            })
            .collect::<Vec<_>>()
            .join(",");
        format!("\"skins\":[{parts}],")
    };
    let buffer_json = match buffer_uri {
        None => format!("{{\"byteLength\":{}}}", bin.len()),
        Some(uri) => format!("{{\"byteLength\":{},\"uri\":\"{uri}\"}}", bin.len()),
    };
    // NOTE: `accessors_json`/`views_json` are rendered after the skin and
    // animation accessors above are pushed, so indices stay valid.
    let views_json = views
        .iter()
        .map(|(offset, length)| {
            format!("{{\"buffer\":0,\"byteOffset\":{offset},\"byteLength\":{length}}}")
        })
        .collect::<Vec<_>>()
        .join(",");
    let accessors_json = accessors.join(",");
    let json = format!(
        "{{\"asset\":{{\"version\":\"2.0\",\"generator\":\"ornis-gltf-fixture\"}},\
        \"scene\":0,\"scenes\":[{{{scene_name_json}\"nodes\":[{roots_json}]}}],\
        \"nodes\":[{nodes_json}],\
        \"meshes\":[{{\"primitives\":[{primitive_json}]{mesh_name_json}}}],\
        {materials_json}{textures_json}{images_json}{skins_json}{animations_json}\
        \"buffers\":[{buffer_json}],\
        \"bufferViews\":[{views_json}],\"accessors\":[{accessors_json}]}}"
    );
    (json, bin, external_files)
}

/// Encodes fixture weights to raw bytes in the requested storage width.
fn encode_weights(weights: &[[f32; 4]], kind: FixtureWeightsKind) -> (u32, Vec<u8>) {
    match kind {
        FixtureWeightsKind::F32 => {
            let mut raw = Vec::new();
            for weight in weights {
                for component in weight {
                    raw.extend_from_slice(&component.to_le_bytes());
                }
            }
            (5126, raw)
        }
        FixtureWeightsKind::U8 => {
            let mut raw = Vec::new();
            for weight in weights {
                for component in weight {
                    raw.push((component.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
            (5121, raw)
        }
        FixtureWeightsKind::U16 => {
            let mut raw = Vec::new();
            for weight in weights {
                for component in weight {
                    raw.extend_from_slice(
                        &((component.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes(),
                    );
                }
            }
            (5123, raw)
        }
    }
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
