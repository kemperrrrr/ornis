//! Geometry-only glTF 2.0 import: `.glb` / `.gltf` bytes to engine-shaped data.
//!
//! This is the first step of the asset pipeline: static triangle geometry
//! plus node transforms, scalar PBR factors, and decoded texture pixels.
//! Skeletal data and animations are a separate track
//! (see `docs/animation-design.md`) and GPU texture upload is a later step
//! — both are skipped honestly here, never stubbed.
//!
//! The crate is intentionally `std`-only plus the `gltf` crate: no `wgpu`,
//! no `ornis-render` dependency. Outputs mirror the render transport shapes
//! field-for-field so wiring is mechanical (see the mapping table); the
//! actual `Scene` assembly in the editor happens later.
//!
//! # Mapping (glTF 2.0 → this crate → `ornis-render` wiring)
//!
//! | glTF source | Output here | Later wiring (`crates/render/src/scene.rs`) |
//! |---|---|---|
//! | default scene, else first scene | [`Model::name`] | `Scene.name` |
//! | every scene node, including mesh-less | one [`ModelNode`] (local TRS, parent) | one entity per node at spawn (Core track) |
//! | node TRS / matrix, **local** | [`ModelNode::local`] ([`ornis_core::Transform`]) | world pose via [`Model::world_transform`] |
//! | node name | [`ModelNode::name`] (`None` when omitted) | `EntityDesc.name` (unnamed → `mesh_{node}_{prim}`) |
//! | primitive `POSITION` | [`LoadedMesh::positions`] (verbatim) | `MeshDesc::Custom.positions` |
//! | primitive indices (`u8`/`u16`/`u32`) | [`LoadedMesh::indices`] as `u32` (verbatim; absent → sequential) | `MeshDesc::Custom.indices` |
//! | primitive `NORMAL` | [`LoadedMesh::normals`] (`Some`, verbatim) | dropped: recomputed at upload |
//! | absent `NORMAL` | [`LoadedMesh::normals`] is `None`; [`LoadedMesh::resolved_normals`] recomputes area-weighted (as `custom_mesh_data`) | `mesh_upload::custom_mesh_data` |
//! | primitive `TEXCOORD_0` | [`LoadedMesh::uvs`] (`Some`, cast to `f32`) | dropped: rebuilt at upload |
//! | absent `TEXCOORD_0` | [`LoadedMesh::uvs`] is `None`; [`LoadedMesh::resolved_uvs`] rebuilds with a box projection (as `custom_mesh_data`) | `mesh_upload::custom_mesh_data` |
//! | primitive `JOINTS_0` (+`JOINTS_1..` when present) | [`LoadedMesh::joints`] (`Some`, top-4 by weight, `u16`) | `SkinnedMesh.joints` via `ornis-animation` builders |
//! | primitive `WEIGHTS_0` (+`WEIGHTS_1..` when present) | [`LoadedMesh::weights`] (`Some`, normalized, `sum == 1`) | `SkinnedMesh.weights` via `ornis-animation` builders |
//! | `skins[]` + node `skin` | [`LoadedSkin`] (`parents`/`inverse_bind`/`joint_names`) + [`ModelNode::skin`] link | `Skeleton` via `ornis-animation` builders |
//! | `animations[]` sampler `input`/`output` per `channel.target.path` | [`Model::skel_clips`] ([`LoadedSkelClip`]) + [`Model::anim_clips`] ([`LoadedAnimClip`]); tracks address [`NodeIdx`] | `SkelClip`/`AnimClip` cold lanes via the animation builders |
//! | `baseColorFactor` / `metallicFactor` / `roughnessFactor` / `emissiveFactor` | [`LoadedMaterial`] scalars | factor kept as [`ornis_core::Metallic`]; `1` → `MaterialDesc::Metal`, else `Dielectric` |
//! | `baseColorTexture` | [`LoadedMaterial::base_color_texture`] (RGBA8) | albedo bind at upload |
//! | `metallicRoughnessTexture` | [`LoadedMaterial::metallic_roughness_texture`] (RGBA8; G = roughness, B = metallic) | roughness/metallic bind at upload |
//! | `emissiveTexture` | [`LoadedMaterial::emissive_texture`] (RGBA8) | emission bind at upload |
//!
//! # Skip rules (honest: skip + counter, never a stub mesh)
//!
//! | Input | Outcome | Counter |
//! |---|---|---|
//! | `mode != TRIANGLES` (points/lines/strips/fans) | primitive skipped, no triangulation in v1 | [`ImportStats::skipped_non_triangle`] |
//! | `JOINTS_0`/`WEIGHTS_0` malformed (missing set, count or width mismatch) | primitive skipped, no stub skin | [`ImportStats::skipped_skinned`] |
//! | more than 4 nonzero influences across sets | top-4 kept, renormalized, one stderr warn per load | — (no counter: shape stays importable) |
//! | `animations[]` with only malformed `CUBICSPLINE`/morph/malformed channels | clip skipped (no tracks assembled) | [`ImportStats::skipped_clips`] |
//! | malformed `CUBICSPLINE` sampler channel (unreadable accessor, output count != `3`× input count) | channel skipped, other channels of the clip still assemble | [`ImportStats::skipped_cubicspline`] |
//! | no `POSITION` attribute | primitive skipped | [`ImportStats::skipped_no_position`] |
//! | empty positions or indices | primitive skipped | [`ImportStats::skipped_empty`] |
//! | index out of range, or unindexed count not a multiple of 3 | primitive skipped | [`ImportStats::skipped_bad_index`] |
//! | morph targets, cameras, lights, extensions, samplers | ignored | — (documented here) |
//! | animations, skeletons (beyond the topology above) | morph-target channels ignored (clip track lands later) | — (documented here) |
//! | node without a mesh | kept in [`Model::nodes`] with an empty primitive list | — |
//! | external buffer URI under [`load_slice`] | `Err(ExternalBuffer)` — use [`load_path`] | — |
//!
//! # Next steps (explicitly NOT in this crate)
//!
//! 1. GPU upload of the decoded [`LoadedImage`] pixels: create the texture,
//!    sampler, and `MaterialDesc` binding in `ornis-render`. Sampling,
//!    filtering, and `texCoord` sets live there, not here (samplers are
//!    ignored on import).
//! 2. `SkelClip`/`AnimClip` assembly from `animations[]` per `docs/animation-design.md`
//!    §4 lands here ([`Model::skel_clips`], [`Model::anim_clips`];
//!    `LINEAR`/`STEP`/`CUBICSPLINE` assemble, malformed `CUBICSPLINE`
//!    channels skip with [`ImportStats::skipped_cubicspline`], morph targets
//!    stay ignored). Tracks address [`NodeIdx`], so a mesh-less parent keeps
//!    its channel.
//! 3. Wiring: [`Model`] → editor `Scene` via `scene_from_model` (host keeps
//!    its own camera/lights/ambient; [`LoadedMesh::into_custom`] feeds
//!    `MeshDesc::Custom`; `metallicFactor` is stored as
//!    [`ornis_core::Metallic`] — a clamped `1` selects `MaterialDesc::Metal`,
//!    every other factor selects `Dielectric` and keeps the number).
//!    Hierarchical spawn is the Core track.

#![warn(missing_docs)]

/// Indices per triangle (flat soup alignment).
const TRIANGLE_VERTS: usize = 3;
/// Legacy `>=` cut used only by [`LoadedMaterial::is_metallic`].
const METALNESS_THRESHOLD: f32 = 0.5;

mod anim;
mod base64;
mod geom;
mod import;
mod model;
mod textures;

#[cfg(test)]
mod fixtures;

pub use anim::{
    LoadedAnimClip, LoadedAnimTrack, LoadedInterpolation, LoadedJointTrack, LoadedKey,
    LoadedKeyTrack, LoadedSkelClip, assemble_clips, node_to_joint_map,
};
pub use import::{load_path, load_slice};
pub use model::{Model, ModelNode, ModelPrimitive, NodeIdx};

/// Vertex index into a mesh vertex list.
///
/// Newtype over raw `u32` soup so vertex indices never mix with document
/// node ids at the type level. Layout is `repr(transparent)` over `u32`
/// (12 bytes per [`Triangle`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TriIndex(pub u32);

impl TriIndex {
    /// Wraps a raw vertex index without validation.
    pub const fn from_raw(index: u32) -> Self {
        Self(index)
    }

    /// Raw `u32` vertex index (for upload transports).
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Vertex position in a slice (`as usize`).
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for TriIndex {
    fn from(index: u32) -> Self {
        Self::from_raw(index)
    }
}

/// One triangle as three vertex indices (CCW from outside).
///
/// Stored as three [`TriIndex`] (12 bytes, `repr(C)`); use
/// [`Triangle::from_raw`]/[`Triangle::as_u32`] at transport boundaries and
/// [`Triangle::index`] for corner access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Triangle(pub TriIndex, pub TriIndex, pub TriIndex);

impl Triangle {
    /// Wraps three raw vertex indices without validation.
    pub const fn from_raw(indices: [u32; 3]) -> Self {
        Self(
            TriIndex(indices[0]),
            TriIndex(indices[1]),
            TriIndex(indices[2]),
        )
    }

    /// Raw `[u32; 3]` triple (for upload transports).
    pub const fn as_u32(self) -> [u32; 3] {
        [self.0.0, self.1.0, self.2.0]
    }

    /// `i`-th corner (`0..3`) as a vertex index. Out-of-range indices
    /// clamp to the last corner (same bit pattern as a saturated read).
    pub const fn index(self, i: usize) -> TriIndex {
        match i {
            0 => self.0,
            1 => self.1,
            _ => self.2,
        }
    }
}

impl From<[u32; 3]> for Triangle {
    fn from(indices: [u32; 3]) -> Self {
        Self::from_raw(indices)
    }
}

/// Triangle soup of one primitive; positions/indices feed `MeshDesc::Custom`.
///
/// Attribute presence is preserved (`Some` = verbatim source data) so hosts
/// that can carry normals/uvs keep them, while the `MeshDesc::Custom` path
/// uses [`LoadedMesh::into_custom`] plus the `resolved_*` fallbacks — the
/// same contract as `mesh_upload::custom_mesh_data` (normals recomputed,
/// uvs box-projected, never transported).
#[derive(Debug, Clone)]
pub struct LoadedMesh {
    /// Vertex positions, copied verbatim (CCW winding preserved).
    pub positions: Vec<[f32; 3]>,
    /// Triangle index list (`u32`, triples, CCW from outside).
    pub indices: Vec<u32>,
    /// Verbatim `NORMAL` when present and well-formed, else `None`.
    pub normals: Option<Vec<[f32; 3]>>,
    /// Verbatim `TEXCOORD_0` (cast to `f32`) when present and well-formed,
    /// else `None`.
    pub uvs: Option<Vec<[f32; 2]>>,
    /// Imported `JOINTS_0` (+`JOINTS_1..`, top-4 by weight) as `u16`
    /// (`Some` = skinned primitive, one entry per vertex).
    pub joints: Option<Vec<[u16; 4]>>,
    /// Imported `WEIGHTS_0` (+`WEIGHTS_1..`) normalized per vertex
    /// (`sum == 1`; zero/non-finite sums fall back to `(1,0,0,0)` on the
    /// first slot, mirroring the animation canonical rule).
    pub weights: Option<Vec<[f32; 4]>>,
}

/// One resolved `skins[]` element: joint topology plus bind inverses.
///
/// Plain arrays (no `glam`, no `ornis-animation` dependency): the wiring
/// feeds these field-for-field into the animation builders
/// (`skeleton_from_import`), which validate and convert.
#[derive(Debug, Clone)]
pub struct LoadedSkin {
    /// Parent joint per joint (`-1` = root); length equals the joint count.
    /// Resolved from the node hierarchy: the parent node of each
    /// `skin.joints` entry, mapped back into skin order (`-1` when the
    /// parent is not a joint of this skin).
    pub parents: Vec<i32>,
    /// Bind-pose inverses in glTF column-major layout (`m[col][row]`);
    /// identity per joint when the accessor is absent or malformed.
    pub inverse_bind: Vec<[[f32; 4]; 4]>,
    /// Joint labels: node names, else `joint_{node_index}` (diagnostics only).
    pub joint_names: Vec<String>,
}

impl LoadedMesh {
    /// Splits the soup into the `MeshDesc::Custom` pair (positions, indices).
    ///
    /// Drops the optional source attributes — the upload path rebuilds them
    /// via [`LoadedMesh::resolved_normals`] / [`LoadedMesh::resolved_uvs`].
    /// Indices round-trip through [`Triangle::from_raw`] /
    /// [`Triangle::as_u32`] so the flat transport stays triple-aligned by
    /// construction (12 bytes per triangle).
    pub fn into_custom(self) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut flat = Vec::with_capacity(self.indices.len());
        for c in self.indices.chunks_exact(TRIANGLE_VERTS) {
            flat.extend_from_slice(&Triangle::from_raw([c[0], c[1], c[2]]).as_u32());
        }
        // Defensive: a hand-built mesh with a non-triple tail keeps its tail
        // verbatim (the importer rejects such soups earlier with
        // `skipped_bad_index`; this stays panic-free regardless).
        let rem = self.indices.len() % TRIANGLE_VERTS;
        if rem != 0 {
            flat.extend_from_slice(&self.indices[self.indices.len() - rem..]);
        }
        (self.positions, flat)
    }

    /// Typed triangle view over the flat index list.
    ///
    /// Chunks via [`Triangle::from_raw`]; a non-triple tail is dropped (the
    /// importer rejects it earlier — see `skipped_bad_index`).
    pub fn triangles(&self) -> Vec<Triangle> {
        self.indices
            .chunks_exact(TRIANGLE_VERTS)
            .map(|c| Triangle::from_raw([c[0], c[1], c[2]]))
            .collect()
    }

    /// Stored normals, or area-weighted recomputation when absent.
    ///
    /// Matches `MeshData::with_computed_normals`
    /// (`crates/mesh-editor/src/mesh_data.rs`): face contributions are
    /// area-weighted (`(b-a)×(c-a)` accumulated unnormalized), degenerate
    /// triangles contribute nothing, and untouched vertices keep `+Y`.
    /// Out-of-range indices are ignored (the importer rejects them earlier;
    /// this stays panic-free on hand-built meshes).
    pub fn resolved_normals(&self) -> Vec<[f32; 3]> {
        if let Some(normals) = &self.normals {
            return normals.clone();
        }
        geom::area_weighted_normals(&self.positions, &self.indices)
    }

    /// Stored uvs, or a box projection when absent.
    ///
    /// Matches `apply_box_project_uvs` (`crates/render/src/mesh_upload.rs`):
    /// each vertex maps planarly from its dominant resolved-normal axis
    /// (`|nx|` → `(z, y)`, `|ny|` → `(x, z)`, else `(x, y)`), normalized over
    /// the soup bounds into `[0, 1]`; degenerate spans collapse to `0.5`.
    pub fn resolved_uvs(&self) -> Vec<[f32; 2]> {
        if let Some(uvs) = &self.uvs {
            return uvs.clone();
        }
        geom::box_project_uvs(&self.positions, &self.resolved_normals())
    }
}

/// Scalar PBR factors of one primitive plus its decoded texture slots.
///
/// `base_color`, `roughness`, `emission` and `metallic` are the glTF
/// factors. Assets wiring stores `metallic` as [`ornis_core::Metallic`]:
/// a clamped `1` selects the `Metal` preset and every other value selects
/// `Dielectric` while keeping the number. [`LoadedMaterial::is_metallic`]
/// is the old `>= 0.5` classification and does not move that number. Each
/// `Some` texture feeds the matching upload bind; `None` means the slot is
/// unbound and the scalar factor stands alone.
#[derive(Debug, Clone)]
pub struct LoadedMaterial {
    /// `baseColorFactor` RGB in linear space (default white).
    pub base_color: [f32; 3],
    /// `metallicFactor` (glTF default `1.0`).
    pub metallic: f32,
    /// `roughnessFactor` (glTF default `1.0`).
    pub roughness: f32,
    /// `emissiveFactor` RGB in linear space (default off).
    pub emission: [f32; 3],
    /// Decoded `baseColorTexture` (RGBA8 albedo multiplier), if present.
    pub base_color_texture: Option<LoadedImage>,
    /// Decoded `metallicRoughnessTexture` (RGBA8; green holds roughness,
    /// blue holds metallic), if present.
    pub metallic_roughness_texture: Option<LoadedImage>,
    /// Decoded `emissiveTexture` (RGBA8 emission multiplier), if present.
    pub emissive_texture: Option<LoadedImage>,
}

impl LoadedMaterial {
    /// Whether `metallicFactor` is at least `0.5`.
    ///
    /// Legacy classification. Assets wiring no longer uses it to choose
    /// a preset or to snap the factor. A bound metallic-roughness texture
    /// does not move the result — the upload shader samples that image
    /// at runtime instead.
    pub fn is_metallic(&self) -> bool {
        self.metallic >= METALNESS_THRESHOLD
    }

    /// Image bound to `role`, if that slot is textured.
    pub fn texture(&self, role: TextureRole) -> Option<&LoadedImage> {
        match role {
            TextureRole::BaseColor => self.base_color_texture.as_ref(),
            TextureRole::MetallicRoughness => self.metallic_roughness_texture.as_ref(),
            TextureRole::Emissive => self.emissive_texture.as_ref(),
        }
    }
}

/// Decoded texture image: always RGBA8, row-major, top row first.
///
/// PNG (`image/png`) and JPEG (`image/jpeg`) sources both land here — the
/// decoder normalizes channels (JPEG gains opaque alpha), so the GPU-upload
/// step needs no format switch. `pixels` holds exactly
/// `width * height * 4` bytes.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// RGBA8 bytes, row-major from the top row.
    pub pixels: Vec<u8>,
}

/// Which material slot a texture image feeds.
///
/// Mirrors the three glTF texture slots this crate resolves; the upload step
/// matches on this to pick the GPU binding. Sampler parameters
/// (filter/wrap) and `texCoord` sets are intentionally not carried — upload
/// uses its own defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureRole {
    /// `baseColorTexture`: albedo multiplier.
    BaseColor,
    /// `metallicRoughnessTexture`: green holds roughness, blue holds metallic.
    MetallicRoughness,
    /// `emissiveTexture`: emission multiplier.
    Emissive,
}

/// Primitive/skip counters for one import; see the skip-rules table.
///
/// All skips are silent per-primitive drops counted here — no stub geometry,
/// no stderr spam (the loader runs at asset time, not per frame).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImportStats {
    /// Nodes visited during scene traversal (with or without meshes).
    pub nodes_visited: u32,
    /// Mesh primitives seen across all visited nodes.
    pub primitives_total: u32,
    /// Primitives that became entities.
    pub entities: u32,
    /// Primitives without `POSITION`.
    pub skipped_no_position: u32,
    /// Primitives whose `mode` is not `TRIANGLES`.
    pub skipped_non_triangle: u32,
    /// Skinned primitives that failed skin-attribute decoding (missing
    /// set, vertex-count or component-width mismatch) — skipped, never stubbed.
    pub skipped_skinned: u32,
    /// Document `animations[]` with no assemblable tracks (only
    /// `CUBICSPLINE`/morph/malformed channels) — skipped, never a stub clip.
    pub skipped_clips: u32,
    /// `CUBICSPLINE` sampler channels skipped (`LINEAR`/`STEP` channels of
    /// the same clip still assemble).
    pub skipped_cubicspline: u32,
    /// Primitives with empty positions or indices.
    pub skipped_empty: u32,
    /// Primitives with out-of-range indices or a malformed count.
    pub skipped_bad_index: u32,
}

impl ImportStats {
    /// True when every seen primitive became an entity (no skips).
    pub fn is_clean(&self) -> bool {
        self.skipped_no_position == 0
            && self.skipped_non_triangle == 0
            && self.skipped_skinned == 0
            && self.skipped_clips == 0
            && self.skipped_cubicspline == 0
            && self.skipped_empty == 0
            && self.skipped_bad_index == 0
    }
}

/// Geometry import failure.
///
/// Typed per cause: `Ok` values never carry error text, and each `Err`
/// names its reason — a missing filesystem sibling ([`ExternalBuffer`])
/// never mixes with undecodable bytes ([`Parse`]). Sample hierarchy:
/// `TextureUploadError` in `ornis-render` (same struct-variant discipline).
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// Bytes are not a parseable glTF 2.0 asset (carries the parser message).
    #[error("invalid glTF asset: {message}")]
    Parse {
        /// Parser message (unstructured: the `gltf` crate reports text).
        message: String,
    },
    /// The document has no scenes to traverse.
    #[error("glTF document has no scenes")]
    NoScene,
    /// A `BIN`-chunk buffer with no binary chunk (truncated `.glb`).
    #[error("GLB buffer has no BIN chunk")]
    MissingBlob,
    /// An external buffer URI seen by [`load_slice`], which never touches
    /// the filesystem — retry with [`load_path`]. Typed as a path (not a
    /// bare `String`) so hosts can join it against an asset directory.
    #[error("external buffer '{0}' needs the filesystem: use load_path")]
    ExternalBuffer(std::path::PathBuf),
    /// A relative buffer/image URI that would leave the asset directory
    /// (absolute path, `..` segment, NUL byte) or has broken
    /// percent-encoding. Rejected before any filesystem access.
    #[error("unsafe glTF URI '{uri}': {reason}")]
    UnsafeUri {
        /// First 48 characters of the offending (raw) URI.
        uri: String,
        /// Why it was rejected.
        reason: &'static str,
    },
    /// Declared `byteLength` exceeds the resolved buffer bytes.
    #[error("buffer {index} too short: declared {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Buffer index in the document.
        index: usize,
        /// Declared `byteLength`.
        expected: usize,
        /// Resolved byte count.
        actual: usize,
    },
    /// A `data:` URI that is not decodable base64 (carries the URI head).
    #[error("undecodable buffer data URI near '{head}'")]
    InvalidDataUri {
        /// First 48 characters of the offending URI.
        head: String,
    },
    /// An image `mimeType` (or file extension) outside the core pair
    /// (`image/png`, `image/jpeg`; carries `image {index}` plus the
    /// offending type).
    #[error("unsupported glTF image: {context}")]
    UnsupportedImage {
        /// `image {index}` plus the offending mime/extension.
        context: String,
    },
    /// Image bytes no PNG/JPEG decoder accepts (carries `image {index}`
    /// plus the size or range context).
    #[error("invalid glTF image: {context}")]
    InvalidImage {
        /// `image {index}` plus the size or range context.
        context: String,
    },
    /// Filesystem failure inside [`load_path`] (asset or sibling buffer).
    #[error("glTF IO error: {0}")]
    Io(#[from] std::io::Error),
}
