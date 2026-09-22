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
//! | default scene, else first scene | [`LoadedScene::name`] | `Scene.name` |
//! | node with a mesh | one [`LoadedEntity`] per mesh primitive | one `EntityDesc` per entity |
//! | node TRS / matrix, world-composed | `translation`, `rotation` (`x,y,z,w`), `scale` | `TransformDesc` verbatim |
//! | node name → mesh name → `mesh_{mi}_{pi}` | [`LoadedEntity::name`] | `EntityDesc.name` |
//! | primitive `POSITION` | [`LoadedMesh::positions`] (verbatim) | `MeshDesc::Custom.positions` |
//! | primitive indices (`u8`/`u16`/`u32`) | [`LoadedMesh::indices`] as `u32` (verbatim; absent → sequential) | `MeshDesc::Custom.indices` |
//! | primitive `NORMAL` | [`LoadedMesh::normals`] (`Some`, verbatim) | dropped: recomputed at upload |
//! | absent `NORMAL` | [`LoadedMesh::normals`] is `None`; [`LoadedMesh::resolved_normals`] recomputes area-weighted (as `custom_mesh_data`) | `mesh_upload::custom_mesh_data` |
//! | primitive `TEXCOORD_0` | [`LoadedMesh::uvs`] (`Some`, cast to `f32`) | dropped: rebuilt at upload |
//! | absent `TEXCOORD_0` | [`LoadedMesh::uvs`] is `None`; [`LoadedMesh::resolved_uvs`] rebuilds with a box projection (as `custom_mesh_data`) | `mesh_upload::custom_mesh_data` |
//! | `baseColorFactor` / `metallicFactor` / `roughnessFactor` / `emissiveFactor` | [`LoadedMaterial`] scalars | `metallic >= 0.5` → `MaterialDesc::Metal`, else `Dielectric` |
//! | `baseColorTexture` | [`LoadedMaterial::base_color_texture`] (RGBA8) | albedo bind at upload |
//! | `metallicRoughnessTexture` | [`LoadedMaterial::metallic_roughness_texture`] (RGBA8; G = roughness, B = metallic) | roughness/metallic bind at upload |
//! | `emissiveTexture` | [`LoadedMaterial::emissive_texture`] (RGBA8) | emission bind at upload |
//!
//! # Skip rules (honest: skip + counter, never a stub mesh)
//!
//! | Input | Outcome | Counter |
//! |---|---|---|
//! | `mode != TRIANGLES` (points/lines/strips/fans) | primitive skipped, no triangulation in v1 | [`ImportStats::skipped_non_triangle`] |
//! | `JOINTS_0` or `WEIGHTS_0` present | primitive skipped (skin track owns it) | [`ImportStats::skipped_skinned`] |
//! | no `POSITION` attribute | primitive skipped | [`ImportStats::skipped_no_position`] |
//! | empty positions or indices | primitive skipped | [`ImportStats::skipped_empty`] |
//! | index out of range, or unindexed count not a multiple of 3 | primitive skipped | [`ImportStats::skipped_bad_index`] |
//! | morph targets, cameras, lights, extensions, samplers | ignored | — (documented here) |
//! | animations, skins, skeletons | ignored (separate track) | — (documented here) |
//! | node without a mesh | traversed for children only | — |
//! | external buffer URI under [`load_slice`] | `Err(ExternalBuffer)` — use [`load_path`] | — |
//!
//! # Next steps (explicitly NOT in this crate)
//!
//! 1. GPU upload of the decoded [`LoadedImage`] pixels: create the texture,
//!    sampler, and `MaterialDesc` binding in `ornis-render`. Sampling,
//!    filtering, and `texCoord` sets live there, not here (samplers are
//!    ignored on import).
//! 2. Skin + animations per `docs/animation-design.md` §4 (`Skeleton`,
//!    `JointPose`, `SkinnedMesh`, `SkelClip` contract).
//! 3. Wiring: `LoadedScene` → `ornis-render` `Scene` (host keeps its own
//!    camera/lights/ambient; [`LoadedMesh::into_custom`] feeds
//!    `MeshDesc::Custom`; [`LoadedMaterial::is_metallic`] picks the
//!    `MaterialDesc` variant).

#![warn(missing_docs)]

mod base64;
mod geom;
mod import;
mod textures;

#[cfg(test)]
mod fixtures;

pub use import::{load_path, load_slice};

/// Geometry-only import result: flat entity list plus skip counters.
///
/// Mirrors `ornis-render` `Scene` without camera/lights/ambient — the host
/// owns those and fills them in at wiring time.
#[derive(Debug, Clone)]
pub struct LoadedScene {
    /// Scene label: glTF scene name, else `"scene"`.
    pub name: String,
    /// One entry per imported mesh primitive, hierarchy flattened.
    pub entities: Vec<LoadedEntity>,
    /// Primitive/skip counters; see the skip-rules table.
    pub stats: ImportStats,
}

/// One imported mesh primitive with its world transform.
///
/// Field layouts match `ornis-render` `EntityDesc`/`TransformDesc` so the
/// later wiring copies them verbatim.
#[derive(Debug, Clone)]
pub struct LoadedEntity {
    /// Display name: node name → mesh name → `mesh_{mesh}_{primitive}`.
    pub name: String,
    /// World-space translation in glTF units.
    pub translation: [f32; 3],
    /// World-space orientation as `(x, y, z, w)`, unit length.
    pub rotation: [f32; 4],
    /// World-space scale per axis (may carry a mirror bake, see [`load_slice`]).
    pub scale: [f32; 3],
    /// Triangle soup plus optional source attributes.
    pub mesh: LoadedMesh,
    /// Scalar PBR factors plus decoded texture slots.
    pub material: LoadedMaterial,
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
}

impl LoadedMesh {
    /// Splits the soup into the `MeshDesc::Custom` pair (positions, indices).
    ///
    /// Drops the optional source attributes — the upload path rebuilds them
    /// via [`LoadedMesh::resolved_normals`] / [`LoadedMesh::resolved_uvs`].
    pub fn into_custom(self) -> (Vec<[f32; 3]>, Vec<u32>) {
        (self.positions, self.indices)
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
/// Wiring rule: [`LoadedMaterial::is_metallic`] picks `MaterialDesc::Metal`,
/// otherwise `MaterialDesc::Dielectric`; `base_color` feeds `base_color`,
/// `roughness` feeds `roughness`, `emission` feeds `emission`. Each `Some`
/// texture feeds the matching upload bind; `None` means the slot is unbound
/// and the scalar factor stands alone.
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
    /// Whether the wiring should pick `Metal` over `Dielectric`.
    ///
    /// Threshold `>= 0.5` on the scalar factor; a bound
    /// metallic-roughness texture does not move the switch — the upload
    /// shader samples it at runtime instead.
    pub fn is_metallic(&self) -> bool {
        self.metallic >= 0.5
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
    /// Primitives carrying `JOINTS_0`/`WEIGHTS_0` (skin track owns them).
    pub skipped_skinned: u32,
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
            && self.skipped_empty == 0
            && self.skipped_bad_index == 0
    }
}

/// Geometry import failure.
#[derive(Debug)]
pub enum ImportError {
    /// Bytes are not a parseable glTF 2.0 asset (carries the parser message).
    Parse(String),
    /// The document has no scenes to traverse.
    NoScene,
    /// A `BIN`-chunk buffer with no binary chunk (truncated `.glb`).
    MissingBlob,
    /// An external buffer URI seen by [`load_slice`], which never touches
    /// the filesystem — retry with [`load_path`].
    ExternalBuffer(String),
    /// Declared `byteLength` exceeds the resolved buffer bytes.
    BufferTooShort {
        /// Buffer index in the document.
        index: usize,
        /// Declared `byteLength`.
        expected: usize,
        /// Resolved byte count.
        actual: usize,
    },
    /// A `data:` URI that is not decodable base64 (carries the URI head).
    InvalidDataUri(String),
    /// An image `mimeType` (or file extension) outside the core pair
    /// (`image/png`, `image/jpeg`; carries `image {index}` plus the
    /// offending type).
    UnsupportedImage(String),
    /// Image bytes no PNG/JPEG decoder accepts (carries `image {index}`
    /// plus the size or range context).
    InvalidImage(String),
    /// Filesystem failure inside [`load_path`] (asset or sibling buffer).
    Io(std::io::Error),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(message) => write!(f, "invalid glTF asset: {message}"),
            Self::NoScene => write!(f, "glTF document has no scenes"),
            Self::MissingBlob => write!(f, "GLB buffer has no BIN chunk"),
            Self::ExternalBuffer(uri) => write!(
                f,
                "external buffer '{uri}' needs the filesystem: use load_path"
            ),
            Self::BufferTooShort {
                index,
                expected,
                actual,
            } => write!(
                f,
                "buffer {index} too short: declared {expected} bytes, got {actual}"
            ),
            Self::InvalidDataUri(head) => {
                write!(f, "undecodable buffer data URI near '{head}'")
            }
            Self::UnsupportedImage(context) => {
                write!(f, "unsupported glTF image: {context}")
            }
            Self::InvalidImage(context) => {
                write!(f, "invalid glTF image: {context}")
            }
            Self::Io(error) => write!(f, "glTF IO error: {error}"),
        }
    }
}

impl std::error::Error for ImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ImportError {
    /// Wraps filesystem failures from [`load_path`] buffer resolution.
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
