//! Document traversal and primitive decoding behind [`load_slice`]/[`load_path`].

use std::collections::HashMap;
use std::path::Path;

use gltf::accessor::DataType;
use gltf::mesh::Semantic;
use gltf::mesh::util::{ReadIndices, ReadJoints, ReadTexCoords, ReadWeights};
use gltf::scene::Transform;
use gltf::{Buffer, Document, Gltf, Node, Primitive, Skin};

use crate::anim::{assemble_clips, node_to_joint_map};
use crate::base64;
use crate::geom::{self, Mat4};
use crate::textures::resolve_images;
use crate::{
    ImportError, ImportStats, LoadedEntity, LoadedImage, LoadedMaterial, LoadedMesh, LoadedScene,
    LoadedSkin,
};

/// Indices per triangle (flat soup alignment).
const TRIANGLE_VERTS: usize = 3;
/// Spatial components in a position / translation.
const VEC3_COMPONENTS: usize = 3;
/// Max joint influences per skinned vertex (glTF top-4 contract).
const MAX_INFLUENCES: usize = 4;
/// Matrix column/row count for bind poses.
const MAT4_DIM: usize = 4;
/// Weight-sum floor before normalize.
const NEAR_ZERO: f32 = 1e-6;

/// Parses a `.glb` or `.gltf` document held in memory.
///
/// `.glb` (binary chunk) and embedded `data:` buffers resolve directly;
/// external buffer URIs fail with [`ImportError::ExternalBuffer`] — this
/// entry point never touches the filesystem, retry with [`load_path`].
/// Scene choice is the glTF default scene, else the first scene.
///
/// # Errors
///
/// [`ImportError::Parse`] on malformed bytes, [`ImportError::NoScene`] on a
/// sceneless document, buffer errors as documented per variant.
pub fn load_slice(bytes: &[u8]) -> Result<LoadedScene, ImportError> {
    let gltf = Gltf::from_slice(bytes).map_err(|error| ImportError::Parse {
        message: error.to_string(),
    })?;
    let buffers = resolve_buffers(&gltf, None)?;
    let images = resolve_images(&gltf, &buffers, None)?;
    import_gltf(&gltf, &buffers, &images)
}

/// Loads a `.glb` or `.gltf` file; sibling buffer URIs resolve against the
/// parent directory, `data:` URIs decode inline.
///
/// Absolute paths and remote schemes are rejected — a geometry loader has no
/// business fetching the network; vendor the buffers next to the asset.
///
/// # Errors
///
/// Same as [`load_slice`], plus [`ImportError::Io`] for unreadable files.
pub fn load_path(path: &Path) -> Result<LoadedScene, ImportError> {
    let bytes = std::fs::read(path)?;
    let gltf = Gltf::from_slice(&bytes).map_err(|error| ImportError::Parse {
        message: error.to_string(),
    })?;
    let parent = path.parent();
    let buffers = resolve_buffers(&gltf, parent)?;
    let images = resolve_images(&gltf, &buffers, parent)?;
    import_gltf(&gltf, &buffers, &images)
}

/// Resolves every document buffer to owned bytes.
///
/// `base_dir` gates external URIs: `None` (i.e. [`load_slice`]) rejects them
/// with [`ImportError::ExternalBuffer`]; `Some` (i.e. [`load_path`]) joins
/// relative URIs under it. `data:` URIs decode inline in both modes.
fn resolve_buffers(gltf: &Gltf, base_dir: Option<&Path>) -> Result<Vec<Vec<u8>>, ImportError> {
    let mut buffers = Vec::with_capacity(gltf.document.buffers().len());
    for buffer in gltf.document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => gltf.blob.clone().ok_or(ImportError::MissingBlob)?,
            gltf::buffer::Source::Uri(uri) if is_data_uri(uri) => decode_data_uri(uri)?,
            gltf::buffer::Source::Uri(uri) => {
                let base = base_dir
                    .ok_or_else(|| ImportError::ExternalBuffer(std::path::PathBuf::from(uri)))?;
                reject_remote_uri(uri)?;
                std::fs::read(base.join(uri))?
            }
        };
        if data.len() < buffer.length() {
            return Err(ImportError::BufferTooShort {
                index: buffer.index(),
                expected: buffer.length(),
                actual: data.len(),
            });
        }
        buffers.push(data);
    }
    Ok(buffers)
}

/// True for `data:...;base64,...` URIs (any media type).
pub(crate) fn is_data_uri(uri: &str) -> bool {
    uri.starts_with("data:") && uri.contains(";base64,")
}

/// Extracts and decodes the payload after the first comma.
pub(crate) fn decode_data_uri(uri: &str) -> Result<Vec<u8>, ImportError> {
    let payload = uri.split_once(',').map_or("", |(_, after)| after);
    base64::decode(payload).map_err(|_| ImportError::InvalidDataUri {
        head: short_head(uri),
    })
}

/// Rejects absolute paths and remote schemes before any filesystem access.
pub(crate) fn reject_remote_uri(uri: &str) -> Result<(), ImportError> {
    if uri.contains("://") || uri.starts_with("//") || uri.starts_with("data:") {
        return Err(ImportError::ExternalBuffer(std::path::PathBuf::from(uri)));
    }
    Ok(())
}

/// First 48 characters of a URI for error messages (URIs can be megabytes).
pub(crate) fn short_head(uri: &str) -> String {
    uri.chars().take(48).collect()
}

/// Traverses the picked scene, flattening node hierarchies to entities.
fn import_gltf(
    gltf: &Gltf,
    buffers: &[Vec<u8>],
    images: &[LoadedImage],
) -> Result<LoadedScene, ImportError> {
    let document = &gltf.document;
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or(ImportError::NoScene)?;
    let mut import = Import::new(buffers, images, parent_map(document));
    for node in scene.nodes() {
        import.visit_node(&node, &geom::IDENTITY);
    }
    if import.truncated_influences {
        eprintln!(
            "ornis-gltf: scene '{}' keeps the top-4 influences per vertex \
             (>4 nonzero weights across JOINTS_n/WEIGHTS_n sets are truncated)",
            scene.name().unwrap_or("scene")
        );
    }
    let node_to_joint = node_to_joint_map(document);
    let (skel_clips, anim_clips) =
        assemble_clips(document, buffers, &node_to_joint, &mut import.stats);
    Ok(LoadedScene {
        name: scene.name().unwrap_or("scene").to_string(),
        entities: import.entities,
        skins: import.skins,
        skel_clips,
        anim_clips,
        stats: import.stats,
    })
}

/// Child → parent node index over the whole document (one walk; skins
/// resolve joint parents through it without re-traversing per skin).
fn parent_map(document: &Document) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for node in document.nodes() {
        for child in node.children() {
            map.insert(child.index(), node.index());
        }
    }
    map
}

/// Traversal state: output entities plus counters.
struct Import<'a> {
    /// Resolved buffer bytes indexed by document buffer index.
    buffers: &'a [Vec<u8>],
    /// Resolved image pixels indexed by document image index.
    images: &'a [LoadedImage],
    /// Finished entities in traversal order.
    entities: Vec<LoadedEntity>,
    /// Resolved skins in document order (see [`LoadedScene::skins`]).
    skins: Vec<LoadedSkin>,
    /// Document skin index → [`Import::skins`] position.
    skin_index: HashMap<usize, usize>,
    /// Child → parent node index over the whole document.
    parents: HashMap<usize, usize>,
    /// Whether any vertex lost nonzero influences to the top-4 cap
    /// (reported once per load, never per frame).
    truncated_influences: bool,
    /// Counters (see the crate skip-rules table).
    stats: ImportStats,
}

impl<'a> Import<'a> {
    /// Borrows resolved buffers and images for one traversal.
    fn new(
        buffers: &'a [Vec<u8>],
        images: &'a [LoadedImage],
        parents: HashMap<usize, usize>,
    ) -> Self {
        Self {
            buffers,
            images,
            entities: Vec::new(),
            skins: Vec::new(),
            skin_index: HashMap::new(),
            parents,
            truncated_influences: false,
            stats: ImportStats::default(),
        }
    }

    /// Visits a node: emits one entity per mesh primitive, then recurses
    /// into children with the composed world matrix.
    fn visit_node(&mut self, node: &Node<'_>, parent: &Mat4) {
        self.stats.nodes_visited += 1;
        let world = geom::mat_mul(parent, &local_matrix(&node.transform()));
        if let Some(mesh) = node.mesh() {
            let mesh_index = mesh.index();
            let mesh_name = mesh.name().map(str::to_string);
            for primitive in mesh.primitives() {
                self.stats.primitives_total += 1;
                if let Some(entity) =
                    self.import_primitive(node, mesh_index, &mesh_name, &primitive, &world)
                {
                    self.stats.entities += 1;
                    self.entities.push(entity);
                }
            }
        }
        for child in node.children() {
            self.visit_node(&child, &world);
        }
    }

    /// Decodes one primitive to an entity, or counts a skip (never a stub).
    ///
    /// Skinned primitives (`JOINTS_0`/`WEIGHTS_0`) import like classic ones
    /// plus the influence arrays; only malformed skin attributes skip the
    /// primitive ([`ImportStats::skipped_skinned`]).
    fn import_primitive(
        &mut self,
        node: &Node<'_>,
        mesh_index: usize,
        mesh_name: &Option<String>,
        primitive: &Primitive<'_>,
        world: &Mat4,
    ) -> Option<LoadedEntity> {
        if primitive.mode() != gltf::mesh::Mode::Triangles {
            self.stats.skipped_non_triangle += 1;
            return None;
        }
        // Local buffer alias: the reader closure below must not capture
        // `self` (later counter writes would collide with that borrow).
        let buffers = self.buffers;
        let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice));
        let skinned = primitive
            .attributes()
            .any(|(semantic, _)| matches!(semantic, Semantic::Joints(_) | Semantic::Weights(_)));
        // Count guards before any read: the reader backend panics on empty
        // slices, and an empty soup is a skip either way.
        let positions_count = primitive
            .get(&Semantic::Positions)
            .map_or(0, |accessor| accessor.count());
        let indices_count = primitive
            .indices()
            .map_or(positions_count, |accessor| accessor.count());
        if positions_count == 0 || indices_count == 0 {
            self.stats.skipped_empty += 1;
            return None;
        }
        let Some(read_positions) = reader.read_positions() else {
            self.stats.skipped_no_position += 1;
            return None;
        };
        let positions: Vec<[f32; VEC3_COMPONENTS]> = read_positions.collect();
        let flat: Vec<u32> = match reader.read_indices() {
            Some(indices) => collect_indices(indices),
            None => (0..positions.len() as u32).collect(),
        };
        if positions.is_empty() || flat.is_empty() {
            self.stats.skipped_empty += 1;
            return None;
        }
        // Typed validation: chunk through `Triangle::from_raw`, then check
        // triple alignment and vertex range via `TriIndex::index`.
        if !flat.len().is_multiple_of(TRIANGLE_VERTS) {
            self.stats.skipped_bad_index += 1;
            return None;
        }
        let triangles: Vec<crate::Triangle> = flat
            .chunks_exact(TRIANGLE_VERTS)
            .map(|c| crate::Triangle::from_raw([c[0], c[1], c[2]]))
            .collect();
        if triangles
            .iter()
            .flat_map(|t| t.as_u32())
            .any(|index| crate::TriIndex::from_raw(index).index() >= positions.len())
        {
            self.stats.skipped_bad_index += 1;
            return None;
        }
        let indices: Vec<u32> = triangles.iter().flat_map(|t| t.as_u32()).collect();
        let vertex_count = positions.len();
        let influences = if skinned {
            match read_influences(primitive, &reader, vertex_count) {
                Some((joints, weights, truncated)) => {
                    self.truncated_influences |= truncated;
                    Some((joints, weights))
                }
                None => {
                    self.stats.skipped_skinned += 1;
                    return None;
                }
            }
        } else {
            None
        };
        let normals = reader
            .read_normals()
            .map(Iterator::collect)
            .filter(|normals: &Vec<[f32; VEC3_COMPONENTS]>| normals.len() == vertex_count);
        let uvs = reader
            .read_tex_coords(0)
            .map(collect_tex_coords)
            .filter(|uvs: &Vec<[f32; 2]>| uvs.len() == vertex_count);
        let (translation, rotation, scale) = geom::decompose(world);
        let (joints, weights) = influences.unzip();
        let skin = node.skin().and_then(|skin| self.resolve_skin(&skin));
        Some(LoadedEntity {
            name: entity_name(node, mesh_name, mesh_index, primitive.index()),
            node: node.index() as u32,
            translation,
            rotation,
            scale,
            mesh: LoadedMesh {
                positions,
                indices,
                normals,
                uvs,
                joints,
                weights,
            },
            skin,
            material: read_material(primitive, self.images),
        })
    }

    /// Resolves a node skin to the scene-level [`LoadedSkin`] table,
    /// memoizing by document skin index.
    ///
    /// Returns [`None`] (unskinned entity) when the skin carries no joints;
    /// the vertex influences above are still imported, so the wiring fails
    /// honestly at skeleton build time instead of reading a stub topology.
    fn resolve_skin(&mut self, skin: &Skin<'_>) -> Option<usize> {
        if let Some(&known) = self.skin_index.get(&skin.index()) {
            return Some(known);
        }
        let loaded = load_skin(skin, self.buffers, &self.parents)?;
        self.skins.push(loaded);
        let position = self.skins.len() - 1;
        self.skin_index.insert(skin.index(), position);
        Some(position)
    }
}

/// Resolves one `skins[]` element: joint parents from the node hierarchy,
/// bind inverses verbatim (identity fallback), joint names for diagnostics.
///
/// Returns [`None`] when the skin names no joints (nothing to sample).
fn load_skin(
    skin: &Skin<'_>,
    buffers: &[Vec<u8>],
    parents: &HashMap<usize, usize>,
) -> Option<LoadedSkin> {
    let joints: Vec<Node<'_>> = skin.joints().collect();
    if joints.is_empty() {
        return None;
    }
    let order: HashMap<usize, usize> = joints
        .iter()
        .enumerate()
        .map(|(position, node)| (node.index(), position))
        .collect();
    let parents = joints
        .iter()
        .map(|node| {
            parents
                .get(&node.index())
                .and_then(|parent| order.get(parent))
                .map_or(-1, |&position| position as i32)
        })
        .collect();
    let inverse_bind = skin
        .reader(|buffer| buffers.get(buffer.index()).map(Vec::as_slice))
        .read_inverse_bind_matrices()
        .map(Iterator::collect)
        .filter(|matrices: &Vec<[[f32; MAT4_DIM]; MAT4_DIM]>| matrices.len() == joints.len())
        .unwrap_or_else(|| vec![identity_bind(); joints.len()]);
    let joint_names = joints
        .iter()
        .map(|node| {
            node.name()
                .map(str::to_string)
                .unwrap_or_else(|| format!("joint_{}", node.index()))
        })
        .collect();
    Some(LoadedSkin {
        parents,
        inverse_bind,
        joint_names,
    })
}

/// Column-major identity bind matrix (glTF layout, `m[col][row]`).
fn identity_bind() -> [[f32; MAT4_DIM]; MAT4_DIM] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Decoded skin influences: per-vertex top-4 joints plus normalized
/// weights, and whether any nonzero influence was truncated.
type SkinInfluences = (Vec<[u16; MAX_INFLUENCES]>, Vec<[f32; MAX_INFLUENCES]>, bool);

/// Reads and merges all dense influence sets from `0` into top-4 `u16`
/// joints plus normalized `f32` weights (design §4.3).
///
/// Returns the per-vertex arrays plus whether any nonzero influence was
/// truncated. [`None`] = malformed skin attributes (missing set 0, vertex
/// count or component-width mismatch): the caller skips the primitive.
///
/// Width guards precede every `read_joints`/`read_weights` call: those
/// backends hit `unreachable!()` on unexpected component types, so a
/// hostile file must fail here as a skip, never as a panic.
fn read_influences<'a, 's, F>(
    primitive: &Primitive<'a>,
    reader: &gltf::mesh::Reader<'a, 's, F>,
    vertex_count: usize,
) -> Option<SkinInfluences>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    if !valid_influence_set(primitive, 0) {
        return None;
    }
    let mut pairs: Vec<Vec<(u16, f32)>> = vec![Vec::new(); vertex_count];
    for set in 0u32..(MAX_INFLUENCES as u32) {
        if primitive.get(&Semantic::Joints(set)).is_none() {
            break;
        }
        if !valid_influence_set(primitive, set) {
            return None;
        }
        let joints = read_joint_set(reader, set)?;
        let weights = read_weight_set(reader, set)?;
        if joints.len() != vertex_count || weights.len() != vertex_count {
            return None;
        }
        for (slot, (joint, weight)) in joints.into_iter().zip(weights).enumerate() {
            for lane in 0..MAX_INFLUENCES {
                pairs[slot].push((joint[lane], weight[lane]));
            }
        }
    }
    let mut truncated = false;
    let mut out_joints = Vec::with_capacity(vertex_count);
    let mut out_weights = Vec::with_capacity(vertex_count);
    for mut slot in pairs {
        // Heaviest first; the sort is stable, so ties keep set order.
        slot.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        truncated |= slot
            .iter()
            .skip(MAX_INFLUENCES)
            .any(|&(_, weight)| weight > 0.0);
        let mut joints = [0u16; MAX_INFLUENCES];
        let mut weights = [0.0f32; MAX_INFLUENCES];
        for (lane, (joint, weight)) in slot.into_iter().take(MAX_INFLUENCES).enumerate() {
            joints[lane] = joint;
            weights[lane] = weight;
        }
        out_joints.push(joints);
        out_weights.push(normalize_weights(weights));
    }
    Some((out_joints, out_weights, truncated))
}

/// Whether both influence accessors of `set` exist with decodable widths
/// (`u8`/`u16` joints, `u8`/`u16`/`f32` weights).
fn valid_influence_set(primitive: &Primitive<'_>, set: u32) -> bool {
    let joints = primitive
        .get(&Semantic::Joints(set))
        .is_some_and(|accessor| matches!(accessor.data_type(), DataType::U8 | DataType::U16));
    let weights = primitive
        .get(&Semantic::Weights(set))
        .is_some_and(|accessor| {
            matches!(
                accessor.data_type(),
                DataType::U8 | DataType::U16 | DataType::F32
            )
        });
    joints && weights
}

/// Decodes one joint set to `u16` (widths pre-checked by
/// [`valid_influence_set`]).
fn read_joint_set<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    set: u32,
) -> Option<Vec<[u16; 4]>>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    match reader.read_joints(set)? {
        ReadJoints::U8(iter) => Some(iter.map(|joint| joint.map(u16::from)).collect()),
        ReadJoints::U16(iter) => Some(iter.collect()),
    }
}

/// Decodes one weight set to `f32` (widths pre-checked by
/// [`valid_influence_set`]).
///
/// Integer storage scales exactly like the `gltf` crate's own `into_f32`
/// cast (`u8 / 255`, `u16 / 65535`); that adapter cannot be built here
/// (`CastingIter::new` is crate-private), so the scale is spelled out.
fn read_weight_set<'a, 's, F>(
    reader: &gltf::mesh::Reader<'a, 's, F>,
    set: u32,
) -> Option<Vec<[f32; 4]>>
where
    F: Clone + Fn(Buffer<'a>) -> Option<&'s [u8]>,
{
    match reader.read_weights(set)? {
        ReadWeights::U8(iter) => Some(
            iter.map(|weight| weight.map(|slot| f32::from(slot) / 255.0))
                .collect(),
        ),
        ReadWeights::U16(iter) => Some(
            iter.map(|weight| weight.map(|slot| f32::from(slot) / 65535.0))
                .collect(),
        ),
        ReadWeights::F32(iter) => Some(iter.collect()),
    }
}

/// Canonical per-vertex weights: a finite positive sum normalizes,
/// otherwise the full weight falls back to the first slot (mirrors the
/// animation canonical rule — joints are untouched).
fn normalize_weights(weights: [f32; MAX_INFLUENCES]) -> [f32; MAX_INFLUENCES] {
    let finite = weights.iter().all(|slot| slot.is_finite());
    let sum: f32 = weights.iter().sum();
    if finite && sum > NEAR_ZERO {
        [
            weights[0] / sum,
            weights[1] / sum,
            weights[2] / sum,
            weights[3] / sum,
        ]
    } else {
        [1.0, 0.0, 0.0, 0.0]
    }
}

/// Local node matrix: TRS composes through the crate convention, an
/// explicit matrix passes through verbatim.
fn local_matrix(transform: &Transform) -> Mat4 {
    match *transform {
        Transform::Matrix { matrix } => matrix,
        Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => geom::mat_from_trs(translation, rotation, scale),
    }
}

/// Entity name: node name → mesh name → `mesh_{mesh}_{primitive}` fallback.
fn entity_name(
    node: &Node<'_>,
    mesh_name: &Option<String>,
    mesh_index: usize,
    primitive_index: usize,
) -> String {
    if let Some(name) = node.name() {
        return name.to_string();
    }
    if let Some(name) = mesh_name {
        return name.clone();
    }
    format!("mesh_{mesh_index}_{primitive_index}")
}

/// Scalar PBR factors plus decoded texture slots; untextured slots are `None`.
fn read_material(primitive: &Primitive<'_>, images: &[LoadedImage]) -> LoadedMaterial {
    let material = primitive.material();
    let pbr = material.pbr_metallic_roughness();
    let base = pbr.base_color_factor();
    LoadedMaterial {
        base_color: [base[0], base[1], base[2]],
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        emission: material.emissive_factor(),
        base_color_texture: texture_image(images, pbr.base_color_texture()),
        metallic_roughness_texture: texture_image(images, pbr.metallic_roughness_texture()),
        emissive_texture: texture_image(images, material.emissive_texture()),
    }
}

/// Clones the resolved pixels for one texture slot (`None` = untextured).
///
/// Image indices come from `Gltf::from_slice` validation, so `.get` only
/// misses on hand-built nonsense — which maps to `None`, never a panic.
fn texture_image(
    images: &[LoadedImage],
    texture: Option<gltf::texture::Info<'_>>,
) -> Option<LoadedImage> {
    let image = texture?.texture().source();
    images.get(image.index()).cloned()
}

/// Casts index iterators (`u8`/`u16`/`u32`) to `u32`.
fn collect_indices(indices: ReadIndices<'_>) -> Vec<u32> {
    match indices {
        ReadIndices::U8(iter) => iter.map(u32::from).collect(),
        ReadIndices::U16(iter) => iter.map(u32::from).collect(),
        ReadIndices::U32(iter) => iter.collect(),
    }
}

/// Normalizes `TEXCOORD_0` (`u8`/`u16`/`f32`) to `f32` pairs.
fn collect_tex_coords(tex: ReadTexCoords<'_>) -> Vec<[f32; 2]> {
    tex.into_f32().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{
        self, FixtureIndices, FixtureInfluence, FixtureMaterial, FixtureNode, FixtureSkin,
        FixtureWeightsKind, build_glb, build_gltf, load_triangle, skinned_triangle, triangle,
    };

    #[test]
    fn parses_triangle_glb_verbatim() {
        let scene = load_triangle();
        assert_eq!(scene.name, "tri-scene");
        assert_eq!(scene.entities.len(), 1);
        let entity = &scene.entities[0];
        assert_eq!(entity.name, "tri-node");
        assert_eq!(entity.translation, [1.0, 2.0, 3.0]);
        assert_eq!(entity.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(entity.scale, [1.0, 1.0, 1.0]);
        assert_eq!(
            entity.mesh.positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(entity.mesh.indices, vec![0, 1, 2]);
        assert_eq!(entity.mesh.normals, Some(vec![[0.0, 0.0, 1.0]; 3]));
        assert_eq!(
            entity.mesh.uvs,
            Some(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        );
        assert_eq!(entity.mesh.resolved_normals(), vec![[0.0, 0.0, 1.0]; 3]);
        assert_eq!(
            entity.mesh.resolved_uvs(),
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        );
        assert_eq!(scene.stats.nodes_visited, 1);
        assert_eq!(scene.stats.primitives_total, 1);
        assert_eq!(scene.stats.entities, 1);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn parses_same_document_as_gltf_data_uri() {
        let json = build_gltf(&triangle());
        let scene = load_slice(json.as_bytes()).expect("data-uri gltf parses");
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(
            scene.entities[0].mesh.positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        assert_eq!(scene.entities[0].mesh.indices, vec![0, 1, 2]);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn unindexed_soup_sequences_indices() {
        let mut fixture = triangle();
        fixture.indices = FixtureIndices::Sequential;
        let scene = load_slice(&build_glb(&fixture)).expect("unindexed parses");
        assert_eq!(scene.entities[0].mesh.indices, vec![0, 1, 2]);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn u8_and_u32_indices_decode() {
        for indices in [
            FixtureIndices::U8(vec![0, 1, 2]),
            FixtureIndices::U32(vec![0, 1, 2]),
        ] {
            let mut fixture = triangle();
            fixture.indices = indices;
            let scene = load_slice(&build_glb(&fixture)).expect("indices decode");
            assert_eq!(scene.entities[0].mesh.indices, vec![0, 1, 2]);
            assert!(scene.stats.is_clean());
        }
    }

    #[test]
    fn trs_maps_verbatim_for_root_node() {
        let half = std::f32::consts::FRAC_PI_4;
        let mut fixture = triangle();
        fixture.nodes[0].rotation = Some([0.0, half.sin(), 0.0, half.cos()]);
        fixture.nodes[0].scale = Some([2.0, 0.5, 4.0]);
        let scene = load_slice(&build_glb(&fixture)).expect("trs parses");
        let entity = &scene.entities[0];
        assert_eq!(entity.translation, [1.0, 2.0, 3.0]);
        // World TRS round-trips through matrices: exact for translation,
        // epsilon for rotation/scale.
        let want_rotation = [0.0, half.sin(), 0.0, half.cos()];
        for (got, want) in entity.rotation.iter().zip(want_rotation) {
            assert!((got - want).abs() < 1e-6, "rotation {:?}", entity.rotation);
        }
        for (got, want) in entity.scale.iter().zip([2.0, 0.5, 4.0]) {
            assert!((got - want).abs() < 1e-6, "scale {:?}", entity.scale);
        }
    }

    #[test]
    fn child_world_transform_composes() {
        let mut fixture = triangle();
        fixture.nodes = vec![
            FixtureNode {
                name: None,
                mesh: false,
                skin: None,
                translation: Some([10.0, 0.0, 0.0]),
                rotation: None,
                scale: None,
                matrix: None,
                children: vec![1],
            },
            FixtureNode {
                name: Some("child".to_string()),
                mesh: true,
                skin: None,
                translation: Some([0.0, 5.0, 0.0]),
                rotation: None,
                scale: None,
                matrix: None,
                children: Vec::new(),
            },
        ];
        let scene = load_slice(&build_glb(&fixture)).expect("hierarchy parses");
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.entities[0].name, "child");
        assert_eq!(scene.entities[0].translation, [10.0, 5.0, 0.0]);
        assert_eq!(scene.stats.nodes_visited, 2);
    }

    #[test]
    fn matrix_node_decomposes_to_trs() {
        // Column-major translation matrix for [7, 8, 9].
        let mut fixture = triangle();
        fixture.nodes[0].translation = None;
        fixture.nodes[0].matrix = Some([
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            7.0, 8.0, 9.0, 1.0,
        ]);
        let scene = load_slice(&build_glb(&fixture)).expect("matrix parses");
        let entity = &scene.entities[0];
        assert_eq!(entity.translation, [7.0, 8.0, 9.0]);
        assert_eq!(entity.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(entity.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn skinned_primitive_imports_with_skin_link() {
        // Phase C: `JOINTS_0`/`WEIGHTS_0` no longer skip the primitive — the
        // influences import verbatim (already canonical here) and the node
        // `skin` resolves to the scene-level table.
        let scene = load_slice(&build_glb(&skinned_triangle())).expect("skinned parses");
        assert_eq!(scene.entities.len(), 1);
        let entity = &scene.entities[0];
        assert_eq!(
            entity.mesh.joints,
            Some(vec![[0, 0, 0, 0]; 3]),
            "u8 joints widen to u16"
        );
        assert_eq!(
            entity.mesh.weights,
            Some(vec![[1.0, 0.0, 0.0, 0.0]; 3]),
            "unit weights stay canonical"
        );
        assert_eq!(entity.skin, Some(0), "node skin links to skins[0]");
        assert_eq!(scene.skins.len(), 1);
        assert_eq!(scene.skins[0].parents, vec![-1], "single joint is a root");
        assert_eq!(scene.skins[0].joint_names, vec!["tri-node".to_string()]);
        assert_eq!(scene.stats.primitives_total, 1);
        assert_eq!(scene.stats.entities, 1);
        assert_eq!(scene.stats.skipped_skinned, 0);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn unskinned_primitive_has_no_influences() {
        // Classic path is untouched: no skin attributes, no skin link.
        let scene = load_triangle();
        assert_eq!(scene.entities[0].mesh.joints, None);
        assert_eq!(scene.entities[0].mesh.weights, None);
        assert_eq!(scene.entities[0].skin, None);
        assert!(scene.skins.is_empty());
    }

    #[test]
    fn weights_normalize_and_zero_sum_falls_back() {
        // `[2, 2, 0, 0]` → `[0.5, 0.5, 0, 0]`; all-zero → `(1,0,0,0)` on
        // the first slot (design §2.1 rule, joints untouched).
        let mut fixture = skinned_triangle();
        let influence = fixture.influence.as_mut().expect("skinned fixture");
        influence.weights = vec![
            [2.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        let scene = load_slice(&build_glb(&fixture)).expect("weights parse");
        assert_eq!(
            scene.entities[0].mesh.weights,
            Some(vec![
                [0.5, 0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ])
        );
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn u16_joints_and_u8_weights_decode() {
        // Non-default storage widths: `u16` joints pass through, `u8`
        // weights scale back (`255 → 1.0`, renormalized); `u16` weights
        // scale by `65535` the same way.
        let mut fixture = skinned_triangle();
        let influence = fixture.influence.as_mut().expect("skinned fixture");
        influence.joints_u16 = true;
        influence.joints = vec![[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8]];
        influence.weights_kind = FixtureWeightsKind::U8;
        influence.weights = vec![[1.0, 0.0, 0.0, 0.0]; 3];
        let scene = load_slice(&build_glb(&fixture)).expect("widths parse");
        assert_eq!(
            scene.entities[0].mesh.joints,
            Some(vec![[1, 2, 3, 4], [0, 0, 0, 0], [5, 6, 7, 8]])
        );
        assert_eq!(
            scene.entities[0].mesh.weights,
            Some(vec![[1.0, 0.0, 0.0, 0.0]; 3])
        );
        assert!(scene.stats.is_clean());

        let influence = fixture.influence.as_mut().expect("skinned fixture");
        influence.weights_kind = FixtureWeightsKind::U16;
        let scene = load_slice(&build_glb(&fixture)).expect("u16 weights parse");
        assert_eq!(
            scene.entities[0].mesh.weights,
            Some(vec![[1.0, 0.0, 0.0, 0.0]; 3])
        );
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn second_influence_set_truncates_to_top4() {
        // `JOINTS_1`/`WEIGHTS_1` add a heavier fifth influence on vertex 0:
        // the lightest set-0 weight drops out, the rest renormalize.
        let mut fixture = skinned_triangle();
        let influence = fixture.influence.as_mut().expect("skinned fixture");
        influence.weights = vec![
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ];
        influence.extra = Some((
            vec![[9, 9, 9, 9], [0, 0, 0, 0], [0, 0, 0, 0]],
            vec![
                [0.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        ));
        let scene = load_slice(&build_glb(&fixture)).expect("two sets parse");
        let (joints, weights) = (
            scene.entities[0].mesh.joints.clone().expect("joints kept"),
            scene.entities[0]
                .mesh
                .weights
                .clone()
                .expect("weights kept"),
        );
        // Vertex 0 keeps the four heaviest: 5.0 (joint 9), 0.4, 0.3, 0.2 —
        // the 0.1 influence is truncated, weights renormalize over 5.9.
        assert_eq!(joints[0], [9, 0, 0, 0]);
        let sum = 5.0 + 0.4 + 0.3 + 0.2;
        for (got, want) in weights[0]
            .iter()
            .zip([5.0 / sum, 0.4 / sum, 0.3 / sum, 0.2 / sum])
        {
            assert!((got - want).abs() < 1e-6, "top-4 weights {weights:?}");
        }
        // Untouched vertices keep their set-0 data.
        assert_eq!(joints[1], [0, 0, 0, 0]);
        assert_eq!(weights[1], [1.0, 0.0, 0.0, 0.0]);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn skin_parents_follow_node_hierarchy() {
        // Root joint 0 with child joint 1: `parents == [-1, 0]`, names from
        // the nodes, and the mesh node links the shared skin once.
        let mut fixture = skinned_triangle();
        fixture.nodes = vec![
            FixtureNode {
                name: Some("root".to_string()),
                mesh: false,
                skin: None,
                translation: None,
                rotation: None,
                scale: None,
                matrix: None,
                children: vec![1],
            },
            FixtureNode {
                name: Some("tip".to_string()),
                mesh: true,
                skin: Some(0),
                translation: Some([1.0, 0.0, 0.0]),
                rotation: None,
                scale: None,
                matrix: None,
                children: Vec::new(),
            },
        ];
        fixture.roots = vec![0];
        fixture.skins = vec![FixtureSkin {
            joints: vec![0, 1],
            inverse_bind: None,
            name: Some("arm".to_string()),
            skeleton: Some(0),
        }];
        let scene = load_slice(&build_glb(&fixture)).expect("hierarchy skin parses");
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.entities[0].skin, Some(0));
        assert_eq!(scene.skins[0].parents, vec![-1, 0]);
        assert_eq!(
            scene.skins[0].joint_names,
            vec!["root".to_string(), "tip".to_string()]
        );
        // Identity fallback: no `inverseBindMatrices` in the fixture.
        assert_eq!(scene.skins[0].inverse_bind.len(), 2);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn inverse_bind_matrices_import_verbatim() {
        // A non-identity bind (translation +X on joint 0) survives the
        // import in column-major layout.
        let mut fixture = skinned_triangle();
        let skin = fixture
            .skins
            .get_mut(0)
            .expect("skinned fixture has a skin");
        skin.inverse_bind = Some(vec![[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            5.0, 0.0, 0.0, 1.0,
        ]]);
        let scene = load_slice(&build_glb(&fixture)).expect("bind parses");
        let bind = &scene.skins[0].inverse_bind[0];
        assert_eq!(bind[3], [5.0, 0.0, 0.0, 1.0], "translation column kept");
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn joint_animation_assembles_skel_clip() {
        // Animated joint (node 0 is the skin's only joint): two `LINEAR`
        // translation clips assemble to skeletal clips, nothing object-side.
        let mut fixture = skinned_triangle();
        fixture.animations = 2;
        let scene = load_slice(&build_glb(&fixture)).expect("animated parses");
        assert_eq!(scene.entities.len(), 1, "skin still imports");
        assert_eq!(scene.skel_clips.len(), 2);
        assert!(scene.anim_clips.is_empty());
        for clip in &scene.skel_clips {
            assert_eq!(clip.duration, 1.0);
            assert_eq!(clip.tracks.len(), 1);
            let track = &clip.tracks[0];
            assert_eq!(track.joint, 0);
            assert_eq!(track.translation.keys.len(), 2);
            assert_eq!(track.translation.keys[0].time, 0.0);
            assert_eq!(track.translation.keys[1].time, 1.0);
            assert_eq!(track.translation.keys[1].value, [1.0, 0.0, 0.0]);
            assert_eq!(
                track.translation.interpolation,
                crate::LoadedInterpolation::Linear
            );
            // Untargeted joint channels stay empty (= identity, not untouched).
            assert!(track.rotation.keys.is_empty());
            assert!(track.scale.keys.is_empty());
        }
        assert_eq!(scene.stats.skipped_clips, 0);
        assert_eq!(scene.stats.skipped_cubicspline, 0);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn plain_node_animation_assembles_anim_clip() {
        // Unskinned node: one `LINEAR` translation clip assembles to an
        // object clip keyed by node index, nothing skeletal-side.
        let mut fixture = triangle();
        fixture.animations = 1;
        let scene = load_slice(&build_glb(&fixture)).expect("animated parses");
        assert!(scene.skel_clips.is_empty());
        assert_eq!(scene.anim_clips.len(), 1);
        let clip = &scene.anim_clips[0];
        assert_eq!(clip.name, "clip_0");
        assert_eq!(clip.duration, 1.0);
        assert!(clip.looping);
        assert_eq!(clip.tracks.len(), 1);
        let track = &clip.tracks[0];
        assert_eq!(track.entity, ornis_core::Entity::new(0));
        assert_eq!(track.translation.keys.len(), 2);
        assert_eq!(track.translation.keys[1].value, [1.0, 0.0, 0.0]);
        assert_eq!(
            track.translation.interpolation,
            crate::LoadedInterpolation::Linear
        );
        // Untargeted object channels stay empty (= untouched, size survives).
        assert!(track.rotation.keys.is_empty());
        assert!(track.scale.keys.is_empty());
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn step_interpolation_assembles_stepped_tracks() {
        let mut fixture = triangle();
        fixture.animations = 1;
        fixture.anim_interp = crate::fixtures::FixtureInterp::Step;
        let scene = load_slice(&build_glb(&fixture)).expect("step parses");
        assert_eq!(scene.anim_clips.len(), 1);
        assert_eq!(
            scene.anim_clips[0].tracks[0].translation.interpolation,
            crate::LoadedInterpolation::Step
        );
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn cubicspline_translation_assembles_cubic_track() {
        // `CUBICSPLINE` assembles now: keys carry the vertex values,
        // tangents land in the cubic lanes in (in, value, out) order —
        // no skip counters, duration from the cubic keys.
        let mut fixture = triangle();
        fixture.animations = 1;
        fixture.anim_interp = crate::fixtures::FixtureInterp::CubicSpline;
        let scene = load_slice(&build_glb(&fixture)).expect("spline parses");
        assert_eq!(scene.anim_clips.len(), 1);
        assert_eq!(scene.anim_clips[0].duration, 1.0);
        let track = &scene.anim_clips[0].tracks[0];
        assert_eq!(track.translation.keys.len(), 2);
        assert_eq!(track.translation.keys[0].time, 0.0);
        assert_eq!(track.translation.keys[1].time, 1.0);
        assert_eq!(track.translation.keys[0].value, [0.0, 0.0, 0.0]);
        assert_eq!(track.translation.keys[1].value, [1.0, 0.0, 0.0]);
        match &track.translation.interpolation {
            crate::LoadedInterpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                assert_eq!(in_tangents, &vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]);
                assert_eq!(out_tangents, &vec![[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]]);
            }
            other => panic!("expected cubic interpolation, got {other:?}"),
        }
        assert!(track.rotation.keys.is_empty());
        assert!(track.scale.keys.is_empty());
        assert_eq!(scene.stats.skipped_cubicspline, 0);
        assert_eq!(scene.stats.skipped_clips, 0);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn cubicspline_on_joint_assembles_skel_track() {
        // Same split on the joint side: the skinned node routes to a
        // skeletal clip with a cubic translation lane.
        let mut fixture = skinned_triangle();
        fixture.animations = 1;
        fixture.anim_interp = crate::fixtures::FixtureInterp::CubicSpline;
        let scene = load_slice(&build_glb(&fixture)).expect("spline parses");
        assert_eq!(scene.skel_clips.len(), 1);
        assert!(scene.anim_clips.is_empty());
        assert_eq!(scene.skel_clips[0].duration, 1.0);
        let track = &scene.skel_clips[0].tracks[0];
        assert_eq!(track.joint, 0);
        assert_eq!(track.translation.keys.len(), 2);
        assert_eq!(track.translation.keys[1].value, [1.0, 0.0, 0.0]);
        match &track.translation.interpolation {
            crate::LoadedInterpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                assert_eq!(in_tangents, &vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]);
                assert_eq!(out_tangents, &vec![[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]]);
            }
            other => panic!("expected cubic interpolation, got {other:?}"),
        }
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn cubicspline_rotation_normalizes_values_keeps_tangents_raw() {
        // Rotation values normalize to unit length (identity fallback, as
        // in the linear path); tangents stay raw derivatives.
        let mut fixture = triangle();
        fixture.animations = 1;
        fixture.anim_path = crate::fixtures::FixtureAnimPath::Rotation;
        fixture.anim_interp = crate::fixtures::FixtureInterp::CubicSpline;
        let scene = load_slice(&build_glb(&fixture)).expect("spline parses");
        assert_eq!(scene.anim_clips.len(), 1);
        let track = &scene.anim_clips[0].tracks[0];
        assert_eq!(track.rotation.keys.len(), 2);
        for key in &track.rotation.keys {
            assert_eq!(key.value, [0.0, 0.0, 0.0, 1.0]);
        }
        match &track.rotation.interpolation {
            crate::LoadedInterpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                assert_eq!(
                    in_tangents,
                    &vec![[10.0, 0.0, 0.0, 0.0], [11.0, 0.0, 0.0, 0.0]]
                );
                assert_eq!(
                    out_tangents,
                    &vec![[20.0, 0.0, 0.0, 0.0], [21.0, 0.0, 0.0, 0.0]]
                );
            }
            other => panic!("expected cubic interpolation, got {other:?}"),
        }
        assert!(track.translation.keys.is_empty());
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn cubicspline_scale_assembles_skel_scale() {
        let mut fixture = skinned_triangle();
        fixture.animations = 1;
        fixture.anim_path = crate::fixtures::FixtureAnimPath::Scale;
        fixture.anim_interp = crate::fixtures::FixtureInterp::CubicSpline;
        let scene = load_slice(&build_glb(&fixture)).expect("spline parses");
        assert_eq!(scene.skel_clips.len(), 1);
        let track = &scene.skel_clips[0].tracks[0];
        assert_eq!(track.scale.keys.len(), 2);
        assert_eq!(track.scale.keys[0].value, [1.0, 1.0, 1.0]);
        assert_eq!(track.scale.keys[1].value, [2.0, 2.0, 2.0]);
        match &track.scale.interpolation {
            crate::LoadedInterpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                assert_eq!(in_tangents, &vec![[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]);
                assert_eq!(out_tangents, &vec![[20.0, 0.0, 0.0], [21.0, 0.0, 0.0]]);
            }
            other => panic!("expected cubic interpolation, got {other:?}"),
        }
        assert!(track.translation.keys.is_empty());
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn mixed_interpolations_assemble_in_one_clip() {
        // One animation, three channels: `LINEAR` translation, `STEP`
        // scale, `CUBICSPLINE` rotation — each lane keeps its own blend.
        // The cubic channel runs longest, so the clip duration (2.0)
        // proves cubic keys feed the duration.
        use crate::fixtures::{
            FixtureAnimPath, FixtureChannel, FixtureInterp, build_mixed_clip_glb,
        };
        let glb = build_mixed_clip_glb(&[
            FixtureChannel {
                node: 0,
                path: FixtureAnimPath::Translation,
                interp: FixtureInterp::Linear,
                times: vec![0.0, 1.0],
            },
            FixtureChannel {
                node: 0,
                path: FixtureAnimPath::Rotation,
                interp: FixtureInterp::CubicSpline,
                times: vec![0.0, 2.0],
            },
            FixtureChannel {
                node: 0,
                path: FixtureAnimPath::Scale,
                interp: FixtureInterp::Step,
                times: vec![0.0, 1.0],
            },
        ]);
        let scene = load_slice(&glb).expect("mixed parses");
        assert!(scene.skel_clips.is_empty());
        assert_eq!(scene.anim_clips.len(), 1);
        let clip = &scene.anim_clips[0];
        assert_eq!(clip.name, "mixed");
        assert_eq!(clip.duration, 2.0);
        assert_eq!(clip.tracks.len(), 1);
        let track = &clip.tracks[0];
        assert_eq!(
            track.translation.interpolation,
            crate::LoadedInterpolation::Linear
        );
        assert_eq!(track.translation.keys.len(), 2);
        assert_eq!(track.scale.interpolation, crate::LoadedInterpolation::Step);
        assert_eq!(track.scale.keys.len(), 2);
        assert_eq!(track.rotation.keys.len(), 2);
        assert_eq!(track.rotation.keys[1].time, 2.0);
        match &track.rotation.interpolation {
            crate::LoadedInterpolation::Cubic {
                in_tangents,
                out_tangents,
            } => {
                assert_eq!(in_tangents.len(), 2);
                assert_eq!(out_tangents.len(), 2);
                assert_eq!(in_tangents[0], [10.0, 0.0, 0.0, 0.0]);
                assert_eq!(out_tangents[1], [21.0, 0.0, 0.0, 0.0]);
            }
            other => panic!("expected cubic interpolation, got {other:?}"),
        }
        assert_eq!(scene.stats.skipped_cubicspline, 0);
        assert_eq!(scene.stats.skipped_clips, 0);
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn malformed_cubicspline_skips_with_counter() {
        // Output count (`4`) is not `3`× the input count (`2`): genuinely
        // unassemblable, so the channel counts and the trackless clip too.
        let mut fixture = triangle();
        fixture.animations = 1;
        fixture.anim_interp = crate::fixtures::FixtureInterp::CubicSpline;
        fixture.anim_output_len_override = Some(4);
        let scene = load_slice(&build_glb(&fixture)).expect("malformed spline parses");
        assert!(scene.skel_clips.is_empty());
        assert!(scene.anim_clips.is_empty());
        assert_eq!(scene.stats.skipped_cubicspline, 1);
        assert_eq!(scene.stats.skipped_clips, 1);
        assert!(!scene.stats.is_clean());
    }

    #[test]
    fn rotation_path_assembles_normalized_quats() {
        let mut fixture = triangle();
        fixture.animations = 1;
        fixture.anim_path = crate::fixtures::FixtureAnimPath::Rotation;
        let scene = load_slice(&build_glb(&fixture)).expect("rotation parses");
        assert_eq!(scene.anim_clips.len(), 1);
        let track = &scene.anim_clips[0].tracks[0];
        assert_eq!(track.rotation.keys.len(), 2);
        for key in &track.rotation.keys {
            let length_squared: f32 = key
                .value
                .iter()
                .map(|component| component * component)
                .sum();
            assert!((length_squared - 1.0).abs() < 1e-6);
        }
        assert!(track.translation.keys.is_empty());
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn scale_path_on_joint_assembles_skel_scale() {
        let mut fixture = skinned_triangle();
        fixture.animations = 1;
        fixture.anim_path = crate::fixtures::FixtureAnimPath::Scale;
        let scene = load_slice(&build_glb(&fixture)).expect("scale parses");
        assert_eq!(scene.skel_clips.len(), 1);
        let track = &scene.skel_clips[0].tracks[0];
        assert_eq!(track.scale.keys.len(), 2);
        assert_eq!(track.scale.keys[0].value, [1.0, 1.0, 1.0]);
        assert_eq!(track.scale.keys[1].value, [2.0, 2.0, 2.0]);
        assert!(track.translation.keys.is_empty());
        assert!(scene.stats.is_clean());
    }

    #[test]
    fn malformed_skin_attributes_skip_with_counter() {
        // Joints/weights count mismatch with the positions: malformed,
        // never a stub.
        let mut fixture = skinned_triangle();
        fixture.influence = Some(FixtureInfluence {
            joints: vec![[0, 0, 0, 0]; 2],
            joints_u16: false,
            weights: vec![[1.0, 0.0, 0.0, 0.0]; 3],
            weights_kind: FixtureWeightsKind::F32,
            extra: None,
        });
        let scene = load_slice(&build_glb(&fixture)).expect("mismatched parses");
        assert!(scene.entities.is_empty());
        assert_eq!(scene.stats.skipped_skinned, 1);
        assert!(!scene.stats.is_clean());
    }

    #[test]
    fn lines_mode_skipped_with_counter() {
        let mut fixture = triangle();
        fixture.mode = Some(1);
        let scene = load_slice(&build_glb(&fixture)).expect("lines parses");
        assert!(scene.entities.is_empty());
        assert_eq!(scene.stats.skipped_non_triangle, 1);
    }

    #[test]
    fn missing_attributes_fall_back_like_custom_mesh_data() {
        let mut fixture = triangle();
        fixture.normals = None;
        fixture.uvs = None;
        let scene = load_slice(&build_glb(&fixture)).expect("bare soup parses");
        let mesh = &scene.entities[0].mesh;
        assert_eq!(mesh.normals, None);
        assert_eq!(mesh.uvs, None);
        // (1,0,0)×(0,1,0) = (0,0,1): recomputed, not transported.
        assert_eq!(mesh.resolved_normals(), vec![[0.0, 0.0, 1.0]; 3]);
        // +Z-dominant → (x, y) over the unit bounds.
        assert_eq!(
            mesh.resolved_uvs(),
            vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        );
    }

    #[test]
    fn bad_index_skipped_with_counter() {
        let mut fixture = triangle();
        fixture.indices = FixtureIndices::U16(vec![0, 1, 9]);
        let scene = load_slice(&build_glb(&fixture)).expect("bad index parses");
        assert!(scene.entities.is_empty());
        assert_eq!(scene.stats.skipped_bad_index, 1);
    }

    #[test]
    fn empty_positions_skipped_with_counter() {
        let mut fixture = triangle();
        fixture.positions = Vec::new();
        fixture.normals = None;
        fixture.uvs = None;
        let scene = load_slice(&build_glb(&fixture)).expect("empty parses");
        assert!(scene.entities.is_empty());
        assert_eq!(scene.stats.skipped_empty, 1);
    }

    #[test]
    fn unindexed_non_multiple_of_three_skipped() {
        let mut fixture = triangle();
        fixture.positions.push([2.0, 2.0, 0.0]);
        fixture.normals = Some(vec![[0.0, 0.0, 1.0]; 4]);
        fixture.uvs = Some(vec![[0.0, 0.0]; 4]);
        fixture.indices = FixtureIndices::Sequential;
        let scene = load_slice(&build_glb(&fixture)).expect("quad soup parses");
        assert!(scene.entities.is_empty());
        assert_eq!(scene.stats.skipped_bad_index, 1);
    }

    #[test]
    fn entity_naming_falls_back_to_mesh_then_position() {
        let mut fixture = triangle();
        fixture.nodes[0].name = None;
        let scene = load_slice(&build_glb(&fixture)).expect("mesh-named parses");
        assert_eq!(scene.entities[0].name, "tri-mesh");

        fixture.mesh_name = None;
        let scene = load_slice(&build_glb(&fixture)).expect("anonymous parses");
        assert_eq!(scene.entities[0].name, "mesh_0_0");
    }

    #[test]
    fn material_factors_map_with_metallic_switch() {
        let mut fixture = triangle();
        fixture.material = Some(FixtureMaterial {
            base_color: [0.25, 0.5, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.3,
            emission: [1.0, 0.0, 0.0],
        });
        let scene = load_slice(&build_glb(&fixture)).expect("material parses");
        let material = &scene.entities[0].material;
        assert_eq!(material.base_color, [0.25, 0.5, 1.0]);
        assert_eq!(material.metallic, 0.0);
        assert_eq!(material.roughness, 0.3);
        assert_eq!(material.emission, [1.0, 0.0, 0.0]);
        assert!(!material.is_metallic());

        let scene = load_triangle();
        assert!(scene.entities[0].material.is_metallic());
    }

    #[test]
    fn into_custom_drops_attributes() {
        let scene = load_triangle();
        let (positions, indices) = scene.entities[0].mesh.clone().into_custom();
        assert_eq!(positions.len(), 3);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn triangles_view_round_trips_raw() {
        use crate::{TriIndex, Triangle};
        let scene = load_triangle();
        let tris = scene.entities[0].mesh.triangles();
        assert_eq!(tris, vec![Triangle::from_raw([0, 1, 2])]);
        assert_eq!(tris[0].as_u32(), [0, 1, 2]);
        assert_eq!(tris[0].index(2), TriIndex::from_raw(2));
        assert_eq!(std::mem::size_of::<Triangle>(), 12);
    }

    #[test]
    fn garbage_bytes_fail_parse() {
        assert!(matches!(
            load_slice(b"definitely not gltf"),
            Err(ImportError::Parse { .. })
        ));
        assert!(matches!(load_slice(&[]), Err(ImportError::Parse { .. })));
    }

    #[test]
    fn parse_error_carries_the_parser_message() {
        let Err(ImportError::Parse { message }) = load_slice(b"definitely not gltf") else {
            panic!("garbage bytes must fail with Parse");
        };
        assert!(!message.is_empty(), "parser message must survive typing");
    }

    #[test]
    fn external_uri_fails_on_slice() {
        let (json, _) = fixtures::build_external_parts();
        assert!(matches!(
            load_slice(json.as_bytes()),
            Err(ImportError::ExternalBuffer(_))
        ));
    }

    #[test]
    fn sceneless_document_fails() {
        let bytes = br#"{"asset":{"version":"2.0"},"scenes":[]}"#;
        assert!(matches!(
            load_slice(bytes),
            Err(ImportError::NoScene) | Err(ImportError::Parse { .. })
        ));
    }

    #[test]
    fn counters_aggregate_across_roots() {
        let mut fixture = triangle();
        fixture.nodes.push(FixtureNode {
            name: Some("lines".to_string()),
            mesh: true,
            skin: None,
            translation: None,
            rotation: None,
            scale: None,
            matrix: None,
            children: Vec::new(),
        });
        fixture.roots = vec![0, 1];
        // Both roots share mesh 0; the mode applies to the single primitive,
        // so emulate the mix by switching the whole primitive to lines and
        // checking aggregation on a second load with triangles.
        fixture.mode = Some(1);
        let skipped = load_slice(&build_glb(&fixture)).expect("lines load");
        assert_eq!(skipped.stats.primitives_total, 2);
        assert_eq!(skipped.stats.skipped_non_triangle, 2);
        assert!(skipped.entities.is_empty());

        fixture.mode = None;
        let imported = load_slice(&build_glb(&fixture)).expect("triangles load");
        assert_eq!(imported.stats.primitives_total, 2);
        assert_eq!(imported.stats.entities, 2);
        assert!(imported.stats.is_clean());
    }

    #[test]
    fn load_path_reads_glb_and_external_gltf() {
        let dir = std::env::temp_dir().join(format!(
            "ornis-gltf-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let glb_path = dir.join("tri.glb");
        std::fs::write(&glb_path, build_glb(&triangle())).expect("write glb");
        let glb_scene = load_path(&glb_path).expect("glb loads from path");
        assert_eq!(glb_scene.entities.len(), 1);

        let (json, bin) = fixtures::build_external_parts();
        std::fs::write(dir.join("ext.gltf"), json).expect("write gltf");
        std::fs::write(dir.join("mesh.bin"), bin).expect("write bin");
        let gltf_scene = load_path(&dir.join("ext.gltf")).expect("external gltf loads");
        assert_eq!(gltf_scene.entities.len(), 1);
        assert_eq!(
            gltf_scene.entities[0].mesh.positions,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        );
        std::fs::remove_dir_all(&dir).expect("temp cleanup");
    }
}
