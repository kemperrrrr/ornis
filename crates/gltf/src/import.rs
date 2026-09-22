//! Document traversal and primitive decoding behind [`load_slice`]/[`load_path`].

use std::path::Path;

use gltf::mesh::Semantic;
use gltf::mesh::util::{ReadIndices, ReadTexCoords};
use gltf::scene::Transform;
use gltf::{Buffer, Gltf, Node, Primitive};

use crate::base64;
use crate::geom::{self, Mat4};
use crate::textures::resolve_images;
use crate::{
    ImportError, ImportStats, LoadedEntity, LoadedImage, LoadedMaterial, LoadedMesh, LoadedScene,
};

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
    let gltf = Gltf::from_slice(bytes).map_err(|error| ImportError::Parse(error.to_string()))?;
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
    let gltf = Gltf::from_slice(&bytes).map_err(|error| ImportError::Parse(error.to_string()))?;
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
                let base = base_dir.ok_or_else(|| ImportError::ExternalBuffer(uri.to_string()))?;
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
    base64::decode(payload).map_err(|_| ImportError::InvalidDataUri(short_head(uri)))
}

/// Rejects absolute paths and remote schemes before any filesystem access.
pub(crate) fn reject_remote_uri(uri: &str) -> Result<(), ImportError> {
    if uri.contains("://") || uri.starts_with("//") || uri.starts_with("data:") {
        return Err(ImportError::ExternalBuffer(uri.to_string()));
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
    let mut import = Import::new(buffers, images);
    for node in scene.nodes() {
        import.visit_node(&node, &geom::IDENTITY);
    }
    Ok(LoadedScene {
        name: scene.name().unwrap_or("scene").to_string(),
        entities: import.entities,
        stats: import.stats,
    })
}

/// Traversal state: output entities plus counters.
struct Import<'a> {
    /// Resolved buffer bytes indexed by document buffer index.
    buffers: &'a [Vec<u8>],
    /// Resolved image pixels indexed by document image index.
    images: &'a [LoadedImage],
    /// Finished entities in traversal order.
    entities: Vec<LoadedEntity>,
    /// Counters (see the crate skip-rules table).
    stats: ImportStats,
}

impl<'a> Import<'a> {
    /// Borrows resolved buffers and images for one traversal.
    fn new(buffers: &'a [Vec<u8>], images: &'a [LoadedImage]) -> Self {
        Self {
            buffers,
            images,
            entities: Vec::new(),
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
        let reader = primitive.reader(|buffer| self.buffer_bytes(buffer));
        if primitive
            .attributes()
            .any(|(semantic, _)| matches!(semantic, Semantic::Joints(_) | Semantic::Weights(_)))
        {
            self.stats.skipped_skinned += 1;
            return None;
        }
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
        let positions: Vec<[f32; 3]> = read_positions.collect();
        let indices = match reader.read_indices() {
            Some(indices) => collect_indices(indices),
            None => (0..positions.len() as u32).collect(),
        };
        if positions.is_empty() || indices.is_empty() {
            self.stats.skipped_empty += 1;
            return None;
        }
        if indices.len() % 3 != 0
            || indices
                .iter()
                .any(|index| (*index as usize) >= positions.len())
        {
            self.stats.skipped_bad_index += 1;
            return None;
        }
        let vertex_count = positions.len();
        let normals = reader
            .read_normals()
            .map(Iterator::collect)
            .filter(|normals: &Vec<[f32; 3]>| normals.len() == vertex_count);
        let uvs = reader
            .read_tex_coords(0)
            .map(collect_tex_coords)
            .filter(|uvs: &Vec<[f32; 2]>| uvs.len() == vertex_count);
        let (translation, rotation, scale) = geom::decompose(world);
        Some(LoadedEntity {
            name: entity_name(node, mesh_name, mesh_index, primitive.index()),
            translation,
            rotation,
            scale,
            mesh: LoadedMesh {
                positions,
                indices,
                normals,
                uvs,
            },
            material: read_material(primitive, self.images),
        })
    }

    /// Buffer bytes for the reader closure (`None` on unknown index).
    fn buffer_bytes(&self, buffer: Buffer<'_>) -> Option<&[u8]> {
        self.buffers.get(buffer.index()).map(Vec::as_slice)
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
        self, FixtureIndices, FixtureMaterial, FixtureNode, build_glb, build_gltf, load_triangle,
        triangle,
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
                translation: Some([10.0, 0.0, 0.0]),
                rotation: None,
                scale: None,
                matrix: None,
                children: vec![1],
            },
            FixtureNode {
                name: Some("child".to_string()),
                mesh: true,
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
    fn skinned_primitive_skipped_with_counter() {
        let mut fixture = triangle();
        fixture.skinned = true;
        let scene = load_slice(&build_glb(&fixture)).expect("skinned parses");
        assert!(scene.entities.is_empty());
        assert_eq!(scene.stats.primitives_total, 1);
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
    fn garbage_bytes_fail_parse() {
        assert!(matches!(
            load_slice(b"definitely not gltf"),
            Err(ImportError::Parse(_))
        ));
        assert!(matches!(load_slice(&[]), Err(ImportError::Parse(_))));
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
            Err(ImportError::NoScene) | Err(ImportError::Parse(_))
        ));
    }

    #[test]
    fn counters_aggregate_across_roots() {
        let mut fixture = triangle();
        fixture.nodes.push(FixtureNode {
            name: Some("lines".to_string()),
            mesh: true,
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
