//! ECS-backed render extraction shared by native and WASM runtimes.
//!
//! Scene descriptions are deserialized at the serialization boundary and
//! inserted as `TransformDesc`, `MeshDesc` and `MaterialDesc` component
//! lanes; the frame payload is read directly from the lanes on demand
//! ([`extract_render_data`] → [`FrameUpload`], X4/Extract-free — no
//! scheduled snapshot round-trip). The scene-backed world itself lives in
//! `ornis_app::GameWorld` — one type for the authoritative host and the
//! browser replica; this module keeps the extraction canon plus the
//! deprecated `RenderWorld` shim below.
//!
//! GPU resources and cameras remain owned by the platform renderer;
//! lighting is the [`RenderLights`] resource (X3). This module
//! deliberately stops at CPU-side instance/material data. That keeps the
//! server/editor world authoritative while allowing a native or browser
//! client to build its own physical GPU representation.

use glam::Mat4;
use glam::Vec3;
use ornis_animation::SkinnedMesh;
use ornis_animation::SkinningMode;
use ornis_assets::scene::MaterialDesc;
use ornis_assets::scene::MeshDesc;
use ornis_assets::scene::Scene;
use ornis_assets::scene::TransformDesc;
use ornis_core::Engine;
use ornis_core::Entity;
use ornis_core::GlobalTransform;
use ornis_core::OpenPBRMaterial;
use ornis_core::SmartStore;
use ornis_core::units::PositiveF32;

use crate::mesh_upload::SoupCache;
use crate::mesh_upload::UploadCache;
use crate::renderer::InstanceData;

mod cull;
mod lights;
mod materials;
mod skinning;
#[cfg(test)]
mod test_util;

pub use cull::CullStats;
pub use cull::cull_frame_upload;
pub use cull::cull_frame_upload_timed;
pub use cull::instance_sphere;
pub use cull::instance_view_depth;
pub use cull::sort_by_depth;
pub use cull::sort_by_depth_timed;
pub use lights::RenderLights;
pub use skinning::CustomMeshEntry;
pub use skinning::MeshPose;
pub use skinning::SkinInfluences;

use self::materials::deduped_material_index;
use self::skinning::bind_pose_entry;
use self::skinning::gpu_joint_palette;
use self::skinning::skinned_entry;

/// Indices per triangle (flat soup alignment).
const TRIANGLE_VERTS: usize = 3;

/// CPU-side render data read from the ECS lanes for one frame (X4
/// Extract-free: a direct-read payload, not a scheduled snapshot —
/// no `Mutex` round-trip).
#[derive(Clone, Debug)]
pub struct FrameUpload {
    /// Maximum sphere tessellation required by the extracted entities.
    pub mesh_params: (u32, u32),
    /// Deduplicated GPU-ready materials: identical [`MaterialDesc`]
    /// values share one entry, and every `instance.material_index`
    /// (both [`Self::instances`] and [`Self::custom_meshes`]) points
    /// into this table.
    pub materials: Vec<OpenPBRMaterial>,
    /// Per-entity model/normal matrices and material indices (procedural
    /// meshes only — drawn with the shared unit meshes, size baked into
    /// the model matrix).
    pub instances: Vec<InstanceData>,
    /// Per-entity custom geometry (one entry per valid `MeshDesc::Custom`
    /// entity — drawn with its own uploaded mesh, never the sphere).
    pub custom_meshes: Vec<CustomMeshEntry>,
}
/// Tessellation floor when no complete renderable entity asks for more
/// (the `FrameUpload::default` `mesh_params`).
const DEFAULT_MESH_PARAMS: (u32, u32) = (32, 24);

impl Default for FrameUpload {
    fn default() -> Self {
        Self {
            mesh_params: DEFAULT_MESH_PARAMS,
            materials: Vec::new(),
            instances: Vec::new(),
            custom_meshes: Vec::new(),
        }
    }
}
/// Deprecated scene world: use `ornis_app::GameWorld` instead.
///
/// Retained (without the `deprecated` attribute, so the stranded
/// `crates/render/tests` users keep passing `clippy -D warnings`) only
/// until those integration tests migrate to the single world type — new
/// code must construct `GameWorld`. Behavior matches `GameWorld`
/// one-to-one: the same [`Engine`], the same scene lanes, the same
/// extraction canon.
pub struct RenderWorld {
    engine: Engine,
    entities: Vec<Entity>,
}

impl Default for RenderWorld {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderWorld {
    /// Creates an empty render world — the scene-loader host (X4: no
    /// scheduled extraction pass; the frame payload is read directly
    /// via [`extract_render_data`]).
    pub fn new() -> Self {
        Self {
            engine: Engine::new(),
            entities: Vec::new(),
        }
    }

    /// Creates a render world populated from a serialized scene description.
    pub fn from_scene(scene: &Scene) -> Self {
        let mut world = Self::new();
        world.replace_scene(scene);
        world
    }

    /// Returns the logical engine used by this render world.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Returns the logical engine for controlled setup or custom systems.
    ///
    /// Scene replacement and frame execution should normally use
    /// [`Self::replace_scene`] and [`Self::run_frame`] so the entity list and
    /// extraction resource remain consistent.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Number of render entities currently represented in the ECS.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Returns the handles of entities populated from the current scene.
    ///
    /// The slice excludes auxiliary entities inserted through
    /// [`Self::engine_mut`], which lets a platform attach hidden runtime
    /// components such as physics bodies without changing the render count.
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Replaces the renderable ECS entities with `scene.entities` and
    /// publishes the scene lighting as the [`RenderLights`] resource (X3).
    ///
    /// The camera stays frame/view state owned by the caller; lights and
    /// ambient are world state now — `RenderSubmit` reads the resource
    /// instead of a hardcoded rig. The next [`Self::run_frame`] refreshes
    /// the extracted snapshot.
    pub fn replace_scene(&mut self, scene: &Scene) {
        let previous = std::mem::take(&mut self.entities);
        if let Some(store) = self.engine.world().store() {
            for entity in previous {
                if store.is_alive(entity) {
                    store.destroy_entity(entity);
                }
            }
        }
        self.entities = insert_scene_entities(&mut self.engine, &scene.entities);
        let _ = self
            .engine
            .world_mut()
            .insert(RenderLights::from_scene(scene));
    }

    /// Publishes time and runs the world schedule for one frame.
    pub fn run_frame(&mut self, delta_seconds: f32) {
        self.engine.run_frame(delta_seconds);
    }

    /// Reads the frame payload directly from the component lanes.
    ///
    /// Equivalent to calling [`extract_render_data`] on this world's
    /// store; provided for callers that own the [`RenderWorld`].
    pub fn frame_upload(&self) -> FrameUpload {
        match self.engine.world().store() {
            Some(store) => extract_render_data(store),
            None => FrameUpload::default(),
        }
    }
}
/// Per-frame honesty counters for [`extract_render_data_with_stats`]:
/// every entity the extraction skips, and every material-table reuse, is
/// counted here instead of dropped silently. All counters start at zero;
/// a clean scene extracts with every counter at zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtractionStats {
    /// Entities skipped for missing mesh or material components
    /// (incomplete lanes — never uploaded, never stubbed).
    pub skipped_incomplete: u32,
    /// `MeshDesc::Custom` entities skipped for an empty soup or a soup
    /// failing validation (no panic, no sphere stub).
    pub skipped_bad_custom: u32,
    /// `SkinnedMesh` entities skipped for inconsistent skin-lane arrays
    /// (length defects, empty binds, bad indices — see
    /// [`extract_render_data_with_stats`]). Missing skeleton/pose freshness
    /// is the skin system's own counter; extraction only trusts the lane
    /// buffers it can validate here.
    pub skipped_bad_skin: u32,
    /// Entities reaching the match-exhaustiveness fallback for mesh
    /// variants the router does not know. Never fires for the known
    /// variants (`Sphere`, `Box`, `Plane`, `Cylinder`, `Custom`) —
    /// nonzero means the router missed a case.
    pub skipped_unknown_mesh: u32,
    /// Material-table reuses: times an identical [`MaterialDesc`] shared
    /// an existing [`FrameUpload::materials`] entry instead of pushing a
    /// new one.
    pub materials_deduped: u32,
    /// Custom-soup conversion cache hits: identical soups reused without
    /// recomputing normals/UVs (see [`SoupCache`], [`UploadCache::Hit`]).
    pub custom_cache_hits: u32,
    /// Custom-soup conversion cache misses: distinct soups converted
    /// this frame (see [`UploadCache::Miss`]).
    pub custom_cache_misses: u32,
}
/// Pose-bearing entities in extraction order.
///
/// [`TransformDesc`] lane order comes first, so flat scenes stay stable.
/// [`GlobalTransform`] entities with no [`TransformDesc`] follow: a spawned
/// model subtree carries the world pose without a per-node desc.
fn posed_entities(store: &SmartStore) -> Vec<Entity> {
    let mut order = Vec::new();
    if let Some(transforms) = store.read_lane::<TransformDesc>() {
        order.extend(transforms.entities.iter().copied());
    }
    if let Some(globals) = store.read_lane::<GlobalTransform>() {
        let transforms = store.read_lane::<TransformDesc>();
        for &entity in &globals.entities {
            if transforms
                .as_ref()
                .is_none_or(|lane| lane.get(entity).is_none())
            {
                order.push(entity);
            }
        }
    }
    order
}

/// World TRS for one renderable.
///
/// [`GlobalTransform`] is the hierarchy pose. Entities that have not been
/// propagated still use [`TransformDesc`], which flat scenes store in world
/// space. [`None`] when the entity has neither.
fn world_trs(
    global: Option<&GlobalTransform>,
    desc: Option<&TransformDesc>,
) -> Option<(Vec3, glam::Quat, Vec3)> {
    if let Some(global) = global {
        Some((global.translation, global.rotation.get(), global.scale))
    } else {
        desc.map(|desc| (desc.translation, desc.rotation.get(), desc.scale))
    }
}
/// Extracts complete renderable entities from the ECS store.
///
/// Thin wrapper over [`extract_render_data_with_stats`] discarding the
/// honesty counters; use the `_with_stats` variant when skip visibility
/// matters. See there for the extraction canon.
pub fn extract_render_data(store: &SmartStore) -> FrameUpload {
    extract_render_data_with_stats(store).0
}
/// Extracts complete renderable entities from the ECS store, counting
/// every skip.
///
/// Entities missing mesh, material, or a pose are skipped
/// ([`ExtractionStats::skipped_incomplete`]). A pose is
/// [`GlobalTransform`] when that component is present, otherwise
/// [`TransformDesc`]. Dense [`TransformDesc`] lane order is the
/// deterministic extraction order; [`GlobalTransform`] entities with no
/// desc follow it. Identical [`MaterialDesc`]
/// values share one [`FrameUpload::materials`] entry (see
/// [`deduped_material_index`]), so `materials.len()` is the number of
/// *distinct* materials, not entities. A [`ornis_core::Surface`] on the
/// entity is applied after that conversion and is part of the dedup key.
///
/// Entities carrying the [`SkinnedMesh`] lane never take the classic paths
/// below: GPU-staged skins land in `custom_meshes` with
/// [`SkinningMode::Gpu`], bind-pose `vertices` plus staged influences
/// ([`MeshPose::Bind`]) and the joint palette (see
/// [`CustomMeshEntry::joint_palette`]); the [`SkinningMode::Cpu`]
/// fallback keeps the skin-system output buffers (`skinned_positions` /
/// `skinned_normals`, world space, [`MeshPose::Skinned`]) on the classic
/// path, and `IDENTITY` matrices either way (design §2.3).
/// Inconsistent lane arrays skip the entity with
/// [`ExtractionStats::skipped_bad_skin`] — even when its classic soup
/// would decode, a claimed skin must not silently render unskinned.
pub fn extract_render_data_with_stats(store: &SmartStore) -> (FrameUpload, ExtractionStats) {
    let mut extracted = FrameUpload::default();
    let mut stats = ExtractionStats::default();
    let order = posed_entities(store);
    if order.is_empty() {
        return (extracted, stats);
    }
    let transforms = store.read_lane::<TransformDesc>();
    let globals = store.read_lane::<GlobalTransform>();
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return (extracted, stats);
    };
    let Some(materials) = store.read_lane::<MaterialDesc>() else {
        return (extracted, stats);
    };
    let skinned = store.read_lane::<SkinnedMesh>();
    let surfaces = store.read_lane::<ornis_core::Surface>();

    // Parallel to `extracted.materials`: the source desc plus an optional
    // surface override, for exact (`PartialEq`) dedup. Linear scan is
    // fine — frames hold tens of distinct materials, not thousands.
    let mut seen: Vec<(MaterialDesc, Option<ornis_core::Surface>)> = Vec::new();
    // Per-frame Custom soup conversion cache: identical soups convert
    // once (see `mesh_upload::SoupCache`). Staging is reserved once from
    // the lane length (capped): the map grows at most once per frame and
    // `custom_meshes` amortizes its pushes the same way.
    /// Cap on per-frame Custom soup cache entries (amortized staging).
    const SOUP_CACHE_CAP: usize = 4096;
    /// Reserved custom-mesh upload slots per frame.
    const CUSTOM_MESH_RESERVE: usize = 256;
    let lane_len = order.len();
    let mut soup_cache = SoupCache::with_capacity(lane_len.min(SOUP_CACHE_CAP));
    extracted
        .custom_meshes
        .reserve(lane_len.min(CUSTOM_MESH_RESERVE));
    for entity in order {
        let Some(mesh) = meshes.get(entity) else {
            stats.skipped_incomplete += 1;
            continue;
        };
        let Some(material) = materials.get(entity) else {
            stats.skipped_incomplete += 1;
            continue;
        };
        // Skin path first: a `SkinnedMesh` lane claims the entity for the
        // pre-skinned world-space payload (any `MeshDesc` variant — the
        // lane's own bind arrays are authoritative, the desc only keeps
        // the completeness triple). Lane defects skip with the skin
        // counter; the classic soup below never runs for claimed entities.
        // Phase D: the joint palette stages alongside (GPU mode) whenever
        // the skeleton validates — otherwise the CPU fallback (same
        // vertices, no palette).
        if let Some(skin) = skinned.as_ref().and_then(|lane| lane.get(entity)) {
            let material_index = deduped_material_index(
                &mut extracted,
                &mut seen,
                material,
                surfaces.as_ref().and_then(|lane| lane.get(entity)).copied(),
                &mut stats,
            );
            let (skinning, joint_palette) = match gpu_joint_palette(store, skin) {
                Some(palette) => (SkinningMode::Gpu, Some(palette)),
                None => (SkinningMode::Cpu, None),
            };
            // GPU entries carry bind-pose rows for the skinned vertex
            // stage (one-line switch now that the draw path binds the
            // palette); CPU entries and every fallback keep the
            // pre-skinned rows on the classic path, pixel-identical.
            let (vertices, indices, pose) = match skinning {
                SkinningMode::Gpu => {
                    let Some((vertices, indices, influences)) = bind_pose_entry(skin) else {
                        stats.skipped_bad_skin += 1;
                        continue;
                    };
                    (vertices, indices, MeshPose::Bind(influences))
                }
                SkinningMode::Cpu => {
                    let Some((vertices, indices)) = skinned_entry(skin) else {
                        stats.skipped_bad_skin += 1;
                        continue;
                    };
                    (vertices, indices, MeshPose::Skinned)
                }
            };
            extracted.custom_meshes.push(CustomMeshEntry {
                vertices,
                indices,
                instance: InstanceData {
                    model_matrix: Mat4::IDENTITY,
                    normal_matrix: Mat4::IDENTITY,
                    material_index,
                },
                skinning,
                joint_palette,
                pose,
            });
            continue;
        }
        let Some((translation, rotation, scale)) = world_trs(
            globals.as_ref().and_then(|lane| lane.get(entity)),
            transforms.as_ref().and_then(|lane| lane.get(entity)),
        ) else {
            stats.skipped_incomplete += 1;
            continue;
        };
        // Per-entity Custom path: CPU-side vertices via the mesh_upload
        // bridge, deduplicated by soup hash within the frame (identical
        // soups convert once). An empty or invalid soup skips the entity —
        // no panic, no sphere stub, no invented collider (same honesty as
        // the physics TriMesh import, which has no sphere proxy).
        // Materials stay in lockstep with both instance lanes because
        // every push below goes through [`deduped_material_index`].
        if let Some((positions, indices)) = mesh.as_custom() {
            if positions.is_empty() || indices.is_empty() {
                stats.skipped_bad_custom += 1;
                continue;
            }
            let Ok(((vertices, soup_indices), outcome)) =
                soup_cache.get_or_convert(positions, indices)
            else {
                stats.skipped_bad_custom += 1;
                continue;
            };
            match outcome {
                UploadCache::Hit => {
                    stats.custom_cache_hits = stats.custom_cache_hits.saturating_add(1);
                }
                UploadCache::Miss => {
                    stats.custom_cache_misses = stats.custom_cache_misses.saturating_add(1);
                }
            }
            let model = Mat4::from_scale_rotation_translation(scale, rotation, translation);
            let material_index = deduped_material_index(
                &mut extracted,
                &mut seen,
                material,
                surfaces.as_ref().and_then(|lane| lane.get(entity)).copied(),
                &mut stats,
            );
            let instance = InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index,
            };
            extracted.custom_meshes.push(CustomMeshEntry {
                vertices,
                indices: soup_indices,
                instance,
                // Classic soup path: bind-pose geometry, never pre-skinned.
                skinning: SkinningMode::Cpu,
                joint_palette: None,
                pose: MeshPose::Skinned,
            });
            continue;
        }
        // Shared procedural batch: every non-Custom primitive lands in
        // `instances` (drawn with its shared unit mesh — the primitive's
        // size is baked into the model scale, exactly like the sphere's
        // radius). Only spheres feed the shared tessellation maximum.
        let size_scale = match mesh {
            MeshDesc::Sphere {
                radius,
                segments,
                rings,
            } => {
                extracted.mesh_params.0 = extracted.mesh_params.0.max(*segments);
                extracted.mesh_params.1 = extracted.mesh_params.1.max(*rings);
                scale * radius.get()
            }
            MeshDesc::Box { size } => scale * Vec3::from_array(size.map(PositiveF32::get)),
            MeshDesc::Plane { size } => {
                Vec3::new(scale[0] * size[0].get(), scale[1], scale[2] * size[1].get())
            }
            MeshDesc::Cylinder { radius, height, .. } => Vec3::new(
                scale[0] * radius.get(),
                scale[1] * height.get(),
                scale[2] * radius.get(),
            ),
            // `as_custom` above handles every `Custom`; this arm is only
            // the match-exhaustiveness fallback (never a stub).
            MeshDesc::Custom { .. } => {
                stats.skipped_unknown_mesh += 1;
                continue;
            }
        };
        let model = Mat4::from_scale_rotation_translation(size_scale, rotation, translation);
        let material_index = deduped_material_index(
            &mut extracted,
            &mut seen,
            material,
            surfaces.as_ref().and_then(|lane| lane.get(entity)).copied(),
            &mut stats,
        );
        extracted.instances.push(InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index,
        });
    }
    (extracted, stats)
}
/// Maximum sphere tessellation over complete renderable entities — the
/// GPU mesh re-create criterion (X2, Extract-free).
///
/// The same canon as [`extract_render_data`]: entities missing mesh,
/// material, or a pose ([`GlobalTransform`] or [`TransformDesc`]) are
/// skipped (even for the maximum), and the result never falls below the
/// `FrameUpload::default` floor (32, 24). Iterator form: the lane walk
/// lives in closures (the sanctioned lenient form, `rustqual.toml`).
pub fn max_mesh_params(store: &SmartStore) -> (u32, u32) {
    let order = posed_entities(store);
    if order.is_empty() {
        return DEFAULT_MESH_PARAMS;
    }
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    let Some(materials) = store.read_lane::<MaterialDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    order
        .iter()
        // Complete entities only: pose plus mesh and material.
        .filter(|&&entity| meshes.get(entity).is_some() && materials.get(entity).is_some())
        .filter_map(|&entity| meshes.get(entity))
        .fold(DEFAULT_MESH_PARAMS, |params, mesh| match mesh {
            MeshDesc::Sphere {
                segments, rings, ..
            } => (params.0.max(*segments), params.1.max(*rings)),
            // Custom soups don't feed the shared sphere tessellation.
            MeshDesc::Custom { .. } => params,
            // Other procedural variants don't feed it either.
            MeshDesc::Box { .. } | MeshDesc::Plane { .. } | MeshDesc::Cylinder { .. } => params,
        })
}
fn insert_scene_entities(
    engine: &mut Engine,
    entities: &[ornis_assets::scene::EntityDesc],
) -> Vec<Entity> {
    let Some(store) = engine.world_mut().store_mut() else {
        return Vec::new();
    };
    let mut handles = Vec::with_capacity(entities.len());
    for entity in entities {
        let handle = store.create_entity();
        store.insert(handle, entity.transform.clone());
        store.insert(handle, entity.mesh.clone());
        store.insert(handle, entity.material.clone());
        handles.push(handle);
    }
    handles
}
#[cfg(test)]
mod tests {
    use super::test_util::push_test_entity;
    use super::test_util::test_material;
    use super::test_util::test_quad;
    use super::test_util::test_sphere;
    use super::*;
    use ornis_assets::scene::MaterialDesc;
    use ornis_assets::scene::MeshDesc;
    use ornis_assets::scene::TransformDesc;
    use ornis_core::Engine;
    use ornis_core::units::Clamped01;

    #[test]
    fn extraction_uses_global_transform_not_the_local_desc() {
        use ornis_core::{SmartStore, Transform, propagate_transforms, set_parent};
        let mut store = SmartStore::new();
        let parent = store.create_entity();
        let child = store.create_entity();
        store.insert(
            parent,
            Transform::from_translation(glam::Vec3::new(10.0, 0.0, 0.0)),
        );
        store.insert(child, Transform::from_translation(glam::Vec3::X));
        store.insert(child, TransformDesc::from_translation(glam::Vec3::X));
        store.insert(
            child,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 8,
                rings: 6,
            },
        );
        store.insert(
            child,
            MaterialDesc::Dielectric {
                base_color: [0.8, 0.2, 0.2],
                roughness: Clamped01::new(0.4),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(0.0),
            },
        );
        set_parent(&mut store, child, parent).expect("parent");
        propagate_transforms(&mut store);
        let extracted = extract_render_data(&store);
        assert_eq!(extracted.instances.len(), 1);
        let translation = extracted.instances[0].model_matrix.w_axis.truncate();
        assert!(
            (translation - glam::Vec3::new(11.0, 0.0, 0.0)).length() < 1e-4,
            "model matrix follows the world pose, got {translation:?}"
        );
    }

    #[test]
    fn extraction_draws_global_transform_without_transform_desc() {
        use ornis_core::{GlobalTransform, SmartStore, Transform};
        let mut store = SmartStore::new();
        let flat = store.create_entity();
        store.insert(
            flat,
            TransformDesc::from_translation(glam::Vec3::new(-2.0, 0.0, 0.0)),
        );
        store.insert(
            flat,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 8,
                rings: 6,
            },
        );
        store.insert(
            flat,
            MaterialDesc::Dielectric {
                base_color: [0.2, 0.8, 0.2],
                roughness: Clamped01::new(0.4),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(0.0),
            },
        );
        let posed = store.create_entity();
        store.insert(
            posed,
            GlobalTransform::from_local(Transform::from_translation(glam::Vec3::new(
                4.0, 0.0, 0.0,
            ))),
        );
        store.insert(
            posed,
            MeshDesc::Sphere {
                radius: PositiveF32::expect_valid(1.0),
                segments: 40,
                rings: 30,
            },
        );
        store.insert(
            posed,
            MaterialDesc::Dielectric {
                base_color: [0.8, 0.2, 0.2],
                roughness: Clamped01::new(0.4),
                emission: [0.0, 0.0, 0.0],
                metallic: ornis_core::Metallic::new(0.0),
            },
        );
        let extracted = extract_render_data(&store);
        assert_eq!(extracted.instances.len(), 2);
        let flat_translation = extracted.instances[0].model_matrix.w_axis.truncate();
        let posed_translation = extracted.instances[1].model_matrix.w_axis.truncate();
        assert!(
            (flat_translation - glam::Vec3::new(-2.0, 0.0, 0.0)).length() < 1e-4,
            "desc-only entity still draws, got {flat_translation:?}"
        );
        assert!(
            (posed_translation - glam::Vec3::new(4.0, 0.0, 0.0)).length() < 1e-4,
            "global-only entity draws, got {posed_translation:?}"
        );
        assert_eq!(max_mesh_params(&store), (40, 30));
    }

    #[test]
    fn custom_quad_routes_per_entity_and_bad_soups_skip_without_stub() {
        // One sphere + one valid Custom quad + one empty + one invalid:
        // the sphere stays in the shared batch, the quad lands in
        // `custom_meshes` with its own geometry, and both bad soups skip
        // the entity entirely (no sphere stub, no material leak). Both
        // valid entities share one material, so dedup yields one entry.
        let mut engine = Engine::new();
        let mut add = |mesh: MeshDesc| {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(handle, TransformDesc::IDENTITY);
            store.insert(handle, mesh);
            store.insert(
                handle,
                MaterialDesc::Dielectric {
                    base_color: [0.8, 0.2, 0.2],
                    roughness: Clamped01::new(0.4),
                    emission: [0.0, 0.0, 0.0],
                    metallic: ornis_core::Metallic::new(0.0),
                },
            );
        };
        add(MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 16,
            rings: 12,
        });
        add(MeshDesc::Custom {
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        });
        add(MeshDesc::Custom {
            positions: Vec::new(),
            indices: Vec::new(),
        });
        add(MeshDesc::Custom {
            positions: vec![[0.0, 0.0, 0.0]],
            indices: vec![0, 0, 7],
        });

        let extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 1, "sphere batch only");
        assert_eq!(extracted.custom_meshes.len(), 1, "only the valid quad");
        // Both valid entities share one material: dedup yields one entry,
        // no material leak from skips.
        assert_eq!(extracted.materials.len(), 1, "deduped, no leak");
        assert_eq!(
            extracted.instances[0].material_index,
            crate::renderer::MaterialIdx::from_raw(0)
        );
        let entry = &extracted.custom_meshes[0];
        assert_eq!(entry.vertices.len(), 4);
        assert_eq!(entry.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(
            entry.instance.material_index,
            crate::renderer::MaterialIdx::from_raw(0)
        );
        // Custom geometry never feeds the shared sphere tessellation.
        assert_eq!(extracted.mesh_params, (32, 24));
    }

    #[test]
    fn box_plane_cylinder_route_to_shared_batch_with_baked_size() {
        // One sphere + Box + Plane + Cylinder + two identical Custom
        // quads: all four procedurals land in `instances` (never in
        // `custom_meshes`), the quads land per-entity with equal geometry
        // (frame cache dedups the conversion), and only the sphere feeds
        // the tessellation maximum. All entities share one material, so
        // dedup yields one entry.
        let material = || MaterialDesc::Dielectric {
            base_color: [0.8, 0.2, 0.2],
            roughness: Clamped01::new(0.4),
            emission: [0.0, 0.0, 0.0],
            metallic: ornis_core::Metallic::new(0.0),
        };
        let mut engine = Engine::new();
        let mut add = |mesh: MeshDesc| {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(handle, TransformDesc::IDENTITY);
            store.insert(handle, mesh);
            store.insert(handle, material());
        };
        add(MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 48,
            rings: 32,
        });
        add(MeshDesc::Box {
            size: [
                PositiveF32::expect_valid(2.0),
                PositiveF32::expect_valid(4.0),
                PositiveF32::expect_valid(6.0),
            ],
        });
        add(MeshDesc::Plane {
            size: [
                PositiveF32::expect_valid(3.0),
                PositiveF32::expect_valid(5.0),
            ],
        });
        add(MeshDesc::Cylinder {
            radius: PositiveF32::expect_valid(2.0),
            height: PositiveF32::expect_valid(7.0),
            radial_segments: 64,
        });
        let quad = MeshDesc::Custom {
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        add(quad.clone());
        add(quad);

        let extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 4, "all procedurals shared");
        assert_eq!(extracted.custom_meshes.len(), 2, "both quads per-entity");
        assert_eq!(extracted.materials.len(), 1, "deduped");
        // Tessellation max comes from the sphere only: Box/Plane/Cylinder
        // (even radial_segments 64) never feed it.
        assert_eq!(extracted.mesh_params, (48, 32));
        // Size baked into the model scale: axis lengths of each instance.
        let axis_lengths = |model: glam::Mat4| {
            [
                model.x_axis.length(),
                model.y_axis.length(),
                model.z_axis.length(),
            ]
        };
        assert_eq!(
            axis_lengths(extracted.instances[0].model_matrix),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(
            axis_lengths(extracted.instances[1].model_matrix),
            [2.0, 4.0, 6.0]
        );
        assert_eq!(
            axis_lengths(extracted.instances[2].model_matrix),
            [3.0, 1.0, 5.0]
        );
        assert_eq!(
            axis_lengths(extracted.instances[3].model_matrix),
            [2.0, 7.0, 2.0]
        );
        // Identical soups convert once: equal geometry in both entries.
        let (first, second) = (&extracted.custom_meshes[0], &extracted.custom_meshes[1]);
        assert_eq!(first.indices, second.indices);
        assert_eq!(first.vertices.len(), second.vertices.len());
        for (a, b) in first.vertices.iter().zip(&second.vertices) {
            assert_eq!(a.position, b.position);
            assert_eq!(a.normal, b.normal);
        }
    }

    #[test]
    fn incomplete_entities_are_skipped() {
        let mut engine = Engine::new();
        let entity = engine
            .world_mut()
            .store_mut()
            .expect("store")
            .create_entity();
        engine
            .world_mut()
            .store_mut()
            .expect("store")
            .insert(entity, TransformDesc::IDENTITY);
        assert!(
            extract_render_data(engine.world().store().expect("store"))
                .instances
                .is_empty()
        );
    }

    #[test]
    fn stats_count_incomplete_entities() {
        // One complete sphere plus one entity without mesh and one
        // without material: only the sphere extracts, both incomplete
        // entities land in `skipped_incomplete`, nothing else fires.
        let mut engine = Engine::new();
        push_test_entity(&mut engine, Some(test_sphere()), Some(test_material()));
        push_test_entity(&mut engine, None, Some(test_material()));
        push_test_entity(&mut engine, Some(test_sphere()), None);

        let (extracted, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 1);
        assert_eq!(
            stats,
            ExtractionStats {
                skipped_incomplete: 2,
                skipped_bad_custom: 0,
                skipped_bad_skin: 0,
                skipped_unknown_mesh: 0,
                materials_deduped: 0,
                custom_cache_hits: 0,
                custom_cache_misses: 0,
            }
        );
    }

    #[test]
    fn stats_count_bad_custom_soups() {
        // One valid quad plus one empty and one invalid soup: only the
        // quad extracts, both bad soups land in `skipped_bad_custom`.
        let mut engine = Engine::new();
        push_test_entity(&mut engine, Some(test_quad()), Some(test_material()));
        push_test_entity(
            &mut engine,
            Some(MeshDesc::Custom {
                positions: Vec::new(),
                indices: Vec::new(),
            }),
            Some(test_material()),
        );
        push_test_entity(
            &mut engine,
            Some(MeshDesc::Custom {
                positions: vec![[0.0, 0.0, 0.0]],
                indices: vec![0, 0, 7],
            }),
            Some(test_material()),
        );

        let (extracted, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(extracted.custom_meshes.len(), 1);
        assert_eq!(extracted.instances.len(), 0);
        assert_eq!(stats.skipped_bad_custom, 2);
        assert_eq!(stats.skipped_incomplete, 0);
        assert_eq!(stats.skipped_unknown_mesh, 0);
    }

    #[test]
    fn stats_unknown_mesh_stays_zero_for_known_variants() {
        // Every known mesh variant routes without touching the
        // match-exhaustiveness fallback: the counter pins that invariant.
        let mut engine = Engine::new();
        for mesh in [
            test_sphere(),
            MeshDesc::Box {
                size: [
                    PositiveF32::expect_valid(2.0),
                    PositiveF32::expect_valid(4.0),
                    PositiveF32::expect_valid(6.0),
                ],
            },
            MeshDesc::Plane {
                size: [
                    PositiveF32::expect_valid(3.0),
                    PositiveF32::expect_valid(5.0),
                ],
            },
            MeshDesc::Cylinder {
                radius: PositiveF32::expect_valid(2.0),
                height: PositiveF32::expect_valid(7.0),
                radial_segments: 12,
            },
            test_quad(),
        ] {
            push_test_entity(&mut engine, Some(mesh), Some(test_material()));
        }

        let (extracted, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 4);
        assert_eq!(extracted.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_unknown_mesh, 0);
        assert_eq!(stats.skipped_incomplete, 0);
        assert_eq!(stats.skipped_bad_custom, 0);
    }

    #[test]
    fn stats_count_deduped_material_reuses() {
        // Three entities share one material (two reuses) plus one distinct
        // material: the table holds two entries, the counter two reuses.
        let mut engine = Engine::new();
        let other = MaterialDesc::Matte {
            base_color: [0.2, 0.2, 0.2],
            roughness: Clamped01::new(0.8),
            metallic: ornis_core::Metallic::new(0.0),
        };
        for material in [test_material(), test_material(), test_material(), other] {
            push_test_entity(&mut engine, Some(test_sphere()), Some(material));
        }

        let (extracted, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 4);
        assert_eq!(extracted.materials.len(), 2);
        assert_eq!(stats.materials_deduped, 2);
        assert_eq!(stats.skipped_incomplete, 0);
    }

    #[test]
    fn wrapper_matches_with_stats_payload() {
        // The legacy signature stays a thin wrapper: same payload as the
        // `_with_stats` variant on a mixed scene.
        let mut engine = Engine::new();
        push_test_entity(&mut engine, Some(test_sphere()), Some(test_material()));
        push_test_entity(&mut engine, Some(test_quad()), Some(test_material()));
        push_test_entity(&mut engine, None, Some(test_material()));

        let store = engine.world().store().expect("store");
        let plain = extract_render_data(store);
        let (with_stats, _) = extract_render_data_with_stats(store);
        assert_eq!(plain.instances.len(), with_stats.instances.len());
        assert_eq!(plain.custom_meshes.len(), with_stats.custom_meshes.len());
        assert_eq!(plain.materials.len(), with_stats.materials.len());
        assert_eq!(plain.mesh_params, with_stats.mesh_params);
    }

    #[test]
    fn identical_soups_convert_once_and_count_hits() {
        // Two identical quads plus one distinct triangle: the frame
        // converts two distinct soups (2 misses), the repeated quad is a
        // cache hit, and both quad entries carry equal geometry.
        let mut engine = Engine::new();
        push_test_entity(&mut engine, Some(test_quad()), Some(test_material()));
        push_test_entity(&mut engine, Some(test_quad()), Some(test_material()));
        push_test_entity(
            &mut engine,
            Some(MeshDesc::Custom {
                positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                indices: vec![0, 1, 2],
            }),
            Some(test_material()),
        );

        let (extracted, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(extracted.custom_meshes.len(), 3, "per-entity entries kept");
        assert_eq!(stats.custom_cache_misses, 2, "quad + triangle converted");
        assert_eq!(stats.custom_cache_hits, 1, "repeated quad reused");
        let (first, second) = (&extracted.custom_meshes[0], &extracted.custom_meshes[1]);
        assert_eq!(first.indices, second.indices);
        assert_eq!(first.vertices.len(), second.vertices.len());
    }
}
