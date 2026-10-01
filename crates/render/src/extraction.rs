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

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    JointPose, Skeleton, SkinnedMesh, SkinningMode, SkinningResources, canonical_staged_weights,
    skinning_matrices,
};
use ornis_core::{Engine, Entity, OpenPBRMaterial, SmartStore};
use serde::{Deserialize, Serialize};

use crate::camera::Frustum;
use crate::mesh::{SkinnedVertex, Vertex};
use crate::mesh_upload::{SoupCache, UploadCache};
use crate::renderer::{InstanceData, LightUploadStats, count_light_drops};
use crate::skinning::{PaletteHandle, SkinBindError, SkinnedDraw};
use ornis_assets::scene::{LightDesc, MaterialDesc, MeshDesc, Scene, ShadowCast, TransformDesc};
use ornis_core::units::PositiveF32;

/// Squared length below which a direction is treated as degenerate.
const DEGENERATE_LEN2: f32 = 1e-12;
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

/// One valid `MeshDesc::Custom` entity: its CPU-side geometry plus the
/// instance pointing at the merged [`FrameUpload::materials`] table.
///
/// The renderer uploads `vertices`/`indices` per entity
/// (`renderer::upload_custom_mesh`) and draws the entry with `instance`.
#[derive(Clone, Debug)]
pub struct CustomMeshEntry {
    /// GPU-ready vertices: pre-skinned world-space rows for
    /// [`MeshPose::Skinned`] entries (classic soup path and CPU fallback),
    /// bind-pose rows for [`MeshPose::Bind`] entries (GPU blend — the
    /// skinned vertex stage reads these plus the influences in
    /// [`CustomMeshEntry::pose`]).
    pub vertices: Vec<Vertex>,
    /// Triangle index list (`u32`, triples, CCW from outside).
    pub indices: Vec<u32>,
    /// Model/normal matrices and the index into `FrameUpload::materials`.
    pub instance: InstanceData,
    /// How this entry is blended (phase D, `docs/animation-design.md` §2.3):
    /// [`SkinningMode::Cpu`] on the classic soup path below (bind-pose
    /// geometry transformed by `instance.model_matrix`, never pre-skinned)
    /// and on the CPU pre-skin fallback; [`SkinningMode::Gpu`] exactly when
    /// the entity carries the [`SkinnedMesh`] lane *and* its joint palette
    /// staged (see [`CustomMeshEntry::joint_palette`]).
    ///
    /// Freshness is the skin system's contract (`skel_skin_cpu` runs
    /// PostFrame before the frame upload); inconsistent lane arrays skip
    /// the entity with [`ExtractionStats::skipped_bad_skin`], never a stub
    /// (see [`extract_render_data_with_stats`]).
    pub skinning: SkinningMode,
    /// Staged GPU joint palette (phase D): final joint matrices
    /// (`model * inverse_bind`) as column-major arrays, one per joint —
    /// the upload bytes behind [`crate::skinning::joint_palette_bytes`].
    ///
    /// `Some` exactly when [`CustomMeshEntry::skinning`] is
    /// [`SkinningMode::Gpu`]; `None` on the classic path and on the CPU
    /// fallback (over-limit skeleton, stale pose, out-of-range joint
    /// indices — anything that would read out of bounds in the shader).
    pub joint_palette: Option<Vec<[[f32; 4]; 4]>>,
    /// Which buffer [`CustomMeshEntry::vertices`] holds: bind-pose rows
    /// (GPU blend) or pre-skinned world-space rows (CPU blend). Always in
    /// lockstep with [`CustomMeshEntry::skinning`] (`Gpu` ⟺ `Bind`,
    /// `Cpu` ⟺ `Skinned`); the draw decision reads both through
    /// [`CustomMeshEntry::draw`].
    pub pose: MeshPose,
}

impl CustomMeshEntry {
    /// Color-draw decision for this entry: [`SkinnedDraw::Gpu`] exactly
    /// when the entry blends on the GPU *and* a palette slot was uploaded
    /// for it, [`SkinnedDraw::Cpu`] otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`SkinBindError::MissingPalette`] when the entry claims the
    /// GPU path but `handle` is `None` (or no palette staged) — the caller
    /// falls back to [`SkinnedDraw::Cpu`], never a panic.
    pub fn draw(&self, handle: Option<PaletteHandle>) -> Result<SkinnedDraw, SkinBindError> {
        match self.skinning {
            SkinningMode::Cpu => Ok(SkinnedDraw::Cpu),
            SkinningMode::Gpu => match (self.joint_palette.is_some(), handle) {
                (true, Some(handle)) => Ok(SkinnedDraw::Gpu(handle)),
                _ => Err(SkinBindError::MissingPalette),
            },
        }
    }

    /// Depth-pre-pass decision for this entry: same rule as
    /// [`CustomMeshEntry::draw`], but the failure carries
    /// [`SkinBindError::ShadowWithoutPalette`] — skinned depth needs the
    /// palette, otherwise the shadow would come from bind-pose geometry.
    ///
    /// # Errors
    ///
    /// Returns [`SkinBindError::ShadowWithoutPalette`] when the entry
    /// claims the GPU path but `handle` is `None` (or no palette staged) —
    /// the caller falls back to [`SkinnedDraw::Cpu`], never a panic.
    pub fn shadow_draw(&self, handle: Option<PaletteHandle>) -> Result<SkinnedDraw, SkinBindError> {
        match self.skinning {
            SkinningMode::Cpu => Ok(SkinnedDraw::Cpu),
            SkinningMode::Gpu => match (self.joint_palette.is_some(), handle) {
                (true, Some(handle)) => Ok(SkinnedDraw::Gpu(handle)),
                _ => Err(SkinBindError::ShadowWithoutPalette),
            },
        }
    }

    /// Interleaved GPU rows for the skinned vertex stage: bind-pose
    /// vertices plus joint influences.
    ///
    /// `Some` exactly for [`MeshPose::Bind`] entries with agreeing lane
    /// lengths; `None` for CPU-blended entries (draw those with the
    /// classic upload) and for length defects (the caller keeps the entry
    /// on the CPU path, never a partial buffer).
    pub fn skinned_gpu_vertices(&self) -> Option<Vec<SkinnedVertex>> {
        let MeshPose::Bind(influences) = &self.pose else {
            return None;
        };
        if influences.joints.len() != self.vertices.len()
            || influences.weights.len() != self.vertices.len()
        {
            return None;
        }
        Some(
            self.vertices
                .iter()
                .zip(influences.joints.iter())
                .zip(influences.weights.iter())
                .map(|((vertex, &joints), &weights)| SkinnedVertex {
                    position: vertex.position,
                    normal: vertex.normal,
                    uv: vertex.uv,
                    tangent: vertex.tangent,
                    joints,
                    weights,
                })
                .collect(),
        )
    }
}

/// Which buffer a [`CustomMeshEntry`] holds: bind-pose rows for the vertex
/// stage to blend, or pre-skinned world-space rows for the classic draw.
///
/// `enum`, never `bool`: the GPU variant carries the per-vertex influences
/// the stage blends with, so a bind-pose buffer without influences is
/// unrepresentable.
#[derive(Clone, Debug)]
pub enum MeshPose {
    /// Bind-pose vertices: the skinned pipeline blends these with
    /// `influences` over the bound palette. Only for
    /// [`SkinningMode::Gpu`] entries.
    Bind(SkinInfluences),
    /// Pre-skinned world-space vertices: the classic pipeline draws these
    /// as-is. Classic soup path and CPU fallback.
    Skinned,
}

impl MeshPose {
    /// Whether this pose carries bind-pose rows for the GPU blend.
    pub const fn is_bind(&self) -> bool {
        matches!(self, Self::Bind(_))
    }
}

/// Per-vertex joint influences for the GPU blend of one
/// [`MeshPose::Bind`] entry.
///
/// Joints are widened `u32` lanes (`SkinnedMesh` stores `u16` — exact);
/// weights are canonicalized at staging (see
/// [`canonical_staged_weights`]) because the vertex stage multiplies raw
/// weights without normalizing. Lengths always agree with the entry's
/// vertex count.
#[derive(Clone, Debug)]
pub struct SkinInfluences {
    /// Influencing joints per vertex (top-4, `< joint count`).
    pub joints: Vec<[u32; 4]>,
    /// Canonicalized influence weights per vertex.
    pub weights: Vec<[f32; 4]>,
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

/// Ambient plus directional lights of the frame as a world resource (X3,
/// Extract-free).
///
/// Written by the scene loader between frames (world `replace_scene`
/// or the platform's equivalent) and only read inside the schedule, so no
/// `Mutex` is needed (same contract as `GpuSurfaceState`). `RenderSubmit`
/// uploads it via [`Self::set_lights_args`] instead of a hardcoded rig.
///
/// The IBL-minimum multipliers ([`Self::ambient_intensity`],
/// [`Self::exposure`]) default to `1.0` (exact no-op) and are accepted by
/// old serialized payloads through `serde` defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderLights {
    /// Ambient RGB contribution.
    pub ambient: [f32; 3],
    /// Scene lights of any kind; the renderer uploads the first eight
    /// (see `renderer::MAX_LIGHTS`) and reports the rest via
    /// [`Self::light_upload_stats`].
    pub lights: Vec<LightDesc>,
    /// IBL-minimum ambient multiplier, baked into the ambient upload by
    /// `Renderer3D::set_lights_full`. Absent in older payloads — defaults
    /// to `1.0` (no-op).
    #[serde(default = "default_ibl_factor")]
    pub ambient_intensity: f32,
    /// IBL-minimum exposure multiplier applied to every light color, baked
    /// into the light upload by `Renderer3D::set_lights_full`. Absent in
    /// older payloads — defaults to `1.0` (no-op).
    #[serde(default = "default_ibl_factor")]
    pub exposure: f32,
}

/// Default IBL-minimum multiplier (`1.0`): the exact no-op for the ambient
/// and exposure uploads, and the `serde` default for older payloads.
fn default_ibl_factor() -> f32 {
    1.0
}

/// The lighting rig `RenderSubmit` hardcoded before X3 — the resource
/// default, so a runtime that never loads a scene renders exactly as it
/// did (gate: zero pixel differences).
const LEGACY_AMBIENT: [f32; 3] = [0.10, 0.10, 0.15];
const LEGACY_KEY_LIGHT: LightDesc = LightDesc::Directional {
    direction: [1.0, 1.0, 1.0],
    intensity: 0.6,
    color: [1.0, 1.0, 1.0],
    shadow: ShadowCast::Disabled,
};
const LEGACY_FILL_LIGHT: LightDesc = LightDesc::Directional {
    direction: [-0.5, 0.5, -0.5],
    intensity: 0.3,
    color: [0.8, 0.8, 1.0],
    shadow: ShadowCast::Disabled,
};

impl Default for RenderLights {
    fn default() -> Self {
        Self {
            ambient: LEGACY_AMBIENT,
            lights: vec![LEGACY_KEY_LIGHT, LEGACY_FILL_LIGHT],
            ambient_intensity: default_ibl_factor(),
            exposure: default_ibl_factor(),
        }
    }
}

impl RenderLights {
    /// The lighting of a serialized scene as the resource (X3).
    ///
    /// Logs a one-line drop report to stderr when the scene exceeds the
    /// renderer limits (lights or shadow slots) — once per scene load,
    /// never per frame. Per-frame callers stay silent and read
    /// [`Self::light_upload_stats`] instead.
    pub fn from_scene(scene: &Scene) -> Self {
        let stats = count_light_drops(&scene.lights);
        if stats.dropped_lights > 0 || stats.dropped_shadows > 0 {
            eprintln!(
                "RenderLights: scene '{}' drops {} light(s) and {} shadow(s) \
                 (renderer limits: 8 lights, 4 shadow layers, 2 point cubes)",
                scene.name, stats.dropped_lights, stats.dropped_shadows
            );
        }
        Self {
            ambient: scene.ambient,
            lights: scene.lights.clone(),
            ambient_intensity: default_ibl_factor(),
            exposure: default_ibl_factor(),
        }
    }

    /// Converts the lights into `Renderer3D::set_lights` arguments —
    /// the scene's [`LightDesc`] list as-is (up to eight entries; the
    /// renderer uploads the rest through the drop report, same as before).
    /// Deliberately silent: the per-frame upload path must not log — drops
    /// are reported once per scene load ([`Self::from_scene`]) and via
    /// [`Self::light_upload_stats`].
    pub fn set_lights_args(&self) -> Vec<LightDesc> {
        self.lights.clone()
    }

    /// Pure preview of what the renderer would drop for this resource:
    /// excess lights beyond the upload window plus shadow requests without
    /// a free layer/cube slot. No GPU access — safe to call per frame.
    pub fn light_upload_stats(&self) -> LightUploadStats {
        count_light_drops(&self.lights)
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
/// Entities missing any of the three render components are skipped
/// ([`ExtractionStats::skipped_incomplete`]). Dense lane order is used
/// as the deterministic extraction order; identical [`MaterialDesc`]
/// values share one [`FrameUpload::materials`] entry (see
/// [`deduped_material_index`]), so `materials.len()` is the number of
/// *distinct* materials, not entities.
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
    let Some(transforms) = store.read_lane::<TransformDesc>() else {
        return (extracted, stats);
    };
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return (extracted, stats);
    };
    let Some(materials) = store.read_lane::<MaterialDesc>() else {
        return (extracted, stats);
    };
    let skinned = store.read_lane::<SkinnedMesh>();

    // Parallel to `extracted.materials`: the source descs, for exact
    // (`PartialEq`) dedup. Linear scan is fine — frames hold tens of
    // distinct materials, not thousands.
    let mut seen: Vec<MaterialDesc> = Vec::new();
    // Per-frame Custom soup conversion cache: identical soups convert
    // once (see `mesh_upload::SoupCache`). Staging is reserved once from
    // the lane length (capped): the map grows at most once per frame and
    // `custom_meshes` amortizes its pushes the same way.
    /// Cap on per-frame Custom soup cache entries (amortized staging).
    const SOUP_CACHE_CAP: usize = 4096;
    /// Reserved custom-mesh upload slots per frame.
    const CUSTOM_MESH_RESERVE: usize = 256;
    let lane_len = transforms.entities.len();
    let mut soup_cache = SoupCache::with_capacity(lane_len.min(SOUP_CACHE_CAP));
    extracted
        .custom_meshes
        .reserve(lane_len.min(CUSTOM_MESH_RESERVE));
    for (&entity, transform) in transforms.entities.iter().zip(&transforms.data) {
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
            let material_index =
                deduped_material_index(&mut extracted, &mut seen, material, &mut stats);
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
            let model = Mat4::from_scale_rotation_translation(
                Vec3::from_array(transform.scale),
                normalized_rotation(transform.rotation),
                Vec3::from_array(transform.translation),
            );
            let material_index =
                deduped_material_index(&mut extracted, &mut seen, material, &mut stats);
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
                Vec3::from_array(transform.scale) * radius.get()
            }
            MeshDesc::Box { size } => {
                Vec3::from_array(transform.scale) * Vec3::from_array(size.map(PositiveF32::get))
            }
            MeshDesc::Plane { size } => Vec3::new(
                transform.scale[0] * size[0].get(),
                transform.scale[1],
                transform.scale[2] * size[1].get(),
            ),
            MeshDesc::Cylinder { radius, height, .. } => Vec3::new(
                transform.scale[0] * radius.get(),
                transform.scale[1] * height.get(),
                transform.scale[2] * radius.get(),
            ),
            // `as_custom` above handles every `Custom`; this arm is only
            // the match-exhaustiveness fallback (never a stub).
            MeshDesc::Custom { .. } => {
                stats.skipped_unknown_mesh += 1;
                continue;
            }
        };
        let model = Mat4::from_scale_rotation_translation(
            size_scale,
            normalized_rotation(transform.rotation),
            Vec3::from_array(transform.translation),
        );
        let material_index =
            deduped_material_index(&mut extracted, &mut seen, material, &mut stats);
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
/// The same canon as [`extract_render_data`]: entities missing any of
/// the three render components are skipped (even for the maximum), and
/// the result never falls below the `FrameUpload::default` floor
/// (32, 24). Iterator form: the lane walk lives in closures (the
/// sanctioned lenient form, `rustqual.toml`).
pub fn max_mesh_params(store: &SmartStore) -> (u32, u32) {
    let Some(transforms) = store.read_lane::<TransformDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    let Some(meshes) = store.read_lane::<MeshDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    let Some(materials) = store.read_lane::<MaterialDesc>() else {
        return DEFAULT_MESH_PARAMS;
    };
    transforms
        .entities
        .iter()
        // Complete entities only: all three render components present.
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

/// Per-frame cull report for [`cull_frame_upload`]: kept versus dropped
/// entries across both instance lanes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CullStats {
    /// Instances and custom entries surviving the frustum test.
    pub kept: usize,
    /// Instances and custom entries fully outside the frustum.
    pub culled: usize,
}

/// Bounding sphere of one drawn instance: the model translation as the
/// center and the longest model-matrix axis as the radius.
///
/// The shared procedural meshes are unit-bounded with the size baked
/// into the model scale (see [`extract_render_data`]), so
/// `max(scale) * 1.0` is the sphere — a conservative overestimate for
/// non-spherical primitives (never falsely culls, may keep a few extra).
pub fn instance_sphere(instance: &InstanceData) -> (Vec3, f32) {
    let model = &instance.model_matrix;
    let radius = model
        .x_axis
        .length()
        .max(model.y_axis.length())
        .max(model.z_axis.length());
    (model.w_axis.truncate(), radius)
}

/// Opt-in frustum culling of an extracted frame: drops the
/// [`FrameUpload`] entries fully outside the six planes of `view_proj`
/// in place and reports the counts.
///
/// Off by default — [`extract_render_data`] never calls this; the
/// caller applies it after extraction when a frame `view_proj` (see
/// [`crate::camera::camera_view_projection`]) is available. Degenerate
/// planes and non-finite bounds fail open (kept, never culled). Cost is
/// `O(n)` sphere tests over both lanes; measure with
/// [`cull_frame_upload_timed`] when the frame budget needs attributing.
pub fn cull_frame_upload(upload: &mut FrameUpload, view_proj: &Mat4) -> CullStats {
    let frustum = Frustum::from_view_proj(view_proj);
    let before = upload.instances.len() + upload.custom_meshes.len();
    upload.instances.retain(|instance| {
        let (center, radius) = instance_sphere(instance);
        frustum.sphere_visible(center, radius)
    });
    upload.custom_meshes.retain(|entry| {
        let (center, radius) = instance_sphere(&entry.instance);
        frustum.sphere_visible(center, radius)
    });
    let kept = upload.instances.len() + upload.custom_meshes.len();
    CullStats {
        kept,
        culled: before - kept,
    }
}

/// View-space depth of one instance: camera-forward distance of its
/// center (`-z` of the view-transformed translation, so larger means
/// farther from the eye).
pub fn instance_view_depth(instance: &InstanceData, view: &Mat4) -> f32 {
    let world = instance.model_matrix.w_axis;
    -(*view * world).z
}

/// Back-to-front depth sort of an extracted frame, in place.
///
/// Reorders both instance lanes by [`instance_view_depth`] descending
/// (farthest first, stable). Order-only: blending and depth state are
/// untouched. Non-finite depths compare equal (stable, never panics).
/// Cost is `O(n log n)` comparisons over both lanes; measure with
/// [`sort_by_depth_timed`] when the frame budget needs attributing.
///
/// Order contract: the current forward submit is a single instanced draw
/// ([`Renderer3D::render_forward`](crate::renderer::Renderer3D::render_forward))
/// and ignores instance order — this sort is a no-op today, kept for
/// future multi-draw submits where back-to-front order is honored.
pub fn sort_by_depth(upload: &mut FrameUpload, view: &Mat4) {
    upload.instances.sort_by(|a, b| {
        instance_view_depth(b, view)
            .partial_cmp(&instance_view_depth(a, view))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    upload.custom_meshes.sort_by(|a, b| {
        instance_view_depth(&b.instance, view)
            .partial_cmp(&instance_view_depth(&a.instance, view))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// [`cull_frame_upload`] plus its wall time: the [`CullStats`] report and
/// the time spent testing both lanes against the six frustum planes
/// (`O(n)` sphere tests). Use the duration to attribute cull cost inside
/// the frame budget; the culled payload is identical to the untimed call.
pub fn cull_frame_upload_timed(
    upload: &mut FrameUpload,
    view_proj: &Mat4,
) -> (CullStats, std::time::Duration) {
    let started = std::time::Instant::now();
    let stats = cull_frame_upload(upload, view_proj);
    (stats, started.elapsed())
}

/// [`sort_by_depth`] plus its wall time (`O(n log n)` comparisons).
/// Use the duration to attribute sort cost inside the frame budget; the
/// reordered payload is identical to the untimed call.
pub fn sort_by_depth_timed(upload: &mut FrameUpload, view: &Mat4) -> std::time::Duration {
    let started = std::time::Instant::now();
    sort_by_depth(upload, view);
    started.elapsed()
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

/// Returns the [`FrameUpload::materials`] index for `material`,
/// pushing its GPU conversion only on first sight (exact `PartialEq`
/// dedup — identical [`MaterialDesc`] values share one entry). Every
/// reuse of an existing entry bumps
/// [`ExtractionStats::materials_deduped`].
fn deduped_material_index(
    extracted: &mut FrameUpload,
    seen: &mut Vec<MaterialDesc>,
    material: &MaterialDesc,
    stats: &mut ExtractionStats,
) -> crate::renderer::MaterialIdx {
    if let Some(index) = seen.iter().position(|known| known == material) {
        stats.materials_deduped += 1;
        return crate::renderer::MaterialIdx::from(index as u32);
    }
    seen.push(material.clone());
    extracted.materials.push(material_to_gpu(material));
    crate::renderer::MaterialIdx::from_raw(extracted.materials.len() as u32 - 1)
}

fn material_to_gpu(material: &MaterialDesc) -> OpenPBRMaterial {
    match material {
        MaterialDesc::Dielectric {
            base_color,
            roughness,
            emission,
        } => {
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb(*base_color);
            output.specular.roughness(roughness.get());
            apply_emission(&mut output, *emission);
            output
        }
        MaterialDesc::Metal {
            base_color,
            roughness,
            emission,
        } => {
            let mut output = OpenPBRMaterial::metal();
            output.base.color_rgb(*base_color);
            output.specular.roughness(roughness.get());
            apply_emission(&mut output, *emission);
            output
        }
        MaterialDesc::Coat {
            base_color,
            coat_weight,
            coat_roughness,
            emission,
        } => {
            let mut output = OpenPBRMaterial::coat();
            output.base.color_rgb(*base_color);
            output.coat.weight(coat_weight.get());
            output.coat.roughness(coat_roughness.get());
            apply_emission(&mut output, *emission);
            output
        }
        MaterialDesc::Matte {
            base_color,
            roughness,
        } => {
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb(*base_color);
            output.base.diffuse_roughness(roughness.get());
            // Matte is diffuse-only: no specular lobe.
            output.specular.weight(0.0);
            output
        }
        MaterialDesc::Glass {
            base_color,
            roughness,
            ior,
        } => {
            let mut output = OpenPBRMaterial::glass();
            output.transmission.color_rgb(*base_color);
            output.specular.roughness(roughness.get());
            output.specular.ior(ior.get());
            output
        }
    }
}

/// Maps an emissive RGB triple onto the OpenPBR emission group.
///
/// The shader evaluates `emission_color * emission_luminance / PI`, so the
/// peak channel becomes the luminance (nits) and the color carries the
/// normalized chromaticity — `luminance * color` reproduces the input
/// exactly. Black input leaves the preset default (luminance 0 = off).
fn apply_emission(output: &mut OpenPBRMaterial, emission: [f32; 3]) {
    let peak = emission[0].max(emission[1]).max(emission[2]).max(0.0);
    if peak > 0.0 {
        output.emission.luminance(peak);
        output
            .emission
            .color_rgb([emission[0] / peak, emission[1] / peak, emission[2] / peak]);
    }
}

fn normalized_rotation(rotation: [f32; 4]) -> Quat {
    let orientation = Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]);
    let length_squared = orientation.length_squared();
    if length_squared.is_finite() && length_squared > DEGENERATE_LEN2 {
        orientation.normalize()
    } else {
        Quat::IDENTITY
    }
}

/// Validated vertex count of one [`SkinnedMesh`] lane set: every bind and
/// output array agrees on the length, and every index lands inside it.
///
/// [`None`] (bad skin) on empty binds, array length defects, or an empty /
/// malformed index list — the caller counts
/// [`ExtractionStats::skipped_bad_skin`]. Joint-index range against the
/// skeleton is the skin system's verdict (it owns that counter); the lane
/// buffers validated here are that system's output contract.
fn skin_vertex_count(mesh: &SkinnedMesh) -> Option<usize> {
    let count = mesh.joints.len();
    if count == 0
        || mesh.weights.len() != count
        || mesh.positions.len() != count
        || mesh.normals.len() != count
        || mesh.skinned_positions.len() != count
        || mesh.skinned_normals.len() != count
        || mesh.uvs.len() != count
    {
        return None;
    }
    if mesh.indices.is_empty()
        || !mesh.indices.len().is_multiple_of(TRIANGLE_VERTS)
        || mesh.indices.iter().any(|index| (*index as usize) >= count)
    {
        return None;
    }
    Some(count)
}

/// Builds the pre-skinned payload of one [`SkinnedMesh`] entity: world-space
/// output buffers as [`Vertex`] rows plus the passthrough bind indices.
///
/// [`None`] (bad skin) when [`skin_vertex_count`] rejects the lanes — the
/// caller counts [`ExtractionStats::skipped_bad_skin`].
fn skinned_entry(mesh: &SkinnedMesh) -> Option<(Vec<Vertex>, Vec<u32>)> {
    skin_vertex_count(mesh)?;
    let vertices = mesh
        .skinned_positions
        .iter()
        .zip(mesh.skinned_normals.iter())
        .zip(mesh.uvs.iter())
        .map(|((&position, &normal), &uv)| Vertex {
            position,
            normal,
            uv,
            tangent: skinned_tangent(normal),
        })
        .collect();
    Some((vertices, mesh.indices.to_vec()))
}

/// Builds the bind-pose payload of one [`SkinnedMesh`] entity for the GPU
/// blend: bind-pose [`Vertex`] rows, the passthrough bind indices, and the
/// staged influences (widened joints, canonicalized weights).
///
/// [`None`] (bad skin) when [`skin_vertex_count`] rejects the lanes — the
/// caller counts [`ExtractionStats::skipped_bad_skin`]. Joint-index range
/// against the skeleton is gated upstream by [`gpu_joint_palette`] (only
/// staged palettes reach the GPU path); the widened lanes here carry the
/// values verbatim.
fn bind_pose_entry(mesh: &SkinnedMesh) -> Option<(Vec<Vertex>, Vec<u32>, SkinInfluences)> {
    skin_vertex_count(mesh)?;
    let vertices = mesh
        .positions
        .iter()
        .zip(mesh.normals.iter())
        .zip(mesh.uvs.iter())
        .map(|((&position, &normal), &uv)| Vertex {
            position,
            normal,
            uv,
            tangent: skinned_tangent(normal),
        })
        .collect();
    let joints = mesh
        .joints
        .iter()
        .map(|lane| {
            [
                lane[0] as u32,
                lane[1] as u32,
                lane[2] as u32,
                lane[3] as u32,
            ]
        })
        .collect();
    let weights = mesh
        .weights
        .iter()
        .map(|&lane| canonical_staged_weights(lane))
        .collect();
    Some((
        vertices,
        mesh.indices.to_vec(),
        SkinInfluences { joints, weights },
    ))
}

/// Stages the GPU joint palette of one [`SkinnedMesh`] entity: final joint
/// matrices (`model * inverse_bind`) as column-major arrays.
///
/// [`None`] (CPU fallback — same pre-skinned vertices, no palette) when the
/// skeleton lanes are absent, the topology is invalid, the pose is stale,
/// or a joint index reaches past the palette (an out-of-bounds read in the
/// shader must never stage). Callers keep the entity on the CPU path; only
/// lane-buffer defects skip it (see [`skinned_entry`]).
fn gpu_joint_palette(store: &SmartStore, mesh: &SkinnedMesh) -> Option<Vec<[[f32; 4]; 4]>> {
    let skeletons = store.read_lane::<Skeleton>()?;
    let poses = store.read_lane::<JointPose>()?;
    let skeleton = skeletons.get(mesh.skeleton)?;
    let count = skeleton.validate().ok()?;
    let pose = poses.get(mesh.skeleton)?;
    if pose.matrices.len() != count {
        return None;
    }
    let staged = SkinningResources::build(
        &skinning_matrices(&pose.matrices, &skeleton.inverse_bind),
        SkinningMode::Gpu,
    )
    .ok()?;
    if mesh
        .joints
        .iter()
        .flatten()
        .any(|index| (*index as usize) >= count)
    {
        return None;
    }
    Some(
        staged
            .palette_matrices()
            .iter()
            .map(Mat4::to_cols_array_2d)
            .collect(),
    )
}

/// Any unit vector orthogonal to a skinned normal (tangent fallback).
///
/// Same contract as `mesh_upload::fallback_tangent` (duplicated here: that
/// module is outside this track's file bounds, so the formula — not the
/// function — is shared): degenerate normals fall back to `+X` so the
/// vertex stays finite.
fn skinned_tangent(normal: [f32; 3]) -> [f32; 3] {
    let direction = Vec3::from_array(normal);
    if direction.length_squared() <= f32::EPSILON {
        return [1.0, 0.0, 0.0];
    }
    direction.normalize().any_orthonormal_vector().to_array()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_core::units::{Clamped01, Ior};

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
            store.insert(
                handle,
                TransformDesc {
                    translation: Vec3::ZERO.to_array(),
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(handle, mesh);
            store.insert(
                handle,
                MaterialDesc::Dielectric {
                    base_color: [0.8, 0.2, 0.2],
                    roughness: Clamped01::new(0.4),
                    emission: [0.0, 0.0, 0.0],
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
    fn identical_materials_deduplicate_to_one_entry() {
        // Three entities with the same `MaterialDesc` share one
        // `FrameUpload::materials` entry; a distinct material adds a
        // second, and every instance index points at its own entry.
        let shared = MaterialDesc::Metal {
            base_color: [0.9, 0.7, 0.1],
            roughness: Clamped01::new(0.2),
            emission: [0.0, 0.0, 0.0],
        };
        let other = MaterialDesc::Matte {
            base_color: [0.2, 0.2, 0.2],
            roughness: Clamped01::new(0.8),
        };
        let mut engine = Engine::new();
        for (i, material) in [shared.clone(), shared.clone(), shared.clone(), other]
            .into_iter()
            .enumerate()
        {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc {
                    translation: [i as f32, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(
                handle,
                MeshDesc::Sphere {
                    radius: PositiveF32::expect_valid(1.0),
                    segments: 16,
                    rings: 12,
                },
            );
            store.insert(handle, material);
        }

        let extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 4);
        assert_eq!(extracted.materials.len(), 2, "3 identical + 1 distinct");
        assert_eq!(
            extracted
                .instances
                .iter()
                .map(|instance| instance.material_index)
                .collect::<Vec<_>>(),
            vec![
                crate::renderer::MaterialIdx::from_raw(0),
                crate::renderer::MaterialIdx::from_raw(0),
                crate::renderer::MaterialIdx::from_raw(0),
                crate::renderer::MaterialIdx::from_raw(1)
            ],
            "shared entries reuse the first index"
        );
    }

    #[test]
    fn material_to_gpu_maps_emission_and_matte() {
        // Emission `[2, 1, 0.5]`: peak 2 nits, normalized chromaticity.
        let gpu = material_to_gpu(&MaterialDesc::Dielectric {
            base_color: [0.5, 0.5, 0.5],
            roughness: Clamped01::new(0.9),
            emission: [2.0, 1.0, 0.5],
        });
        assert_eq!(gpu.emission.params[0], 2.0);
        assert_eq!(gpu.emission.color[0], 1.0);
        assert_eq!(gpu.emission.color[1], 0.5);
        assert_eq!(gpu.emission.color[2], 0.25);
        // Black emission leaves the preset default (luminance 0 = off).
        let off = material_to_gpu(&MaterialDesc::Metal {
            base_color: [0.9, 0.7, 0.1],
            roughness: Clamped01::new(0.2),
            emission: [0.0, 0.0, 0.0],
        });
        assert_eq!(off.emission.params[0], 0.0);
        // Matte: diffuse albedo + roughness, no specular lobe.
        let matte = material_to_gpu(&MaterialDesc::Matte {
            base_color: [0.2, 0.4, 0.6],
            roughness: Clamped01::new(0.7),
        });
        assert_eq!(matte.base.color[0], 0.2);
        assert_eq!(matte.base.color[1], 0.4);
        assert_eq!(matte.base.color[2], 0.6);
        assert_eq!(matte.base.params[2], 0.0, "metalness 0");
        assert!((matte.base.params[1] - 0.7).abs() < f32::EPSILON);
        assert_eq!(matte.specular.params[0], 0.0, "no specular lobe");
    }

    #[test]
    fn material_to_gpu_maps_glass() {
        // Glass tint → transmission color, roughness/ior → specular lobe;
        // the preset keeps full transmission over thin walls.
        let gpu = material_to_gpu(&MaterialDesc::Glass {
            base_color: [0.9, 0.95, 1.0],
            roughness: Clamped01::new(0.05),
            ior: Ior::new(1.33),
        });
        assert_eq!(gpu.transmission.color[0], 0.9);
        assert_eq!(gpu.transmission.color[1], 0.95);
        assert_eq!(gpu.transmission.color[2], 1.0);
        assert_eq!(gpu.transmission.params[0], 1.0, "full transmission");
        assert!((gpu.specular.params[1] - 0.05).abs() < f32::EPSILON);
        assert!((gpu.specular.params[2] - 1.33).abs() < f32::EPSILON);
    }

    #[test]
    fn three_identical_glass_materials_share_one_entry() {
        // Spec case: 3 entities with the same `MaterialDesc` → one
        // `FrameUpload::materials` entry, every instance pointing at it.
        let shared = MaterialDesc::Glass {
            base_color: [0.9, 0.95, 1.0],
            roughness: Clamped01::new(0.05),
            ior: Ior::new(1.5),
        };
        let mut engine = Engine::new();
        for i in 0..3 {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc {
                    translation: [i as f32, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(
                handle,
                MeshDesc::Sphere {
                    radius: PositiveF32::expect_valid(1.0),
                    segments: 16,
                    rings: 12,
                },
            );
            store.insert(handle, shared.clone());
        }

        let extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 3);
        assert_eq!(extracted.materials.len(), 1, "3 identical → 1 entry");
        assert!(
            extracted.instances.iter().all(
                |instance| instance.material_index == crate::renderer::MaterialIdx::from_raw(0)
            )
        );
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
        };
        let mut engine = Engine::new();
        let mut add = |mesh: MeshDesc| {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc {
                    translation: Vec3::ZERO.to_array(),
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
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
        engine.world_mut().store_mut().expect("store").insert(
            entity,
            TransformDesc {
                translation: Vec3::ZERO.to_array(),
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: Vec3::ONE.to_array(),
            },
        );
        assert!(
            extract_render_data(engine.world().store().expect("store"))
                .instances
                .is_empty()
        );
    }

    /// Test helper: one entity with a transform plus optional mesh and
    /// material lanes (missing lanes = incomplete entity).
    fn push_test_entity(
        engine: &mut Engine,
        mesh: Option<MeshDesc>,
        material: Option<MaterialDesc>,
    ) {
        let store = engine.world_mut().store_mut().expect("store");
        let handle = store.create_entity();
        store.insert(
            handle,
            TransformDesc {
                translation: Vec3::ZERO.to_array(),
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: Vec3::ONE.to_array(),
            },
        );
        if let Some(mesh) = mesh {
            store.insert(handle, mesh);
        }
        if let Some(material) = material {
            store.insert(handle, material);
        }
    }

    fn test_material() -> MaterialDesc {
        MaterialDesc::Dielectric {
            base_color: [0.8, 0.2, 0.2],
            roughness: Clamped01::new(0.4),
            emission: [0.0, 0.0, 0.0],
        }
    }

    fn test_sphere() -> MeshDesc {
        MeshDesc::Sphere {
            radius: PositiveF32::expect_valid(1.0),
            segments: 16,
            rings: 12,
        }
    }

    fn test_quad() -> MeshDesc {
        MeshDesc::Custom {
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            indices: vec![0, 1, 2, 0, 2, 3],
        }
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
    fn default_lights_reproduce_the_legacy_hardcoded_rig() {
        // X3: `RenderSubmit` no longer inlines a lighting rig — the
        // resource default must be exactly the old hardcoded arguments
        // (gate: zero pixel differences).
        let rig = RenderLights::default();
        assert_eq!(rig.ambient, [0.10, 0.10, 0.15]);
        assert!(matches!(
            rig.set_lights_args().as_slice(),
            [
                LightDesc::Directional {
                    direction: [1.0, 1.0, 1.0],
                    intensity: key,
                    color: [1.0, 1.0, 1.0],
                    shadow: ShadowCast::Disabled,
                },
                LightDesc::Directional {
                    direction: [-0.5, 0.5, -0.5],
                    intensity: fill,
                    color: [0.8, 0.8, 1.0],
                    shadow: ShadowCast::Disabled,
                },
            ] if *key == 0.6 && *fill == 0.3
        ));
    }

    #[test]
    fn cull_frame_upload_drops_offscreen_spheres() {
        // 100 spheres: 50 near the origin (visible), 50 at x = 100 (far
        // outside the side plane). Culling is opt-in — extraction keeps
        // all 100, `cull_frame_upload` drops the offscreen half.
        let mut engine = Engine::new();
        for i in 0..100 {
            let x = if i < 50 { i as f32 * 0.05 } else { 100.0 };
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc {
                    translation: [x, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(handle, test_sphere());
            store.insert(handle, test_material());
        }
        let mut extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 100, "extraction never culls");

        let view = (
            Vec3::new(0.0, 2.5, 9.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
        );
        let (view_proj, _) = crate::camera::camera_view_projection(view, (160, 90));
        let stats = cull_frame_upload(&mut extracted, &view_proj);
        assert!(stats.culled > 0, "offscreen half culled: {stats:?}");
        assert_eq!(stats.culled, 50, "{stats:?}");
        assert_eq!(stats.kept, 50, "{stats:?}");
        assert_eq!(extracted.instances.len(), 50);
    }

    #[test]
    fn cull_frame_upload_keeps_custom_lane_in_lockstep() {
        // One visible quad plus one offscreen quad: the custom lane is
        // culled by the same sphere test as the shared batch.
        let mut upload = FrameUpload::default();
        let quad = test_quad();
        let (positions, indices) = quad.as_custom().expect("quad is custom");
        let (vertices, soup_indices) =
            crate::mesh_upload::custom_vertices(positions, indices).expect("quad valid");
        for x in [0.0, 100.0] {
            let model = Mat4::from_translation(Vec3::new(x, 0.0, 0.0));
            upload.custom_meshes.push(CustomMeshEntry {
                vertices: vertices.clone(),
                indices: soup_indices.clone(),
                instance: InstanceData {
                    model_matrix: model,
                    normal_matrix: model.inverse().transpose(),
                    material_index: crate::renderer::MaterialIdx::from_raw(0),
                },
                skinning: SkinningMode::Cpu,
                joint_palette: None,
                pose: MeshPose::Skinned,
            });
        }
        let view = (
            Vec3::new(0.0, 2.5, 9.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
        );
        let (view_proj, _) = crate::camera::camera_view_projection(view, (160, 90));
        let stats = cull_frame_upload(&mut upload, &view_proj);
        assert_eq!(stats.kept, 1, "{stats:?}");
        assert_eq!(stats.culled, 1, "{stats:?}");
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

    #[test]
    fn timed_cull_and_sort_match_untimed_payloads() {
        // The timed wrappers report the same payload as the plain calls
        // plus a wall-time measurement for frame-budget attribution.
        let mut engine = Engine::new();
        for i in 0..10 {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc {
                    translation: [i as f32 * 0.05, 0.0, -(i as f32)],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(handle, test_sphere());
            store.insert(handle, test_material());
        }
        let view = (
            Vec3::new(0.0, 2.5, 9.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
        );
        let (view_proj, _) = crate::camera::camera_view_projection(view, (160, 90));
        let mut timed = extract_render_data(engine.world().store().expect("store"));
        let mut plain = timed.clone();
        let (stats, elapsed) = cull_frame_upload_timed(&mut timed, &view_proj);
        let expected = cull_frame_upload(&mut plain, &view_proj);
        assert_eq!(stats, expected);
        assert_eq!(timed.instances.len(), plain.instances.len());
        let _ = elapsed;

        let mut timed_sort = extract_render_data(engine.world().store().expect("store"));
        let mut plain_sort = timed_sort.clone();
        let sort_elapsed = sort_by_depth_timed(&mut timed_sort, &Mat4::IDENTITY);
        sort_by_depth(&mut plain_sort, &Mat4::IDENTITY);
        assert_eq!(
            timed_sort
                .instances
                .iter()
                .map(|instance| instance.model_matrix.w_axis.z)
                .collect::<Vec<_>>(),
            plain_sort
                .instances
                .iter()
                .map(|instance| instance.model_matrix.w_axis.z)
                .collect::<Vec<_>>()
        );
        let _ = sort_elapsed;
    }

    #[test]
    fn sort_by_depth_orders_back_to_front() {
        // Identity view (camera at the origin looking down -Z): depth is
        // `-z`, so z = -1/-5/-10 sorts to -10/-5/-1. Both lanes sort.
        let mut upload = FrameUpload::default();
        for z in [-5.0, -10.0, -1.0] {
            let model = Mat4::from_translation(Vec3::new(0.0, 0.0, z));
            upload.instances.push(InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: crate::renderer::MaterialIdx::from_raw(0),
            });
        }
        sort_by_depth(&mut upload, &Mat4::IDENTITY);
        let depths: Vec<f32> = upload
            .instances
            .iter()
            .map(|instance| instance_view_depth(instance, &Mat4::IDENTITY))
            .collect();
        assert_eq!(depths, vec![10.0, 5.0, 1.0], "{depths:?}");
        let zs: Vec<f32> = upload
            .instances
            .iter()
            .map(|instance| instance.model_matrix.w_axis.z)
            .collect();
        assert_eq!(zs, vec![-10.0, -5.0, -1.0], "{zs:?}");
    }

    /// Test helper: a single-joint skeleton root plus one skinned triangle
    /// entity (identity pose → CPU vertices equal the bind positions).
    /// Returns the mesh entity (for lane corruption) — the root is the
    /// mesh's `skeleton` link.
    fn push_skinned_triangle(engine: &mut Engine) -> Entity {
        {
            let store = engine.world_mut().store_mut().expect("store");
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<SkinnedMesh>();
        }
        let root = {
            let store = engine.world_mut().store_mut().expect("store");
            let root = store.create_entity();
            store.insert(
                root,
                Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["root".to_string()]),
            );
            store.insert(root, JointPose::identity(1));
            root
        };
        let positions = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        {
            let store = engine.world_mut().store_mut().expect("store");
            let mesh = store.create_entity();
            store.insert(
                mesh,
                TransformDesc {
                    translation: Vec3::ZERO.to_array(),
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(
                mesh,
                MeshDesc::Custom {
                    positions: positions.clone(),
                    indices: vec![0, 1, 2],
                },
            );
            store.insert(mesh, test_material());
            store.insert(
                mesh,
                SkinnedMesh::new(
                    root,
                    vec![[0, 0, 0, 0]; 3],
                    vec![[1.0, 0.0, 0.0, 0.0]; 3],
                    positions,
                    vec![[0.0, 0.0, 1.0]; 3],
                    vec![[0.0, 0.0]; 3],
                    vec![0, 1, 2],
                ),
            );
            mesh
        }
    }

    #[test]
    fn skinned_entry_stages_gpu_palette_with_bind_pose_vertices() {
        // Valid skin: GPU mode with the staged palette, while `vertices`
        // carry the bind pose (identity skin → bind positions) for the
        // skinned vertex stage; the influences ride `pose`. Parity: the
        // palette-blend of the bind data matches the CPU skin output
        // within the допуск.
        use ornis_animation::{CPU_GPU_TOLERANCE, blend_vertex_reference};
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Gpu);
        assert!(entry.pose.is_bind());
        assert_eq!(entry.instance.model_matrix, Mat4::IDENTITY);
        let palette = entry.joint_palette.as_ref().expect("palette staged");
        assert_eq!(palette.len(), 1);
        assert_eq!(Mat4::from_cols_array_2d(&palette[0]), Mat4::IDENTITY);
        assert_eq!(entry.vertices[0].position, [1.0, 0.0, 0.0]);
        // Bind pose rides the entry: joints widened, weights canonical.
        let MeshPose::Bind(influences) = &entry.pose else {
            panic!("gpu entry must carry bind influences");
        };
        assert_eq!(influences.joints.len(), 3);
        assert_eq!(influences.joints[0], [0, 0, 0, 0]);
        assert_eq!(influences.weights[0], [1.0, 0.0, 0.0, 0.0]);
        // Parity: palette-blend(bind) ≈ CPU-skinned positions.
        let bind = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let matrices = [Mat4::IDENTITY];
        for (index, vertex) in entry.vertices.iter().enumerate() {
            let (position, _) = blend_vertex_reference(
                &matrices,
                [0, 0, 0, 0],
                [1.0, 0.0, 0.0, 0.0],
                bind[index],
                [0.0, 0.0, 1.0],
            );
            let drift = (Vec3::from_array(position) - Vec3::from_array(vertex.position)).length();
            assert!(drift < CPU_GPU_TOLERANCE, "vertex {index} drifts {drift}");
        }
        // The interleaved GPU rows zip bind vertices with influences.
        let gpu_rows = entry.skinned_gpu_vertices().expect("bind rows");
        assert_eq!(gpu_rows.len(), 3);
        assert_eq!(gpu_rows[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(gpu_rows[0].joints, [0, 0, 0, 0]);
        assert_eq!(gpu_rows[0].weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn missing_skeleton_falls_back_to_cpu_without_palette() {
        // No skeleton/pose lanes: the entity still extracts (valid CPU
        // buffers) but in CPU mode with no palette — never a GPU claim
        // the shader could read out of bounds with.
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);
        // Destroy every skeleton root (same honesty as `RenderWorld` scene
        // replacement): the mesh lane outlives its skeleton.
        let roots: Vec<Entity> = {
            let store = engine.world().store().expect("store");
            store
                .read_lane::<Skeleton>()
                .map(|lane| lane.entities.clone())
                .unwrap_or_default()
        };
        for root in roots {
            let store = engine.world_mut().store_mut().expect("store");
            if store.is_alive(root) {
                store.destroy_entity(root);
            }
        }

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Cpu);
        assert!(entry.joint_palette.is_none());
        assert!(matches!(entry.pose, MeshPose::Skinned));
        assert!(entry.skinned_gpu_vertices().is_none());
        assert_eq!(entry.vertices[0].position, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn over_limit_skeleton_falls_back_to_cpu_without_palette() {
        // 129 joints: the palette cannot stage (typed overflow), so the
        // entry extracts in CPU mode with no palette — never truncated.
        use ornis_animation::JointLimit;
        let over = JointLimit::GPU.index() + 1;
        let mut engine = Engine::new();
        {
            let store = engine.world_mut().store_mut().expect("store");
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<SkinnedMesh>();
            let root = store.create_entity();
            store.insert(
                root,
                Skeleton::new(
                    vec![None; over],
                    vec![Mat4::IDENTITY; over],
                    vec!["joint".to_string(); over],
                ),
            );
            store.insert(root, JointPose::identity(over));
            let mesh = store.create_entity();
            store.insert(
                mesh,
                TransformDesc {
                    translation: Vec3::ZERO.to_array(),
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(
                mesh,
                MeshDesc::Custom {
                    positions: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    indices: vec![0, 1, 2],
                },
            );
            store.insert(mesh, test_material());
            store.insert(
                mesh,
                SkinnedMesh::new(
                    root,
                    vec![[0, 0, 0, 0]; 3],
                    vec![[1.0, 0.0, 0.0, 0.0]; 3],
                    vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    vec![[0.0, 0.0, 1.0]; 3],
                    vec![[0.0, 0.0]; 3],
                    vec![0, 1, 2],
                ),
            );
        }

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Cpu);
        assert!(entry.joint_palette.is_none());
        assert!(matches!(entry.pose, MeshPose::Skinned));
        assert!(entry.skinned_gpu_vertices().is_none());
    }

    /// Test helper: a two-joint skeleton root with a rigid rotated pose
    /// plus one skinned triangle entity. The palette is a 90° Z-rotation
    /// with a translation on joint 1, so the GPU blend and the CPU skin
    /// must agree within the parity допуск (rigid joints are exact up to
    /// FMA ordering).
    fn push_rotated_two_joint_triangle(engine: &mut Engine) {
        {
            let store = engine.world_mut().store_mut().expect("store");
            store.register::<Skeleton>();
            store.register::<JointPose>();
            store.register::<SkinnedMesh>();
        }
        let root = {
            let store = engine.world_mut().store_mut().expect("store");
            let root = store.create_entity();
            store.insert(
                root,
                Skeleton::new(
                    vec![None, Some(ornis_animation::JointId::from_raw(0))],
                    vec![Mat4::IDENTITY; 2],
                    vec!["root".to_string(), "child".to_string()],
                ),
            );
            let rotated = Mat4::from_rotation_translation(
                glam::Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                Vec3::X,
            );
            store.insert(
                root,
                JointPose {
                    matrices: vec![Mat4::IDENTITY, rotated],
                },
            );
            root
        };
        {
            let store = engine.world_mut().store_mut().expect("store");
            let mesh = store.create_entity();
            store.insert(
                mesh,
                TransformDesc {
                    translation: Vec3::ZERO.to_array(),
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    scale: Vec3::ONE.to_array(),
                },
            );
            store.insert(
                mesh,
                MeshDesc::Custom {
                    positions: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    indices: vec![0, 1, 2],
                },
            );
            store.insert(mesh, test_material());
            store.insert(
                mesh,
                SkinnedMesh::new(
                    root,
                    vec![[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]],
                    vec![
                        [1.0, 0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.0, 0.0],
                    ],
                    vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    vec![[0.0, 0.0, 1.0]; 3],
                    vec![[0.0, 0.0]; 3],
                    vec![0, 1, 2],
                ),
            );
        }
    }

    #[test]
    fn gpu_bind_blend_matches_cpu_skin_within_tolerance() {
        // Cpu-draw vs Gpu-draw parity at the entry level: the staged bind
        // data blended through the GPU reference formula lands within
        // `CPU_GPU_TOLERANCE` of the CPU skin output for a rigid palette
        // (normals: linear part vs inverse-transpose coincide on rigid
        // joints; positions differ only by FMA ordering).
        use ornis_animation::{CPU_GPU_TOLERANCE, blend_vertex_reference, skin_vertices};
        let mut engine = Engine::new();
        push_rotated_two_joint_triangle(&mut engine);

        let (upload, stats) =
            extract_render_data_with_stats(engine.world().store().expect("store"));
        assert_eq!(upload.custom_meshes.len(), 1);
        assert_eq!(stats.skipped_bad_skin, 0);
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Gpu);
        let palette = entry
            .joint_palette
            .as_ref()
            .expect("palette staged")
            .iter()
            .map(Mat4::from_cols_array_2d)
            .collect::<Vec<_>>();
        assert_eq!(palette.len(), 2);
        let MeshPose::Bind(influences) = &entry.pose else {
            panic!("gpu entry must carry bind influences");
        };
        // CPU draw: the skin system's own blend over the same palette.
        let joints_u16 = [[1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]];
        let weights = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ];
        let bind_positions = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let bind_normals = [[0.0, 0.0, 1.0]; 3];
        let (cpu_positions, cpu_normals) = skin_vertices(
            &palette,
            &joints_u16,
            &weights,
            &bind_positions,
            &bind_normals,
        );
        // GPU draw: the reference blend over the staged entry data.
        for index in 0..3 {
            let (gpu_position, gpu_normal) = blend_vertex_reference(
                &palette,
                influences.joints[index].map(|joint| joint as u16),
                influences.weights[index],
                entry.vertices[index].position,
                entry.vertices[index].normal,
            );
            let position_drift =
                (Vec3::from_array(gpu_position) - Vec3::from_array(cpu_positions[index])).length();
            let normal_drift =
                (Vec3::from_array(gpu_normal) - Vec3::from_array(cpu_normals[index])).length();
            assert!(
                position_drift < CPU_GPU_TOLERANCE,
                "vertex {index} position drifts {position_drift}"
            );
            assert!(
                normal_drift < CPU_GPU_TOLERANCE,
                "vertex {index} normal drifts {normal_drift}"
            );
        }
        // Spot check: joint-1-only vertex rides the rotation+translation.
        assert!(
            (Vec3::from_array(cpu_positions[0]) - Vec3::new(1.0, 1.0, 0.0)).length()
                < CPU_GPU_TOLERANCE
        );
    }

    #[test]
    fn draw_resolves_gpu_with_handle_and_falls_back_without_palette() {
        // Gpu entries resolve through the uploaded handle; a missing
        // palette (or slot) falls back to Cpu without panicking — the
        // color path reports `MissingPalette`, the shadow path
        // `ShadowWithoutPalette` (bind-pose shadows are never drawn).
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);
        let upload = extract_render_data(engine.world().store().expect("store"));
        let entry = &upload.custom_meshes[0];
        assert_eq!(entry.skinning, SkinningMode::Gpu);
        let handle = PaletteHandle::from_raw(0);
        assert_eq!(entry.draw(Some(handle)), Ok(SkinnedDraw::Gpu(handle)));
        assert_eq!(
            entry.shadow_draw(Some(handle)),
            Ok(SkinnedDraw::Gpu(handle))
        );
        assert_eq!(entry.draw(None), Err(SkinBindError::MissingPalette));
        assert_eq!(
            entry.shadow_draw(None),
            Err(SkinBindError::ShadowWithoutPalette)
        );
        // Both failures degrade to the CPU draw, never a panic.
        assert_eq!(
            entry.draw(None).unwrap_or(SkinnedDraw::Cpu),
            SkinnedDraw::Cpu
        );
        assert_eq!(
            entry.shadow_draw(None).unwrap_or(SkinnedDraw::Cpu),
            SkinnedDraw::Cpu
        );
        // Cpu entries ignore handles entirely.
        let mut cpu_entry = entry.clone();
        cpu_entry.skinning = SkinningMode::Cpu;
        cpu_entry.joint_palette = None;
        cpu_entry.pose = MeshPose::Skinned;
        assert_eq!(cpu_entry.draw(Some(handle)), Ok(SkinnedDraw::Cpu));
        assert_eq!(cpu_entry.shadow_draw(Some(handle)), Ok(SkinnedDraw::Cpu));
        assert_eq!(cpu_entry.draw(None), Ok(SkinnedDraw::Cpu));
    }

    #[test]
    fn skinned_gpu_vertices_reject_cpu_pose_and_length_defects() {
        // Cpu entries have no GPU rows; a Bind pose whose lanes disagree
        // with the vertex count yields None instead of a partial buffer.
        let mut engine = Engine::new();
        push_skinned_triangle(&mut engine);
        let upload = extract_render_data(engine.world().store().expect("store"));
        let mut entry = upload.custom_meshes[0].clone();
        entry.pose = MeshPose::Skinned;
        assert!(entry.skinned_gpu_vertices().is_none());
        let mut broken = upload.custom_meshes[0].clone();
        let MeshPose::Bind(influences) = &mut broken.pose else {
            panic!("gpu entry must carry bind influences");
        };
        influences.joints.pop();
        assert!(broken.skinned_gpu_vertices().is_none());
    }
}
