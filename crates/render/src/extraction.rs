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
use ornis_core::{Engine, Entity, OpenPBRMaterial, SmartStore};
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use crate::camera::Frustum;
use crate::mesh::Vertex;
use crate::mesh_upload::custom_vertices_cached;
use crate::renderer::{InstanceData, LightUploadStats, count_light_drops};
use crate::scene::{LightDesc, MaterialDesc, MeshDesc, Scene, TransformDesc};

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
    /// GPU-ready vertices (`mesh_upload::custom_vertices_cached` output).
    pub vertices: Vec<Vertex>,
    /// Triangle index list (`u32`, triples, CCW from outside).
    pub indices: Vec<u32>,
    /// Model/normal matrices and the index into `FrameUpload::materials`.
    pub instance: InstanceData,
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
    shadow: false,
};
const LEGACY_FILL_LIGHT: LightDesc = LightDesc::Directional {
    direction: [-0.5, 0.5, -0.5],
    intensity: 0.3,
    color: [0.8, 0.8, 1.0],
    shadow: false,
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
        extract_render_data(self.engine.world().store().expect("render world store"))
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
    /// Entities reaching the match-exhaustiveness fallback for mesh
    /// variants the router does not know. Never fires for the known
    /// variants (`Sphere`, `Box`, `Plane`, `Cylinder`, `Custom`) —
    /// nonzero means the router missed a case.
    pub skipped_unknown_mesh: u32,
    /// Material-table reuses: times an identical [`MaterialDesc`] shared
    /// an existing [`FrameUpload::materials`] entry instead of pushing a
    /// new one.
    pub materials_deduped: u32,
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

    // Parallel to `extracted.materials`: the source descs, for exact
    // (`PartialEq`) dedup. Linear scan is fine — frames hold tens of
    // distinct materials, not thousands.
    let mut seen: Vec<MaterialDesc> = Vec::new();
    // Per-frame Custom soup conversion cache: identical soups convert
    // once (see `mesh_upload::custom_vertices_cached`).
    let mut custom_cache: HashMap<u64, (Vec<Vertex>, Vec<u32>)> = HashMap::new();
    for (&entity, transform) in transforms.entities.iter().zip(&transforms.data) {
        let Some(mesh) = meshes.get(entity) else {
            stats.skipped_incomplete += 1;
            continue;
        };
        let Some(material) = materials.get(entity) else {
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
            let Ok((vertices, soup_indices)) =
                custom_vertices_cached(positions, indices, &mut custom_cache)
            else {
                stats.skipped_bad_custom += 1;
                continue;
            };
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
                Vec3::from_array(transform.scale) * *radius
            }
            MeshDesc::Box { size } => Vec3::from_array(transform.scale) * Vec3::from_array(*size),
            MeshDesc::Plane { size } => Vec3::new(
                transform.scale[0] * size[0],
                transform.scale[1],
                transform.scale[2] * size[1],
            ),
            MeshDesc::Cylinder { radius, height, .. } => Vec3::new(
                transform.scale[0] * *radius,
                transform.scale[1] * *height,
                transform.scale[2] * *radius,
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
/// planes and non-finite bounds fail open (kept, never culled).
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

fn insert_scene_entities(
    engine: &mut Engine,
    entities: &[crate::scene::EntityDesc],
) -> Vec<Entity> {
    let store = engine.world_mut().store_mut().expect("render world store");
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
) -> u32 {
    if let Some(index) = seen.iter().position(|known| known == material) {
        stats.materials_deduped += 1;
        return index as u32;
    }
    seen.push(material.clone());
    extracted.materials.push(material_to_gpu(material));
    extracted.materials.len() as u32 - 1
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
            output.specular.roughness(*roughness);
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
            output.specular.roughness(*roughness);
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
            output.coat.weight(*coat_weight);
            output.coat.roughness(*coat_roughness);
            apply_emission(&mut output, *emission);
            output
        }
        MaterialDesc::Matte {
            base_color,
            roughness,
        } => {
            let mut output = OpenPBRMaterial::dielectric();
            output.base.color_rgb(*base_color);
            output.base.diffuse_roughness(*roughness);
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
            output.specular.roughness(*roughness);
            output.specular.ior(*ior);
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
    if length_squared.is_finite() && length_squared > 1e-12 {
        orientation.normalize()
    } else {
        Quat::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
                    roughness: 0.4,
                    emission: [0.0, 0.0, 0.0],
                },
            );
        };
        add(MeshDesc::Sphere {
            radius: 1.0,
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
        assert_eq!(extracted.instances[0].material_index, 0);
        let entry = &extracted.custom_meshes[0];
        assert_eq!(entry.vertices.len(), 4);
        assert_eq!(entry.indices, vec![0, 1, 2, 0, 2, 3]);
        assert_eq!(entry.instance.material_index, 0);
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
            roughness: 0.2,
            emission: [0.0, 0.0, 0.0],
        };
        let other = MaterialDesc::Matte {
            base_color: [0.2, 0.2, 0.2],
            roughness: 0.8,
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
                    radius: 1.0,
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
            vec![0, 0, 0, 1],
            "shared entries reuse the first index"
        );
    }

    #[test]
    fn material_to_gpu_maps_emission_and_matte() {
        // Emission `[2, 1, 0.5]`: peak 2 nits, normalized chromaticity.
        let gpu = material_to_gpu(&MaterialDesc::Dielectric {
            base_color: [0.5, 0.5, 0.5],
            roughness: 0.9,
            emission: [2.0, 1.0, 0.5],
        });
        assert_eq!(gpu.emission.params[0], 2.0);
        assert_eq!(gpu.emission.color[0], 1.0);
        assert_eq!(gpu.emission.color[1], 0.5);
        assert_eq!(gpu.emission.color[2], 0.25);
        // Black emission leaves the preset default (luminance 0 = off).
        let off = material_to_gpu(&MaterialDesc::Metal {
            base_color: [0.9, 0.7, 0.1],
            roughness: 0.2,
            emission: [0.0, 0.0, 0.0],
        });
        assert_eq!(off.emission.params[0], 0.0);
        // Matte: diffuse albedo + roughness, no specular lobe.
        let matte = material_to_gpu(&MaterialDesc::Matte {
            base_color: [0.2, 0.4, 0.6],
            roughness: 0.7,
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
            roughness: 0.05,
            ior: 1.33,
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
            roughness: 0.05,
            ior: 1.5,
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
                    radius: 1.0,
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
            extracted
                .instances
                .iter()
                .all(|instance| instance.material_index == 0)
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
            roughness: 0.4,
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
            radius: 1.0,
            segments: 48,
            rings: 32,
        });
        add(MeshDesc::Box {
            size: [2.0, 4.0, 6.0],
        });
        add(MeshDesc::Plane { size: [3.0, 5.0] });
        add(MeshDesc::Cylinder {
            radius: 2.0,
            height: 7.0,
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
            roughness: 0.4,
            emission: [0.0, 0.0, 0.0],
        }
    }

    fn test_sphere() -> MeshDesc {
        MeshDesc::Sphere {
            radius: 1.0,
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
                skipped_unknown_mesh: 0,
                materials_deduped: 0,
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
                size: [2.0, 4.0, 6.0],
            },
            MeshDesc::Plane { size: [3.0, 5.0] },
            MeshDesc::Cylinder {
                radius: 2.0,
                height: 7.0,
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
            roughness: 0.8,
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
                    shadow: false,
                },
                LightDesc::Directional {
                    direction: [-0.5, 0.5, -0.5],
                    intensity: fill,
                    color: [0.8, 0.8, 1.0],
                    shadow: false,
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
                    material_index: 0,
                },
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
    fn sort_by_depth_orders_back_to_front() {
        // Identity view (camera at the origin looking down -Z): depth is
        // `-z`, so z = -1/-5/-10 sorts to -10/-5/-1. Both lanes sort.
        let mut upload = FrameUpload::default();
        for z in [-5.0, -10.0, -1.0] {
            let model = Mat4::from_translation(Vec3::new(0.0, 0.0, z));
            upload.instances.push(InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: 0,
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
}
