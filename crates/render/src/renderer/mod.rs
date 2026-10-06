//! The deferred `Renderer3D`: imperative hybrid pipeline of
//! gbuffer (5 MRT) -> lighting -> forward -> composite passes, plus the bloom
//! chain. This is the production implementation behind
//! [`crate::render_backend::RenderBackend`]; see also [`crate::frame_exec`]
//! for the render-graph-driven equivalent.

use crate::extraction::CustomMeshEntry;
use crate::mesh::{Mesh, SkinnedVertex, Vertex};
use crate::shaders;
use crate::skinning::{
    GBUFFER_SKINNED_RESOURCES, PALETTE_BYTE_SIZE, PaletteHandle, SkinnedDraw, palette_upload_bytes,
    skinned_entry_point, wgsl_vertex_source_skinned,
};
use crate::textures::{
    CpuImage, GpuTexture, MaterialTextureSet, TextureCache, TextureRole,
    sampler_descriptor_for_role, upload_texture,
};
use glam::Mat4;
use ornis_animation::SkinningMode;
use ornis_assets::scene::LightDesc;
use ornis_core::material::{OPENPBR_MATERIAL_SIZE, OpenPBRMaterial};
use ornis_macros::WgslStruct;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

mod buffers;
mod construct;
mod deferred;
mod forward;
mod gbuffer;
mod ibl_bind;
mod lights;
mod meshes;
mod post;
mod shadows;
#[cfg(test)]
mod test_util;

pub use buffers::{InstanceData, MaterialIdx, PerObjectGpu};
// Path-stability re-export: `crate::renderer::staging_capacity_for_instances`
// before the split; no in-crate users outside its module right now.
#[allow(unused_imports)]
pub(crate) use buffers::staging_capacity_for_instances;
pub use deferred::LightingPass;
pub use forward::{
    BlendMode, ForwardPass, TexturedForwardPass, TransparencyError, TransparencyOptions,
    forward_blend_state, sort_by_depth,
};
pub use gbuffer::{GBufferTextures, GbufferTargets};
pub(crate) use lights::{GpuLight, LIGHT_KIND_DIRECTIONAL, LightingUniform};
pub use lights::{LightUploadStats, MAX_LIGHTS, ShadingDebug, count_light_drops};
// Path-stability re-exports: `crate::renderer::X` before the split; no
// in-crate users outside their module right now.
#[allow(unused_imports)]
pub(crate) use lights::{GpuLightKind, LIGHT_KIND_POINT, LIGHT_KIND_SPOT};
pub use meshes::{
    CustomGbufferDraw, StagedCustomMesh, custom_draw_items, upload_custom_mesh,
    upload_skinned_mesh, upload_vertex_rows,
};
pub use post::{BloomPass, CompositeInputs, CompositePass, CompositeResources, FogInputs};
pub(crate) use post::{BloomUniform, FogPipeline, FogUniform};
pub use shadows::{
    POINT_SHADOW_CUBES, SHADOW_CUBE_NEAR, SHADOW_CUBE_SIZE, SHADOW_LAYERS, SHADOW_ORTHO_HALF,
    SHADOW_SIZE, shadow_fit_for_bounds,
};

use shadows::CUBE_FACE_COUNT;

/// Shared read guard that recovers from a poisoned [`std::sync::RwLock`].
fn read_lock<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}

/// Exclusive write guard that recovers from a poisoned [`std::sync::RwLock`].
fn write_lock<T>(lock: &std::sync::RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

/// CPU-updated uniform buffer: the host writes it, the shader reads it.
///
/// `BitOr` on wgpu usages is not `const`, so this is `union` (which is).
const CPU_UNIFORM_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST);

/// Color target rendered into by one pass and sampled by a later pass.
///
/// `BitOr` on wgpu usages is not `const`, so this is `union` (which is).
const RENDER_TARGET_USAGE: wgpu::TextureUsages =
    wgpu::TextureUsages::RENDER_ATTACHMENT.union(wgpu::TextureUsages::TEXTURE_BINDING);

/// Frame-global camera uniform (binding shared by every pass).
///
/// The WGSL `Camera` declaration is generated from this layout
/// ([`CameraUniform::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "Camera")]
pub struct CameraUniform {
    /// View-projection matrix.
    pub view_proj: [[f32; 4]; 4],
    /// Its inverse: reconstructs world position from depth in the lighting pass.
    pub inv_view_proj: [[f32; 4]; 4],
    /// World-space eye position (`w` = 1) for specular falloff.
    pub camera_pos: [f32; 4],
}

/// Native MSAA sample count: 4x multisampled color/depth with resolve into
/// the presented/observed texture.
///
/// This is the count the native shell passes as `sample_count` (see
/// [`Renderer3D::new`] and
/// [`crate::render_backend::RenderBackendConfig::sample_count`) after gating
/// it through [`negotiate_sample_count`]. Pass policy per target:
///
/// * geometry layers (g-buffer float colors, forward HDR color) render 4x
///   multisampled and resolve into single-sample textures that downstream
///   passes sample — shaders and layouts stay single-sample there;
/// * g-buffer depth and the integer material-id layer have no resolve target
///   in `wgpu`, so they stay multisampled and are loaded as sample 0 (see
///   [`crate::shaders::resource_stays_multisampled`]);
/// * fullscreen passes (lighting output, composite, bloom, fog target) and
///   shadow maps stay single-sample, while the frame-plan pool follows this
///   count for its geometry layers (sampling the single-sample resolves
///   owned here — see [`Renderer3D::forward_resolve_view`]): every
///   pipeline's `multisample.count` matches its attachments.
pub const MSAA_SAMPLE_COUNT: u32 = 4;

/// Single-sample count: the default everywhere (headless gates, plan pool,
/// shadow maps, fullscreen passes) and the documented fallback when the
/// adapter cannot do [`MSAA_SAMPLE_COUNT`] (see [`negotiate_sample_count`]).
/// The 1x path never allocates resolve targets and records no resolves, so
/// existing pixel-parity gates stay green by construction.
pub const SINGLE_SAMPLE_COUNT: u32 = 1;

/// Scene-linear color of the deferred lighting pass.
///
/// Half-float, not the surface format: an 8-bit sRGB target clamps every
/// channel to 1 before the composite's single ACES, so a grazing highlight
/// becomes a hard white pixel and a walking normal quantizes into a
/// whole-body flicker. The composite still tonemaps once into the
/// swapchain.
pub const DEFERRED_HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Clamps a requested sample count to the supported set (`{1, 4}`):
/// [`MSAA_SAMPLE_COUNT`] passes through, anything else (including 0, 2, 8)
/// falls back to [`SINGLE_SAMPLE_COUNT`]. Pure, so unit tests pin it without
/// a GPU. [`Renderer3D::new`] applies this; capability gating lives in
/// [`negotiate_sample_count`].
pub fn normalize_sample_count(requested: u32) -> u32 {
    if requested == MSAA_SAMPLE_COUNT {
        MSAA_SAMPLE_COUNT
    } else {
        SINGLE_SAMPLE_COUNT
    }
}

/// Negotiates the MSAA sample count for `requested` against `adapter`:
/// returns [`MSAA_SAMPLE_COUNT`] only when 4x multisampling is available
/// for every texture the MSAA path creates multisampled (g-buffer colors,
/// material id, depth, forward color) plus resolve support for every format
/// it resolves into (the float colors); otherwise returns
/// [`SINGLE_SAMPLE_COUNT`].
///
/// Two halves must both pass. First the WebGPU-guaranteed baseline (what the
/// device enforces for the feature-less device requests used throughout this
/// crate): adapter-specific flags are known to over-claim — native Metal
/// reports `MULTISAMPLE_X4` for `R32Uint`, whose creation then fails, which
/// is why the material-id layer uses `R16Uint` (spec-guaranteed multisample).
/// Second the adapter's own flags, for adapters weaker than the spec
/// (software rasterizers may miss a format). Either half failing falls back
/// to single-sample rendering instead of a validation panic: the lavapipe/CI
/// contract. The function itself never panics. Non-4 requests normalize to
/// 1 without consulting any table.
pub fn negotiate_sample_count(adapter: &wgpu::Adapter, requested: u32) -> u32 {
    use wgpu::TextureFormatFeatureFlags as Flags;
    if normalize_sample_count(requested) != MSAA_SAMPLE_COUNT {
        return SINGLE_SAMPLE_COUNT;
    }
    /// Formats the MSAA path creates multisampled (geometry colors, the
    /// integer material id, depth, forward color).
    const MSAA_STORAGE: [wgpu::TextureFormat; 5] = [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rg16Float,
        wgpu::TextureFormat::R16Uint,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Depth32Float,
    ];
    /// Formats the MSAA path resolves into (the filterable float colors;
    /// depth and the integer id have no resolve target and stay multisampled).
    const RESOLVE_TARGETS: [wgpu::TextureFormat; 3] = [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rg16Float,
        wgpu::TextureFormat::Rgba16Float,
    ];
    // Spec-guaranteed baseline for feature-less devices (see above).
    let guaranteed_ok = MSAA_STORAGE.iter().all(|format| {
        format
            .guaranteed_format_features(wgpu::Features::empty())
            .flags
            .contains(Flags::MULTISAMPLE_X4)
    }) && RESOLVE_TARGETS.iter().all(|format| {
        format
            .guaranteed_format_features(wgpu::Features::empty())
            .flags
            .contains(Flags::MULTISAMPLE_RESOLVE)
    });
    // The adapter must not be weaker than the spec baseline.
    let adapter_ok = MSAA_STORAGE.iter().all(|format| {
        adapter
            .get_texture_format_features(*format)
            .flags
            .contains(Flags::MULTISAMPLE_X4)
    }) && RESOLVE_TARGETS.iter().all(|format| {
        adapter
            .get_texture_format_features(*format)
            .flags
            .contains(Flags::MULTISAMPLE_RESOLVE)
    });
    if guaranteed_ok && adapter_ok {
        MSAA_SAMPLE_COUNT
    } else {
        SINGLE_SAMPLE_COUNT
    }
}

/// Components in an RGB / xyz triple.
const VEC3_COMPONENTS: usize = 3;

/// Midpoint / half-extent scale.
const HALF: f32 = 0.5;

/// Indices per triangle.
const TRIANGLE_VERTS: usize = 3;

/// Components in an RGBA / homogeneous vector.
const VEC4_COMPONENTS: usize = 4;

/// Vertices in a fullscreen triangle-strip quad (`draw(0..4)`).
const FULLSCREEN_QUAD_VERTS: u32 = 4;

/// Initial per-object instance buffer capacity (grows on demand).
const INITIAL_MAX_OBJECTS: u32 = 256;

/// Initial material buffer capacity (grows on demand).
const INITIAL_MAX_MATERIALS: u32 = 64;

/// Default ambient when no scene lighting has been uploaded yet.
const DEFAULT_AMBIENT_RGB: [f32; 3] = [0.03, 0.03, 0.05];

/// The four core GPU buffers (camera / per-object / material / lighting).
struct CoreBuffers {
    camera: wgpu::Buffer,
    per_object: wgpu::Buffer,
    material: wgpu::Buffer,
    lighting: wgpu::Buffer,
}

/// The deferred 3D renderer: owns GPU buffers, pipelines and persistent
/// targets; drives the hybrid deferred+forward+bloom frame via its
/// `render_*` methods or the all-in-one [`Renderer3D::render_scene`].
///
/// Interior mutability note: `upload_*` may reallocate growable buffers
/// and rebuild their bind groups, so the buffers live behind an `RwLock`
/// — every other method only reads them, and the parallel command
/// recording path (`Sync`) keeps working. External callers keep the plain
/// `&self` signatures; no `&mut` threading through the schedule systems
/// or the WASM loop was needed.
pub struct Renderer3D {
    camera_buffer: wgpu::Buffer,
    per_object_buffer: std::sync::RwLock<wgpu::Buffer>,
    material_buffer: std::sync::RwLock<wgpu::Buffer>,
    lighting_buffer: wgpu::Buffer,
    _bind_group_layout: wgpu::BindGroupLayout,
    _bind_group: wgpu::BindGroup,
    _pipeline: wgpu::RenderPipeline,
    pbr_texture: wgpu::Texture,
    pbr_texture_view: wgpu::TextureView,
    sample_count: u32,
    /// Exposure multiplier applied by [`set_lights`](Self::set_lights);
    /// `1.0` (default) is the exact no-op. Set from
    /// [`crate::render_backend::RenderBackendConfig::exposure`].
    exposure: f32,
    /// Transparency mode of the forward pipeline (see
    /// [`TransparencyOptions`]); default off (golden-pinned `REPLACE`).
    transparency: TransparencyOptions,
    max_objects: std::sync::atomic::AtomicU32,
    max_materials: std::sync::atomic::AtomicU32,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    gbuffer: GBufferTextures,
    gbuffer_pipeline: wgpu::RenderPipeline,
    gbuffer_bind_group_layout: wgpu::BindGroupLayout,
    gbuffer_bind_group: std::sync::RwLock<wgpu::BindGroup>,
    /// Skinned g-buffer pipeline: [`wgsl_vertex_source_skinned`] vertex
    /// stage (palette blend) with the classic fragment stage and the same
    /// five MRT targets. Draws interleaved [`SkinnedVertex`] buffers; bind
    /// groups are built per draw from `skinned_bind_group_layout` (no
    /// cached group — the palette slot differs per entry).
    skinned_pipeline: wgpu::RenderPipeline,
    /// Bind-group layout of the skinned passes (see
    /// [`GBUFFER_SKINNED_RESOURCES`]: camera, per-objects, materials plus
    /// the binding-3 palette).
    skinned_bind_group_layout: wgpu::BindGroupLayout,
    /// Depth-only skinned pipeline for the 2D shadow pre-pass (same vertex
    /// stage, no fragment): skinned entries cast skinned depth, never
    /// bind-pose depth.
    skinned_shadow_pipeline: wgpu::RenderPipeline,
    /// Mirrored (`front_face: Cw`) depth-only skinned pipeline for the
    /// point-shadow cube faces (see [`point_cube_face_vp`]).
    skinned_shadow_cube_pipeline: wgpu::RenderPipeline,
    /// Packed joint-palette storage: one [`PALETTE_BYTE_SIZE`] slot per
    /// staged entry, written by [`upload_skin_palettes`](Self::upload_skin_palettes).
    palette_buffer: std::sync::RwLock<wgpu::Buffer>,
    /// Capacity of [`Renderer3D::palette_buffer`] in palette slots; grows
    /// on demand like the per-object buffer.
    max_palettes: std::sync::atomic::AtomicU32,
    /// Slots written by the last [`upload_skin_palettes`](Self::upload_skin_palettes)
    /// call: draw handles at or past this are stale and record no commands
    /// (exact no-op, never an out-of-bounds bind).
    palette_count: std::sync::atomic::AtomicU32,
    lighting_pass: LightingPass,
    forward_pass: ForwardPass,
    /// Textured-forward pipeline plus fallbacks, built on demand by
    /// [`ensure_textured_forward`](Renderer3D::ensure_textured_forward);
    /// `None` until then (legacy flows never build it).
    textured_forward: Option<TexturedForwardPass>,
    composite_pass: CompositePass,
    /// Linear sampler shared by composite/bloom full-screen passes.
    composite_sampler: wgpu::Sampler,
    /// Bloom chain pipelines and params buffer.
    bloom_pass: BloomPass,
    /// Opt-in distance-fog pass (disabled by default: never runs unless a
    /// [`crate::frame_passes::FogPass`] with an enabled state records it).
    fog: FogPipeline,
    /// Shadow-map array (one depth layer per light slot) plus per-layer
    /// views, light-space VP uniform buffers, the depth-only pipeline,
    /// and the comparison sampler used by both lighting entries.
    shadow_maps: wgpu::Texture,
    shadow_views: [wgpu::TextureView; SHADOW_LAYERS],
    /// Full-array view for sampling (`texture_depth_2d_array`).
    shadow_array_view: wgpu::TextureView,
    shadow_vp_buffers: [wgpu::Buffer; SHADOW_LAYERS],
    shadow_pipeline: wgpu::RenderPipeline,
    shadow_sampler: wgpu::Sampler,
    /// Shadow-casting lights assigned by the last
    /// [`set_lights`](Self::set_lights) call (layers `0..count`).
    shadow_count: std::sync::atomic::AtomicU32,
    /// Directional-shadow fit: `(center, half-extent)` set via
    /// [`set_shadow_bounds`](Self::set_shadow_bounds) from the scene AABB.
    /// `None` (default) keeps the legacy ±[`SHADOW_ORTHO_HALF`] box around
    /// the origin, so scenes that never set bounds render pixel-identical.
    shadow_fit: std::sync::RwLock<Option<([f32; 3], f32)>>,
    /// Upload report of the last [`set_lights`](Self::set_lights) /
    /// [`set_lights_full`](Self::set_lights_full) call; read via
    /// [`light_upload_stats`](Self::light_upload_stats).
    last_light_stats: std::sync::RwLock<LightUploadStats>,
    /// Point-light shadow cubes: one depth cube per slot (6 faces),
    /// per-face views, sampling array view, per-face VP uniforms, and
    /// the active cube count. `params.w` on a point light indexes the
    /// cube slot (a separate index space from the 2D layers above —
    /// the evaluator picks the pool by light kind).
    shadow_cube_maps: wgpu::Texture,
    shadow_cube_views: [wgpu::TextureView; POINT_SHADOW_CUBES * CUBE_FACE_COUNT],
    shadow_cube_array_view: wgpu::TextureView,
    shadow_cube_vp_buffers: [wgpu::Buffer; POINT_SHADOW_CUBES * CUBE_FACE_COUNT],
    /// Mirrored depth-only pipeline for the cube faces (their VPs
    /// mirror NDC y — see [`point_cube_face_vp`] — so winding flips).
    shadow_cube_pipeline: wgpu::RenderPipeline,
    point_shadow_count: std::sync::atomic::AtomicU32,
    /// Split-sum IBL textures. Default is 1×1 black with weight 0.
    ibl: crate::ibl::IblTargets,
    /// LUT + black-face bytes copied into [`Self::ibl`] on the first pass
    /// that samples it (`new` has no queue).
    ibl_staging: wgpu::Buffer,
    /// Set once the IBL textures hold finite texels.
    ibl_uploaded: std::sync::atomic::AtomicBool,
    /// `f32` bits of the IBL weight, so [`set_lights`](Self::set_lights)
    /// (`&self`) can rewrite the uniform without clearing IBL.
    ibl_weight_bits: std::sync::atomic::AtomicU32,
    /// Set once [`Self::set_explicit_environment_weight`] runs. While set,
    /// cube binds keep this weight instead of the automatic 0/1.
    ibl_weight_explicit: std::sync::atomic::AtomicBool,
    /// `f32` bits of the prefilter's highest mip.
    ibl_max_mip_bits: std::sync::atomic::AtomicU32,
    /// [`ShadingDebug`] discriminant, so [`set_lights`](Self::set_lights)
    /// (`&self`) can rewrite the uniform without clearing the debug view.
    shading_debug: std::sync::atomic::AtomicU32,
}

impl Renderer3D {
    /// MSAA sample count this renderer was built with (see
    /// [`crate::render_backend::RenderBackendConfig::sample_count`]):
    /// [`SINGLE_SAMPLE_COUNT`] (1) or [`MSAA_SAMPLE_COUNT`] (4) — anything
    /// else passed to [`new`](Self::new) normalizes to 1.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Single-sample view of the forward HDR resolve target, `Some` only in
    /// MSAA mode (see [`MSAA_SAMPLE_COUNT`]): the frame-plan composite and
    /// bloom bright-pass sample this at 4x, where the pooled forward layer
    /// itself is multisampled. `None` at 1x (sample the pooled view there).
    pub fn forward_resolve_view(&self) -> Option<&wgpu::TextureView> {
        self.forward_pass.resolve_view.as_ref()
    }

    /// Exposure multiplier applied by [`set_lights`](Self::set_lights);
    /// `1.0` (default) is the exact no-op. Set from
    /// [`crate::render_backend::RenderBackendConfig::exposure`].
    pub fn exposure(&self) -> f32 {
        self.exposure
    }

    /// Replace the exposure multiplier applied by future
    /// [`set_lights`](Self::set_lights) calls (default `1.0`).
    pub fn set_exposure(&mut self, exposure: f32) {
        self.exposure = exposure;
    }

    /// Deferred lighting debug term. Stored across [`set_lights`](Self::set_lights);
    /// the next light upload writes it into `debug_view`. [`ShadingDebug::Beauty`]
    /// keeps the frame unchanged. Forward draws ignore the selector.
    pub fn set_shading_debug(&self, view: ShadingDebug) {
        self.shading_debug
            .store(view as u32, std::sync::atomic::Ordering::Relaxed);
    }

    /// Transparency mode of the forward pipeline (see
    /// [`TransparencyOptions`]).
    pub fn transparency(&self) -> TransparencyOptions {
        self.transparency
    }

    /// Reallocate all size-dependent textures and re-record dependent
    /// pipelines after the output extent changed. Extents are clamped to >= 1.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let width = width.max(1);
        let height = height.max(1);
        self.width = width;
        self.height = height;

        self.pbr_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pbr render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            // Fullscreen lighting output: single-sample in all modes (see `new`).
            sample_count: SINGLE_SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: DEFERRED_HDR_FORMAT,
            usage: RENDER_TARGET_USAGE,
            view_formats: &[],
        });
        self.pbr_texture_view = self
            .pbr_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gbuffer = Self::create_gbuffer(device, width, height, self.sample_count);
        let (gbuffer_pipeline, gbuffer_bind_group_layout, gbuffer_bind_group) =
            Self::create_gbuffer_pipeline(
                device,
                &self.gbuffer,
                &self.camera_buffer,
                &read_lock(&self.per_object_buffer),
                &read_lock(&self.material_buffer),
                self.sample_count,
            );
        self.gbuffer_pipeline = gbuffer_pipeline;
        self.gbuffer_bind_group_layout = gbuffer_bind_group_layout;
        *write_lock(&self.gbuffer_bind_group) = gbuffer_bind_group;

        self.lighting_pass =
            Self::create_lighting_pass(device, &self.pbr_texture_view, self.sample_count);

        self.forward_pass = Self::create_forward_pass(
            device,
            &self.camera_buffer,
            &read_lock(&self.per_object_buffer),
            &read_lock(&self.material_buffer),
            &self.lighting_buffer,
            &self.shadow_array_view,
            &self.shadow_sampler,
            &self.shadow_cube_array_view,
            &self.ibl,
            width,
            height,
            self.sample_count,
            self.transparency,
        );

        self.composite_pass = Self::create_composite_pass(device, self.format);
    }

    /// View of the final lit HDR image produced by the legacy-path lighting pass.
    pub fn pbr_view(&self) -> &wgpu::TextureView {
        &self.pbr_texture_view
    }

    /// Bytes allocated by the persistent textures of the legacy path: the
    /// five g-buffer MRTs plus g-buffer depth, lighting target, forward
    /// color, shadow-map array, point-shadow cubes, and — when
    /// [`MSAA_SAMPLE_COUNT`] is active — the single-sample MSAA resolve
    /// targets.
    pub fn texture_budget(&self) -> u64 {
        let bpp = crate::transient_pool::format_bytes_per_pixel;
        let w = self.width as u64;
        let h = self.height as u64;
        let s = self.sample_count as u64;
        let gbuffer = (bpp(wgpu::TextureFormat::Rgba8Unorm)
            + bpp(wgpu::TextureFormat::Rg16Float)
            + bpp(wgpu::TextureFormat::R16Uint)
            + bpp(wgpu::TextureFormat::Rg16Float)
            + bpp(wgpu::TextureFormat::Rgba16Float)
            + bpp(wgpu::TextureFormat::Depth32Float)) as u64
            * w
            * h
            * s;
        let pbr = bpp(DEFERRED_HDR_FORMAT) as u64 * w * h;
        let forward = bpp(wgpu::TextureFormat::Rgba16Float) as u64 * w * h * s;
        // Resolve targets are single-sample (`None` at 1x): four g-buffer
        // float layers plus the forward HDR color, only when MSAA is active.
        let resolves = if self.gbuffer.resolves.is_some() {
            (bpp(wgpu::TextureFormat::Rgba8Unorm)
                + bpp(wgpu::TextureFormat::Rg16Float)
                + bpp(wgpu::TextureFormat::Rg16Float)
                + bpp(wgpu::TextureFormat::Rgba16Float)) as u64
                * w
                * h
        } else {
            0
        } + if self.forward_pass.resolve_view.is_some() {
            bpp(wgpu::TextureFormat::Rgba16Float) as u64 * w * h
        } else {
            0
        };
        let shadow_size = self.shadow_maps.size();
        let shadow = bpp(wgpu::TextureFormat::Depth32Float) as u64
            * shadow_size.width as u64
            * shadow_size.height as u64
            * shadow_size.depth_or_array_layers as u64;
        let cube_size = self.shadow_cube_maps.size();
        let cubes = bpp(wgpu::TextureFormat::Depth32Float) as u64
            * cube_size.width as u64
            * cube_size.height as u64
            * cube_size.depth_or_array_layers as u64;
        gbuffer + pbr + forward + resolves + shadow + cubes
    }

    /// Upload the camera uniform: view-projection, its inverse (computed here)
    /// and eye position. Call once per frame before rendering.
    pub fn set_camera(&self, queue: &wgpu::Queue, view_proj: &[[f32; 4]; 4], camera_pos: [f32; 3]) {
        let inv_view_proj = glam::Mat4::from_cols_array_2d(view_proj)
            .inverse()
            .to_cols_array_2d();
        let uniform = CameraUniform {
            view_proj: *view_proj,
            inv_view_proj,
            camera_pos: [camera_pos[0], camera_pos[1], camera_pos[2], 1.0],
        };
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn msaa_sample_counts_normalize_to_one_or_four() {
        // The native MSAA count and the single-sample default/fallback.
        assert_eq!(MSAA_SAMPLE_COUNT, 4);
        assert_eq!(SINGLE_SAMPLE_COUNT, 1);
        // 4 passes through; everything else (including 0, 2, 8) falls back
        // to single-sample — `new` can never build a 2x/8x renderer.
        assert_eq!(normalize_sample_count(4), 4);
        for requested in [0, 1, 2, 3, 5, 8, 16, u32::MAX] {
            assert_eq!(
                normalize_sample_count(requested),
                1,
                "requested {requested} must fall back to 1x"
            );
        }
    }
}

#[cfg(test)]
#[path = "../msaa_edge_gpu.rs"]
mod msaa_edge_gpu;
