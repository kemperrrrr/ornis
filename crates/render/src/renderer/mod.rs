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

pub(crate) use buffers::staging_capacity_for_instances;
pub use buffers::{InstanceData, MaterialIdx, PerObjectGpu};
pub use deferred::LightingPass;
pub use forward::{
    BlendMode, ForwardPass, TexturedForwardPass, TransparencyError, TransparencyOptions,
    forward_blend_state, sort_by_depth,
};
pub use gbuffer::{GBufferTextures, GbufferTargets};
pub(crate) use lights::{
    GpuLight, GpuLightKind, LIGHT_KIND_DIRECTIONAL, LIGHT_KIND_POINT, LIGHT_KIND_SPOT,
    LightingUniform,
};
pub use lights::{LightUploadStats, MAX_LIGHTS, ShadingDebug, count_light_drops};
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
    use ornis_assets::scene::ShadowCast;

    fn clip_of(vp: [[f32; 4]; 4], p: [f32; 3]) -> glam::Vec4 {
        glam::Mat4::from_cols_array_2d(&vp) * glam::Vec4::new(p[0], p[1], p[2], 1.0)
    }

    fn ndc_of(vp: [[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let c = clip_of(vp, p);
        [c.x / c.w, c.y / c.w, c.z / c.w]
    }

    fn assert_valid_wgsl(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name} must parse: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} must validate: {e}"));
    }

    #[test]
    fn skinned_pipeline_sources_validate_with_naga() {
        // The skinned pipelines assemble the skinned vertex stage with the
        // classic fragment stage: both modules the renderer compiles must
        // validate (the shadow variants reuse the same vertex module).
        assert_valid_wgsl(
            "skinned_vertex",
            &crate::skinning::wgsl_vertex_source_skinned(),
        );
        assert_valid_wgsl("skinned_fragment", &crate::shaders::gbuffer_fragment());
        assert_eq!(crate::skinning::skinned_entry_point(), "vs_main_skinned");
    }

    #[test]
    fn skinned_upload_rejects_empty_and_bad_indices() {
        // Validation needs a device: skipped headless, like every probe.
        let Some((device, _)) = try_device() else {
            eprintln!("no GPU adapter; skipping skinned upload probe");
            return;
        };
        use crate::mesh_upload::UploadError;
        let vertex = SkinnedVertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            joints: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        assert!(matches!(
            upload_skinned_mesh(&device, &[], &[0, 1, 2]),
            Err(UploadError::EmptyMesh)
        ));
        assert!(matches!(
            upload_skinned_mesh(&device, &[vertex], &[]),
            Err(UploadError::EmptyMesh)
        ));
        assert!(matches!(
            upload_skinned_mesh(&device, &[vertex], &[0, 1]),
            Err(UploadError::InvalidMesh(_))
        ));
        assert!(matches!(
            upload_skinned_mesh(&device, &[vertex], &[0, 1, 7]),
            Err(UploadError::InvalidMesh(_))
        ));
        assert!(upload_skinned_mesh(&device, &[vertex; 3], &[0, 1, 2]).is_ok());
    }

    /// Skinned color + shadow draws bind the palette and record without
    /// panicking; a stale handle is an exact no-op. Skipped when no
    /// adapter is available.
    #[test]
    fn skinned_entry_renders_and_shadows_without_panic() {
        let Some((device, queue)) = try_device() else {
            eprintln!("no GPU adapter; skipping skinned draw probe");
            return;
        };
        const W: u32 = 64;
        const H: u32 = 64;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: W,
            height: H,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let renderer = Renderer3D::new(&device, &surface_config, 1);
        // Bind-group assembly pins the layout contract: four rows off the
        // skinned resource table, palette at binding 3, vertex-only.
        assert_eq!(GBUFFER_SKINNED_RESOURCES.len(), 4);
        let palette_row = GBUFFER_SKINNED_RESOURCES
            .iter()
            .find(|r| r.binding == 3)
            .expect("palette at binding 3");
        assert_eq!(palette_row.name, "palette");
        assert_eq!(
            palette_row.visibility,
            wgpu::ShaderStages::VERTEX,
            "palette is vertex-only"
        );
        {
            let per_object = read_lock(&renderer.per_object_buffer);
            let material = read_lock(&renderer.material_buffer);
            let palette = read_lock(&renderer.palette_buffer);
            let entries =
                crate::shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                    "camera" => Some(renderer.camera_buffer.as_entire_binding()),
                    "per_objects" => Some(per_object.as_entire_binding()),
                    "materials" => Some(material.as_entire_binding()),
                    "palette" => Some(palette.as_entire_binding()),
                    _ => None,
                })
                .expect("skinned resources resolve");
            assert_eq!(entries.len(), 4);
            assert_eq!(
                entries.iter().map(|e| e.binding).collect::<Vec<_>>(),
                vec![0, 1, 2, 3]
            );
        }

        let mut red = ornis_core::OpenPBRMaterial::dielectric();
        red.base.color_rgb([0.8, 0.2, 0.2]);
        red.specular.roughness(HALF);
        renderer.upload_materials(&device, &queue, &[red]);
        // Identity palette over one triangle: the vertex stage passes the
        // bind pose through (model is identity, like the extraction's
        // skinned entries).
        let row = |position: [f32; 3]| SkinnedVertex {
            position,
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0],
            joints: [0, 0, 0, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let mesh = upload_skinned_mesh(
            &device,
            &[
                row([-HALF, -HALF, 0.0]),
                row([HALF, -HALF, 0.0]),
                row([0.0, HALF, 0.0]),
            ],
            &[0, 1, 2],
        )
        .expect("triangle valid");
        let palette_bytes =
            crate::skinning::joint_palette_bytes(&[glam::Mat4::IDENTITY]).expect("fits");
        assert_eq!(palette_bytes.len(), PALETTE_BYTE_SIZE);
        let handles = renderer.upload_skin_palettes(&device, &queue, &[palette_bytes]);
        assert_eq!(handles, vec![PaletteHandle::from_raw(0)]);
        // Identity view-projection: clip = world, depth 0 < clear 1.
        renderer.set_camera(
            &queue,
            &[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [0.0, 0.0, 0.0],
        );
        // One shadowed light so the skinned depth pre-pass draws.
        renderer.set_lights_full(
            &queue,
            [0.1, 0.1, 0.15],
            1.0,
            1.0,
            &[ornis_assets::scene::LightDesc::Directional {
                direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(0.2, 1.0, 0.3))
                    .expect("non-zero direction"),
                intensity: 1.2,
                color: [1.0, 1.0, 1.0],
                shadow: ShadowCast::Enabled,
            }],
        );
        let g = GbufferTargets {
            albedo: &renderer.gbuffer.albedo_view,
            normal: &renderer.gbuffer.normal_view,
            material_id: &renderer.gbuffer.material_id_view,
            world_position: &renderer.gbuffer.world_position_view,
            material_params: &renderer.gbuffer.material_params_view,
            depth: &renderer.gbuffer.depth_view,
        };
        let instance = InstanceData {
            model_matrix: glam::Mat4::IDENTITY,
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("skinned probe encoder"),
        });
        renderer.render_skinned_entry(
            &device,
            &queue,
            &mut encoder,
            &g,
            &mesh,
            &instance,
            handles[0],
        );
        renderer.render_skinned_shadows(&device, &mut encoder, &mesh, handles[0]);
        // Stale handle: exact no-op, never an out-of-bounds bind.
        renderer.render_skinned_entry(
            &device,
            &queue,
            &mut encoder,
            &g,
            &mesh,
            &instance,
            PaletteHandle::from_raw(99),
        );
        renderer.render_skinned_shadows(&device, &mut encoder, &mesh, PaletteHandle::from_raw(99));
        queue.submit(std::iter::once(encoder.finish()));
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
    }

    #[test]
    fn dir_shadow_vp_centers_origin_with_light_depth_order() {
        // Light above: to-light = +Y, eye at +Y·30 looking at origin.
        let vp = dir_shadow_vp(glam::Vec3::Y);
        let center = ndc_of(vp, [0.0, 0.0, 0.0]);
        assert!(center[0].abs() < 1e-5 && center[1].abs() < 1e-5);
        assert!((0.0..=1.0).contains(&center[2]));
        // Nearer the light (higher Y) = smaller depth.
        let hi = ndc_of(vp, [0.0, 5.0, 0.0])[2];
        let lo = ndc_of(vp, [0.0, -5.0, 0.0])[2];
        assert!(hi < center[2] && center[2] < lo, "{hi} {center:?} {lo}");
        // Box rim stays inside the frustum.
        for p in [
            [12.0, 0.0, 0.0],
            [-12.0, 0.0, 0.0],
            [0.0, 0.0, 12.0],
            [0.0, 0.0, -12.0],
        ] {
            let n = ndc_of(vp, p);
            assert!(
                n[0].abs() <= 1.0 + 1e-4 && n[1].abs() <= 1.0 + 1e-4,
                "{p:?} -> {n:?}"
            );
        }
    }

    #[test]
    fn spot_shadow_vp_points_down_its_axis() {
        let vp = spot_shadow_vp([0.0, 6.0, 0.0], glam::Vec3::NEG_Y, 35.0, 40.0);
        let hit = ndc_of(vp, [0.0, 0.0, 0.0]);
        assert!(hit[0].abs() < 1e-4 && hit[1].abs() < 1e-4);
        assert!((0.0..=1.0).contains(&hit[2]));
        // Behind the light has negative clip w (mirrored projection).
        assert!(clip_of(vp, [0.0, 8.0, 0.0]).w < 0.0);
        // Off-axis outside the 35° cone (half-width ≈ 4.2 at depth 6;
        // the shadow frame maps world X onto NDC Y here).
        let side = ndc_of(vp, [5.0, 0.0, 0.0]);
        assert!(side[0].abs().max(side[1].abs()) > 1.0, "{side:?}");
    }

    #[test]
    fn point_cube_face_vp_centers_its_axis() {
        // Light at origin-ish, range 20: each face centers its axis
        // with directx depth order (nearer = smaller).
        let pos = [2.0, 3.0, 4.0];
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, axis) in axes.iter().enumerate() {
            let vp = point_cube_face_vp(pos, 20.0, face);
            let target = [
                pos[0] + axis[0] * 5.0,
                pos[1] + axis[1] * 5.0,
                pos[2] + axis[2] * 5.0,
            ];
            let hit = ndc_of(vp, target);
            assert!(
                hit[0].abs() < 1e-4 && hit[1].abs() < 1e-4,
                "face {face}: {hit:?}"
            );
            assert!((0.0..=1.0).contains(&hit[2]), "face {face}: {hit:?}");
            // Nearer along the axis = smaller depth (matches the
            // analytic sampling formula's monotonicity).
            let near = ndc_of(
                vp,
                [
                    pos[0] + axis[0] * 2.0,
                    pos[1] + axis[1] * 2.0,
                    pos[2] + axis[2] * 2.0,
                ],
            )[2];
            let far = ndc_of(
                vp,
                [
                    pos[0] + axis[0] * 10.0,
                    pos[1] + axis[1] * 10.0,
                    pos[2] + axis[2] * 10.0,
                ],
            )[2];
            assert!(
                near < hit[2] && hit[2] < far,
                "face {face}: {near} {hit:?} {far}"
            );
        }
    }

    #[test]
    fn point_cube_face_vp_matches_sampler_frame() {
        // Texel placement must match the hardware cube-sampling frame
        // (GL convention: +X: (−z,−y), −X: (+z,−y), +Y: (+x,+z),
        // −Y: (+x,−z), +Z: (+x,−y), −Z: (−x,−y)). Rasterization maps
        // NDC y+1 onto texture row 0 while the sampler reads v=0 from
        // the top, so a +u-source offset lands at positive NDC x and a
        // +v-source offset at NEGATIVE NDC y (the V-mirror in
        // `point_cube_face_vp`). Regression pin: without the mirror
        // the sampler reads mirrored texels (all-lit point shadows).
        let pos = [2.0, 3.0, 4.0];
        let axes: [[[f32; 3]; 2]; 6] = [
            [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
        ];
        let face_axis = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for (face, ([u, v], axis)) in axes.iter().zip(face_axis.iter()).enumerate() {
            let vp = point_cube_face_vp(pos, 20.0, face);
            let base = [
                pos[0] + axis[0] * 5.0,
                pos[1] + axis[1] * 5.0,
                pos[2] + axis[2] * 5.0,
            ];
            let pu = [base[0] + u[0], base[1] + u[1], base[2] + u[2]];
            let pv = [base[0] + v[0], base[1] + v[1], base[2] + v[2]];
            let nu = ndc_of(vp, pu);
            let nv = ndc_of(vp, pv);
            assert!(nu[0] > 0.05, "face {face}: +u offset -> {nu:?}");
            assert!(nv[1] < -0.05, "face {face}: +v offset -> {nv:?}");
        }
    }

    fn dir_probe(direction: [f32; 3], shadow: ornis_assets::scene::ShadowCast) -> LightDesc {
        LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::from_array(direction))
                .expect("non-zero direction"),
            intensity: 1.0,
            color: [1.0, 1.0, 1.0],
            shadow,
        }
    }

    #[test]
    fn per_object_material_index_substitutes_to_u32() {
        use crate::shaders::interface::{GbufferFragmentInput, GbufferVertexOutput};
        // The DSL substitutes the transparent `MaterialIdx` newtype to WGSL
        // `u32`: the declaration texts are unchanged from the raw-`u32` era.
        assert!(
            PerObjectGpu::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            PerObjectGpu::WGSL_SOURCE
        );
        assert_eq!(
            PerObjectGpu::FIELD_NAMES,
            &["model", "normal_matrix", "material_index"]
        );
        assert!(
            GbufferVertexOutput::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            GbufferVertexOutput::WGSL_SOURCE
        );
        assert!(
            GbufferFragmentInput::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            GbufferFragmentInput::WGSL_SOURCE
        );
        // CPU layout is unchanged: transparent wrapper, same offsets/size.
        assert_eq!(std::mem::size_of::<MaterialIdx>(), 4);
        assert_eq!(std::mem::size_of::<PerObjectGpu>(), 144);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, material_index), 128);
        // The derived declaration validates with naga.
        let mut module = naga::Module::default();
        let handle = PerObjectGpu::naga_add_type(&mut module);
        let naga::TypeInner::Struct { members, span } = &module.types[handle].inner else {
            panic!("PerObjectGpu must lower to a naga struct");
        };
        assert_eq!(*span, 144);
        assert_eq!(members[2].offset, 128);
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("PerObjectGpu declaration must validate");
    }

    #[test]
    fn lighting_uniform_spells_eight_lights() {
        // Multipliers stay CPU-baked. `debug_view` is the only added
        // shader-visible field; it occupies the former trailing pad.
        assert!(
            LightingUniform::WGSL_SOURCE.contains("array<Light, 8>"),
            "{}",
            LightingUniform::WGSL_SOURCE
        );
        assert!(LightingUniform::WGSL_SOURCE.contains("debug_view: u32"));
        assert_eq!(
            LightingUniform::FIELD_NAMES,
            &[
                "ambient_color",
                "lights",
                "light_count",
                "ibl_weight",
                "ibl_max_mip",
                "debug_view"
            ]
        );
        assert_eq!(
            std::mem::offset_of!(LightingUniform, debug_view)
                - std::mem::offset_of!(LightingUniform, ibl_max_mip),
            4
        );
        let built = build_lighting_uniform(
            [0.0; VEC3_COMPONENTS],
            1.0,
            1.0,
            &[],
            None,
            0.0,
            0.0,
            ShadingDebug::Shadow as u32,
        );
        assert_eq!(built.uniform.debug_view, ShadingDebug::Shadow as u32);
        assert_eq!(ShadingDebug::Beauty as u32, 0);
    }

    #[test]
    fn ten_lights_upload_eight_and_drop_two() {
        let lights: Vec<LightDesc> = (0..10)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        let built = build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &lights, None, 0.0, 0.0, 0);
        assert_eq!(
            built.stats,
            LightUploadStats {
                uploaded: 8,
                dropped_lights: 2,
                dropped_shadows: 0,
            }
        );
        assert_eq!(built.uniform.light_count, 8);
        // The public preview agrees with the upload path (single logic).
        assert_eq!(count_light_drops(&lights), built.stats);
    }

    #[test]
    fn shadow_overflow_counts_dropped_shadows() {
        // Five shadowed directionals over four 2D layers.
        let dirs: Vec<LightDesc> = (0..5)
            .map(|_| dir_probe([0.0, 1.0, 0.0], ornis_assets::scene::ShadowCast::Enabled))
            .collect();
        let stats = count_light_drops(&dirs);
        assert_eq!(stats.uploaded, 5);
        assert_eq!(stats.dropped_lights, 0);
        assert_eq!(stats.dropped_shadows, 1, "{stats:?}");
        // Three shadowed points over two cubes.
        let points: Vec<LightDesc> = (0..3)
            .map(|i| LightDesc::Point {
                position: glam::Vec3::new(i as f32, 4.0, 6.0),
                intensity: 100.0,
                color: [1.0, 1.0, 1.0],
                range: ornis_core::units::Meters::new(30.0),
                shadow: ShadowCast::Enabled,
            })
            .collect();
        let stats = count_light_drops(&points);
        assert_eq!(stats.dropped_shadows, 1, "{stats:?}");
        // A shadow request on a dropped excess light counts too.
        let mut lights: Vec<LightDesc> = (0..8)
            .map(|_| dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Disabled))
            .collect();
        lights.push(dir_probe(
            [0.0, 1.0, 0.0],
            ornis_assets::scene::ShadowCast::Enabled,
        ));
        lights.push(dir_probe(
            [0.0, 1.0, 0.0],
            ornis_assets::scene::ShadowCast::Enabled,
        ));
        let stats = count_light_drops(&lights);
        assert_eq!(stats.dropped_lights, 2, "{stats:?}");
        assert_eq!(stats.dropped_shadows, 2, "{stats:?}");
    }

    #[test]
    fn legacy_shadow_path_matches_dir_shadow_vp() {
        // `fit: None` must reproduce the legacy matrix bit-for-bit:
        // directional-only scenes stay pixel-identical.
        let light = dir_probe([1.0, 1.0, 1.0], ornis_assets::scene::ShadowCast::Enabled);
        let built = build_lighting_uniform([0.1, 0.1, 0.15], 1.0, 1.0, &[light], None, 0.0, 0.0, 0);
        let v = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        assert_eq!(built.uniform.lights[0].shadow_vp, dir_shadow_vp(v));
        assert_eq!(built.uniform.lights[0].params[3], 0.0);
    }

    #[test]
    fn fitted_shadow_covers_radius_50_scene() {
        let (center, half) = shadow_fit_for_bounds([-50.0; 3], [50.0; 3]);
        assert_eq!(center, [0.0, 0.0, 0.0]);
        assert!((half - 50.0).abs() < 1e-6, "{half}");
        let vp = dir_shadow_vp_fitted(glam::Vec3::Y, center, half);
        for x in [-50.0, 50.0] {
            for y in [-50.0, 50.0] {
                for z in [-50.0, 50.0] {
                    let n = ndc_of(vp, [x, y, z]);
                    assert!(
                        n[0].abs() <= 1.0 + 1e-3 && n[1].abs() <= 1.0 + 1e-3,
                        "{x},{y},{z} -> {n:?}"
                    );
                    assert!((0.0..=1.0).contains(&n[2]), "{x},{y},{z} -> {n:?}");
                }
            }
        }
    }

    #[test]
    fn default_fit_keeps_legacy_box() {
        assert_eq!(
            shadow_fit_for_bounds([-1.0; 3], [1.0; 3]),
            ([0.0; 3], SHADOW_ORTHO_HALF)
        );
        assert_eq!(SHADOW_ORTHO_HALF, 12.0);
        assert_eq!(
            shadow_fit_for_bounds([f32::NAN; 3], [0.0; 3]),
            ([0.0; 3], SHADOW_ORTHO_HALF)
        );
    }

    #[test]
    fn transparency_defaults_to_replace_and_sorts_far_first() {
        // Default mode is opaque: the forward pipeline blends with REPLACE
        // (golden-pinned); opt-in selects ALPHA_BLENDING.
        assert_eq!(TransparencyOptions::default().mode, BlendMode::Opaque);
        assert!(!TransparencyOptions::default().sorted_alpha());
        assert!(!BlendMode::Opaque.is_transparent());
        assert!(BlendMode::Transparent.is_transparent());
        assert_eq!(
            forward_blend_state(TransparencyOptions::default()),
            wgpu::BlendState::REPLACE
        );
        assert_eq!(
            forward_blend_state(TransparencyOptions::new(BlendMode::Transparent)),
            wgpu::BlendState::ALPHA_BLENDING
        );
        assert_eq!(BlendMode::Opaque.blend_state(), wgpu::BlendState::REPLACE);
        assert_eq!(
            BlendMode::Transparent.blend_state(),
            wgpu::BlendState::ALPHA_BLENDING
        );
        // Legacy bool polarity is preserved.
        assert_eq!(BlendMode::from(false), BlendMode::Opaque);
        assert_eq!(BlendMode::from(true), BlendMode::Transparent);
        assert_eq!(TransparencyOptions::from(false).mode, BlendMode::Opaque);
        // Opacity classification: 1.0+ is opaque, below is transparent,
        // non-finite is rejected.
        assert_eq!(BlendMode::from_opacity(1.0), Ok(BlendMode::Opaque));
        assert_eq!(BlendMode::from_opacity(2.0), Ok(BlendMode::Opaque));
        assert_eq!(BlendMode::from_opacity(HALF), Ok(BlendMode::Transparent));
        assert_eq!(BlendMode::from_opacity(0.0), Ok(BlendMode::Transparent));
        assert!(matches!(
            BlendMode::from_opacity(f32::NAN),
            Err(TransparencyError::NonFiniteOpacity(_))
        ));
        assert!(matches!(
            BlendMode::from_opacity(f32::INFINITY),
            Err(TransparencyError::NonFiniteOpacity(_))
        ));
        // Depth sort: far first, stable on ties, empty stays empty.
        // Transparent mode requires this order before uploading (see
        // `TransparencyOptions`; the whole-frame twin is
        // `crate::extraction::sort_by_depth`).
        assert_eq!(sort_by_depth(&[]), Vec::<u32>::new());
        assert_eq!(sort_by_depth(&[2.0]), vec![0]);
        assert_eq!(sort_by_depth(&[1.0, 5.0, 3.0]), vec![1, 2, 0]);
        assert_eq!(sort_by_depth(&[2.0, 2.0, 1.0]), vec![0, 1, 2]);
    }

    #[test]
    fn fog_uniform_layout_matches_wgsl() {
        // Layout gates (precedent: Camera/Lighting via `WgslStruct`):
        // `color: vec3<f32>` at 0, `density: f32` at 12, 16 bytes total.
        assert_eq!(std::mem::size_of::<FogUniform>(), 16);
        assert_eq!(std::mem::offset_of!(FogUniform, color), 0);
        assert_eq!(std::mem::offset_of!(FogUniform, density), 12);
        assert_eq!(FogUniform::FIELD_NAMES, &["color", "density"]);
        assert!(
            FogUniform::WGSL_SOURCE.contains("color: vec3<f32>"),
            "{}",
            FogUniform::WGSL_SOURCE
        );
        assert!(
            FogUniform::WGSL_SOURCE.contains("density: f32"),
            "{}",
            FogUniform::WGSL_SOURCE
        );
        let packed = FogUniform::pack([HALF, 0.6, 0.7], 0.1);
        let bytes = bytemuck::bytes_of(&packed);
        assert_eq!(bytes.len(), 16);
        assert_eq!(
            &bytes[0..12],
            bytemuck::cast_slice::<f32, u8>(&[HALF, 0.6, 0.7])
        );
    }

    #[test]
    fn ibl_multipliers_scale_upload_and_default_is_noop() {
        let lights = vec![LightDesc::Directional {
            direction: ornis_core::units::UnitVec3::normalize(glam::Vec3::new(1.0, 1.0, 1.0))
                .expect("non-zero direction"),
            intensity: 2.0,
            color: [HALF, 0.25, 0.125],
            shadow: ShadowCast::Disabled,
        }];
        let base =
            build_lighting_uniform([HALF, 0.25, 0.125], 1.0, 1.0, &lights, None, 0.0, 0.0, 0);
        assert_eq!(base.uniform.ambient_color, [HALF, 0.25, 0.125, 1.0]);
        assert_eq!(base.uniform.lights[0].color, [HALF, 0.25, 0.125, 2.0]);
        let scaled =
            build_lighting_uniform([HALF, 0.25, 0.125], 2.0, 4.0, &lights, None, 0.0, 0.0, 0);
        assert_eq!(scaled.uniform.ambient_color, [1.0, HALF, 0.25, 1.0]);
        assert_eq!(scaled.uniform.lights[0].color, [2.0, 1.0, HALF, 2.0]);
    }

    #[test]
    fn staging_reserve_is_exact_and_stable_at_same_size() {
        // The upload path reserves once up front: exact fit for the
        // frame, and re-reserving the same size never reallocates.
        assert_eq!(staging_capacity_for_instances(300), 300);
        let mut staging: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(300));
        assert!(staging.capacity() >= 300);
        let capacity = staging.capacity();
        staging.reserve(staging_capacity_for_instances(300).saturating_sub(staging.len()));
        assert_eq!(staging.capacity(), capacity, "same size: no realloc");
    }

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

    /// Adapter handle, or `None` on headless CI without a GPU.
    fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                flags: wgpu::InstanceFlags::empty(),
                backend_options: wgpu::BackendOptions::default(),
                memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
                display: None,
            });
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions::default())
                .await
                .ok()?;
            adapter
                .request_device(&wgpu::DeviceDescriptor::default())
                .await
                .ok()
        })
    }

    /// Explicit IBL weight reaches the lighting uniform and survives a
    /// cube bind. Without the setter, no cube stays at 0 and a cube
    /// becomes 1. Skipped when no adapter is available.
    #[test]
    fn explicit_environment_weight_survives_cube_bind() {
        /// Side of the headless surface used only to construct the renderer.
        const SIDE: u32 = 4;
        /// Explicit weight, distinct from both automatic endpoints.
        const EXPLICIT_WEIGHT: f32 = 0.35;
        let Some((device, queue)) = try_device() else {
            return;
        };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8Unorm,
            width: SIDE,
            height: SIDE,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        let cube = crate::ibl::EnvironmentCube::solid(ornis_core::units::Color::WHITE, SIDE);
        let mut automatic = Renderer3D::new(&device, &config, 1);
        assert_eq!(automatic.ibl_weight_for_tests(), 0.0);
        automatic.set_image_based_light(&device, &queue, None);
        assert_eq!(automatic.ibl_weight_for_tests(), 0.0);
        automatic.set_image_based_light(&device, &queue, Some(&cube));
        assert_eq!(automatic.ibl_weight_for_tests(), 1.0);
        let mip = automatic.ibl_max_mip_for_tests();
        assert!(mip > 0.0, "a bound cube publishes a prefilter mip");
        automatic.set_lights(&queue, [0.1, 0.1, 0.1], &[]);
        assert_eq!(automatic.ibl_weight_for_tests(), 1.0);
        assert_eq!(automatic.ibl_max_mip_for_tests(), mip);

        let mut explicit = Renderer3D::new(&device, &config, 1);
        explicit.set_explicit_environment_weight(&queue, EXPLICIT_WEIGHT);
        assert_eq!(explicit.ibl_weight_for_tests(), EXPLICIT_WEIGHT);
        explicit.set_image_based_light(&device, &queue, Some(&cube));
        assert_eq!(explicit.ibl_weight_for_tests(), EXPLICIT_WEIGHT);
        assert!(explicit.ibl_max_mip_for_tests() > 0.0);
        explicit.set_lights(&queue, [0.1, 0.1, 0.1], &[]);
        assert_eq!(explicit.ibl_weight_for_tests(), EXPLICIT_WEIGHT);
        let kept_mip = explicit.ibl_max_mip_for_tests();
        explicit.set_explicit_environment_weight(&queue, EXPLICIT_WEIGHT);
        assert_eq!(explicit.ibl_max_mip_for_tests(), kept_mip);
    }

    /// GPU/CPU parity smoke for the enabled fog mix: a solid HDR layer over
    /// a cleared (far-plane) depth buffer, fogged on the GPU, must land
    /// within tolerance of ACES([`crate::frame_passes::apply_fog`]) fed with
    /// the same view-space distance the shader reconstructs.
    ///
    /// The fresh `Renderer3D` camera is the identity (eye at the origin),
    /// so the reconstruction is exact on paper: NDC `(u*2-1, 1-v*2, 1)`
    /// maps to itself and the distance is its length. Skipped when no
    /// adapter is available.
    #[test]
    fn fog_enabled_gpu_matches_cpu_apply_fog() {
        const W: u32 = 32;
        const H: u32 = 2;
        const BPP: u32 = 4;
        const ROW: u32 = W * BPP; // 128 — under the 256 copy alignment…
        // …so pad rows to 256 bytes for the readback copy.
        const PADDED_ROW: u32 = 256;
        const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        const INPUT: [f32; 3] = [0.2, 0.4, 0.6];
        const FOG_COLOR: [f32; 3] = [0.9, 0.1, 0.1];
        const DENSITY: f32 = 3.0;
        // Tolerance covers u8 quantization (1/255) plus f32 exp/MAD
        // ordering between CPU and GPU.
        const TOL: f32 = 0.03;

        fn run_case(density: f32) -> Option<[f32; 3]> {
            let (device, queue) = try_device()?;
            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: FMT,
                width: W,
                height: H,
                present_mode: wgpu::PresentMode::AutoNoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            };
            let renderer = Renderer3D::new(&device, &surface_config, 1);

            let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            let input_byte = [byte(INPUT[0]), byte(INPUT[1]), byte(INPUT[2]), 255];
            let extent = wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            };
            let hdr_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fog parity hdr"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let mut hdr_data = vec![0u8; (ROW * H) as usize];
            for px in hdr_data.chunks_exact_mut(4) {
                px.copy_from_slice(&input_byte);
            }
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &hdr_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &hdr_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW),
                    rows_per_image: Some(H),
                },
                extent,
            );
            // Far-plane depth: clear-only pass over a depth texture.
            let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fog parity depth"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let target_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fog parity target"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FMT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let target = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
            let hdr = hdr_tex.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fog parity encoder"),
            });
            // Initialize both attachments first (`render_fog` loads the
            // target instead of clearing it).
            {
                let clear_depth = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fog parity init"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &clear_depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
            renderer.render_fog(
                &device,
                &queue,
                &mut encoder,
                FogInputs {
                    hdr: &hdr,
                    depth: &depth_view,
                    target: &target,
                    color: FOG_COLOR,
                    density,
                },
            );
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fog parity readback"),
                size: (PADDED_ROW * H) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(PADDED_ROW),
                        rows_per_image: Some(H),
                    },
                },
                extent,
            );
            queue.submit([encoder.finish()]);

            let slice = buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .ok();
            rx.recv().ok()?.ok()?;
            let view = slice.get_mapped_range().unwrap();
            // Center-ish texel (x=16, y=1): deterministic uv, same math
            // the CPU reference uses below.
            let px = (PADDED_ROW + 16 * BPP) as usize;
            let pixel = [
                view[px] as f32 / 255.0,
                view[px + 1] as f32 / 255.0,
                view[px + 2] as f32 / 255.0,
            ];
            Some(pixel)
        }

        // Reconstructed distance for texel (16, 1) under the identity
        // camera: uv = ((16+0.5)/32, (1+0.5)/2), NDC z = 1 (cleared far).
        let u = (16.0 + HALF) / W as f32;
        let v = (1.0 + HALF) / H as f32;
        let dist = ((2.0 * u - 1.0).powi(2) + (1.0 - 2.0 * v).powi(2) + 1.0).sqrt();
        let fog = crate::frame_passes::FogState::Enabled(
            crate::frame_passes::FogSettings::try_from_raw(FOG_COLOR, DENSITY)
                .expect("positive density"),
        );
        let tonemap = |rgb: [f32; 3]| {
            let mapped = crate::shaders::math::aces_tonemap::eval(glam::Vec3::from(rgb));
            [mapped.x, mapped.y, mapped.z]
        };
        // The pass mixes in scene-linear space, then applies the same ACES
        // the composite uses, because fog replaces that present.
        let expected = tonemap(crate::frame_passes::apply_fog(INPUT, dist, fog));
        let tonemapped_input = tonemap(INPUT);
        let tonemapped_fog = tonemap(FOG_COLOR);

        let Some(px) = run_case(DENSITY) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for i in 0..3 {
            assert!(
                (px[i] - expected[i]).abs() <= TOL,
                "channel {i}: gpu={px:?} cpu={expected:?} (dist={dist})"
            );
        }
        // High density visibly moved toward the fog color…
        for i in 0..3 {
            assert!(
                (px[i] - tonemapped_fog[i]).abs() < (tonemapped_input[i] - tonemapped_fog[i]).abs(),
                "no fog movement: {px:?}"
            );
        }
        // …while a near-zero density keeps the tonemapped input.
        let faint = crate::frame_passes::FogState::Enabled(
            crate::frame_passes::FogSettings::try_from_raw(FOG_COLOR, 1.0e-4)
                .expect("positive density"),
        );
        let faint_expected = tonemap(crate::frame_passes::apply_fog(INPUT, dist, faint));
        let Some(faint_px) = run_case(1.0e-4) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for i in 0..3 {
            assert!(
                (faint_px[i] - faint_expected[i]).abs() <= TOL,
                "faint channel {i}: gpu={faint_px:?} cpu={faint_expected:?}"
            );
            assert!(
                (faint_px[i] - tonemapped_input[i]).abs() < 0.02,
                "near-zero density drifted: {faint_px:?}"
            );
        }
    }
}

#[cfg(test)]
#[path = "../msaa_edge_gpu.rs"]
mod msaa_edge_gpu;
