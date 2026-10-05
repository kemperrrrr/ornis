//! Ornis render library: deferred [`renderer::Renderer3D`], the
//! frame-plan layer ([`transient_pool`]/[`system`]/[`frame_exec`]/[`frame_passes`]),
//! procedural meshes and the WGSL shader assembly. Scene description types
//! live in `ornis-assets`; this crate only projects them.
#![warn(missing_docs)]
/// Client-side orbit camera and backend-neutral input consumer.
pub mod camera;
/// Final PBR/UI blend pass (legacy path).
pub mod composite;
/// Shared ECS-to-render extraction and logical render-world frame host.
pub mod extraction;
/// Typed replacements for `bool`/`u32` frame-plan flags.
pub mod flags;
/// wgpu executor mapping plan slots to textures and running passes.
pub mod frame_exec;
/// Typed pass implementations wired into the frame plan.
pub mod frame_passes;
/// GPU resources as ECS singletons for the unified scheduler (S7 design).
/// Native-only: stores wgpu `Device`/`Queue`/`Surface`/`CommandBuffer` as
/// `World` resources (`Send + Sync` bound), but wgpu's web backend types
/// are `!Send`/`!Sync` (Rc, JS callbacks). The wasm path renders through
/// `RenderWorld` extraction + `ornis-wasm`, not through this module.
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_resources;
/// Typed directional light spawned into a world light rig.
pub mod light;
/// GPU mesh representation and primitive generation.
pub mod mesh;
/// Upload of `ornis-mesh-editor` mesh data to the GPU.
pub mod mesh_upload;
/// Backend-neutral rendering trait plus its factory.
pub mod render_backend;
/// The deferred [`renderer::Renderer3D`] and its passes.
pub mod renderer;
/// E1 (S5e) bridge: frame passes projected as ordinary core `Schedule`
/// systems (declaration twins; level parity pinned by `scheduler_parity`).
pub mod schedule_bridge;
/// WGSL shader assembly and Rust-side BRDF math kernels.
pub mod shaders;
/// Phase D GPU skinning: joint-palette layout, skinned vertex stage and
/// palette upload bytes (CPU-skinned extraction stays the fallback).
pub mod skinning;
/// Typed plan systems + single declaration registry (d3).
pub mod system;
/// glTF texture image upload (`LoadedImage` → `wgpu` texture + sampler).
pub mod textures;
/// Transient pool — the dynamic half of the dissolved frame-plan shell
/// (d2): declaration snapshots compile into shared layouts here.
pub mod transient_pool;

pub use camera::{OrbitCamera, install_orbit_camera, read_orbit_camera};
pub use composite::CompositePass as LegacyCompositePass;
pub use extraction::{
    ExtractionStats, FrameUpload, MeshPose, RenderLights, RenderWorld, SkinInfluences,
    extract_render_data, extract_render_data_with_stats, max_mesh_params,
};
pub use flags::{
    Access as AccessKind, Bloom, CompositeTechnique, DepthOwnership, PassState, ResourceBacking,
    SamplerKind, ShadowCast,
};
pub use frame_exec::{FrameExecutor, FrameIds, PassViews, RenderFrame3D, Technique};
pub use frame_passes::{
    FogDensity, FogPass, FogPlacement, FogSettings, FogState, FogWiring, apply_fog, fog_factor,
};
pub use light::DirectionalLight;
pub use mesh::{Mesh, SkinnedVertex, Vertex, create_sphere};
pub use mesh_upload::{
    ConvertedSoup, SoupCache, SoupHash, UploadCache, UploadError, to_vertices, upload_mesh_data,
};
pub use ornis_core::{OPENPBR_MATERIAL_SIZE, OPENPBR_MATERIAL_VEC4_COUNT, OpenPBRMaterial};
/// Unified explicit-ordering edge error (Phase A, audit §4.2); the same type
/// `ornis_core` re-exports for systems.
pub use ornis_schedule::OrderError;
pub use render_backend::{
    RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};
pub use renderer::{
    BlendMode, CameraUniform, CompositeInputs, CompositePass, CustomGbufferDraw, FogInputs,
    ForwardPass, GBufferTextures, GbufferTargets, InstanceData, LightingPass, MSAA_SAMPLE_COUNT,
    MaterialIdx, PerObjectGpu, Renderer3D, SINGLE_SAMPLE_COUNT, StagedCustomMesh,
    TransparencyError, TransparencyOptions, custom_draw_items, forward_blend_state,
    negotiate_sample_count, normalize_sample_count, sort_by_depth, upload_vertex_rows,
};
pub use schedule_bridge::{ProjectionError, try_project_schedule};
pub use skinning::{
    GBUFFER_SKINNED_RESOURCES, JointPalette, PALETTE_BYTE_SIZE, PaletteHandle, SkinBindError,
    SkinJoint, SkinnedDraw, joint_palette_bytes, palette_upload_bytes, skinned_entry_point,
    wgsl_vertex_source_skinned,
};
pub use system::{
    Access, AccessSet, ClearBlack, ClearTransparent, ClearValue, ClearWhite, Frame, FramePass,
    FrameResource, Read, Resolver, ResourceKind, SystemSet, SystemViews, Write, WriteClear,
};
pub use textures::{
    CpuImage, GpuTexture, MaterialTextureSet, TextureCache, TextureHandle, TextureRole,
    TextureUploadError,
};
pub use transient_pool::{
    Budget, BudgetExceeded, FrameLayout, PassId, PassLayout, PassName, PoolSlot, ResourceId,
    ResourceLayout, ResourceName, SizePolicy, TextureSpec, TransientPool, format_bytes_per_pixel,
};
