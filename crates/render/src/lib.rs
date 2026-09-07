//! Ornis render library: deferred [`renderer::Renderer3D`], the
//! frame-plan layer ([`transient_pool`]/[`system`]/[`frame_exec`]/[`frame_passes`]),
//! procedural meshes, scene descriptions and the WGSL shader assembly.
#![warn(missing_docs)]
/// Client-side orbit camera and backend-neutral input consumer.
pub mod camera;
/// Final PBR/UI blend pass (legacy path).
pub mod composite;
/// Shared ECS-to-render extraction and logical render-world frame host.
pub mod extraction;
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
/// GPU mesh representation and primitive generation.
pub mod mesh;
/// Backend-neutral rendering trait plus its factory.
pub mod render_backend;
/// The deferred [`renderer::Renderer3D`] and its passes.
pub mod renderer;
/// RON-serializable scene description types.
pub mod scene;
/// E1 (S5e) bridge: frame passes projected as ordinary core `Schedule`
/// systems (declaration twins; level parity pinned by `scheduler_parity`).
pub mod schedule_bridge;
/// WGSL shader assembly and Rust-side BRDF math kernels.
pub mod shaders;
/// Typed plan systems + single declaration registry (d3).
pub mod system;
/// Local-to-world transform component.
pub mod transform;
/// Transient pool — the dynamic half of the dissolved frame-plan shell
/// (d2): declaration snapshots compile into shared layouts here.
pub mod transient_pool;

pub use camera::{OrbitCamera, install_orbit_camera, read_orbit_camera};
pub use composite::CompositePass as LegacyCompositePass;
pub use extraction::{
    RenderExtracted, RenderLights, RenderWorld, extract_render_data, install_render_extract,
    max_mesh_params,
};
pub use frame_exec::{FrameExecutor, FrameIds, PassViews, RenderFrame3D, Technique};
pub use mesh::{Mesh, Vertex, create_sphere};
pub use ornis_core::{OPENPBR_MATERIAL_SIZE, OPENPBR_MATERIAL_VEC4_COUNT, OpenPBRMaterial};
/// Unified explicit-ordering edge error (Phase A, audit §4.2); the same type
/// `ornis_core` re-exports for systems.
pub use ornis_schedule::OrderError;
pub use render_backend::{
    RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};
pub use renderer::{
    CameraUniform, CompositeInputs, CompositePass, ForwardPass, GBufferTextures, GbufferTargets,
    InstanceData, LightingPass, PerObjectGpu, Renderer3D,
};
pub use schedule_bridge::{ProjectionError, try_project_schedule};
pub use system::{
    Access, AccessSet, ClearBlack, ClearTransparent, ClearValue, ClearWhite, Frame, FramePass,
    FrameResource, Read, Resolver, ResourceKind, SystemSet, SystemViews, Write, WriteClear,
};
pub use transform::Transform;
pub use transient_pool::{
    Budget, BudgetExceeded, FrameLayout, PassId, PassLayout, PoolSlot, ResourceId, ResourceLayout,
    SizePolicy, TextureSpec, TransientPool, format_bytes_per_pixel,
};
