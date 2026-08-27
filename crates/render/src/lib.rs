//! Ornis render library: deferred [`renderer::Renderer3D`], the
//! frame-plan layer ([`frame_plan`]/[`frame_exec`]/[`frame_passes`]),
//! procedural meshes, scene descriptions and the WGSL shader assembly.
#![warn(missing_docs)]
/// Final PBR/UI blend pass (legacy path).
pub mod composite;
/// wgpu executor mapping plan slots to textures and running passes.
pub mod frame_exec;
/// Typed pass implementations wired into the frame plan.
pub mod frame_passes;
/// Pure immediate-mode render graph layout (lifetimes, pooling, budgets).
pub mod frame_plan;
/// GPU mesh representation and primitive generation.
pub mod mesh;
/// Backend-neutral rendering trait plus its factory.
pub mod render_backend;
/// The deferred [`renderer::Renderer3D`] and its passes.
pub mod renderer;
/// RON-serializable scene description types.
pub mod scene;
/// WGSL shader assembly and Rust-side BRDF math kernels.
pub mod shaders;
/// Typed plan systems: resources, access sets and pass traits.
pub mod system;
/// Local-to-world transform component.
pub mod transform;

pub use composite::CompositePass as LegacyCompositePass;
pub use frame_exec::{FrameExecutor, FrameIds, PassViews, RenderFrame3D, Technique};
pub use frame_plan::{
    Budget, BudgetExceeded, FrameLayout, FramePlan, PassContext, PassId, PassLayout, PoolSlot,
    ResourceId, ResourceLayout, SizePolicy, TextureSpec, format_bytes_per_pixel,
};
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
pub use system::{
    Access, AccessSet, ClearBlack, ClearTransparent, ClearValue, ClearWhite, Frame, FramePass,
    FrameResource, Read, Resolver, ResourceKind, SystemSet, SystemViews, Write, WriteClear,
};
pub use transform::Transform;
