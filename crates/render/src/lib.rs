pub mod composite;
pub mod frame_exec;
pub mod frame_passes;
pub mod frame_plan;
pub mod mesh;
pub mod render_backend;
pub mod renderer;
pub mod scene;
pub mod shader;
pub mod shaders;
pub mod system;
pub mod transform;

pub use composite::CompositePass as LegacyCompositePass;
pub use frame_exec::{FrameExecutor, FrameIds, PassViews, RenderFrame3D, Technique};
pub use frame_plan::{
    Budget, BudgetExceeded, FrameLayout, FramePlan, PassContext, PassId, PassLayout, PoolSlot,
    ResourceId, ResourceLayout, SizePolicy, TextureSpec, format_bytes_per_pixel,
};
pub use mesh::{Mesh, Vertex, create_sphere};
pub use ornis_core::{OPENPBR_MATERIAL_SIZE, OPENPBR_MATERIAL_VEC4_COUNT, OpenPBRMaterial};
/// Единая ошибка явных рёбер порядка (Фаза A, аудит §4.2) — тот же тип,
/// что реэкспортирует `ornis_core` для систем.
pub use ornis_schedule::OrderError;
pub use render_backend::{
    RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};
pub use renderer::{
    CameraUniform, CompositeInputs, CompositePass, ForwardPass, GBufferTextures, GbufferTargets,
    InstanceData, LightingPass, PerObjectGpu, Renderer3D,
};
pub use shader::{
    COMPOSITE_FRAGMENT, COMPOSITE_VERTEX, GBUFFER_FRAGMENT, GBUFFER_VERTEX, LIGHTING_FRAGMENT,
    LIGHTING_VERTEX, PBR_FRAGMENT, PBR_VERTEX,
};
pub use system::{
    Access, AccessSet, ClearBlack, ClearTransparent, ClearValue, ClearWhite, Frame, FramePass,
    FrameResource, Read, Resolver, ResourceKind, SystemSet, SystemViews, Write, WriteClear,
};
pub use transform::Transform;
