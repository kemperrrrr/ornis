pub mod composite;
pub mod graph_frame;
pub mod graph_passes;
pub mod mesh;
pub mod render_backend;
pub mod render_graph;
pub mod renderer;
pub mod scene;
pub mod shader;
pub mod shaders;
pub mod system;
pub mod transform;

pub use composite::CompositePass as LegacyCompositePass;
pub use graph_frame::{GraphExecutor, GraphIds, PassViews, RenderGraph3D, Technique};
pub use mesh::{Mesh, Vertex, create_sphere};
pub use ornis_core::{OPENPBR_MATERIAL_SIZE, OPENPBR_MATERIAL_VEC4_COUNT, OpenPBRMaterial};
/// Единая ошибка явных рёбер порядка (Фаза A, аудит §4.2) — тот же тип,
/// что реэкспортирует `ornis_core` для систем.
pub use ornis_schedule::OrderError;
pub use render_backend::{
    RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};
pub use render_graph::{
    Budget, BudgetExceeded, GraphLayout, PassContext, PassId, PassLayout, PoolSlot, RenderGraph,
    ResourceId, ResourceLayout, SizePolicy, TextureSpec, format_bytes_per_pixel,
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
    Access, AccessSet, ClearBlack, ClearTransparent, ClearValue, ClearWhite, Frame, GraphPass,
    GraphResource, Read, Resolver, ResourceKind, SystemSet, SystemViews, Write, WriteClear,
};
pub use transform::Transform;
