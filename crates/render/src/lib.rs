pub mod composite;
pub mod mesh;
pub mod render_backend;
pub mod render_graph;
pub mod renderer;
pub mod scene;
pub mod shader;
pub mod shaders;
pub mod transform;

pub use composite::CompositePass as LegacyCompositePass;
pub use mesh::{Mesh, Vertex, create_sphere};
pub use ornis_core::{OPENPBR_MATERIAL_SIZE, OPENPBR_MATERIAL_VEC4_COUNT, OpenPBRMaterial};
pub use render_backend::{
    RenderBackend, RenderBackendConfig, RenderContext, create_render_backend,
};
pub use render_graph::{
    GraphLayout, PassContext, PassId, PassLayout, PoolSlot, RenderGraph, ResourceId,
    ResourceLayout, SizePolicy, TextureSpec,
};
pub use renderer::{
    CameraUniform, CompositePass, ForwardPass, GBufferTextures, InstanceData, LightingPass,
    PerObjectGpu, Renderer3D,
};
pub use shader::{
    COMPOSITE_FRAGMENT, COMPOSITE_VERTEX, GBUFFER_FRAGMENT, GBUFFER_VERTEX, LIGHTING_FRAGMENT,
    LIGHTING_VERTEX, PBR_FRAGMENT, PBR_VERTEX,
};
pub use transform::Transform;
