pub mod material;
pub mod mesh;
pub mod transform;
pub mod renderer;
pub mod shader;
pub mod composite;

pub use material::{OpenPBRMaterial, OPENPBR_MATERIAL_VEC4_COUNT, OPENPBR_MATERIAL_SIZE};
pub use mesh::{Mesh, Vertex, create_sphere};
pub use transform::Transform;
pub use renderer::{Renderer3D, InstanceData, CameraUniform, PerObjectGpu, GBufferTextures, LightingPass, ForwardPass, CompositePass};
pub use composite::CompositePass as LegacyCompositePass;
pub use shader::{PBR_VERTEX, PBR_FRAGMENT, GBUFFER_VERTEX, GBUFFER_FRAGMENT, LIGHTING_VERTEX, LIGHTING_FRAGMENT, COMPOSITE_VERTEX, COMPOSITE_FRAGMENT};

use glam::Mat4;

pub trait RenderBackend {
    fn begin_frame(&mut self);
    fn draw_mesh(&mut self, mesh: &Mesh, material: &OpenPBRMaterial, transform: &Mat4);
    fn end_frame(&mut self) -> wgpu::TextureView;
}
