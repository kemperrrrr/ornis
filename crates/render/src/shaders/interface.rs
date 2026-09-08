//! Shader-interface mirrors: vertex inputs, varyings, fragment I/O.
//!
//! Each struct here is the single source of truth for one WGSL interface
//! struct used by the render passes: the `@location`/`@builtin`/
//! `@interpolate` assignment is generated from the field attributes via
//! [`WgslInterface`](ornis_macros::WgslInterface), and the pass modules
//! splice `WGSL_SOURCE` into the assembled shaders. Unlike buffer layouts,
//! interfaces have no CPU-side memory contract — the structs below are
//! declaration-only (hence the module-level `allow(dead_code)`; stage-body
//! signatures will take them as parameters once the DSL covers entries).
//!
//! Naming: the Rust type names are pass-scoped (`HdrVertexOut`, …); the WGSL
//! names they emit (`#[wgsl(name)]`) match the legacy `shaders/wgsl/*.wgsl`
//! references.

#![allow(dead_code)]

use ornis_macros::WgslInterface;

/// HDR composite vertex output (fullscreen quad).
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "CompositeVertexOutput")]
pub(crate) struct HdrVertexOutput {
    /// Clip-space quad corner.
    #[wgsl(builtin = "position")]
    pub clip_position: [f32; 4],
    /// Quad UV.
    #[wgsl(location = 0)]
    pub uv: [f32; 2],
}

/// HDR composite fragment varying (fullscreen quad).
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "QuadVertexOutput")]
pub(crate) struct HdrFragmentOut {
    /// Clip-space quad corner.
    #[wgsl(builtin = "position")]
    pub clip_position: [f32; 4],
    /// Quad UV.
    #[wgsl(location = 0)]
    pub uv: [f32; 2],
}

/// Bloom fullscreen-quad varying.
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "BloomVertexOutput")]
pub(crate) struct BloomVertexOut {
    /// Clip-space quad corner.
    #[wgsl(builtin = "position")]
    pub clip_position: [f32; 4],
    /// Quad UV.
    #[wgsl(location = 0)]
    pub uv: [f32; 2],
}

/// Legacy UI-composite varying (`vs`/`fs` entries).
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "VertexOutput")]
pub(crate) struct UiCompositeOut {
    /// Clip-space quad corner (named `position` in this pass).
    #[wgsl(builtin = "position")]
    pub position: [f32; 4],
    /// Quad UV.
    #[wgsl(location = 0)]
    pub uv: [f32; 2],
}

/// G-buffer / forward-PBR vertex input (mesh attributes).
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "VertexInput")]
pub(crate) struct GbufferVertexInput {
    /// Object-space position.
    #[wgsl(location = 0)]
    pub position: [f32; 3],
    /// Object-space normal.
    #[wgsl(location = 1)]
    pub normal: [f32; 3],
    /// Texture coordinates.
    #[wgsl(location = 2)]
    pub uv: [f32; 2],
    /// Object-space tangent.
    #[wgsl(location = 3)]
    pub tangent: [f32; 3],
}

/// G-buffer / forward-PBR vertex output (world-space varying).
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "VertexOutput")]
pub(crate) struct GbufferVertexOutput {
    /// Clip-space position.
    #[wgsl(builtin = "position")]
    pub clip_position: [f32; 4],
    /// World-space position.
    #[wgsl(location = 0)]
    pub world_position: [f32; 3],
    /// World-space normal.
    #[wgsl(location = 1)]
    pub world_normal: [f32; 3],
    /// Texture coordinates.
    #[wgsl(location = 2)]
    pub uv: [f32; 2],
    /// World-space tangent.
    #[wgsl(location = 3)]
    pub world_tangent: [f32; 3],
    /// Material index (flat: no interpolation).
    #[wgsl(location = 4, interpolate = "flat")]
    pub material_index: u32,
}

/// G-buffer / forward-PBR fragment input (matches the vertex output).
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "FragmentInput")]
pub(crate) struct GbufferFragmentInput {
    /// World-space position.
    #[wgsl(location = 0)]
    pub world_position: [f32; 3],
    /// World-space normal.
    #[wgsl(location = 1)]
    pub world_normal: [f32; 3],
    /// Texture coordinates.
    #[wgsl(location = 2)]
    pub uv: [f32; 2],
    /// World-space tangent.
    #[wgsl(location = 3)]
    pub world_tangent: [f32; 3],
    /// Material index (flat: no interpolation).
    #[wgsl(location = 4, interpolate = "flat")]
    pub material_index: u32,
}

/// G-buffer (5-MRT) fragment output.
#[derive(Clone, Copy, Debug, WgslInterface)]
#[wgsl(name = "GBufferOutput")]
pub(crate) struct GbufferOutput {
    /// Base albedo + opacity.
    #[wgsl(location = 0)]
    pub albedo: [f32; 4],
    /// Octahedral-encoded normal.
    #[wgsl(location = 1)]
    pub normal: [f32; 2],
    /// Material identifier.
    #[wgsl(location = 2)]
    pub material_id: u32,
    /// World-space position (xy).
    #[wgsl(location = 3)]
    pub world_pos: [f32; 2],
    /// Packed material parameters.
    #[wgsl(location = 4)]
    pub mat_params: [f32; 4],
}
