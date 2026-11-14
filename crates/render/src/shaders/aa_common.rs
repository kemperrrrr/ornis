//! Shared WGSL pieces of the anti-aliasing passes: the fullscreen-quad
//! vertex entry and the colour-space helpers the resolve kernels work in.
//!
//! The edge filter moved to `crates/gpu-aa-edge` (translated from Rust); its
//! luma weights live in that kernel, the temporal resolve here still needs the
//! helpers below.
//!
//! The bodies are raw WGSL (not `#[stage]`): the temporal resolve needs
//! indexed loops and the edge filters a per-mode branch, neither of which
//! the DSL covers. The CPU reference of the same math lives in
//! [`crate::antialias`].

/// Fullscreen vertex entry shared by the anti-aliasing passes
/// (triangle-strip quad).
pub const VERTEX_ENTRY: &str = r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertexOutput {
    return QuadVertexOutput(quad[vertex_index], uvs[vertex_index]);
}
"#;

/// Reversible YCoCg encoding of an RGB colour.
///
/// The chroma axes are decorrelated from the luma, so clamping the
/// history in this space keeps its brightness while rejecting the colour
/// drift that shows as ghosting. Matches
/// [`crate::antialias::ycocg_encode`].
pub const YCOCG_HELPER: &str = r#"
fn aa_ycocg_encode(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn aa_ycocg_decode(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}
"#;
