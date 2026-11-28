//! Fullscreen-quad vertex stage: translated from Rust through LLVM IR.
//!
//! The stage lives in `crates/gpu-quad`: `#[gpu_entry(vertex)] vs_main_quad`
//! reads `@builtin(vertex_index)`, derives the column and row bits, writes
//! `@builtin(position)` and the `@location(0)` UV, and nothing else. The passes
//! that draw a quad — deferred lighting and both AO methods — bind this one
//! module for their vertex stage and their own artifact for the fragment, so
//! the corner/UV table exists once instead of once per pass.
//!
//! The entry is named `vs_main_quad`, not `vs_main`: a pass that keeps its own
//! `#[gpu_entry(vertex)]` under the plain name can depend on `ornis-gpu-quad`
//! without the two entries colliding in that crate's build (the metadata
//! global is named after the entry).
//!
//! Regenerate with
//! `cargo xtask build-shaders --package ornis-gpu-quad --entry vs_main_quad`;
//! `xtask build-shaders --check` keeps the artifact in step with the kernel.

/// Vertex entry point of the translated module.
pub const VS_ENTRY: &str = "vs_main_quad";

/// Full WGSL source of the shared quad vertex stage.
#[must_use]
pub fn wgsl_vertex_source() -> String {
    include_str!("quad_vertex_generated.wgsl").to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage has to be a vertex stage that reads the vertex index and
    /// writes both outputs — a fragment-only module here would fail pipeline
    /// creation at runtime, not at build time.
    #[test]
    fn the_artifact_is_the_quad_vertex_stage() {
        let src = wgsl_vertex_source();
        assert!(src.contains("@vertex"), "artifact must declare @vertex");
        assert!(
            src.contains("fn vs_main_quad(@builtin(vertex_index) vertex_index: u32)"),
            "entry must read the vertex index:\n{src}"
        );
        assert!(src.contains("@builtin(position) position: vec4<f32>"));
        assert!(src.contains("@location(0) o0_: vec2<f32>"));
        assert!(!src.contains("@fragment"), "no fragment stage here");
    }

    /// Corners and UVs are the [`STANDARD_QUAD`]/[`STANDARD_UVS`] table spelled
    /// as two independent per-corner decisions. Pixel gates on the passes that
    /// draw it would catch a wrong corner as a flipped frame, but this pins the
    /// four corners and their four UVs literally, so a transposed bit (which
    /// only swaps two texels of a still-plausible image) cannot slip through.
    #[test]
    fn the_artifact_spells_the_shared_table() {
        use super::super::{STANDARD_QUAD, STANDARD_UVS};

        let src = wgsl_vertex_source();
        for [x, y, z, w] in STANDARD_QUAD {
            let corner = format!("vec4<f32>({x}f, {y}f, {z}f, {w}f)");
            assert!(src.contains(&corner), "missing corner {corner}:\n{src}");
        }
        // The UV write is built from the two hoisted lanes, so the four
        // `vec2<f32>(u, v)` literals are not in the text; assert the values
        // that do appear and that every UV lane of the table is among them.
        let lanes: Vec<String> = STANDARD_UVS
            .iter()
            .flatten()
            .map(|v| format!("{v}f"))
            .collect();
        for lane in &lanes {
            assert!(src.contains(lane.as_str()), "missing UV lane {lane}");
        }

        // Exactly one write per output, and the four corners are the whole
        // position story: a fifth corner write would mean an arm disagrees
        // with the table while still containing all four spellings.
        assert_eq!(
            src.matches("vertex_out.position = vec4<f32>(").count(),
            4,
            "one position write per corner:\n{src}"
        );
        assert_eq!(
            src.matches("vertex_out.o0_ = vec2<f32>(").count(),
            1,
            "the UV is written once from the hoisted lanes:\n{src}"
        );
    }
}
