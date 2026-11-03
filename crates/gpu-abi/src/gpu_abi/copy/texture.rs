//! Explicit-LOD wasm texture sampling import.

use crate::layout::GpuVec2;
use crate::layout::GpuVec4;

use super::Word4;

#[link(wasm_import_module = "ornis_gpu")]
unsafe extern "C" {
    fn texture_sample_2d_lod(
        tex_group: u32,
        tex_binding: u32,
        samp_group: u32,
        samp_binding: u32,
        u: f32,
        v: f32,
        level: f32,
    ) -> Word4;
}

pub(super) fn wasm_sample_lod<const TG: u32, const TB: u32, const SG: u32, const SB: u32>(
    uv: GpuVec2,
    level: f32,
) -> GpuVec4 {
    // SAFETY: the ornis_gpu import returns four initialized f32 lanes and
    // does not load from guest linear memory.
    let word = unsafe { texture_sample_2d_lod(TG, TB, SG, SB, uv.x, uv.y, level) };
    GpuVec4 {
        x: word.x,
        y: word.y,
        z: word.z,
        w: word.w,
    }
}
