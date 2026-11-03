//! Texture imports used by the wasm32 ABI.

use crate::layout::GpuVec2;
use crate::layout::GpuVec4;

use super::Word4;

#[link(wasm_import_module = "ornis_gpu")]
unsafe extern "C" {
    fn texture_sample_2d(
        tex_group: u32,
        tex_binding: u32,
        samp_group: u32,
        samp_binding: u32,
        u: f32,
        v: f32,
    ) -> Word4;
    fn texture_sample_2d_lod(
        tex_group: u32,
        tex_binding: u32,
        samp_group: u32,
        samp_binding: u32,
        u: f32,
        v: f32,
        level: f32,
    ) -> Word4;
    fn texture_load_2d(group: u32, binding: u32, x: i32, y: i32) -> Word4;
    fn texture_width_2d(group: u32, binding: u32) -> u32;
    fn texture_height_2d(group: u32, binding: u32) -> u32;
}

pub(in crate::gpu_abi) fn wasm_sample<const TG: u32, const TB: u32, const SG: u32, const SB: u32>(
    uv: GpuVec2,
) -> GpuVec4 {
    let word = call_gpu_import!(texture_sample_2d(TG, TB, SG, SB, uv.x, uv.y));
    GpuVec4 {
        x: word.x,
        y: word.y,
        z: word.z,
        w: word.w,
    }
}

pub(in crate::gpu_abi) fn wasm_sample_lod<
    const TG: u32,
    const TB: u32,
    const SG: u32,
    const SB: u32,
>(
    uv: GpuVec2,
    level: f32,
) -> GpuVec4 {
    let word = call_gpu_import!(texture_sample_2d_lod(
        TG, TB, SG, SB, uv.x, uv.y, level
    ));
    GpuVec4 {
        x: word.x,
        y: word.y,
        z: word.z,
        w: word.w,
    }
}

pub(in crate::gpu_abi) fn call_texture_load_2d(
    group: u32,
    binding: u32,
    x: i32,
    y: i32,
) -> GpuVec4 {
    let word = call_gpu_import!(texture_load_2d(group, binding, x, y));
    GpuVec4 {
        x: word.x,
        y: word.y,
        z: word.z,
        w: word.w,
    }
}

pub(in crate::gpu_abi) fn call_texture_width_2d(group: u32, binding: u32) -> u32 {
    call_gpu_import!(texture_width_2d(group, binding))
}

pub(in crate::gpu_abi) fn call_texture_height_2d(group: u32, binding: u32) -> u32 {
    call_gpu_import!(texture_height_2d(group, binding))
}
