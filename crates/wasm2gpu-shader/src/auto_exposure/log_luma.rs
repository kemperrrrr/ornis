//! Per-pixel selection, compositing, and log-luminance kernel.

use ornis_gpu_abi::storage_at_f32;
use ornis_gpu_abi::storage_at_u32;
use ornis_gpu_abi::storage_store_f32;

use super::FRAME_BINDING;
use super::INPUT_BINDING;
use super::LOG_LUMINANCE_BINDING;
use super::TEXEL_LANES;
use super::math::log_luminance;
use super::math::selected_channel;

const FORWARD_RGBA_OFFSET: u32 = TEXEL_LANES / 2;
const FORWARD_GREEN_OFFSET: u32 = 1;
const FORWARD_BLUE_OFFSET: u32 = 2;
const FORWARD_ALPHA_OFFSET: u32 = 3;

/// Wasm entry for per-pixel HDR log-luminance; writes `log_luminance[id]`.
///
/// Binding 0 stores eight packed `f32` lanes per pixel (deferred RGBA then
/// forward RGBA); binding 1 stores the resulting scalar log luminance. Frame
/// binding 3 contains pixel count at word 0 and layer mode at word 1.
#[unsafe(no_mangle)]
pub extern "C" fn ae_log_luma(id: u32) {
    let pixel_count = storage_at_u32::<0, FRAME_BINDING>(0);
    if id < pixel_count {
        let mode = storage_at_u32::<0, FRAME_BINDING>(1);
        let base = id * TEXEL_LANES;
        let deferred_r = storage_at_f32::<0, INPUT_BINDING>(base);
        let deferred_g = storage_at_f32::<0, INPUT_BINDING>(base + 1);
        let deferred_b = storage_at_f32::<0, INPUT_BINDING>(base + 2);
        let forward_base = base + FORWARD_RGBA_OFFSET;
        let forward_r = storage_at_f32::<0, INPUT_BINDING>(forward_base);
        let forward_g = storage_at_f32::<0, INPUT_BINDING>(forward_base + FORWARD_GREEN_OFFSET);
        let forward_b = storage_at_f32::<0, INPUT_BINDING>(forward_base + FORWARD_BLUE_OFFSET);
        let forward_alpha = storage_at_f32::<0, INPUT_BINDING>(forward_base + FORWARD_ALPHA_OFFSET);
        let red = selected_channel(mode, deferred_r, forward_r, forward_alpha);
        let green = selected_channel(mode, deferred_g, forward_g, forward_alpha);
        let blue = selected_channel(mode, deferred_b, forward_b, forward_alpha);
        storage_store_f32::<0, LOG_LUMINANCE_BINDING>(id, log_luminance(red, green, blue));
    }
}
