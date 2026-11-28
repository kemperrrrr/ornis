//! Fullscreen panel blit for the stats window, translated from stable Rust.
//!
//! The stats window draws a CPU-composed [`Canvas`] into a texture and blits it
//! one texel per pixel. The shader is a fullscreen triangle from
//! [`vertex_index`] plus a fragment stage that reads `@builtin(position)` and
//! `textureLoad`s the clamps each axis to the last texel — the surface is
//! resized independently of the panel, so a read at `width` would be out of
//! bounds rather than simply black.
//!
//! The pair is one crate and one checked-in module
//! (`crates/runner/src/stats/blit_generated.wgsl`), because a vertex and a
//! fragment stage of one pass share a module and its interface: the vertex
//! writes only `@builtin(position)`, and the fragment has no `@location`
//! inputs at all — it maps framebuffer pixels to texels itself.
//!
//! [`Canvas`]: https://docs.rs/ornis-runner

#![no_std]
#![warn(missing_docs)]

/// Name of the vertex entry, as `#[gpu_entry(vertex)]` exports it.
///
/// The stats window names it in `entry_point` and the artifact carries it, so
/// a rename has to be a compile error there rather than a pipeline creation
/// failure at run time.
pub const VERTEX_ENTRY: &str = "vs_main";

/// Name of the fragment entry, as `#[gpu_entry(fragment)]` exports it.
pub const FRAGMENT_ENTRY: &str = "fs_main";

use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::fragment_position_lane;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::vertex_index;
use ornis_gpu_shader::vertex_position;

/// The blit triangle has three vertices.
const TRIANGLE_VERTICES: u32 = 3;
/// Corners come from these two bits of the vertex index.
const CORNER_MASK: u32 = 2;
/// Positions map the `0..2` corner grid onto clip space `-1..1`.
const CLIP_SCALE: f32 = 2.0;
/// Half the corner grid's span: `corner * 2 - 1` is `-1..1`.
const CLIP_OFFSET: f32 = 1.0;
/// Lane of `@builtin(position)` the blit reads as the pixel column.
const POSITION_X: u32 = 0;
/// Lane of `@builtin(position)` the blit reads as the pixel row.
const POSITION_Y: u32 = 1;
/// Lane of the alpha channel: an `RGBA` texel is four lanes.
const ALPHA_LANE: u32 = 3;

/// One axis of the texel coordinate for a framebuffer pixel.
///
/// The surface and the panel are resized independently, so the last row and
/// column of pixels can land one past the panel: the coordinate clamps to
/// `dim - 1` instead of reading out of bounds. `position` is the
/// `@builtin(position)` coordinate, which is a pixel centre, so truncating it
/// is the pixel index.
#[inline(always)]
pub(crate) fn texel_axis(position: f32, dim: i32) -> u32 {
    let pixel = position as i32;
    let last = dim - 1;
    let clamped = if pixel > last { last } else { pixel };
    let mut word = 0;
    if clamped > 0 {
        word = clamped as u32;
    }
    word
}

/// Fullscreen-triangle vertex stage.
///
/// Three vertices cover the target: `(0,0)`, `(2,0)`, `(0,2)` in corner space
/// scaled to clip space, which puts the triangle's covered area exactly over
/// the viewport with no vertex buffer and no index buffer.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main")]`.
#[gpu_entry(vertex)]
pub fn vs_main() {
    let index = vertex_index();
    const {
        assert!(
            TRIANGLE_VERTICES == 3,
            "a fullscreen triangle has 3 vertices"
        )
    };
    // `(index << 1) & 2` and `index & 2`: the corner grid, one lane at a time.
    let corner_x = (index << 1) & CORNER_MASK;
    let corner_y = index & CORNER_MASK;
    let x = corner_x as f32 * CLIP_SCALE - CLIP_OFFSET;
    let y = corner_y as f32 * CLIP_SCALE - CLIP_OFFSET;
    vertex_position(x, y, 0.0, 1.0);
}

/// One texel of the panel, copied to the framebuffer pixel at the same place.
///
/// No sampler: the panel is sampled at exact texel centres, so a filtered
/// `textureSample` would blur a 1:1 copy. The four lanes are stored
/// individually because the SDK's fragment writers take scalars.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "fs_main")]`.
#[gpu_entry(fragment)]
pub fn fs_main(#[gpu(texture_2d, group = 0, binding = 0)] panel: Texture2d) {
    let x = texel_axis(fragment_position_lane::<POSITION_X>(), panel.width());
    let y = texel_axis(fragment_position_lane::<POSITION_Y>(), panel.height());
    let r = panel.load::<0>(x, y);
    let g = panel.load::<1>(x, y);
    let b = panel.load::<2>(x, y);
    let a = panel.load::<ALPHA_LANE>(x, y);
    fragment_store(r, g, b, a);
}

#[cfg(test)]
mod tests {
    use super::texel_axis;

    /// The coordinate policy, against the hand-written shader it replaces:
    /// `min(i32(position), dim - 1)` truncated to an unsigned index. A pixel
    /// past the panel reads the last texel instead of out of bounds, and a
    /// negative coordinate (which a resize mid-frame can produce) reads the
    /// first one.
    #[test]
    fn texel_axis_clamps_to_the_panel() {
        let master = |position: f32, dim: i32| -> u32 {
            let pixel = position as i32;
            let last = dim - 1;
            let clamped = if pixel > last { last } else { pixel };
            clamped.max(0) as u32
        };
        for dim in [1_i32, 2, 3, 64, 420, 380] {
            for position in [
                0.0_f32, 0.5, 1.0, 1.9, 63.5, 379.0, 380.0, 419.9, 10_000.0, -1.0,
            ] {
                assert_eq!(
                    texel_axis(position, dim),
                    master(position, dim),
                    "position {position}, dim {dim}"
                );
            }
            // The last pixel is the last texel, which is the whole point.
            assert_eq!(texel_axis((dim - 1) as f32 + 0.5, dim), (dim - 1) as u32);
        }
        assert_eq!(texel_axis(0.0, 1), 0, "one-texel panel reads texel zero");
        assert_eq!(texel_axis(-5.0, 8), 0, "negative coordinate clamps up");
    }
}
