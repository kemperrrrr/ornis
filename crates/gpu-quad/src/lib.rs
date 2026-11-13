//! Fullscreen-quad vertex helper for the translated post passes.
//!
//! Every post pass draws one quad from `@builtin(vertex_index)` and nothing
//! else, so the corner table and the two `@location(0)` writes live here once.
//! A pass keeps its own `#[gpu_entry(vertex)]` — one entry per crate, because
//! the metadata global carries a fixed export name — and calls
//! [`standard_quad`] or [`ui_quad`] from it. Both are `#[inline(always)]`, so
//! the compiled entry is the same straight-line code it would be written by
//! hand, and the translator never sees a call.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::vertex_index;
use ornis_gpu_shader::vertex_position;
use ornis_gpu_shader::vertex_store_2;

/// Bit of `vertex_index` that selects the right column of the quad.
const COLUMN_BIT: u32 = 1;
/// Bit of `vertex_index` that selects the top row of the quad.
const ROW_BIT: u32 = 2;

/// Write the standard quad's corner and UV for this `vertex_index`.
///
/// The corners are arithmetic on the index bits, not a table: a table would
/// live in this crate's object file, and a pass that links it sees only an
/// external symbol, which the translator cannot read (it has no initializer in
/// the pass's IR).
#[inline(always)]
pub fn standard_quad() {
    quad(COLUMN_BIT, ROW_BIT);
}

/// Write the UI composite quad's corner and UV for this `vertex_index`.
///
/// The UI layer is drawn with this winding: left column first, then the right
/// one, rows interleaved.
#[inline(always)]
pub fn ui_quad() {
    quad(ROW_BIT, COLUMN_BIT);
}

/// Write the quad corner and UV, reading the column and row from the two index
/// bits the caller names.
#[inline(always)]
fn quad(column_bit: u32, row_bit: u32) {
    let index = vertex_index();
    let right = index & column_bit != 0;
    let top = index & row_bit != 0;
    corner(if right { 1.0 } else { -1.0 }, if top { 1.0 } else { -1.0 });
    uv(if right { 1.0 } else { 0.0 }, if top { 0.0 } else { 1.0 });
}

/// Write the clip-space corner of the fullscreen quad.
#[inline(always)]
fn corner(x: f32, y: f32) {
    vertex_position(x, y, 0.0, 1.0);
}

/// Write the `@location(0)` UV of the same corner.
#[inline(always)]
fn uv(u: f32, v: f32) {
    vertex_store_2::<0>(u, v);
}
