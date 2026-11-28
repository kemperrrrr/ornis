//! Fullscreen-quad vertex helper for the translated post passes.
//!
//! Every post pass draws one quad from `@builtin(vertex_index)` and nothing
//! else, so the corner table and the two `@location(0)` writes live here once.
//! The crate also exports the vertex stage itself, [`vs_main_quad`], so a pass
//! points its pipeline at that entry instead of assembling a vertex stage of
//! its own. It is not named plain `vs_main`: a pass keeps its own
//! `#[gpu_entry(vertex)]` under that name (the metadata global is named after
//! the entry, so two `vs_main`s in one crate's build would collide), and a pass
//! that depends on this crate would then carry both.

#![no_std]
#![warn(missing_docs)]

use ornis_gpu_shader::vertex_index;
use ornis_gpu_shader::vertex_position;
use ornis_gpu_shader::vertex_store_2;

/// Bit of `vertex_index` that selects the right column of the quad.
const COLUMN_BIT: u32 = 1;

/// Standard quad vertex stage: one triangle strip, corners from
/// `vertex_index`.
///
/// Named for the shape rather than for the stage, so a pass that keeps its own
/// `vs_main` can depend on this crate without two entries colliding. The name
/// still starts with `vs_main` — `xtask build-shaders` reads the stage from
/// that prefix, and so does the translator.
// qual:api — wasm export declared by `#[gpu_entry]`; rustqual reads the source
// attribute, not the macro's `#[unsafe(export_name = "vs_main_quad")]`.
#[ornis_gpu_shader::gpu_entry(vertex)]
pub fn vs_main_quad() {
    standard_quad();
}
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
