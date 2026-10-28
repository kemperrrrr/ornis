//! Import module and function names shared with `ornis-wasm2gpu`.
//!
//! The toy translator treats these as GPU opcodes. `raw_read` returns four
//! `f32` lanes (a `vec4` at a byte offset). `texture_sample_2d` is a call,
//! never an `i32.load` from linear memory. Group and binding are separate
//! immediate operands, not bits packed into the offset.

/// Wasm import module (`(import "ornis_gpu" ...)`).
pub const MODULE: &str = "ornis_gpu";

/// `(param i32 i32 i32) (result f32 f32 f32 f32)` — group, binding, byte offset.
pub const RAW_READ: &str = "raw_read";

/// Scalar sibling of [`RAW_READ`]: `(param i32 i32 i32) (result f32)`.
pub const RAW_READ_F32: &str = "raw_read_f32";

/// `(param i32 i32 i32 f32 f32 f32 f32)` — group, binding, offset, four lanes.
pub const RAW_WRITE: &str = "raw_write";

/// `(param i32 i32 i32) (result f32 f32 f32 f32)` — group, binding, element index.
///
/// Group and binding are constant immediates: that pair is the binding's
/// provenance. The index may be a runtime value and lowers to an access
/// chain on that one storage global. A runtime group or binding is rejected;
/// bindless descriptor indexing is not this opcode.
pub const STORAGE_READ: &str = "storage_read";

/// `(param i32 i32 i32 f32 f32 f32 f32)` — group, binding, element index, xyzw.
///
/// Same provenance rule as [`STORAGE_READ`]. The store lands on
/// `array<vec4<f32>>[index]`, not on a bit-packed pointer.
pub const STORAGE_WRITE: &str = "storage_write";

/// `(param i32 i32 i32 i32 f32 f32) (result f32 f32 f32 f32)`.
///
/// Operands are texture group, texture binding, sampler group, sampler
/// binding, then UV. The binding pair is constant provenance; UV is data.
pub const TEXTURE_SAMPLE_2D: &str = "texture_sample_2d";

/// Vertex entry the toy translator expects.
pub const VS_MAIN: &str = "vs_main";

/// Fragment entry the toy translator expects.
pub const FS_MAIN: &str = "fs_main";

/// Whether `name` is a toy GPU import the offline translator may see.
///
/// Group and binding stay separate operands; this predicate only names the
/// opcode. Unknown imports are rejected rather than treated as memory loads.
pub fn is_toy_import(name: &str) -> bool {
    matches!(
        name,
        RAW_READ | RAW_READ_F32 | RAW_WRITE | STORAGE_READ | STORAGE_WRITE | TEXTURE_SAMPLE_2D
    )
}
