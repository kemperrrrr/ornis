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

/// `(param i32 i32 i32) (result f32)` — storage `array<f32>`.
pub const STORAGE_READ_F32: &str = "storage_read_f32";

/// `(param i32 i32 i32 f32)` — store one `f32` element.
pub const STORAGE_WRITE_F32: &str = "storage_write_f32";

/// `(param i32 i32 i32) (result i32)` — storage `array<u32>`.
///
/// The wasm result is `i32` bits. The lowerer treats them as `u32`.
pub const STORAGE_READ_U32: &str = "storage_read_u32";

/// `(param i32 i32 i32 i32)` — store one `u32` element.
pub const STORAGE_WRITE_U32: &str = "storage_write_u32";

/// `(param i32 i32 i32) (result i32)` — storage `array<i32>`.
pub const STORAGE_READ_I32: &str = "storage_read_i32";

/// `(param i32 i32 i32 i32)` — store one `i32` element.
pub const STORAGE_WRITE_I32: &str = "storage_write_i32";

/// `(param i32 i32 i32) (result f32 f32)` — storage `array<vec2<f32>>`.
pub const STORAGE_READ_VEC2: &str = "storage_read_vec2";

/// `(param i32 i32 i32 f32 f32)` — store one `vec2<f32>` element.
pub const STORAGE_WRITE_VEC2: &str = "storage_write_vec2";

/// `(param i32 i32 i32) (result f32 f32 f32)` — storage `array<vec3<f32>>`.
pub const STORAGE_READ_VEC3: &str = "storage_read_vec3";

/// `(param i32 i32 i32 f32 f32 f32)` — store one `vec3<f32>` element.
pub const STORAGE_WRITE_VEC3: &str = "storage_write_vec3";

/// `(param i32 i32 i32 i32) (result …)` — storage array of a struct.
///
/// The fourth immediate is the struct id from the `ornis_gpu.types` custom
/// section. Result lanes are the struct's fields, flattened in order.
pub const STORAGE_READ_STRUCT: &str = "storage_read_struct";

/// `(param i32 i32 i32 i32 …)` — store one struct element.
///
/// The fourth immediate is the struct id. The lanes that follow are the
/// flattened fields, in declaration order.
pub const STORAGE_WRITE_STRUCT: &str = "storage_write_struct";

/// Custom section that lists struct fields for [`STORAGE_READ_STRUCT`].
pub const TYPES_SECTION: &str = "ornis_gpu.types";

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
        RAW_READ
            | RAW_READ_F32
            | RAW_WRITE
            | STORAGE_READ
            | STORAGE_WRITE
            | STORAGE_READ_F32
            | STORAGE_WRITE_F32
            | STORAGE_READ_U32
            | STORAGE_WRITE_U32
            | STORAGE_READ_I32
            | STORAGE_WRITE_I32
            | STORAGE_READ_VEC2
            | STORAGE_WRITE_VEC2
            | STORAGE_READ_VEC3
            | STORAGE_WRITE_VEC3
            | STORAGE_READ_STRUCT
            | STORAGE_WRITE_STRUCT
            | TEXTURE_SAMPLE_2D
    )
}
