//! Stable Rust representations for values exchanged with GPU buffers.
//!
//! These wrappers deliberately avoid Rust layout types whose representation is
//! not suitable for `bytemuck` or WGSL. They are ordinary Rust values at the
//! API boundary, but their storage representation is explicit and portable.

/// A boolean stored as a four-byte unsigned integer for GPU buffers.
///
/// Boolean values in host-shareable WGSL buffers are represented as a four-byte
/// `u32` (`0` or `1`). Rust's native `bool` is intentionally not accepted
/// by [`ornis_macros::WgslStruct`], because its memory representation is not a
/// GPU buffer contract. Use `GpuBool::from(true)` or [`GpuBool::new`]
/// whenever a buffer field is logically boolean.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBool(u32);

impl GpuBool {
    /// The canonical encoded value for `true` or `false`.
    pub const fn new(value: bool) -> Self {
        Self(value as u32)
    }

    /// Construct from an already encoded value, normalizing non-zero values
    /// to `true` so values read from a shader have one canonical form.
    pub const fn from_u32(value: u32) -> Self {
        Self((value != 0) as u32)
    }

    /// Return the logical boolean value.
    pub const fn get(self) -> bool {
        self.0 != 0
    }

    /// Return the exact four-byte value uploaded to the GPU.
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

impl From<bool> for GpuBool {
    fn from(value: bool) -> Self {
        Self::new(value)
    }
}

impl From<GpuBool> for bool {
    fn from(value: GpuBool) -> Self {
        value.get()
    }
}
