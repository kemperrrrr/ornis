//! Element types a storage binding can hold, and the import each one lowers to.
//!
//! A binding's ABI is the element's, not the buffer's: `array<f32>` reads one
//! word, `array<atomic<u32>>` reserves a slot, `array<vec4<u32>>` counts in
//! chunks. Splitting them into per-element impls keeps the facade in
//! [`crate::Storage`] free of that dispatch and means a new element type is two
//! trait impls plus the import it needs.
//!
//! Host builds compile these too (a kernel crate is checked with
//! `cargo check`), and every method diverges there: a binding has no bytes to
//! touch outside a dispatched shader.

/// Element whose binding has a length the shader can read.
pub trait StorageLen {
    /// Element count of the binding at `(group, binding)`.
    fn len(group: u32, binding: u32) -> u32;
}

/// Element a storage binding can be read from.
pub trait StorageLoad {
    /// The Rust type one element reads as.
    type Value;

    /// Read element `index`.
    fn load(group: u32, binding: u32, index: u32) -> Self::Value;
}

/// Element a read-write storage binding can be written to.
pub trait StorageStore {
    /// The Rust type one element writes as.
    type Value;

    /// Write `value` at `index`.
    fn store(group: u32, binding: u32, index: u32, value: Self::Value);
}

/// Element marker of an `array<atomic<u32>>` storage binding.
///
/// The marker picks `atomicAdd`; the element itself is a plain `u32`, and the
/// host must bind the same buffer as an atomic array.
pub enum AtomicU32 {}

/// One `vec4<u32>` chunk of a storage binding.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Vec4U32 {
    /// X lane.
    pub x: u32,
    /// Y lane.
    pub y: u32,
    /// Z lane.
    pub z: u32,
    /// W lane.
    pub w: u32,
}

impl StorageLen for f32 {
    #[inline(always)]
    fn len(_group: u32, _binding: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::len_f32(_group, _binding)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl StorageLoad for f32 {
    type Value = f32;

    #[inline(always)]
    fn load(_group: u32, _binding: u32, _index: u32) -> f32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::read_f32(_group, _binding, _index)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl StorageStore for f32 {
    type Value = f32;

    #[inline(always)]
    fn store(_group: u32, _binding: u32, _index: u32, _value: f32) {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::write_f32(_group, _binding, _index, _value);
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl StorageLen for u32 {
    #[inline(always)]
    fn len(_group: u32, _binding: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::len_u32(_group, _binding)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl StorageLoad for u32 {
    type Value = u32;

    #[inline(always)]
    fn load(_group: u32, _binding: u32, _index: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::read_u32(_group, _binding, _index)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl StorageStore for u32 {
    type Value = u32;

    #[inline(always)]
    fn store(_group: u32, _binding: u32, _index: u32, _value: u32) {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::write_u32(_group, _binding, _index, _value);
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl StorageLen for Vec4U32 {
    #[inline(always)]
    fn len(_group: u32, _binding: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::len_vec4_u32(_group, _binding)
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::host_unavailable()
    }
}

impl AtomicU32 {
    /// Reserve a slot: `atomicAdd(&binding[index], value)`.
    ///
    /// Returns the value the word held before the add, which compaction kernels
    /// use as their dense output slot.
    #[inline(always)]
    pub fn atomic_add(group: u32, binding: u32, index: u32, value: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::atomic_add_u32(group, binding, index, value)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (group, binding, index, value);
            crate::host_unavailable()
        }
    }
}
