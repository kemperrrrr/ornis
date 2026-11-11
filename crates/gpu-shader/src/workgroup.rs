//! Workgroup memory.
//!
//! The translator gives a kernel one `array<u32, 256>` and one
//! `array<atomic<u32>, 256>`. They are not bind-group bindings, so they are
//! not `#[gpu]` parameters: a kernel calls these functions, and the import
//! names the array.

/// One word of the non-atomic workgroup array.
pub struct WorkgroupU32;

impl WorkgroupU32 {
    /// Load word `index`.
    #[inline(always)]
    pub fn load(index: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::workgroup_load(index)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = index;
            crate::host_unavailable()
        }
    }

    /// Store `value` at `index`.
    #[inline(always)]
    pub fn store(index: u32, value: u32) {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::workgroup_store(index, value);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (index, value);
            crate::host_unavailable()
        }
    }
}

/// One word of the workgroup atomic array.
pub struct WorkgroupAtomicU32;

impl WorkgroupAtomicU32 {
    /// Load word `index`.
    #[inline(always)]
    pub fn load(index: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::workgroup_atomic_load(index)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = index;
            crate::host_unavailable()
        }
    }

    /// Store `value` at `index`.
    #[inline(always)]
    pub fn store(index: u32, value: u32) {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::workgroup_atomic_store(index, value);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (index, value);
            crate::host_unavailable()
        }
    }

    /// `atomicAdd` at `index`. Returns the previous word.
    #[inline(always)]
    pub fn add(index: u32, value: u32) -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            crate::abi::workgroup_atomic_add(index, value)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = (index, value);
            crate::host_unavailable()
        }
    }
}

/// Wait until every invocation in the workgroup reaches this call.
#[inline(always)]
pub fn workgroup_barrier() {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::workgroup_barrier();
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::host_unavailable()
}
