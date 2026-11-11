//! Shader-stage builtins.
//!
//! The invocation id, the local index, and the fragment UV are not resources
//! a bind group provides. The axis or lane is a const generic so the
//! translator sees a literal, which is the only form it lowers.

/// Lane `AXIS` (`0..3`) of `@builtin(global_invocation_id)`.
#[inline(always)]
#[must_use]
pub fn global_invocation_lane<const AXIS: u32>() -> u32 {
    const { assert!(AXIS < 3, "global_invocation_id has three lanes") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::global_invocation_lane(AXIS)
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::host_unavailable()
}

/// `@builtin(local_invocation_index)` of the compute entry.
#[inline(always)]
#[must_use]
pub fn local_invocation_index() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::local_invocation_index()
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::host_unavailable()
}

/// Lane `LANE` (`0..2`) of the fragment's `@location(0)` UV.
#[inline(always)]
#[must_use]
pub fn fragment_uv<const LANE: u32>() -> f32 {
    const { assert!(LANE < 2, "fragment UV is a vec2") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::fragment_uv_lane(LANE)
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::host_unavailable()
}

/// Write the fragment's `@location(0)` color.
#[inline(always)]
pub fn fragment_store(r: f32, g: f32, b: f32, a: f32) {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::fragment_store(r, g, b, a);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (r, g, b, a);
        crate::host_unavailable()
    }
}
