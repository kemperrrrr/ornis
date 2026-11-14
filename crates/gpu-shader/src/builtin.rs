//! Shader-stage builtins and stage interface values.
//!
//! The invocation id, the local index, and the fragment UV are not resources
//! a bind group provides. The axis or lane is a const generic so the
//! translator sees a literal, which is the only form it lowers.
//!
//! The same rule carries the raster stages: a vertex entry reads
//! `@builtin(vertex_index)`, `@builtin(instance_index)`, and its `@location`
//! inputs from functions below, and writes `@builtin(position)` plus located
//! outputs through the `vertex_store_*` writers. A fragment entry reads any
//! `@location` with [`fragment_input_lane`] and writes any target with
//! [`fragment_store_at`]. Locations and lanes are const generics for the same
//! reason the axis is: the translator only lowers immediates, and the declared
//! `vecN` width of each interface value follows from the lanes actually used.

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

/// `@builtin(vertex_index)` of the vertex entry.
#[inline(always)]
#[must_use]
pub fn vertex_index() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::vertex_index()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::host_unavailable()
    }
}

/// `@builtin(instance_index)` of the vertex entry.
#[inline(always)]
#[must_use]
pub fn instance_index() -> u32 {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::instance_index()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::host_unavailable()
    }
}

/// Lane `LANE` of the `LANES`-wide vertex input at `LOC`.
///
/// The width is a const generic and not inferred from the lanes the body
/// reads: the vertex and fragment stages are separate compilations of separate
/// crates, and the two declarations must name the same `vecN<f32>`.
#[inline(always)]
#[must_use]
pub fn vertex_input_lane<const LOC: u32, const LANES: u32, const LANE: u32>() -> f32 {
    const { assert!(LOC < 16, "a vertex input is one of 16 locations") };
    const { assert!(LANES >= 1 && LANES <= 4, "an interface value has 1-4 lanes") };
    const { assert!(LANE < LANES, "the lane is inside the declared width") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::vertex_input_lane(LOC, LANES, LANE)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::host_unavailable()
    }
}

/// Clip-space `@builtin(position)` of the vertex output.
#[inline(always)]
pub fn vertex_position(x: f32, y: f32, z: f32, w: f32) {
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::vertex_position(x, y, z, w);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (x, y, z, w);
        crate::host_unavailable()
    }
}

/// Write the `vec2<f32>` vertex output at `LOC`.
#[inline(always)]
pub fn vertex_store_2<const LOC: u32>(x: f32, y: f32) {
    const { assert!(LOC < 16, "a vertex output is one of 16 locations") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::vertex_store_2(LOC, x, y);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (x, y);
        crate::host_unavailable()
    }
}

/// Write the `vec4<f32>` vertex output at `LOC`.
#[inline(always)]
pub fn vertex_store_4<const LOC: u32>(x: f32, y: f32, z: f32, w: f32) {
    const { assert!(LOC < 16, "a vertex output is one of 16 locations") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::vertex_store_4(LOC, x, y, z, w);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (x, y, z, w);
        crate::host_unavailable()
    }
}

/// Lane `LANE` of the `LANES`-wide fragment input at `LOC`.
///
/// The vertex stage writes the interpolant at the same location with the same
/// width; a mismatch is a pipeline error, so the width is declared here too.
#[inline(always)]
#[must_use]
pub fn fragment_input_lane<const LOC: u32, const LANES: u32, const LANE: u32>() -> f32 {
    const { assert!(LOC < 16, "a fragment input is one of 16 locations") };
    const { assert!(LANES >= 1 && LANES <= 4, "an interface value has 1-4 lanes") };
    const { assert!(LANE < LANES, "the lane is inside the declared width") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::fragment_input_lane(LOC, LANES, LANE)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::host_unavailable()
    }
}

/// Write the fragment output at `LOC`, one lane per color attachment.
///
/// A single `@location(0)` target keeps the bare `vec4<f32>` entry result;
/// two or more targets become an output struct with one member each.
#[inline(always)]
pub fn fragment_store_at<const LOC: u32>(r: f32, g: f32, b: f32, a: f32) {
    const { assert!(LOC < 16, "a fragment output is one of 16 locations") };
    #[cfg(target_arch = "wasm32")]
    {
        crate::abi::fragment_store_location(LOC, r, g, b, a);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (r, g, b, a);
        crate::host_unavailable()
    }
}
