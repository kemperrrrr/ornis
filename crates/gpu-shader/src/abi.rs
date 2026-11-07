//! The `ornis_gpu` imports the SDK methods lower to.
//!
//! This is the SDK's only `unsafe` module, mirroring `ornis-gpu-abi`'s rule that
//! the import shims live in one place. Each wrapper takes the operands a typed
//! method already knows (group, binding, element index) and calls one declared
//! import; nothing here dereferences a guest pointer, and every call is a plain
//! `ornis_gpu` opcode the translator recognizes.
//!
//! Only `wasm32` builds carry the imports. On the host the same wrappers are
//! never called — the resource methods diverge through
//! [`crate::host_unavailable`] before reaching them — so the declarations are
//! `cfg`-ed out and the module is empty.

#![cfg(target_arch = "wasm32")]

// SAFETY: every declaration below is one `ornis_gpu` import, and the wrappers
// further down are the only callers. The imports neither dereference a guest
// pointer nor expose linear memory: they name a binding by its `(group,
// binding)` pair and an element by index, and the host validated both when it
// built the bind group.
#[link(wasm_import_module = "ornis_gpu")]
unsafe extern "C" {
    #[link_name = "storage_len_f32"]
    fn storage_len_f32(group: u32, binding: u32) -> u32;
    #[link_name = "storage_len_u32"]
    fn storage_len_u32(group: u32, binding: u32) -> u32;
    #[link_name = "storage_len_u32x4"]
    fn storage_len_u32x4(group: u32, binding: u32) -> u32;
    #[link_name = "storage_read_f32"]
    fn storage_read_f32(group: u32, binding: u32, index: u32) -> f32;
    #[link_name = "storage_write_f32"]
    fn storage_write_f32(group: u32, binding: u32, index: u32, value: f32);
    #[link_name = "storage_read_u32"]
    fn storage_read_u32(group: u32, binding: u32, index: u32) -> u32;
    #[link_name = "storage_write_u32"]
    fn storage_write_u32(group: u32, binding: u32, index: u32, value: u32);
    #[link_name = "storage_atomic_add_u32"]
    fn storage_atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32;
    #[link_name = "uniform_field_u32"]
    fn uniform_field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32;
    #[link_name = "uniform_nested_field_f32"]
    fn uniform_nested_field_f32(group: u32, binding: u32, sid: u32, field: u32, inner: u32) -> f32;
    #[link_name = "uniform_field_vec4_component_f32"]
    fn uniform_field_vec4_component_f32(
        group: u32,
        binding: u32,
        sid: u32,
        field: u32,
        component: u32,
    ) -> f32;
    #[link_name = "gpu_exp"]
    fn gpu_exp(value: f32) -> f32;
    #[link_name = "uniform_field_vec4_plane_dot"]
    fn uniform_field_vec4_plane_dot(
        group: u32,
        binding: u32,
        sid: u32,
        field: u32,
        x: f32,
        y: f32,
        z: f32,
    ) -> f32;
    #[link_name = "gpu_dot3_xyz"]
    fn gpu_dot3_xyz(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32;
    #[link_name = "gpu_length3_xyz"]
    fn gpu_length3_xyz(x: f32, y: f32, z: f32) -> f32;
}

/// Call one declared import.
///
/// The `unsafe` operation lives here, once, instead of in each wrapper: every
/// caller is a typed method whose operands came from a parameter attribute, and
/// the import contract is the same for all of them.
macro_rules! call {
    ($import:ident($($operand:expr),* $(,)?)) => {{
        // SAFETY: `$import` is one of the `ornis_gpu` imports declared above.
        // It names a binding by the `(group, binding)` its caller's type
        // carries and an element by index; it neither dereferences a guest
        // pointer nor exposes linear memory.
        unsafe { $import($($operand),*) }
    }};
}

/// Element count of `array<f32>` at `(group, binding)`.
#[inline(always)]
pub(crate) fn len_f32(group: u32, binding: u32) -> u32 {
    call!(storage_len_f32(group, binding))
}

/// Element count of `array<u32>` at `(group, binding)`.
#[inline(always)]
pub(crate) fn len_u32(group: u32, binding: u32) -> u32 {
    call!(storage_len_u32(group, binding))
}

/// Element count of `array<vec4<u32>>` at `(group, binding)`.
#[inline(always)]
pub(crate) fn len_vec4_u32(group: u32, binding: u32) -> u32 {
    call!(storage_len_u32x4(group, binding))
}

/// Read element `index` of `array<f32>`.
#[inline(always)]
pub(crate) fn read_f32(group: u32, binding: u32, index: u32) -> f32 {
    call!(storage_read_f32(group, binding, index))
}

/// Write element `index` of `array<f32>`.
#[inline(always)]
pub(crate) fn write_f32(group: u32, binding: u32, index: u32, value: f32) {
    call!(storage_write_f32(group, binding, index, value));
}

/// Read element `index` of `array<u32>`.
#[inline(always)]
pub(crate) fn read_u32(group: u32, binding: u32, index: u32) -> u32 {
    call!(storage_read_u32(group, binding, index))
}

/// Write element `index` of `array<u32>`.
#[inline(always)]
pub(crate) fn write_u32(group: u32, binding: u32, index: u32, value: u32) {
    call!(storage_write_u32(group, binding, index, value));
}

/// `atomicAdd(&binding[index], value)`, returning the previous value.
#[inline(always)]
pub(crate) fn atomic_add_u32(group: u32, binding: u32, index: u32, value: u32) -> u32 {
    call!(storage_atomic_add_u32(group, binding, index, value))
}

/// Lane `component` of `vec4` field `field`.
#[inline(always)]
pub(crate) fn vec4_component(
    group: u32,
    binding: u32,
    sid: u32,
    field: u32,
    component: u32,
) -> f32 {
    call!(uniform_field_vec4_component_f32(
        group, binding, sid, field, component
    ))
}

/// `exp(value)` — one WGSL `exp`.
#[inline(always)]
pub(crate) fn exp(value: f32) -> f32 {
    call!(gpu_exp(value))
}

/// `u32` field `field` of the uniform struct `sid`.
#[inline(always)]
pub(crate) fn field_u32(group: u32, binding: u32, sid: u32, field: u32) -> u32 {
    call!(uniform_field_u32(group, binding, sid, field))
}

/// `f32` field `inner` of the struct stored in field `field`.
// qual:allow(test_quality, untested) reason: "wasm import shim; the nested-struct test translates the call in LLVM IR, and the host build never executes it"
#[inline(always)]
pub(crate) fn nested_field_f32(group: u32, binding: u32, sid: u32, field: u32, inner: u32) -> f32 {
    call!(uniform_nested_field_f32(group, binding, sid, field, inner))
}

/// `dot((x, y, z, 1), plane)` for the `vec4` field `field`.
// qual:allow(srp, max_parameters=7) reason: "one operand per import argument: group, binding, struct id, field, and the three point lanes; bundling them into a struct would allocate in the shader"
#[inline(always)]
pub(crate) fn plane_distance(
    group: u32,
    binding: u32,
    sid: u32,
    field: u32,
    x: f32,
    y: f32,
    z: f32,
) -> f32 {
    call!(uniform_field_vec4_plane_dot(
        group, binding, sid, field, x, y, z
    ))
}

/// Dot product of two `vec3<f32>` values.
// qual:allow(srp, max_parameters=6) reason: "the import takes six scalar lanes, and the translator maps them positionally to vec3 components"
#[inline(always)]
pub(crate) fn dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32 {
    call!(gpu_dot3_xyz(ax, ay, az, bx, by, bz))
}

/// Length of a `vec3<f32>` value.
#[inline(always)]
pub(crate) fn length3(x: f32, y: f32, z: f32) -> f32 {
    call!(gpu_length3_xyz(x, y, z))
}
