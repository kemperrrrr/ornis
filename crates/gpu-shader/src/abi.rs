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
    #[link_name = "uniform_field_f32"]
    fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32;
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
    #[link_name = "gpu_pow"]
    fn gpu_pow(base: f32, exponent: f32) -> f32;
    #[link_name = "gpu_atan2"]
    fn gpu_atan2(y: f32, x: f32) -> f32;
    #[link_name = "gpu_smoothstep"]
    fn gpu_smoothstep(low: f32, high: f32, x: f32) -> f32;
    #[link_name = "gpu_sqrt"]
    fn gpu_sqrt(value: f32) -> f32;
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
    #[link_name = "gpu_dot2_xy"]
    fn gpu_dot2_xy(ax: f32, ay: f32, bx: f32, by: f32) -> f32;
    #[link_name = "gpu_dot3_xyz"]
    fn gpu_dot3_xyz(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32;
    #[link_name = "gpu_length3_xyz"]
    fn gpu_length3_xyz(x: f32, y: f32, z: f32) -> f32;
    #[link_name = "gpu_log2"]
    fn gpu_log2(value: f32) -> f32;
    #[link_name = "gpu_exp2"]
    fn gpu_exp2(value: f32) -> f32;
    #[link_name = "gpu_abs"]
    fn gpu_abs(value: f32) -> f32;
    #[link_name = "gpu_sign"]
    fn gpu_sign(value: f32) -> f32;
    #[link_name = "gpu_clamp"]
    fn gpu_clamp(value: f32, low: f32, high: f32) -> f32;
    #[link_name = "gpu_mix"]
    fn gpu_mix(start: f32, end: f32, factor: f32) -> f32;
    #[link_name = "gpu_fract"]
    fn gpu_fract(value: f32) -> f32;
    #[link_name = "gpu_cos"]
    fn gpu_cos(value: f32) -> f32;
    #[link_name = "gpu_sin"]
    fn gpu_sin(value: f32) -> f32;
    #[link_name = "gpu_acos"]
    fn gpu_acos(value: f32) -> f32;
    #[link_name = "gpu_cross3_lane"]
    fn gpu_cross3_lane(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, lane: u32) -> f32;
    #[link_name = "gpu_normalize3_lane"]
    fn gpu_normalize3_lane(x: f32, y: f32, z: f32, lane: u32) -> f32;
    #[link_name = "texture_load_2d_f32"]
    fn texture_load_2d_f32(group: u32, binding: u32, x: u32, y: u32) -> f32;
    #[link_name = "texture_load_2d_u32"]
    fn texture_load_2d_u32(group: u32, binding: u32, x: u32, y: u32) -> u32;
    #[link_name = "texture_load_2d_f32x4"]
    fn texture_load_2d_f32x4(group: u32, binding: u32, x: u32, y: u32, lane: u32) -> f32;
    #[link_name = "texture_sample_level_2d_f32x4"]
    fn texture_sample_level_2d_f32x4(
        group: u32,
        binding: u32,
        sampler_group: u32,
        sampler_binding: u32,
        x: f32,
        y: f32,
        lane: u32,
    ) -> f32;
    #[link_name = "texture_width_2d"]
    fn texture_width_2d(group: u32, binding: u32) -> i32;
    #[link_name = "texture_height_2d"]
    fn texture_height_2d(group: u32, binding: u32) -> i32;
    #[link_name = "texture_width_2d_f32x4"]
    fn texture_width_2d_f32x4(group: u32, binding: u32) -> i32;
    #[link_name = "texture_height_2d_f32x4"]
    fn texture_height_2d_f32x4(group: u32, binding: u32) -> i32;
    #[link_name = "workgroup_load_u32"]
    fn workgroup_load_u32(index: u32) -> u32;
    #[link_name = "workgroup_store_u32"]
    fn workgroup_store_u32(index: u32, value: u32);
    #[link_name = "workgroup_atomic_load_u32"]
    fn workgroup_atomic_load_u32(index: u32) -> u32;
    #[link_name = "workgroup_atomic_store_u32"]
    fn workgroup_atomic_store_u32(index: u32, value: u32);
    #[link_name = "workgroup_atomic_add_u32"]
    fn workgroup_atomic_add_u32(index: u32, value: u32) -> u32;
    #[link_name = "workgroup_barrier"]
    fn import_workgroup_barrier();
    #[link_name = "global_invocation_lane"]
    fn import_global_invocation_lane(axis: u32) -> u32;
    #[link_name = "local_invocation_index"]
    fn import_local_invocation_index() -> u32;
    #[link_name = "fragment_uv_lane"]
    fn import_fragment_uv_lane(lane: u32) -> f32;
    #[link_name = "fragment_store_f32x4"]
    fn fragment_store_f32x4(r: f32, g: f32, b: f32, a: f32);
    #[link_name = "vertex_index"]
    fn import_vertex_index() -> u32;
    #[link_name = "instance_index"]
    fn import_instance_index() -> u32;
    #[link_name = "vertex_input_lane"]
    fn import_vertex_input_lane(location: u32, lanes: u32, lane: u32) -> f32;
    #[link_name = "vertex_position"]
    fn import_vertex_position(x: f32, y: f32, z: f32, w: f32);
    #[link_name = "vertex_store_f32x2"]
    fn import_vertex_store_f32x2(location: u32, x: f32, y: f32);
    #[link_name = "vertex_store_f32x4"]
    fn import_vertex_store_f32x4(location: u32, x: f32, y: f32, z: f32, w: f32);
    #[link_name = "fragment_input_lane"]
    fn import_fragment_input_lane(location: u32, lanes: u32, lane: u32) -> f32;
    #[link_name = "fragment_store_location"]
    fn import_fragment_store_location(location: u32, r: f32, g: f32, b: f32, a: f32);
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

/// `sqrt(value)` — one WGSL `sqrt`.
#[inline(always)]
pub(crate) fn sqrt(value: f32) -> f32 {
    call!(gpu_sqrt(value))
}

/// `smoothstep(low, high, x)` — one WGSL `smoothstep`.
#[inline(always)]
pub(crate) fn smoothstep(low: f32, high: f32, x: f32) -> f32 {
    call!(gpu_smoothstep(low, high, x))
}

/// `pow(base, exponent)` — one WGSL `pow`.
#[inline(always)]
pub(crate) fn pow(base: f32, exponent: f32) -> f32 {
    call!(gpu_pow(base, exponent))
}

/// `atan2(y, x)` — one WGSL `atan2`.
#[inline(always)]
pub(crate) fn atan2(y: f32, x: f32) -> f32 {
    call!(gpu_atan2(y, x))
}

/// `exp(value)` — one WGSL `exp`.
#[inline(always)]
pub(crate) fn exp(value: f32) -> f32 {
    call!(gpu_exp(value))
}

/// `f32` field `field` of the uniform struct `sid`.
#[inline(always)]
pub(crate) fn field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32 {
    call!(uniform_field_f32(group, binding, sid, field))
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

/// Dot product of two `vec2<f32>` values.
#[inline(always)]
pub(crate) fn dot2(ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    call!(gpu_dot2_xy(ax, ay, bx, by))
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

/// `log2(value)`.
#[inline(always)]
pub(crate) fn log2(value: f32) -> f32 {
    call!(gpu_log2(value))
}

/// `exp2(value)`.
#[inline(always)]
pub(crate) fn exp2(value: f32) -> f32 {
    call!(gpu_exp2(value))
}

/// Absolute value.
#[inline(always)]
pub(crate) fn abs(value: f32) -> f32 {
    call!(gpu_abs(value))
}

/// Sign of `value`: `-1`, `0`, or `1`.
#[inline(always)]
pub(crate) fn sign(value: f32) -> f32 {
    call!(gpu_sign(value))
}

/// Clamp `value` into `low..=high`.
#[inline(always)]
pub(crate) fn clamp(value: f32, low: f32, high: f32) -> f32 {
    call!(gpu_clamp(value, low, high))
}

/// Linear interpolation `start + (end - start) * factor`.
#[inline(always)]
pub(crate) fn mix(start: f32, end: f32, factor: f32) -> f32 {
    call!(gpu_mix(start, end, factor))
}

/// Fractional part of `value`.
#[inline(always)]
pub(crate) fn fract(value: f32) -> f32 {
    call!(gpu_fract(value))
}

/// Cosine.
#[inline(always)]
pub(crate) fn cos(value: f32) -> f32 {
    call!(gpu_cos(value))
}

/// Sine.
#[inline(always)]
pub(crate) fn sin(value: f32) -> f32 {
    call!(gpu_sin(value))
}

/// Arccosine.
#[inline(always)]
pub(crate) fn acos(value: f32) -> f32 {
    call!(gpu_acos(value))
}

/// Lane `lane` of `cross(a, b)`.
// qual:allow(srp, max_parameters=7) reason: "one operand per import argument: six vector lanes and the result lane; a Vec3 argument would reintroduce the struct the translator does not see"
#[inline(always)]
pub(crate) fn cross3_lane(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, lane: u32) -> f32 {
    call!(gpu_cross3_lane(ax, ay, az, bx, by, bz, lane))
}

/// Lane `lane` of `normalize(x, y, z)`.
#[inline(always)]
pub(crate) fn normalize3_lane(x: f32, y: f32, z: f32, lane: u32) -> f32 {
    call!(gpu_normalize3_lane(x, y, z, lane))
}

/// First lane of a `texture_2d<u32>` texel at mip 0.
#[inline(always)]
pub(crate) fn texture_load_u32(group: u32, binding: u32, x: u32, y: u32) -> u32 {
    call!(texture_load_2d_u32(group, binding, x, y))
}

/// Scalar load of a depth texture at mip 0.
#[inline(always)]
pub(crate) fn texture_load_depth(group: u32, binding: u32, x: u32, y: u32) -> f32 {
    call!(texture_load_2d_f32(group, binding, x, y))
}

/// One lane of a sampled `texture_2d<f32>` load at mip 0.
#[inline(always)]
pub(crate) fn texture_load_f32x4(group: u32, binding: u32, x: u32, y: u32, lane: u32) -> f32 {
    call!(texture_load_2d_f32x4(group, binding, x, y, lane))
}

/// One lane of `textureSampleLevel(tex, sampler, vec2(x, y), 0.0)`.
// qual:allow(srp, max_parameters=7) reason: "one operand per import argument: texture binding, sampler binding, uv, and the lane"
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn texture_sample_level_f32x4(
    group: u32,
    binding: u32,
    sampler_group: u32,
    sampler_binding: u32,
    x: f32,
    y: f32,
    lane: u32,
) -> f32 {
    call!(texture_sample_level_2d_f32x4(
        group,
        binding,
        sampler_group,
        sampler_binding,
        x,
        y,
        lane
    ))
}

/// Width of a depth texture.
#[inline(always)]
pub(crate) fn texture_width_depth(group: u32, binding: u32) -> i32 {
    call!(texture_width_2d(group, binding))
}

/// Height of a depth texture.
#[inline(always)]
pub(crate) fn texture_height_depth(group: u32, binding: u32) -> i32 {
    call!(texture_height_2d(group, binding))
}

/// Width of a sampled `texture_2d<f32>`.
#[inline(always)]
pub(crate) fn texture_width_f32x4(group: u32, binding: u32) -> i32 {
    call!(texture_width_2d_f32x4(group, binding))
}

/// Height of a sampled `texture_2d<f32>`.
#[inline(always)]
pub(crate) fn texture_height_f32x4(group: u32, binding: u32) -> i32 {
    call!(texture_height_2d_f32x4(group, binding))
}

/// Load one word of the non-atomic workgroup array.
#[inline(always)]
pub(crate) fn workgroup_load(index: u32) -> u32 {
    call!(workgroup_load_u32(index))
}

/// Store one word of the non-atomic workgroup array.
#[inline(always)]
pub(crate) fn workgroup_store(index: u32, value: u32) {
    call!(workgroup_store_u32(index, value));
}

/// Load one word of the workgroup atomic array.
#[inline(always)]
pub(crate) fn workgroup_atomic_load(index: u32) -> u32 {
    call!(workgroup_atomic_load_u32(index))
}

/// Store one word of the workgroup atomic array.
#[inline(always)]
pub(crate) fn workgroup_atomic_store(index: u32, value: u32) {
    call!(workgroup_atomic_store_u32(index, value));
}

/// `atomicAdd` into the workgroup atomic array. Returns the previous word.
#[inline(always)]
pub(crate) fn workgroup_atomic_add(index: u32, value: u32) -> u32 {
    call!(workgroup_atomic_add_u32(index, value))
}

/// Workgroup execution barrier.
#[inline(always)]
pub(crate) fn workgroup_barrier() {
    call!(import_workgroup_barrier());
}

/// Lane `axis` of `@builtin(global_invocation_id)`.
#[inline(always)]
pub(crate) fn global_invocation_lane(axis: u32) -> u32 {
    call!(import_global_invocation_lane(axis))
}

/// `@builtin(local_invocation_index)`.
#[inline(always)]
pub(crate) fn local_invocation_index() -> u32 {
    call!(import_local_invocation_index())
}

/// Lane of the fragment's `@location(0)` UV.
#[inline(always)]
pub(crate) fn fragment_uv_lane(lane: u32) -> f32 {
    call!(import_fragment_uv_lane(lane))
}

/// Write the fragment's `@location(0)` color.
#[inline(always)]
pub(crate) fn fragment_store(r: f32, g: f32, b: f32, a: f32) {
    call!(fragment_store_f32x4(r, g, b, a));
}

/// `@builtin(vertex_index)` of the vertex entry.
#[inline(always)]
pub(crate) fn vertex_index() -> u32 {
    call!(import_vertex_index())
}

/// `@builtin(instance_index)` of the vertex entry.
#[inline(always)]
pub(crate) fn instance_index() -> u32 {
    call!(import_instance_index())
}

/// Lane `lane` of the vertex `@location(location)` input of `lanes` lanes.
#[inline(always)]
pub(crate) fn vertex_input_lane(location: u32, lanes: u32, lane: u32) -> f32 {
    call!(import_vertex_input_lane(location, lanes, lane))
}

/// Write the clip-space `@builtin(position)` of the vertex entry.
#[inline(always)]
pub(crate) fn vertex_position(x: f32, y: f32, z: f32, w: f32) {
    call!(import_vertex_position(x, y, z, w));
}

/// Write a `vec2<f32>` vertex output at `location`.
#[inline(always)]
pub(crate) fn vertex_store_2(location: u32, x: f32, y: f32) {
    call!(import_vertex_store_f32x2(location, x, y));
}

/// Write a `vec4<f32>` vertex output at `location`.
#[inline(always)]
pub(crate) fn vertex_store_4(location: u32, x: f32, y: f32, z: f32, w: f32) {
    call!(import_vertex_store_f32x4(location, x, y, z, w));
}

/// Lane `lane` of the fragment `@location(location)` input of `lanes` lanes.
#[inline(always)]
pub(crate) fn fragment_input_lane(location: u32, lanes: u32, lane: u32) -> f32 {
    call!(import_fragment_input_lane(location, lanes, lane))
}

/// Write the fragment output at `location`: a `vec4<f32>` of RGBA lanes.
#[inline(always)]
pub(crate) fn fragment_store_location(location: u32, r: f32, g: f32, b: f32, a: f32) {
    call!(import_fragment_store_location(location, r, g, b, a));
}
