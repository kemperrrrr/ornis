//! Bilateral AO blur for the stable Rust to WGSL path.
//!
//! Four taps on a cross over the half-res raw AO factor, weighted by the
//! change in metric eye distance (the `bilateral_weight` curve), so occlusion
//! never bleeds over silhouette edges. The centre tap is weighted `1`.
//!
//! The raw factor is sampled with the binding's filtering sampler (level 0);
//! the depth lookup and the weights come from `textureLoad` and the shared
//! world reconstruction, exactly like the raw AO kernels.

#![no_std]
#![allow(clippy::all)]

use ornis_gpu_shader::DepthTexture2d;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::Sampler;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::TYPE_TAG_VEC4;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::exp;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::fragment_uv;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::length3;

/// Host camera uniform: thirteen `vec4` columns.
pub struct Camera;

impl GpuStruct for Camera {
    /// The camera uniform is struct 0.
    const ID: u32 = 0;
    /// `view_proj`, `inv_view_proj`, `camera_pos`, `previous_view_proj`.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_VEC4; 13];
}

/// Host `AoParams`: radius, intensity, bias, blur sigma.
pub struct AoParams;

impl GpuStruct for AoParams {
    /// The AO uniform is struct 1.
    const ID: u32 = 1;
    /// Four `f32` fields.
    const FIELD_TAGS: &'static [u8] = &[TYPE_TAG_F32; 4];
}

const EPS: f32 = 1.0e-6;
/// Offset of each cross tap in texels of the raw AO target.
const TAP_OFFSET: f32 = 1.5;

type CameraUniform = Uniform<Camera, 0, 4>;
type Depth = DepthTexture2d<0, 2>;

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "inv_view_proj is Camera fields 4..7; the indices are the host layout"
fn inv_view_lane<const LANE: usize>(camera: CameraUniform, x: f32, y: f32, z: f32) -> f32 {
    camera.vec4_lane::<4, LANE>() * x
        + camera.vec4_lane::<5, LANE>() * y
        + camera.vec4_lane::<6, LANE>() * z
        + camera.vec4_lane::<7, LANE>()
}

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "NDC remap of a depth sample, matching reconstruct_world_pos"
fn world_lane<const LANE: usize>(camera: CameraUniform, uvx: f32, uvy: f32, depth: f32) -> f32 {
    let nx = uvx * 2.0 - 1.0;
    let ny = 1.0 - uvy * 2.0;
    let value = inv_view_lane::<LANE>(camera, nx, ny, depth);
    let w = f32::max(inv_view_lane::<3>(camera, nx, ny, depth), EPS);
    value / w
}

/// Eye distance of the depth sample at `(uvx, uvy)`.
#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "camera_pos is Camera field 8; the indices are the host layout"
fn eye_distance(camera: CameraUniform, uvx: f32, uvy: f32, depth: f32) -> f32 {
    let wx = world_lane::<0>(camera, uvx, uvy, depth);
    let wy = world_lane::<1>(camera, uvx, uvy, depth);
    let wz = world_lane::<2>(camera, uvx, uvy, depth);
    let eye_x = camera.vec4_lane::<8, 0>();
    let eye_y = camera.vec4_lane::<8, 1>();
    let eye_z = camera.vec4_lane::<8, 2>();
    length3(eye_x - wx, eye_y - wy, eye_z - wz)
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

/// Weight of the tap at `uv + (dx, dy) * texel` for the bilateral curve.
#[inline(always)]
// qual:allow(srp, max_parameters=12) reason: "one tap; the arguments are the entry's live values, and a struct would allocate in the shader"
#[allow(clippy::too_many_arguments)]
fn tap_weight(
    depth_tex: Depth,
    camera: CameraUniform,
    uvx: f32,
    uvy: f32,
    dx: f32,
    dy: f32,
    center_z: f32,
    sigma: f32,
    dims_x: f32,
    dims_y: f32,
    texel_x: f32,
    texel_y: f32,
) -> f32 {
    let sample_uvx = uvx + dx * texel_x;
    let sample_uvy = uvy + dy * texel_y;
    let clamped_x = clamp(sample_uvx, 0.0, 1.0);
    let clamped_y = clamp(sample_uvy, 0.0, 1.0);
    let sx = min_u32((clamped_x * dims_x) as u32, (dims_x - 1.0) as u32);
    let sy = min_u32((clamped_y * dims_y) as u32, (dims_y - 1.0) as u32);
    let sample_z = eye_distance(camera, clamped_x, clamped_y, depth_tex.load(sx, sy));
    let delta = (center_z - sample_z) / sigma;
    exp(-delta * delta)
}

/// Fragment entry. The vertex quad passes UV at `@location(0)`.
#[gpu_entry(fragment)]
// qual:allow(complexity, magic_numbers) reason: "four unrolled cross taps stay visible side by side; a loop would need an index-selected offset table the translator cannot index"
pub fn fs_main(
    #[gpu(texture_2d, group = 0, binding = 0)] ao_raw: Texture2d,
    #[gpu(sampler, group = 0, binding = 1)] blur_sampler: Sampler,
    #[gpu(depth_texture_2d, group = 0, binding = 2)] depth_tex: DepthTexture2d,
    #[gpu(uniform, group = 0, binding = 3)] ao: Uniform<AoParams>,
    #[gpu(uniform, group = 0, binding = 4)] camera: Uniform<Camera>,
) {
    let uvx = fragment_uv::<0>();
    let uvy = fragment_uv::<1>();
    let raw = ao_raw.sample_r(blur_sampler, uvx, uvy);
    let dims_x = depth_tex.width() as f32;
    let dims_y = depth_tex.height() as f32;
    let cx = min_u32((uvx * dims_x) as u32, (dims_x - 1.0) as u32);
    let cy = min_u32((uvy * dims_y) as u32, (dims_y - 1.0) as u32);
    let center_z = eye_distance(camera, uvx, uvy, depth_tex.load(cx, cy));
    let texel_x = 1.0 / (ao_raw.width() as f32);
    let texel_y = 1.0 / (ao_raw.height() as f32);
    let sigma = f32::max(ao.field_f32::<3>(), EPS);
    let mut acc = raw;
    let mut weight = 1.0f32;
    let w0 = tap_weight(
        depth_tex, camera, uvx, uvy, TAP_OFFSET, 0.0, center_z, sigma, dims_x, dims_y, texel_x,
        texel_y,
    );
    acc += ao_raw.sample_r(blur_sampler, uvx + TAP_OFFSET * texel_x, uvy) * w0;
    weight += w0;
    let w1 = tap_weight(
        depth_tex,
        camera,
        uvx,
        uvy,
        -TAP_OFFSET,
        0.0,
        center_z,
        sigma,
        dims_x,
        dims_y,
        texel_x,
        texel_y,
    );
    acc += ao_raw.sample_r(blur_sampler, uvx - TAP_OFFSET * texel_x, uvy) * w1;
    weight += w1;
    let w2 = tap_weight(
        depth_tex, camera, uvx, uvy, 0.0, TAP_OFFSET, center_z, sigma, dims_x, dims_y, texel_x,
        texel_y,
    );
    acc += ao_raw.sample_r(blur_sampler, uvx, uvy + TAP_OFFSET * texel_y) * w2;
    weight += w2;
    let w3 = tap_weight(
        depth_tex,
        camera,
        uvx,
        uvy,
        0.0,
        -TAP_OFFSET,
        center_z,
        sigma,
        dims_x,
        dims_y,
        texel_x,
        texel_y,
    );
    acc += ao_raw.sample_r(blur_sampler, uvx, uvy - TAP_OFFSET * texel_y) * w3;
    weight += w3;
    let visibility = acc / weight;
    fragment_store(visibility, visibility, visibility, 1.0);
}
