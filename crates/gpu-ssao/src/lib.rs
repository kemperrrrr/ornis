//! SSAO spiral taps for the stable Rust to WGSL path.
//!
//! One fragment reads the fullscreen-quad UV, the g-buffer depth and
//! octahedral normal, and the same `Camera` / `AoParams` uniforms the
//! renderer uploads. Eight spiral taps on the normal hemisphere are
//! reprojected through `view_proj`; a tap occludes when the scene in front of
//! the probe point is nearer the eye than the probe itself (minus the bias).
//!
//! The body is identical at 1x and 4x: `--multisampled` only changes the
//! declared texture types and reads sample 0, so this crate never names
//! `texture_multisampled_2d`.

#![no_std]
#![allow(clippy::all)]

use ornis_gpu_shader::DepthTexture2d;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::TYPE_TAG_VEC4;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::cos;
use ornis_gpu_shader::cross_lane;
use ornis_gpu_shader::fract;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::fragment_uv;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::length3;
use ornis_gpu_shader::normalize_lane;
use ornis_gpu_shader::sin;

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

const CLEAR_DEPTH: f32 = 1.0;
const EPS: f32 = 1.0e-6;
/// One full turn in radians, matching the former hand-written literal.
const TURN: f32 = 6.28318530718;
/// Golden-angle spiral step of the eight taps.
const GOLDEN_ANGLE: f32 = 2.399963;
/// Taps per pixel.
const TAPS: u32 = 8;
/// Upward pull of the tap direction so the hemisphere never lies in the plane.
const HEMISPHERE_Z: f32 = 0.75;
/// `ao_interleaved_gradient` weights: 0.06711056, 0.00583715, 52.9829189.
const HASH_X: f32 = 0.06711056;
const HASH_Y: f32 = 0.00583715;
const HASH_Z: f32 = 52.9829189;

type CameraUniform = Uniform<Camera, 0, 2>;
type Depth = DepthTexture2d<0, 0>;

#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "view_proj is Camera fields 0..3; the indices are the host layout"
fn view_lane<const LANE: usize>(camera: CameraUniform, x: f32, y: f32, z: f32) -> f32 {
    camera.vec4_lane::<0, LANE>() * x
        + camera.vec4_lane::<1, LANE>() * y
        + camera.vec4_lane::<2, LANE>() * z
        + camera.vec4_lane::<3, LANE>()
}

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

/// Occlusion contributed by tap `index`.
///
/// Returns `0` when the probe lands off-screen, behind the near plane, over
/// background depth, or on a sample that sits behind the probe. The early
/// returns are the former `continue`s of the hand-written loop; inlining turns
/// them into branches of the caller.
#[inline(always)]
// qual:allow(complexity, magic_numbers) reason: "one tap; the arguments are the loop's live values, and splitting it would restate the reprojection"
// qual:allow(srp, max_parameters=21) reason: "one tap; the arguments are the loop's live values, and splitting it would restate the reprojection"
#[allow(clippy::too_many_arguments)]
fn tap(
    depth_tex: Depth,
    camera: CameraUniform,
    ox: f32,
    oy: f32,
    oz: f32,
    tx: f32,
    ty: f32,
    tz: f32,
    bx: f32,
    by: f32,
    bz: f32,
    nx: f32,
    ny: f32,
    nz: f32,
    index: u32,
    cos_a: f32,
    sin_a: f32,
    dims_x: f32,
    dims_y: f32,
    radius: f32,
    bias: f32,
) -> f32 {
    let fi = (index as f32) + 0.5;
    let tap_r = fi / (TAPS as f32);
    let tap_a = fi * GOLDEN_ANGLE;
    let raw_x = cos(tap_a) * tap_r;
    let raw_y = sin(tap_a) * tap_r;
    let tap_x = raw_x * cos_a - raw_y * sin_a;
    let tap_y = raw_x * sin_a + raw_y * cos_a;
    let dir_x = tx * tap_x + bx * tap_y + nx * HEMISPHERE_Z;
    let dir_y = ty * tap_x + by * tap_y + ny * HEMISPHERE_Z;
    let dir_z = tz * tap_x + bz * tap_y + nz * HEMISPHERE_Z;
    let dx = normalize_lane::<0>(dir_x, dir_y, dir_z);
    let dy = normalize_lane::<1>(dir_x, dir_y, dir_z);
    let dz = normalize_lane::<2>(dir_x, dir_y, dir_z);
    let probe_x = ox + dx * radius;
    let probe_y = oy + dy * radius;
    let probe_z = oz + dz * radius;
    let clip_w = f32::max(view_lane::<3>(camera, probe_x, probe_y, probe_z), EPS);
    let ndc_z = view_lane::<2>(camera, probe_x, probe_y, probe_z) / clip_w;
    if ndc_z > 1.0 {
        return 0.0;
    }
    let suv_x = view_lane::<0>(camera, probe_x, probe_y, probe_z) / clip_w * 0.5 + 0.5;
    let suv_y = 0.5 - view_lane::<1>(camera, probe_x, probe_y, probe_z) / clip_w * 0.5;
    if suv_x < 0.0 || suv_x > 1.0 || suv_y < 0.0 || suv_y > 1.0 {
        return 0.0;
    }
    let sx = (suv_x * dims_x) as u32;
    let sy = (suv_y * dims_y) as u32;
    let sample_depth = depth_tex.load(sx, sy);
    if sample_depth >= CLEAR_DEPTH {
        return 0.0;
    }
    let sample_x = world_lane::<0>(camera, suv_x, suv_y, sample_depth);
    let sample_y = world_lane::<1>(camera, suv_x, suv_y, sample_depth);
    let sample_z = world_lane::<2>(camera, suv_x, suv_y, sample_depth);
    let range = length3(probe_x - sample_x, probe_y - sample_y, probe_z - sample_z);
    if range < EPS {
        return 0.0;
    }
    let eye_x = camera.vec4_lane::<8, 0>();
    let eye_y = camera.vec4_lane::<8, 1>();
    let eye_z = camera.vec4_lane::<8, 2>();
    let eye_probe = length3(eye_x - probe_x, eye_y - probe_y, eye_z - probe_z);
    let eye_sample = length3(eye_x - sample_x, eye_y - sample_y, eye_z - sample_z);
    if eye_sample >= eye_probe - bias {
        return 0.0;
    }
    f32::max(0.0, 1.0 - range / radius)
}

/// Fragment entry. The vertex quad passes UV at `@location(0)`.
#[gpu_entry(fragment)]
pub fn fs_main(
    #[gpu(depth_texture_2d, group = 0, binding = 0)] depth_tex: DepthTexture2d,
    #[gpu(texture_2d, group = 0, binding = 1)] normal_tex: Texture2d,
    #[gpu(uniform, group = 0, binding = 2)] camera: Uniform<Camera>,
    #[gpu(uniform, group = 0, binding = 3)] ao: Uniform<AoParams>,
) {
    let uvx = fragment_uv::<0>();
    let uvy = fragment_uv::<1>();
    let dims_x = depth_tex.width() as f32;
    let dims_y = depth_tex.height() as f32;
    let xi = (uvx * dims_x) as u32;
    let yi = (uvy * dims_y) as u32;
    let depth = depth_tex.load(xi, yi);
    if depth >= CLEAR_DEPTH {
        fragment_store(1.0, 1.0, 1.0, 1.0);
        return;
    }
    let enc_x = normal_tex.load::<0>(xi, yi);
    let enc_y = normal_tex.load::<1>(xi, yi);
    let nz = 1.0 - abs(enc_x) - abs(enc_y);
    let fold = f32::max(-nz, 0.0);
    // Same fold as the renderer's `octahedral_decode`: the positive lanes
    // shift by `-t`, the negative ones by `+t`.
    let raw_nx = enc_x + if enc_x >= 0.0 { -fold } else { fold };
    let raw_ny = enc_y + if enc_y >= 0.0 { -fold } else { fold };
    let nnx = normalize_lane::<0>(raw_nx, raw_ny, nz);
    let nny = normalize_lane::<1>(raw_nx, raw_ny, nz);
    let nnz = normalize_lane::<2>(raw_nx, raw_ny, nz);
    let ox = world_lane::<0>(camera, uvx, uvy, depth);
    let oy = world_lane::<1>(camera, uvx, uvy, depth);
    let oz = world_lane::<2>(camera, uvx, uvy, depth);
    // `cross(n, up) + (0, 0, EPS)` keeps the tangent defined at the pole.
    let cross_x = cross_lane::<0>(nnx, nny, nnz, 0.0, 1.0, 0.0);
    let cross_y = cross_lane::<1>(nnx, nny, nnz, 0.0, 1.0, 0.0);
    let cross_z = cross_lane::<2>(nnx, nny, nnz, 0.0, 1.0, 0.0) + EPS;
    let tx = normalize_lane::<0>(cross_x, cross_y, cross_z);
    let ty = normalize_lane::<1>(cross_x, cross_y, cross_z);
    let tz = normalize_lane::<2>(cross_x, cross_y, cross_z);
    let bx = cross_lane::<0>(nnx, nny, nnz, tx, ty, tz);
    let by = cross_lane::<1>(nnx, nny, nnz, tx, ty, tz);
    let bz = cross_lane::<2>(nnx, nny, nnz, tx, ty, tz);
    let hash = fract(HASH_Z * fract((xi as f32) * HASH_X + (yi as f32) * HASH_Y));
    let angle = hash * TURN;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let radius = ao.field_f32::<0>();
    let intensity = ao.field_f32::<1>();
    let bias = ao.field_f32::<2>();
    let mut occluded = 0.0f32;
    let mut index = 0u32;
    while index < TAPS {
        occluded += tap(
            depth_tex, camera, ox, oy, oz, tx, ty, tz, bx, by, bz, nnx, nny, nnz, index, cos_a,
            sin_a, dims_x, dims_y, radius, bias,
        );
        index += 1;
    }
    let visibility = clamp(1.0 - (occluded / (TAPS as f32)) * intensity, 0.0, 1.0);
    fragment_store(visibility, visibility, visibility, 1.0);
}
