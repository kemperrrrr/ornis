// qual:allow(srp, file_length=360) reason: "one GTAO march: the horizon clamp, the vec2 IGN dot, and the raw homogeneous divide stay next to the slice loop"
//! GTAO horizon march for the stable Rust to WGSL path.
//!
//! One fragment reads the fullscreen-quad UV, the g-buffer depth and
//! octahedral normal, and the same `Camera` / `AoParams` uniforms the
//! renderer uploads. The kernel is a safe `fn` on the shader SDK: the
//! fragment entry, the textures, and the uniforms are parameters, and the
//! math is SDK functions. It does not declare `extern "C"`.

#![no_std]

use ornis_gpu_shader::DepthTexture2d;
use ornis_gpu_shader::GpuStruct;
use ornis_gpu_shader::TYPE_TAG_F32;
use ornis_gpu_shader::TYPE_TAG_VEC4;
use ornis_gpu_shader::Texture2d;
use ornis_gpu_shader::Uniform;
use ornis_gpu_shader::abs;
use ornis_gpu_shader::acos;
use ornis_gpu_shader::clamp;
use ornis_gpu_shader::cos;
use ornis_gpu_shader::cross_lane;
use ornis_gpu_shader::dot2;
use ornis_gpu_shader::dot3;
use ornis_gpu_shader::fract;
use ornis_gpu_shader::fragment_store;
use ornis_gpu_shader::fragment_uv;
use ornis_gpu_shader::gpu_entry;
use ornis_gpu_shader::length3;
use ornis_gpu_shader::mix;
use ornis_gpu_shader::normalize_lane;
use ornis_gpu_shader::sign;
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
/// `ao_interleaved_gradient` weights: `dot(pixel, (x, y))` then `fract(z * fract(.))`.
const HASH_X: f32 = 0.06711056;
const HASH_Y: f32 = 0.00583715;
const HASH_Z: f32 = 52.982_918;
/// π at `f32` precision (same value the kernel used as a literal).
const PI: f32 = core::f32::consts::PI;
/// π/2 at `f32` precision.
const HALF_PI: f32 = core::f32::consts::FRAC_PI_2;

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
    let w = inv_view_lane::<3>(camera, nx, ny, depth);
    // Master divides by the raw homogeneous w, including w < EPS and w < 0.
    homogeneous_divide(value, w)
}

/// `value / w` with the sign and magnitude of `w` left unchanged.
///
/// The hand-written `reconstruct_world_pos` returns `view.xyz / view.w`.
/// This path divides by the raw homogeneous `w` the same way, including
/// when `|w|` is below `EPS` or `w` is negative. Clamping `w` up to `EPS`
/// would move those reconstructions and would flip a negative denominator.
#[inline(always)]
#[must_use]
pub fn homogeneous_divide(value: f32, w: f32) -> f32 {
    value / w
}

/// Master's `fract(magic.z * fract(dot(vec2(pixel), magic.xy)))`.
///
/// The dot is the `vec2` builtin. A scalar `x * cx + y * cy` contracts in
/// the opposite order and changes the slice angle.
#[inline(always)]
fn interleaved_gradient(pixel_x: u32, pixel_y: u32) -> f32 {
    let dot = dot2(pixel_x as f32, pixel_y as f32, HASH_X, HASH_Y);
    fract(HASH_Z * fract(dot))
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

/// Texel index of `uv` in a texture of `dim` texels.
///
/// `uv == 1` truncates to `dim`, one past the last texel. The hand-written
/// `gtao_horizon_cos` clamps with `min(u32(uv * dims), dims - 1)` so the
/// load reads that edge texel. An unclamped `textureLoad` is out of bounds;
/// a backend zero is not the last texel.
#[inline(always)]
#[must_use]
pub fn clamped_texel(uv: f32, dim: f32) -> u32 {
    min_u32((uv * dim) as u32, (dim - 1.0) as u32)
}

#[inline(always)]
// qual:allow(srp, max_parameters=14) reason: "one horizon tap; the closed form is the XeGTAO sample and the resources stay arguments so the body does not restate bindings"
// A struct here would need layout lowering the translator does not need for
// inlined helpers; the kernel keeps one flat signature like the WGSL.
#[allow(clippy::too_many_arguments)]
fn horizon(
    depth_tex: Depth,
    camera: CameraUniform,
    uvx: f32,
    uvy: f32,
    dims_x: f32,
    dims_y: f32,
    cx: f32,
    cy: f32,
    cz: f32,
    vvx: f32,
    vvy: f32,
    vvz: f32,
    min_cos: f32,
    radius: f32,
) -> f32 {
    let mut cos_h = -1.0f32;
    if uvx >= 0.0 && uvy >= 0.0 && uvx <= 1.0 && uvy <= 1.0 {
        let x = clamped_texel(uvx, dims_x);
        let y = clamped_texel(uvy, dims_y);
        let sample_depth = depth_tex.load(x, y);
        if sample_depth < CLEAR_DEPTH {
            let sx = world_lane::<0>(camera, uvx, uvy, sample_depth);
            let sy = world_lane::<1>(camera, uvx, uvy, sample_depth);
            let sz = world_lane::<2>(camera, uvx, uvy, sample_depth);
            let dx = sx - cx;
            let dy = sy - cy;
            let dz = sz - cz;
            let distance = length3(dx, dy, dz);
            if distance > EPS && distance < radius {
                let falloff = clamp(1.0 - distance / f32::max(radius, EPS), 0.0, 1.0);
                let along = dot3(dx / distance, dy / distance, dz / distance, vvx, vvy, vvz);
                cos_h = mix(min_cos, along, falloff);
            }
        }
    }
    cos_h
}

/// Fragment entry. The vertex quad passes UV at `@location(0)`.
#[gpu_entry(fragment)]
// qual:allow(complexity, magic_numbers) reason: "the GTAO march is one fragment; splitting it would hide the slice loop"
// qual:allow(complexity, max_function_lines=150) reason: "the GTAO march is one fragment; splitting it would hide the slice loop"
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
    let nx = enc_x + if enc_x >= 0.0 { -fold } else { fold };
    let ny = enc_y + if enc_y >= 0.0 { -fold } else { fold };
    let nnx = normalize_lane::<0>(nx, ny, nz);
    let nny = normalize_lane::<1>(nx, ny, nz);
    let nnz = normalize_lane::<2>(nx, ny, nz);
    let cx = world_lane::<0>(camera, uvx, uvy, depth);
    let cy = world_lane::<1>(camera, uvx, uvy, depth);
    let cz = world_lane::<2>(camera, uvx, uvy, depth);
    let eye_x = camera.vec4_lane::<8, 0>();
    let eye_y = camera.vec4_lane::<8, 1>();
    let eye_z = camera.vec4_lane::<8, 2>();
    let vlen = f32::max(length3(eye_x - cx, eye_y - cy, eye_z - cz), EPS);
    let vvx = (eye_x - cx) / vlen;
    let vvy = (eye_y - cy) / vlen;
    let vvz = (eye_z - cz) / vlen;
    let rx = world_lane::<0>(camera, uvx + 1.0 / dims_x, uvy, depth);
    let ry = world_lane::<1>(camera, uvx + 1.0 / dims_x, uvy, depth);
    let rz = world_lane::<2>(camera, uvx + 1.0 / dims_x, uvy, depth);
    let ux = world_lane::<0>(camera, uvx, uvy + 1.0 / dims_y, depth);
    let uy = world_lane::<1>(camera, uvx, uvy + 1.0 / dims_y, depth);
    let uz = world_lane::<2>(camera, uvx, uvy + 1.0 / dims_y, depth);
    let sxx = normalize_lane::<0>(rx - cx + EPS, ry - cy, rz - cz);
    let sxy = normalize_lane::<1>(rx - cx + EPS, ry - cy, rz - cz);
    let sxz = normalize_lane::<2>(rx - cx + EPS, ry - cy, rz - cz);
    let syx = normalize_lane::<0>(ux - cx, uy - cy + EPS, uz - cz);
    let syy = normalize_lane::<1>(ux - cx, uy - cy + EPS, uz - cz);
    let syz = normalize_lane::<2>(ux - cx, uy - cy + EPS, uz - cz);
    let radius = ao.field_f32::<0>();
    let intensity = ao.field_f32::<1>();
    let edge_x = cx + sxx * radius;
    let edge_y = cy + sxy * radius;
    let edge_z = cz + sxz * radius;
    let clip_x = view_lane::<0>(camera, edge_x, edge_y, edge_z);
    let clip_y = view_lane::<1>(camera, edge_x, edge_y, edge_z);
    let clip_w = f32::max(view_lane::<3>(camera, edge_x, edge_y, edge_z), EPS);
    let edge_u = clip_x / clip_w * 0.5 + 0.5;
    let edge_v = 0.5 - clip_y / clip_w * 0.5;
    let uv_radius = f32::max(length3(edge_u - uvx, edge_v - uvy, 0.0), 1.0 / dims_x);
    let hash = interleaved_gradient(xi, yi);
    let mut visibility = 0.0f32;
    let mut slice = 0u32;
    while slice < 2 {
        let phi = hash * PI + (slice as f32) * HALF_PI;
        let cphi = cos(phi);
        let sphi = sin(phi);
        let dir_x = sxx * cphi + syx * sphi;
        let dir_y = sxy * cphi + syy * sphi;
        let dir_z = sxz * cphi + syz * sphi;
        let sdx = normalize_lane::<0>(dir_x, dir_y, dir_z);
        let sdy = normalize_lane::<1>(dir_x, dir_y, dir_z);
        let sdz = normalize_lane::<2>(dir_x, dir_y, dir_z);
        let raw_x = cross_lane::<0>(sdx, sdy, sdz, vvx, vvy, vvz) + EPS;
        let raw_y = cross_lane::<1>(sdx, sdy, sdz, vvx, vvy, vvz);
        let raw_z = cross_lane::<2>(sdx, sdy, sdz, vvx, vvy, vvz);
        let ax = normalize_lane::<0>(raw_x, raw_y, raw_z);
        let ay = normalize_lane::<1>(raw_x, raw_y, raw_z);
        let az = normalize_lane::<2>(raw_x, raw_y, raw_z);
        let along = dot3(nnx, nny, nnz, ax, ay, az);
        let proj_x = nnx - ax * along;
        let proj_y = nny - ay * along;
        let proj_z = nnz - az * along;
        let proj_len = f32::max(length3(proj_x, proj_y, proj_z), EPS);
        let cos_norm = clamp(
            dot3(proj_x, proj_y, proj_z, vvx, vvy, vvz) / proj_len,
            0.0,
            1.0,
        );
        let normal_angle = sign(dot3(sdx, sdy, sdz, proj_x, proj_y, proj_z)) * acos(cos_norm);
        let min_pos = cos(normal_angle + HALF_PI);
        let min_neg = cos(normal_angle - HALF_PI);
        let mut horizon_pos = min_pos;
        let mut horizon_neg = min_neg;
        let mut step = 1u32;
        while step <= 4 {
            let off = uv_radius * (step as f32) / 4.0;
            let ox = cphi * off;
            let oy = sphi * off;
            horizon_pos = f32::max(
                horizon_pos,
                horizon(
                    depth_tex,
                    camera,
                    uvx + ox,
                    uvy + oy,
                    dims_x,
                    dims_y,
                    cx,
                    cy,
                    cz,
                    vvx,
                    vvy,
                    vvz,
                    min_pos,
                    radius,
                ),
            );
            horizon_neg = f32::max(
                horizon_neg,
                horizon(
                    depth_tex,
                    camera,
                    uvx - ox,
                    uvy - oy,
                    dims_x,
                    dims_y,
                    cx,
                    cy,
                    cz,
                    vvx,
                    vvy,
                    vvz,
                    min_neg,
                    radius,
                ),
            );
            step += 1;
        }
        let h_pos = acos(clamp(horizon_pos, -1.0, 1.0));
        let h_neg = -acos(clamp(horizon_neg, -1.0, 1.0));
        let n = normal_angle;
        let arc_pos = 0.25 * (-cos(2.0 * h_pos - n) + cos(n) + 2.0 * h_pos * sin(n));
        let arc_neg = 0.25 * (-cos(2.0 * h_neg - n) + cos(n) + 2.0 * h_neg * sin(n));
        visibility += proj_len * (arc_pos + arc_neg);
        slice += 1;
    }
    let occlusion = clamp(mix(1.0, visibility / 2.0, intensity), 0.0, 1.0);
    fragment_store(occlusion, occlusion, occlusion, 1.0);
}
