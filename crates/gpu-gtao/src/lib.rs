//! GTAO horizon march for the stable Rust to WGSL path.
//!
//! One fragment reads the fullscreen-quad UV, the g-buffer depth and
//! octahedral normal, and the same `Camera` / `AoParams` uniforms the
//! renderer uploads. The translator only has scalar imports, so the
//! matrices are columns of `vec4` and each sample stays three lanes.

#![no_std]
#![allow(clippy::all)]
#![allow(unsafe_op_in_unsafe_fn)]

unsafe extern "C" {
    fn fragment_uv_lane(lane: u32) -> f32;
    fn fragment_store_f32x4(r: f32, g: f32, b: f32, a: f32);
    fn texture_load_2d_f32(group: u32, binding: u32, x: u32, y: u32) -> f32;
    fn texture_load_2d_f32x4(group: u32, binding: u32, x: u32, y: u32, lane: u32) -> f32;
    fn texture_width_2d(group: u32, binding: u32) -> i32;
    fn texture_height_2d(group: u32, binding: u32) -> i32;
    fn gpu_cos(value: f32) -> f32;
    fn gpu_sin(value: f32) -> f32;
    fn gpu_acos(value: f32) -> f32;
    fn gpu_clamp(value: f32, low: f32, high: f32) -> f32;
    fn gpu_mix(start: f32, end: f32, factor: f32) -> f32;
    fn gpu_sign(value: f32) -> f32;
    fn gpu_dot3(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) -> f32;
    fn gpu_length3(x: f32, y: f32, z: f32) -> f32;
    fn gpu_cross3_lane(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32, lane: u32) -> f32;
    fn gpu_normalize3_lane(x: f32, y: f32, z: f32, lane: u32) -> f32;
    fn gpu_fract(value: f32) -> f32;
    fn uniform_field_f32(group: u32, binding: u32, sid: u32, field: u32) -> f32;
    fn uniform_field_vec4_component_f32(
        group: u32,
        binding: u32,
        sid: u32,
        field: u32,
        lane: u32,
    ) -> f32;
}
#[used]
#[cfg_attr(
    target_arch = "wasm32",
    unsafe(link_section = "ornis_gpu.shader_meta")
)]
#[unsafe(export_name = "ornis_gpu_shader_meta")]
static ORNIS_GPU_SHADER_META: [u8; 60] =
    *b"entry=fs_main;workgroup=1,1,1;uniform=0:6666666666666,1:1111";
const CLEAR_DEPTH: f32 = 1.0;
const EPS: f32 = 1.0e-6;
const PI: f32 = 3.1415927;
const HALF_PI: f32 = 1.5707964;

#[inline(always)]
// qual:allow(magic_numbers, complexity) reason: "column lane of a host mat4; the indices are the Camera field layout"
unsafe fn mat_lane(base: u32, lane: u32, x: f32, y: f32, z: f32) -> f32 {
    uniform_field_vec4_component_f32(0, 2, 0, base, lane) * x
        + uniform_field_vec4_component_f32(0, 2, 0, base + 1, lane) * y
        + uniform_field_vec4_component_f32(0, 2, 0, base + 2, lane) * z
        + uniform_field_vec4_component_f32(0, 2, 0, base + 3, lane)
}

#[inline(always)]
// qual:allow(magic_numbers) reason: "NDC remap of a depth sample, matching reconstruct_world_pos"
unsafe fn world_lane(lane: u32, uvx: f32, uvy: f32, depth: f32) -> f32 {
    let nx = uvx * 2.0 - 1.0;
    let ny = 1.0 - uvy * 2.0;
    let value = mat_lane(4, lane, nx, ny, depth);
    let w = f32::max(mat_lane(4, 3, nx, ny, depth), EPS);
    value / w
}

#[inline(always)]
// qual:allow(magic_numbers, complexity, unsafe) reason: "one horizon tap; the closed form is the XeGTAO sample"
unsafe fn horizon(
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
        let x = (uvx * dims_x) as u32;
        let y = (uvy * dims_y) as u32;
        let sample_depth = texture_load_2d_f32(0, 0, x, y);
        if sample_depth < CLEAR_DEPTH {
            let sx = world_lane(0, uvx, uvy, sample_depth);
            let sy = world_lane(1, uvx, uvy, sample_depth);
            let sz = world_lane(2, uvx, uvy, sample_depth);
            let dx = sx - cx;
            let dy = sy - cy;
            let dz = sz - cz;
            let distance = gpu_length3(dx, dy, dz);
            if distance > EPS && distance < radius {
                let falloff = gpu_clamp(1.0 - distance / f32::max(radius, EPS), 0.0, 1.0);
                let along = gpu_dot3(dx / distance, dy / distance, dz / distance, vvx, vvy, vvz);
                cos_h = gpu_mix(min_cos, along, falloff);
            }
        }
    }
    cos_h
}

/// Fragment entry. The vertex quad passes UV at `@location(0)`.
#[unsafe(export_name = "fs_main")]
// qual:allow(magic_numbers, complexity, max_function_lines=400, unsafe) reason: "the GTAO march is one fragment; splitting it would hide the slice loop"
pub extern "C" fn fs_main() {
    // SAFETY: these calls are the translator's GPU imports, not host memory.
    unsafe {
        let uvx = fragment_uv_lane(0);
        let uvy = fragment_uv_lane(1);
        let dims_x = texture_width_2d(0, 0) as f32;
        let dims_y = texture_height_2d(0, 0) as f32;
        let xi = (uvx * dims_x) as u32;
        let yi = (uvy * dims_y) as u32;
        let depth = texture_load_2d_f32(0, 0, xi, yi);
        if depth >= CLEAR_DEPTH {
            fragment_store_f32x4(1.0, 1.0, 1.0, 1.0);
            return;
        }
        let enc_x = texture_load_2d_f32x4(0, 1, xi, yi, 0);
        let enc_y = texture_load_2d_f32x4(0, 1, xi, yi, 1);
        let nz = 1.0 - f32::abs(enc_x) - f32::abs(enc_y);
        let fold = f32::max(-nz, 0.0);
        let nx = enc_x + if enc_x >= 0.0 { fold } else { -fold };
        let ny = enc_y + if enc_y >= 0.0 { fold } else { -fold };
        let nnx = gpu_normalize3_lane(nx, ny, nz, 0);
        let nny = gpu_normalize3_lane(nx, ny, nz, 1);
        let nnz = gpu_normalize3_lane(nx, ny, nz, 2);
        let cx = world_lane(0, uvx, uvy, depth);
        let cy = world_lane(1, uvx, uvy, depth);
        let cz = world_lane(2, uvx, uvy, depth);
        let eye_x = uniform_field_vec4_component_f32(0, 2, 0, 8, 0);
        let eye_y = uniform_field_vec4_component_f32(0, 2, 0, 8, 1);
        let eye_z = uniform_field_vec4_component_f32(0, 2, 0, 8, 2);
        let vlen = f32::max(gpu_length3(eye_x - cx, eye_y - cy, eye_z - cz), EPS);
        let vvx = (eye_x - cx) / vlen;
        let vvy = (eye_y - cy) / vlen;
        let vvz = (eye_z - cz) / vlen;
        let rx = world_lane(0, uvx + 1.0 / dims_x, uvy, depth);
        let ry = world_lane(1, uvx + 1.0 / dims_x, uvy, depth);
        let rz = world_lane(2, uvx + 1.0 / dims_x, uvy, depth);
        let ux = world_lane(0, uvx, uvy + 1.0 / dims_y, depth);
        let uy = world_lane(1, uvx, uvy + 1.0 / dims_y, depth);
        let uz = world_lane(2, uvx, uvy + 1.0 / dims_y, depth);
        let sxx = gpu_normalize3_lane(rx - cx + EPS, ry - cy, rz - cz, 0);
        let sxy = gpu_normalize3_lane(rx - cx + EPS, ry - cy, rz - cz, 1);
        let sxz = gpu_normalize3_lane(rx - cx + EPS, ry - cy, rz - cz, 2);
        let syx = gpu_normalize3_lane(ux - cx, uy - cy + EPS, uz - cz, 0);
        let syy = gpu_normalize3_lane(ux - cx, uy - cy + EPS, uz - cz, 1);
        let syz = gpu_normalize3_lane(ux - cx, uy - cy + EPS, uz - cz, 2);
        let radius = uniform_field_f32(0, 3, 1, 0);
        let intensity = uniform_field_f32(0, 3, 1, 1);
        let edge_x = cx + sxx * radius;
        let edge_y = cy + sxy * radius;
        let edge_z = cz + sxz * radius;
        let clip_x = mat_lane(0, 0, edge_x, edge_y, edge_z);
        let clip_y = mat_lane(0, 1, edge_x, edge_y, edge_z);
        let clip_w = f32::max(mat_lane(0, 3, edge_x, edge_y, edge_z), EPS);
        let edge_u = clip_x / clip_w * 0.5 + 0.5;
        let edge_v = 0.5 - clip_y / clip_w * 0.5;
        let uv_radius = f32::max(gpu_length3(edge_u - uvx, edge_v - uvy, 0.0), 1.0 / dims_x);
        let hash =
            gpu_fract(52.9829189 * gpu_fract((xi as f32) * 0.06711056 + (yi as f32) * 0.00583715));
        let mut visibility = 0.0f32;
        let mut slice = 0u32;
        while slice < 2 {
            let phi = hash * PI + (slice as f32) * HALF_PI;
            let cphi = gpu_cos(phi);
            let sphi = gpu_sin(phi);
            let dir_x = sxx * cphi + syx * sphi;
            let dir_y = sxy * cphi + syy * sphi;
            let dir_z = sxz * cphi + syz * sphi;
            let sdx = gpu_normalize3_lane(dir_x, dir_y, dir_z, 0);
            let sdy = gpu_normalize3_lane(dir_x, dir_y, dir_z, 1);
            let sdz = gpu_normalize3_lane(dir_x, dir_y, dir_z, 2);
            let raw_x = gpu_cross3_lane(sdx, sdy, sdz, vvx, vvy, vvz, 0) + EPS;
            let raw_y = gpu_cross3_lane(sdx, sdy, sdz, vvx, vvy, vvz, 1);
            let raw_z = gpu_cross3_lane(sdx, sdy, sdz, vvx, vvy, vvz, 2);
            let ax = gpu_normalize3_lane(raw_x, raw_y, raw_z, 0);
            let ay = gpu_normalize3_lane(raw_x, raw_y, raw_z, 1);
            let az = gpu_normalize3_lane(raw_x, raw_y, raw_z, 2);
            let along = gpu_dot3(nnx, nny, nnz, ax, ay, az);
            let proj_x = nnx - ax * along;
            let proj_y = nny - ay * along;
            let proj_z = nnz - az * along;
            let proj_len = f32::max(gpu_length3(proj_x, proj_y, proj_z), EPS);
            let cos_norm = gpu_clamp(
                gpu_dot3(proj_x, proj_y, proj_z, vvx, vvy, vvz) / proj_len,
                0.0,
                1.0,
            );
            let normal_angle =
                gpu_sign(gpu_dot3(sdx, sdy, sdz, proj_x, proj_y, proj_z)) * gpu_acos(cos_norm);
            let min_pos = gpu_cos(normal_angle + HALF_PI);
            let min_neg = gpu_cos(normal_angle - HALF_PI);
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
            let h_pos = gpu_acos(gpu_clamp(horizon_pos, -1.0, 1.0));
            let h_neg = -gpu_acos(gpu_clamp(horizon_neg, -1.0, 1.0));
            let n = normal_angle;
            let arc_pos =
                0.25 * (-gpu_cos(2.0 * h_pos - n) + gpu_cos(n) + 2.0 * h_pos * gpu_sin(n));
            let arc_neg =
                0.25 * (-gpu_cos(2.0 * h_neg - n) + gpu_cos(n) + 2.0 * h_neg * gpu_sin(n));
            visibility += proj_len * (arc_pos + arc_neg);
            slice += 1;
        }
        let ao = gpu_clamp(gpu_mix(1.0, visibility / 2.0, intensity), 0.0, 1.0);
        fragment_store_f32x4(ao, ao, ao, 1.0);
    }
}
