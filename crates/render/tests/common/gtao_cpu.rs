//! CPU march of the GTAO kernel against axis-aligned boxes.
//!
//! Horizons come from the same four screen steps as the shader. Depth is the
//! hit through the texel centre (what the g-buffer stores); the sample uv is
//! reconstructed at that depth, matching `gtao_horizon_cos`. Each horizon is
//! clamped with [`gtao_clamp_horizon`](ornis_render::gtao_clamp_horizon) and
//! integrated with [`gtao_sample`](ornis_render::gtao_sample), then blurred
//! and bilinearly upsampled the way the frame reads `ao`.

#![allow(dead_code)]

use ornis_render::ao::DEFAULT_AO_BLUR_SIGMA;

/// Full-frame width the ledge gate renders at.
const FRAME_W: f32 = 320.0;
/// Full-frame height the ledge gate renders at.
const FRAME_H: f32 = 180.0;
/// Half-resolution edge the AO targets use.
const AO_SCALE: f32 = 2.0;

/// One instanced box: center and full side length, matching `create_box`.
pub struct Aabb {
    /// Minimum corner.
    pub min: glam::Vec3,
    /// Maximum corner.
    pub max: glam::Vec3,
}

/// Box from a translation and a full side length.
pub fn aabb(translation: [f32; 3], scale: [f32; 3]) -> Aabb {
    let center = glam::Vec3::from_array(translation);
    let half = glam::Vec3::from_array(scale) * 0.5;
    Aabb {
        min: center - half,
        max: center + half,
    }
}

struct Cam {
    eye: glam::Vec3,
    vp: glam::Mat4,
    inv: glam::Mat4,
}

struct Pixel {
    uv: glam::Vec2,
    center: glam::Vec3,
    normal: glam::Vec3,
    view_vec: glam::Vec3,
    screen_x: glam::Vec3,
    screen_y: glam::Vec3,
    uv_radius: f32,
    radius: f32,
}

/// Blurred, upsampled AO factor at a full-frame uv.
pub fn expected_ao(
    eye: [f32; 3],
    view_proj: glam::Mat4,
    boxes: &[Aabb],
    full_uv: glam::Vec2,
    radius: f32,
) -> f32 {
    let cam = Cam {
        eye: glam::Vec3::from_array(eye),
        vp: view_proj,
        inv: view_proj.inverse(),
    };
    let (x0, y0, tx, ty) = texel_lerp(full_uv);
    let mut acc = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let wx = if dx == 0 { 1.0 - tx } else { tx };
            let wy = if dy == 0 { 1.0 - ty } else { ty };
            acc += wx * wy * blurred(&cam, boxes, radius, x0 + dx, y0 + dy);
        }
    }
    acc
}

fn reconstruct(inv: glam::Mat4, uv: glam::Vec2, depth: f32) -> glam::Vec3 {
    let ndc = glam::Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = inv * ndc;
    world.truncate() / world.w
}

fn project_uv(vp: glam::Mat4, point: glam::Vec3) -> glam::Vec2 {
    let clip = vp * point.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    glam::Vec2::new(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5)
}

/// One axis of a ray-box slab test.
struct Slab {
    origin: f32,
    dir: f32,
    min: f32,
    max: f32,
    axis: glam::Vec3,
}

fn ray_aabb(origin: glam::Vec3, dir: glam::Vec3, box_: &Aabb) -> Option<(f32, glam::Vec3)> {
    let mut t_min = 0.0_f32;
    let mut t_max = f32::MAX;
    let mut normal = glam::Vec3::Y;
    let origin_a = origin.to_array();
    let dir_a = dir.to_array();
    let min_a = box_.min.to_array();
    let max_a = box_.max.to_array();
    for (i, axis) in [glam::Vec3::X, glam::Vec3::Y, glam::Vec3::Z]
        .into_iter()
        .enumerate()
    {
        if !slab(
            &mut t_min,
            &mut t_max,
            &mut normal,
            Slab {
                origin: origin_a[i],
                dir: dir_a[i],
                min: min_a[i],
                max: max_a[i],
                axis,
            },
        ) {
            return None;
        }
    }
    (t_min >= 1.0e-4).then_some((t_min, normal))
}

fn slab(t_min: &mut f32, t_max: &mut f32, normal: &mut glam::Vec3, span: Slab) -> bool {
    if span.dir.abs() <= 1.0e-8 {
        return (span.min..=span.max).contains(&span.origin);
    }
    let inv = 1.0 / span.dir;
    let (t_near, t_far, face) = if span.dir > 0.0 {
        (
            (span.min - span.origin) * inv,
            (span.max - span.origin) * inv,
            -span.axis,
        )
    } else {
        (
            (span.max - span.origin) * inv,
            (span.min - span.origin) * inv,
            span.axis,
        )
    };
    if t_near > *t_min {
        *t_min = t_near;
        *normal = face;
    }
    *t_max = t_far.min(*t_max);
    *t_max >= *t_min
}

fn closest(
    origin: glam::Vec3,
    dir: glam::Vec3,
    boxes: &[Aabb],
) -> Option<(glam::Vec3, glam::Vec3)> {
    let mut best_t = f32::MAX;
    let mut best_n = glam::Vec3::Y;
    for box_ in boxes {
        if let Some((t, normal)) = ray_aabb(origin, dir, box_)
            && t < best_t
        {
            best_t = t;
            best_n = normal;
        }
    }
    (best_t < f32::MAX).then_some((origin + dir * best_t, best_n))
}

fn hit(cam: &Cam, uv: glam::Vec2, boxes: &[Aabb]) -> Option<(glam::Vec3, glam::Vec3)> {
    let dir = (reconstruct(cam.inv, uv, 1.0) - cam.eye).normalize();
    let (point, normal) = closest(cam.eye, dir, boxes)?;
    let clip = cam.vp * point.extend(1.0);
    (clip.z / clip.w < 1.0 - 1.0e-4).then_some((point, normal))
}

/// Texel the shader's `textureLoad` reads for `uv` (`floor(uv * dims)`).
fn texel_coord(uv: glam::Vec2) -> (u32, u32) {
    let x = (uv.x * FRAME_W).floor().clamp(0.0, FRAME_W - 1.0) as u32;
    let y = (uv.y * FRAME_H).floor().clamp(0.0, FRAME_H - 1.0) as u32;
    (x, y)
}

/// Depth and normal stored for `uv`: the surface through that texel center.
/// A miss is the clear depth, which the shader still reconstructs.
fn texel_depth(cam: &Cam, uv: glam::Vec2, boxes: &[Aabb]) -> (f32, Option<glam::Vec3>) {
    let (tx, ty) = texel_coord(uv);
    let center_uv = glam::Vec2::new((tx as f32 + 0.5) / FRAME_W, (ty as f32 + 0.5) / FRAME_H);
    match hit(cam, center_uv, boxes) {
        Some((point, normal)) => {
            let clip = cam.vp * point.extend(1.0);
            (clip.z / clip.w, Some(normal))
        }
        None => (1.0, None),
    }
}

fn pixel_at(cam: &Cam, boxes: &[Aabb], uv: glam::Vec2, radius: f32) -> Option<Pixel> {
    let (depth, normal) = texel_depth(cam, uv, boxes);
    let normal = normal?;
    let center = reconstruct(cam.inv, uv, depth);
    let view_vec = (cam.eye - center).normalize();
    let dims = glam::Vec2::new(FRAME_W, FRAME_H);
    let step_x = glam::Vec2::new(1.0 / dims.x, 0.0);
    let step_y = glam::Vec2::new(0.0, 1.0 / dims.y);
    let screen_x =
        (reconstruct(cam.inv, uv + step_x, depth) - center + glam::Vec3::X * 1.0e-6).normalize();
    let screen_y =
        (reconstruct(cam.inv, uv + step_y, depth) - center + glam::Vec3::Y * 1.0e-6).normalize();
    let edge_uv = project_uv(cam.vp, center + screen_x * radius);
    Some(Pixel {
        uv,
        center,
        normal,
        view_vec,
        screen_x,
        screen_y,
        uv_radius: (edge_uv - uv).length().max(1.0 / dims.x),
        radius,
    })
}

fn gradient(px: u32, py: u32) -> f32 {
    let fract_dot = (px as f32 * 0.06711056 + py as f32 * 0.00583715).fract();
    // Same f32 as the shader literal `52.9829189` (Jimenez interleaved gradient).
    (52.982_918 * fract_dot).fract()
}

fn slice_base(uv: glam::Vec2) -> f32 {
    let px = (uv.x * FRAME_W) as u32;
    let py = (uv.y * FRAME_H) as u32;
    gradient(px, py) * std::f32::consts::PI
}

fn sample_cos(cam: &Cam, boxes: &[Aabb], pixel: &Pixel, uv: glam::Vec2, min_cos: f32) -> f32 {
    if !(0.0..=1.0).contains(&uv.x) || !(0.0..=1.0).contains(&uv.y) {
        return -1.0;
    }
    let (depth, normal) = texel_depth(cam, uv, boxes);
    if normal.is_none() {
        return -1.0;
    }
    let point = reconstruct(cam.inv, uv, depth);
    let delta = point - pixel.center;
    let distance = delta.length();
    if distance <= 1.0e-6 || distance >= pixel.radius {
        return -1.0;
    }
    let falloff = (1.0 - distance / pixel.radius).clamp(0.0, 1.0);
    let cos_h = delta.dot(pixel.view_vec) / distance;
    min_cos + (cos_h - min_cos) * falloff
}

fn sign_of(value: f32) -> f32 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn one_slice(cam: &Cam, boxes: &[Aabb], pixel: &Pixel, phi: f32) -> f32 {
    let (cos_phi, sin_phi) = (phi.cos(), phi.sin());
    let slice_dir = (pixel.screen_x * cos_phi + pixel.screen_y * sin_phi).normalize();
    let axis = (slice_dir.cross(pixel.view_vec) + glam::Vec3::X * 1.0e-6).normalize();
    let projected = pixel.normal - axis * pixel.normal.dot(axis);
    let proj_len = projected.length().max(1.0e-6);
    let cos_norm = (projected.dot(pixel.view_vec) / proj_len).clamp(0.0, 1.0);
    let normal_angle = sign_of(slice_dir.dot(projected)) * cos_norm.acos();
    let min_pos = (normal_angle + std::f32::consts::FRAC_PI_2).cos();
    let min_neg = (normal_angle - std::f32::consts::FRAC_PI_2).cos();
    let mut horizon_pos = -1.0_f32;
    let mut horizon_neg = -1.0_f32;
    let slice_uv = glam::Vec2::new(cos_phi, sin_phi);
    for step in 1..=4 {
        let off = slice_uv * (pixel.uv_radius * step as f32 / 4.0);
        horizon_pos = horizon_pos.max(sample_cos(cam, boxes, pixel, pixel.uv + off, min_pos));
        horizon_neg = horizon_neg.max(sample_cos(cam, boxes, pixel, pixel.uv - off, min_neg));
    }
    let positive = horizon_pos.clamp(-1.0, 1.0).acos();
    let negative = -horizon_neg.clamp(-1.0, 1.0).acos();
    // `gtao_sample` clamps one slice to 1. The kernel only clamps after
    // averaging both slices, and an open off-axis slice integrates above
    // 1. A quarter of the projected length stays under that clamp; scaling
    // back recovers the kernel's unclamped slice.
    const SLICE_SCALE: f32 = 1.0 / (2.0 + 2.0);
    ornis_render::gtao_sample(
        ornis_render::gtao_clamp_horizon(positive, normal_angle),
        ornis_render::gtao_clamp_horizon(negative, normal_angle),
        normal_angle,
        proj_len * SLICE_SCALE,
    ) / SLICE_SCALE
}

fn raw_at(cam: &Cam, boxes: &[Aabb], uv: glam::Vec2, radius: f32) -> f32 {
    let Some(pixel) = pixel_at(cam, boxes, uv, radius) else {
        return 1.0;
    };
    let base = slice_base(uv);
    let half = std::f32::consts::FRAC_PI_2;
    ((one_slice(cam, boxes, &pixel, base) + one_slice(cam, boxes, &pixel, base + half)) * 0.5)
        .clamp(0.0, 1.0)
}

fn half_uv(hx: i32, hy: i32) -> glam::Vec2 {
    let half = glam::Vec2::new(FRAME_W, FRAME_H) / AO_SCALE;
    glam::Vec2::new((hx as f32 + 0.5) / half.x, (hy as f32 + 0.5) / half.y)
}

fn raw_texel(cam: &Cam, boxes: &[Aabb], radius: f32, hx: i32, hy: i32) -> f32 {
    let half = glam::Vec2::new(FRAME_W, FRAME_H) / AO_SCALE;
    if hx < 0 || hy < 0 || hx >= half.x as i32 || hy >= half.y as i32 {
        return 1.0;
    }
    raw_at(cam, boxes, half_uv(hx, hy), radius)
}

/// Half-texel the full-frame `uv` falls between, plus the bilinear weights.
fn texel_lerp(uv: glam::Vec2) -> (i32, i32, f32, f32) {
    let half = glam::Vec2::new(FRAME_W, FRAME_H) / AO_SCALE;
    let fx = uv.x * half.x - 0.5;
    let fy = uv.y * half.y - 0.5;
    let x0 = fx.floor() as i32;
    let y0 = fy.floor() as i32;
    (x0, y0, fx - x0 as f32, fy - y0 as f32)
}

fn bilinear_raw(cam: &Cam, boxes: &[Aabb], radius: f32, uv: glam::Vec2) -> f32 {
    let (x0, y0, tx, ty) = texel_lerp(uv);
    let mut acc = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let wx = if dx == 0 { 1.0 - tx } else { tx };
            let wy = if dy == 0 { 1.0 - ty } else { ty };
            acc += wx * wy * raw_texel(cam, boxes, radius, x0 + dx, y0 + dy);
        }
    }
    acc
}

fn eye_distance(cam: &Cam, boxes: &[Aabb], uv: glam::Vec2) -> f32 {
    let (depth, _) = texel_depth(cam, uv, boxes);
    (cam.eye - reconstruct(cam.inv, uv, depth)).length()
}

fn blurred(cam: &Cam, boxes: &[Aabb], radius: f32, hx: i32, hy: i32) -> f32 {
    let uv = half_uv(hx, hy);
    let center = raw_texel(cam, boxes, radius, hx, hy);
    let center_z = eye_distance(cam, boxes, uv);
    let texel = glam::Vec2::new(AO_SCALE / FRAME_W, AO_SCALE / FRAME_H);
    let mut acc = center;
    let mut weight = 1.0;
    for off in [
        glam::Vec2::new(1.5, 0.0),
        glam::Vec2::new(-1.5, 0.0),
        glam::Vec2::new(0.0, 1.5),
        glam::Vec2::new(0.0, -1.5),
    ] {
        let sample_uv = (uv + off * texel).clamp(glam::Vec2::ZERO, glam::Vec2::ONE);
        let delta = (center_z - eye_distance(cam, boxes, sample_uv)) / DEFAULT_AO_BLUR_SIGMA;
        let w = (-delta * delta).exp();
        acc += bilinear_raw(cam, boxes, radius, sample_uv) * w;
        weight += w;
    }
    acc / weight
}
