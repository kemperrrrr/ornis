//! Synthetic depth with an interior edge and a distinct last column/row.
//!
//! Pixel (0, 0) has interleaved-gradient hash 0, so GTAO slice 0 steps along
//! +U. The radius is solved so step 1 of 4 lands on UV x = 1. That point is
//! a quarter of the radius from the centre, inside the horizon: master loads
//! texel `width - 1`, and an unclamped cast loads `width`.

use ornis_render::ao::{DEFAULT_AO_BIAS, DEFAULT_AO_BLUR_SIGMA, DEFAULT_AO_RADIUS};

/// Depth and AO extent. A power of two keeps the pixel-centre UV exact.
pub const W: u32 = 64;
/// Depth and AO extent.
pub const H: u32 = 64;
/// Guard used by both kernels (`1e-6`).
const EPS: f32 = 1.0e-6;
/// Near plane of the orthographic camera, metres.
const NEAR: f32 = 0.5;
/// Far plane of the orthographic camera, metres.
const FAR: f32 = 4.0;
/// Eye position: looking down −Z at the origin.
const EYE: [f32; 3] = [0.0, 0.0, 2.0];

/// One AO shot: camera, kernel uniforms, and the depth image.
pub struct Shot {
    /// View-projection, column-major, matching `CameraUniform`.
    pub view_proj: [[f32; 4]; 4],
    /// World-space eye.
    pub eye: [f32; 3],
    /// Kernel radius, metres.
    pub radius: f32,
    /// Kernel intensity.
    pub intensity: f32,
    /// SSAO bias, metres.
    pub bias: f32,
    /// Bilateral sigma, metres.
    pub blur_sigma: f32,
    /// Hardware depth, row-major, `W * H` texels. The right column stays on
    /// the near plane so UV = 1 is a horizon hit; the bottom row is far;
    /// `x >= W/2` is an interior step.
    pub depth: Vec<f32>,
    /// Scale applied to `view_proj`, with the inverse scaled by the
    /// reciprocal. `1` leaves the camera unchanged. A large scale makes
    /// the reconstruction `w` smaller than `1e-6`.
    pub proj_scale: f32,
}

/// 8-bit channel diff of two RGBA images, in the review's units.
pub struct LevelDiff {
    /// Largest per-pixel max-channel difference, in 8-bit levels.
    pub max_level: u8,
    /// Pixels whose max channel differs by more than 5 levels.
    pub over_5: usize,
    /// Mean absolute red-channel difference, in 0..1.
    pub mae: f32,
    /// Pixels compared.
    pub pixels: usize,
}

/// Orthographic DirectX projection: NDC z is 0 at `near` and 1 at `far`.
fn ortho_dx() -> glam::Mat4 {
    let (left, right, bottom, top) = (-1.0, 1.0, -1.0, 1.0);
    let sx = 2.0 / (right - left);
    let sy = 2.0 / (top - bottom);
    let sz = 1.0 / (NEAR - FAR);
    let tx = -(right + left) / (right - left);
    let ty = -(top + bottom) / (top - bottom);
    let tz = NEAR / (NEAR - FAR);
    glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, sz, 0.0),
        glam::Vec4::new(tx, ty, tz, 1.0),
    )
}

fn view_proj() -> glam::Mat4 {
    let eye = glam::Vec3::from_array(EYE);
    let view = glam::camera::rh::view::look_at_mat4(eye, glam::Vec3::ZERO, glam::Vec3::Y);
    ortho_dx() * view
}

fn reconstruct(inv: glam::Mat4, uv: glam::Vec2, depth: f32) -> glam::Vec3 {
    let ndc = glam::Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = inv * ndc;
    world.truncate() / world.w
}

fn project_uv(vp: glam::Mat4, point: glam::Vec3) -> glam::Vec2 {
    let clip = vp * point.extend(1.0);
    let w = clip.w.max(EPS);
    glam::Vec2::new(clip.x / w * 0.5 + 0.5, 0.5 - clip.y / w * 0.5)
}

fn ndc_z(vp: glam::Mat4, world: glam::Vec3) -> f32 {
    let clip = vp * world.extend(1.0);
    clip.z / clip.w
}

/// Pixel-centre UV of texel `(x, y)`, matching the fullscreen quad.
pub fn pixel_uv(x: u32, y: u32) -> glam::Vec2 {
    glam::Vec2::new((x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / H as f32)
}

/// UV of GTAO slice-0 step 1 at pixel (0, 0) for `radius`.
///
/// Hash of texel (0, 0) is 0, so the slice direction is +U and step 1
/// moves by `uv_radius / 4`.
fn border_sample_uv(vp: glam::Mat4, plane_depth: f32, radius: f32) -> glam::Vec2 {
    let inv = vp.inverse();
    let uv = pixel_uv(0, 0);
    let center = reconstruct(inv, uv, plane_depth);
    let right = reconstruct(inv, uv + glam::Vec2::new(1.0 / W as f32, 0.0), plane_depth);
    let screen_x = (right - center + glam::Vec3::new(EPS, 0.0, 0.0)).normalize();
    let edge_uv = project_uv(vp, center + screen_x * radius);
    let uv_radius = (edge_uv - uv).length().max(1.0 / W as f32);
    uv + glam::Vec2::new(uv_radius * 0.25, 0.0)
}

fn radius_for_border(vp: glam::Mat4, plane_depth: f32) -> f32 {
    // Two radii past one texel, where `uv_radius` is the geometric length
    // and the sample U is linear in the radius.
    let x1 = border_sample_uv(vp, plane_depth, 1.0).x;
    let x2 = border_sample_uv(vp, plane_depth, 2.0).x;
    let mut radius = 1.0 + (1.0 - x1) / (x2 - x1);
    if border_sample_uv(vp, plane_depth, radius).x.to_bits() == 1.0f32.to_bits() {
        return radius;
    }
    let mut bits = radius.to_bits();
    for _ in 0..128 {
        bits = bits.saturating_sub(1);
        radius = f32::from_bits(bits);
        if border_sample_uv(vp, plane_depth, radius).x.to_bits() == 1.0f32.to_bits() {
            return radius;
        }
    }
    radius
}

fn depth_image(plane: f32, step: f32, edge: f32) -> Vec<f32> {
    let mut depth = vec![plane; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let value = if y + 1 == H {
                edge
            } else if x + 1 == W {
                // Same surface as the left side, so the UV=1 tap is a hit
                // a quarter-radius away. Depth 0 (an out-of-bounds load)
                // reconstructs somewhere else.
                plane
            } else if x >= W / 2 {
                step
            } else {
                plane
            };
            depth[(y * W + x) as usize] = value;
        }
    }
    depth
}

/// Border scene: step 1 of pixel (0, 0) lands on UV x = 1.
pub fn border_shot() -> Shot {
    let vp = view_proj();
    let plane = ndc_z(vp, glam::Vec3::ZERO);
    let step = ndc_z(vp, glam::Vec3::new(0.0, 0.0, -0.6));
    let edge = ndc_z(vp, glam::Vec3::new(0.0, 0.0, -1.2));
    Shot {
        view_proj: vp.to_cols_array_2d(),
        eye: EYE,
        radius: radius_for_border(vp, plane),
        intensity: 1.0,
        bias: DEFAULT_AO_BIAS,
        blur_sigma: DEFAULT_AO_BLUR_SIGMA,
        depth: depth_image(plane, step, edge),
        proj_scale: 1.0,
    }
}

/// Same image at the production default radius, so taps stay local.
pub fn local_shot() -> Shot {
    let mut shot = border_shot();
    shot.radius = DEFAULT_AO_RADIUS;
    shot
}

/// `CameraUniform` bytes: view-projection, inverse, eye, previous matrix.
pub fn camera_bytes(shot: &Shot) -> Vec<u8> {
    let vp = glam::Mat4::from_cols_array_2d(&shot.view_proj);
    let inv = vp.inverse();
    let scale = shot.proj_scale;
    let mut bytes = Vec::with_capacity(208);
    // Scale the projection and the inverse by reciprocal factors. Inverting
    // the scaled matrix again would not be the same bits.
    push_mat(&mut bytes, (vp * scale).to_cols_array_2d());
    push_mat(&mut bytes, (inv * scale.recip()).to_cols_array_2d());
    for value in [shot.eye[0], shot.eye[1], shot.eye[2], 1.0] {
        bytes.extend(value.to_le_bytes());
    }
    push_mat(&mut bytes, (vp * scale).to_cols_array_2d());
    bytes
}

/// `AoParams` bytes: radius, intensity, bias, blur sigma.
pub fn ao_bytes(shot: &Shot) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for value in [shot.radius, shot.intensity, shot.bias, shot.blur_sigma] {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn push_mat(bytes: &mut Vec<u8>, mat: [[f32; 4]; 4]) {
    for column in mat {
        for value in column {
            bytes.extend(value.to_le_bytes());
        }
    }
}

/// Raw AO factor: open on the left, dark past the interior edge, darker
/// still on the image border. Red holds the factor.
pub fn ao_raw_rgba() -> Vec<u8> {
    let mut pixels = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let factor = if x + 1 == W || y + 1 == H {
                10
            } else if x >= W / 2 {
                40
            } else {
                255
            };
            let i = ((y * W + x) * 4) as usize;
            pixels[i] = factor;
            pixels[i + 1] = factor;
            pixels[i + 2] = factor;
            pixels[i + 3] = 255;
        }
    }
    pixels
}

/// Review metrics: max level, count above 5 levels, mean |Δred| / 255.
pub fn level_diff(master: &[u8], generated: &[u8]) -> LevelDiff {
    assert_eq!(master.len(), generated.len());
    let mut max_level = 0u8;
    let mut over_5 = 0usize;
    let mut acc = 0.0f64;
    let mut pixels = 0usize;
    for (a, b) in master.chunks_exact(4).zip(generated.chunks_exact(4)) {
        let delta = (0..3)
            .map(|lane| a[lane].abs_diff(b[lane]))
            .max()
            .unwrap_or(0);
        max_level = max_level.max(delta);
        over_5 += usize::from(delta > 5);
        acc += f64::from(a[0].abs_diff(b[0])) / 255.0;
        pixels += 1;
    }
    LevelDiff {
        max_level,
        over_5,
        mae: (acc / pixels as f64) as f32,
        pixels,
    }
}

#[test]
fn border_sample_is_inside_radius_and_oob_depth_is_not() {
    let shot = border_shot();
    let vp = glam::Mat4::from_cols_array_2d(&shot.view_proj);
    let inv = vp.inverse();
    let uv = pixel_uv(0, 0);
    let plane = shot.depth[0];
    let center = reconstruct(inv, uv, plane);
    let at_edge = glam::Vec2::new(1.0, uv.y);
    let edge_depth = shot.depth[(W - 1) as usize];
    let clamped = reconstruct(inv, at_edge, edge_depth);
    let oob = reconstruct(inv, at_edge, 0.0);
    let clamped_d = (clamped - center).length();
    assert!(
        clamped_d > EPS && clamped_d < shot.radius,
        "last texel must be a horizon hit, dist {clamped_d} radius {}",
        shot.radius
    );
    assert!(
        (clamped - oob).length() > 0.05,
        "depth 0 at UV=1 must not reconstruct to the last texel"
    );
}

#[test]
fn border_step_lands_on_uv_one() {
    let shot = border_shot();
    let vp = glam::Mat4::from_cols_array_2d(&shot.view_proj);
    let plane = shot.depth[0];
    let sample = border_sample_uv(vp, plane, shot.radius);
    assert!(
        (0.0..1.0).contains(&plane) && plane.is_finite(),
        "plane depth {plane}"
    );
    let edge = *shot.depth.last().expect("depth");
    assert!(
        edge > plane && edge < 1.0,
        "edge depth {edge} vs plane {plane}"
    );
    assert_eq!(
        sample.x.to_bits(),
        1.0f32.to_bits(),
        "step 4 UV x is {sample:?}, radius {}",
        shot.radius
    );
    let index = (sample.x * W as f32) as u32;
    assert_eq!(index, W, "unclamped uv=1 is the out-of-bounds index");
    assert_eq!(index.min(W - 1), W - 1);
}
