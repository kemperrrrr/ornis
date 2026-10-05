//! Split-sum image-based lighting: a roughness-independent BRDF LUT, a
//! prefiltered specular environment (mip `i` is roughness `i / (mips-1)`),
//! and a cosine-weighted irradiance cube.
//!
//! The bake is CPU-side and shared with the GPU upload. With no environment
//! the renderer binds 1×1 black cubes and a weight of zero, so the lighting
//! integral stays the direct-light result.

use ornis_core::units::{Clamped01, Color, UnitVec3};

/// Cube faces in wgpu / GL order: +X, −X, +Y, −Y, +Z, −Z.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CubeFace {
    /// +X.
    PositiveX = 0,
    /// −X.
    NegativeX = 1,
    /// +Y.
    PositiveY = 2,
    /// −Y.
    NegativeY = 3,
    /// +Z.
    PositiveZ = 4,
    /// −Z.
    NegativeZ = 5,
}

impl CubeFace {
    /// All six faces in [`CubeFace`] order.
    pub const ALL: [Self; 6] = [
        Self::PositiveX,
        Self::NegativeX,
        Self::PositiveY,
        Self::NegativeY,
        Self::PositiveZ,
        Self::NegativeZ,
    ];

    /// Layer index in a wgpu cube (0..6).
    pub const fn index(self) -> u32 {
        self as u32
    }
}

/// Scale (R) and bias (G) of the split-sum specular BRDF integral.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitSum {
    scale: f32,
    bias: f32,
}

impl SplitSum {
    /// Fresnel scale term (LUT red).
    pub const fn scale(self) -> f32 {
        self.scale
    }

    /// Fresnel bias term (LUT green).
    pub const fn bias(self) -> f32 {
        self.bias
    }

    fn from_pair(scale: f32, bias: f32) -> Self {
        Self { scale, bias }
    }
}

/// One face of an environment, stored as linear [`Color`] texels.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvironmentCube {
    size: u32,
    faces: [Vec<Color>; 6],
}

impl EnvironmentCube {
    /// Solid cube. `face_size` of 0 becomes 1.
    pub fn solid(color: Color, face_size: u32) -> Self {
        filled(face_size.max(1), color)
    }

    /// Replaces one face with a solid color (same resolution).
    pub fn with_solid_face(mut self, face: CubeFace, color: Color) -> Self {
        self.faces[face.index() as usize] = solid_face(self.size, color);
        self
    }

    /// Texels along one edge.
    pub const fn face_size(&self) -> u32 {
        self.size
    }

    /// Bilinear sample. Directions follow the wgpu cube convention.
    pub fn sample(&self, dir: UnitVec3) -> Color {
        color_from_vec(self.sample_vec(dir.get()))
    }

    fn sample_vec(&self, dir: glam::Vec3) -> glam::Vec3 {
        let hit = face_hit(dir);
        sample_face(self.face_colors(hit.face), self.size, hit.u, hit.v)
    }

    pub(crate) fn face_colors(&self, face: CubeFace) -> &[Color] {
        &self.faces[face.index() as usize]
    }
}

/// Split-sum specular BRDF LUT, row-major, R = scale, G = bias.
#[derive(Debug, Clone, PartialEq)]
pub struct BrdfLut {
    size: u32,
    texels: Vec<SplitSum>,
}

impl BrdfLut {
    /// Edge length in texels.
    pub const fn side(&self) -> u32 {
        self.size
    }

    /// Texel at `(x, y)` where `x` is NoV and `y` is roughness, both in
    /// `[0, 1]` across the texture.
    pub fn texel(&self, x: u32, y: u32) -> SplitSum {
        self.texels[lut_index(self.size, x, y)]
    }
}

/// Integrate the split-sum BRDF at one `(NoV, roughness)` pair.
pub fn integrate_brdf(nov: Clamped01, roughness: Clamped01) -> SplitSum {
    SplitSum::from_pair(
        split_sum_scale(nov.get(), roughness.get()),
        split_sum_bias(nov.get(), roughness.get()),
    )
}

/// Bake a square split-sum LUT. `size` of 0 becomes 1.
pub fn bake_brdf_lut(size: u32) -> BrdfLut {
    let side = size.max(1);
    BrdfLut {
        size: side,
        texels: lut_texels(side),
    }
}

/// Prefiltered specular cube at `roughness` (0 copies the environment).
pub fn convolve_specular(env: &EnvironmentCube, roughness: Clamped01) -> EnvironmentCube {
    filtered_cube(env, env.face_size(), roughness.get())
}

/// Cosine-weighted irradiance cube. A constant environment of color `C`
/// integrates to `C` (`∫ cos θ dω / π = 1`).
pub fn convolve_irradiance(env: &EnvironmentCube) -> EnvironmentCube {
    irradiance_cube(env, env.face_size())
}

/// Mip chain of [`convolve_specular`], mip 0 at roughness 0.
pub(crate) struct SpecularChain {
    mips: Vec<EnvironmentCube>,
}

impl SpecularChain {
    pub(crate) fn from_environment(env: &EnvironmentCube) -> Self {
        Self {
            mips: collect_mips(env),
        }
    }

    pub(crate) fn mips(&self) -> &[EnvironmentCube] {
        &self.mips
    }

    /// Highest mip index (`0` when the chain is a single level).
    pub(crate) fn max_mip(&self) -> f32 {
        max_mip_from_count(self.mips.len() as u32)
    }
}

const SAMPLE_COUNT: u32 = 32;
const SAMPLE_COUNT_F: f32 = 32.0;
const MIRROR_ROUGHNESS: f32 = 1.0e-4;
const TEXEL_CENTER: f32 = 0.5;
const TWO_PI: f32 = std::f32::consts::TAU;
const XI_MAX: f32 = 0.999_999;
const UP_PARALLEL: f32 = 0.999;
const EPS: f32 = 1.0e-6;
const SMITH_FOLD: f32 = 4.0;
const SMITH_HALF: f32 = 0.5;
const FC_POWER: i32 = 5;
const VDC_SHIFT_16: u32 = 16;
const VDC_SHIFT_1: u32 = 1;
const VDC_SHIFT_2: u32 = 2;
const VDC_SHIFT_4: u32 = 4;
const VDC_SHIFT_8: u32 = 8;
const VDC_A: u32 = 0x5555_5555;
const VDC_B: u32 = 0xAAAA_AAAA;
const VDC_C: u32 = 0x3333_3333;
const VDC_D: u32 = 0xCCCC_CCCC;
const VDC_E: u32 = 0x0F0F_0F0F;
const VDC_F: u32 = 0xF0F0_F0F0;
const VDC_G: u32 = 0x00FF_00FF;
const VDC_H: u32 = 0xFF00_FF00;
const VDC_SCALE: f32 = 2.328_306_4e-10;
const RGBA16_BYTES: u32 = 8;
const COPY_ALIGN: u32 = 256;
const LUT_SIDE: u32 = 32;
const CUBE_LAYERS: u32 = 6;
const F16_SIGN_SHIFT: u32 = 16;
const F16_SIGN_MASK: u32 = 0x8000;
const F32_EXP_SHIFT: u32 = 23;
const F32_EXP_MASK: u32 = 0xff;
const F32_MANT_MASK: u32 = 0x007f_ffff;
const F32_EXP_BIAS: i32 = 127;
const F16_EXP_BIAS: i32 = 15;
const F16_EXP_MAX: i32 = 31;
const F16_INF_BITS: u32 = 0x7c00;
const F16_NAN_BITS: u32 = 0x7e00;
const F16_EXP_SHIFT: i32 = 10;
const F16_MANT_SHIFT: u32 = 13;
const F16_ROUND: u32 = 0x1000;
const F16_ROUND_MASK: u32 = 0x1fff;
const F32_IMPLICIT: u32 = 0x0080_0000;
const SUBNORMAL_CUTOFF: i32 = -10;
const ALPHA_CHANNEL: usize = 3;
const CHAN_BYTES: usize = 2;
const PACKED_RGBA: usize = 8;

const fn align_row(tight: u32) -> u32 {
    let rem = tight % COPY_ALIGN;
    if rem == 0 {
        tight
    } else {
        tight + (COPY_ALIGN - rem)
    }
}

const fn lut_row_stride() -> u32 {
    align_row(LUT_SIDE * RGBA16_BYTES)
}

const fn face_row_stride() -> u32 {
    align_row(RGBA16_BYTES)
}

/// Byte length of the baked BRDF LUT in the initial staging buffer.
pub(crate) const fn lut_staging_bytes() -> u64 {
    lut_row_stride() as u64 * LUT_SIDE as u64
}

/// Byte length of one 1×1 black cube face in the staging buffer.
pub(crate) const fn black_face_bytes() -> u64 {
    face_row_stride() as u64
}

fn filled(size: u32, color: Color) -> EnvironmentCube {
    let face = solid_face(size, color);
    EnvironmentCube {
        size,
        faces: std::array::from_fn(|_| face.clone()),
    }
}

fn solid_face(size: u32, color: Color) -> Vec<Color> {
    vec![color; face_len(size)]
}

fn face_len(size: u32) -> usize {
    size as usize * size as usize
}

fn color_from_vec(v: glam::Vec3) -> Color {
    Color::linear_rgb(v.x, v.y, v.z)
}

fn color_to_vec(color: Color) -> glam::Vec3 {
    let rgb = color.to_linear_rgb().as_array();
    glam::Vec3::new(rgb[0], rgb[1], rgb[2])
}

struct FaceHit {
    face: CubeFace,
    u: f32,
    v: f32,
}

fn face_hit(dir: glam::Vec3) -> FaceHit {
    match major_axis(dir) {
        0 => hit_x(dir),
        1 => hit_y(dir),
        _ => hit_z(dir),
    }
}

fn major_axis(dir: glam::Vec3) -> u32 {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();
    if ax >= ay && ax >= az {
        0
    } else if ay >= az {
        1
    } else {
        2
    }
}

fn hit_x(dir: glam::Vec3) -> FaceHit {
    let ax = dir.x.abs().max(EPS);
    if dir.x >= 0.0 {
        FaceHit {
            face: CubeFace::PositiveX,
            u: -dir.z / ax,
            v: -dir.y / ax,
        }
    } else {
        FaceHit {
            face: CubeFace::NegativeX,
            u: dir.z / ax,
            v: -dir.y / ax,
        }
    }
}

fn hit_y(dir: glam::Vec3) -> FaceHit {
    let ay = dir.y.abs().max(EPS);
    if dir.y >= 0.0 {
        FaceHit {
            face: CubeFace::PositiveY,
            u: dir.x / ay,
            v: dir.z / ay,
        }
    } else {
        FaceHit {
            face: CubeFace::NegativeY,
            u: dir.x / ay,
            v: -dir.z / ay,
        }
    }
}

fn hit_z(dir: glam::Vec3) -> FaceHit {
    let az = dir.z.abs().max(EPS);
    if dir.z >= 0.0 {
        FaceHit {
            face: CubeFace::PositiveZ,
            u: dir.x / az,
            v: -dir.y / az,
        }
    } else {
        FaceHit {
            face: CubeFace::NegativeZ,
            u: -dir.x / az,
            v: -dir.y / az,
        }
    }
}

fn sample_face(texels: &[Color], size: u32, u: f32, v: f32) -> glam::Vec3 {
    let c = bilinear_coords(size, u, v);
    blend_bilinear(
        texel_vec(texels, size, c.x0, c.y0),
        texel_vec(texels, size, c.x1, c.y0),
        texel_vec(texels, size, c.x0, c.y1),
        texel_vec(texels, size, c.x1, c.y1),
        c.tx,
        c.ty,
    )
}

struct Bilinear {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    tx: f32,
    ty: f32,
}

fn bilinear_coords(size: u32, u: f32, v: f32) -> Bilinear {
    let x = texel_index(size, u);
    let y = texel_index(size, v);
    Bilinear {
        x0: x.index0,
        y0: y.index0,
        x1: x.index1,
        y1: y.index1,
        tx: x.frac,
        ty: y.frac,
    }
}

struct AxisSample {
    index0: u32,
    index1: u32,
    frac: f32,
}

fn texel_index(size: u32, t: f32) -> AxisSample {
    let max_i = size - 1;
    let p = ((t + 1.0) * TEXEL_CENTER * size as f32 - TEXEL_CENTER).clamp(0.0, max_i as f32);
    let i0 = p.floor() as u32;
    let i1 = (i0 + 1).min(max_i);
    AxisSample {
        index0: i0,
        index1: i1,
        frac: p - i0 as f32,
    }
}

fn blend_bilinear(
    c00: glam::Vec3,
    c10: glam::Vec3,
    c01: glam::Vec3,
    c11: glam::Vec3,
    tx: f32,
    ty: f32,
) -> glam::Vec3 {
    let a = c00 * (1.0 - tx) + c10 * tx;
    let b = c01 * (1.0 - tx) + c11 * tx;
    a * (1.0 - ty) + b * ty
}

fn texel_vec(texels: &[Color], size: u32, x: u32, y: u32) -> glam::Vec3 {
    color_to_vec(texels[index_of(size, x, y)])
}

fn index_of(size: u32, x: u32, y: u32) -> usize {
    (y * size + x) as usize
}

fn lut_texels(size: u32) -> Vec<SplitSum> {
    let mut texels = Vec::new();
    for y in 0..size {
        push_lut_row(&mut texels, size, y);
    }
    texels
}

fn push_lut_row(texels: &mut Vec<SplitSum>, size: u32, y: u32) {
    for x in 0..size {
        texels.push(integrate_at(lut_nov(x, size), lut_roughness(y, size)));
    }
}

fn lut_index(size: u32, x: u32, y: u32) -> usize {
    index_of(size, x, y)
}

fn lut_nov(x: u32, size: u32) -> f32 {
    (x as f32 + TEXEL_CENTER) / size as f32
}

fn lut_roughness(y: u32, size: u32) -> f32 {
    lut_nov(y, size)
}

fn integrate_at(nov: f32, roughness: f32) -> SplitSum {
    SplitSum::from_pair(
        split_sum_scale(nov, roughness),
        split_sum_bias(nov, roughness),
    )
}

fn split_sum_scale(nov: f32, roughness: f32) -> f32 {
    reduce_scale(&gather_split(nov, roughness))
}

fn split_sum_bias(nov: f32, roughness: f32) -> f32 {
    reduce_bias(&gather_split(nov, roughness))
}

struct SplitSample {
    scale: f32,
    bias: f32,
}

fn gather_split(nov: f32, roughness: f32) -> Vec<SplitSample> {
    let mut samples = Vec::new();
    for index in 0..SAMPLE_COUNT {
        samples.push(split_sample(nov, roughness, index));
    }
    samples
}

fn split_sample(nov: f32, roughness: f32, index: u32) -> SplitSample {
    mirror_or_sample(roughness, sample_contribution(nov, roughness, index))
}

fn mirror_or_sample(roughness: f32, sample: SplitSample) -> SplitSample {
    if roughness <= MIRROR_ROUGHNESS {
        SplitSample {
            scale: 1.0,
            bias: 0.0,
        }
    } else {
        sample
    }
}

fn sample_contribution(nov: f32, roughness: f32, index: u32) -> SplitSample {
    contribution_from(nov, roughness, half_vector_z(roughness, index))
}

fn half_vector_z(roughness: f32, index: u32) -> glam::Vec3 {
    ggx_half(roughness, hammersley_x(index), radical_inverse(index))
}

fn hammersley_x(index: u32) -> f32 {
    index as f32 / SAMPLE_COUNT_F
}

fn radical_inverse(mut bits: u32) -> f32 {
    bits = (bits << VDC_SHIFT_16) | (bits >> VDC_SHIFT_16);
    bits = ((bits & VDC_A) << VDC_SHIFT_1) | ((bits & VDC_B) >> VDC_SHIFT_1);
    bits = ((bits & VDC_C) << VDC_SHIFT_2) | ((bits & VDC_D) >> VDC_SHIFT_2);
    bits = ((bits & VDC_E) << VDC_SHIFT_4) | ((bits & VDC_F) >> VDC_SHIFT_4);
    bits = ((bits & VDC_G) << VDC_SHIFT_8) | ((bits & VDC_H) >> VDC_SHIFT_8);
    bits as f32 * VDC_SCALE
}

fn ggx_half(roughness: f32, xi_x: f32, xi_y: f32) -> glam::Vec3 {
    let a = roughness * roughness;
    let a2 = a * a;
    let phi = TWO_PI * xi_x;
    let xi = xi_y.min(XI_MAX);
    let cos_theta = ((1.0 - xi) / (1.0 + (a2 - 1.0) * xi)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    glam::Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

fn contribution_from(nov: f32, roughness: f32, h: glam::Vec3) -> SplitSample {
    let alpha = roughness * roughness;
    let vx = (1.0 - nov * nov).max(0.0).sqrt();
    let voh = (vx * h.x + nov * h.z).max(0.0);
    let nol = (2.0 * voh * h.z - nov).max(0.0);
    if nol <= 0.0 {
        return SplitSample {
            scale: 0.0,
            bias: 0.0,
        };
    }
    let noh = h.z.max(EPS);
    let vis = smith_v(nov, nol, alpha);
    let g_vis = vis * SMITH_FOLD * nol * voh / noh;
    let fc = (1.0 - voh).powi(FC_POWER);
    SplitSample {
        scale: (1.0 - fc) * g_vis,
        bias: fc * g_vis,
    }
}

fn smith_v(nov: f32, nol: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let ggxv = nov * (nol * nol * (1.0 - a2) + a2).max(EPS).sqrt();
    let ggxl = nol * (nov * nov * (1.0 - a2) + a2).max(EPS).sqrt();
    SMITH_HALF / (ggxv + ggxl).max(EPS)
}

fn reduce_scale(samples: &[SplitSample]) -> f32 {
    let mut scale = 0.0;
    for sample in samples {
        scale += sample.scale;
    }
    scale / SAMPLE_COUNT_F
}

fn reduce_bias(samples: &[SplitSample]) -> f32 {
    let mut bias = 0.0;
    for sample in samples {
        bias += sample.bias;
    }
    bias / SAMPLE_COUNT_F
}

fn collect_mips(env: &EnvironmentCube) -> Vec<EnvironmentCube> {
    let mut mips = Vec::new();
    for index in 0..mip_count(env.face_size()) {
        mips.push(prefilter_level(env, index));
    }
    mips
}

fn mip_count(size: u32) -> u32 {
    let mut count = 1u32;
    let mut side = size.max(1);
    while side > 1 {
        side /= 2;
        count += 1;
    }
    count
}

fn prefilter_level(env: &EnvironmentCube, index: u32) -> EnvironmentCube {
    let levels = mip_count(env.face_size());
    filtered_cube(
        env,
        level_size(env.face_size(), index),
        level_roughness(index, levels),
    )
}

fn level_size(base: u32, index: u32) -> u32 {
    (base >> index).max(1)
}

fn level_roughness(index: u32, levels: u32) -> f32 {
    if levels <= 1 {
        0.0
    } else {
        index as f32 / (levels - 1) as f32
    }
}

fn max_mip_from_count(count: u32) -> f32 {
    if count <= 1 { 0.0 } else { (count - 1) as f32 }
}

fn filtered_cube(env: &EnvironmentCube, size: u32, roughness: f32) -> EnvironmentCube {
    EnvironmentCube {
        size,
        faces: filtered_faces(env, size, roughness),
    }
}

fn filtered_faces(env: &EnvironmentCube, size: u32, roughness: f32) -> [Vec<Color>; 6] {
    std::array::from_fn(|face| filter_face(env, CubeFace::ALL[face], size, roughness))
}

fn filter_face(env: &EnvironmentCube, face: CubeFace, size: u32, roughness: f32) -> Vec<Color> {
    let mut texels = Vec::new();
    for y in 0..size {
        push_filtered_row(&mut texels, env, face, size, y, roughness);
    }
    texels
}

fn push_filtered_row(
    texels: &mut Vec<Color>,
    env: &EnvironmentCube,
    face: CubeFace,
    size: u32,
    y: u32,
    roughness: f32,
) {
    for x in 0..size {
        texels.push(filter_texel(env, face, size, x, y, roughness));
    }
}

fn filter_texel(
    env: &EnvironmentCube,
    face: CubeFace,
    size: u32,
    x: u32,
    y: u32,
    roughness: f32,
) -> Color {
    let dir = texel_direction(face, size, x, y);
    color_from_vec(filter_direction(env, dir, roughness))
}

fn texel_direction(face: CubeFace, size: u32, x: u32, y: u32) -> glam::Vec3 {
    face_direction(face, axis_coord(x, size), axis_coord(y, size))
}

fn axis_coord(i: u32, size: u32) -> f32 {
    (i as f32 + TEXEL_CENTER) / size as f32 * 2.0 - 1.0
}

fn face_direction(face: CubeFace, u: f32, v: f32) -> glam::Vec3 {
    match face {
        CubeFace::PositiveX => glam::Vec3::new(1.0, -v, -u),
        CubeFace::NegativeX => glam::Vec3::new(-1.0, -v, u),
        CubeFace::PositiveY => glam::Vec3::new(u, 1.0, v),
        CubeFace::NegativeY => glam::Vec3::new(u, -1.0, -v),
        CubeFace::PositiveZ => glam::Vec3::new(u, -v, 1.0),
        CubeFace::NegativeZ => glam::Vec3::new(-u, -v, -1.0),
    }
}

fn filter_direction(env: &EnvironmentCube, dir: glam::Vec3, roughness: f32) -> glam::Vec3 {
    reduce_weighted(&gather_prefilter(env, dir, roughness))
}

struct WeightedRgb {
    rgb: glam::Vec3,
    weight: f32,
}

fn gather_prefilter(env: &EnvironmentCube, n: glam::Vec3, roughness: f32) -> Vec<WeightedRgb> {
    let mut samples = Vec::new();
    for index in 0..SAMPLE_COUNT {
        samples.push(prefilter_sample(env, n, roughness, index));
    }
    samples
}

fn prefilter_sample(
    env: &EnvironmentCube,
    n: glam::Vec3,
    roughness: f32,
    index: u32,
) -> WeightedRgb {
    let l = sample_direction(n, roughness, index);
    WeightedRgb {
        rgb: env.sample_vec(l),
        weight: direction_weight(n, l),
    }
}

fn sample_direction(n: glam::Vec3, roughness: f32, index: u32) -> glam::Vec3 {
    let h = world_from_local(
        n,
        ggx_half(roughness, hammersley_x(index), radical_inverse(index)),
    );
    reflect_vec(n, h)
}

fn reflect_vec(n: glam::Vec3, h: glam::Vec3) -> glam::Vec3 {
    let voh = n.x * h.x + n.y * h.y + n.z * h.z;
    glam::Vec3::new(
        h.x * voh * 2.0 - n.x,
        h.y * voh * 2.0 - n.y,
        h.z * voh * 2.0 - n.z,
    )
}

fn direction_weight(n: glam::Vec3, l: glam::Vec3) -> f32 {
    (n.x * l.x + n.y * l.y + n.z * l.z).max(0.0)
}

fn reduce_weighted(samples: &[WeightedRgb]) -> glam::Vec3 {
    scale_by_weight(fold_rgb(samples), fold_weight(samples))
}

fn fold_rgb(samples: &[WeightedRgb]) -> glam::Vec3 {
    let mut sum = glam::Vec3::ZERO;
    for sample in samples {
        sum += sample.rgb * sample.weight;
    }
    sum
}

fn fold_weight(samples: &[WeightedRgb]) -> f32 {
    let mut weight = 0.0;
    for sample in samples {
        weight += sample.weight;
    }
    weight
}

fn scale_by_weight(sum: glam::Vec3, weight: f32) -> glam::Vec3 {
    if weight <= 0.0 {
        glam::Vec3::ZERO
    } else {
        sum * (1.0 / weight)
    }
}

fn world_from_local(n: glam::Vec3, local: glam::Vec3) -> glam::Vec3 {
    let parallel = n.z.abs() >= UP_PARALLEL;
    let up_x = if parallel { 1.0 } else { 0.0 };
    let up_z = if parallel { 0.0 } else { 1.0 };
    let cx = -up_z * n.y;
    let cy = up_z * n.x - up_x * n.z;
    let cz = up_x * n.y;
    let t_len = (cx * cx + cy * cy + cz * cz).sqrt().max(EPS);
    let tx = cx / t_len;
    let ty = cy / t_len;
    let tz = cz / t_len;
    let bx = n.y * tz - n.z * ty;
    let by = n.z * tx - n.x * tz;
    let bz = n.x * ty - n.y * tx;
    glam::Vec3::new(
        tx * local.x + bx * local.y + n.x * local.z,
        ty * local.x + by * local.y + n.y * local.z,
        tz * local.x + bz * local.y + n.z * local.z,
    )
}

fn irradiance_cube(env: &EnvironmentCube, size: u32) -> EnvironmentCube {
    EnvironmentCube {
        size,
        faces: std::array::from_fn(|face| irradiance_face(env, CubeFace::ALL[face], size)),
    }
}

fn irradiance_face(env: &EnvironmentCube, face: CubeFace, size: u32) -> Vec<Color> {
    let mut texels = Vec::new();
    for y in 0..size {
        push_irradiance_row(&mut texels, env, face, size, y);
    }
    texels
}

fn push_irradiance_row(
    texels: &mut Vec<Color>,
    env: &EnvironmentCube,
    face: CubeFace,
    size: u32,
    y: u32,
) {
    for x in 0..size {
        texels.push(irradiance_texel(env, face, size, x, y));
    }
}

fn irradiance_texel(env: &EnvironmentCube, face: CubeFace, size: u32, x: u32, y: u32) -> Color {
    let dir = texel_direction(face, size, x, y);
    color_from_vec(reduce_irradiance(&gather_irradiance(env, dir)))
}

fn gather_irradiance(env: &EnvironmentCube, n: glam::Vec3) -> Vec<glam::Vec3> {
    let mut samples = Vec::new();
    for index in 0..SAMPLE_COUNT {
        samples.push(irradiance_sample(env, n, index));
    }
    samples
}

fn irradiance_sample(env: &EnvironmentCube, n: glam::Vec3, index: u32) -> glam::Vec3 {
    env.sample_vec(cosine_direction(n, index))
}

fn cosine_direction(n: glam::Vec3, index: u32) -> glam::Vec3 {
    world_from_local(n, cosine_local(index))
}

fn cosine_local(index: u32) -> glam::Vec3 {
    let phi = TWO_PI * hammersley_x(index);
    let cos_theta = radical_inverse(index).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    glam::Vec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

fn reduce_irradiance(samples: &[glam::Vec3]) -> glam::Vec3 {
    let mut sum = glam::Vec3::ZERO;
    for sample in samples {
        sum += *sample;
    }
    sum / SAMPLE_COUNT_F
}

/// RGBA16F bytes for one cube face, rows padded to the copy alignment.
pub(crate) fn face_bytes(colors: &[Color], size: u32) -> Vec<u8> {
    let mut bytes = zero_bytes(face_byte_len(size));
    fill_face(&mut bytes, colors, size);
    bytes
}

/// RGBA16F bytes for the BRDF LUT (R = scale, G = bias).
pub(crate) fn lut_bytes(lut: &BrdfLut) -> Vec<u8> {
    let mut bytes = zero_bytes(lut_byte_len(lut.side()));
    fill_lut(&mut bytes, lut);
    bytes
}

fn face_byte_len(size: u32) -> usize {
    row_stride(size) as usize * size as usize
}

fn lut_byte_len(size: u32) -> usize {
    face_byte_len(size)
}

fn zero_bytes(len: usize) -> Vec<u8> {
    vec![0u8; len]
}

fn row_stride(width: u32) -> u32 {
    align_row(tight_row(width))
}

fn tight_row(width: u32) -> u32 {
    width * RGBA16_BYTES
}

fn fill_face(bytes: &mut [u8], colors: &[Color], size: u32) {
    for y in 0..size {
        fill_color_row(bytes, colors, size, y);
    }
}

fn fill_color_row(bytes: &mut [u8], colors: &[Color], size: u32, y: u32) {
    for x in 0..size {
        write_color(bytes, colors, size, x, y);
    }
}

fn write_color(bytes: &mut [u8], colors: &[Color], size: u32, x: u32, y: u32) {
    copy_packed(
        bytes,
        texel_offset(row_stride(size), x, y),
        pack_color(texel_at(colors, size, x, y)),
    );
}

fn texel_at(colors: &[Color], size: u32, x: u32, y: u32) -> Color {
    colors[index_of(size, x, y)]
}

fn texel_offset(stride: u32, x: u32, y: u32) -> usize {
    y as usize * stride as usize + x as usize * RGBA16_BYTES as usize
}

fn fill_lut(bytes: &mut [u8], lut: &BrdfLut) {
    for y in 0..lut.side() {
        fill_lut_row(bytes, lut, y);
    }
}

fn fill_lut_row(bytes: &mut [u8], lut: &BrdfLut, y: u32) {
    for x in 0..lut.side() {
        write_split(bytes, lut, x, y);
    }
}

fn write_split(bytes: &mut [u8], lut: &BrdfLut, x: u32, y: u32) {
    copy_packed(
        bytes,
        texel_offset(row_stride(lut.side()), x, y),
        pack_split(lut.texel(x, y)),
    )
}

fn pack_color(color: Color) -> [u8; 8] {
    pack_rgb(color.to_linear_rgb().as_array())
}

fn pack_rgb(rgb: [f32; 3]) -> [u8; PACKED_RGBA] {
    let mut out = [0u8; PACKED_RGBA];
    put_channel(&mut out, 0, rgb[0]);
    put_channel(&mut out, 1, rgb[1]);
    put_channel(&mut out, 2, rgb[2]);
    put_channel(&mut out, ALPHA_CHANNEL, 1.0);
    out
}

fn pack_split(sum: SplitSum) -> [u8; PACKED_RGBA] {
    let mut out = [0u8; PACKED_RGBA];
    put_channel(&mut out, 0, sum.scale());
    put_channel(&mut out, 1, sum.bias());
    put_channel(&mut out, 2, 0.0);
    put_channel(&mut out, ALPHA_CHANNEL, 1.0);
    out
}

fn put_channel(out: &mut [u8; PACKED_RGBA], channel: usize, value: f32) {
    write_le(out, channel_offset(channel), f32_to_f16(value));
}

fn channel_offset(channel: usize) -> usize {
    channel * CHAN_BYTES
}

fn write_le(out: &mut [u8; PACKED_RGBA], offset: usize, bits: u16) {
    let bytes = bits.to_le_bytes();
    out[offset] = bytes[0];
    out[offset + 1] = bytes[1];
}

fn copy_packed(bytes: &mut [u8], offset: usize, packed: [u8; PACKED_RGBA]) {
    bytes[offset..offset + packed.len()].copy_from_slice(&packed);
}

/// Round a finite `f32` to IEEE binary16 (1.0 → `0x3C00`).
pub(crate) fn f32_to_f16(value: f32) -> u16 {
    pack_f16(value.to_bits())
}

fn pack_f16(bits: u32) -> u16 {
    let sign = f16_sign(bits);
    match exponent_class(bits) {
        F16Class::Zero => sign,
        F16Class::Inf => sign | F16_INF_BITS as u16,
        F16Class::Nan => sign | F16_NAN_BITS as u16,
        F16Class::Overflow => sign | F16_INF_BITS as u16,
        F16Class::Subnormal => subnormal_bits(sign, bits),
        F16Class::Normal => normal_bits(sign, bits),
    }
}

#[derive(Clone, Copy)]
enum F16Class {
    Zero,
    Inf,
    Nan,
    Overflow,
    Subnormal,
    Normal,
}

fn f16_sign(bits: u32) -> u16 {
    ((bits >> F16_SIGN_SHIFT) & F16_SIGN_MASK) as u16
}

fn exponent_class(bits: u32) -> F16Class {
    let exp = f32_exp(bits);
    let half_exp = exp - F32_EXP_BIAS + F16_EXP_BIAS;
    if exp == F32_EXP_MASK as i32 {
        nan_or_inf(bits)
    } else if half_exp >= F16_EXP_MAX {
        F16Class::Overflow
    } else if half_exp <= 0 {
        zero_or_sub(half_exp)
    } else {
        F16Class::Normal
    }
}

fn f32_exp(bits: u32) -> i32 {
    ((bits >> F32_EXP_SHIFT) & F32_EXP_MASK) as i32
}

fn nan_or_inf(bits: u32) -> F16Class {
    if bits & F32_MANT_MASK == 0 {
        F16Class::Inf
    } else {
        F16Class::Nan
    }
}

fn zero_or_sub(half_exp: i32) -> F16Class {
    if half_exp < SUBNORMAL_CUTOFF {
        F16Class::Zero
    } else {
        F16Class::Subnormal
    }
}

fn subnormal_bits(sign: u16, bits: u32) -> u16 {
    let half_exp = f32_exp(bits) - F32_EXP_BIAS + F16_EXP_BIAS;
    let shift = (1 - half_exp) as u32;
    let mantissa = (bits & F32_MANT_MASK) | F32_IMPLICIT;
    let rounded = mantissa >> (shift + F16_MANT_SHIFT);
    sign | rounded as u16
}

fn normal_bits(sign: u16, bits: u32) -> u16 {
    let half_exp = (f32_exp(bits) - F32_EXP_BIAS + F16_EXP_BIAS) as u32;
    let mantissa = bits & F32_MANT_MASK;
    let mut half_mant = mantissa >> F16_MANT_SHIFT;
    let remainder = mantissa & F16_ROUND_MASK;
    if remainder > F16_ROUND || (remainder == F16_ROUND && (half_mant & 1) == 1) {
        half_mant += 1;
    }
    sign | (half_exp << F16_EXP_SHIFT as u32) as u16 | half_mant as u16
}

/// GPU images for one IBL set: prefiltered specular, irradiance, BRDF LUT.
pub(crate) struct IblTargets {
    pub prefilter: wgpu::Texture,
    pub prefilter_view: wgpu::TextureView,
    pub irradiance: wgpu::Texture,
    pub irradiance_view: wgpu::TextureView,
    pub lut: wgpu::Texture,
    pub lut_view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    /// Weight written into [`crate::renderer::LightingUniform`].
    pub weight: f32,
    /// Highest prefilter mip, written next to `weight`.
    pub max_mip: f32,
}

impl IblTargets {
    /// Bind one of the IBL resource names, or `None` for anything else.
    pub(crate) fn binding(&self, name: &str) -> Option<wgpu::BindingResource<'_>> {
        match name {
            "prefilter_cube" => Some(wgpu::BindingResource::TextureView(&self.prefilter_view)),
            "irradiance_cube" => Some(wgpu::BindingResource::TextureView(&self.irradiance_view)),
            "brdf_lut" => Some(wgpu::BindingResource::TextureView(&self.lut_view)),
            "ibl_sampler" => Some(wgpu::BindingResource::Sampler(&self.sampler)),
            _ => None,
        }
    }
}

/// The `black` constructor above cannot see the textures it just created
/// when building views. Views are created by these helpers from the
/// textures after they exist — see [`IblTargets::attach_views`].
impl IblTargets {
    pub(crate) fn attach_views(mut self) -> Self {
        self.prefilter_view = self.prefilter.create_view(&cube_view_desc());
        self.irradiance_view = self.irradiance.create_view(&cube_view_desc());
        self.lut_view = self
            .lut
            .create_view(&wgpu::TextureViewDescriptor::default());
        self
    }
}

/// Replace the placeholder views. `black` builds textures first.
pub(crate) fn black_targets(device: &wgpu::Device) -> IblTargets {
    IblTargets {
        prefilter: create_cube(device, "ibl prefilter", 1, 1),
        prefilter_view: placeholder_view(device),
        irradiance: create_cube(device, "ibl irradiance", 1, 1),
        irradiance_view: placeholder_view(device),
        lut: create_lut(device),
        lut_view: placeholder_view(device),
        sampler: ibl_sampler(device),
        weight: 0.0,
        max_mip: 0.0,
    }
    .attach_views()
}

fn placeholder_view(device: &wgpu::Device) -> wgpu::TextureView {
    create_lut(device).create_view(&wgpu::TextureViewDescriptor::default())
}

/// Upload `env` (or black, when `None`) and return textures the shader can sample.
pub(crate) fn upload_targets(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    env: Option<&EnvironmentCube>,
) -> IblTargets {
    match env {
        None => upload_black(device, queue),
        Some(env) => upload_env(device, queue, env),
    }
}

fn upload_black(device: &wgpu::Device, queue: &wgpu::Queue) -> IblTargets {
    let targets = black_targets(device);
    let lut = bake_brdf_lut(LUT_SIDE);
    queue.write_texture(
        lut_copy(&targets.lut),
        &lut_bytes(&lut),
        lut_layout(),
        lut_extent(),
    );
    write_black_cube(queue, &targets.prefilter);
    write_black_cube(queue, &targets.irradiance);
    targets
}

fn upload_env(device: &wgpu::Device, queue: &wgpu::Queue, env: &EnvironmentCube) -> IblTargets {
    let chain = SpecularChain::from_environment(env);
    let irradiance = convolve_irradiance(env);
    let mips = chain.mips().len() as u32;
    let targets = IblTargets {
        prefilter: create_cube(device, "ibl prefilter", env.face_size(), mips),
        prefilter_view: placeholder_view(device),
        irradiance: create_cube(device, "ibl irradiance", irradiance.face_size(), 1),
        irradiance_view: placeholder_view(device),
        lut: create_lut(device),
        lut_view: placeholder_view(device),
        sampler: ibl_sampler(device),
        weight: 1.0,
        max_mip: chain.max_mip(),
    }
    .attach_views();
    write_chain(queue, &targets.prefilter, chain.mips());
    write_cube_mip(queue, &targets.irradiance, &irradiance, 0);
    let lut = bake_brdf_lut(LUT_SIDE);
    queue.write_texture(
        lut_copy(&targets.lut),
        &lut_bytes(&lut),
        lut_layout(),
        lut_extent(),
    );
    targets
}

fn write_black_cube(queue: &wgpu::Queue, texture: &wgpu::Texture) {
    let black = EnvironmentCube::solid(Color::BLACK, 1);
    write_cube_mip(queue, texture, &black, 0);
}

fn write_chain(queue: &wgpu::Queue, texture: &wgpu::Texture, mips: &[EnvironmentCube]) {
    for (level, cube) in mips.iter().enumerate() {
        write_cube_mip(queue, texture, cube, level as u32);
    }
}

fn write_cube_mip(queue: &wgpu::Queue, texture: &wgpu::Texture, cube: &EnvironmentCube, mip: u32) {
    for face in CubeFace::ALL {
        queue.write_texture(
            face_copy(texture, mip, face),
            &face_bytes(cube.face_colors(face), cube.face_size()),
            face_layout(cube.face_size()),
            face_extent(cube.face_size()),
        );
    }
}

fn create_cube(device: &wgpu::Device, label: &'static str, size: u32, mips: u32) -> wgpu::Texture {
    device.create_texture(&cube_descriptor(label, size, mips))
}

fn create_lut(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&lut_descriptor())
}

fn cube_descriptor(label: &'static str, size: u32, mips: u32) -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some(label),
        size: cube_extent(size),
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: cube_usage(),
        view_formats: &[],
    }
}

fn lut_descriptor() -> wgpu::TextureDescriptor<'static> {
    wgpu::TextureDescriptor {
        label: Some("ibl brdf lut"),
        size: wgpu::Extent3d {
            width: LUT_SIDE,
            height: LUT_SIDE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: cube_usage(),
        view_formats: &[],
    }
}

fn cube_extent(size: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: CUBE_LAYERS,
    }
}

fn cube_usage() -> wgpu::TextureUsages {
    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST
}

fn cube_view_desc() -> wgpu::TextureViewDescriptor<'static> {
    wgpu::TextureViewDescriptor {
        label: Some("ibl cube"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    }
}

fn ibl_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&ibl_sampler_desc())
}

fn ibl_sampler_desc() -> wgpu::SamplerDescriptor<'static> {
    wgpu::SamplerDescriptor {
        label: Some("ibl sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    }
}

fn face_copy(texture: &wgpu::Texture, mip: u32, face: CubeFace) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: mip,
        origin: wgpu::Origin3d {
            x: 0,
            y: 0,
            z: face.index(),
        },
        aspect: wgpu::TextureAspect::All,
    }
}

fn face_layout(size: u32) -> wgpu::TexelCopyBufferLayout {
    wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(row_stride(size)),
        rows_per_image: Some(size),
    }
}

fn face_extent(size: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: size,
        height: size,
        depth_or_array_layers: 1,
    }
}

fn lut_copy(texture: &wgpu::Texture) -> wgpu::TexelCopyTextureInfo<'_> {
    wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    }
}

fn lut_layout() -> wgpu::TexelCopyBufferLayout {
    face_layout(LUT_SIDE)
}

fn lut_extent() -> wgpu::Extent3d {
    face_extent(LUT_SIDE)
}

/// Initial staging bytes: the real LUT, then six 1×1 black faces.
pub(crate) fn initial_staging_bytes() -> Vec<u8> {
    let mut bytes = lut_bytes(&bake_brdf_lut(LUT_SIDE));
    append_black_faces(&mut bytes);
    bytes
}

fn append_black_faces(bytes: &mut Vec<u8>) {
    for _face in CubeFace::ALL {
        bytes.extend(face_bytes(
            EnvironmentCube::solid(Color::BLACK, 1).face_colors(CubeFace::PositiveX),
            1,
        ));
    }
}

/// Record copies that fill the black IBL textures from [`initial_staging_bytes`].
pub(crate) fn encode_initial_upload(
    encoder: &mut wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    targets: &IblTargets,
) {
    encoder.copy_buffer_to_texture(lut_src(staging), lut_dst(&targets.lut), lut_extent());
    copy_black_faces(encoder, staging, &targets.prefilter);
    copy_black_faces(encoder, staging, &targets.irradiance);
}

fn lut_src(staging: &wgpu::Buffer) -> wgpu::TexelCopyBufferInfo<'_> {
    wgpu::TexelCopyBufferInfo {
        buffer: staging,
        layout: wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(lut_row_stride()),
            rows_per_image: Some(LUT_SIDE),
        },
    }
}

fn lut_dst(texture: &wgpu::Texture) -> wgpu::TexelCopyTextureInfo<'_> {
    lut_copy(texture)
}

fn copy_black_faces(
    encoder: &mut wgpu::CommandEncoder,
    staging: &wgpu::Buffer,
    texture: &wgpu::Texture,
) {
    for face in CubeFace::ALL {
        encoder.copy_buffer_to_texture(
            black_src(staging, face),
            face_copy(texture, 0, face),
            face_extent(1),
        );
    }
}

fn black_src(staging: &wgpu::Buffer, face: CubeFace) -> wgpu::TexelCopyBufferInfo<'_> {
    wgpu::TexelCopyBufferInfo {
        buffer: staging,
        layout: wgpu::TexelCopyBufferLayout {
            offset: black_offset(face),
            bytes_per_row: Some(face_row_stride()),
            rows_per_image: Some(1),
        },
    }
}

fn black_offset(face: CubeFace) -> u64 {
    staged_black_offset(face.index())
}

fn staged_black_offset(index: u32) -> u64 {
    lut_staging_bytes() + index as u64 * black_face_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roughness_zero_split_sum_is_mirror() {
        let sum = integrate_brdf(Clamped01::ONE, Clamped01::ZERO);
        assert!((sum.scale() - 1.0).abs() < 1.0e-5, "{}", sum.scale());
        assert!(sum.bias().abs() < 1.0e-5, "{}", sum.bias());
    }

    #[test]
    fn rough_split_sum_stays_in_range_and_scale_falls() {
        let rough = integrate_brdf(Clamped01::ONE, Clamped01::ONE);
        assert!(rough.scale().is_finite() && rough.bias().is_finite());
        assert!(
            (0.3..0.9).contains(&rough.scale()),
            "scale {}",
            rough.scale()
        );
        assert!((0.0..0.35).contains(&rough.bias()), "bias {}", rough.bias());
        let mid = integrate_brdf(Clamped01::ONE, Clamped01::new(0.25));
        assert!(
            mid.scale() > rough.scale(),
            "mid {} rough {}",
            mid.scale(),
            rough.scale()
        );
    }

    #[test]
    fn cube_directions_round_trip_and_faces_do_not_bleed() {
        let env = EnvironmentCube::solid(Color::BLACK, 4)
            .with_solid_face(CubeFace::PositiveZ, Color::linear_rgb(1.0, 0.0, 0.0));
        let red = env.sample(UnitVec3::Z);
        let black =
            env.sample(UnitVec3::new(glam::Vec3::new(0.0, 0.0, -1.0)).unwrap_or(UnitVec3::Z));
        let rgb = red.to_linear_rgb().as_array();
        assert!(rgb[0] > 0.9 && rgb[1] < 0.05, "{rgb:?}");
        assert!(black.to_linear_rgb().as_array()[0] < 0.05);
        for face in CubeFace::ALL {
            let dir = texel_direction(face, 4, 2, 2);
            let hit = face_hit(dir);
            assert_eq!(hit.face, face);
        }
    }

    #[test]
    fn constant_env_irradiance_matches_color_and_mirror_prefilter_copies() {
        let white = EnvironmentCube::solid(Color::WHITE, 4);
        let irr = convolve_irradiance(&white);
        let sample = irr.sample(UnitVec3::Z).to_linear_rgb().as_array();
        assert!(
            (sample[0] - 1.0).abs() < 0.05 && (sample[1] - 1.0).abs() < 0.05,
            "{sample:?}"
        );
        let red = EnvironmentCube::solid(Color::BLACK, 4)
            .with_solid_face(CubeFace::PositiveZ, Color::linear_rgb(1.0, 0.0, 0.0));
        let sharp = convolve_specular(&red, Clamped01::ZERO);
        let sharp_z = sharp.sample(UnitVec3::Z).to_linear_rgb().as_array();
        assert!(sharp_z[0] > 0.9, "{sharp_z:?}");
        let broad = convolve_specular(&red, Clamped01::ONE);
        let broad_z = broad.sample(UnitVec3::Z).to_linear_rgb().as_array();
        assert!(
            broad_z[0] < sharp_z[0],
            "rough prefilter should blur the red face, sharp {sharp_z:?} broad {broad_z:?}"
        );
    }

    #[test]
    fn one_packs_as_f16() {
        assert_eq!(f32_to_f16(1.0), 0x3C00);
        assert_eq!(f32_to_f16(0.0), 0);
    }
}
