//! Bounding sphere of one packed light, read as scalar `f32` words.
//!
//! A `GpuLight` is 144 bytes (`align(16)`): `kind`, `direction`, `position`,
//! `color`, `params`, then the shadow matrix this pass never reads. Point
//! lights use `(position, range)`. Spots use the cone's bounding sphere, and
//! fall back to the point sphere when the cone is wider than [`COS_FLOOR`].

use ornis_gpu_shader::ReadOnly;
use ornis_gpu_shader::Storage;
use ornis_gpu_shader::Vec3;

use crate::COS_FLOOR;
use crate::LIGHT_WORDS;
use crate::SPOT;

/// Word of `kind.x` inside a light.
const KIND: u32 = 0;
/// Word of `direction.x`.
const DIRECTION: u32 = 4;
/// Word of `position.x`.
const POSITION: u32 = 8;
/// Word of `params.x` (range). `params.z`, the outer cosine, is two later.
const RANGE: u32 = 16;
/// `range * (1 + tan²) / 2`.
const HALF: f32 = 0.5;

#[inline(always)]
fn word(lights: &Storage<f32, 0, 2, ReadOnly>, light: u32, offset: u32) -> f32 {
    lights.load(light * LIGHT_WORDS + offset)
}

/// `(center, radius)` of light `index`.
#[inline(always)]
pub(crate) fn bound(lights: &Storage<f32, 0, 2, ReadOnly>, index: u32) -> (Vec3, f32) {
    let center = Vec3::new(
        word(lights, index, POSITION),
        word(lights, index, POSITION + 1),
        word(lights, index, POSITION + 2),
    );
    let range = word(lights, index, RANGE);
    let cos_outer = word(lights, index, RANGE + 2);
    if word(lights, index, KIND) < SPOT || cos_outer < COS_FLOOR {
        return (center, range);
    }
    let tan_sq = (1.0 - cos_outer * cos_outer) / (cos_outer * cos_outer);
    let radius = range * (1.0 + tan_sq) * HALF;
    let axis = Vec3::new(
        word(lights, index, DIRECTION),
        word(lights, index, DIRECTION + 1),
        word(lights, index, DIRECTION + 2),
    );
    let length = axis.length();
    let scaled = Vec3::new(
        center.x + axis.x / length * radius,
        center.y + axis.y / length * radius,
        center.z + axis.z / length * radius,
    );
    (scaled, radius)
}
