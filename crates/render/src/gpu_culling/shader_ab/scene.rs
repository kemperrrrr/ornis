//! Shared spheres and frustum planes for the production cull shader and the
//! wasm2gpu flag kernel.
//!
//! Both shaders see the same centers, radii, and planes. The production
//! path stores each sphere as a model matrix (center in column 3, radius
//! on the column lengths `instance_sphere` reads). The flag kernel stores
//! a packed `vec4` and planes as one uniform lane per component, which is
//! the guest ABI the lifter emits.

use super::super::CULL_PARAMS_BYTE_SIZE;
use super::super::CullParams;
use super::super::FRUSTUM_PLANE_COUNT;

/// Spheres in the small scene.
const FEW: u32 = 256;
/// Spheres in the large scenes, including both worst cases.
const MANY: u32 = 8_192;
/// Spacing of the line of spheres, in world units.
const SPACING: f32 = 1.0;
/// Sphere radius stored in every record.
const RADIUS: f32 = 0.25;
/// Half-width of the slab that keeps a band of the line.
const SLAB: f32 = 32.0;
/// Plane `w` that passes every finite sphere in the bench.
const PASS_W: f32 = 1.0e6;
/// Last-plane `w` for the late-reject row. The line reaches about
/// `±MANY/2`, so this has to sit past the rightmost center; a nearer
/// plane would let the positive-x tail through.
const REJECT_W: f32 = -1.0e6;

/// Bytes of one `f32` lane.
const F32_BYTES: usize = std::mem::size_of::<f32>();
/// Bytes of one `u32` lane.
const U32_BYTES: usize = std::mem::size_of::<u32>();
/// Production `PerObject` stride, taken from the GPU record.
pub(super) const OBJECT_BYTES: usize = std::mem::size_of::<crate::renderer::PerObjectGpu>();
/// `material_index` byte offset inside that record.
pub(super) const MATERIAL_AT: usize =
    std::mem::offset_of!(crate::renderer::PerObjectGpu, material_index);
/// Column 0, lane x: radius for `length(model[0].xyz)`.
const COL0_X: usize = 0;
/// Column 1, lane y: radius for `length(model[1].xyz)`.
const COL1_Y: usize = 20;
/// Column 2, lane z: radius for `length(model[2].xyz)`.
const COL2_Z: usize = 40;
/// Translation column of the model matrix.
const COL3_X: usize = 48;
const COL3_Y: usize = 52;
const COL3_Z: usize = 56;
const COL3_W: usize = 60;
/// Guest sphere record: four tightly packed lanes.
const SPHERE_BYTES: usize = F32_BYTES * 4;
/// Guest uniform stride: one component per 16-byte slot.
const LANE_STRIDE: usize = 16;
/// Components per plane in the guest uniform.
const LANES: usize = 4;
/// `u32` words in the guest count binding.
const COUNT_WORDS: usize = 4;
/// Byte offset of [`CullParams::count`].
const PARAM_COUNT_AT: usize = std::mem::offset_of!(CullParams, count);

/// Which input distribution a row of the bench uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A short line, slab frustum, most spheres culled.
    Few,
    /// The same slab on a long line.
    Many,
    /// Every sphere passes every plane (production copies the full record).
    WorstVisible,
    /// Every sphere fails only the last plane (no early-out, no copy).
    WorstLate,
}

/// One sphere the two shaders both receive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sphere {
    /// World-space center.
    pub center: [f32; 3],
    /// Radius.
    pub radius: f32,
}

/// One bench scene: the spheres, the six planes, and which case it is.
pub(crate) struct Scene {
    /// Case name for the table.
    pub kind: Kind,
    /// Source objects, in slot order.
    pub spheres: Vec<Sphere>,
    /// Inward frustum planes, `xyz` normal and `w`.
    pub planes: [[f32; 4]; FRUSTUM_PLANE_COUNT],
}

/// The four rows the bench times, in table order.
#[cfg(feature = "test-util")]
pub(crate) fn scenes() -> Vec<Scene> {
    vec![
        scene(Kind::Few),
        scene(Kind::Many),
        scene(Kind::WorstVisible),
        scene(Kind::WorstLate),
    ]
}

/// Build one case. The line of spheres is shared; only the planes change
/// between the large rows.
pub(crate) fn scene(kind: Kind) -> Scene {
    Scene {
        spheres: spheres(count_of(kind)),
        planes: planes_of(kind),
        kind,
    }
}

/// Stable row name.
pub(crate) fn kind_name(kind: Kind) -> &'static str {
    row_spec(kind).name
}

/// Slots whose sphere passes every plane, in increasing order.
pub(crate) fn visible_ids(scene: &Scene) -> Vec<u32> {
    scene
        .spheres
        .iter()
        .enumerate()
        .filter(|(_, sphere)| inside(sphere, &scene.planes))
        .map(|(index, _)| index as u32)
        .collect()
}

struct RowSpec {
    name: &'static str,
    count: u32,
    planes: PlaneSet,
}

enum PlaneSet {
    Slab,
    Open,
    Late,
}

fn row_spec(kind: Kind) -> RowSpec {
    let (name, count, planes) = match kind {
        Kind::Few => ("few", FEW, PlaneSet::Slab),
        Kind::Many => ("many", MANY, PlaneSet::Slab),
        Kind::WorstVisible => ("worst_all_visible", MANY, PlaneSet::Open),
        Kind::WorstLate => ("worst_late_reject", MANY, PlaneSet::Late),
    };
    RowSpec {
        name,
        count,
        planes,
    }
}

fn count_of(kind: Kind) -> u32 {
    row_spec(kind).count
}

fn planes_of(kind: Kind) -> [[f32; 4]; FRUSTUM_PLANE_COUNT] {
    match row_spec(kind).planes {
        PlaneSet::Open => open_planes(),
        PlaneSet::Late => late_planes(),
        PlaneSet::Slab => slab_planes(),
    }
}

fn spheres(count: u32) -> Vec<Sphere> {
    (0..count).map(|index| sphere_at(index, count)).collect()
}

fn sphere_at(index: u32, count: u32) -> Sphere {
    let offset = (count - 1) as f32 * SPACING / 2.0;
    Sphere {
        center: [index as f32 * SPACING - offset, 0.0, 0.0],
        radius: RADIUS,
    }
}

fn slab_planes() -> [[f32; 4]; FRUSTUM_PLANE_COUNT] {
    [
        [-1.0, 0.0, 0.0, SLAB],
        [1.0, 0.0, 0.0, SLAB],
        [0.0, 0.0, 0.0, PASS_W],
        [0.0, 0.0, 0.0, PASS_W],
        [0.0, 0.0, 0.0, PASS_W],
        [0.0, 0.0, 0.0, PASS_W],
    ]
}

fn open_planes() -> [[f32; 4]; FRUSTUM_PLANE_COUNT] {
    [[0.0, 0.0, 0.0, PASS_W]; FRUSTUM_PLANE_COUNT]
}

fn late_planes() -> [[f32; 4]; FRUSTUM_PLANE_COUNT] {
    let mut planes = open_planes();
    planes[FRUSTUM_PLANE_COUNT - 1] = [1.0, 0.0, 0.0, REJECT_W];
    planes
}

fn inside(sphere: &Sphere, planes: &[[f32; 4]; FRUSTUM_PLANE_COUNT]) -> bool {
    planes.iter().all(|plane| !culled_by(sphere, plane))
}

fn culled_by(sphere: &Sphere, plane: &[f32; 4]) -> bool {
    let dot =
        plane[0] * sphere.center[0] + plane[1] * sphere.center[1] + plane[2] * sphere.center[2];
    dot + plane[3] < -sphere.radius
}

/// `CullParams` bytes: six tight `vec4` planes and the instance count.
pub(crate) fn pack_params(scene: &Scene) -> Vec<u8> {
    let mut bytes = vec![0u8; CULL_PARAMS_BYTE_SIZE as usize];
    scene.planes.iter().enumerate().for_each(|(index, plane)| {
        write_param_plane(&mut bytes, index, plane);
    });
    store_u32(&mut bytes, PARAM_COUNT_AT, scene.spheres.len() as u32);
    bytes
}

/// Production `PerObject` array. `material_index` is the source slot.
pub(crate) fn pack_objects(spheres: &[Sphere]) -> Vec<u8> {
    let mut bytes = vec![0u8; spheres.len() * OBJECT_BYTES];
    spheres
        .iter()
        .enumerate()
        .for_each(|(index, sphere)| write_object(&mut bytes, index, sphere));
    bytes
}

/// Guest sphere storage: `vec4(center, radius)` per slot.
pub(crate) fn pack_flag_spheres(spheres: &[Sphere]) -> Vec<u8> {
    let mut bytes = vec![0u8; spheres.len() * SPHERE_BYTES];
    spheres
        .iter()
        .enumerate()
        .for_each(|(index, sphere)| write_flag_sphere(&mut bytes, index, sphere));
    bytes
}

/// Guest plane uniform: each component in its own 16-byte slot, value in `.x`.
pub(crate) fn pack_flag_planes(planes: &[[f32; 4]; FRUSTUM_PLANE_COUNT]) -> Vec<u8> {
    let mut bytes = vec![0u8; FRUSTUM_PLANE_COUNT * LANES * LANE_STRIDE];
    planes
        .iter()
        .enumerate()
        .for_each(|(index, plane)| write_flag_plane(&mut bytes, index, plane));
    bytes
}

/// Guest count binding: the active length in the first word.
pub(crate) fn pack_flag_count(count: u32) -> Vec<u8> {
    let mut bytes = vec![0u8; COUNT_WORDS * U32_BYTES];
    store_u32(&mut bytes, 0, count);
    bytes
}

fn write_param_plane(bytes: &mut [u8], index: usize, plane: &[f32; 4]) {
    write_strided(bytes, index * LANES * F32_BYTES, F32_BYTES, plane);
}

fn write_object(bytes: &mut [u8], index: usize, sphere: &Sphere) {
    let base = index * OBJECT_BYTES;
    write_placed(bytes, base, &object_lanes(sphere));
    store_u32(bytes, base + MATERIAL_AT, index as u32);
}

fn object_lanes(sphere: &Sphere) -> [(usize, f32); 7] {
    [
        (COL0_X, sphere.radius),
        (COL1_Y, sphere.radius),
        (COL2_Z, sphere.radius),
        (COL3_X, sphere.center[0]),
        (COL3_Y, sphere.center[1]),
        (COL3_Z, sphere.center[2]),
        (COL3_W, 1.0),
    ]
}

fn write_flag_sphere(bytes: &mut [u8], index: usize, sphere: &Sphere) {
    write_strided(
        bytes,
        index * SPHERE_BYTES,
        F32_BYTES,
        &[
            sphere.center[0],
            sphere.center[1],
            sphere.center[2],
            sphere.radius,
        ],
    );
}

fn write_flag_plane(bytes: &mut [u8], index: usize, plane: &[f32; 4]) {
    write_strided(bytes, index * LANES * LANE_STRIDE, LANE_STRIDE, plane);
}

fn write_strided(bytes: &mut [u8], base: usize, stride: usize, lanes: &[f32]) {
    lanes
        .iter()
        .enumerate()
        .for_each(|(lane, value)| store_f32(bytes, base + lane * stride, *value));
}

fn write_placed(bytes: &mut [u8], base: usize, lanes: &[(usize, f32)]) {
    lanes
        .iter()
        .for_each(|(at, value)| store_f32(bytes, base + at, *value));
}

fn store_f32(bytes: &mut [u8], at: usize, value: f32) {
    let end = at + F32_BYTES;
    bytes[at..end].copy_from_slice(&value.to_le_bytes());
}

fn store_u32(bytes: &mut [u8], at: usize, value: u32) {
    let end = at + U32_BYTES;
    bytes[at..end].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
#[path = "scene_tests.rs"]
mod tests;
