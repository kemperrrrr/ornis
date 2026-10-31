use super::super::super::CULL_PARAMS_BYTE_SIZE;
use super::FEW;
use super::Kind;
use super::MANY;
use super::MATERIAL_AT;
use super::OBJECT_BYTES;
use super::RADIUS;
use super::kind_name;
use super::pack_flag_count;
use super::pack_flag_planes;
use super::pack_flag_spheres;
use super::pack_objects;
use super::pack_params;
use super::scene;
use super::visible_ids;

#[test]
fn slab_keeps_the_same_band_on_the_short_and_long_line() {
    let few = scene(Kind::Few);
    let many = scene(Kind::Many);
    assert_eq!(few.spheres.len(), FEW as usize);
    assert_eq!(many.spheres.len(), MANY as usize);
    assert_eq!(visible_ids(&few).len(), 64);
    assert_eq!(visible_ids(&many).len(), 64);
    assert_eq!(kind_name(Kind::Few), "few");
}

#[test]
fn worst_rows_are_all_visible_or_fully_rejected() {
    let all = scene(Kind::WorstVisible);
    let late = scene(Kind::WorstLate);
    assert_eq!(visible_ids(&all).len(), all.spheres.len());
    assert!(visible_ids(&late).is_empty());
    assert_eq!(kind_name(Kind::WorstLate), "worst_late_reject");
}

#[test]
fn production_and_flag_packs_store_the_same_first_sphere() {
    let scene = scene(Kind::Few);
    let objects = pack_objects(&scene.spheres);
    let flags = pack_flag_spheres(&scene.spheres);
    assert_eq!(objects.len(), scene.spheres.len() * OBJECT_BYTES);
    assert_eq!(flags.len(), scene.spheres.len() * super::SPHERE_BYTES);
    let radius = f32::from_le_bytes(
        objects[super::COL0_X..super::COL0_X + 4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(radius, RADIUS);
    let slot = u32::from_le_bytes(objects[MATERIAL_AT..MATERIAL_AT + 4].try_into().unwrap());
    assert_eq!(slot, 0);
    let flag_radius = f32::from_le_bytes(
        flags[super::SPHERE_BYTES - 4..super::SPHERE_BYTES]
            .try_into()
            .unwrap(),
    );
    assert_eq!(flag_radius, RADIUS);
}

#[test]
fn params_and_flag_planes_carry_the_count_and_the_first_component() {
    let scene = scene(Kind::Few);
    let params = pack_params(&scene);
    assert_eq!(params.len(), CULL_PARAMS_BYTE_SIZE as usize);
    let count = u32::from_le_bytes(
        params[super::PARAM_COUNT_AT..super::PARAM_COUNT_AT + 4]
            .try_into()
            .unwrap(),
    );
    assert_eq!(count, FEW);
    let planes = pack_flag_planes(&scene.planes);
    let nx = f32::from_le_bytes(planes[0..4].try_into().unwrap());
    assert_eq!(nx, scene.planes[0][0]);
    let count_bytes = pack_flag_count(FEW);
    assert_eq!(
        u32::from_le_bytes(count_bytes[0..4].try_into().unwrap()),
        FEW
    );
}
