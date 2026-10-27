//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! `Walk_Loop` by name in a native window, with the §8.8 control rig
//! planting both feet on the floor stance while the cycle plays.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames).

use glam::Vec3;
use ornis_app::{AnimatorAccess, BasePull, ControlRig, Effector, GameWorld, ModelSpawn, TwoBoneIk};
use ornis_assets::Model;
use ornis_assets::scene::ShadowCast;
use ornis_core::{Clamped01, Color, Degrees, Lux, Position, UnitVec3};
use ornis_render::{DirectionalLight, OrbitCamera};

/// Base-pull weight of the demo rig: the pelvis leans halfway into the
/// foot hold.
const PELVIS_PULL: f32 = 0.5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut world = GameWorld::new();
    world.set_title("Ornis — Animation Demo");
    let mannequin = world.load::<Model>("assets/starter/ual1_standard.glb")?;
    let hero = world.spawn(ModelSpawn {
        model: mannequin,
        position: Position::new(0.0, 0.0, 0.0),
        ..Default::default()
    })?;
    // `direction` points toward the light. The key sits in the camera
    // octant so the front is lit; the fill and linear ambient keep the
    // far side readable. `#1A1A26` is sRGB, so it uploaded as ~0.01 linear.
    world.set_ambient(Color::linear_rgb(0.10, 0.10, 0.15));
    world.spawn(DirectionalLight {
        direction: UnitVec3::new(Vec3::new(1.0, 1.0, 1.0))?,
        illuminance: Lux(0.6),
        color: Color::WHITE,
        ..Default::default()
    });
    world.spawn(DirectionalLight {
        direction: UnitVec3::new(Vec3::new(-0.5, 0.5, -0.5))?,
        illuminance: Lux(0.3),
        color: Color::linear_rgb(0.8, 0.8, 1.0),
        shadow: ShadowCast::Disabled,
    });
    world.spawn(OrbitCamera::looking_at(Vec3::new(2.5, 1.8, 3.5), Vec3::Y).with_fov(Degrees(45.0)));
    world.entity_mut(hero).animator()?.play("Walk_Loop")?;
    // §8.8 control rig: world-space goals on the mannequin's stance —
    // both ankles held on their floor points and the pelvis free to lean
    // into the hold (`BasePull`). The skeleton root lives under the model
    // root; joints resolve by their glTF names. Goals go stale as the
    // root moves — the mannequin walks in place here, and a moving host
    // re-sets the rig from its update loop every frame.
    let skel_root = world.entity_mut(hero).animator()?.targets()?[0];
    let entity = world.entity_mut(skel_root);
    let thigh_l = entity.joint_named("thigh_l").expect("thigh_l");
    let calf_l = entity.joint_named("calf_l").expect("calf_l");
    let foot_l = entity.joint_named("foot_l").expect("foot_l");
    let thigh_r = entity.joint_named("thigh_r").expect("thigh_r");
    let calf_r = entity.joint_named("calf_r").expect("calf_r");
    let foot_r = entity.joint_named("foot_r").expect("foot_r");
    let pelvis = entity.joint_named("pelvis").expect("pelvis");
    let chain = |root, mid, end, target: Vec3| {
        Effector::TwoBone(TwoBoneIk {
            root,
            mid,
            end,
            target,
            weight: Clamped01::ONE,
        })
    };
    let rig = ControlRig {
        goals: vec![
            chain(thigh_l, calf_l, foot_l, Vec3::new(0.089, 0.104, -0.036)),
            chain(thigh_r, calf_r, foot_r, Vec3::new(-0.089, 0.104, -0.036)),
        ],
        base: Some(BasePull {
            joint: pelvis,
            weight: Clamped01::new(PELVIS_PULL),
        }),
    };
    world.entity_mut(skel_root).set_control_rig(rig);
    ornis::run(world)
}
