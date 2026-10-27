//! Animation demo: the vendored starter mannequin
//! (`assets/starter/ual1_standard.glb`, Quaternius UAL-1, CC0) playing
//! `Walk_Loop` by name in a native window, with the §8.8 control rig
//! planting both feet on the floor stance while the cycle plays, an
//! §8.6 upper-body gesture layer (`Idle_Talking_Loop` masked above the
//! spine), and clip sync marks plus footstep events authored on the
//! walk clip.
//!
//! Run with: `cargo run --example anim` (`--frames N` exits after N
//! frames).

use glam::Vec3;
use ornis_app::{
    AnimEvent, Animator, AnimatorAccess, BasePull, ControlRig, Effector, GameWorld, JointMask,
    ModelSpawn, SkelLayer, SkelLayerStack, SkelPlayer, SyncMark, TwoBoneIk,
};
use ornis_assets::Model;
use ornis_assets::scene::ShadowCast;
use ornis_core::{Clamped01, Color, Degrees, Entity, Lux, Position, Seconds, UnitVec3};
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
    let skel_root = world
        .entity_mut(hero)
        .animator()?
        .targets()?
        .first()
        .copied()
        .expect("the mannequin spawns one skeleton root");
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
    assert!(
        world.entity_mut(skel_root).set_control_rig(rig),
        "the demo rig installs on the mannequin"
    );
    author_gesture_layer(&mut world, hero, skel_root)?;
    ornis::run(world)
}

// qual:allow(test_quality, untested) reason: "demo-only authoring — exercised by running the anim example (cargo run --example anim), mirroring the benchmark-only kernels in core/benches"
/// §8.6 layer authoring: the talking loop overlays the walk above the
/// spine. The mask zeroes the legs, so the foot rig keeps working on
/// the walk pose underneath; a moving host re-sets the stack the same
/// way from its update loop. Clip meta: foot contacts drive sync marks
/// (§8.4) and the step events hosts would react to (§8.1).
fn author_gesture_layer(
    world: &mut GameWorld,
    hero: Entity,
    skel_root: Entity,
) -> Result<(), Box<dyn std::error::Error>> {
    let (walk_clip, talk_clip) = {
        let store = world.engine().world().store().ok_or("store")?;
        let anim = store
            .read_lane::<Animator>()
            .ok_or("animator lane")?
            .get(hero)
            .cloned()
            .ok_or("animator on the root")?;
        (
            anim.clip("Walk_Loop").ok_or("Walk_Loop")?,
            anim.clip("Idle_Talking_Loop").ok_or("Idle_Talking_Loop")?,
        )
    };
    let skeleton = world
        .entity_mut(skel_root)
        .skeleton()
        .ok_or("skeleton under the root")?;
    let mut weights = vec![Clamped01::ZERO; skeleton.joint_count()];
    for (weight, name) in weights.iter_mut().zip(skeleton.joint_names.iter()) {
        let upper = name.starts_with("spine_")
            || name.starts_with("clavicle_")
            || name.starts_with("upperarm_")
            || name.starts_with("lowerarm_")
            || name.starts_with("hand_")
            || name.starts_with("neck_")
            || name == "Head";
        *weight = if upper {
            Clamped01::ONE
        } else {
            Clamped01::ZERO
        };
    }
    let upper = world.create_skel_mask(JointMask {
        name: "upper_body".into(),
        weights,
    });
    assert!(
        world.entity_mut(skel_root).set_skel_layers(SkelLayerStack {
            layers: vec![SkelLayer {
                player: SkelPlayer {
                    clip: talk_clip,
                    time: Seconds::ZERO,
                    speed: Seconds::new(1.0),
                    weight: Clamped01::new(0.85),
                    playing: true,
                    looping: true,
                },
                mask: upper,
            }],
        }),
        "the gesture layer installs on the mannequin"
    );
    author_clip_meta(world, walk_clip.0);
    Ok(())
}

// qual:allow(test_quality, untested) reason: "demo-only authoring — exercised by running the anim example (cargo run --example anim), mirroring the benchmark-only kernels in core/benches"
/// Clip meta: foot contacts drive sync marks (§8.4) and the step events
/// hosts would react to (§8.1).
fn author_clip_meta(world: &mut GameWorld, walk: Entity) {
    assert!(
        world.entity_mut(walk).set_clip_sync_marks(vec![
            SyncMark {
                name: "contact_l".into(),
                at: 0.10,
            },
            SyncMark {
                name: "contact_r".into(),
                at: 0.60,
            },
        ]),
        "foot contact marks install on Walk_Loop"
    );
    assert!(
        world.entity_mut(walk).set_clip_events(vec![
            AnimEvent {
                name: "footstep".into(),
                at: 0.10,
            },
            AnimEvent {
                name: "footstep".into(),
                at: 0.60,
            },
        ]),
        "footstep events install on Walk_Loop"
    );
}
