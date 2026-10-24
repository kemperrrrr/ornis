//! Typed ragdoll facade: binds a skeleton root to a Featherstone chain
//! (`docs/ragdoll-design.md` §3, §7). The physics-free model lives in
//! [`ornis_animation::RagdollDesc`]; this module owns the install seam —
//! the chain assembles in [`crate::ragdoll_runtime`], the per-frame fill
//! runs in [`crate::ragdoll_sync`], and the facade re-exports the §4
//! seams the host talks to.

use glam::Vec3;
use ornis_animation::{
    JointMask, JointTrs, MaskId, RagdollBlend, RagdollDesc, RagdollError, RagdollMode, Skeleton,
};
use ornis_core::units::Clamped01;
use ornis_core::{Entity, SmartStore};

pub use crate::ragdoll_runtime::{RagdollBoneView, RagdollRuntime, build_ragdoll};
pub use crate::ragdoll_sync::{RagdollSyncSystem, update_ragdoll};

// qual:api — typed ragdoll input for hosts; in-tree callers land with
// the load wiring, hosts own the install decision.
/// Installs the ragdoll chain of `desc` on the skeleton `root` (§7):
/// binds against the root's [`Skeleton`] lane and registers the cold
/// description plus the hot runtime. Refuses (writing nothing) without a
/// skeleton on the root or on a description [`RagdollDesc::validate`]
/// otherwise.
pub fn set_ragdoll(
    store: &mut SmartStore,
    root: Entity,
    desc: RagdollDesc,
    gravity: Vec3,
) -> Result<(), RagdollError> {
    let Some(skeleton) = store
        .read_lane::<Skeleton>()
        .and_then(|skeletons| skeletons.get(root).cloned())
    else {
        return Err(RagdollError::NoSkeleton);
    };
    let runtime = build_ragdoll(&skeleton, &desc, gravity)?;
    let joints = skeleton.parents.len();
    store.register_cold::<RagdollDesc>();
    store.register::<RagdollRuntime>();
    store.register::<RagdollBlend>();
    store.register::<RagdollMode>();
    store.insert_cold(root, desc);
    store.insert(root, runtime);
    store.insert(
        root,
        RagdollBlend {
            physical: vec![JointTrs::IDENTITY; joints],
            weights: vec![Clamped01::ZERO; joints],
            skipped_frames: 0,
        },
    );
    store.insert(root, RagdollMode::Animated);
    Ok(())
}

// qual:api — typed ragdoll removal for hosts; in-tree callers land with
// the load wiring, hosts own the install decision.
/// Removes the ragdoll chain from `root`, returning whether one was
/// installed (the second call finds nothing).
pub fn clear_ragdoll(store: &mut SmartStore, root: Entity) -> bool {
    crate::anim_wiring::clear_root_lanes::<RagdollDesc, RagdollRuntime>(store, root)
}

// qual:api — typed mode switch for hosts; in-tree callers land with the
// gameplay bindings, hosts own the impact/getup policy.
/// Switches the pose owner of `root` (§2.1): `Animated`, `Physical` or
/// `Blend { mask, weight }`. Refuses (writing nothing) without an
/// installed chain ([`RagdollError::Missing`]) or with a mask that is
/// missing or sized differently from the skeleton ([`RagdollError::Mask`]).
pub fn set_ragdoll_mode(
    store: &mut SmartStore,
    root: Entity,
    mode: RagdollMode,
) -> Result<(), RagdollError> {
    let Some(joints) = store
        .read_lane::<Skeleton>()
        .and_then(|skeletons| skeletons.get(root).map(|skeleton| skeleton.parents.len()))
    else {
        return Err(RagdollError::NoSkeleton);
    };
    if !store
        .read_lane::<RagdollRuntime>()
        .is_some_and(|runtimes| runtimes.contains(root))
    {
        return Err(RagdollError::Missing);
    }
    if let RagdollMode::Blend {
        mask: MaskId(mask), ..
    } = mode
    {
        let sized = store
            .read_cold_lane::<JointMask>()
            .and_then(|masks| masks.get(mask).cloned())
            .is_some_and(|mask| mask.weights.len() == joints);
        if !sized {
            return Err(RagdollError::Mask);
        }
    }
    store.register::<RagdollMode>();
    store.insert(root, mode);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat};
    use ornis_animation::{JointId, RagdollBase, RagdollBone, RagdollMotor, RagdollShape};
    use ornis_core::{GlobalTransform, UnitQuat};
    use ornis_physics::engine::PhysicsEngine;
    use ornis_physics::joint::JointKind;

    use crate::ragdoll_runtime::{FULL_EDGE, hinge_joint};

    /// Three-joint chain `0 <- 1 <- 2`, one metre apart at rest.
    fn chain() -> Skeleton {
        Skeleton::new(
            vec![None, Some(JointId::from_raw(0)), Some(JointId::from_raw(1))],
            vec![Default::default(); 3],
            vec!["hips".to_string(), "spine".to_string(), "head".to_string()],
        )
        .with_rest(vec![
            JointTrs::IDENTITY,
            JointTrs {
                translation: Vec3::Y,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            JointTrs {
                translation: Vec3::Y,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ])
    }

    /// A sound bone on `joint` under `parent`.
    fn bone(joint: usize, parent: Option<usize>) -> RagdollBone {
        RagdollBone {
            joint: JointId::from_raw(joint as u32),
            shape: RagdollShape::Box {
                half_extents: Vec3::ONE,
            },
            axis: Vec3::X,
            limit: None,
            motor: RagdollMotor {
                max_torque: 0.0,
                damping: 0.0,
                friction: 0.0,
            },
            density: 1.0,
            parent: parent.map(|index| JointId::from_raw(index as u32)),
        }
    }

    /// The three-bone description of `chain()`.
    fn desc() -> RagdollDesc {
        RagdollDesc {
            bones: vec![bone(0, None), bone(1, Some(0)), bone(2, Some(1))],
            base: RagdollBase::Floating,
        }
    }

    #[test]
    fn build_ragdoll_sits_on_the_bind_pose() {
        {
            let runtime = build_ragdoll(&chain(), &desc(), Vec3::new(0.0, -1.0, 0.0))
                .expect("the sound chain assembles");
            assert_eq!(runtime.bone_count(), 3);
            let frame = |bone: usize| {
                runtime
                    .bone(bone)
                    .map(|view| (view.position, view.rotation, view.parent, view.hinge))
            };
            assert_eq!(
                frame(0).map(|(position, rotation, parent, _)| (position, rotation, parent)),
                Some((Vec3::ZERO, Quat::IDENTITY, None))
            );
            assert_eq!(
                frame(1).map(|(position, rotation, parent, _)| (position, rotation, parent)),
                Some((Vec3::Y, Quat::IDENTITY, Some(0)))
            );
            assert_eq!(
                frame(2).map(|(position, rotation, parent, _)| (position, rotation, parent)),
                Some((Vec3::Y * FULL_EDGE, Quat::IDENTITY, Some(1)))
            );
            assert!(runtime.bone(0).is_some_and(|view| view.hinge.is_none()));
            assert!(runtime.bone(1).is_some_and(|view| view.hinge.is_some()));
        }
    }

    /// A two-bone rig whose rest rotates 90° about Z and scales 2× on
    /// the root, with the child 1 m along the parent's local x.
    fn scaled_rig() -> Skeleton {
        use std::f32::consts::FRAC_PI_2;
        Skeleton::new(
            vec![None, Some(JointId::from_raw(0))],
            vec![Default::default(); 2],
            vec!["hips".to_string(), "spine".to_string()],
        )
        .with_rest(vec![
            JointTrs {
                translation: Vec3::ZERO,
                rotation: Quat::from_rotation_z(FRAC_PI_2),
                scale: Vec3::splat(FULL_EDGE),
            },
            JointTrs {
                translation: Vec3::X,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ])
    }

    /// The two-bone description of [`scaled_rig`].
    fn scaled_desc() -> RagdollDesc {
        RagdollDesc {
            bones: vec![bone(0, None), bone(1, Some(0))],
            base: RagdollBase::Floating,
        }
    }

    /// The expected bind landmarks of [`scaled_rig`]: the root frame is
    /// the 90°-Z turn at the origin; the child sits `2 × (1,0,0)` rotated
    /// into `(0,2,0)` — the 2× rest scale lengthens the lever arm.
    fn scaled_landmarks() -> Vec<(Vec3, Quat)> {
        use std::f32::consts::FRAC_PI_2;
        vec![
            (Vec3::ZERO, Quat::from_rotation_z(FRAC_PI_2)),
            (
                Vec3::new(0.0, FULL_EDGE, 0.0),
                Quat::from_rotation_z(FRAC_PI_2),
            ),
        ]
    }

    /// Whether two poses agree: positions within `1e-5`, rotations by
    /// quaternion alignment (independently computed representations of one
    /// rotation differ by float noise, and `acos` turns that noise into
    /// milliradians — the dot product does not).
    fn near(a: (Vec3, Quat), b: (Vec3, Quat)) -> bool {
        (a.0 - b.0).length() < 1e-5 && a.1.dot(b.1).abs() > 0.9999
    }

    #[test]
    fn bind_landmarks_follow_the_scaled_rest() {
        {
            let skeleton = scaled_rig();
            let expected = scaled_landmarks();
            let runtime = build_ragdoll(&skeleton, &scaled_desc(), Vec3::ZERO)
                .expect("the scaled rig assembles");
            for (bone, landmark) in expected.iter().enumerate() {
                let view = runtime.bone(bone).expect("bone");
                assert!(near((view.position, view.rotation), *landmark));
                let in_engine = runtime
                    .with_engine(|engine| {
                        engine
                            .get_body(view.body)
                            .map(|body| (body.position, body.orientation))
                    })
                    .flatten()
                    .expect("body in the engine");
                assert!(near(in_engine, *landmark), "engine body {bone}");
            }
            // The hinge lands on the scaled lever arm: in the parent frame
            // the anchor is the 2 m offset, and the shared axis is x.
            let bones = scaled_desc().bones;
            let hinge = hinge_joint(&bones[1], &expected, &bones[0]).expect("hinge");
            if let JointKind::Revolute {
                local_anchor_a,
                local_anchor_b,
                local_axis_a,
                local_axis_b,
                ..
            } = hinge
            {
                assert!(local_anchor_a.distance(Vec3::X * FULL_EDGE) < 1e-5);
                assert_eq!(local_anchor_b, Vec3::ZERO);
                assert!(local_axis_a.distance(Vec3::X) < 1e-5);
                assert!(local_axis_b.distance(Vec3::X) < 1e-5);
            } else {
                panic!("the bone hinges are revolute");
            }
        }
    }

    #[test]
    fn build_ragdoll_is_repeatable() {
        {
            let skeleton = scaled_rig();
            let first = build_ragdoll(&skeleton, &scaled_desc(), Vec3::ZERO).expect("first");
            let second = build_ragdoll(&skeleton, &scaled_desc(), Vec3::ZERO).expect("second");
            for bone in 0..2 {
                assert_eq!(first.bone(bone), second.bone(bone));
            }
        }
    }

    #[test]
    fn build_ragdoll_refuses_a_broken_description() {
        {
            let mut broken = desc();
            broken.bones[1].axis = Vec3::ZERO;
            assert!(build_ragdoll(&chain(), &broken, Vec3::ZERO).is_err());
        }
    }

    /// The verdict of [`set_ragdoll`] and what the lanes hold afterwards.
    fn lane_state(desc: RagdollDesc, with_skeleton: bool) -> (bool, bool, bool) {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        if with_skeleton {
            store.register::<Skeleton>();
            store.insert(root, chain());
        }
        let set = set_ragdoll(&mut store, root, desc, Vec3::ZERO).is_ok();
        let held_desc = store
            .read_cold_lane::<RagdollDesc>()
            .is_some_and(|descs| descs.contains(root));
        let held_runtime = store
            .read_lane::<RagdollRuntime>()
            .is_some_and(|runtimes| runtimes.contains(root));
        (set, held_desc, held_runtime)
    }

    #[test]
    fn set_ragdoll_installs_and_refuses_cleanly() {
        {
            assert_eq!(
                lane_state(desc(), true),
                (true, true, true),
                "the sound chain installs"
            );
            assert_eq!(
                lane_state(desc(), false),
                (false, false, false),
                "no skeleton — nothing is written"
            );
            let mut broken = desc();
            broken.bones[1].axis = Vec3::ZERO;
            assert_eq!(
                lane_state(broken, true),
                (false, false, false),
                "a broken description writes nothing"
            );
        }
    }

    #[test]
    fn set_ragdoll_installs_the_blend_lanes() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        let blend = blend.get(root).unwrap();
        assert_eq!(blend.weights, vec![Clamped01::ZERO; 3]);
        assert_eq!(blend.physical, vec![JointTrs::IDENTITY; 3]);
        assert_eq!(blend.skipped_frames, 0);
        assert_eq!(
            store
                .read_lane::<RagdollMode>()
                .and_then(|m| m.get(root).copied()),
            Some(RagdollMode::Animated)
        );
    }

    #[test]
    fn set_ragdoll_mode_checks_the_mask() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();

        // A mask sized for another skeleton is refused.
        let wrong = store.create_entity();
        store.register_cold::<JointMask>();
        store.insert_cold(
            wrong,
            JointMask {
                name: "wrong".to_string(),
                weights: vec![Clamped01::ONE; 2],
            },
        );
        assert_eq!(
            set_ragdoll_mode(
                &mut store,
                root,
                RagdollMode::Blend {
                    mask: MaskId(wrong),
                    weight: Clamped01::new(0.5),
                }
            ),
            Err(RagdollError::Mask)
        );
        assert_eq!(
            store
                .read_lane::<RagdollMode>()
                .and_then(|m| m.get(root).copied()),
            Some(RagdollMode::Animated),
            "a refused switch writes nothing"
        );

        // A mask of the joint count is accepted.
        let right = store.create_entity();
        store.insert_cold(
            right,
            JointMask {
                name: "right".to_string(),
                weights: vec![Clamped01::ONE; 3],
            },
        );
        let mode = RagdollMode::Blend {
            mask: MaskId(right),
            weight: Clamped01::new(0.5),
        };
        assert_eq!(set_ragdoll_mode(&mut store, root, mode), Ok(()));
        assert_eq!(
            store
                .read_lane::<RagdollMode>()
                .and_then(|m| m.get(root).copied()),
            Some(mode)
        );
    }

    #[test]
    fn update_ragdoll_weights_follow_the_mode() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        store.register::<GlobalTransform>();
        store.insert(
            root,
            GlobalTransform {
                translation: Vec3::ZERO,
                rotation: UnitQuat::IDENTITY,
                scale: Vec3::ONE,
            },
        );
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();

        // Physical: every bone joint owns the pose.
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        assert_eq!(
            blend.get(root).map(|b| b.weights.clone()),
            Some(vec![Clamped01::ONE; 3])
        );
        drop(blend);

        // Blend: mask x weight per joint (0.25·0.5, 1·0.5, 0.5·0.5).
        let mask = store.create_entity();
        store.register_cold::<JointMask>();
        store.insert_cold(
            mask,
            JointMask {
                name: "blend".to_string(),
                weights: vec![Clamped01::new(0.25), Clamped01::ONE, Clamped01::new(0.5)],
            },
        );
        set_ragdoll_mode(
            &mut store,
            root,
            RagdollMode::Blend {
                mask: MaskId(mask),
                weight: Clamped01::new(0.5),
            },
        )
        .unwrap();
        assert!(update_ragdoll(&store, root));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        assert_eq!(
            blend.get(root).map(|b| b.weights.clone()),
            Some(vec![
                Clamped01::new(0.125),
                Clamped01::new(0.5),
                Clamped01::new(0.25)
            ])
        );
    }

    #[test]
    fn update_ragdoll_zeroes_weights_on_a_broken_frame() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        store.register::<GlobalTransform>();
        store.insert(
            root,
            GlobalTransform {
                translation: Vec3::ZERO,
                rotation: UnitQuat::IDENTITY,
                scale: Vec3::ONE,
            },
        );
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root));

        // A non-finite body pose zeroes the frame and counts it.
        let runtime = store
            .read_lane::<RagdollRuntime>()
            .and_then(|runtimes| runtimes.get(root).cloned())
            .unwrap();
        let handle = runtime.body_handle(0).unwrap();
        runtime.with_engine(|engine| {
            engine.get_body_mut(handle).unwrap().position = Vec3::NAN;
        });
        assert!(!update_ragdoll(&store, root));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        let blend = blend.get(root).unwrap();
        assert_eq!(blend.weights, vec![Clamped01::ZERO; 3]);
        assert_eq!(blend.skipped_frames, 1);
    }

    #[test]
    fn sample_local_round_trips_the_chain_pose() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();
        let runtime = store
            .read_lane::<RagdollRuntime>()
            .and_then(|runtimes| runtimes.get(root).cloned())
            .unwrap();

        // Identity entity: the sampled locals are the chain's rest locals.
        let mut out = vec![JointTrs::IDENTITY; 3];
        assert!(runtime.sample_local(&desc(), Mat4::IDENTITY, &mut out));
        assert_eq!(out[0], JointTrs::IDENTITY);
        assert_eq!(out[1].translation, Vec3::Y);
        assert_eq!(out[1].rotation, Quat::IDENTITY);
        assert_eq!(out[2].translation, Vec3::Y);

        // Folding the sampled locals reproduces the body world poses —
        // the physical pose = world transforms of the chain (§4.1).
        let mut world = Mat4::IDENTITY;
        let mut folded = Vec::new();
        for local in &out {
            let local = Mat4::from_scale_rotation_translation(
                local.scale,
                local.rotation,
                local.translation,
            );
            world *= local;
            folded.push(world);
        }
        let bodies = runtime.with_engine(|engine| {
            (0..3)
                .map(|index| {
                    let handle = runtime.body_handle(index).unwrap();
                    let body = engine.get_body(handle).unwrap();
                    Mat4::from_scale_rotation_translation(
                        Vec3::ONE,
                        body.orientation,
                        body.position,
                    )
                })
                .collect::<Vec<_>>()
        });
        for (folded, body) in folded.iter().zip(bodies.unwrap()) {
            assert_eq!(*folded, body);
        }
    }

    #[test]
    fn clear_ragdoll_removes_the_chain() {
        {
            let mut store = SmartStore::new();
            let root = store.create_entity();
            store.register::<Skeleton>();
            store.insert(root, chain());
            assert!(set_ragdoll(&mut store, root, desc(), Vec3::ZERO).is_ok());
            assert!(clear_ragdoll(&mut store, root));
            assert!(!clear_ragdoll(&mut store, root), "already clear");
        }
    }
}
