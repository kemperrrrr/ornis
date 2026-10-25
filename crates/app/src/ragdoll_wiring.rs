//! Typed ragdoll facade: binds a skeleton root to a Featherstone chain
//! (`docs/ragdoll-design.md` §3, §7). The physics-free model lives in
//! [`ornis_animation::RagdollDesc`]; this module owns the install seam —
//! the chain assembles in [`crate::ragdoll_runtime`], the per-frame fill
//! runs in [`crate::ragdoll_sync`], and the facade re-exports the §4
//! seams the host talks to.

use glam::{Quat, Vec3};
use ornis_animation::{
    JointMask, JointTrs, MaskId, RagdollBlend, RagdollDesc, RagdollError, RagdollFade, RagdollMode,
    Skeleton,
};
use ornis_core::units::Clamped01;
use ornis_core::{Entity, Seconds, SmartStore};
use ornis_physics::RigidBody;

pub use crate::ragdoll_proxy::RagdollProxySync;
pub use crate::ragdoll_runtime::{RagdollBoneView, RagdollProxy, RagdollRuntime, build_ragdoll};
pub use crate::ragdoll_sync::{RagdollSyncSystem, update_ragdoll};

// qual:api — typed ragdoll input for hosts; in-tree callers land with
// the load wiring, hosts own the install decision.
/// Installs the ragdoll chain of `desc` on the skeleton `root` (§7):
/// binds against the root's [`Skeleton`] lane and registers the cold
/// description plus the hot runtime. Refuses (writing nothing) without a
/// skeleton on the root or on a description [`RagdollDesc::validate`]
/// otherwise. A reinstall replaces the chain outright — an in-flight
/// weight transition dies with the replaced state, like on
/// [`clear_ragdoll`].
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
    let links = desc.bones.len();
    // A reinstall owns the root anew: an in-flight weight transition (§4.2)
    // belongs to the replaced chain and dies with it, like on
    // [`clear_ragdoll`].
    if let Some(mut fades) = store.write_lane::<RagdollFade>() {
        fades.remove(root);
    }
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
            anim_rotations: vec![Quat::IDENTITY; joints],
            pinned: vec![None; joints],
            skipped_frames: 0,
        },
    );
    store.insert(root, RagdollMode::Animated);
    spawn_proxies(store, root, links);
    Ok(())
}

/// World proxies of `root`'s chain (§8.5): one [`RagdollProxy`] entity per
/// bone — [`RagdollProxySync`](crate::ragdoll_proxy::RagdollProxySync) owns
/// their [`ornis_physics::RigidBody`] binding and drive. Replaces any
/// previous proxies of the root (reinstall owns the root anew).
fn spawn_proxies(store: &mut SmartStore, root: Entity, links: usize) {
    clear_proxies(store, root);
    store.register::<RagdollProxy>();
    for link in 0..links {
        let proxy = store.create_entity();
        store.insert(proxy, RagdollProxy { root, link });
    }
}

/// Destroys the world proxies of `root` (§8.5), dropping their solver
/// bodies first so nothing keeps simulating on a dead entity. Returns the
/// number of proxies removed.
fn clear_proxies(store: &mut SmartStore, root: Entity) -> usize {
    let doomed: Vec<Entity> = store
        .read_lane::<RagdollProxy>()
        .map(|proxies| {
            proxies
                .entities
                .iter()
                .zip(&proxies.data)
                .filter(|(_, proxy)| proxy.root == root)
                .map(|(&entity, _)| entity)
                .collect()
        })
        .unwrap_or_default();
    if let Some(mut bodies) = store.write_lane::<RigidBody>() {
        for &entity in &doomed {
            bodies.remove(entity);
        }
    }
    if let Some(mut proxies) = store.write_lane::<RagdollProxy>() {
        for &entity in &doomed {
            proxies.remove(entity);
        }
    }
    for entity in doomed.iter().copied() {
        store.destroy_entity(entity);
    }
    doomed.len()
}

// qual:api — typed ragdoll removal for hosts; in-tree callers land with
// the load wiring, hosts own the install decision.
/// Removes the ragdoll chain from `root`, returning whether one was
/// installed (the second call finds nothing).
pub fn clear_ragdoll(store: &mut SmartStore, root: Entity) -> bool {
    let had = crate::anim_wiring::clear_root_lanes::<RagdollDesc, RagdollRuntime>(store, root);
    crate::anim_wiring::clear_root_hot_lanes::<RagdollBlend, RagdollMode>(store, root);
    clear_proxies(store, root);
    if let Some(mut fades) = store.write_lane::<RagdollFade>() {
        fades.remove(root);
    }
    had
}

/// The mode contract shared by the mode entries (§6): the skeleton joint
/// count, the installed chain and the mask sizing. Worker of
/// [`set_ragdoll_mode`] and [`set_ragdoll_mode_fade`]. Leaf.
fn checked_mode(
    store: &SmartStore,
    root: Entity,
    mode: RagdollMode,
) -> Result<usize, RagdollError> {
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
    Ok(joints)
}

// qual:api — typed mode switch for hosts; in-tree callers land with the
// gameplay bindings, hosts own the impact/getup policy.
/// Switches the pose owner of `root` (§2.1): `Animated`, `Physical` or
/// `Blend { mask, weight }`, applied at once. Refuses (writing nothing)
/// without an installed chain ([`RagdollError::Missing`]) or with a mask
/// that is missing or sized differently from the skeleton
/// ([`RagdollError::Mask`]). A set is instant by nature (§4.2): it cuts
/// any in-flight weight transition and re-owns every joint — the live
/// pins of [`set_ragdoll_weight`] die with it.
pub fn set_ragdoll_mode(
    store: &mut SmartStore,
    root: Entity,
    mode: RagdollMode,
) -> Result<(), RagdollError> {
    checked_mode(store, root, mode)?;
    if let Some(mut fades) = store.write_lane::<RagdollFade>() {
        fades.remove(root);
    }
    if let Some(mut blends) = store.write_lane::<RagdollBlend>()
        && let Some(blend) = blends.get_mut(root)
    {
        for pin in &mut blend.pinned {
            *pin = None;
        }
    }
    // `insert` self-registers the lane (`ensure_lane`).
    store.insert(root, mode);
    Ok(())
}

// qual:api — typed fading mode switch for hosts; in-tree callers land
// with the gameplay bindings, hosts own the impact/getup policy.
/// Switches the pose owner like [`set_ragdoll_mode`], but the live
/// weights lerp to the mode's over `duration` seconds (§4.2, linear —
/// `SkelFade` mechanics). A non-positive duration cuts at once; a
/// non-finite one is refused ([`RagdollError::Fade`]). Refuses with
/// [`RagdollError::Busy`] while a transition is in flight — the flying
/// progress is never interrupted (§4.2). Live pins die with the call,
/// like on [`set_ragdoll_mode`].
pub fn set_ragdoll_mode_fade(
    store: &mut SmartStore,
    root: Entity,
    mode: RagdollMode,
    duration: Seconds,
) -> Result<(), RagdollError> {
    if !duration.get().is_finite() {
        return Err(RagdollError::Fade);
    }
    checked_mode(store, root, mode)?;
    if store
        .read_lane::<RagdollFade>()
        .is_some_and(|fades| fades.contains(root))
    {
        return Err(RagdollError::Busy);
    }
    if duration.get() <= 0.0 {
        return set_ragdoll_mode(store, root, mode);
    }
    let from = store
        .read_lane::<RagdollBlend>()
        .and_then(|blends| blends.get(root).map(|blend| blend.weights.clone()))
        .unwrap_or_default();
    // The instant set re-owns every joint (pins die, the mode lands);
    // the transition starts from the weights snapshotted above. `insert`
    // self-registers the fade lane (`ensure_lane`).
    set_ragdoll_mode(store, root, mode)?;
    store.insert(
        root,
        RagdollFade {
            from,
            duration,
            elapsed: Seconds::ZERO,
        },
    );
    Ok(())
}

// qual:api — typed per-joint weight pin for hosts; in-tree callers land
// with the gameplay bindings, hosts own the impact/getup policy.
/// Pins the live weight of one skeleton joint (§7 «точечный вес», the
/// "shot off the arm" scenario): the pin wins over the mode and any
/// in-flight transition until a mode set clears it. Returns `false`
/// (writing nothing) without an installed chain, for a joint outside the
/// skeleton, or when `w` is outside `[0, 1]` or non-finite.
pub fn set_ragdoll_weight(store: &mut SmartStore, root: Entity, joint: usize, w: f32) -> bool {
    if !w.is_finite() || !(0.0..=1.0).contains(&w) {
        return false;
    }
    let Some(mut blends) = store.write_lane::<RagdollBlend>() else {
        return false;
    };
    let Some(blend) = blends.get_mut(root) else {
        return false;
    };
    let Some(pin) = blend.pinned.get_mut(joint) else {
        return false;
    };
    *pin = Some(Clamped01::new(w));
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat};
    use ornis_animation::{
        JointId, RagdollBase, RagdollBone, RagdollLimit, RagdollMotor, RagdollShape, ragdoll_blend,
    };
    use ornis_core::{Engine, FixedTime, GlobalTransform, UnitQuat};
    use ornis_physics::engine::PhysicsEngine;
    use ornis_physics::joint::JointKind;

    use crate::physics_runtime::install_physics;
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
    fn scaled_landmarks() -> Vec<(Vec3, Quat, Vec3)> {
        use std::f32::consts::FRAC_PI_2;
        vec![
            (
                Vec3::ZERO,
                Quat::from_rotation_z(FRAC_PI_2),
                Vec3::splat(FULL_EDGE),
            ),
            (
                Vec3::new(0.0, FULL_EDGE, 0.0),
                Quat::from_rotation_z(FRAC_PI_2),
                Vec3::splat(FULL_EDGE),
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

    /// One fixed ragdoll tick (§5): the step length the host ticks at.
    fn tick() -> Seconds {
        Seconds::new(1.0 / 60.0)
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
                assert!(near(
                    (view.position, view.rotation),
                    (landmark.0, landmark.1)
                ));
                let in_engine = runtime
                    .with_engine(|engine| {
                        engine
                            .get_body(view.body)
                            .map(|body| (body.position, body.orientation))
                    })
                    .flatten()
                    .expect("body in the engine");
                assert!(
                    near(in_engine, (landmark.0, landmark.1)),
                    "engine body {bone}"
                );
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
                translation: Vec3::new(5.0, -2.0, 1.0),
                rotation: UnitQuat::try_from_quat(Quat::from_rotation_z(1.0)).unwrap(),
                scale: Vec3::splat(2.0),
            },
        );
        // Zero gravity: the seam checks below read the bind pose exactly
        // through the ABA step — falling under gravity is covered by
        // `physical_tick_falls_and_animated_tick_holds`.
        set_ragdoll(&mut store, root, desc(), Vec3::ZERO).unwrap();

        // Physical: every bone joint owns the pose. The entity transform
        // above must not leak into the sample — the chain and the seam
        // live in the skeleton's model space (§4.1).
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        assert_eq!(
            blend.get(root).map(|b| b.weights.clone()),
            Some(vec![Clamped01::ONE; 3])
        );
        let physical = blend.get(root).map(|b| b.physical.clone()).unwrap();
        // Pose checks ride the seam tolerance, not exact f32 equality —
        // float noise in the fixtures must not break the verdict.
        assert!(near(
            (physical[0].translation, physical[0].rotation),
            (Vec3::ZERO, Quat::IDENTITY)
        ));
        assert!(near(
            (physical[1].translation, physical[1].rotation),
            (Vec3::Y, Quat::IDENTITY)
        ));
        assert!(near(
            (physical[2].translation, physical[2].rotation),
            (Vec3::Y, Quat::IDENTITY)
        ));
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
        assert!(update_ragdoll(&store, root, tick()));
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
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root, tick()));

        // A non-finite body pose zeroes the frame and counts it.
        let runtime = store
            .read_lane::<RagdollRuntime>()
            .and_then(|runtimes| runtimes.get(root).cloned())
            .unwrap();
        let handle = runtime.body_handle(0).unwrap();
        runtime.with_engine(|engine| {
            engine.get_body_mut(handle).unwrap().position = Vec3::NAN;
        });
        assert!(!update_ragdoll(&store, root, tick()));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        let blend = blend.get(root).unwrap();
        assert_eq!(blend.weights, vec![Clamped01::ZERO; 3]);
        assert_eq!(blend.skipped_frames, 1);
    }

    /// Shared rig of the runtime-reading tests (§4.1 seam checks): a
    /// fresh store with the [`chain`] skeleton and the [`desc`] ragdoll
    /// installed at `gravity`, plus the live [`RagdollRuntime`] handle —
    /// the scaffold these tests otherwise repeat verbatim.
    fn runtime_rig(gravity: Vec3) -> (SmartStore, Entity, RagdollRuntime) {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), gravity).unwrap();
        let runtime = store
            .read_lane::<RagdollRuntime>()
            .and_then(|runtimes| runtimes.get(root).cloned())
            .unwrap();
        (store, root, runtime)
    }

    #[test]
    fn set_ragdoll_reinstall_drops_a_flying_fade() {
        let (mut store, root, _runtime) = runtime_rig(Vec3::ZERO);
        set_ragdoll_mode_fade(&mut store, root, RagdollMode::Physical, Seconds::new(0.5)).unwrap();
        assert!(
            store
                .read_lane::<RagdollFade>()
                .is_some_and(|fades| fades.contains(root))
        );
        set_ragdoll(&mut store, root, desc(), Vec3::ZERO).unwrap();
        assert!(
            !store
                .read_lane::<RagdollFade>()
                .is_some_and(|fades| fades.contains(root))
        );
    }

    /// One broken variant of the sound [`desc`]/[`chain`] and the refusal
    /// it must map to.
    type BreakCase = (fn(&mut Skeleton, &mut RagdollDesc), RagdollError);

    /// Every refusal of [`set_ragdoll`] is typed and writes nothing (§6):
    /// one broken field per case over the sound [`desc`]/[`chain`].
    #[test]
    fn set_ragdoll_maps_the_full_error_set_writing_nothing() {
        let cases: [BreakCase; 9] = [
            (
                |_, broken| {
                    broken.bones.clear();
                },
                RagdollError::EmptyChain,
            ),
            (
                |_, broken| {
                    broken.bones[2].joint = broken.bones[1].joint;
                },
                RagdollError::Joint,
            ),
            (
                |_, broken| {
                    broken.bones[1].axis = Vec3::ZERO;
                },
                RagdollError::Axis,
            ),
            (
                |_, broken| {
                    broken.bones[1].density = 0.0;
                },
                RagdollError::Density,
            ),
            (
                |_, broken| {
                    broken.bones[1].limit = Some(RagdollLimit {
                        min: 1.0,
                        max: -1.0,
                    });
                },
                RagdollError::Limit,
            ),
            (
                |_, broken| {
                    broken.bones[1].motor.damping = -1.0;
                },
                RagdollError::Motor,
            ),
            (
                |_, broken| {
                    broken.bones[1].shape = RagdollShape::Capsule {
                        radius: -1.0,
                        half_height: 0.0,
                    };
                },
                RagdollError::Shape,
            ),
            (
                |_, broken| {
                    broken.bones[1].parent = None;
                },
                RagdollError::Parent,
            ),
            (
                |skeleton, broken| {
                    skeleton.parents = vec![None, None, Some(JointId::from_raw(1))].into();
                    broken.bones[1].parent = None;
                },
                RagdollError::Roots,
            ),
        ];
        for (break_it, want) in cases {
            let mut skeleton = chain();
            let mut broken = desc();
            break_it(&mut skeleton, &mut broken);
            let mut store = SmartStore::new();
            let root = store.create_entity();
            store.register::<Skeleton>();
            store.insert(root, skeleton);
            assert_eq!(
                set_ragdoll(&mut store, root, broken, Vec3::ZERO),
                Err(want),
                "{want:?}"
            );
            assert_eq!(
                (
                    store
                        .read_cold_lane::<RagdollDesc>()
                        .is_some_and(|descs| descs.contains(root)),
                    store
                        .read_lane::<RagdollRuntime>()
                        .is_some_and(|runtimes| runtimes.contains(root)),
                ),
                (false, false),
                "{want:?} writes nothing"
            );
        }
    }

    /// The mode facades refuse an uninstalled root ([`RagdollError::Missing`])
    /// and a non-finite transition duration ([`RagdollError::Fade`]); the
    /// `bool` entries answer `false` without a chain (§7).
    #[test]
    fn mode_facades_refuse_uninstalled_roots_and_bad_transitions() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        assert_eq!(
            set_ragdoll_mode(&mut store, root, RagdollMode::Physical),
            Err(RagdollError::Missing)
        );
        assert_eq!(
            set_ragdoll_mode_fade(&mut store, root, RagdollMode::Physical, Seconds::new(1.0)),
            Err(RagdollError::Missing)
        );
        assert!(!update_ragdoll(&store, root, tick()));
        assert!(!set_ragdoll_weight(&mut store, root, 0, 1.0));
        set_ragdoll(&mut store, root, desc(), Vec3::ZERO).unwrap();
        assert_eq!(
            set_ragdoll_mode_fade(
                &mut store,
                root,
                RagdollMode::Physical,
                Seconds::new(f32::NAN)
            ),
            Err(RagdollError::Fade)
        );
    }

    /// The frame repeats the pose (§8.4): two identical chains stepped in
    /// lockstep land bit-equal — the fixed tick is deterministic.
    #[test]
    fn identical_chains_land_bit_identical_poses() {
        let dt = tick();
        let (mut first_store, first_root, _first) = runtime_rig(Vec3::new(0.0, -9.8, 0.0));
        let (mut second_store, second_root, _second) = runtime_rig(Vec3::new(0.0, -9.8, 0.0));
        set_ragdoll_mode(&mut first_store, first_root, RagdollMode::Physical).unwrap();
        set_ragdoll_mode(&mut second_store, second_root, RagdollMode::Physical).unwrap();
        for _ in 0..32 {
            assert!(update_ragdoll(&first_store, first_root, dt));
            assert!(update_ragdoll(&second_store, second_root, dt));
        }
        let first = first_store.read_lane::<RagdollBlend>().unwrap();
        let second = second_store.read_lane::<RagdollBlend>().unwrap();
        let first = first.get(first_root).unwrap();
        let second = second.get(second_root).unwrap();
        assert_eq!(first.physical, second.physical, "bit-equal chain poses");
        assert_eq!(first.weights, second.weights);
        assert_eq!(first.skipped_frames, second.skipped_frames);
    }

    #[test]
    fn sample_local_round_trips_the_chain_pose() {
        let (_store, _root, runtime) = runtime_rig(Vec3::new(0.0, -9.8, 0.0));

        // Model space: the sampled locals are the chain's rest locals.
        let mut out = vec![JointTrs::IDENTITY; 3];
        assert!(runtime.sample_local(&desc(), &mut out));
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
    fn scaled_sample_restores_the_rest_pose() {
        let rig = scaled_rig();
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, rig.clone());
        // Zero gravity keeps the chain on its bind pose through the step.
        set_ragdoll(&mut store, root, scaled_desc(), Vec3::ZERO).unwrap();
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        let blend = blend.get(root).unwrap();

        // At rest the sample lands back in the skeleton's rest — the
        // inverse seam undoes the accumulated bind scale (§4.1). The
        // physical scale field stays one; the blend keeps the animated
        // scale.
        let rest = &rig.rest;
        for (joint, sampled) in blend.physical.iter().enumerate() {
            assert!(
                (sampled.translation - rest[joint].translation).length() < 1e-5,
                "joint {joint} translation {:?} vs {:?}",
                sampled.translation,
                rest[joint].translation
            );
            assert!(
                sampled.rotation.dot(rest[joint].rotation).abs() > 0.9999,
                "joint {joint} rotation"
            );
            assert_eq!(sampled.scale, Vec3::ONE);
        }
    }

    /// A rig whose chain skips a joint: the path `0 <- 2 <- 3` runs past
    /// the scaled decoy sibling `1`, so the chain's bone slots (`0, 1, 2`)
    /// differ from their skeleton joint indices (`0, 2, 3`). The decoy
    /// folds a 6× accumulated bind scale — reading the bind table by bone
    /// slot instead of the bone's joint would divide by it.
    fn sparse_rig() -> Skeleton {
        use std::f32::consts::FRAC_PI_2;
        Skeleton::new(
            vec![
                None,
                Some(JointId::from_raw(0)),
                Some(JointId::from_raw(0)),
                Some(JointId::from_raw(2)),
            ],
            vec![Default::default(); 4],
            vec![
                "hips".to_string(),
                "leaf".to_string(),
                "spine".to_string(),
                "head".to_string(),
            ],
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
                scale: Vec3::splat(3.0),
            },
            JointTrs {
                translation: Vec3::X,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
            JointTrs {
                translation: Vec3::X,
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            },
        ])
    }

    /// The chain `0 <- 2 <- 3` of [`sparse_rig`] — bone `1`'s joint is
    /// `2`, bone `2`'s joint is `3`.
    fn sparse_desc() -> RagdollDesc {
        RagdollDesc {
            bones: vec![bone(0, None), bone(2, Some(0)), bone(3, Some(2))],
            base: RagdollBase::Floating,
        }
    }

    #[test]
    fn sparse_sample_restores_the_rest_pose() {
        let rig = sparse_rig();
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, rig.clone());
        // Zero gravity keeps the chain on its bind pose through the step.
        set_ragdoll(&mut store, root, sparse_desc(), Vec3::ZERO).unwrap();
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        let blend = blend.get(root).unwrap();

        // At rest the sample lands back in the skeleton's rest on every
        // chain joint. The inverse seam must divide by the parent *bone's*
        // accumulated bind scale (joint 2's fold), not the bind of the
        // same-numbered skeleton slot (the decoy joint 1 folds 6×).
        let rest = &rig.rest;
        for &joint in &[0usize, 2, 3] {
            let sampled = &blend.physical[joint];
            assert!(
                (sampled.translation - rest[joint].translation).length() < 1e-5,
                "joint {joint} translation {:?} vs {:?}",
                sampled.translation,
                rest[joint].translation
            );
            assert!(
                sampled.rotation.dot(rest[joint].rotation).abs() > 0.9999,
                "joint {joint} rotation"
            );
            assert_eq!(sampled.scale, Vec3::ONE);
        }
    }

    #[test]
    fn clear_ragdoll_returns_the_pose_to_animation() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root, tick()));

        // Clearing removes the blend and mode lanes too — `apply_ragdoll`
        // finds nothing and the pose is animated again (§7).
        assert!(clear_ragdoll(&mut store, root));
        assert!(
            !store
                .read_lane::<RagdollBlend>()
                .is_some_and(|lane| lane.contains(root))
        );
        assert!(
            !store
                .read_lane::<RagdollMode>()
                .is_some_and(|lane| lane.contains(root))
        );
        assert!(set_ragdoll_mode(&mut store, root, RagdollMode::Physical).is_err());
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
            assert!(
                !store
                    .read_lane::<RagdollBlend>()
                    .is_some_and(|lane| lane.contains(root))
            );
            assert!(
                !store
                    .read_lane::<RagdollMode>()
                    .is_some_and(|lane| lane.contains(root))
            );
            assert!(!clear_ragdoll(&mut store, root), "already clear");
        }
    }

    #[test]
    fn physical_tick_falls_and_animated_tick_holds() {
        let (mut store, root, runtime) = runtime_rig(Vec3::new(0.0, -9.8, 0.0));
        let pos = |runtime: &RagdollRuntime| {
            let handle = runtime.body_handle(0).unwrap();
            runtime
                .with_engine(|engine| engine.get_body(handle).map(|body| body.position))
                .flatten()
                .unwrap()
        };
        // Animated never steps the chain (ABA has no sleeping — §5).
        let held = pos(&runtime);
        for _ in 0..8 {
            assert!(update_ragdoll(&store, root, tick()));
        }
        assert!((pos(&runtime) - held).length() < 1e-5);
        // Physical takes one fixed step per tick — the free root falls.
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        for _ in 0..8 {
            assert!(update_ragdoll(&store, root, tick()));
        }
        let after = pos(&runtime);
        assert!(after.y < held.y - 1e-3, "the chain falls: {after:?}");
    }

    #[test]
    fn drives_hold_the_animated_half() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        // Welded root, gravity along -z so the x-hinges feel the head's
        // weight; the driven spine hinge carries a real torque budget.
        let mut description = desc();
        description.base = RagdollBase::Fixed;
        description.bones[1].motor.max_torque = 200.0;
        description.bones[1].motor.damping = 10.0;
        set_ragdoll(&mut store, root, description, Vec3::new(0.0, 0.0, -9.8)).unwrap();
        // Upper-body ragdoll: head physical, hips and spine animated —
        // the animated half is held by the drives (§4.2).
        let mask = store.create_entity();
        store.register_cold::<JointMask>();
        store.insert_cold(
            mask,
            JointMask {
                name: "upper".to_string(),
                weights: vec![Clamped01::ZERO, Clamped01::ZERO, Clamped01::ONE],
            },
        );
        set_ragdoll_mode(
            &mut store,
            root,
            RagdollMode::Blend {
                mask: MaskId(mask),
                weight: Clamped01::ONE,
            },
        )
        .unwrap();
        // The sample's role: the animated locals stay at rest.
        store
            .write_lane::<RagdollBlend>()
            .and_then(|mut blends| {
                let blend = blends.get_mut(root)?;
                blend.anim_rotations = vec![Quat::IDENTITY; 3];
                Some(())
            })
            .expect("blend lane");
        let runtime = store
            .read_lane::<RagdollRuntime>()
            .and_then(|runtimes| runtimes.get(root).cloned())
            .unwrap();
        let angle = |bone: usize| {
            let hinge = runtime.bone(bone).unwrap().hinge.unwrap();
            runtime
                .with_engine(|engine| engine.joint_angle(hinge))
                .flatten()
                .unwrap()
        };
        let motor = |bone: usize| {
            let hinge = runtime.bone(bone).unwrap().hinge.unwrap();
            runtime
                .with_engine(|engine| engine.joint_motor(hinge))
                .flatten()
        };
        for _ in 0..36 {
            assert!(update_ragdoll(&store, root, tick()));
        }
        // The driven spine hinge holds its target while the physical head
        // hangs on it; the head hinge (weight 1) has no servo at all.
        let held = angle(1);
        assert!(motor(1).is_some(), "w < 1 drives the hinge");
        assert!(motor(2).is_none(), "w = 1 gasses the servo");
        // Physical gasses every servo (§4.2) — the same hinge sags under
        // the head's weight, and the drive was what held it.
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        for _ in 0..36 {
            assert!(update_ragdoll(&store, root, tick()));
        }
        let sagged = angle(1);
        assert!(sagged.abs() > 0.1, "the free hinge sags: {sagged:?}");
        assert!(
            held.abs() * 2.0 < sagged.abs(),
            "the drive held {held:?} vs free {sagged:?}"
        );
    }

    #[test]
    fn ragdoll_returns_to_animation() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::new(0.0, -9.8, 0.0)).unwrap();
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        for _ in 0..6 {
            assert!(update_ragdoll(&store, root, tick()));
        }
        // Back to Animated: the weights drop to zero and the blend leaves
        // the animated pose alone (§5 — root motion is animated again).
        set_ragdoll_mode(&mut store, root, RagdollMode::Animated).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        let blend = blend.get(root).unwrap();
        assert_eq!(blend.weights, vec![Clamped01::ZERO; 3]);
        let mut animated = vec![JointTrs::IDENTITY; 3];
        ragdoll_blend(&mut animated, &blend.physical, &blend.weights);
        assert_eq!(animated, vec![JointTrs::IDENTITY; 3]);
    }

    #[test]
    fn weight_transition_progresses_linearly() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::ZERO).unwrap();
        // Half a second of `Animated -> Physical`: 0.3 s in, the weights
        // stand at 0.3 (SkelFade mechanics — §4.2, linear).
        set_ragdoll_mode_fade(&mut store, root, RagdollMode::Physical, Seconds::new(1.0)).unwrap();
        for _ in 0..3 {
            assert!(update_ragdoll(&store, root, Seconds::new(0.1)));
        }
        {
            let blend = store.write_lane::<RagdollBlend>().unwrap();
            let blend = blend.get(root).unwrap();
            assert!(
                (blend.weights[0].get() - 0.3).abs() < 1e-5,
                "{:?}",
                blend.weights
            );
        }
        // The flying transition is never interrupted (§4.2).
        assert_eq!(
            set_ragdoll_mode_fade(&mut store, root, RagdollMode::Animated, Seconds::new(1.0)),
            Err(RagdollError::Busy)
        );
        // An unusable duration is refused outright.
        assert_eq!(
            set_ragdoll_mode_fade(
                &mut store,
                root,
                RagdollMode::Animated,
                Seconds::new(f32::NAN)
            ),
            Err(RagdollError::Fade)
        );
        // An instant set cuts at once (§4.2).
        set_ragdoll_mode(&mut store, root, RagdollMode::Physical).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        {
            let blend = store.write_lane::<RagdollBlend>().unwrap();
            assert_eq!(
                blend.get(root).map(|blend| blend.weights.clone()),
                Some(vec![Clamped01::ONE; 3])
            );
        }
        // A non-positive fade duration behaves like the instant set.
        set_ragdoll_mode_fade(&mut store, root, RagdollMode::Animated, Seconds::ZERO).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        {
            let blend = store.write_lane::<RagdollBlend>().unwrap();
            assert_eq!(
                blend.get(root).map(|blend| blend.weights.clone()),
                Some(vec![Clamped01::ZERO; 3])
            );
        }
    }

    #[test]
    fn set_ragdoll_weight_pins_a_joint() {
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        set_ragdoll(&mut store, root, desc(), Vec3::ZERO).unwrap();
        // "Shot off the arm": joint 1 goes physical while the mode says
        // animated — the pin wins until a mode set clears it (§7).
        assert!(set_ragdoll_weight(&mut store, root, 1, 1.0));
        assert!(!set_ragdoll_weight(&mut store, root, 9, 1.0));
        assert!(!set_ragdoll_weight(&mut store, root, 1, f32::NAN));
        assert!(update_ragdoll(&store, root, tick()));
        {
            let blend = store.write_lane::<RagdollBlend>().unwrap();
            assert_eq!(
                blend.get(root).map(|blend| blend.weights.clone()),
                Some(vec![Clamped01::ZERO, Clamped01::ONE, Clamped01::ZERO])
            );
        }
        set_ragdoll_mode(&mut store, root, RagdollMode::Animated).unwrap();
        assert!(update_ragdoll(&store, root, tick()));
        let blend = store.write_lane::<RagdollBlend>().unwrap();
        assert_eq!(
            blend.get(root).map(|blend| blend.weights.clone()),
            Some(vec![Clamped01::ZERO; 3])
        );
    }

    /// Runs the sample pass over a rest clip with the ragdoll blend filled
    /// as given and returns the published model translation of joint 1
    /// (the seam check of §2.3 through `SkelSampleSystem`).
    fn sampled_joint_one(weights: Vec<Clamped01>, physical: Vec<JointTrs>) -> Vec3 {
        use ornis_animation::{
            ClipId, Interpolation, JointPose, JointTrack, KeyTrack, SkelClip, SkelPlayer,
            run_skel_sample,
        };
        use ornis_core::{Resources, Time};
        let mut store = SmartStore::new();
        let root = store.create_entity();
        store.register::<Skeleton>();
        store.insert(root, chain());
        // Empty tracks: every joint samples its authored rest.
        let clip = store.create_entity();
        store.register_cold::<SkelClip>();
        store.insert_cold(
            clip,
            SkelClip {
                name: "rest".to_string(),
                duration: 1.0,
                tracks: vec![JointTrack {
                    joint: JointId::from_raw(0),
                    node: None,
                    translation: KeyTrack {
                        keys: Vec::new(),
                        interpolation: Interpolation::Linear,
                    },
                    rotation: KeyTrack {
                        keys: Vec::new(),
                        interpolation: Interpolation::Linear,
                    },
                    scale: KeyTrack {
                        keys: Vec::new(),
                        interpolation: Interpolation::Linear,
                    },
                }],
            },
        );
        store.register::<SkelPlayer>();
        store.insert(
            root,
            SkelPlayer {
                clip: ClipId(clip),
                time: Seconds::ZERO,
                speed: Seconds::new(1.0),
                weight: Clamped01::ONE,
                playing: true,
                looping: true,
            },
        );
        store.register::<JointPose>();
        store.insert(root, JointPose::identity(3));
        set_ragdoll(&mut store, root, desc(), Vec3::ZERO).unwrap();
        store
            .write_lane::<RagdollBlend>()
            .and_then(|mut blends| {
                let blend = blends.get_mut(root)?;
                blend.physical = physical;
                blend.weights = weights;
                Some(())
            })
            .expect("blend lane");
        let mut resources = Resources::new();
        resources.insert(Time::new());
        resources.insert(store);
        let _ = run_skel_sample(&resources);
        let store = resources.get::<SmartStore>().expect("store");
        let poses = store.read_lane::<JointPose>().expect("poses");
        poses.get(root).expect("root pose").matrices[1].transform_point3(Vec3::ZERO)
    }

    #[test]
    fn skel_sample_sees_the_ragdoll_blend() {
        let shift = Vec3::new(9.0, 9.0, 9.0);
        let physical = {
            let mut physical = vec![JointTrs::IDENTITY; 3];
            physical[1].translation = shift;
            physical
        };
        // Physical owns the joint: the published pose carries the shift.
        let owned = sampled_joint_one(vec![Clamped01::ONE; 3], physical.clone());
        assert!((owned - shift).length() < 1e-4, "{owned:?}");
        // Animated: the blend leaves the animated (rest) pose alone.
        let held = sampled_joint_one(vec![Clamped01::ZERO; 3], physical.clone());
        assert!((held - Vec3::Y).length() < 1e-4, "{held:?}");
        // A length mismatch skips the blend — animation wins the frame.
        let mut short = physical;
        short.truncate(1);
        let skipped = sampled_joint_one(vec![Clamped01::ONE; 3], short);
        assert!((skipped - Vec3::Y).length() < 1e-4, "{skipped:?}");
    }

    /// Gravity of the §8.5 coupling drop tests.
    const DROP_GRAVITY_Y: f32 = -9.81;
    /// The world floor's top surface in the drop tests.
    const FLOOR_TOP_Y: f32 = -3.5;
    /// A settled one-metre-half-extent link cannot sink below the floor
    /// top — anything under this line fell through.
    const SINK_LINE_Y: f32 = -3.0;
    /// Fixed frames of the landing test: two seconds of dropping.
    const LANDING_TICKS: usize = 120;
    /// Fixed frames of the determinism test: contact-rich mid-bounce.
    const COUPLE_TICKS: usize = 64;

    /// World floor of the §8.5 coupling tests: a wide static box whose
    /// top sits at [`FLOOR_TOP_Y`].
    fn floor_body() -> RigidBody {
        let half = FULL_EDGE * 0.5;
        RigidBody::new_box(
            Vec3::new(0.0, FLOOR_TOP_Y - half, 0.0),
            Vec3::new(20.0, half, 20.0),
            0.0,
        )
    }

    /// Engine-level rig of the §8.5 coupling tests: a world floor plus a
    /// physical three-bone chain dropped onto it, the root riding
    /// `root_pose`. The full installed plan couples them (proxy drive →
    /// world step → drain/fold → ABA step).
    fn coupled_rig_at(root_pose: GlobalTransform) -> (Engine, Entity) {
        let gravity = Vec3::new(0.0, DROP_GRAVITY_Y, 0.0);
        let mut engine = Engine::new();
        let root = {
            let store = engine.world_mut().store_mut().expect("world store");
            let floor = store.create_entity();
            store.insert(floor, floor_body());
            let root = store.create_entity();
            store.register::<Skeleton>();
            store.insert(root, chain());
            store.insert(root, root_pose);
            set_ragdoll(store, root, desc(), gravity).unwrap();
            set_ragdoll_mode(store, root, RagdollMode::Physical).unwrap();
            root
        };
        install_physics(&mut engine, gravity);
        crate::install_skeletal_animation(&mut engine);
        (engine, root)
    }

    /// [`coupled_rig_at`] with the identity root (the common case).
    fn coupled_rig() -> (Engine, Entity) {
        coupled_rig_at(GlobalTransform::IDENTITY)
    }

    /// Steps `engine` `ticks` fixed frames through the installed plan.
    fn run_fixed(engine: &mut Engine, ticks: usize) {
        let dt = FixedTime::default().delta_seconds();
        for _ in 0..ticks {
            engine.run_frame(dt);
        }
    }

    /// Rig plus `ticks` fixed frames of the §8.5 coupling runs.
    fn coupled_drop(ticks: usize) -> (Engine, Entity) {
        let (mut engine, root) = coupled_rig();
        run_fixed(&mut engine, ticks);
        (engine, root)
    }

    /// Chain link positions in the root's model space of the §8.5
    /// coupling runs (the chain runs model-space; the root pose folds it
    /// to the world, see [`world_positions`]).
    fn chain_positions(engine: &Engine, root: Entity) -> Vec<Vec3> {
        let store = engine.world().store().expect("world store");
        let runtimes = store.read_lane::<RagdollRuntime>().expect("runtimes");
        let runtime = runtimes.get(root).expect("chain");
        let mut positions = Vec::new();
        for link in 0..runtime.bone_count() {
            let handle = runtime.body_handle(link).expect("link body");
            let position = runtime
                .with_engine(|aba| aba.get_body(handle).map(|body| body.position))
                .flatten()
                .expect("link pose");
            positions.push(position);
        }
        positions
    }

    /// [`chain_positions`] folded to the world through the root's
    /// [`GlobalTransform`] (`world = rotation · model + translation`).
    fn world_positions(engine: &Engine, root: Entity) -> Vec<Vec3> {
        let store = engine.world().store().expect("world store");
        let pose = store
            .read_lane::<GlobalTransform>()
            .and_then(|globals| globals.get(root).copied())
            .unwrap_or(GlobalTransform::IDENTITY);
        chain_positions(engine, root)
            .into_iter()
            .map(|position| pose.rotation.get() * position + pose.translation)
            .collect()
    }

    /// §8.5 end-to-end: a physical chain dropped onto a world floor must
    /// land on it. Without the proxy coupling the ABA chain knows no
    /// world collision and falls straight through — the floor's catch is
    /// the drained-impulse channel working (proxies → `apply_impulse`).
    #[test]
    fn world_floor_catches_a_falling_chain() {
        let (engine, root) = coupled_drop(LANDING_TICKS);
        for (link, position) in chain_positions(&engine, root).iter().enumerate() {
            assert!(
                position.y > SINK_LINE_Y,
                "bone {link} fell through the floor: y={}",
                position.y
            );
        }
    }

    /// §8.5 end-to-end under a non-identity root: the fold's `R⁻¹(p−t)`
    /// and the drive's `R·v` must land the chain on the world floor the
    /// same way as the identity root — the coupling is world-correct,
    /// not model-silent.
    #[test]
    fn world_floor_catches_a_shifted_yawed_root() {
        let yaw = UnitQuat::normalize(Quat::from_rotation_y(0.7)).expect("unit yaw");
        let (mut engine, root) = coupled_rig_at(GlobalTransform {
            translation: Vec3::new(2.5, 6.0, -1.5),
            rotation: yaw,
            scale: Vec3::ONE,
        });
        run_fixed(&mut engine, LANDING_TICKS);
        for (link, position) in world_positions(&engine, root).iter().enumerate() {
            assert!(
                position.y > SINK_LINE_Y,
                "bone {link} fell through the floor: y={}",
                position.y
            );
        }
    }

    /// §8.5 negative control: without the proxy channel the ABA chain
    /// knows no world collision and falls through the floor — the catch
    /// above is the coupling working, not a fixture fluke.
    #[test]
    fn world_floor_needs_the_coupling() {
        let (mut engine, root) = coupled_rig();
        {
            let store = engine.world_mut().store_mut().expect("world store");
            let mut proxies = store.write_lane::<RagdollProxy>().expect("proxies");
            let unbound: Vec<Entity> = proxies.entities.clone();
            for entity in unbound {
                proxies.remove(entity);
            }
        }
        run_fixed(&mut engine, LANDING_TICKS);
        let positions = chain_positions(&engine, root);
        assert!(
            positions.iter().any(|position| position.y < SINK_LINE_Y),
            "chain must fall through without the coupling: {positions:?}"
        );
    }

    /// §8.5 end-to-end determinism: two identical coupled runs land
    /// bit-identical chain poses — the drain is order-stable (sorted
    /// pair/point) and the fold applies the very same impulses.
    #[test]
    fn coupled_impulses_land_bit_identical_poses() {
        let (first_engine, first_root) = coupled_drop(COUPLE_TICKS);
        let (second_engine, second_root) = coupled_drop(COUPLE_TICKS);
        assert_eq!(
            chain_positions(&first_engine, first_root),
            chain_positions(&second_engine, second_root),
            "bit-identical coupled chain poses"
        );
    }
}
