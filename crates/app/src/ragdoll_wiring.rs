//! Typed ragdoll facade: binds a skeleton root to a Featherstone chain
//! (`docs/ragdoll-design.md` §3, §7). The physics-free model lives in
//! [`ornis_animation::RagdollDesc`]; this module owns the seam — chain
//! assembly from the bind pose and the install/remove inputs the host
//! talks to.

use std::sync::{Arc, Mutex};

use glam::{Quat, Vec3};
use ornis_animation::{RagdollBase, RagdollBone, RagdollDesc, RagdollShape, Skeleton};
use ornis_core::units::Radians;
use ornis_core::{Entity, SmartStore};
use ornis_physics::articulated::{BaseJoint, FeatherstoneEngine};
use ornis_physics::body::{BodyHandle, RigidBody};
use ornis_physics::engine::PhysicsEngine;
use ornis_physics::joint::{JointHandle, JointKind, RevoluteLimit};

/// Full edge from a half-extent (shape volumes below).
const FULL_EDGE: f32 = 2.0;
/// Hemispherical caps share of a capsule volume (`4/3`).
const CAPSULE_CAPS_SHARE: f32 = 4.0 / 3.0;

/// Hot state of one ragdoll root: its own articulated chain plus the
/// bind-pose frames the world→local seam reads against (§4.1). Registered
/// as a hot lane on the skeleton root (the lane's `Clone` bound is why the
/// chain rides behind an `Arc` — cloning the runtime shares the chain, it
/// never copies it); the chain holds no state but the physics bodies
/// themselves.
#[derive(Clone)]
pub struct RagdollRuntime {
    /// The articulated chain (one per ragdoll root).
    engine: Arc<Mutex<FeatherstoneEngine>>,
    /// Physical link per bone (description order).
    bodies: Vec<BodyHandle>,
    /// Hinge per bone; `None` on the chain root.
    joints: Vec<Option<JointHandle>>,
    /// Bind-pose world transform of each bone's joint frame.
    bind: Vec<(Vec3, Quat)>,
    /// Parent bone index per bone; `None` on the chain root.
    parents: Vec<Option<usize>>,
}

/// World→local seam data of one bone (`docs/ragdoll-design.md` §4.1):
/// the bind frame the physical pose is read against and the chain handles
/// behind it.
#[derive(Debug, Clone, Copy)]
pub struct RagdollBoneView {
    /// Bind-pose world position of the bone's joint frame.
    pub position: Vec3,
    /// Bind-pose world rotation of the bone's joint frame.
    pub rotation: Quat,
    /// Physical link of the bone.
    pub body: BodyHandle,
    /// Hinge handle (`None` on the chain root).
    pub hinge: Option<JointHandle>,
    /// Parent bone index (`None` on the chain root).
    pub parent: Option<usize>,
}

impl RagdollRuntime {
    /// Number of bones in the chain: the chain's own link count, falling
    /// back to the handle table if the lock is poisoned.
    // qual:api — per-bone seam data for hosts; the world→local read
    // (slice 2) is its first in-tree caller.
    pub fn bone_count(&self) -> usize {
        self.with_engine(|engine| engine.body_count())
            .unwrap_or(self.bodies.len())
    }

    /// Seam data of bone `bone` ([`RagdollBoneView`]).
    // qual:api — per-bone seam data for hosts; the world→local read
    // (slice 2) is its first in-tree caller.
    pub fn bone(&self, bone: usize) -> Option<RagdollBoneView> {
        let (position, rotation) = self.bind.get(bone).copied()?;
        Some(RagdollBoneView {
            position,
            rotation,
            body: self.bodies.get(bone).copied()?,
            hinge: self.joints.get(bone).copied().flatten(),
            parent: self.parents.get(bone).copied().flatten(),
        })
    }

    /// Runs `f` over the articulated chain (the stepping seam of §4);
    /// `None` means the chain lock is poisoned (bad skin downstream).
    // qual:api — stepping seam for hosts; the fixed-step bridge (slice 2)
    // is its first in-tree caller.
    pub fn with_engine<R>(&self, f: impl FnOnce(&mut FeatherstoneEngine) -> R) -> Option<R> {
        self.engine.lock().ok().map(|mut engine| f(&mut engine))
    }
}

/// Assembles the physical chain of `desc` under `skeleton` at the bind
/// pose (§3): every bone becomes a dynamic link whose frame is its joint
/// bind frame, hinges sit on the bind anchors, and the root link takes
/// `desc.base`. Returns `None` when [`RagdollDesc::validate`] refuses the
/// description or physics cannot assemble a hinge.
pub fn build_ragdoll(
    skeleton: &Skeleton,
    desc: &RagdollDesc,
    gravity: Vec3,
) -> Option<RagdollRuntime> {
    desc.validate(skeleton).ok()?;
    let bind = bind_worlds(skeleton);
    let mut engine = FeatherstoneEngine::new(gravity);
    let mut handles = Vec::with_capacity(desc.bones.len());
    for (index, bone) in desc.bones.iter().enumerate() {
        let (position, orientation) = bind[bone.joint.index()];
        let mut body = bone_body(bone, position);
        body.orientation = orientation;
        let base = match desc.base {
            RagdollBase::Fixed => BaseJoint::Fixed,
            RagdollBase::Floating => BaseJoint::Floating,
        };
        let root = index == chain_root(&desc.bones);
        handles.push(if root {
            engine.add_root(body, base)
        } else {
            engine.add_body(body)
        });
    }
    let mut joints = Vec::with_capacity(desc.bones.len());
    for (index, bone) in desc.bones.iter().enumerate() {
        let Some(parent) = chain_parent(index, &desc.bones) else {
            joints.push(None);
            continue;
        };
        let hinge = hinge_joint(bone, &bind, &desc.bones[parent])?;
        let handle = engine
            .add_joint(handles[parent], handles[index], hinge)
            .ok()?;
        engine.set_joint_damping(handle, bone.motor.damping);
        engine.set_joint_friction(handle, bone.motor.friction);
        joints.push(Some(handle));
    }
    let parents = (0..desc.bones.len())
        .map(|index| chain_parent(index, &desc.bones))
        .collect();
    Some(RagdollRuntime {
        engine: Arc::new(Mutex::new(engine)),
        bodies: handles,
        joints,
        bind,
        parents,
    })
}

// qual:api — typed ragdoll input for hosts; in-tree callers land with
// the load wiring, hosts own the install decision.
/// Installs the ragdoll chain of `desc` on the skeleton `root` (§7):
/// binds against the root's [`Skeleton`] lane and registers the cold
/// description plus the hot runtime. Refuses (writing nothing) without a
/// skeleton on the root or on a description [`RagdollDesc::validate`]
/// rejects.
pub fn set_ragdoll(store: &mut SmartStore, root: Entity, desc: RagdollDesc, gravity: Vec3) -> bool {
    let Some(skeleton) = store
        .read_lane::<Skeleton>()
        .and_then(|skeletons| skeletons.get(root).cloned())
    else {
        return false;
    };
    let Some(runtime) = build_ragdoll(&skeleton, &desc, gravity) else {
        return false;
    };
    store.register_cold::<RagdollDesc>();
    store.register::<RagdollRuntime>();
    store.insert_cold(root, desc);
    store.insert(root, runtime);
    true
}

// qual:api — typed ragdoll removal for hosts; in-tree callers land with
// the load wiring, hosts own the install decision.
/// Removes the ragdoll chain from `root`, returning whether one was
/// installed (the second call finds nothing).
pub fn clear_ragdoll(store: &mut SmartStore, root: Entity) -> bool {
    crate::anim_wiring::clear_root_lanes::<RagdollDesc, RagdollRuntime>(store, root)
}

/// Bind-pose world transforms (position, rotation) per joint, folded from
/// [`Skeleton::rest`] along the parents in depth order (scale is
/// skin-space and never enters physics). Leaf.
fn bind_worlds(skeleton: &Skeleton) -> Vec<(Vec3, Quat)> {
    let count = skeleton.parents.len();
    let mut depths = Vec::with_capacity(count);
    for parent_of in skeleton.parents.iter() {
        let mut depth = 0;
        let mut cursor = *parent_of;
        while let Some(parent) = cursor {
            depth += 1;
            cursor = skeleton.parents[parent.index()];
        }
        depths.push(depth);
    }
    let mut order: Vec<usize> = (0..count).collect();
    order.sort_by_key(|&joint| depths[joint]);
    let mut out = vec![(Vec3::ZERO, Quat::IDENTITY); count];
    for &joint in &order {
        let local = skeleton.rest[joint];
        let (position, rotation) = match skeleton.parents[joint] {
            Some(parent) => out[parent.index()],
            None => (Vec3::ZERO, Quat::IDENTITY),
        };
        out[joint] = (
            position + rotation * local.translation,
            rotation * local.rotation,
        );
    }
    out
}

/// Dynamic body of `bone` centred on its joint frame (§3.1: the link
/// frame is the joint frame in the first slices). Leaf.
fn bone_body(bone: &RagdollBone, position: Vec3) -> RigidBody {
    let mass = match bone.shape {
        RagdollShape::Capsule {
            radius,
            half_height,
        } => {
            let disc = std::f32::consts::PI * radius * radius;
            bone.density * (disc * (FULL_EDGE * half_height) + CAPSULE_CAPS_SHARE * disc * radius)
        }
        RagdollShape::Box { half_extents } => {
            bone.density
                * (FULL_EDGE * half_extents.x)
                * (FULL_EDGE * half_extents.y)
                * (FULL_EDGE * half_extents.z)
        }
    };
    match bone.shape {
        RagdollShape::Capsule {
            radius,
            half_height,
        } => RigidBody::new_capsule(position, radius, half_height, mass),
        RagdollShape::Box { half_extents } => RigidBody::new_box(position, half_extents, mass),
    }
}

/// Revolute hinge of `bone` under `parent`, anchored on the bind frames
/// of §3.2: `local_anchor_b` is zero (the child frame *is* the joint) and
/// both axes are the same world axis. Leaf.
fn hinge_joint(
    bone: &RagdollBone,
    bind: &[(Vec3, Quat)],
    parent: &RagdollBone,
) -> Option<JointKind> {
    let (joint_position, joint_rotation) = bind[bone.joint.index()];
    let (parent_position, parent_rotation) = bind[parent.joint.index()];
    let limit = match bone.limit {
        Some(window) => Some(RevoluteLimit::try_new(
            Radians::new(window.min),
            Radians::new(window.max),
        )?),
        None => None,
    };
    Some(JointKind::Revolute {
        local_anchor_a: parent_rotation.inverse() * (joint_position - parent_position),
        local_anchor_b: Vec3::ZERO,
        local_axis_a: parent_rotation.inverse() * (joint_rotation * bone.axis),
        local_axis_b: bone.axis,
        limit,
        motor: None,
    })
}

/// Index of the chain root bone (the one with `parent: None`); the
/// description validates to exactly one. Leaf.
fn chain_root(bones: &[RagdollBone]) -> usize {
    let mut root = 0;
    for (index, bone) in bones.iter().enumerate() {
        if bone.parent.is_none() {
            root = index;
        }
    }
    root
}

/// Bone index of `bones[index]`'s parent bone. Leaf.
fn chain_parent(index: usize, bones: &[RagdollBone]) -> Option<usize> {
    let parent = bones[index].parent?;
    let mut found = None;
    for (candidate, bone) in bones.iter().enumerate() {
        if bone.joint == parent {
            found = Some(candidate);
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use ornis_animation::{JointId, JointTrs, RagdollMotor};

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

    #[test]
    fn build_ragdoll_refuses_a_broken_description() {
        {
            let mut broken = desc();
            broken.bones[1].axis = Vec3::ZERO;
            assert!(build_ragdoll(&chain(), &broken, Vec3::ZERO).is_none());
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
        let set = set_ragdoll(&mut store, root, desc, Vec3::ZERO);
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
    fn clear_ragdoll_removes_the_chain() {
        {
            let mut store = SmartStore::new();
            let root = store.create_entity();
            store.register::<Skeleton>();
            store.insert(root, chain());
            assert!(set_ragdoll(&mut store, root, desc(), Vec3::ZERO));
            assert!(clear_ragdoll(&mut store, root));
            assert!(!clear_ragdoll(&mut store, root), "already clear");
        }
    }
}
