//! Ragdoll chain assembly: the hot [`RagdollRuntime`] of one skeleton
//! root — its own Featherstone chain plus the bind frames the world→local
//! seam reads against (`docs/ragdoll-design.md` §3, §4). The physics-free
//! model lives in [`ornis_animation::RagdollDesc`]; this module builds the
//! chain from a validated description.

use std::sync::{Arc, Mutex};

use glam::{Mat4, Quat, Vec3};
use ornis_animation::{
    JointTrs, RagdollBase, RagdollBone, RagdollDesc, RagdollError, RagdollShape, Skeleton,
};
use ornis_core::units::Radians;
use ornis_physics::articulated::{BaseJoint, FeatherstoneEngine};
use ornis_physics::body::{BodyHandle, RigidBody};
use ornis_physics::engine::PhysicsEngine;
use ornis_physics::joint::{JointHandle, JointKind, RevoluteLimit};

/// Full edge from a half-extent (shape volumes below).
pub(crate) const FULL_EDGE: f32 = 2.0;
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
    /// Bind-pose frame of each bone's joint: world position, world
    /// rotation and the accumulated world scale — the factor the inverse
    /// seam of [`RagdollRuntime::sample_local`] undoes (bodies are rigid
    /// and carry no scale). Indexed by bone (description order), not by
    /// skeleton joint slot — on sparse chains the two differ, and the
    /// seam must read the parent *bone's* fold.
    bind: Vec<(Vec3, Quat, Vec3)>,
    /// Parent bone index per bone; `None` on the chain root.
    parents: Vec<Option<usize>>,
}

/// World→local seam data of one bone (`docs/ragdoll-design.md` §4.1):
/// the bind frame the physical pose is read against and the chain handles
/// behind it.
#[derive(Debug, Clone, Copy, PartialEq)]
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
        let (position, rotation, _) = self.bind.get(bone).copied()?;
        Some(RagdollBoneView {
            position,
            rotation,
            body: self.bodies.get(bone).copied()?,
            hinge: self.joints.get(bone).copied().flatten(),
            parent: self.parents.get(bone).copied().flatten(),
        })
    }

    /// Handle of the body built for chain bone `index` — the §4 seam for
    /// hosts driving motors/impacts through `get_body_mut`.
    // qual:api — host seam; None without the chain or the bone.
    pub fn body_handle(&self, index: usize) -> Option<BodyHandle> {
        self.bodies.get(index).copied()
    }

    /// Runs `f` over the articulated chain (the stepping seam of §4);
    /// `None` means the chain lock is poisoned (bad skin downstream).
    // qual:api — stepping seam for hosts; the fixed-step bridge lands in
    // slice 3 (the chain does not step yet — see design §5).
    pub fn with_engine<R>(&self, f: impl FnOnce(&mut FeatherstoneEngine) -> R) -> Option<R> {
        self.engine.lock().ok().map(|mut engine| f(&mut engine))
    }

    /// Model→local sample of §4.1: writes each bone's physical local pose
    /// into `out[bone.joint]` (the skeleton slot the blend mixes). The
    /// chain and the sample live in the skeleton's **model space** — the
    /// bodies sit at the bind fold of [`Skeleton::rest`] and the chain
    /// root is taken against the identity matrix; the entity's
    /// `GlobalTransform` enters later, in `compose_model_matrices`, as on
    /// the animation side. A bone local is taken against its parent bone's
    /// body: rotation `R_p⁻¹·R_c`, translation divided per axis by the
    /// parent bone's accumulated bind scale (the very factor `bind_worlds`
    /// folded in for its joint — rigid bodies carry no scale, so the seam
    /// undoes it to land back in the skeleton's scaled local frames). The
    /// output scale is left at one: the blend keeps the animated scale. Fails on a
    /// poisoned lock, a missing body or a non-finite pose, writing nothing
    /// further. Hot path: no allocations.
    pub fn sample_local(&self, desc: &RagdollDesc, out: &mut [JointTrs]) -> bool {
        let Ok(engine) = self.engine.lock() else {
            return false;
        };
        for (index, bone) in desc.bones.iter().enumerate() {
            let Some(handle) = self.bodies.get(index).copied() else {
                return false;
            };
            let Some(body) = engine.get_body(handle) else {
                return false;
            };
            if !body.position.is_finite() || !body.orientation.is_finite() {
                return false;
            }
            let (translation, rotation) = match self.parents.get(index).copied().flatten() {
                Some(parent) => {
                    let Some((_, _, parent_scale)) = self.bind.get(parent).copied() else {
                        return false;
                    };
                    let Some(handle) = self.bodies.get(parent).copied() else {
                        return false;
                    };
                    let Some(parent_body) = engine.get_body(handle) else {
                        return false;
                    };
                    if !parent_body.position.is_finite() || !parent_body.orientation.is_finite() {
                        return false;
                    }
                    let inverse = parent_body.orientation.inverse();
                    let rotation = inverse * body.orientation;
                    let offset = inverse * (body.position - parent_body.position);
                    if parent_scale.min_element() == 0.0 || !parent_scale.is_finite() {
                        return false;
                    }
                    (offset / parent_scale, rotation)
                }
                None => (body.position, body.orientation),
            };
            let Some(slot) = out.get_mut(usize::from(bone.joint)) else {
                return false;
            };
            *slot = JointTrs {
                translation,
                rotation,
                scale: Vec3::ONE,
            };
        }
        true
    }
}

/// Assembles the physical chain of `desc` under `skeleton` at the bind
/// pose (§3): every bone becomes a dynamic link whose frame is its joint
/// bind frame, hinges sit on the bind anchors, and the root link takes
/// `desc.base`. Refuses (writing nothing) when [`RagdollDesc::validate`]
/// does, or [`RagdollError::Assemble`] when physics cannot assemble a
/// hinge of an otherwise valid description.
pub fn build_ragdoll(
    skeleton: &Skeleton,
    desc: &RagdollDesc,
    gravity: Vec3,
) -> Result<RagdollRuntime, RagdollError> {
    desc.validate(skeleton)?;
    let joint_bind = bind_worlds(skeleton);
    let mut engine = FeatherstoneEngine::new(gravity);
    let mut handles = Vec::with_capacity(desc.bones.len());
    let root_bone = chain_root(&desc.bones);
    let base = match desc.base {
        RagdollBase::Fixed => BaseJoint::Fixed,
        RagdollBase::Floating => BaseJoint::Floating,
    };
    for (index, bone) in desc.bones.iter().enumerate() {
        let (position, orientation, _) = joint_bind[bone.joint.index()];
        let mut body = bone_body(bone, position);
        body.orientation = orientation;
        handles.push(if index == root_bone {
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
        let hinge = hinge_joint(bone, &joint_bind, &desc.bones[parent])?;
        let handle = engine
            .add_joint(handles[parent], handles[index], hinge)
            .map_err(|_| RagdollError::Assemble)?;
        engine.set_joint_damping(handle, bone.motor.damping);
        engine.set_joint_friction(handle, bone.motor.friction);
        joints.push(Some(handle));
    }
    let parents = (0..desc.bones.len())
        .map(|index| chain_parent(index, &desc.bones))
        .collect();
    // Bind frames ride per bone, not per joint slot: on sparse chains a
    // bone's slot in the description differs from its skeleton joint, and
    // the inverse seam indexes the parent by bone (§4.1).
    let bind = desc
        .bones
        .iter()
        .map(|bone| joint_bind[bone.joint.index()])
        .collect();
    Ok(RagdollRuntime {
        engine: Arc::new(Mutex::new(engine)),
        bodies: handles,
        joints,
        bind,
        parents,
    })
}

/// Bind-pose frames (world position, world rotation, accumulated world
/// scale) per joint, folded from [`Skeleton::rest`] along the parents in
/// depth order as full T·R·S matrices — the same composition the skin's
/// `compose_model_matrices` uses, so parent scale lands in the world
/// positions (typical glTF rigs scale the root by 0.01). Only position and
/// rotation enter physics; the scale rides along for the inverse seam of
/// [`RagdollRuntime::sample_local`]. Leaf.
fn bind_worlds(skeleton: &Skeleton) -> Vec<(Vec3, Quat, Vec3)> {
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
    let mut matrices = vec![Mat4::IDENTITY; count];
    let mut out = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); count];
    for &joint in &order {
        let local = skeleton.rest[joint];
        let local =
            Mat4::from_scale_rotation_translation(local.scale, local.rotation, local.translation);
        let world = match skeleton.parents[joint] {
            Some(parent) => matrices[parent.index()] * local,
            None => local,
        };
        matrices[joint] = world;
        let (scale, rotation, position) = world.to_scale_rotation_translation();
        out[joint] = (position, rotation, scale);
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
pub(crate) fn hinge_joint(
    bone: &RagdollBone,
    bind: &[(Vec3, Quat, Vec3)],
    parent: &RagdollBone,
) -> Result<JointKind, RagdollError> {
    let (joint_position, joint_rotation, _) = bind[bone.joint.index()];
    let (parent_position, parent_rotation, _) = bind[parent.joint.index()];
    // `RevoluteLimit::try_new` checks exactly the window rule of
    // `RagdollDesc::validate`, so this failure is unreachable for
    // validated descriptions — it maps instead of swallowing.
    let limit = match bone.limit {
        Some(window) => Some(
            RevoluteLimit::try_new(Radians::new(window.min), Radians::new(window.max))
                .ok_or(RagdollError::Limit)?,
        ),
        None => None,
    };
    Ok(JointKind::Revolute {
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
