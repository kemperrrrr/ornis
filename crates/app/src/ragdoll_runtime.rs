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
use ornis_core::Entity;
use ornis_core::units::Radians;
use ornis_physics::articulated::{BaseJoint, FeatherstoneEngine};
use ornis_physics::body::{BodyHandle, RigidBody};
use ornis_physics::engine::PhysicsEngine;
use ornis_physics::joint::{JointHandle, JointKind, RevoluteLimit};

/// Full edge from a half-extent (shape volumes below).
pub(crate) const FULL_EDGE: f32 = 2.0;
/// Hemispherical caps share of a capsule volume (`4/3`).
const CAPSULE_CAPS_SHARE: f32 = 4.0 / 3.0;

/// The simulated chain behind a ragdoll root (§11, G5 path): which solver
/// owns the bones. Skinned-pose ragdoll rides the ABA chain (contacts
/// through the §8.5 proxy coupling); contact/rope scenes ride the
/// world's maximal-coordinate joints (G5) directly.
#[derive(Clone)]
pub enum RagdollChain {
    /// Featherstone ABA chain (one per ragdoll root, shared on clone).
    Aba(Arc<Mutex<FeatherstoneEngine>>),
    /// World-solver ragdoll: the bones are world bodies and `JointKind`
    /// joints of `PhysicsRuntime`; no chain engine at all.
    G5,
}

/// Hot state of one ragdoll root: its own articulated chain plus the
/// bind-pose frames the world→local seam reads against (§4.1). Registered
/// as a hot lane on the skeleton root (the lane's `Clone` bound is why the
/// chain rides behind an `Arc` — cloning the runtime shares the chain, it
/// never copies it); the chain holds no state but the physics bodies
/// themselves.
#[derive(Clone)]
pub struct RagdollRuntime {
    /// The simulated chain behind the root (§11): the ABA engine, or
    /// nothing — the G5 path rides world bodies and `JointKind` joints of
    /// the world solver (`PhysicsRuntime`), contacts and ropes native.
    pub(crate) chain: RagdollChain,
    /// Physical link per bone (description order). Empty on the G5 path —
    /// the links are world bodies (`templates` below), not chain handles.
    pub(crate) bodies: Vec<BodyHandle>,
    /// Bind-pose body templates per bone (G5 path only, §11): the proxies
    /// insert them as [`RigidBody`] components and the world solver binds
    /// them as its own bodies. Empty on the ABA path — its template comes
    /// from the live chain body ([`RagdollRuntime::body_template`]).
    pub(crate) templates: Vec<RigidBody>,
    /// Hinge per bone; `None` on the chain root.
    pub(crate) joints: Vec<Option<JointHandle>>,
    /// Bind-pose frame of each bone's joint: world position, world
    /// rotation and the accumulated world scale — the factor the inverse
    /// seam of [`RagdollRuntime::sample_local`] undoes (bodies are rigid
    /// and carry no scale). Indexed by bone (description order), not by
    /// skeleton joint slot — on sparse chains the two differ, and the
    /// seam must read the parent *bone's* fold.
    pub(crate) bind: Vec<(Vec3, Quat, Vec3)>,
    /// Parent bone index per bone; `None` on the chain root.
    pub(crate) parents: Vec<Option<usize>>,
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

/// Marker of one world-proxy entity of the contact coupling
/// (`docs/ragdoll-design.md` §8.5): the proxy mirrors chain bone `link` of
/// the ragdoll `root` into the world solver. [`RagdollProxySync`]
/// (crate::ragdoll_proxy) owns its `RigidBody` component — present while
/// the bone is physical, removed while animation owns the pose. Hot lane
/// (the coupling reads it every tick).
///
/// [`RagdollProxySync`]: crate::ragdoll_proxy::RagdollProxySync
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagdollProxy {
    /// Ragdoll root the proxied chain belongs to.
    pub root: Entity,
    /// Chain bone index (description order).
    pub link: usize,
}

impl RagdollRuntime {
    /// Number of bones in the chain: the chain's own link count, falling
    /// back to the handle table if the lock is poisoned.
    // qual:api — per-bone seam data for hosts; the world→local read
    // (slice 2) is its first in-tree caller.
    pub fn bone_count(&self) -> usize {
        self.with_engine(|engine| engine.body_count())
            .unwrap_or(self.bind.len())
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

    /// Body template of chain bone `index` (§8.5): a clone of the link's
    /// rigid body — shape, mass and inertia of the bone at its live pose —
    /// for hosts mirroring the link into another solver (the world proxies
    /// of the contact coupling). Callers re-aim the pose/velocities and
    /// own the collision filter.
    // qual:api — §8.5 proxy seam; the world coupling is its first caller.
    pub fn body_template(&self, index: usize) -> Option<RigidBody> {
        if self.is_g5() {
            return self.templates.get(index).cloned();
        }
        let handle = self.bodies.get(index).copied()?;
        self.with_engine(|engine| engine.get_body(handle).cloned())
            .flatten()
    }

    /// Applies one world impulse `impulse` at `point` to chain bone `link`
    /// (`docs/ragdoll-design.md` §8.5): velocity-level,
    /// momentum-conserving — the impulse rides `f_ext` for exactly one
    /// chain step. Coordinates are the chain's **model space** (hosts
    /// transform in and out). `false` without the bone or on a poisoned
    /// lock.
    // qual:api — §8.5 contact readback worker; RagdollSyncSystem is the
    // first caller.
    pub fn apply_contact_impulse(&self, link: usize, point: Vec3, impulse: Vec3) -> bool {
        let Some(handle) = self.bodies.get(link).copied() else {
            return false;
        };
        self.with_engine(|engine| {
            engine.apply_impulse(handle, point, impulse);
        })
        .is_some()
    }

    /// Runs `f` over the articulated chain (the stepping seam of §4);
    /// `None` means the chain lock is poisoned (bad skin downstream) or
    /// the root rides the G5 path (§11 — there is no chain engine).
    // qual:api — stepping seam for hosts; the fixed-step bridge lands in
    // slice 3 (the chain does not step yet — see design §5).
    pub fn with_engine<R>(&self, f: impl FnOnce(&mut FeatherstoneEngine) -> R) -> Option<R> {
        let RagdollChain::Aba(engine) = &self.chain else {
            return None;
        };
        engine.lock().ok().map(|mut engine| f(&mut engine))
    }

    /// Whether the root rides the G5 path (§11): world bodies and
    /// `JointKind` joints of the world solver, no chain engine.
    #[must_use]
    pub fn is_g5(&self) -> bool {
        matches!(self.chain, RagdollChain::G5)
    }

    /// Bind-pose frames of the bones (§4.1 seam), description order.
    pub(crate) fn bind_frames(&self) -> &[(Vec3, Quat, Vec3)] {
        &self.bind
    }

    /// Parent bone index per bone (§4.1 seam), description order.
    pub(crate) fn parents(&self) -> &[Option<usize>] {
        &self.parents
    }

    /// The G5 world joint handle of bone `link` (`None` on the root, on
    /// the ABA path, or while the bone is unbound).
    pub fn g5_joint(&self, link: usize) -> Option<JointHandle> {
        self.joints.get(link).copied().flatten()
    }

    /// Records the G5 world joint of bone `link` (bind path), or clears
    /// it (`None`) when the bone unbinds — the solver drops joints with
    /// their bodies, the table must follow (§11).
    pub(crate) fn set_g5_joint(&mut self, link: usize, handle: Option<JointHandle>) -> bool {
        let Some(slot) = self.joints.get_mut(link) else {
            return false;
        };
        *slot = handle;
        true
    }

    /// Independent deep copy of the runtime (`docs/ragdoll-design.md`
    /// §2.2): the ABA chain state is **copied** into a fresh engine, so
    /// mutating the copy never touches this runtime's chain and vice
    /// versa — the snapshot seam for editor undo and multiplayer
    /// rollback (keep the copy, swap it back in later). Body and joint
    /// handles are dense engine indices and stay valid in the copy.
    ///
    /// The G5 path (§11) owns no simulated state of its own — its bodies
    /// and joints live in the world solver, and `deep_clone` copies only
    /// the runtime's own tables (bind frames, templates, handles); the
    /// world snapshot is `PhysicsRuntime`'s domain. A poisoned chain
    /// lock still deep-clones (the data behind the poison is readable).
    // qual:api — snapshot/undo seam for hosts; the clone-isolation tests
    // are its first in-tree callers.
    pub fn deep_clone(&self) -> Self {
        let mut copy = self.clone();
        if let RagdollChain::Aba(engine) = &self.chain {
            let fresh = match engine.lock() {
                Ok(live) => (*live).clone(),
                Err(poisoned) => poisoned.into_inner().clone(),
            };
            copy.chain = RagdollChain::Aba(Arc::new(Mutex::new(fresh)));
        }
        copy
    }

    /// Model→local sample of §4.1 (ABA seam): writes each bone's physical
    /// local pose into `out[bone.joint]` from the chain bodies. The sample
    /// lives in the skeleton's **model space** — see `sample_local_into`
    /// (the seam core, `ragdoll_g5`) for the seam itself. Fails on a
    /// poisoned lock, a missing body or a non-finite pose, writing
    /// nothing further. Hot path: no allocations.
    pub fn sample_local(&self, desc: &RagdollDesc, out: &mut [JointTrs]) -> bool {
        self.with_engine(|engine| {
            crate::ragdoll_g5::sample_local_into(desc, out, &self.bind, &self.parents, |index| {
                let handle = self.bodies.get(index).copied()?;
                let body = engine.get_body(handle)?;
                Some((body.position, body.orientation))
            })
        })
        .unwrap_or(false)
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
        chain: RagdollChain::Aba(Arc::new(Mutex::new(engine))),
        bodies: handles,
        templates: Vec::new(),
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
pub(crate) fn bind_worlds(skeleton: &Skeleton) -> Vec<(Vec3, Quat, Vec3)> {
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
pub(crate) fn bone_body(bone: &RagdollBone, position: Vec3) -> RigidBody {
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
pub(crate) fn chain_parent(index: usize, bones: &[RagdollBone]) -> Option<usize> {
    let parent = bones[index].parent?;
    let mut found = None;
    for (candidate, bone) in bones.iter().enumerate() {
        if bone.joint == parent {
            found = Some(candidate);
        }
    }
    found
}
