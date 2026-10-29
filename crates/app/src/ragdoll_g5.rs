//! G5: the ragdoll of world bodies and `JointKind` joints
//! (`docs/ragdoll-design.md` §11): the assembly of the maximal-coordinate
//! chain behind [`crate::ragdoll_runtime::RagdollRuntime::is_g5`], the
//! world→local seam core both chain variants share, and the world joint
//! bind of the proxy drive. Contacts and ropes are the world solver's
//! own; the skinned-pose ragdoll stays on the ABA chain.

use glam::{Quat, Vec3};
use ornis_animation::{JointTrs, RagdollDesc, RagdollError, Skeleton};
use ornis_core::{ComponentStore, Entity, SmartStore};
use ornis_physics::body::RigidBody;

use crate::physics_runtime::PhysicsRuntime;
use crate::ragdoll_runtime::{
    RagdollBoneView, RagdollProxy, RagdollRuntime, bind_worlds, bone_body, chain_parent,
    hinge_joint,
};

/// Model→local seam core of §4.1 (shared by the ABA and G5 seams): writes
/// each bone's physical local pose into `out[bone.joint]` (the skeleton
/// slot the blend mixes) from the per-bone world poses `body_at` serves
/// (bone index in description order). The chain and the sample live in
/// the skeleton's **model space** — the bodies sit at the bind fold of
/// [`Skeleton::rest`] and the chain root is taken against the identity
/// matrix; the entity's `GlobalTransform` enters later, in
/// `compose_model_matrices`, as on the animation side. A bone local is
/// taken against its parent bone's body (see [`local_against_parent`]).
/// The output scale is left at one: the blend keeps the animated scale.
/// A missing body or a non-finite pose fails the sample, writing nothing
/// further. Hot path: no allocations.
pub(crate) fn sample_local_into(
    desc: &RagdollDesc,
    out: &mut [JointTrs],
    bind: &[(Vec3, Quat, Vec3)],
    parents: &[Option<usize>],
    mut body_at: impl FnMut(usize) -> Option<(Vec3, Quat)>,
) -> bool {
    for (index, bone) in desc.bones.iter().enumerate() {
        let Some((position, orientation)) = body_at(index) else {
            return false;
        };
        if !position.is_finite() || !orientation.is_finite() {
            return false;
        }
        let local = match parents.get(index).copied().flatten() {
            Some(parent) => local_against_parent(parent, position, orientation, bind, &mut body_at),
            None => Some((position, orientation)),
        };
        let Some((translation, rotation)) = local else {
            return false;
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

/// The local pose of a child bone against its parent bone's body (§4.1):
/// rotation `R_p⁻¹·R_c`, translation divided per axis by the parent
/// bone's accumulated bind scale (the very factor `bind_worlds` folded in
/// for its joint — rigid bodies carry no scale, so the seam undoes it to
/// land back in the skeleton's scaled local frames). `None` on a missing
/// bind frame, a missing parent body or a non-finite pose.
fn local_against_parent(
    parent: usize,
    position: Vec3,
    orientation: Quat,
    bind: &[(Vec3, Quat, Vec3)],
    body_at: &mut impl FnMut(usize) -> Option<(Vec3, Quat)>,
) -> Option<(Vec3, Quat)> {
    let (_, _, parent_scale) = bind.get(parent).copied()?;
    let (parent_position, parent_orientation) = body_at(parent)?;
    if !parent_position.is_finite() || !parent_orientation.is_finite() {
        return None;
    }
    if parent_scale.min_element() == 0.0 || !parent_scale.is_finite() {
        return None;
    }
    let inverse = parent_orientation.inverse();
    let offset = inverse * (position - parent_position);
    Some((offset / parent_scale, inverse * orientation))
}

/// Assembles the G5 ragdoll of `desc` under `skeleton` at the bind pose
/// (§11): no chain engine — the bones ride world bodies and `JointKind`
/// joints of the world solver ([`PhysicsRuntime`]), so contacts and ropes
/// are native and the §8.5 proxy coupling does not apply. The seam
/// (`bind`, `parents`) matches `build_ragdoll` and the body templates
/// carry the bind-pose links the proxies insert; the joint handles fill
/// in at bind time (the world owns the joints and drops them with the
/// bodies). Refuses (writing nothing) when [`RagdollDesc::validate`]
/// does, or [`RagdollError::Assemble`] when the hinge of an otherwise
/// valid description cannot be assembled — the geometry is probed here
/// so install fails, not the first bind. `desc.base` selects the root
/// behaviour on bind: `Fixed` pins the root link to the entity's bind
/// frame, `Floating` leaves it to the world.
pub fn build_g5_ragdoll(
    skeleton: &Skeleton,
    desc: &RagdollDesc,
) -> Result<RagdollRuntime, RagdollError> {
    desc.validate(skeleton)?;
    let joint_bind = bind_worlds(skeleton);
    let mut templates = Vec::with_capacity(desc.bones.len());
    let mut joints = Vec::with_capacity(desc.bones.len());
    for (index, bone) in desc.bones.iter().enumerate() {
        let (position, orientation, _) = joint_bind[bone.joint.index()];
        let mut body = bone_body(bone, position);
        body.orientation = orientation;
        templates.push(body);
        // Probe the hinge geometry now so install fails, not the first
        // bind; the world joints fill the slots at bind time (§11).
        if let Some(parent) = chain_parent(index, &desc.bones) {
            hinge_joint(bone, &joint_bind, &desc.bones[parent])?;
        }
        joints.push(None);
    }
    let parents = (0..desc.bones.len())
        .map(|index| chain_parent(index, &desc.bones))
        .collect();
    let bind = desc
        .bones
        .iter()
        .map(|bone| joint_bind[bone.joint.index()])
        .collect();
    Ok(RagdollRuntime {
        chain: crate::ragdoll_runtime::RagdollChain::G5,
        bodies: Vec::new(),
        templates,
        joints,
        bind,
        parents,
    })
}

/// The world handles a G5 bind works with (§11): the entity store of the
/// bound bodies and the solver that owns the joints.
pub(crate) struct G5World<'a> {
    /// The lane the proxy bodies live in.
    pub(crate) bodies: &'a mut ComponentStore<RigidBody>,
    /// The world solver the joints are created in.
    pub(crate) physics: &'a mut PhysicsRuntime,
}

/// G5 joint bind (§11): gives every bound bone whose parent bone is bound
/// too its world joint once — the description's own hinge geometry
/// ([`hinge_joint`], one vocabulary with the ABA chain). The solver drops
/// joints with their bodies and the unbind path clears the table, so a
/// recreated bind recreates the joint. Worker of `drive_proxies`.
pub(crate) fn bind_g5_joints(
    store: &SmartStore,
    root: Entity,
    runtime: &mut RagdollRuntime,
    desc: &RagdollDesc,
    world: &mut G5World<'_>,
) {
    let G5World { bodies, physics } = world;
    let Some(proxies) = store.read_lane::<RagdollProxy>() else {
        return;
    };
    for (index, bone) in desc.bones.iter().enumerate() {
        let Some(parent_index) = chain_parent(index, &desc.bones) else {
            continue;
        };
        if runtime.g5_joint(index).is_some() {
            continue;
        }
        let Some((child, parent)) = g5_link_pair(&proxies, root, index, parent_index) else {
            continue;
        };
        if !bodies.contains(child) || !bodies.contains(parent) {
            continue;
        }
        let Ok(hinge) = hinge_joint(bone, runtime.bind_frames(), &desc.bones[parent_index]) else {
            continue;
        };
        if let Some(handle) = physics.add_joint_between(bodies, parent, child, hinge) {
            runtime.set_g5_joint(index, Some(handle));
        }
    }
}

/// The proxy entities of a G5 link pair (`child`, `parent`), when both
/// proxies of `root` exist. Leaf lookup of [`bind_g5_joints`].
fn g5_link_pair(
    proxies: &ComponentStore<RagdollProxy>,
    root: Entity,
    index: usize,
    parent_index: usize,
) -> Option<(Entity, Entity)> {
    let bone_entity = |link: usize| {
        proxies
            .entities
            .iter()
            .zip(&proxies.data)
            .find(|(_, proxy)| proxy.root == root && proxy.link == link)
            .map(|(&entity, _)| entity)
    };
    Some((bone_entity(index)?, bone_entity(parent_index)?))
}

/// First bind of a proxy (§8.5/§11, worker of `drive_proxies`):
/// inserts its bind-pose template body with the collision filter of
/// its layer bit. Leaf.
pub(crate) fn bind_proxy_body(
    runtime: &RagdollRuntime,
    proxy: &RagdollProxy,
    entity: Entity,
    bodies: &mut ComponentStore<RigidBody>,
    bone_count: usize,
) {
    if bodies.contains(entity) {
        return;
    }
    let Some(mut body) = runtime.body_template(proxy.link) else {
        return;
    };
    body.set_collision_filter(proxy_layer(proxy.link), proxy_mask(bone_count, proxy.link));
    bodies.insert(entity, body);
}

/// §11 Fixed-base root (worker of `drive_proxies`): pins the root link
/// to the entity's bind frame (`drive_pose`, kinematic anchor) — the
/// world owns the rest. Leaf.
pub(crate) fn pin_g5_fixed_root(
    view: RagdollBoneView,
    root_pose: (Vec3, Quat),
    entity: Entity,
    bodies: &mut ComponentStore<RigidBody>,
    physics: &mut PhysicsRuntime,
) {
    let (root_position, root_rotation) = root_pose;
    let position = root_rotation * view.position + root_position;
    let rotation = root_rotation * view.rotation;
    let Some(body) = bodies.get_mut(entity) else {
        return;
    };
    body.position = position;
    body.orientation = rotation;
    body.velocity = Vec3::ZERO;
    body.angular_velocity = Vec3::ZERO;
    if let Some(handle) = physics.handle_of(entity) {
        physics.drive_pose(handle, position, rotation);
    }
}

/// The world's default collision layer bit (§8.5 proxy filters): bit 0
/// stays the world's, bone `i` takes bit `i + 1`.
const WORLD_LAYER_BIT: u32 = 1;
/// Top layer bit a proxy bone can own (§8.5): u32 bit 31 — longer chains
/// share it (their direct neighbours stay masked out).
const MAX_PROXY_CHAIN_BITS: usize = 31;
/// Adjacency half-width of the §8.5 self-collision mask: the bone and its
/// direct neighbours never collide.
const ADJACENT_BONES: usize = 1;

/// Layer bit of bone `link` (§8.5): bit 0 is the world's default layer,
/// bone `i` takes bit `i + 1` — chains past [`MAX_PROXY_CHAIN_BITS`]
/// bones share the top bit.
pub(crate) fn proxy_layer(link: usize) -> u32 {
    1u32 << ((link + 1).min(MAX_PROXY_CHAIN_BITS) as u32)
}

/// Collision mask of bone `link` (§8.5): the world's bit plus every chain
/// bone except the bone itself and its direct neighbours — parent/child
/// pairs cannot collide (the hinge keeps them together), cousins can.
pub(crate) fn proxy_mask(links: usize, link: usize) -> u32 {
    let mut mask = WORLD_LAYER_BIT;
    for other in 0..links {
        if other.abs_diff(link) > ADJACENT_BONES {
            mask |= proxy_layer(other);
        }
    }
    mask
}
