//! Two-bone and look-at solver mechanics (`docs/animation-design.md`
//! §8.2, §8.6): forward kinematics, the aim fold and the local-TRS
//! writes. Pure functions of `(pose, constraint)`; [`FkScratch`] carries
//! the per-frame buffers, so a solve allocates nothing. The effector
//! model and the sample stage live in [`crate::ik`].

use glam::{Quat, Vec3};

use crate::ik::{LookAt, TwoBoneIk};
use crate::{JointId, JointTrs};

/// Bone-length floor for the solver: degenerate chains never move.
const ZERO_LENGTH: f32 = 1e-6;

/// Squared length below which an aim axis or a target offset is not a
/// direction. Shared by the look-at solver and the effector validation.
const AIM_EPSILON_SQ: f32 = 1e-12;

/// A finite direction: not NaN and longer than [`AIM_EPSILON_SQ`].
pub(crate) fn axis_ok(axis: Vec3) -> bool {
    axis.is_finite() && axis.length_squared() > AIM_EPSILON_SQ
}

/// The aim of one chain: the model-space parent rotations of the base
/// and hinge, the model-space deltas that put the effector on the
/// target, and how far to apply them.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Aim {
    parent_root: Quat,
    parent_mid: Quat,
    delta_root: Quat,
    delta_mid: Quat,
    weight: f32,
}

/// The rotation that takes direction `from` onto direction `to` — the
/// solver's rotation construction, its own seam.
fn arc_rotation(from: Vec3, to: Vec3) -> Quat {
    Quat::from_rotation_arc(from, to)
}

/// Reusable forward-kinematics buffers of the IK solvers (design §8.2).
///
/// The caller keeps one across calls — hosts hold it in their own state,
/// the sampler keeps one in its frame scratch — so [`solve_two_bone_ik`]
/// and [`solve_look_at`] allocate nothing per solve (the same
/// `*Scratch` contract as the fade shares and the tree weights). Filled
/// in place: one buffer serves skeletons of any joint count back to
/// back.
#[derive(Debug, Default)]
pub struct FkScratch {
    /// Model-space rotations of every joint, parents before children.
    pub(crate) model_rot: Vec<Quat>,
    /// Model-space positions of every joint, parents before children.
    pub(crate) model_pos: Vec<Vec3>,
}

/// Forward kinematics into `buf` (in place): model-space rotations and
/// positions of every joint (parents before children). Pure.
pub(crate) fn fk_model(trs: &[JointTrs], parents: &[Option<JointId>], buf: &mut FkScratch) {
    buf.model_rot.clear();
    buf.model_rot.resize(trs.len(), Quat::IDENTITY);
    buf.model_pos.clear();
    buf.model_pos.resize(trs.len(), Vec3::ZERO);
    for (index, local) in trs.iter().enumerate() {
        let (rotation, origin) = match parents.get(index).copied().flatten() {
            Some(parent) if parent.index() < index => (
                buf.model_rot[parent.index()] * local.rotation,
                buf.model_pos[parent.index()] + buf.model_rot[parent.index()] * local.translation,
            ),
            _ => (local.rotation, local.translation),
        };
        buf.model_rot[index] = rotation;
        buf.model_pos[index] = origin;
    }
}

/// The aim that puts the `root → mid → end` chain on `target` (clamped
/// to arm reach) — or `None` for a non-chain or degenerate geometry.
/// Pure: validation, the fold and the bend axis kept from the pose.
pub(crate) fn fold_chain(
    fk: &FkScratch,
    parents: &[Option<JointId>],
    chain: (usize, usize, usize),
    target: Vec3,
    weight: f32,
) -> Option<Aim> {
    let (root, mid, end) = chain;
    let bound = root.max(mid).max(end);
    if bound >= fk.model_pos.len()
        || parents.len() != fk.model_pos.len()
        || !target.is_finite()
        || weight <= 0.0
    {
        return None;
    }
    // Chain check: the walk from `end` upward passes `mid`, then `root`.
    let (mut found_mid, mut found_root) = (false, false);
    let mut cursor = parents[end];
    for _ in 0..parents.len() {
        let Some(id) = cursor else {
            break;
        };
        let index = id.index();
        if index == mid {
            found_mid = true;
        }
        if index == root {
            found_root = found_mid;
        }
        if index >= parents.len() {
            break;
        }
        cursor = parents[index];
    }
    if !found_root {
        return None;
    }
    let (model_rot, model_pos) = (&fk.model_rot[..], &fk.model_pos[..]);
    let (u, v) = (
        model_pos[mid] - model_pos[root],
        model_pos[end] - model_pos[mid],
    );
    let (l1, l2) = (u.length(), v.length());
    if l1 <= ZERO_LENGTH || l2 <= ZERO_LENGTH {
        return None;
    }
    let to_target = target - model_pos[root];
    let d_raw = to_target.length();
    if d_raw <= ZERO_LENGTH {
        return None;
    }
    let dir_t = to_target / d_raw;
    // Bend axis: keep the pose's own bend plane; a straight chain on the
    // target line bends in a deterministic arbitrary plane.
    let mut side = u / l1 - dir_t * (u / l1).dot(dir_t);
    if side.length_squared() <= ZERO_LENGTH * ZERO_LENGTH {
        side = v / l2 - dir_t * (v / l2).dot(dir_t);
    }
    if side.length_squared() <= ZERO_LENGTH * ZERO_LENGTH {
        side = dir_t.any_orthonormal_vector();
    }
    let side = side.normalize();
    let min_reach = (l1 - l2).abs() + ZERO_LENGTH;
    let d = d_raw.clamp(min_reach, (l1 + l2 - ZERO_LENGTH).max(min_reach));
    let cos_root = ((l1 * l1 + d * d - l2 * l2) / (2.0 * l1 * d)).clamp(-1.0, 1.0);
    let (sin_root, cos_root) = cos_root.acos().sin_cos();
    let u_desired = dir_t * cos_root + side * sin_root;
    let v_desired = (dir_t * d - u_desired * l1).normalize_or_zero();
    if v_desired == Vec3::ZERO {
        return None;
    }
    let delta_root = arc_rotation(u / l1, u_desired);
    let delta_mid = arc_rotation(delta_root * (v / l2), v_desired);
    Some(Aim {
        parent_root: parents[root]
            .filter(|id| id.index() < model_rot.len())
            .map_or(Quat::IDENTITY, |id| model_rot[id.index()]),
        parent_mid: parents[mid]
            .filter(|id| id.index() < model_rot.len())
            .map_or(Quat::IDENTITY, |id| model_rot[id.index()]),
        delta_root,
        delta_mid,
        weight,
    })
}

/// Writes `aim` back as local rotations of the chain base and hinge,
/// slerped by the aim weight. Pure.
fn write_aim(trs: &mut [JointTrs], chain: (usize, usize, usize), aim: Aim) {
    let (root, mid, _) = chain;
    let root_frame = aim.parent_root;
    let mid_frame = aim.delta_root * aim.parent_mid;
    let solved_root = root_frame.inverse() * aim.delta_root * root_frame * trs[root].rotation;
    let solved_mid = mid_frame.inverse() * aim.delta_mid * mid_frame * trs[mid].rotation;
    trs[root].rotation = if aim.weight >= 1.0 {
        solved_root
    } else {
        trs[root].rotation.slerp(solved_root, aim.weight)
    };
    trs[mid].rotation = if aim.weight >= 1.0 {
        solved_mid
    } else {
        trs[mid].rotation.slerp(solved_mid, aim.weight)
    };
}

/// Solves one two-bone constraint against a local-TRS pose.
///
/// Rotates `root` and `mid` in their local frames so `end` reaches
/// `constraint.target` (clamped to arm reach), slerped by the
/// constraint weight. `parents` is the skeleton parent chain (parents
/// before children); `fk` is the caller's reusable [`FkScratch`] — one
/// scratch serves every solve of a frame. Returns whether the pose
/// changed; degenerate input (bad joints, non-chain, zero-length
/// bones, zero-length target offset, weight `0`) is a strict no-op.
pub fn solve_two_bone_ik(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    constraint: &TwoBoneIk,
    fk: &mut FkScratch,
) -> bool {
    let chain = (
        constraint.root.index(),
        constraint.mid.index(),
        constraint.end.index(),
    );
    fk_model(trs, parents, fk);
    let aim = fold_chain(
        fk,
        parents,
        chain,
        constraint.target,
        constraint.weight.get(),
    );
    let Some(aim) = aim else {
        return false;
    };
    write_aim(trs, chain, aim);
    true
}

/// Model-space rotation of `index`'s parent (`IDENTITY` at a root).
pub(crate) fn parent_model_rot(
    model_rot: &[Quat],
    parents: &[Option<JointId>],
    index: usize,
) -> Quat {
    match parents.get(index).copied().flatten() {
        Some(parent) if parent.index() < index && parent.index() < model_rot.len() => {
            model_rot[parent.index()]
        }
        _ => Quat::IDENTITY,
    }
}

/// Local rotation that points `look.aim` at `look.target`, or `None`
/// when the aim is degenerate. `fk` is the caller's [`FkScratch`],
/// filled for this pose.
fn look_at_local(
    trs: &[JointTrs],
    parents: &[Option<JointId>],
    look: &LookAt,
    fk: &mut FkScratch,
) -> Option<Quat> {
    let index = look.joint.index();
    if index >= trs.len() || parents.len() != trs.len() || look.weight.get() <= 0.0 {
        return None;
    }
    if !look.target.is_finite() || !axis_ok(look.aim) {
        return None;
    }
    fk_model(trs, parents, fk);
    let (model_rot, model_pos) = (&fk.model_rot[..], &fk.model_pos[..]);
    let offset = look.target - model_pos[index];
    let current = model_rot[index] * look.aim;
    if !axis_ok(offset) || !axis_ok(current) {
        return None;
    }
    let delta = arc_rotation(current.normalize(), offset.normalize());
    let parent = parent_model_rot(model_rot, parents, index);
    Some(parent.inverse() * delta * model_rot[index])
}

/// Solves one look-at against a local-TRS pose.
///
/// Rotates `look.joint` in its parent frame so its local `aim` axis
/// points at `look.target`, slerped by the weight. `fk` is the
/// caller's reusable [`FkScratch`]. Returns whether the pose changed;
/// degenerate input (joint out of range, weight `0`, non-finite or
/// zero-length `aim`, zero-length target offset) is a strict no-op.
pub fn solve_look_at(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    look: &LookAt,
    fk: &mut FkScratch,
) -> bool {
    let Some(solved) = look_at_local(trs, parents, look, fk) else {
        return false;
    };
    let index = look.joint.index();
    let weight = look.weight.get();
    trs[index].rotation = if weight >= 1.0 {
        solved
    } else {
        trs[index].rotation.slerp(solved, weight)
    };
    true
}
