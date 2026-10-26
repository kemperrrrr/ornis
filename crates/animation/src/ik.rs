//! IK effectors (`docs/animation-design.md` §8.2, §8.6): the two-bone
//! chain, the single-joint look-at and the ordered per-root stack. A
//! separate module from the pose pipeline because authoring data,
//! chain geometry and the sample stage are different responsibilities.
//! Full-body IK and a control rig are not this layer.

use glam::{Quat, Vec3};

use ornis_core::Clamped01;

use crate::{JointId, JointTrs};

/// Bone-length floor for the solver: degenerate chains never move.
const ZERO_LENGTH: f32 = 1e-6;

/// Squared length below which an aim axis or a target offset is not a
/// direction. Shared by the look-at solver and [`Effector::installable`].
const AIM_EPSILON_SQ: f32 = 1e-12;

/// One two-bone IK constraint (design §8.2): aim the `root → mid → end`
/// chain (knee/elbow chains) at `target`.
///
/// Pure data: the solver [`solve_two_bone_ik`] is a stateless function of
/// `(pose, target)` — determinism is the determinism of its input. The
/// pole (bend plane) is implicit: the solved chain bends in its own
/// current plane (a straight chain at the target line bends in a
/// deterministic arbitrary plane).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoneIk {
    /// Chain base joint; its local rotation is aimed at the target.
    pub root: JointId,
    /// Hinge joint (knee/elbow); its local rotation folds the chain.
    pub mid: JointId,
    /// Effector joint the target is reached with.
    pub end: JointId,
    /// Goal position of `end`, in the skeleton model space of the root
    /// (the seam to world space is the host's, as with ragdoll §8.5).
    pub target: Vec3,
    /// Blend strength in `[0, 1]`: `1` aims as solved, `0.5` slerps the
    /// local rotations halfway (the effector then falls short of the
    /// target by exactly that much).
    pub weight: Clamped01,
}

/// One authored aim on a skeleton root (design §8.6).
///
/// `TwoBone` folds a knee/elbow chain; `LookAt` turns one joint so a
/// local axis points at a target. [`IkStack`] applies them in order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Effector {
    /// Two-bone chain (knee/elbow), design §8.2.
    TwoBone(TwoBoneIk),
    /// Single joint aiming a local axis at a point, design §8.6.
    LookAt(LookAt),
}

impl Effector {
    /// Whether installing this effector on a skeleton of `joints` bones
    /// is well-formed: three distinct in-range chain joints and a finite
    /// target, or an in-range look-at joint, a finite target and a finite
    /// non-zero aim. Chain ancestry is checked when the solver runs.
    pub fn installable(&self, joints: usize) -> bool {
        match self {
            Self::TwoBone(chain) => chain.installable(joints),
            Self::LookAt(look) => look.installable(joints),
        }
    }
}

impl TwoBoneIk {
    /// In-range distinct joints and a finite target (design §8.2).
    fn installable(&self, joints: usize) -> bool {
        let (root, mid, end) = (self.root.index(), self.mid.index(), self.end.index());
        root < joints
            && mid < joints
            && end < joints
            && root != mid
            && mid != end
            && root != end
            && self.target.is_finite()
    }
}

/// Aim `joint`'s local `aim` axis at `target` (design §8.6).
///
/// Pure data: [`solve_look_at`] is a stateless function of
/// `(pose, target, aim)`. `aim` is a direction in the joint's local
/// frame; its length does not matter past a non-zero finite vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LookAt {
    /// Joint whose local rotation is aimed.
    pub joint: JointId,
    /// Goal position in the skeleton model space of the root (the seam
    /// to world space is the host's, as with [`TwoBoneIk::target`]).
    pub target: Vec3,
    /// Local-space axis that should point at `target`.
    pub aim: Vec3,
    /// Blend strength in `[0, 1]`: `1` writes the solved local rotation,
    /// `0` leaves the joint alone.
    pub weight: Clamped01,
}

impl LookAt {
    /// In-range joint, finite target, finite non-zero aim.
    fn installable(&self, joints: usize) -> bool {
        self.joint.index() < joints && self.target.is_finite() && axis_ok(self.aim)
    }
}

/// Ordered IK effectors of one skeleton root — the hot lane on the
/// root (`Clone + Send + Sync` per the contract).
///
/// Installed explicitly (`SmartStore::insert` / `set_skel_ik`); a root
/// without the stack is never touched. Effectors apply in vector order
/// after the tree/layer blend and **before** the ragdoll blend (design
/// §8.2: where physics weights are up, physics wins — IK never argues
/// with ragdoll).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IkStack {
    /// Effectors applied in order; later ones solve the pose the earlier
    /// ones produced.
    pub effectors: Vec<Effector>,
}

/// The aim of one chain: the model-space parent rotations of the base
/// and hinge, the model-space deltas that put the effector on the
/// target, and how far to apply them.
#[derive(Debug, Clone, Copy)]
struct Aim {
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
    model_rot: Vec<Quat>,
    /// Model-space positions of every joint, parents before children.
    model_pos: Vec<Vec3>,
}

/// Forward kinematics into `buf` (in place): model-space rotations and
/// positions of every joint (parents before children). Pure.
fn fk_model(trs: &[JointTrs], parents: &[Option<JointId>], buf: &mut FkScratch) {
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
fn fold_chain(
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

/// A finite direction: not NaN and longer than [`AIM_EPSILON_SQ`].
fn axis_ok(axis: Vec3) -> bool {
    axis.is_finite() && axis.length_squared() > AIM_EPSILON_SQ
}

/// Model-space rotation of `index`'s parent (`IDENTITY` at a root).
fn parent_model_rot(model_rot: &[Quat], parents: &[Option<JointId>], index: usize) -> Quat {
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

/// Solves one effector against the pose the earlier effectors left.
fn apply_effector(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    effector: &Effector,
    fk: &mut FkScratch,
) {
    match effector {
        Effector::TwoBone(constraint) => {
            solve_two_bone_ik(trs, parents, constraint, fk);
        }
        Effector::LookAt(look) => {
            solve_look_at(trs, parents, look, fk);
        }
    }
}

/// Applies `stack`'s effectors in order to the local-TRS pose — the
/// pipeline stage (design §8.2, §8.6), shared by every skel sample
/// path. `fk` is the frame's shared [`FkScratch`]: the stack runs
/// allocation-free.
pub(crate) fn apply_ik_stack(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    stack: &IkStack,
    fk: &mut FkScratch,
) {
    for effector in &stack.effectors {
        apply_effector(trs, parents, effector, fk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A straight `count`-joint chain along +X with unit bones; the
    /// root sits at the origin.
    fn chain(count: usize) -> (Vec<JointTrs>, Vec<Option<JointId>>) {
        let trs = (0..count)
            .map(|index| JointTrs {
                translation: if index == 0 { Vec3::ZERO } else { Vec3::X },
                rotation: Quat::IDENTITY,
                scale: Vec3::ONE,
            })
            .collect();
        let parents = (0..count)
            .map(|index| {
                if index == 0 {
                    None
                } else {
                    Some(JointId::from_raw((index - 1) as u32))
                }
            })
            .collect();
        (trs, parents)
    }

    /// The model-space position of `joint` after `solve` (a chain: each
    /// parent is the previous joint).
    fn model_position(trs: &[JointTrs], joint: usize) -> Vec3 {
        let mut position = trs[0].translation;
        let mut rotation = trs[0].rotation;
        for local in trs.iter().skip(1).take(joint) {
            position += rotation * local.translation;
            rotation *= local.rotation;
        }
        position
    }

    /// Root at 0, mid at 1, end at 2 (a two-bone chain), solved to
    /// `target` at `weight`.
    fn solve_to(target: Vec3, weight: f32) -> (Vec<JointTrs>, Vec<Option<JointId>>, bool) {
        let (mut trs, parents) = chain(3);
        let moved = solve_two_bone_ik(
            &mut trs,
            &parents,
            &TwoBoneIk {
                root: JointId::from_raw(0),
                mid: JointId::from_raw(1),
                end: JointId::from_raw(2),
                target,
                weight: Clamped01::new(weight),
            },
            &mut FkScratch::default(),
        );
        (trs, parents, moved)
    }

    #[test]
    fn a_reachable_target_aims_the_effector_at_it() {
        let (trs, _parents, moved) = solve_to(Vec3::new(1.0, 1.0, 0.0), 1.0);
        assert!(moved);
        let end = model_position(&trs, 2);
        assert!(
            (end - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-4,
            "end at {end:?}"
        );
    }

    #[test]
    fn an_unreachable_target_stretches_to_full_reach() {
        // On-axis of a straight chain: the degenerate bend-plane case.
        let (trs, _parents, moved) = solve_to(Vec3::new(10.0, 0.0, 0.0), 1.0);
        assert!(moved);
        let end = model_position(&trs, 2);
        assert!((end - Vec3::X * 2.0).length() < 1e-4, "end at {end:?}");
    }

    #[test]
    fn bone_lengths_survive_the_solver() {
        let (trs, _parents, _) = solve_to(Vec3::new(0.5, 1.2, -0.3), 1.0);
        let l1 = (model_position(&trs, 1) - model_position(&trs, 0)).length();
        let l2 = (model_position(&trs, 2) - model_position(&trs, 1)).length();
        assert!((l1 - 1.0).abs() < 1e-4 && (l2 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn zero_weight_is_a_noop() {
        let (trs, _parents, moved) = solve_to(Vec3::new(1.0, 1.0, 0.0), 0.0);
        assert!(!moved);
        assert_eq!(trs[0].rotation, Quat::IDENTITY);
        assert_eq!(trs[1].rotation, Quat::IDENTITY);
    }

    #[test]
    fn half_weight_slerps_halfway() {
        let (full, _parents, _) = solve_to(Vec3::new(1.0, 1.0, 0.0), 1.0);
        let (half, _parents_half, moved) = solve_to(Vec3::new(1.0, 1.0, 0.0), 0.5);
        assert!(moved);
        let halfway = Quat::IDENTITY.slerp(full[0].rotation, 0.5);
        assert!(half[0].rotation.abs_diff_eq(halfway, 1e-4));
    }

    #[test]
    fn degenerate_and_invalid_chains_never_move() {
        // One scratch serves every solve of a test — and of a frame.
        let mut fk = FkScratch::default();
        // Zero-length first bone.
        let (mut trs, parents) = chain(3);
        trs[1].translation = Vec3::ZERO;
        assert!(!solve_two_bone_ik(
            &mut trs,
            &parents,
            &TwoBoneIk {
                root: JointId::from_raw(0),
                mid: JointId::from_raw(1),
                end: JointId::from_raw(2),
                target: Vec3::new(1.0, 1.0, 0.0),
                weight: Clamped01::new(1.0),
            },
            &mut fk,
        ));
        // End is not a descendant of mid (siblings).
        let (mut trs, mut parents) = chain(3);
        parents[2] = parents[1];
        assert!(!solve_two_bone_ik(
            &mut trs,
            &parents,
            &TwoBoneIk {
                root: JointId::from_raw(0),
                mid: JointId::from_raw(1),
                end: JointId::from_raw(2),
                target: Vec3::new(1.0, 1.0, 0.0),
                weight: Clamped01::new(1.0),
            },
            &mut fk,
        ));
        assert_eq!(trs[0].rotation, Quat::IDENTITY);
    }

    #[test]
    fn stacks_apply_in_order() {
        let (mut trs, parents) = chain(3);
        let stack = IkStack {
            effectors: vec![
                Effector::TwoBone(TwoBoneIk {
                    root: JointId::from_raw(0),
                    mid: JointId::from_raw(1),
                    end: JointId::from_raw(2),
                    target: Vec3::new(1.0, 1.0, 0.0),
                    weight: Clamped01::new(1.0),
                }),
                Effector::TwoBone(TwoBoneIk {
                    root: JointId::from_raw(0),
                    mid: JointId::from_raw(1),
                    end: JointId::from_raw(2),
                    target: Vec3::new(1.0, -1.0, 0.0),
                    weight: Clamped01::new(1.0),
                }),
            ],
        };
        apply_ik_stack(&mut trs, &parents, &stack, &mut FkScratch::default());
        // The later constraint re-solves the pose the earlier one left.
        let end = model_position(&trs, 2);
        assert!(
            (end - Vec3::new(1.0, -1.0, 0.0)).length() < 1e-4,
            "end at {end:?}"
        );
    }

    /// Look-at on the chain root: local +X toward world +Y swings the
    /// child onto +Y. Weight 0 and a zero axis leave the pose alone.
    #[test]
    fn look_at_aims_the_child_and_a_zero_weight_does_not() {
        let mut fk = FkScratch::default();
        let (mut trs, parents) = chain(2);
        let aim = LookAt {
            joint: JointId::from_raw(0),
            target: Vec3::Y,
            aim: Vec3::X,
            weight: Clamped01::new(1.0),
        };
        assert!(solve_look_at(&mut trs, &parents, &aim, &mut fk));
        let child = model_position(&trs, 1);
        assert!((child - Vec3::Y).length() < 1e-4, "child at {child:?}");

        let (mut trs, parents) = chain(2);
        let held = LookAt {
            weight: Clamped01::new(0.0),
            ..aim
        };
        assert!(!solve_look_at(&mut trs, &parents, &held, &mut fk));
        assert_eq!(trs[0].rotation, Quat::IDENTITY);
        assert!(!solve_look_at(
            &mut trs,
            &parents,
            &LookAt {
                aim: Vec3::ZERO,
                ..aim
            },
            &mut fk,
        ));
        assert_eq!(trs[0].rotation, Quat::IDENTITY);
    }

    /// The solved rotation is written in the parent frame: a chain
    /// already bent onto +Y still aims the child's local +X at world +X.
    #[test]
    fn look_at_writes_in_the_parent_frame() {
        let (mut trs, parents) = chain(3);
        trs[0].rotation = Quat::from_rotation_arc(Vec3::X, Vec3::Y);
        let moved = solve_look_at(
            &mut trs,
            &parents,
            &LookAt {
                joint: JointId::from_raw(1),
                target: Vec3::new(1.0, 1.0, 0.0),
                aim: Vec3::X,
                weight: Clamped01::new(1.0),
            },
            &mut FkScratch::default(),
        );
        assert!(moved);
        let child = model_position(&trs, 2);
        assert!(
            (child - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-4,
            "child at {child:?}"
        );
    }

    /// The scratch is filled in place: one buffer serves solves of
    /// different skeletons back to back (§8.2, no per-frame allocs).
    #[test]
    fn one_fk_scratch_serves_solves_of_any_skeleton() {
        let mut fk = FkScratch::default();
        let constraint = TwoBoneIk {
            root: JointId::from_raw(0),
            mid: JointId::from_raw(1),
            end: JointId::from_raw(2),
            target: Vec3::new(1.0, 1.0, 0.0),
            weight: Clamped01::new(1.0),
        };
        let (mut short, short_parents) = chain(3);
        assert!(solve_two_bone_ik(
            &mut short,
            &short_parents,
            &constraint,
            &mut fk
        ));
        assert_eq!(fk.model_pos.len(), 3);
        // A different skeleton refills the same buffer in place.
        let (mut long, long_parents) = chain(6);
        assert!(solve_look_at(
            &mut long,
            &long_parents,
            &LookAt {
                joint: JointId::from_raw(5),
                target: Vec3::new(6.0, 1.0, 0.0),
                aim: Vec3::X,
                weight: Clamped01::new(1.0),
            },
            &mut fk,
        ));
        assert_eq!(fk.model_pos.len(), 6);
        // And back down: the refill never carries stale joints.
        let (mut short, short_parents) = chain(3);
        assert!(solve_two_bone_ik(
            &mut short,
            &short_parents,
            &constraint,
            &mut fk
        ));
        assert_eq!(fk.model_pos.len(), 3);
        let end = model_position(&short, 2);
        assert!(
            (end - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-4,
            "end at {end:?}"
        );
    }
}
