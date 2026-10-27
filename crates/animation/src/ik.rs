//! IK effectors (`docs/animation-design.md` §8.2, §8.6–§8.8): the
//! two-bone chain, the single-joint look-at, the ordered per-root stack,
//! the §8.7 base pull and the world-space [`ControlRig`]. A separate
//! module from the pose pipeline because authoring data, chain geometry
//! and the sample stage are different responsibilities.

use glam::{Mat4, Vec3};

use ornis_core::Clamped01;

use crate::ik_solve::{axis_ok, fk_model, fold_chain, parent_model_rot};
use crate::{JointId, JointTrs};

pub use crate::ik_solve::{FkScratch, solve_look_at, solve_two_bone_ik};

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

/// Optional base pull of one [`IkStack`] (design §8.7): the base joint
/// (pelvis) translates toward effector targets the chains cannot reach,
/// so the body leans/ducks into the pose instead of stretching bones.
/// Pure data; the stack stage runs the pull before the effectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasePull {
    /// Joint that carries the body; its local translation moves.
    pub joint: JointId,
    /// How much of the weighted shortfall reaches the base, `[0, 1]`:
    /// `1` puts the mean shortfall into the base, `0` leaves it alone.
    pub weight: Clamped01,
}

impl BasePull {
    /// Whether installing this base pull on a skeleton of `joints`
    /// bones is well-formed: an in-range base joint.
    pub fn installable(&self, joints: usize) -> bool {
        self.joint.index() < joints
    }
}

/// Ordered IK effectors of one skeleton root — the hot lane on the
/// root (`Clone + Send + Sync` per the contract).
///
/// Installed explicitly (`SmartStore::insert` / `set_skel_ik`); a root
/// without the stack is never touched. Effectors apply in vector order
/// after the tree/layer blend and **before** the ragdoll blend (design
/// §8.2: where physics weights are up, physics wins — IK never argues
/// with ragdoll). With a [`BasePull`], the base joint first leans into
/// the unreachable two-bone targets (design §8.7).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IkStack {
    /// Effectors applied in order; later ones solve the pose the earlier
    /// ones produced.
    pub effectors: Vec<Effector>,
    /// Optional full-body base pull, run before the effectors (§8.7).
    pub base: Option<BasePull>,
}

/// A control rig: IK goals authored in **world space** (design §8.8).
///
/// Goal data matches [`Effector`] and [`BasePull`], but every `target`
/// is a world point. [`ControlRig::into_stack`] maps the goals through
/// the root's world matrix into the model-space [`IkStack`] the sample
/// stage solves — the same «author in world, solve in model» seam as
/// ragdoll contact goals.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ControlRig {
    /// Goals applied in order, as [`IkStack::effectors`]; each `target`
    /// is a world point.
    pub goals: Vec<Effector>,
    /// Optional §8.7 base pull (joint identity — space-independent).
    pub base: Option<BasePull>,
}

impl ControlRig {
    /// The model-space [`IkStack`] for the skeleton root whose world
    /// matrix is `root_world` (its `GlobalTransform`): every goal
    /// `target` maps through the inverse, the base passes through.
    /// A singular matrix yields non-finite targets: inserted directly,
    /// the solvers' §8.7 gate skips them (no panic without `glam-assert`);
    /// through the typed facade the validation rejects the whole set.
    /// Pure.
    pub fn into_stack(self, root_world: Mat4) -> IkStack {
        let to_model = root_world.inverse();
        let effectors = self
            .goals
            .into_iter()
            .map(|goal| match goal {
                Effector::TwoBone(chain) => Effector::TwoBone(TwoBoneIk {
                    root: chain.root,
                    mid: chain.mid,
                    end: chain.end,
                    target: to_model.transform_point3(chain.target),
                    weight: chain.weight,
                }),
                Effector::LookAt(look) => Effector::LookAt(LookAt {
                    joint: look.joint,
                    target: to_model.transform_point3(look.target),
                    aim: look.aim,
                    weight: look.weight,
                }),
            })
            .collect();
        IkStack {
            effectors,
            base: self.base,
        }
    }
}

/// The model-space shift of the §8.7 base pull: the weighted mean of
/// the two-bone shortfalls (target distance past chain reach), scaled
/// by the pull weight — `None` when no effector pulls. Only effectors
/// the solver would run count: [`fold_chain`] gates non-finite targets,
/// non-chains and degenerate geometry out (a no-op effector never moves
/// the base). Reachable chains count with a zero shortfall and hold the
/// base. Pure.
fn base_shift(
    fk: &FkScratch,
    parents: &[Option<JointId>],
    effectors: &[Effector],
    weight: f32,
) -> Option<Vec3> {
    let (mut total, mut weight_sum) = (Vec3::ZERO, 0.0);
    for effector in effectors {
        let chain = match effector {
            Effector::TwoBone(chain) => chain,
            Effector::LookAt(_) => continue,
        };
        let strength = chain.weight.get();
        let triple = (chain.root.index(), chain.mid.index(), chain.end.index());
        if strength <= 0.0 || fold_chain(fk, parents, triple, chain.target, strength).is_none() {
            continue;
        }
        let (root, mid, end) = triple;
        let reach = (fk.model_pos[mid] - fk.model_pos[root]).length()
            + (fk.model_pos[end] - fk.model_pos[mid]).length();
        let to_target = chain.target - fk.model_pos[root];
        let distance = to_target.length();
        if reach <= 0.0 || distance <= 0.0 {
            continue;
        }
        total += to_target / distance * ((distance - reach).max(0.0) * strength);
        weight_sum += strength;
    }
    if weight_sum <= 0.0 {
        return None;
    }
    Some(total / weight_sum * weight)
}

/// Runs the §8.7 base pull: `base.joint`'s local translation takes the
/// weighted shortfall of the two-bone effectors, written in the joint's
/// parent frame. Strict no-op for an out-of-range base, weight `0` or a
/// non-finite shift.
fn pull_base(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    base: &BasePull,
    effectors: &[Effector],
    fk: &mut FkScratch,
) {
    let index = base.joint.index();
    if index >= trs.len() || base.weight.get() <= 0.0 {
        return;
    }
    fk_model(trs, parents, fk);
    if let Some(shift) = base_shift(fk, parents, effectors, base.weight.get())
        && shift.is_finite()
    {
        let parent = parent_model_rot(&fk.model_rot[..], parents, index);
        trs[index].translation += parent.inverse() * shift;
    }
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

/// Applies `stack` in order to the local-TRS pose — the pipeline stage
/// (design §8.2, §8.6), shared by every skel sample path: the §8.7 base
/// pull first (when set), then every effector in vector order. `fk` is
/// the frame's shared [`FkScratch`]: the stack runs allocation-free.
pub(crate) fn apply_ik_stack(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    stack: &IkStack,
    fk: &mut FkScratch,
) {
    if let Some(base) = &stack.base {
        pull_base(trs, parents, base, &stack.effectors, fk);
    }
    for effector in &stack.effectors {
        apply_effector(trs, parents, effector, fk);
    }
}

#[cfg(test)]
mod tests {
    use std::f32::consts::FRAC_PI_2;

    use glam::Quat;

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
            base: None,
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

    /// The §8.7 base pull: an unreachable target moves the base by the
    /// shortfall, so the chain then reaches exactly.
    #[test]
    fn the_base_pull_moves_the_base_into_an_unreachable_target() {
        let (mut trs, parents) = chain(3);
        let stack = IkStack {
            effectors: vec![Effector::TwoBone(TwoBoneIk {
                root: JointId::from_raw(0),
                mid: JointId::from_raw(1),
                end: JointId::from_raw(2),
                target: Vec3::new(10.0, 0.0, 0.0),
                weight: Clamped01::new(1.0),
            })],
            base: Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(1.0),
            }),
        };
        apply_ik_stack(&mut trs, &parents, &stack, &mut FkScratch::default());
        // Reach 2, target at 10: the base takes the shortfall of 8.
        assert!(
            (trs[0].translation - Vec3::X * 8.0).length() < 1e-4,
            "base at {:?}",
            trs[0].translation
        );
        let end = model_position(&trs, 2);
        assert!((end - Vec3::X * 10.0).length() < 1e-4, "end at {end:?}");
    }

    /// Look-at effectors never pull and a pull weight of `0` leaves the
    /// base alone — the §8.7 base pull is a two-bone shortfall only.
    #[test]
    fn a_zero_pull_weight_and_look_ats_leave_the_base_alone() {
        let mut fk = FkScratch::default();
        let (mut trs, parents) = chain(3);
        let look = IkStack {
            effectors: vec![Effector::LookAt(LookAt {
                joint: JointId::from_raw(0),
                target: Vec3::new(10.0, 10.0, 0.0),
                aim: Vec3::X,
                weight: Clamped01::new(1.0),
            })],
            base: Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(1.0),
            }),
        };
        apply_ik_stack(&mut trs, &parents, &look, &mut fk);
        assert_eq!(trs[0].translation, Vec3::ZERO);
        let (mut trs, parents) = chain(3);
        let muted = IkStack {
            effectors: vec![Effector::TwoBone(TwoBoneIk {
                root: JointId::from_raw(0),
                mid: JointId::from_raw(1),
                end: JointId::from_raw(2),
                target: Vec3::new(10.0, 0.0, 0.0),
                weight: Clamped01::new(1.0),
            })],
            base: Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(0.0),
            }),
        };
        apply_ik_stack(&mut trs, &parents, &muted, &mut fk);
        assert_eq!(trs[0].translation, Vec3::ZERO);
    }

    /// The pull of several two-bone effectors is the weighted mean of
    /// their shortfalls (§8.7).
    #[test]
    fn the_pull_is_the_weighted_mean_of_the_shortfalls() {
        let (mut trs, parents) = chain(3);
        let first = TwoBoneIk {
            root: JointId::from_raw(0),
            mid: JointId::from_raw(1),
            end: JointId::from_raw(2),
            target: Vec3::new(10.0, 0.0, 0.0),
            weight: Clamped01::new(1.0),
        };
        let mut second = first;
        second.target = Vec3::new(0.0, 10.0, 0.0);
        let stack = IkStack {
            effectors: vec![Effector::TwoBone(first), Effector::TwoBone(second)],
            base: Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(1.0),
            }),
        };
        apply_ik_stack(&mut trs, &parents, &stack, &mut FkScratch::default());
        // Shortfall 8 along +X and along +Y, mean (4, 4, 0).
        assert!(
            (trs[0].translation - Vec3::new(4.0, 4.0, 0.0)).length() < 1e-4,
            "base at {:?}",
            trs[0].translation
        );
    }

    /// One two-bone effector of the `chain(3)` shape at `target` — the
    /// shared fixture of the base-pull tests (rustqual DRY-001).
    fn target_at(target: Vec3) -> Effector {
        Effector::TwoBone(TwoBoneIk {
            root: JointId::from_raw(0),
            mid: JointId::from_raw(1),
            end: JointId::from_raw(2),
            target,
            weight: Clamped01::new(1.0),
        })
    }

    /// The §8.7 pull of `effectors` on `chain(3)` at pull `weight`,
    /// answered as the posed chain (rustqual DRY-001 keeps the stage in
    /// one place).
    fn pulled(effectors: Vec<Effector>, weight: f32) -> Vec<JointTrs> {
        let (mut trs, parents) = chain(3);
        apply_ik_stack(
            &mut trs,
            &parents,
            &IkStack {
                effectors,
                base: Some(BasePull {
                    joint: JointId::from_raw(0),
                    weight: Clamped01::new(weight),
                }),
            },
            &mut FkScratch::default(),
        );
        trs
    }

    /// A no-op effector never moves the base (§8.7): a non-finite
    /// target is gated out exactly like the solver gates it — pose and
    /// base stay untouched.
    #[test]
    fn a_non_finite_target_never_pulls_the_base() {
        let trs = pulled(vec![target_at(Vec3::NAN)], 1.0);
        assert_eq!(trs[0].translation, Vec3::ZERO);
        assert_eq!(trs[1].rotation, Quat::IDENTITY);
    }

    /// A reachable chain counts into the §8.7 mean with a zero
    /// shortfall — it holds the base (the far pull is halved by one
    /// anchored chain, not applied in full).
    #[test]
    fn a_reachable_chain_holds_the_base_against_a_far_target() {
        let trs = pulled(
            vec![
                target_at(Vec3::new(10.0, 0.0, 0.0)),
                target_at(Vec3::new(1.0, 0.0, 0.0)),
            ],
            1.0,
        );
        assert!(
            (trs[0].translation - Vec3::X * 4.0).length() < 1e-4,
            "base at {:?}",
            trs[0].translation
        );
    }

    /// The §8.8 control rig: world goals map into the root's model
    /// space through its world matrix (rotation included); the base
    /// passes through unchanged.
    #[test]
    fn the_control_rig_maps_world_goals_into_model_space() {
        let root_world =
            Mat4::from_rotation_translation(Quat::from_rotation_z(FRAC_PI_2), Vec3::X * 10.0);
        let rig = ControlRig {
            goals: vec![
                target_at(Vec3::new(10.0, 2.0, 0.0)),
                Effector::LookAt(LookAt {
                    joint: JointId::from_raw(0),
                    target: Vec3::new(10.0, 2.0, 0.0),
                    aim: Vec3::X,
                    weight: Clamped01::new(1.0),
                }),
            ],
            base: Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(1.0),
            }),
        };
        let stack = rig.into_stack(root_world);
        let chain = match stack.effectors[0] {
            Effector::TwoBone(chain) => chain,
            _ => panic!("the first goal stays a chain"),
        };
        let look = match stack.effectors[1] {
            Effector::LookAt(look) => look,
            _ => panic!("the second goal stays a look"),
        };
        // World (10, 2, 0) against a root at (10, 0, 0) turned 90°
        // about Z: the model-space goal is (2, 0, 0).
        assert!(
            (chain.target - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4,
            "chain at {:?}",
            chain.target
        );
        assert!(
            (look.target - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-4,
            "look at {:?}",
            look.target
        );
        assert_eq!(
            stack.base,
            Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(1.0),
            })
        );
    }

    /// A singular root matrix yields non-finite goals: the §8.7 gate
    /// drops them, the pose stays untouched, and `into_stack` does not
    /// panic (without `glam-assert`).
    #[test]
    fn a_singular_root_matrix_never_moves_the_pose() {
        let rig = ControlRig {
            goals: vec![target_at(Vec3::new(10.0, 2.0, 0.0))],
            base: Some(BasePull {
                joint: JointId::from_raw(0),
                weight: Clamped01::new(1.0),
            }),
        };
        let stack = rig.into_stack(Mat4::from_scale(Vec3::ZERO));
        let (mut trs, parents) = chain(3);
        apply_ik_stack(&mut trs, &parents, &stack, &mut FkScratch::default());
        assert_eq!(trs[0].translation, Vec3::ZERO);
        assert_eq!(trs[1].rotation, Quat::IDENTITY);
    }
}
