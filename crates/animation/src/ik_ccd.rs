//! Tail-chain CCD (`docs/animation-design.md` §8.6): the `root..=end`
//! chain of any length two-bone cannot fold — tails, guts, tendrils.
//! A separate module from the two-bone/look-at solvers (§8.2, §8.6) so
//! each solver family stays a cohesive file.

use crate::ik::CcdIk;
use crate::ik_solve::{FkScratch, arc_rotation, axis_ok, fk_model, parent_model_rot};
use crate::{JointId, JointTrs};

/// Iteration cap of the tail CCD (design §8.6): a fixed count keeps
/// the solve deterministic and bounded — tails settle far sooner.
const CCD_ITERATIONS: usize = 8;

/// Convergence radius of the tip on the target: inside it the tail
/// stops iterating early.
const CCD_REACH_EPS: f32 = 1e-4;

/// Same-rotation cutoff of one CCD write: below it the joint counts
/// as unmoved.
const CCD_WRITE_EPS: f32 = 1e-6;

/// Whether `root` sits on the parent walk from `end` (the chain check
/// of the solvers — ancestry runs with the solver, not at install).
fn is_ancestor(parents: &[Option<JointId>], root: usize, end: usize) -> bool {
    let mut cursor = parents[end];
    for _ in 0..parents.len() {
        let Some(id) = cursor else {
            return false;
        };
        let index = id.index();
        if index == root {
            return true;
        }
        if index >= parents.len() {
            return false;
        }
        cursor = parents[index];
    }
    false
}

/// One CCD step: rotates `joint` in its parent frame so the direction
/// to the tip turns onto the direction to the target, refreshing `fk`.
/// Returns whether the local rotation actually changed. Pure.
fn ccd_step(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    constraint: &CcdIk,
    joint: usize,
    fk: &mut FkScratch,
) -> bool {
    let (tip, origin) = (fk.model_pos[constraint.end.index()], fk.model_pos[joint]);
    let (to_tip, to_target) = (tip - origin, constraint.target - origin);
    if !axis_ok(to_tip) || !axis_ok(to_target) {
        return false;
    }
    let delta = arc_rotation(to_tip.normalize(), to_target.normalize());
    let parent = parent_model_rot(&fk.model_rot[..], parents, joint);
    let solved = parent.inverse() * delta * fk.model_rot[joint];
    if solved.abs_diff_eq(trs[joint].rotation, CCD_WRITE_EPS) {
        return false;
    }
    trs[joint].rotation = solved;
    fk_model(trs, parents, fk);
    true
}

/// One tip-to-base CCD sweep: every chain joint but the tip, FK
/// refreshed after each write. Returns whether anything moved. Pure.
fn ccd_sweep(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    constraint: &CcdIk,
    fk: &mut FkScratch,
) -> bool {
    let (root, end) = (constraint.root.index(), constraint.end.index());
    let mut moved = false;
    let mut cursor = parents[end];
    while let Some(id) = cursor {
        let joint = id.index();
        if joint >= trs.len() {
            break;
        }
        moved |= ccd_step(trs, parents, constraint, joint, fk);
        if joint == root {
            break;
        }
        cursor = parents[joint];
    }
    moved
}

/// Solves one tail-chain CCD against a local-TRS pose (design §8.6).
///
/// Rotates every joint of the chain `root..=end` but the tip,
/// tip-to-base, until the tip reaches `constraint.target` or
/// `CCD_ITERATIONS` sweeps run out, then slerps every chain rotation
/// toward its input by the constraint weight. Rotations only — bone
/// lengths survive. `parents` is the skeleton parent chain (parents
/// before children); `fk` is the caller's reusable [`FkScratch`] — one
/// scratch serves every solve of a frame. Returns whether any local
/// rotation changed; degenerate input (bad joints, `root` not an
/// ancestor of `end`, non-finite target, weight `0`) is a strict no-op.
pub fn solve_ccd_ik(
    trs: &mut [JointTrs],
    parents: &[Option<JointId>],
    constraint: &CcdIk,
    fk: &mut FkScratch,
) -> bool {
    let (root, end) = (constraint.root.index(), constraint.end.index());
    if constraint.weight.get() <= 0.0
        || !constraint.target.is_finite()
        || parents.len() != trs.len()
        || root >= trs.len()
        || end >= trs.len()
        || root == end
        || !is_ancestor(parents, root, end)
    {
        return false;
    }
    fk.orig_rot.clear();
    fk.orig_rot.extend(trs.iter().map(|local| local.rotation));
    let mut moved = false;
    for _ in 0..CCD_ITERATIONS {
        fk_model(trs, parents, fk);
        if (fk.model_pos[end] - constraint.target).length() <= CCD_REACH_EPS {
            break;
        }
        moved |= ccd_sweep(trs, parents, constraint, fk);
    }
    if !moved {
        return false;
    }
    // Weight blend (design §8.6): `0.5` lands each chain joint halfway
    // between its input and solved rotation, the tip then falls short.
    let weight = constraint.weight.get();
    if weight < 1.0 {
        let mut cursor = parents[end];
        while let Some(id) = cursor {
            let joint = id.index();
            if joint >= trs.len() {
                break;
            }
            trs[joint].rotation = fk.orig_rot[joint].slerp(trs[joint].rotation, weight);
            if joint == root {
                break;
            }
            cursor = parents[joint];
        }
    }
    true
}
