//! Ragdoll model: the physics-free description of a skeleton↔chain
//! binding ([`RagdollDesc`] bones, [`RagdollMode`] pose ownership) — see
//! `docs/ragdoll-design.md` §2.1. The seam that talks to
//! `ornis-physics` lives in the app crate (`ragdoll_wiring`).

use glam::Vec3;

use crate::blend::{JointTrs, MaskId};
use crate::{JointId, Skeleton};
use ornis_core::units::Clamped01;

/// Shape of one ragdoll bone body (explicit sizes — `docs/ragdoll-design.md`
/// §10.2; the seam computes mass from `RagdollBone::density`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RagdollShape {
    /// Capsule standing along the hinge-perpendicular bone frame: `radius`
    /// positive, `half_height` of the cylindrical part non-negative (m) —
    /// zero gives a sphere.
    Capsule {
        /// Capsule radius (m).
        radius: f32,
        /// Half-height of the cylindrical part (m), zero = sphere.
        half_height: f32,
    },
    /// Box in the joint frame: `half_extents` in meters, each positive.
    Box {
        /// Half-extents along the joint-frame axes (m).
        half_extents: Vec3,
    },
}

/// Angular travel window of a revolute ragdoll joint (rad, relative to the
/// bind pose). `min == max` locks the hinge on the bind twist.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagdollLimit {
    /// Lower bound (rad).
    pub min: f32,
    /// Upper bound (rad).
    pub max: f32,
}

/// Drive parameters of a revolute ragdoll joint. The seam applies
/// `damping`/`friction` to the physical hinge and keeps `max_torque` as the
/// budget of the animation-following servo (`docs/ragdoll-design.md` §4.2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagdollMotor {
    /// Torque budget of the follow servo (N·m, non-negative).
    pub max_torque: f32,
    /// Joint damping (non-negative).
    pub damping: f32,
    /// Joint friction (non-negative).
    pub friction: f32,
}

/// One ragdoll bone: a skeleton joint bound to a physical link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RagdollBone {
    /// Skeleton joint this bone poses.
    pub joint: JointId,
    /// Physical body shape.
    pub shape: RagdollShape,
    /// Hinge axis in the joint's bind frame (finite, non-zero; the seam
    /// normalizes it).
    pub axis: Vec3,
    /// Travel window of the hinge; `None` = free hinge.
    pub limit: Option<RagdollLimit>,
    /// Drive parameters of the hinge.
    pub motor: RagdollMotor,
    /// Body density (kg/m³, finite, positive); mass = volume × density.
    pub density: f32,
    /// Joint of the parent bone in the chain: another bone of this
    /// description and a skeleton ancestor of `joint`. `None` marks the
    /// chain root — exactly one bone in [`RagdollDesc::bones`] may do so.
    pub parent: Option<JointId>,
}

/// How the chain root is attached to the world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagdollBase {
    /// The root link is welded where it spawns.
    Fixed,
    /// The root link is free (hips lead a free fall).
    Floating,
}

/// Whole binding: bones (parent-before-child not required — the seam
/// attaches root links) plus the root base.
#[derive(Debug, Clone, PartialEq)]
pub struct RagdollDesc {
    /// Bones of the chain; every `parent` names another bone here.
    pub bones: Vec<RagdollBone>,
    /// Attachment of the chain root.
    pub base: RagdollBase,
}

/// Who writes the posed joints (`docs/ragdoll-design.md` §2.1; the same
/// per-entry switch shape as [`crate::skinning::SkinningMode`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RagdollMode {
    /// The animation pipeline poses every joint; physics sleeps.
    Animated,
    /// Physics poses every joint; animated root motion stops.
    Physical,
    /// Per-joint mix: joints with mask weight `1` take physics fully, `0`
    /// keeps the animated pose; `weight` scales the mask before the blend.
    Blend {
        /// Playlist mask scaled by `weight` (length = skeleton joint count).
        mask: MaskId,
        /// Global strength of the physical side.
        weight: Clamped01,
    },
}

/// Per-frame ragdoll input of one skeleton root (hot lane, filled by the
/// app's ragdoll seam before sampling — `docs/ragdoll-design.md` §2.3):
/// physical local poses and per-joint blend weights (`1` = physics).
/// Joints outside the ragdoll chain keep zero weights.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RagdollBlend {
    /// Physical local pose per skeleton joint (model space, §4.1; the
    /// scale field is unused — the blend keeps the animated scale). On a
    /// bad frame the fill may leave a partially written buffer behind,
    /// but it zeroes the weights first, so nothing of it blends.
    pub physical: Vec<JointTrs>,
    /// Blend weight per skeleton joint (`1.0` takes the physical pose).
    pub weights: Vec<Clamped01>,
    /// Frames where the physical sample was unusable (animation won).
    pub skipped_frames: u32,
}

/// Blends the physical pose over `animated` joint by joint (§4.1):
/// translation and rotation mix (`lerp`/`slerp`) toward the physical
/// joint at the joint's weight; the **scale stays animated** — physics
/// carries none, and the inverse seam lands in the skeleton's scaled
/// local frames, so even weight 1 keeps the authored scale. Weight `0`
/// leaves the joint untouched. Truncates at the shortest of the three
/// slices — the fill sizes them all to the joint count. Pure: no
/// allocations, no state.
pub fn ragdoll_blend(animated: &mut [JointTrs], physical: &[JointTrs], weights: &[Clamped01]) {
    for ((pose, source), weight) in animated.iter_mut().zip(physical).zip(weights) {
        let strength = weight.get();
        if strength <= 0.0 {
            continue;
        }
        let scale = pose.scale;
        *pose = pose.lerp(*source, strength);
        pose.scale = scale;
    }
}

/// Why [`RagdollDesc::validate`] refused a description. The facade writes
/// nothing on any of these (`docs/ragdoll-design.md` §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RagdollError {
    /// A chain needs at least one bone.
    #[error("ragdoll needs at least one bone")]
    EmptyChain,
    /// A bone joint is outside the skeleton or named twice.
    #[error("bone joint out of the skeleton range or duplicated")]
    Joint,
    /// A hinge axis is zero or not finite.
    #[error("hinge axis must be finite and non-zero")]
    Axis,
    /// A limit is not finite or `min > max`.
    #[error("revolute limit must be finite and min <= max")]
    Limit,
    /// A motor parameter is negative or not finite.
    #[error("motor parameters must be finite and non-negative")]
    Motor,
    /// A shape size is negative or not finite.
    #[error("shape sizes must be finite and non-negative")]
    Shape,
    /// Physics refused to assemble a hinge of an otherwise valid
    /// description (unreachable in practice; the reason stays typed).
    #[error("physics could not assemble the chain")]
    Assemble,
    /// The root has no [`Skeleton`] lane to bind against.
    #[error("the root has no Skeleton lane")]
    NoSkeleton,
    /// A density is not positive or not finite.
    #[error("density must be finite and positive")]
    Density,
    /// A parent is not the skeleton parent's bone of the description
    /// (the chain must mirror the skeleton — §4.1 needs it so every local
    /// blend lands in its own skeleton slot).
    #[error("bone parent must be the skeleton parent's bone of the description")]
    Parent,
    /// The bones do not form one chain rooted at the skeleton root.
    #[error("the bones must form one chain rooted at the skeleton root")]
    Roots,
    /// A blend mask is missing or does not match the joint count.
    #[error("blend mask missing or sized differently from the skeleton")]
    Mask,
    /// No ragdoll chain is installed on the root.
    #[error("no ragdoll chain installed on the root")]
    Missing,
}

impl RagdollDesc {
    /// Checks the description against `skeleton` — every rule of
    /// `docs/ragdoll-design.md` §6 that the model can see (mode masks are
    /// store data and checked by the facade). Pure and allocation-free.
    pub fn validate(&self, skeleton: &Skeleton) -> Result<(), RagdollError> {
        if self.bones.is_empty() {
            return Err(RagdollError::EmptyChain);
        }
        for (index, _) in self.bones.iter().enumerate() {
            bone_ok(index, &self.bones, &skeleton.parents)?;
        }
        let mut roots = 0;
        for bone in &self.bones {
            if bone.parent.is_none() {
                roots += 1;
            }
        }
        if roots != 1 {
            return Err(RagdollError::Roots);
        }
        Ok(())
    }
}

/// Per-bone rules of [`RagdollDesc::validate`] (`docs/ragdoll-design.md`
/// §6): the joint identity, the physical numbers, the parent link. Leaf
/// over the three checks below.
fn bone_ok(
    index: usize,
    bones: &[RagdollBone],
    parents: &[Option<JointId>],
) -> Result<(), RagdollError> {
    joint_ok(index, bones, parents)?;
    numbers_ok(&bones[index])?;
    parent_ok(&bones[index], bones, parents)
}

/// Joint identity of `bones[index]`: in skeleton range, named once. Leaf.
fn joint_ok(
    index: usize,
    bones: &[RagdollBone],
    parents: &[Option<JointId>],
) -> Result<(), RagdollError> {
    let bone = &bones[index];
    if usize::from(bone.joint) >= parents.len() {
        return Err(RagdollError::Joint);
    }
    for (other, candidate) in bones.iter().enumerate() {
        if other != index && candidate.joint == bone.joint {
            return Err(RagdollError::Joint);
        }
    }
    Ok(())
}

/// Physical numbers of `bone`: axis, density, shape and the window/drive
/// below. Leaf.
fn numbers_ok(bone: &RagdollBone) -> Result<(), RagdollError> {
    if !bone.axis.is_finite() || bone.axis.length_squared() == 0.0 {
        return Err(RagdollError::Axis);
    }
    if !bone.density.is_finite() || bone.density <= 0.0 {
        return Err(RagdollError::Density);
    }
    limit_ok(bone.limit)?;
    motor_ok(&bone.motor)?;
    if !shape_ok(&bone.shape) {
        return Err(RagdollError::Shape);
    }
    Ok(())
}

/// Travel window of a hinge: finite bounds in order. Leaf.
fn limit_ok(limit: Option<RagdollLimit>) -> Result<(), RagdollError> {
    if let Some(limit) = limit
        && (!limit.min.is_finite() || !limit.max.is_finite() || limit.min > limit.max)
    {
        return Err(RagdollError::Limit);
    }
    Ok(())
}

/// Drive parameters of a hinge: finite and non-negative. Leaf.
fn motor_ok(motor: &RagdollMotor) -> Result<(), RagdollError> {
    if !motor.max_torque.is_finite()
        || motor.max_torque < 0.0
        || !motor.damping.is_finite()
        || motor.damping < 0.0
        || !motor.friction.is_finite()
        || motor.friction < 0.0
    {
        return Err(RagdollError::Motor);
    }
    Ok(())
}

/// Parent link of `bone`: `parent` must be the bone of the joint's direct
/// skeleton parent, and a root bone must sit on the skeleton root — the
/// chain mirrors the skeleton so every local blend lands in its own
/// skeleton slot (§4.1). Leaf.
fn parent_ok(
    bone: &RagdollBone,
    bones: &[RagdollBone],
    parents: &[Option<JointId>],
) -> Result<(), RagdollError> {
    let skeleton_parent = parents[usize::from(bone.joint)];
    match bone.parent {
        None => {
            if skeleton_parent.is_some() {
                return Err(RagdollError::Parent);
            }
        }
        Some(parent) => {
            if skeleton_parent != Some(parent) {
                return Err(RagdollError::Parent);
            }
            if !bones.iter().any(|candidate| candidate.joint == parent) {
                return Err(RagdollError::Parent);
            }
        }
    }
    Ok(())
}

/// Whether every shape size is finite and positive. Leaf.
fn shape_ok(shape: &RagdollShape) -> bool {
    match shape {
        RagdollShape::Capsule {
            radius,
            half_height,
        } => radius.is_finite() && *radius > 0.0 && half_height.is_finite() && *half_height >= 0.0,
        RagdollShape::Box { half_extents } => {
            half_extents.is_finite()
                && half_extents.x > 0.0
                && half_extents.y > 0.0
                && half_extents.z > 0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    /// Skeleton `0 <- 1 <- 2` (a three-joint chain).
    fn chain() -> Skeleton {
        Skeleton::new(
            vec![None, Some(JointId::from_raw(0)), Some(JointId::from_raw(1))],
            vec![Default::default(); 3],
            vec!["hips".to_string(), "spine".to_string(), "head".to_string()],
        )
    }

    /// A sound bone on `joint` under `parent`.
    fn bone(joint: usize, parent: Option<usize>) -> RagdollBone {
        RagdollBone {
            joint: JointId::from_raw(joint as u32),
            shape: RagdollShape::Capsule {
                radius: 0.1,
                half_height: 0.2,
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

    /// Description of `bones` under a floating base.
    fn desc(bones: Vec<RagdollBone>) -> RagdollDesc {
        RagdollDesc {
            bones,
            base: RagdollBase::Floating,
        }
    }

    #[test]
    fn a_sound_chain_validates() {
        {
            let sound = desc(vec![bone(0, None), bone(1, Some(0)), bone(2, Some(1))]);
            assert_eq!(sound.validate(&chain()), Ok(()));
        }
    }

    /// Two skeleton roots side by side.
    fn pair() -> Skeleton {
        Skeleton::new(
            vec![None, None],
            vec![Default::default(); 2],
            vec!["hips".to_string(), "tail".to_string()],
        )
    }

    #[test]
    fn empty_and_multi_root_chains_refuse() {
        {
            assert_eq!(
                desc(Vec::new()).validate(&chain()),
                Err(RagdollError::EmptyChain)
            );
            let two_roots = desc(vec![bone(0, None), bone(1, None)]);
            assert_eq!(two_roots.validate(&pair()), Err(RagdollError::Roots));
            let no_root = desc(vec![bone(0, Some(1)), bone(1, Some(0))]);
            assert_eq!(no_root.validate(&chain()), Err(RagdollError::Parent));
        }
    }

    #[test]
    fn joint_names_must_be_unique_and_in_range() {
        {
            let out_of_range = desc(vec![bone(3, None)]);
            assert_eq!(out_of_range.validate(&chain()), Err(RagdollError::Joint));
            let duplicated = desc(vec![bone(0, None), bone(0, None)]);
            assert_eq!(duplicated.validate(&chain()), Err(RagdollError::Joint));
        }
    }

    #[test]
    fn parents_must_mirror_the_skeleton() {
        {
            // Bone 2 under bone 0 would skip bone 1 — its local blend slot
            // would not line up with the physical frame (§4.1).
            let skipped = desc(vec![bone(0, None), bone(1, Some(0)), bone(2, Some(0))]);
            assert_eq!(skipped.validate(&chain()), Err(RagdollError::Parent));
            let not_a_bone = desc(vec![bone(0, None), bone(2, Some(1))]);
            assert_eq!(not_a_bone.validate(&chain()), Err(RagdollError::Parent));
            let detached_root = desc(vec![bone(1, None), bone(2, Some(1))]);
            assert_eq!(detached_root.validate(&chain()), Err(RagdollError::Parent));
        }
    }

    #[test]
    fn numbers_must_be_physical() {
        {
            let mut bad_axis = desc(vec![bone(0, None)]);
            bad_axis.bones[0].axis = Vec3::ZERO;
            assert_eq!(bad_axis.validate(&chain()), Err(RagdollError::Axis));
            let mut bad_limit = desc(vec![bone(0, None)]);
            bad_limit.bones[0].limit = Some(RagdollLimit {
                min: 0.0,
                max: f32::NEG_INFINITY,
            });
            assert_eq!(bad_limit.validate(&chain()), Err(RagdollError::Limit));
            let mut bad_motor = desc(vec![bone(0, None)]);
            bad_motor.bones[0].motor.damping = -1.0;
            assert_eq!(bad_motor.validate(&chain()), Err(RagdollError::Motor));
            let mut bad_density = desc(vec![bone(0, None)]);
            bad_density.bones[0].density = 0.0;
            assert_eq!(bad_density.validate(&chain()), Err(RagdollError::Density));
            let mut bad_shape = desc(vec![bone(0, None)]);
            bad_shape.bones[0].shape = RagdollShape::Box {
                half_extents: Vec3::ZERO,
            };
            assert_eq!(bad_shape.validate(&chain()), Err(RagdollError::Shape));
            let mut sphere = desc(vec![bone(0, None)]);
            sphere.bones[0].shape = RagdollShape::Capsule {
                radius: 0.1,
                half_height: 0.0,
            };
            assert_eq!(sphere.validate(&chain()), Ok(()), "zero height = sphere");
        }
    }

    /// A one-joint pose at `translation`.
    fn pose(translation: f32) -> JointTrs {
        JointTrs {
            translation: Vec3::splat(translation),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    #[test]
    fn ragdoll_blend_mixes_by_exact_weights() {
        {
            let mut animated = [pose(0.0)];
            ragdoll_blend(&mut animated, &[pose(1.0)], &[Clamped01::new(0.5)]);
            assert_eq!(animated[0].translation, Vec3::splat(0.5));
            let mut animated = [pose(0.0)];
            ragdoll_blend(&mut animated, &[pose(1.0)], &[Clamped01::ONE]);
            assert_eq!(animated[0].translation, Vec3::splat(1.0));
            let mut animated = [pose(0.0)];
            ragdoll_blend(&mut animated, &[pose(1.0)], &[Clamped01::ZERO]);
            assert_eq!(animated[0].translation, Vec3::splat(0.0));
        }
    }
}
