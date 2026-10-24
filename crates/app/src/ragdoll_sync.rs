//! Per-frame ragdoll sync: fills the [`RagdollBlend`] lane of every
//! ragdoll root from the physical chain before the sample pass reads it
//! (`docs/ragdoll-design.md` §2.3, §4.1).

use glam::Mat4;
use ornis_animation::{
    JointMask, MaskId, RagdollBlend, RagdollBone, RagdollDesc, RagdollMode, Skeleton,
};
use ornis_core::units::Clamped01;
use ornis_core::{Entity, GlobalTransform, Resources, SmartStore, System, SystemAccess};

use crate::ragdoll_runtime::RagdollRuntime;

/// Per-frame fill of the [`RagdollBlend`] lane (§2.3, before sampling):
/// the mode's weights over the bone joints and the chain's world→local
/// sample. A broken sample (poisoned lock, missing body, non-finite
/// pose) zeroes the weights — animation wins the frame — and bumps
/// [`RagdollBlend::skipped_frames`].
pub fn update_ragdoll(store: &SmartStore, root: Entity) -> bool {
    let Some(desc) = store
        .read_cold_lane::<RagdollDesc>()
        .and_then(|descs| descs.get(root).cloned())
    else {
        return false;
    };
    let Some(runtime) = store
        .read_lane::<RagdollRuntime>()
        .and_then(|runtimes| runtimes.get(root).cloned())
    else {
        return false;
    };
    let Some(entity_world) = store.read_lane::<GlobalTransform>().and_then(|transforms| {
        transforms.get(root).map(|global| {
            Mat4::from_scale_rotation_translation(
                global.scale,
                global.rotation.get(),
                global.translation,
            )
        })
    }) else {
        return false;
    };
    let mode = store
        .read_lane::<RagdollMode>()
        .and_then(|modes| modes.get(root).copied())
        .unwrap_or(RagdollMode::Animated);
    let mask = match mode {
        RagdollMode::Blend {
            mask: MaskId(mask), ..
        } => store
            .read_cold_lane::<JointMask>()
            .and_then(|masks| masks.get(mask).cloned()),
        _ => None,
    };
    let Some(mut blend) = store.write_lane::<RagdollBlend>() else {
        return false;
    };
    let Some(blend) = blend.get_mut(root) else {
        return false;
    };
    let sampled = runtime.sample_local(&desc, entity_world, &mut blend.physical);
    for (joint, weight) in blend.weights.iter_mut().enumerate() {
        *weight = sampled_weight(mode, mask.as_ref(), desc.bones.as_slice(), joint);
    }
    if !sampled {
        for weight in &mut blend.weights {
            *weight = Clamped01::ZERO;
        }
        blend.skipped_frames += 1;
    }
    sampled
}

/// Weight of skeleton `joint` under `mode` (zero outside the chain).
/// Leaf.
fn sampled_weight(
    mode: RagdollMode,
    mask: Option<&JointMask>,
    bones: &[RagdollBone],
    joint: usize,
) -> Clamped01 {
    let mut on_bone = false;
    for bone in bones {
        if usize::from(bone.joint) == joint {
            on_bone = true;
        }
    }
    if !on_bone {
        return Clamped01::ZERO;
    }
    match mode {
        RagdollMode::Animated => Clamped01::ZERO,
        RagdollMode::Physical => Clamped01::ONE,
        RagdollMode::Blend { weight, .. } => {
            let entry = mask
                .and_then(|mask| mask.weights.get(joint))
                .copied()
                .unwrap_or(Clamped01::ZERO);
            Clamped01::new(entry.get() * weight.get())
        }
    }
}

/// Fills the [`RagdollBlend`] lanes of every ragdoll root before the
/// sample pass (§2.3). Registered before `skel_sample`.
pub struct RagdollSyncSystem;

impl RagdollSyncSystem {
    /// Creates the fill system.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RagdollSyncSystem {
    /// Creates the fill system (see [`RagdollSyncSystem::new`]).
    fn default() -> Self {
        Self::new()
    }
}

impl System for RagdollSyncSystem {
    fn name(&self) -> &'static str {
        "ragdoll_sync"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<GlobalTransform>()
            .reads_lane::<RagdollMode>()
            .reads_lane::<RagdollRuntime>()
            .reads_lane::<Skeleton>()
            .writes_lane::<RagdollBlend>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtimes) = store.read_lane::<RagdollRuntime>() else {
            return;
        };
        let roots: Vec<Entity> = runtimes.entities.to_vec();
        drop(runtimes);
        for root in roots {
            update_ragdoll(store, root);
        }
    }
}
