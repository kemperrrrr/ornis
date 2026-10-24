//! Per-frame ragdoll sync: fills the [`RagdollBlend`] lane of every
//! ragdoll root from the physical chain before the sample pass reads it
//! (`docs/ragdoll-design.md` §2.3, §4.1).

use ornis_animation::{JointMask, MaskId, RagdollBlend, RagdollBone, RagdollDesc, RagdollMode};
use ornis_core::units::Clamped01;
use ornis_core::{Entity, Resources, SmartStore, System, SystemAccess};

use crate::ragdoll_runtime::RagdollRuntime;

/// Per-frame fill of the [`RagdollBlend`] lane (§2.3, before sampling):
/// the mode's weights over the bone joints and the chain's model→local
/// sample. A broken sample (poisoned lock, missing body, non-finite
/// pose) zeroes the weights — animation wins the frame — and bumps
/// [`RagdollBlend::skipped_frames`]. Borrow-based: the lane guards live
/// for the fill — nothing is cloned, nothing is allocated (per-lane
/// locks never nest on the same type).
pub fn update_ragdoll(store: &SmartStore, root: Entity) -> bool {
    let Some(runtimes) = store.read_lane::<RagdollRuntime>() else {
        return false;
    };
    let Some(runtime) = runtimes.get(root) else {
        return false;
    };
    fill_ragdoll(store, root, runtime)
}

/// The fill itself with the borrowed runtime in hand — the system passes
/// the lane value straight from its iteration. Worker of
/// [`update_ragdoll`].
fn fill_ragdoll(store: &SmartStore, root: Entity, runtime: &RagdollRuntime) -> bool {
    let Some(descs) = store.read_cold_lane::<RagdollDesc>() else {
        return false;
    };
    let Some(desc) = descs.get(root) else {
        return false;
    };
    let mode = store
        .read_lane::<RagdollMode>()
        .and_then(|modes| modes.get(root).copied())
        .unwrap_or(RagdollMode::Animated);
    let mask_entity = match mode {
        RagdollMode::Blend {
            mask: MaskId(mask), ..
        } => Some(mask),
        _ => None,
    };
    let masks = store.read_cold_lane::<JointMask>();
    let mask = mask_entity.and_then(|entity| masks.as_ref().and_then(|masks| masks.get(entity)));
    let Some(mut blend) = store.write_lane::<RagdollBlend>() else {
        return false;
    };
    let Some(blend) = blend.get_mut(root) else {
        return false;
    };
    let sampled = runtime.sample_local(desc, &mut blend.physical);
    fill_weights(mode, mask, &desc.bones, &mut blend.weights);
    if !sampled {
        for weight in &mut blend.weights {
            *weight = Clamped01::ZERO;
        }
        blend.skipped_frames += 1;
    }
    sampled
}

/// Mode weights of the bone joints (§2.3): `weights` is zeroed, then each
/// bone's joint takes its mode weight — joints outside the chain stay at
/// zero. O(joints + bones) per root and frame. Leaf.
fn fill_weights(
    mode: RagdollMode,
    mask: Option<&JointMask>,
    bones: &[RagdollBone],
    weights: &mut [Clamped01],
) {
    for weight in weights.iter_mut() {
        *weight = Clamped01::ZERO;
    }
    for bone in bones {
        let Some(weight) = weights.get_mut(usize::from(bone.joint)) else {
            continue;
        };
        *weight = match mode {
            RagdollMode::Animated => Clamped01::ZERO,
            RagdollMode::Physical => Clamped01::ONE,
            RagdollMode::Blend {
                weight: strength, ..
            } => {
                let entry = mask
                    .and_then(|mask| mask.weights.get(usize::from(bone.joint)))
                    .copied()
                    .unwrap_or(Clamped01::ZERO);
                Clamped01::new(entry.get() * strength.get())
            }
        };
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
        // Hot lanes only — cold `RagdollDesc`/`JointMask` have no
        // declaration slot in `SystemAccess`.
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<RagdollMode>()
            .reads_lane::<RagdollRuntime>()
            .writes_lane::<RagdollBlend>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtimes) = store.read_lane::<RagdollRuntime>() else {
            return;
        };
        for &root in &runtimes.entities {
            if let Some(runtime) = runtimes.get(root) {
                fill_ragdoll(store, root, runtime);
            }
        }
    }
}
