//! Per-tick ragdoll sync: one fixed chain tick plus the fill of the
//! [`RagdollBlend`] lane of every ragdoll root before the sample pass
//! reads it (`docs/ragdoll-design.md` §2.3, §4, §5).

use std::sync::Mutex;

use glam::Vec3;
use ornis_animation::{
    JointMask, JointTrs, MaskId, RagdollBlend, RagdollBone, RagdollDesc, RagdollFade, RagdollMode,
    Skeleton, fade_progress,
};
use ornis_core::units::Clamped01;
use ornis_core::{
    Entity, FixedTime, GlobalTransform, Resources, Seconds, SmartStore, System, SystemAccess,
};
use ornis_physics::body::RigidBody;
use ornis_physics::engine::PhysicsEngine;
use ornis_physics::joint::JointMotor;

use crate::physics_runtime::PhysicsRuntime;
use crate::ragdoll_g5::sample_local_into;
use crate::ragdoll_runtime::{RagdollProxy, RagdollRuntime};

/// Follow-servo stiffness of a driven hinge (1/s²; `AccelerationBased`,
/// so it scales with mass by construction). Damping and the torque budget
/// ride the bone's [`ornis_animation::RagdollMotor`] (§4.2).
const DRIVE_STIFFNESS: f32 = 100.0;
/// Quaternion half-angle factor: `2·atan2` reads the rad angle out of a
/// unit quaternion's twist.
const TWIST_FACTOR: f32 = 2.0;

/// Per-tick ragdoll work of one root (§5): weight-transition clock →
/// live weights (mode, then the transition lerp, then the live pins) →
/// drive motors and one fixed ABA step (both skipped in `Animated` —
/// ABA has no sleeping) → the model→local sample into `blend.physical`.
/// A broken sample zeroes the weights — animation wins the frame — and
/// bumps [`RagdollBlend::skipped_frames`]. Borrow-based: the lane guards
/// live for the fill, nothing is cloned and nothing is allocated per
/// tick (buffers and the transition snapshot ride the lanes, §6). Worker
/// of [`update_ragdoll`] and [`RagdollSyncSystem`].
fn tick_ragdoll(store: &SmartStore, root: Entity, runtime: &RagdollRuntime, dt: Seconds) -> bool {
    let Some(descs) = store.read_cold_lane::<RagdollDesc>() else {
        return false;
    };
    let Some(desc) = descs.get(root) else {
        return false;
    };
    let skeletons = store.read_lane::<Skeleton>();
    let skeleton = skeletons.as_ref().and_then(|lane| lane.get(root));
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
    let Some(mut blend_lane) = store.write_lane::<RagdollBlend>() else {
        return false;
    };
    let Some(blend) = blend_lane.get_mut(root) else {
        return false;
    };
    // Weight-transition clock (§4.2): the `from` snapshot is lerped by
    // borrow straight from its lane — no copy per tick (§6) — and a
    // finished transition dies in the same lane open.
    let mut fade_lane = store.write_lane::<RagdollFade>();
    let mut fade_done = false;
    let (progress, from) = match fade_lane.as_mut().and_then(|fades| fades.get_mut(root)) {
        Some(fade) => {
            fade.elapsed = Seconds::new(fade.elapsed.get() + dt.get());
            fade_done = fade.elapsed.get() >= fade.duration.get();
            (
                fade_progress(fade.duration, fade.elapsed),
                Some(fade.from.as_slice()),
            )
        }
        None => (1.0, None),
    };
    fill_weights(mode, mask, &desc.bones, blend, progress, from);
    if fade_done && let Some(fades) = fade_lane.as_mut() {
        fades.remove(root);
    }
    // G5 (§11) has no chain to step — the world step is the physics; the
    // ABA path drives its servos and takes one fixed chain step (§5).
    if !runtime.is_g5() && !matches!(mode, RagdollMode::Animated) {
        step_chain(runtime, desc, skeleton, blend, dt);
    }
    let sampled = if runtime.is_g5() {
        sample_local_g5(store, root, runtime, desc, &mut blend.physical)
    } else {
        runtime.sample_local(desc, &mut blend.physical)
    };
    if !sampled {
        for weight in &mut blend.weights {
            *weight = Clamped01::ZERO;
        }
        blend.skipped_frames += 1;
    }
    sampled
}

/// One chain tick of §5: every hinge below weight 1 gets a follow servo
/// aimed at the animated local rotation ([`RagdollBlend::anim_rotations`],
/// §4.2: the anim local against the bind local, projected on the hinge
/// axis — the twist angle in rad from the assembly pose; without a
/// recorded rotation or on non-finite data the servo stays as it was),
/// weight 1 clears the servo (physics owns the joint), and the chain
/// takes one fixed ABA step of `dt`. Fails closed on a poisoned chain
/// lock. The projection lives in the drive loop on purpose (§4.2): a
/// free helper would open duplicate-fragment windows (rustqual
/// `min_statements = 3`). Leaf.
fn step_chain(
    runtime: &RagdollRuntime,
    desc: &RagdollDesc,
    skeleton: Option<&Skeleton>,
    blend: &RagdollBlend,
    dt: Seconds,
) -> bool {
    runtime
        .with_engine(|engine| {
            if let Some(skeleton) = skeleton {
                for (index, bone) in desc.bones.iter().enumerate() {
                    let Some(hinge) = runtime.bone(index).and_then(|view| view.hinge) else {
                        continue;
                    };
                    let weight = blend
                        .weights
                        .get(usize::from(bone.joint))
                        .map(|weight| weight.get())
                        .unwrap_or(0.0);
                    if weight >= 1.0 {
                        let _ = engine.set_joint_motor(hinge, None);
                        continue;
                    }
                    let target = {
                        let joint = usize::from(bone.joint);
                        let Some(&anim) = blend.anim_rotations.get(joint) else {
                            continue;
                        };
                        let Some(rest) = skeleton.rest.get(joint) else {
                            continue;
                        };
                        let axis = bone.axis.normalize_or_zero();
                        if axis == Vec3::ZERO {
                            continue;
                        }
                        let delta = rest.rotation.inverse() * anim;
                        TWIST_FACTOR
                            * Vec3::new(delta.x, delta.y, delta.z)
                                .dot(axis)
                                .atan2(delta.w)
                    };
                    if !target.is_finite() {
                        continue;
                    }
                    if let Some(motor) = JointMotor::position(
                        target,
                        DRIVE_STIFFNESS,
                        bone.motor.damping,
                        bone.motor.max_torque,
                    ) {
                        let _ = engine.set_joint_motor(hinge, Some(motor));
                    }
                }
            }
            engine.step(dt.get());
            true
        })
        .unwrap_or(false)
}

/// Mode weights of the bone joints (§2.3): the target weights of `mode`
/// over the bone joints (joints outside the chain stay at zero), lerped
/// from the transition snapshot `from` at `progress` (§4.2), then the
/// live pins of `set_ragdoll_weight` overwrite their joints. O(joints +
/// bones). Leaf.

fn fill_weights(
    mode: RagdollMode,
    mask: Option<&JointMask>,
    bones: &[RagdollBone],
    blend: &mut RagdollBlend,
    progress: f32,
    from: Option<&[Clamped01]>,
) {
    for weight in blend.weights.iter_mut() {
        *weight = Clamped01::ZERO;
    }
    for bone in bones {
        let Some(weight) = blend.weights.get_mut(usize::from(bone.joint)) else {
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
    if let Some(from) = from {
        for (weight, source) in blend.weights.iter_mut().zip(from) {
            let target = weight.get();
            *weight = Clamped01::new(source.get() + (target - source.get()) * progress);
        }
    }
    for (weight, pin) in blend.weights.iter_mut().zip(&blend.pinned) {
        if let Some(pin) = pin {
            *weight = *pin;
        }
    }
}

/// G5 readback of §11: the §4.1 model→local sample taken from the world
/// bodies — the proxies' live [`RigidBody`] poses after
/// `physics_sync_out` — against the same seam as the ABA chain. A missing
/// proxy or body fails the sample like a missing chain body. Leaf.
fn sample_local_g5(
    store: &SmartStore,
    root: Entity,
    runtime: &RagdollRuntime,
    desc: &RagdollDesc,
    out: &mut [JointTrs],
) -> bool {
    let (Some(proxies), Some(bodies)) = (
        store.read_lane::<RagdollProxy>(),
        store.read_lane::<RigidBody>(),
    ) else {
        return false;
    };
    sample_local_into(
        desc,
        out,
        runtime.bind_frames(),
        runtime.parents(),
        |link| {
            let entity = proxies
                .entities
                .iter()
                .zip(&proxies.data)
                .filter(|(_, proxy)| proxy.root == root && proxy.link == link)
                .map(|(&entity, _)| entity)
                .next()?;
            bodies
                .get(entity)
                .map(|body| (body.position, body.orientation))
        },
    )
}

/// One ragdoll tick of `root` (§5): the fixed chain step plus the
/// [`RagdollBlend`] fill before sampling. `dt` is the fixed step the host
/// ticks at ([`RagdollSyncSystem`] passes `FixedTime`; tests advance one
/// tick per call). Borrow-based: the lane value is passed straight from
/// its iteration, nothing is cloned.
pub fn update_ragdoll(store: &SmartStore, root: Entity, dt: Seconds) -> bool {
    let Some(runtimes) = store.read_lane::<RagdollRuntime>() else {
        return false;
    };
    let Some(runtime) = runtimes.get(root) else {
        return false;
    };
    tick_ragdoll(store, root, runtime, dt)
}

/// Ticks the chain and fills the [`RagdollBlend`] lanes of every ragdoll
/// root (§5). Registered on the **fixed** plan: one fixed ABA step per
/// fixed tick, and the fixed loop runs before the frame schedule
/// engine-wide, so the sample pass always reads the last fill (§2.3).
pub struct RagdollSyncSystem;

impl RagdollSyncSystem {
    /// Creates the tick-and-fill system.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RagdollSyncSystem {
    /// Creates the tick-and-fill system (see [`RagdollSyncSystem::new`]).
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
            .reads::<FixedTime>()
            .reads::<SmartStore>()
            .reads_lane::<GlobalTransform>()
            .reads_lane::<RagdollMode>()
            .reads_lane::<RagdollProxy>()
            .reads_lane::<RagdollRuntime>()
            .reads_lane::<RagdollProxy>()
            .reads_lane::<RigidBody>()
            .reads_lane::<Skeleton>()
            .writes::<Mutex<PhysicsRuntime>>()
            .writes_lane::<RagdollBlend>()
            .writes_lane::<RagdollFade>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(time) = resources.get::<FixedTime>() else {
            return;
        };
        let dt = Seconds::new(time.delta_seconds());
        // §8.5 explicit coupling: fold the drained world contacts into the
        // chains first — `apply_impulse` feeds this tick's ABA step; the
        // one-tick latency lives in the proxy pose (proxies follow the
        // chain pose of the previous tick), not in the delivery.
        crate::ragdoll_proxy::fold_world_contacts(resources, store);
        let Some(runtimes) = store.read_lane::<RagdollRuntime>() else {
            return;
        };
        for &root in &runtimes.entities {
            if let Some(runtime) = runtimes.get(root) {
                tick_ragdoll(store, root, runtime, dt);
            }
        }
    }
}
