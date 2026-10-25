//! World proxies of the ragdoll contact coupling
//! (`docs/ragdoll-design.md` §8.5): one [`RagdollProxy`] entity per chain
//! bone mirrored into the world solver, driven from the chain before the
//! world step, and the drained contact impulses folded back into the
//! chain before its ABA step. The fixed-plan order is
//! `ragdoll_proxy_sync → physics_sync_in → physics_step →
//! physics_sync_out → ragdoll_sync` (see
//! [`crate::install_skeletal_animation`]).

use std::sync::Mutex;

use glam::{Quat, Vec3};
use ornis_animation::{RagdollBlend, RagdollBone, RagdollDesc, RagdollMode};
use ornis_core::{Entity, FixedTime, GlobalTransform, Resources, SmartStore, System, SystemAccess};
use ornis_physics::PhysicsEngine;
use ornis_physics::body::{BodyHandle, RigidBody};

use crate::physics_runtime::PhysicsRuntime;
use crate::ragdoll_runtime::{RagdollProxy, RagdollRuntime};

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
fn proxy_layer(link: usize) -> u32 {
    1u32 << ((link + 1).min(MAX_PROXY_CHAIN_BITS) as u32)
}

/// Collision mask of bone `link` (§8.5): the world's bit plus every chain
/// bone except the bone itself and its direct neighbours — parent/child
/// pairs cannot collide (the hinge keeps them together), cousins can.
fn proxy_mask(links: usize, link: usize) -> u32 {
    let mut mask = WORLD_LAYER_BIT;
    for other in 0..links {
        if other.abs_diff(link) > ADJACENT_BONES {
            mask |= proxy_layer(other);
        }
    }
    mask
}

/// Whether the world proxy of `bone` is bound this tick (§8.5): physical
/// mode binds always, `Blend` binds at a non-zero live weight, `Animated`
/// never binds — an unbound proxy manufactures no false impulses.
fn proxy_bound(mode: RagdollMode, blend: Option<&RagdollBlend>, bone: &RagdollBone) -> bool {
    match mode {
        RagdollMode::Animated => false,
        RagdollMode::Physical => true,
        RagdollMode::Blend { .. } => blend
            .and_then(|blend| blend.weights.get(usize::from(bone.joint)))
            .is_some_and(|weight| weight.get() > 0.0),
    }
}

/// Live world state of proxied bone `link` (§8.5): the chain body's model
/// pose/velocities taken to the world frame through `root_pose`.
/// Velocities are pre-compensated `+ gravity·dt` — the solver integrates
/// its own gravity into a dynamic proxy before solving, and the
/// compensation lands the step on the driven velocity (proxies carry the
/// link mass but not the world's gravity: the ABA chain owns it, §8.5).
/// `None` without the bone body or on a poisoned chain lock.
fn proxy_world_state(
    runtime: &RagdollRuntime,
    link: usize,
    root_pose: (Vec3, Quat),
    gravity: Vec3,
    dt: f32,
) -> Option<(Vec3, Quat, Vec3, Vec3)> {
    let handle = runtime.body_handle(link)?;
    let state = runtime.with_engine(|engine| {
        let body = engine.get_body(handle)?;
        Some((
            body.position,
            body.orientation,
            body.velocity,
            body.angular_velocity,
        ))
    });
    let (position, rotation, velocity, angular) = state.flatten()?;
    let (root_position, root_rotation) = root_pose;
    Some((
        root_rotation * position + root_position,
        root_rotation * rotation,
        root_rotation * velocity + gravity * dt,
        root_rotation * angular,
    ))
}

/// §8.5 world proxies of one ragdoll root: each bone's proxy rides a
/// [`RigidBody`] component while [`proxy_bound`] holds and loses it while
/// animation owns the pose. Bound proxies are driven to the chain's live
/// pose every tick — the lane gets the world state (the world solver's
/// `sync_in` binds new proxies and drops stale ones), the bound solver
/// body gets an explicit [`PhysicsRuntime::drive_pose`] teleport
/// (dynamic bodies are solver-authoritative for pose sync).
fn drive_proxies(
    store: &SmartStore,
    root: Entity,
    runtime: &RagdollRuntime,
    physics: &mut PhysicsRuntime,
    dt: f32,
) {
    let Some(descs) = store.read_cold_lane::<RagdollDesc>() else {
        return;
    };
    let Some(desc) = descs.get(root) else {
        return;
    };
    let mode = store
        .read_lane::<RagdollMode>()
        .and_then(|modes| modes.get(root).copied())
        .unwrap_or(RagdollMode::Animated);
    let blend_lane = store.read_lane::<RagdollBlend>();
    let blend = blend_lane.as_ref().and_then(|lane| lane.get(root));
    let Some(proxies) = store.read_lane::<RagdollProxy>() else {
        return;
    };
    let root_pose = store
        .read_lane::<GlobalTransform>()
        .and_then(|globals| globals.get(root).copied())
        .map(|global| (global.translation, global.rotation.get()))
        .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
    let gravity = physics.gravity();
    let mut bodies_lane = store.write_lane::<RigidBody>();
    for (entity, proxy) in proxies.entities.iter().zip(&proxies.data) {
        if proxy.root != root {
            continue;
        }
        let Some(bone) = desc.bones.get(proxy.link) else {
            continue;
        };
        let Some(bodies) = bodies_lane.as_mut() else {
            continue;
        };
        if !proxy_bound(mode, blend, bone) {
            bodies.remove(*entity);
            continue;
        }
        let Some((position, rotation, velocity, angular)) =
            proxy_world_state(runtime, proxy.link, root_pose, gravity, dt)
        else {
            continue;
        };
        if !bodies.contains(*entity) {
            let Some(mut body) = runtime.body_template(proxy.link) else {
                continue;
            };
            body.set_collision_filter(
                proxy_layer(proxy.link),
                proxy_mask(desc.bones.len(), proxy.link),
            );
            bodies.insert(*entity, body);
        }
        let Some(body) = bodies.get_mut(*entity) else {
            continue;
        };
        body.position = position;
        body.orientation = rotation;
        body.velocity = velocity;
        body.angular_velocity = angular;
        if let Some(handle) = physics.handle_of(*entity) {
            physics.drive_pose(handle, position, rotation);
        }
    }
}

/// Drains the world solver's contact impulses and folds each proxy
/// reaction into its chain (§8.5 explicit coupling): the record's
/// `impulse` is delivered to `a`, its negation to `b`, and each side that
/// names a bound proxy is applied to its chain bone in the root's model
/// space — before the ABA step of the same tick. Returns the number of
/// applied records. Worker of [`crate::ragdoll_sync::RagdollSyncSystem`].
pub(crate) fn fold_world_contacts(resources: &Resources, store: &SmartStore) -> usize {
    let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
        return 0;
    };
    let mut physics = runtime_resource
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let drained = physics.drain_contact_impulses();
    if drained.is_empty() {
        return 0;
    }
    let Some(proxies) = store.read_lane::<RagdollProxy>() else {
        return 0;
    };
    let mut bound: Vec<(BodyHandle, Entity, usize)> = Vec::new();
    for (entity, proxy) in proxies.entities.iter().zip(&proxies.data) {
        if let Some(handle) = physics.handle_of(*entity) {
            bound.push((handle, proxy.root, proxy.link));
        }
    }
    let runtimes = store.read_lane::<RagdollRuntime>();
    let globals = store.read_lane::<GlobalTransform>();
    let mut applied = 0;
    for record in &drained {
        for (handle, impulse) in [(record.a, record.impulse), (record.b, -record.impulse)] {
            let Some(&(_, root, link)) = bound.iter().find(|(bound, _, _)| *bound == handle) else {
                continue;
            };
            let Some(runtime) = runtimes.as_ref().and_then(|runtimes| runtimes.get(root)) else {
                continue;
            };
            let (root_position, root_rotation) = globals
                .as_ref()
                .and_then(|globals| globals.get(root))
                .map(|global| (global.translation, global.rotation.get()))
                .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
            let point = root_rotation.inverse() * (record.point - root_position);
            let impulse = root_rotation.inverse() * impulse;
            applied += usize::from(runtime.apply_contact_impulse(link, point, impulse));
        }
    }
    applied
}

/// §8.5 explicit coupling, fixed plan: drives every ragdoll root's world
/// proxies to the chain's live pose before the world step
/// (`ragdoll_proxy_sync → physics_sync_in` is an explicit edge).
pub struct RagdollProxySync;

impl RagdollProxySync {
    /// Creates the proxy-drive system.
    pub fn new() -> Self {
        Self
    }
}

impl Default for RagdollProxySync {
    /// Creates the proxy-drive system (see [`RagdollProxySync::new`]).
    fn default() -> Self {
        Self::new()
    }
}

impl System for RagdollProxySync {
    fn name(&self) -> &'static str {
        "ragdoll_proxy_sync"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<FixedTime>()
            .reads::<SmartStore>()
            .reads_lane::<GlobalTransform>()
            .reads_lane::<RagdollBlend>()
            .reads_lane::<RagdollMode>()
            .reads_lane::<RagdollProxy>()
            .reads_lane::<RagdollRuntime>()
            .writes::<Mutex<PhysicsRuntime>>()
            .writes_lane::<RigidBody>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(runtime_resource) = resources.get::<Mutex<PhysicsRuntime>>() else {
            return;
        };
        let mut physics = runtime_resource
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // §8.5 opt-in readback: the orchestrator pays for the impulse
        // accumulation exactly while ragdoll proxies may collide.
        let tracked = store
            .read_lane::<RagdollProxy>()
            .is_some_and(|proxies| !proxies.data.is_empty());
        physics.set_contact_impulse_tracking(tracked);
        let dt = resources
            .get::<FixedTime>()
            .map(|time| time.delta_seconds())
            .unwrap_or(0.0);
        let Some(runtimes) = store.read_lane::<RagdollRuntime>() else {
            return;
        };
        for &root in &runtimes.entities {
            if let Some(runtime) = runtimes.get(root) {
                drive_proxies(store, root, runtime, &mut physics, dt);
            }
        }
    }
}
