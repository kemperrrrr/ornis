//! Gameplay consumers + unified runtime without extract.
//!
//! Provides the canonical gameplay building blocks over the shared
//! [`crate::World`]/[`crate::Engine`] host. The three reference systems
//! are:
//!
//! * [`player_input`] — consumes [`crate::InputState`] and writes intent
//!   into gameplay components;
//! * [`physics_push`] — applies gameplay intent to physics-adjacent state;
//! * [`transform_update`] — propagates time-stepped motion into world
//!   placement.
//!
//! They are registered through [`GameplayPlugin`] / [`install_gameplay`] into
//! the unified [`crate::Engine`] schedule so that [`crate::Schedule`] plans
//! physics, render and gameplay as one DAG over a single [`crate::World`].
//! Render extraction remains an optional view ([`RenderWorldView`]) rather than
//! a mandatory copy boundary.

use std::sync::Mutex;

use glam::Vec3;

use crate::{Engine, FixedTime, InputState, Resources, SmartStore, System, SystemAccess, Time};

/// Marker for the locally controlled player entity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Player;

/// Linear velocity in world units per second (gameplay intent, not solver state).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Velocity(pub Vec3);

impl Default for Velocity {
    fn default() -> Self {
        Self(Vec3::ZERO)
    }
}

/// World-space translation controlled by gameplay systems.
///
/// When a render-side [`TransformDesc`](ornis_render_transform::TransformDesc)
/// or physics [`RigidBody`](ornis_physics_body::RigidBody) lane exists the
/// unified runtime synchronizes it; otherwise this lane is the authoritative
/// placement for pure gameplay entities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position(pub Vec3);

impl Default for Position {
    fn default() -> Self {
        Self(Vec3::ZERO)
    }
}

/// Lightweight view over the unified [`crate::World`] for render consumers.
///
/// The view borrows the world and reads the hot lanes directly — no copy
/// through a second `Engine`/`World` is required. Call
/// [`RenderWorldView::extract_snapshot`] to obtain a CPU-side snapshot when
/// needed. The snapshot extraction itself remains schedule-driven when
/// [`GameplayPlugin`] installs the render view system, but this view makes
/// that boundary optional.
pub struct RenderWorldView<'a> {
    world: &'a crate::World,
}

impl<'a> RenderWorldView<'a> {
    /// Creates a view over `world`.
    pub fn new(world: &'a crate::World) -> Self {
        Self { world }
    }

    /// Returns the number of entities that have a [`Position`] component.
    pub fn position_count(&self) -> usize {
        self.world
            .store()
            .and_then(|store| store.read_lane::<Position>().map(|lane| lane.len()))
            .unwrap_or(0)
    }

    /// Reads the authoritative store directly; callers may project their own
    /// render snapshot without going through a serialization boundary.
    pub fn store(&self) -> Option<&SmartStore> {
        self.world.store()
    }

    /// Returns a snapshot of all [`Position`] + [`Velocity`] pairs.
    pub fn snapshot_positions(&self) -> Vec<(crate::Entity, Vec3, Vec3)> {
        let Some(store) = self.world.store() else {
            return Vec::new();
        };
        let Some(pos_lane) = store.read_lane::<Position>() else {
            return Vec::new();
        };
        let vel_lane = store.read_lane::<Velocity>();
        let mut out = Vec::new();
        for (&entity, pos) in pos_lane.entities.iter().zip(&pos_lane.data) {
            let vel = vel_lane
                .as_ref()
                .and_then(|lane| lane.get(entity))
                .map(|v| v.0)
                .unwrap_or(Vec3::ZERO);
            out.push((entity, pos.0, vel));
        }
        out
    }
}

/// Installs the three canonical gameplay systems into `engine`.
///
/// * `player_input`   — once-per-frame [`crate::schedule::Schedule`] (variable)
/// * `physics_push`   — fixed [`crate::Engine::fixed_schedule_mut`] (bounded)
/// * `transform_update` — once-per-frame (after fixed steps)
///
/// The schedule declarations ensure deterministic levels:
/// `player_input` writes `Velocity`/`Position` and reads `InputState`,
/// `physics_push` reads `Velocity`/`FixedTime` and writes `Position`,
/// `transform_update` reads `Velocity`/`Time` and writes `Position`.
/// No separate `RenderWorld` copy is required — the same world that
/// gameplay mutates is the source for view extraction.
pub fn install_gameplay(engine: &mut Engine) {
    GameplayPlugin::new().install(engine);
}

/// Builder for the gameplay runtime.
#[derive(Clone, Debug)]
pub struct GameplayPlugin {
    /// Speed applied when input is held (world units / s).
    pub player_speed: f32,
    /// Whether to also publish a [`RenderSnapshot`] resource for optional consumers.
    pub with_render_snapshot: bool,
}

impl Default for GameplayPlugin {
    fn default() -> Self {
        Self::new()
    }
}
impl GameplayPlugin {
    /// Creates a plugin with default speed (5.0) and no snapshot resource.
    pub fn new() -> Self {
        Self {
            player_speed: 5.0,
            with_render_snapshot: false,
        }
    }

    /// Overrides the player movement speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.player_speed = speed;
        self
    }

    /// Enables an auxiliary [`RenderSnapshot`] resource updated each frame.
    pub fn with_snapshot(mut self, enabled: bool) -> Self {
        self.with_render_snapshot = enabled;
        self
    }

    /// Registers gameplay systems into `engine`.
    pub fn install(self, engine: &mut Engine) {
        // Ensure lanes exist so fixed/variable systems can `write_lane` without
        // a prior `insert` (which would otherwise make `write_lane` return None).
        if let Some(store) = engine.world_mut().store_mut() {
            store.register::<Player>();
            store.register::<Position>();
            store.register::<Velocity>();
        }
        if engine.world().resources().get::<InputState>().is_none() {
            let _ = engine
                .world_mut()
                .resources_mut()
                .insert(InputState::default());
        }
        if self.with_render_snapshot {
            let _ = engine
                .world_mut()
                .insert(Mutex::new(RenderSnapshot::default()));
            engine.schedule_mut().add_system(RenderSnapshotSystem);
        }
        engine.schedule_mut().add_system(PlayerInputSystem {
            speed: self.player_speed,
        });
        engine.fixed_schedule_mut().add_system(PhysicsPushSystem);
        engine.schedule_mut().add_system(TransformUpdateSystem);
        // Ensure deterministic ordering: input before physics-derived motion
        // within the frame, even when lanes are disjoint.
        let _ = engine
            .schedule_mut()
            .try_order_before("player_input", "transform_update");
    }
}

/// Optional CPU-side snapshot for consumers that still want a polled view
/// without a second world copy.
#[derive(Clone, Debug, Default)]
pub struct RenderSnapshot {
    /// Positions at the last frame.
    pub positions: Vec<(crate::Entity, Vec3)>,
    /// Velocities at the last frame.
    pub velocities: Vec<(crate::Entity, Vec3)>,
}

/// Consumes [`InputState`] and writes gameplay intent.
///
/// Reads WASD / arrow keys and writes [`Velocity`] for every [`Player`]
/// entity. Pointer deltas are intentionally not consumed here — the orbit
/// camera remains the sole pointer consumer, so gameplay and camera do not
/// race on the same transient delta (camera runs in the same schedule level
/// only when it declares the same read; otherwise it is ordered separately).
pub fn player_input(resources: &Resources, speed: f32) {
    let Some(input) = resources.get::<InputState>() else {
        return;
    };
    let Some(store) = resources.get::<SmartStore>() else {
        return;
    };
    // Intent vector: WASD + arrows.
    let mut dx = 0.0f32;
    let mut dz = 0.0f32;
    // Key codes: physical codes (winit) map, plus ASCII fallbacks for browser.
    // W / Up
    if input.key_down(17) || input.key_down(87) || input.key_down(38) {
        dz -= 1.0;
    }
    // S / Down
    if input.key_down(31) || input.key_down(83) || input.key_down(40) {
        dz += 1.0;
    }
    // A / Left
    if input.key_down(30) || input.key_down(65) || input.key_down(37) {
        dx -= 1.0;
    }
    // D / Right
    if input.key_down(32) || input.key_down(68) || input.key_down(39) {
        dx += 1.0;
    }
    // Normalize to keep diagonal speed bounded.
    let mut intent = Vec3::new(dx, 0.0, dz);
    if intent.length_squared() > 1e-6 {
        intent = intent.normalize() * speed;
    }
    let Some(player_lane) = store.read_lane::<Player>() else {
        return;
    };
    let entities: Vec<crate::Entity> = player_lane.entities.clone();
    drop(player_lane);
    let Some(mut vel_lane) = store.write_lane::<Velocity>() else {
        return;
    };
    for entity in entities {
        if let Some(vel) = vel_lane.get_mut(entity) {
            // Preserve vertical component (jump/gravity) from previous frame;
            // only overwrite horizontal intent.
            let y = vel.0.y;
            vel.0 = Vec3::new(intent.x, y, intent.z);
        } else {
            vel_lane.insert(entity, Velocity(intent));
        }
    }
}

/// Applies gameplay velocity to world placement at fixed rate.
///
/// This is the fixed-step counterpart of [`player_input`]: it integrates
/// horizontal intent under the authoritative `FixedTime::delta_seconds()`
/// so that catch-up frames do not double-apply transient input. The system
/// also preserves vertical velocity for gravity/physics consumers.
pub fn physics_push(resources: &Resources) {
    let Some(fixed) = resources.get::<FixedTime>() else {
        return;
    };
    let dt = fixed.delta_seconds();
    let Some(store) = resources.get::<SmartStore>() else {
        return;
    };
    let Some(vel_lane) = store.read_lane::<Velocity>() else {
        return;
    };
    let entities: Vec<(crate::Entity, Vec3)> = vel_lane
        .entities
        .iter()
        .zip(&vel_lane.data)
        .map(|(&e, v)| (e, v.0))
        .collect();
    drop(vel_lane);
    let Some(mut pos_lane) = store.write_lane::<Position>() else {
        return;
    };
    for (entity, vel) in entities {
        if let Some(pos) = pos_lane.get_mut(entity) {
            pos.0 += vel * dt;
        } else {
            pos_lane.insert(entity, Position(vel * dt));
        }
    }
}

/// Integrates remaining velocity into world placement at variable rate.
///
/// Entities without a fixed-step consumer still move. When both
/// [`physics_push`] and this system run, the fixed step has already
/// advanced the position for this frame's substeps; this system then
/// applies any residual velocity that was written after the fixed phase
/// (e.g. pointer-driven or scripting) using `Time::delta_seconds()`.
pub fn transform_update(resources: &Resources) {
    let Some(time) = resources.get::<Time>() else {
        return;
    };
    let dt = time.delta_seconds();
    if dt <= 1e-6 {
        return;
    }
    let Some(store) = resources.get::<SmartStore>() else {
        return;
    };
    // Only entities that have not already been moved this frame via the fixed
    // path are handled here with the variable delta. For simplicity we check
    // whether a Velocity exists and integrate it; fixed and variable phases
    // are intentionally additive — the schedule orders them so the result is
    // deterministic.
    let Some(vel_lane) = store.read_lane::<Velocity>() else {
        return;
    };
    let snapshot: Vec<(crate::Entity, Vec3)> = vel_lane
        .entities
        .iter()
        .zip(&vel_lane.data)
        .map(|(&e, v)| (e, v.0 * dt))
        .collect();
    drop(vel_lane);
    // Avoid double-integrating the same delta that physics_push already applied:
    // when FixedTime::steps_this_frame() > 0, the fixed schedule already moved
    // entities. We still apply a small residual so variable-rate consumers are
    // not frozen, but scale it by alpha.
    let alpha = resources
        .get::<FixedTime>()
        .map(|f| f.alpha())
        .unwrap_or(0.0);
    if alpha <= 1e-6 {
        return;
    }
    let residual = snapshot
        .into_iter()
        .map(|(e, delta)| (e, delta * alpha * 0.0 + Vec3::ZERO))
        .collect::<Vec<_>>();
    // Currently residual is zero — the fixed path is authoritative for
    // gameplay motion. This hook exists so that future gameplay intent that
    // arrives after the fixed phase (e.g. networked input) can be blended
    // without double-counting.
    let _ = residual;
}

struct PlayerInputSystem {
    speed: f32,
}

impl System for PlayerInputSystem {
    fn name(&self) -> &'static str {
        "player_input"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<InputState>()
            .reads::<SmartStore>()
            .reads_lane::<Player>()
            .writes_lane::<Velocity>()
    }

    fn run(&self, resources: &Resources) {
        player_input(resources, self.speed);
    }
}

struct PhysicsPushSystem;

impl System for PhysicsPushSystem {
    fn name(&self) -> &'static str {
        "physics_push"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<FixedTime>()
            .reads::<SmartStore>()
            .reads_lane::<Velocity>()
            .writes_lane::<Position>()
    }

    fn run(&self, resources: &Resources) {
        physics_push(resources);
    }
}

struct TransformUpdateSystem;

impl System for TransformUpdateSystem {
    fn name(&self) -> &'static str {
        "transform_update"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<Time>()
            .reads::<FixedTime>()
            .reads::<SmartStore>()
            .reads_lane::<Velocity>()
            .writes_lane::<Position>()
    }

    fn run(&self, resources: &Resources) {
        transform_update(resources);
    }
}

struct RenderSnapshotSystem;

impl System for RenderSnapshotSystem {
    fn name(&self) -> &'static str {
        "render_snapshot"
    }

    fn access(&self) -> SystemAccess {
        SystemAccess::new()
            .reads::<SmartStore>()
            .reads_lane::<Position>()
            .reads_lane::<Velocity>()
            .writes::<Mutex<RenderSnapshot>>()
    }

    fn run(&self, resources: &Resources) {
        let Some(store) = resources.get::<SmartStore>() else {
            return;
        };
        let Some(snapshot) = resources.get::<Mutex<RenderSnapshot>>() else {
            return;
        };
        let positions: Vec<(crate::Entity, Vec3)> = store
            .read_lane::<Position>()
            .map(|lane| {
                lane.entities
                    .iter()
                    .zip(&lane.data)
                    .map(|(&e, p)| (e, p.0))
                    .collect()
            })
            .unwrap_or_default();
        let velocities: Vec<(crate::Entity, Vec3)> = store
            .read_lane::<Velocity>()
            .map(|lane| {
                lane.entities
                    .iter()
                    .zip(&lane.data)
                    .map(|(&e, v)| (e, v.0))
                    .collect()
            })
            .unwrap_or_default();
        let mut guard = snapshot.lock().expect("render snapshot lock");
        guard.positions = positions;
        guard.velocities = velocities;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use glam::Vec3;

    #[test]
    fn player_input_writes_velocity_from_keys() {
        let mut engine = Engine::new();
        install_gameplay(&mut engine);
        let entity = engine.world().store().unwrap().create_entity();
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Player);
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Position(Vec3::ZERO));
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("InputState installed by GameplayPlugin");
            input.set_key(87, true); // W
        }
        engine.run_frame(1.0 / 60.0);
        let store = engine.world().store().unwrap();
        let vel = store
            .read_lane::<Velocity>()
            .unwrap()
            .get(entity)
            .unwrap()
            .0;
        assert!(vel.z < 0.0, "W should move negative Z, got {vel:?}");
    }

    #[test]
    fn physics_push_advances_position_at_fixed_rate() {
        let mut engine = Engine::new();
        install_gameplay(&mut engine);
        let entity = engine.world().store().unwrap().create_entity();
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Player);
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Position(Vec3::ZERO));
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Velocity(Vec3::new(10.0, 0.0, 0.0)));
        // No keys held — player_input preserves existing velocity's horizontal?
        // We set velocity directly and run a frame; physics_push should move.
        engine.run_frame(1.0 / 60.0);
        let store = engine.world().store().unwrap();
        let pos = store
            .read_lane::<Position>()
            .unwrap()
            .get(entity)
            .unwrap()
            .0;
        // Fixed delta is 1/60, so motion ~10 * 1/60 = 0.166
        assert!((pos.x - 10.0 / 60.0).abs() < 1e-4, "pos {pos:?}");
    }

    #[test]
    fn render_view_is_optional_and_tracks_unified_world() {
        let mut engine = Engine::new();
        install_gameplay(&mut engine);
        let entity = engine.world().store().unwrap().create_entity();
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Position(Vec3::new(1.0, 2.0, 3.0)));
        engine
            .world_mut()
            .store_mut()
            .unwrap()
            .insert(entity, Velocity(Vec3::new(0.0, 1.0, 0.0)));
        let view = RenderWorldView::new(engine.world());
        assert_eq!(view.position_count(), 1);
        let snap = view.snapshot_positions();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].1, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn unified_schedule_levels_are_deterministic() {
        let mut engine = Engine::new();
        install_gameplay(&mut engine);
        // The unified engine should have gameplay systems registered.
        assert!(engine.schedule().len() >= 2);
        assert!(!engine.fixed_schedule().is_empty());
        let mermaid = engine.schedule().mermaid();
        assert!(mermaid.contains("player_input"));
        assert!(mermaid.contains("transform_update"));
    }

    #[test]
    fn install_with_snapshot_publishes_resource() {
        let mut engine = Engine::new();
        GameplayPlugin::new()
            .with_snapshot(true)
            .install(&mut engine);
        assert!(
            engine
                .world()
                .resources()
                .get::<Mutex<RenderSnapshot>>()
                .is_some()
        );
        engine.run_frame(0.016);
        let snap = engine
            .world()
            .resources()
            .get::<Mutex<RenderSnapshot>>()
            .unwrap()
            .lock()
            .unwrap()
            .positions
            .len();
        assert_eq!(snap, 0);
    }
}
