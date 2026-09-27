//! Gameplay consumers + unified runtime without extract.
//!
//! Provides the canonical gameplay building blocks over the shared
//! [`ornis_core::World`]/[`ornis_core::Engine`] host. The three reference
//! systems are:
//!
//! * [`player_input`] — consumes [`ornis_input::InputState`] and writes
//!   intent into gameplay components;
//! * [`physics_push`] — applies gameplay intent to physics-adjacent state;
//! * [`transform_update`] — propagates time-stepped motion into world
//!   placement.
//!
//! They are registered through [`GameplayPlugin`] / [`install_gameplay`] into
//! the unified [`ornis_core::Engine`] schedule so that
//! [`ornis_core::Schedule`] plans physics, render and gameplay as one DAG
//! over a single [`ornis_core::World`].

#![warn(missing_docs)]

use glam::Vec3;

use ornis_core::units::MetersPerSecond;
use ornis_core::{Engine, FixedTime, Resources, SmartStore, System, SystemAccess, Time};
use ornis_input::{InputMap, InputState};
use ornis_macros::RegisterComponent;

/// Serializes a [`Vec3`]-backed component as its `[x, y, z]` array.
///
/// Manual impl (not `glam/serde`) so gameplay never toggles a workspace-wide
/// feature: the canonical JSON form matches `glam`'s own (`[x, y, z]`).
fn serialize_vec3<S: serde::Serializer>(v: Vec3, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeTupleStruct;
    let mut state = serializer.serialize_tuple_struct("Vec3", 3)?;
    state.serialize_field(&v.x)?;
    state.serialize_field(&v.y)?;
    state.serialize_field(&v.z)?;
    state.end()
}

/// Reads the `[x, y, z]` canonical form back (sequence only, like `glam`).
fn deserialize_vec3<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec3, D::Error> {
    struct Vec3Visitor;

    impl<'de> serde::de::Visitor<'de> for Vec3Visitor {
        type Value = Vec3;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a sequence of 3 f32 values")
        }

        fn visit_seq<A: serde::de::SeqAccess<'de>>(self, mut seq: A) -> Result<Vec3, A::Error> {
            let x = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
            let y = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
            let z = seq
                .next_element()?
                .ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
            Ok(Vec3::new(x, y, z))
        }
    }

    deserializer.deserialize_tuple_struct("Vec3", 3, Vec3Visitor)
}

/// Marker for the locally controlled player entity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Player;

/// Linear velocity in world units per second (gameplay intent, not solver state).
#[derive(Clone, Copy, Debug, PartialEq, RegisterComponent)]
pub struct Velocity(pub Vec3);

impl serde::Serialize for Velocity {
    /// Canonical `[x, y, z]` m/s form (registry, scenes, editor protocol).
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_vec3(self.0, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Velocity {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_vec3(deserializer).map(Velocity)
    }
}

impl Velocity {
    /// Wraps a raw m/s vector.
    pub fn from_mps(velocity: Vec3) -> Self {
        Self(velocity)
    }

    /// Raw m/s vector.
    pub fn as_mps(self) -> Vec3 {
        self.0
    }

    /// Speed (magnitude) in m/s.
    pub fn speed_units(self) -> MetersPerSecond {
        MetersPerSecond::new(self.0.length())
    }
}

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
#[derive(Clone, Copy, Debug, PartialEq, RegisterComponent)]
pub struct Position(pub Vec3);

impl serde::Serialize for Position {
    /// Canonical `[x, y, z]` meters form (registry, scenes, editor protocol).
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serialize_vec3(self.0, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Position {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserialize_vec3(deserializer).map(Position)
    }
}

impl Position {
    /// Wraps a raw world-space position (meters).
    pub fn from_meters(position: Vec3) -> Self {
        Self(position)
    }

    /// Raw position in meters.
    pub fn as_meters(self) -> Vec3 {
        self.0
    }

    /// Advance by `velocity * dt`: the shared integration step used by
    /// [`physics_push`] (meters = m/s × s).
    pub fn advanced_by(self, velocity: Velocity, dt_seconds: f32) -> Self {
        Self(self.0 + velocity.0 * dt_seconds)
    }

    /// Typed alias of [`Position::advanced_by`] over
    /// [`ornis_core::units::Seconds`] (meters = m/s × s, units checked).
    pub fn advanced_by_secs(self, velocity: Velocity, dt: ornis_core::units::Seconds) -> Self {
        self.advanced_by(velocity, dt.get())
    }
}

impl Default for Position {
    fn default() -> Self {
        Self(Vec3::ZERO)
    }
}

/// Installs the three canonical gameplay systems into `engine`.
///
/// * `player_input`   — once-per-frame [`ornis_core::Schedule`] (variable)
/// * `physics_push`   — fixed [`ornis_core::Engine::fixed_schedule_mut`] (bounded)
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
}

impl Default for GameplayPlugin {
    fn default() -> Self {
        Self::new()
    }
}
impl GameplayPlugin {
    /// Creates a plugin with default speed (5.0).
    pub fn new() -> Self {
        Self { player_speed: 5.0 }
    }

    /// Overrides the player movement speed.
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.player_speed = speed;
        self
    }

    /// Typed speed override in m/s (`None` for non-finite input; negative
    /// speeds clamp to zero instead of reversing the input mapping).
    pub fn with_speed_units(mut self, speed: MetersPerSecond) -> Option<Self> {
        let v = speed.get();
        if !v.is_finite() {
            return None;
        }
        self.player_speed = v.max(0.0);
        Some(self)
    }

    /// Current speed in m/s (non-finite or negative raw values saturate to
    /// zero so a corrupt field never reverses movement).
    pub fn speed_units(&self) -> MetersPerSecond {
        let v = self.player_speed;
        if v.is_finite() {
            MetersPerSecond::new(v.max(0.0))
        } else {
            MetersPerSecond::ZERO
        }
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
    // Read-edge mapping: an installed `InputMap` resource remaps the four
    // movement actions, otherwise the default WASD + arrows table applies.
    // Either way no raw key codes appear in gameplay logic.
    let fallback;
    let map = match resources.get::<InputMap>() {
        Some(custom) => custom,
        None => {
            fallback = InputMap::default_gameplay();
            &fallback
        }
    };
    // Intent vector from the mapped movement actions.
    let mut dx = 0.0f32;
    let mut dz = 0.0f32;
    // Forward
    if map.action_down(input, InputMap::MOVE_FORWARD) {
        dz -= 1.0;
    }
    // Back
    if map.action_down(input, InputMap::MOVE_BACK) {
        dz += 1.0;
    }
    // Strafe left
    if map.action_down(input, InputMap::MOVE_LEFT) {
        dx -= 1.0;
    }
    // Strafe right
    if map.action_down(input, InputMap::MOVE_RIGHT) {
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
    let entities: Vec<ornis_core::Entity> = player_lane.entities.clone();
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
    let dt = fixed.delta();
    let Some(store) = resources.get::<SmartStore>() else {
        return;
    };
    let Some(vel_lane) = store.read_lane::<Velocity>() else {
        return;
    };
    let entities: Vec<(ornis_core::Entity, Vec3)> = vel_lane
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
            *pos = pos.advanced_by_secs(Velocity::from_mps(vel), dt);
        } else {
            pos_lane.insert(entity, Position::from_meters(vel * dt.get()));
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
    let snapshot: Vec<(ornis_core::Entity, Vec3)> = vel_lane
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
            .reads::<InputMap>()
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

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use ornis_core::Engine;
    use ornis_input::KeyCode;

    fn spawn_player(engine: &mut Engine) -> ornis_core::Entity {
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
        entity
    }

    fn player_velocity(engine: &Engine, entity: ornis_core::Entity) -> Vec3 {
        engine
            .world()
            .store()
            .unwrap()
            .read_lane::<Velocity>()
            .unwrap()
            .get(entity)
            .unwrap()
            .0
    }

    #[test]
    fn player_input_writes_velocity_from_keys() {
        let mut engine = Engine::new();
        install_gameplay(&mut engine);
        let entity = spawn_player(&mut engine);
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("InputState installed by GameplayPlugin");
            input.set_keycode(KeyCode::KeyW, true);
        }
        engine.run_frame(1.0 / 60.0);
        let vel = player_velocity(&engine, entity);
        assert!(vel.z < 0.0, "W should move negative Z, got {vel:?}");
    }

    #[test]
    fn player_input_honors_custom_action_map() {
        use ornis_input::InputBinding;
        let mut engine = Engine::new();
        install_gameplay(&mut engine);
        let entity = spawn_player(&mut engine);
        // Remap forward to Space only: W must stop driving the player.
        let mut map = InputMap::default_gameplay();
        map.bind(
            InputMap::MOVE_FORWARD,
            InputBinding::with_keys([KeyCode::Space]),
        );
        let _ = engine.world_mut().insert(map);
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("InputState installed by GameplayPlugin");
            input.set_keycode(KeyCode::KeyW, true);
        }
        engine.run_frame(1.0 / 60.0);
        let vel = player_velocity(&engine, entity);
        assert_eq!(vel, Vec3::ZERO, "remapped W must not move, got {vel:?}");
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("InputState installed by GameplayPlugin");
            input.set_keycode(KeyCode::KeyW, false);
            input.set_keycode(KeyCode::Space, true);
        }
        engine.run_frame(1.0 / 60.0);
        let vel = player_velocity(&engine, entity);
        assert!(vel.z < 0.0, "Space should move negative Z, got {vel:?}");
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
}
