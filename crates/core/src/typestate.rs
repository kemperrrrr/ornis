//! Phantom phases and scalar newtypes for the frame boundary.
//!
//! [`Building`] and [`Running`] are zero-sized markers carried as phantom
//! parameters by [`World`](crate::World), [`Engine`](crate::Engine) and
//! `GameWorld`: schedule/system registration belongs to the building phase,
//! the frame loop owns mutation once built. The parameters default to the
//! running role, so existing `World`/`Engine` paths keep compiling; the
//! phased constructors (`new_building` + `build`) are the opt-in strict
//! route. Scalar newtypes ([`Seconds`], [`Frame`], [`Tick`],
//! [`FixedSteps`], [`SceneVersion`]) replace bare `f32`/`u64`/`u32` at the
//! boundary without changing serialization: each one encodes as the plain
//! underlying number, keeping serde JSON and WASM snapshots stable.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Entity;

/// Build-time phase: systems and resources may be registered.
///
/// A [`World`](crate::World)/[`Engine`](crate::Engine) in this phase has not
/// entered the frame loop yet; [`Engine::build`](crate::Engine::build)
/// seals it into [`Running`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Building {
    _private: (),
}

/// Run-time phase: the frame loop owns mutation.
///
/// This is the default parameter of [`World`](crate::World) and
/// [`Engine`](crate::Engine), so a bare `Engine` keeps meaning a running
/// engine exactly as before this module existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Running {
    _private: (),
}

/// Authoritative scene role: the native/editor host owns mutations.
///
/// This is the default parameter of `GameWorld`; a bare `GameWorld` keeps
/// meaning the authoritative instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Authoritative {
    _private: (),
}

/// Replica scene role: the browser viewport mirrors serialized snapshots.
///
/// Same world type as [`Authoritative`], only the transport differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Replica {
    _private: (),
}

/// Enforces at compile time that a phantom parameter is a phase marker.
///
/// Implemented only for [`Building`] and [`Running`]; downstream code
/// cannot smuggle an arbitrary type into `World<S>`/`Engine<S>`.
pub trait Phase: Clone + Copy + PartialEq + Default {}

/// Build-time phase marker for the [`Phase`] bound.
impl Phase for Building {}

/// Run-time phase marker for the [`Phase`] bound.
impl Phase for Running {}

/// Enforces at compile time that a phantom parameter is a scene role.
///
/// Implemented only for [`Authoritative`] and [`Replica`].
pub trait SceneRole: Clone + Copy + PartialEq + Default {}

/// Authoritative role marker for the [`SceneRole`] bound.
impl SceneRole for Authoritative {}

/// Replica role marker for the [`SceneRole`] bound.
impl SceneRole for Replica {}

/// Duration in seconds at the frame boundary.
///
/// Transparent over `f32`: serializes as a plain JSON number, so snapshots
/// and WASM payloads are byte-identical to the previous bare-`f32` form.
/// The field is public by the `units` convention (all scalar wrappers
/// construct transparently); prefer [`Seconds::new`] at call sites.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Seconds(pub f32);

impl Seconds {
    /// Zero duration.
    pub const ZERO: Self = Self(0.0);

    /// Wraps a raw second count.
    pub fn new(seconds: f32) -> Self {
        Self(seconds)
    }

    /// Returns the raw second count.
    pub fn get(self) -> f32 {
        self.0
    }
}

impl From<f32> for Seconds {
    /// Wraps a raw second count without validation.
    fn from(seconds: f32) -> Self {
        Self(seconds)
    }
}

impl From<Seconds> for f32 {
    /// Unwraps back to the raw second count.
    fn from(seconds: Seconds) -> Self {
        seconds.0
    }
}

impl Serialize for Seconds {
    /// Encodes as the plain underlying number.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_f32(self.0)
    }
}

impl<'de> Deserialize<'de> for Seconds {
    /// Decodes from the plain underlying number.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        f32::deserialize(deserializer).map(Self)
    }
}

/// Variable-rate frame counter published by [`Time`](crate::Time).
///
/// Transparent over `u64`: serializes as a plain JSON number.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Frame(u64);

impl Frame {
    /// Frame zero: no frame has been published yet.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw frame count.
    pub fn new(frame: u64) -> Self {
        Self(frame)
    }

    /// Returns the raw frame count.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Returns the next frame, saturating instead of wrapping.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl From<u64> for Frame {
    /// Wraps a raw frame count.
    fn from(frame: u64) -> Self {
        Self(frame)
    }
}

impl From<Frame> for u64 {
    /// Unwraps back to the raw frame count.
    fn from(frame: Frame) -> Self {
        frame.0
    }
}

impl Serialize for Frame {
    /// Encodes as the plain underlying number.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for Frame {
    /// Decodes from the plain underlying number.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u64::deserialize(deserializer).map(Self)
    }
}

/// Fixed-rate simulation tick published by [`FixedTime`](crate::FixedTime).
///
/// Transparent over `u64`: serializes as a plain JSON number.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tick(u64);

impl Tick {
    /// Tick zero: no fixed update has run yet.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw tick count.
    pub fn new(tick: u64) -> Self {
        Self(tick)
    }

    /// Returns the raw tick count.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Returns the next tick, saturating instead of wrapping.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl From<u64> for Tick {
    /// Wraps a raw tick count.
    fn from(tick: u64) -> Self {
        Self(tick)
    }
}

impl From<Tick> for u64 {
    /// Unwraps back to the raw tick count.
    fn from(tick: Tick) -> Self {
        tick.0
    }
}

impl Serialize for Tick {
    /// Encodes as the plain underlying number.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for Tick {
    /// Decodes from the plain underlying number.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u64::deserialize(deserializer).map(Self)
    }
}

/// Number of fixed updates scheduled for one frame.
///
/// Transparent over `u32`: serializes as a plain JSON number.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FixedSteps(u32);

impl FixedSteps {
    /// No fixed update scheduled.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw step count.
    pub fn new(steps: u32) -> Self {
        Self(steps)
    }

    /// Returns the raw step count.
    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for FixedSteps {
    /// Wraps a raw step count.
    fn from(steps: u32) -> Self {
        Self(steps)
    }
}

impl From<FixedSteps> for u32 {
    /// Unwraps back to the raw step count.
    fn from(steps: FixedSteps) -> Self {
        steps.0
    }
}

impl Serialize for FixedSteps {
    /// Encodes as the plain underlying number.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u32(self.0)
    }
}

impl<'de> Deserialize<'de> for FixedSteps {
    /// Decodes from the plain underlying number.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u32::deserialize(deserializer).map(Self)
    }
}

/// Monotonic scene-mutation counter shared by `GameWorld` and the editor.
///
/// Frame execution never bumps it; only scene replacement and entity
/// mutation do. Transparent over `u64`: serializes as a plain JSON number,
/// so `/api/status` and `/api/scene` payloads are unchanged.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SceneVersion(u64);

impl SceneVersion {
    /// Initial version: no scene mutation has happened yet.
    pub const ZERO: Self = Self(0);

    /// Wraps a raw version counter.
    pub fn new(version: u64) -> Self {
        Self(version)
    }

    /// Returns the raw version counter.
    pub fn get(self) -> u64 {
        self.0
    }

    /// Advances the counter by one, saturating instead of wrapping.
    pub fn bump(&mut self) {
        self.0 = self.0.saturating_add(1);
    }

    /// Returns the counter advanced by one, saturating instead of wrapping.
    #[must_use]
    pub fn bumped(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl From<u64> for SceneVersion {
    /// Wraps a raw version counter.
    fn from(version: u64) -> Self {
        Self(version)
    }
}

impl From<SceneVersion> for u64 {
    /// Unwraps back to the raw version counter.
    fn from(version: SceneVersion) -> Self {
        version.0
    }
}

impl PartialEq<u64> for SceneVersion {
    /// Compares against a raw version counter (test and snapshot shorthand).
    fn eq(&self, other: &u64) -> bool {
        self.0 == *other
    }
}

impl PartialEq<SceneVersion> for u64 {
    /// Compares a raw version counter against a [`SceneVersion`].
    fn eq(&self, other: &SceneVersion) -> bool {
        *self == other.0
    }
}

impl Serialize for SceneVersion {
    /// Encodes as the plain underlying number.
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

impl<'de> Deserialize<'de> for SceneVersion {
    /// Decodes from the plain underlying number.
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        u64::deserialize(deserializer).map(Self)
    }
}

/// Handles of the entities populated from the current scene.
///
/// Newtype over the scene-entity list so `GameWorld` exposes scene
/// membership distinctly from auxiliary runtime entities. Auxiliary
/// entities inserted through the engine never enter this list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SceneEntities {
    entities: Vec<Entity>,
}

impl SceneEntities {
    /// Creates an empty scene-entity list.
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
        }
    }

    /// Wraps handles already populated from a scene description.
    pub fn from_vec(entities: Vec<Entity>) -> Self {
        Self { entities }
    }

    /// Returns the scene-entity handles.
    pub fn as_slice(&self) -> &[Entity] {
        &self.entities
    }

    /// Number of scene entities currently represented in the ECS.
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the scene currently represents no entity.
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Iterates over the scene-entity handles.
    pub fn iter(&self) -> std::slice::Iter<'_, Entity> {
        self.entities.iter()
    }
}

impl From<Vec<Entity>> for SceneEntities {
    /// Wraps handles already populated from a scene description.
    fn from(entities: Vec<Entity>) -> Self {
        Self::from_vec(entities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_newtypes_round_trip_as_plain_numbers() {
        assert_eq!(
            serde_json::to_string(&Seconds::new(0.5)).expect("seconds serializes"),
            "0.5"
        );
        assert_eq!(
            serde_json::to_string(&Frame::new(3)).expect("frame serializes"),
            "3"
        );
        assert_eq!(
            serde_json::to_string(&Tick::new(7)).expect("tick serializes"),
            "7"
        );
        assert_eq!(
            serde_json::to_string(&FixedSteps::new(8)).expect("steps serialize"),
            "8"
        );
        assert_eq!(
            serde_json::to_string(&SceneVersion::new(5)).expect("version serializes"),
            "5"
        );
        let version: SceneVersion =
            serde_json::from_str("5").expect("version deserializes from a number");
        assert_eq!(version.get(), 5);
    }

    #[test]
    fn scene_version_bump_is_monotonic_and_saturating() {
        let mut version = SceneVersion::ZERO;
        version.bump();
        assert_eq!(version.get(), 1);
        assert_eq!(version.bumped().get(), 2);
        assert_eq!(SceneVersion::new(u64::MAX).bumped().get(), u64::MAX);
    }

    #[test]
    fn scene_entities_wraps_scene_membership_only() {
        let list = SceneEntities::from_vec(vec![Entity::new(1)]);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
        assert_eq!(list.as_slice().len(), 1);
        assert!(SceneEntities::new().is_empty());
    }
}
