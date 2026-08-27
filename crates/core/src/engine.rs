//! Minimal frame runner for the logical engine world.
//!
//! [`Engine`] owns one [`World`] and one [`Schedule`]. It publishes the
//! current [`Time`] resource before running the schedule, giving future
//! physics, rendering and gameplay domains a common frame boundary without
//! coupling the core crate to any particular backend.

use crate::{Schedule, World};

/// Per-frame clock published in the [`World`] resource map.
///
/// `Time` is updated by [`Engine::run_frame`]. Systems should declare a read
/// access to `Time` and use the snapshot supplied for their frame; the value
/// remains unchanged while the schedule is executing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Time {
    delta_seconds: f32,
    elapsed_seconds: f64,
    frame: u64,
}

impl Time {
    /// Creates a clock at frame zero with no elapsed time.
    pub const fn new() -> Self {
        Self {
            delta_seconds: 0.0,
            elapsed_seconds: 0.0,
            frame: 0,
        }
    }

    /// Duration of the current frame in seconds.
    pub fn delta_seconds(self) -> f32 {
        self.delta_seconds
    }

    /// Total simulated frame time in seconds.
    pub fn elapsed_seconds(self) -> f64 {
        self.elapsed_seconds
    }

    /// Number of frames that have been published, starting at one after the
    /// first successful [`Engine::run_frame`] call.
    pub fn frame(self) -> u64 {
        self.frame
    }

    fn advance(&mut self, delta_seconds: f32) {
        assert!(
            delta_seconds.is_finite() && delta_seconds >= 0.0,
            "frame delta must be finite and non-negative, got {delta_seconds}"
        );
        self.delta_seconds = delta_seconds;
        self.elapsed_seconds += f64::from(delta_seconds);
        self.frame = self.frame.saturating_add(1);
    }
}

/// Owns a logical [`World`] and the systems that process its frames.
///
/// This is intentionally a small, backend-neutral host. Physics, rendering,
/// input and assets are registered as resources/systems by higher layers;
/// the core runner only establishes the ordering boundary and publishes
/// [`Time`].
pub struct Engine {
    world: World,
    schedule: Schedule,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Creates an empty engine with a fresh [`World`] and a zeroed [`Time`]
    /// resource.
    pub fn new() -> Self {
        let mut world = World::new();
        let _ = world.insert(Time::new());
        Self {
            world,
            schedule: Schedule::new(),
        }
    }

    /// Returns the logical world for read-only inspection.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Returns the logical world for setup and resource registration.
    ///
    /// Domain resources should be registered between frame calls. Replacing
    /// the `Time` resource is supported; the next frame recreates it if it
    /// has been removed.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Returns the frame schedule for read-only inspection.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Returns the frame schedule for system registration and configuration.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// Runs one frame with `delta_seconds` and publishes [`Time`] first.
    ///
    /// The delta must be finite and non-negative. Systems observe the same
    /// immutable time snapshot during this call; domain-specific mutable
    /// state continues to use the scheduler's declared resource/lane access
    /// contract.
    pub fn run_frame(&mut self, delta_seconds: f32) {
        if let Some(time) = self.world.resources_mut().get_mut::<Time>() {
            time.advance(delta_seconds);
        } else {
            let mut time = Time::new();
            time.advance(delta_seconds);
            let _ = self.world.insert(time);
        }
        self.world.run(&self.schedule);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Resources, System, SystemAccess};
    use std::sync::{Arc, Mutex};

    struct CaptureTime(Arc<Mutex<Vec<Time>>>);

    impl System for CaptureTime {
        fn name(&self) -> &'static str {
            "capture_time"
        }

        fn access(&self) -> SystemAccess {
            SystemAccess::new().reads::<Time>()
        }

        fn run(&self, resources: &Resources) {
            let time = *resources.get::<Time>().expect("engine publishes Time");
            self.0.lock().expect("capture lock").push(time);
        }
    }

    #[test]
    fn run_frame_publishes_monotonic_time() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine
            .schedule_mut()
            .add_system(CaptureTime(captured.clone()));

        engine.run_frame(0.25);
        engine.run_frame(0.5);

        let captured = captured.lock().expect("capture lock");
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].frame(), 1);
        assert_eq!(captured[0].delta_seconds(), 0.25);
        assert_eq!(captured[0].elapsed_seconds(), 0.25);
        assert_eq!(captured[1].frame(), 2);
        assert_eq!(captured[1].delta_seconds(), 0.5);
        assert_eq!(captured[1].elapsed_seconds(), 0.75);
    }

    #[test]
    fn engine_keeps_world_and_schedule_together() {
        let mut engine = Engine::new();
        let _ = engine.world_mut().insert(42_u32);
        assert!(engine.world().store().is_some());
        assert!(engine.schedule().is_empty());
    }

    #[test]
    fn missing_time_resource_is_recreated_on_next_frame() {
        let mut engine = Engine::new();
        assert!(engine.world_mut().remove::<Time>().is_some());

        engine.run_frame(0.1);

        let time = engine
            .world()
            .resources()
            .get::<Time>()
            .expect("Time is restored");
        assert_eq!(time.frame(), 1);
        assert_eq!(time.delta_seconds(), 0.1);
    }

    #[test]
    #[should_panic(expected = "frame delta must be finite and non-negative")]
    fn run_frame_rejects_invalid_delta() {
        let mut engine = Engine::new();
        engine.run_frame(f32::NAN);
    }
}
