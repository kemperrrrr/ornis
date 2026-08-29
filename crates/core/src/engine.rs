//! Minimal frame runner for the logical engine world.
//!
//! [`Engine`] owns one [`World`] and a variable-rate [`Schedule`] plus a
//! fixed-rate schedule. It publishes the current [`Time`] and [`FixedTime`]
//! resources before running domain systems, giving gameplay and physics a
//! common bounded fixed-update boundary without coupling the core crate to a
//! particular backend.

use crate::{InputState, Schedule, World};

/// Default simulation step used by the backend-neutral fixed-update host.
pub const DEFAULT_FIXED_DELTA_SECONDS: f32 = 1.0 / 60.0;

/// Default maximum number of fixed updates an engine frame may catch up.
pub const DEFAULT_MAX_FIXED_STEPS_PER_FRAME: u32 = 8;

/// Per-frame clock published in the [`World`] resource map.
///
/// `Time` is updated by [`Engine::run_frame`]. Systems should declare a read
/// access to `Time` and use the snapshot supplied for their frame; the value
/// remains unchanged while either schedule is executing.
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

    /// Duration of the current variable-rate frame in seconds.
    pub fn delta_seconds(self) -> f32 {
        self.delta_seconds
    }

    /// Total elapsed variable-rate frame time in seconds.
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

/// Fixed-rate simulation clock published in the [`World`] resource map.
///
/// The engine accumulates variable-rate frame deltas and runs at most
/// `max_steps_per_frame` fixed updates per frame. Excess catch-up time is
/// deliberately dropped after the cap, preventing a long render hitch from
/// creating an unbounded spiral of simulation work. `alpha()` is the
/// fractional remainder for optional render interpolation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FixedTime {
    delta_seconds: f32,
    accumulator_seconds: f32,
    elapsed_seconds: f64,
    tick: u64,
    steps_this_frame: u32,
    current_step: u32,
    alpha: f32,
    dropped_seconds: f64,
    max_steps_per_frame: u32,
}

impl Default for FixedTime {
    fn default() -> Self {
        Self::new(
            DEFAULT_FIXED_DELTA_SECONDS,
            DEFAULT_MAX_FIXED_STEPS_PER_FRAME,
        )
    }
}

impl FixedTime {
    /// Creates a fixed clock with `fixed_delta_seconds` and a per-frame cap.
    ///
    /// A fixed step must be finite and positive. The cap must be non-zero so
    /// every accepted frame can make progress when enough time is available.
    /// The accumulator starts empty; use [`Engine::run_frame`] to advance it.
    pub fn new(fixed_delta_seconds: f32, max_steps_per_frame: u32) -> Self {
        assert!(
            fixed_delta_seconds.is_finite() && fixed_delta_seconds > 0.0,
            "fixed delta must be finite and positive, got {fixed_delta_seconds}"
        );
        assert!(
            max_steps_per_frame > 0,
            "maximum fixed steps per frame must be positive"
        );
        Self {
            delta_seconds: fixed_delta_seconds,
            accumulator_seconds: 0.0,
            elapsed_seconds: 0.0,
            tick: 0,
            steps_this_frame: 0,
            current_step: 0,
            alpha: 0.0,
            dropped_seconds: 0.0,
            max_steps_per_frame,
        }
    }

    /// Duration of one fixed simulation update in seconds.
    pub fn delta_seconds(self) -> f32 {
        self.delta_seconds
    }

    /// Unconsumed fraction of simulation time after the current frame.
    pub fn accumulator_seconds(self) -> f32 {
        self.accumulator_seconds
    }

    /// Fractional remainder divided by the fixed step, in `0.0..1.0`.
    ///
    /// Render interpolation may use this value while fixed systems should
    /// consume the exact [`Self::delta_seconds`] instead.
    pub fn alpha(self) -> f32 {
        self.alpha
    }

    /// Total simulated time advanced by fixed updates, in seconds.
    pub fn elapsed_seconds(self) -> f64 {
        self.elapsed_seconds
    }

    /// Number of fixed updates executed since the clock was created.
    pub fn tick(self) -> u64 {
        self.tick
    }

    /// Number of fixed updates scheduled during the current frame.
    pub fn steps_this_frame(self) -> u32 {
        self.steps_this_frame
    }

    /// One-based index of the current frame's fixed update.
    ///
    /// During a fixed system this identifies the update being executed. After
    /// the frame it remains the last index until the next frame; it is zero
    /// when the current frame had no fixed updates.
    pub fn current_step(self) -> u32 {
        self.current_step
    }

    /// Maximum fixed updates the host may execute for one frame.
    pub fn max_steps_per_frame(self) -> u32 {
        self.max_steps_per_frame
    }

    /// Cumulative simulation time discarded by the hitch protection cap.
    pub fn dropped_seconds(self) -> f64 {
        self.dropped_seconds
    }

    fn begin_frame(&mut self, frame_delta: f32) -> u32 {
        debug_assert!(frame_delta.is_finite() && frame_delta >= 0.0);
        let max_accumulator = self.delta_seconds * (self.max_steps_per_frame as f32 + 1.0);
        let available_capacity = max_accumulator - self.accumulator_seconds;
        if frame_delta > available_capacity {
            self.dropped_seconds += f64::from(frame_delta - available_capacity);
            self.accumulator_seconds = max_accumulator;
        } else {
            self.accumulator_seconds += frame_delta;
        }

        // Keep one extra step in the bounded accumulator as a deterministic
        // hitch buffer, then drop it if the execution cap is reached. This
        // preserves the fractional remainder while ensuring the loop below
        // performs no more than the configured number of fixed updates.
        let mut remaining = self.accumulator_seconds;
        let mut available_steps = 0_u32;
        let probe_limit = self.max_steps_per_frame.saturating_add(1);
        while remaining >= self.delta_seconds && available_steps < probe_limit {
            remaining -= self.delta_seconds;
            available_steps += 1;
        }
        let steps = available_steps.min(self.max_steps_per_frame);
        let dropped_steps = available_steps.saturating_sub(steps);
        self.dropped_seconds += f64::from(dropped_steps) * f64::from(self.delta_seconds);
        self.accumulator_seconds = remaining + steps as f32 * self.delta_seconds;
        self.steps_this_frame = steps;
        self.current_step = 0;
        self.update_alpha();
        steps
    }

    fn start_step(&mut self) {
        debug_assert!(self.accumulator_seconds >= self.delta_seconds);
        self.accumulator_seconds = (self.accumulator_seconds - self.delta_seconds).max(0.0);
        self.tick = self.tick.saturating_add(1);
        self.elapsed_seconds += f64::from(self.delta_seconds);
        self.current_step = self.current_step.saturating_add(1);
        self.update_alpha();
    }

    fn update_alpha(&mut self) {
        self.alpha = (self.accumulator_seconds / self.delta_seconds).clamp(0.0, 1.0);
    }
}

/// Owns a logical [`World`] and the systems that process its frames.
///
/// The variable-rate schedule runs once per frame after zero or more runs of
/// the fixed-rate schedule. Physics and fixed gameplay systems belong in
/// [`Engine::fixed_schedule_mut`]; render extraction and other once-per-frame
/// consumers belong in [`Engine::schedule_mut`]. This is intentionally a
/// small, backend-neutral host: domain algorithms remain registered by
/// higher layers and the core runner does not choose a physics or render
/// backend.
pub struct Engine {
    world: World,
    schedule: Schedule,
    fixed_schedule: Schedule,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    /// Creates an empty engine with fresh [`World`], [`Time`], [`FixedTime`]
    /// and [`InputState`] resources.
    pub fn new() -> Self {
        let mut world = World::new();
        let _ = world.insert(Time::new());
        let _ = world.insert(FixedTime::default());
        let _ = world.insert(InputState::new());
        Self {
            world,
            schedule: Schedule::new(),
            fixed_schedule: Schedule::new(),
        }
    }

    /// Returns the logical world for read-only inspection.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Returns the logical world for setup and resource registration.
    ///
    /// Domain resources should be registered between frame calls. Replacing
    /// the `Time` or `FixedTime` resource is supported; the next frame
    /// recreates a missing clock with its default configuration.
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Returns the variable-rate frame schedule for read-only inspection.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Returns the variable-rate frame schedule for system registration.
    ///
    /// It runs once after all fixed updates for the frame. Render extraction
    /// should normally be registered here so it observes final fixed-step
    /// poses without being repeated for each substep.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// Returns the fixed-rate schedule for read-only inspection.
    pub fn fixed_schedule(&self) -> &Schedule {
        &self.fixed_schedule
    }

    /// Returns the fixed-rate schedule for system registration.
    ///
    /// Every system in this schedule runs once per fixed update selected by
    /// the accumulator. Systems should read [`FixedTime`] for the exact step
    /// duration and declare all resource/lane accesses normally.
    pub fn fixed_schedule_mut(&mut self) -> &mut Schedule {
        &mut self.fixed_schedule
    }

    /// Registers one fixed-rate system and returns the engine for chaining.
    pub fn add_fixed_system<S: crate::System + 'static>(&mut self, system: S) -> &mut Self {
        self.fixed_schedule.add_system(system);
        self
    }

    /// Runs one frame with `delta_seconds` and publishes [`Time`] first.
    ///
    /// The fixed accumulator is advanced before the fixed schedule runs. Each
    /// selected fixed update receives the same frame-level input snapshot and
    /// the current [`FixedTime`] step; the variable-rate schedule then runs
    /// once. Input events accumulated by a platform adapter are visible to
    /// both schedules during the frame. Consumers of transient pointer/wheel
    /// deltas should normally run in the once-per-frame schedule so a catch-up
    /// frame does not apply one event repeatedly. After all systems finish,
    /// transient deltas are cleared from [`InputState`]; held keys/buttons
    /// persist.
    ///
    /// The delta must be finite and non-negative. Domain-specific mutable
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

        if self.world.resources().get::<FixedTime>().is_none() {
            let _ = self.world.insert(FixedTime::default());
        }
        let fixed_steps = self
            .world
            .resources_mut()
            .get_mut::<FixedTime>()
            .expect("engine publishes FixedTime")
            .begin_frame(delta_seconds);

        for _ in 0..fixed_steps {
            self.world
                .resources_mut()
                .get_mut::<FixedTime>()
                .expect("engine publishes FixedTime")
                .start_step();
            self.world.run(&self.fixed_schedule);
        }
        self.world.run(&self.schedule);

        if let Some(input) = self.world.resources_mut().get_mut::<InputState>() {
            input.clear_frame_transients();
        }
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

    struct CaptureFixedTime(Arc<Mutex<Vec<FixedTime>>>);

    impl System for CaptureFixedTime {
        fn name(&self) -> &'static str {
            "capture_fixed_time"
        }

        fn access(&self) -> SystemAccess {
            SystemAccess::new().reads::<FixedTime>()
        }

        fn run(&self, resources: &Resources) {
            let time = *resources
                .get::<FixedTime>()
                .expect("engine publishes FixedTime");
            self.0.lock().expect("capture lock").push(time);
        }
    }

    struct Trace {
        fixed: bool,
    }

    impl System for Trace {
        fn name(&self) -> &'static str {
            if self.fixed {
                "trace_fixed"
            } else {
                "trace_frame"
            }
        }

        fn access(&self) -> SystemAccess {
            let access = SystemAccess::new().reads::<TraceLog>();
            if self.fixed {
                access.reads::<FixedTime>()
            } else {
                access.reads::<Time>()
            }
        }

        fn run(&self, resources: &Resources) {
            let log = resources.get::<TraceLog>().expect("trace log");
            log.0
                .lock()
                .expect("trace lock")
                .push(if self.fixed { "fixed" } else { "frame" });
        }
    }

    struct TraceLog(Arc<Mutex<Vec<&'static str>>>);

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
    fn fixed_schedule_is_bounded_and_runs_before_frame_schedule() {
        let fixed = Arc::new(Mutex::new(Vec::new()));
        let frame = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine
            .fixed_schedule_mut()
            .add_system(CaptureFixedTime(fixed.clone()));
        engine.schedule_mut().add_system(CaptureTime(frame.clone()));

        let delta = FixedTime::default().delta_seconds();
        engine.run_frame(delta * 100.0);

        assert_eq!(fixed.lock().expect("fixed capture lock").len(), 8);
        assert_eq!(frame.lock().expect("frame capture lock").len(), 1);
        let fixed_time = *engine
            .world()
            .resources()
            .get::<FixedTime>()
            .expect("fixed time");
        assert_eq!(fixed_time.steps_this_frame(), 8);
        assert_eq!(fixed_time.tick(), 8);
        assert!(fixed_time.dropped_seconds() > 0.0);
        assert!(fixed_time.accumulator_seconds() < delta);
    }

    #[test]
    fn fixed_schedule_preserves_partial_time_and_orders_before_frame() {
        let trace = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        let _ = engine.world_mut().insert(TraceLog(trace.clone()));
        engine
            .fixed_schedule_mut()
            .add_system(Trace { fixed: true });
        engine.schedule_mut().add_system(Trace { fixed: false });

        let delta = FixedTime::default().delta_seconds();
        engine.run_frame(delta * 0.5);
        assert_eq!(*trace.lock().expect("trace lock"), vec!["frame"]);
        trace.lock().expect("trace lock").clear();
        engine.run_frame(delta * 0.5);

        let entries = trace.lock().expect("trace lock").clone();
        assert_eq!(entries, vec!["fixed", "frame"]);
        let fixed_time = *engine
            .world()
            .resources()
            .get::<FixedTime>()
            .expect("fixed time");
        assert_eq!(fixed_time.current_step(), 1);
        assert_eq!(fixed_time.tick(), 1);
        assert!(fixed_time.alpha() < 0.01);
    }

    #[test]
    fn engine_keeps_world_and_schedules_together() {
        let mut engine = Engine::new();
        let _ = engine.world_mut().insert(42_u32);
        assert!(engine.world().store().is_some());
        assert!(engine.schedule().is_empty());
        assert!(engine.fixed_schedule().is_empty());
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
    fn missing_fixed_time_resource_is_recreated_on_next_frame() {
        let mut engine = Engine::new();
        assert!(engine.world_mut().remove::<FixedTime>().is_some());

        engine.run_frame(FixedTime::default().delta_seconds());

        let fixed_time = engine
            .world()
            .resources()
            .get::<FixedTime>()
            .expect("FixedTime is restored");
        assert_eq!(fixed_time.tick(), 1);
    }

    #[test]
    #[should_panic(expected = "frame delta must be finite and non-negative")]
    fn run_frame_rejects_invalid_delta() {
        let mut engine = Engine::new();
        engine.run_frame(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "fixed delta must be finite and positive")]
    fn fixed_time_rejects_invalid_step() {
        let _ = FixedTime::new(0.0, 8);
    }

    #[test]
    #[should_panic(expected = "maximum fixed steps per frame must be positive")]
    fn fixed_time_rejects_zero_step_cap() {
        let _ = FixedTime::new(DEFAULT_FIXED_DELTA_SECONDS, 0);
    }
}
