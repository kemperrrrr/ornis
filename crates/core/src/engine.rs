//! Minimal frame runner for the logical engine world.
//!
//! [`Engine`] owns one [`World`] and a variable-rate [`Schedule`] plus a
//! fixed-rate schedule. It publishes the current [`Time`] and [`FixedTime`]
//! resources before running domain systems, giving gameplay and physics a
//! common bounded fixed-update boundary without coupling the core crate to a
//! particular backend.

use std::marker::PhantomData;

use crate::typestate::{Building, FixedSteps, Frame, Phase, Running, Seconds, Tick};
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

    /// Duration of the current variable-rate frame as a [`Seconds`] newtype.
    pub fn delta(self) -> Seconds {
        Seconds::new(self.delta_seconds)
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

    /// Published frame count as a [`Frame`] newtype.
    pub fn frame_id(self) -> Frame {
        Frame::new(self.frame)
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

    /// Advances the clock by a [`Seconds`] delta (same contract as the
    /// `f32` path: finite and non-negative).
    #[allow(dead_code)]
    fn advance_secs(&mut self, delta: Seconds) {
        self.advance(delta.get());
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

    /// Creates a fixed clock from [`Seconds`]/[`FixedSteps`] newtypes.
    ///
    /// Same contract as [`FixedTime::new`]: the step must be finite and
    /// positive, the cap non-zero.
    pub fn new_secs(fixed_delta: Seconds, max_steps_per_frame: FixedSteps) -> Self {
        Self::new(fixed_delta.get(), max_steps_per_frame.get())
    }

    /// Duration of one fixed simulation update in seconds.
    pub fn delta_seconds(self) -> f32 {
        self.delta_seconds
    }

    /// Duration of one fixed simulation update as a [`Seconds`] newtype.
    pub fn delta(self) -> Seconds {
        Seconds::new(self.delta_seconds)
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

    /// Fixed-update count as a [`Tick`] newtype.
    pub fn tick_id(self) -> Tick {
        Tick::new(self.tick)
    }

    /// Number of fixed updates scheduled during the current frame.
    pub fn steps_this_frame(self) -> u32 {
        self.steps_this_frame
    }

    /// Fixed updates scheduled during the current frame as [`FixedSteps`].
    pub fn steps(self) -> FixedSteps {
        FixedSteps::new(self.steps_this_frame)
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

    /// Advances one replay/journal tick: no accumulator, no `begin_frame`.
    ///
    /// Implements the `run_fixed_step` clock contract from
    /// `docs/networking-design.md` §2.1/§6.1: the accumulator is dropped
    /// (`0`), `alpha` is `0`, and `steps_this_frame`/`current_step` read `1`
    /// because a journaled tick has no frame around it. Keeping these fields
    /// canonical is what makes `FixedTime` bytes independent of the host's
    /// frame slicing, so a snapshot taken after a replay tick matches one
    /// taken from the same tick in a live `run_frame`.
    fn start_replay_step(&mut self) {
        self.accumulator_seconds = 0.0;
        self.tick = self.tick.saturating_add(1);
        self.elapsed_seconds += f64::from(self.delta_seconds);
        self.steps_this_frame = 1;
        self.current_step = 1;
        self.alpha = 0.0;
    }

    fn update_alpha(&mut self) {
        self.alpha = (self.accumulator_seconds / self.delta_seconds).clamp(0.0, 1.0);
    }
}

/// Named execution stage over the existing fixed/frame schedule split.
///
/// Stages are logical plans, not extra worlds: `PreUpdate` and `Input` run
/// once per frame before the fixed catch-up loop, `Gameplay` is the bounded
/// fixed-rate plan, `PostFrame` is the once-per-frame plan after all
/// fixed updates, and `Last` is the frame-final plan that runs after the
/// frame structural queue has been applied. Physically `PostFrame` reuses the
/// legacy variable [`Engine::schedule`] storage, `Gameplay` reuses
/// [`Engine::fixed_schedule`], and `Last` has its own schedule; the
/// pre-existing entry points delegate unchanged; see
/// [`Engine::stage_schedule`].
///
/// `#[non_exhaustive]`: stages are an engine-owned execution order, and a new
/// plan must not break downstream exhaustive `match`es (it did once when
/// `Last` landed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Stage {
    /// Between-frame input ingest (reserved, runs first, once per frame).
    PreUpdate,
    /// Once-per-frame input consumers (runs second, once per frame).
    Input,
    /// Fixed-step intent and physics (bounded catch-up loop).
    Gameplay,
    /// Pose propagation and audio/render extraction views (runs after all
    /// fixed updates).
    PostFrame,
    /// Frame-final plan: the frame structural queue
    /// ([`FrameStructuralQueue`](crate::mutation::FrameStructuralQueue)) is
    /// applied first, then registered systems run once per frame — so a
    /// `Last` system observes this frame's `add`/`remove`/`spawn` results.
    ///
    /// Ops queued *by* a `Last` system cannot be applied in the same frame
    /// (the single apply already happened); they land at the next frame's
    /// apply, still in FIFO order. Queue frame-final structural work from
    /// `PostFrame`.
    Last,
}

impl Stage {
    /// Short stable label of the stage.
    pub fn name(self) -> &'static str {
        match self {
            Stage::PreUpdate => "pre_update",
            Stage::Input => "input",
            Stage::Gameplay => "gameplay",
            Stage::PostFrame => "post_frame",
            Stage::Last => "last",
        }
    }

    /// Whether the stage runs inside the bounded fixed catch-up loop.
    pub fn is_fixed(self) -> bool {
        matches!(self, Stage::Gameplay)
    }
}

/// Checks that a structural writer's queue kind matches the stage it runs in.
///
/// The fixed queue drains after every fixed tick and the frame queue once per
/// frame; putting a writer in the wrong plan is what makes results depend on
/// FPS (a frame-queued spawn from `Gameplay` would land at a frame boundary in
/// the middle of the fixed schedule). Registration is the only cheap place to
/// catch it — the access set names the queue, and the stage names the plan.
///
/// # Panics
/// When a fixed writer sits outside `Stage::Gameplay`, or a frame writer
/// inside it. Both are setup bugs, so this fails at registration instead of
/// producing a plausible but frame-dependent simulation.
fn assert_writer_stage_matches(stage: Stage, system: &'static str, access: &crate::SystemAccess) {
    let writes = |id| access.writes.contains(&id);
    let fixed_writer = writes(std::any::TypeId::of::<crate::mutation::FixedStructuralQueue>());
    let frame_writer = writes(std::any::TypeId::of::<crate::mutation::FrameStructuralQueue>());
    if fixed_writer && !stage.is_fixed() {
        panic!(
            "engine: system `{system}` declares `StructuralWriter<FixedStep>` but is registered in \
             stage `{}` — the fixed queue drains after every fixed tick, so fixed writers belong \
             only to `Stage::Gameplay` (use `StructuralWriter<FrameStep>` for once-per-frame stages)",
            stage.name()
        );
    }
    if frame_writer && stage.is_fixed() {
        panic!(
            "engine: system `{system}` declares `StructuralWriter<FrameStep>` but is registered in \
             stage `{}` — the frame queue drains once per frame, so frame writers cannot live in \
             the fixed plan; a spawn queued here would land at a frame-dependent moment. Use \
             `StructuralWriter<FixedStep>`, which applies after this tick",
            stage.name()
        );
    }
}

/// Owns a logical [`World`] and the systems that process its frames.
///
/// The variable-rate schedule runs once per frame after zero or more runs of
/// the fixed-rate schedule. Physics and fixed gameplay systems belong in
/// the fixed plan; render extraction and other once-per-frame consumers
/// belong in the variable plan. This is intentionally a small,
/// backend-neutral host: domain algorithms remain registered by higher
/// layers and the core runner does not choose a physics or render backend.
///
/// Staged plans ([`Stage`]) run in `PreUpdate → Input → Gameplay(fixed) →
/// PostFrame → Last` order; `PreUpdate`/`Input` are new once-per-frame
/// schedules ahead of the fixed loop, `Gameplay`/`PostFrame` delegate to
/// the pre-existing `fixed_schedule`/`schedule` storage, and `Last` runs
/// once after the deferred structural queue is applied.
///
/// Typestate: the `State` phantom tracks the build/run phase
/// ([`Building`]/[`Running`], default [`Running`]). The canonical
/// construction route is [`EngineBuilder`] (or [`Engine::new_building`] +
/// [`Engine::build`]): schedule/system registration (`schedule_mut`,
/// `fixed_schedule_mut`, `stage_schedule_mut`, `add_stage_system`,
/// `add_fixed_system`) belongs to the [`Building`] phase, and `build()`
/// seals the engine before the frame loop starts. The same `*_mut`
/// accessors stay available on [`Engine<Running>`] as a documented
/// legacy escape hatch for late-registration hosts (the `install_*`
/// family: `install_gameplay`, `install_orbit_camera`,
/// `install_gpu_resources`, `install_unified_runtime` and siblings take
/// `&mut Engine<Running>`), so every existing host compiles unchanged.
/// New setup code should prefer the builder; the `Running` mutators must
/// only run between frames, never while systems execute.
pub struct Engine<State: Phase = Running> {
    world: World<State>,
    schedule: Schedule,
    fixed_schedule: Schedule,
    pre_update: Schedule,
    input: Schedule,
    last: Schedule,
    phase: PhantomData<State>,
}

impl<State: Phase> Default for Engine<State> {
    /// Creates an empty engine with fresh [`World`], [`Time`], [`FixedTime`]
    /// and [`InputState`] resources.
    fn default() -> Self {
        Self::new_in_phase()
    }
}

impl Engine<Running> {
    /// Creates an empty running engine with fresh [`World`], [`Time`],
    /// [`FixedTime`] and [`InputState`] resources.
    ///
    /// This direct constructor stays for the late-registration hosts that
    /// install systems after the engine exists (the `install_*` family);
    /// new setup code should prefer the canonical [`EngineBuilder`].
    pub fn new() -> Self {
        Self::new_in_phase()
    }
}

impl Engine<Building> {
    /// Creates an empty build-phase engine for registration-heavy setup.
    ///
    /// This is the canonical construction route (together with
    /// [`EngineBuilder`]): register systems and resources through the
    /// `*_mut` accessors below, then seal the engine with [`Engine::build`]
    /// before entering the frame loop.
    pub fn new_building() -> Self {
        Self::new_in_phase()
    }

    /// Seals a build-phase engine into the running phase.
    ///
    /// The world (including the authoritative [`SmartStore`](crate::SmartStore))
    /// and all staged schedules move over untouched; only the phase marker
    /// changes. Frame execution should start after this call.
    #[must_use]
    pub fn build(self) -> Engine<Running> {
        Engine {
            world: self.world.build(),
            schedule: self.schedule,
            fixed_schedule: self.fixed_schedule,
            pre_update: self.pre_update,
            input: self.input,
            last: self.last,
            phase: PhantomData,
        }
    }
}

/// Canonical `#[must_use]` builder for an [`Engine`].
///
/// The builder owns an [`Engine<Building>`] and is the strict construction
/// route: configure the fixed clock, register systems and resources through
/// the `*_mut` accessors, then `build()` seals a running engine. The fixed
/// configuration is applied once at seal time, so it cannot be widened
/// again by accident afterwards.
///
/// [`Engine<Running>`] keeps its own `schedule_mut`/`fixed_schedule_mut`/
/// `stage_schedule_mut` accessors as a documented legacy escape hatch for
/// the `install_*` family (50+ call sites taking `&mut Engine`), which is
/// deliberately not migrated; new code should register on the builder.
#[must_use]
pub struct EngineBuilder {
    inner: Engine<Building>,
    fixed_delta: Seconds,
    max_steps: FixedSteps,
}

impl std::fmt::Debug for EngineBuilder {
    /// Shows the pending fixed-clock configuration (not the schedules).
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineBuilder")
            .field("fixed_delta", &self.fixed_delta)
            .field("max_steps", &self.max_steps)
            .finish_non_exhaustive()
    }
}

impl EngineBuilder {
    /// Starts a builder with the default fixed clock
    /// ([`DEFAULT_FIXED_DELTA_SECONDS`], [`DEFAULT_MAX_FIXED_STEPS_PER_FRAME`])
    /// and empty build-phase schedules.
    pub fn new() -> Self {
        Self {
            inner: Engine::new_building(),
            fixed_delta: Seconds::new(DEFAULT_FIXED_DELTA_SECONDS),
            max_steps: FixedSteps::new(DEFAULT_MAX_FIXED_STEPS_PER_FRAME),
        }
    }

    /// Sets the fixed simulation step (applied to the sealed engine).
    pub fn fixed_delta(mut self, fixed_delta: Seconds) -> Self {
        self.fixed_delta = fixed_delta;
        self
    }

    /// Sets the maximum fixed updates per frame (applied to the sealed
    /// engine; hitch protection cap).
    pub fn max_steps(mut self, max_steps: FixedSteps) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Returns the build-phase world for setup and resource registration.
    ///
    /// Canonical registration accessor: domain resources registered here
    /// move into the sealed engine untouched.
    pub fn world_mut(&mut self) -> &mut World<Building> {
        self.inner.world_mut()
    }

    /// Returns the build-phase variable-rate schedule for registration.
    ///
    /// Canonical accessor for the [`Stage::PostFrame`] plan; runs once
    /// after all fixed updates for the frame.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        self.inner.schedule_mut()
    }

    /// Returns the build-phase fixed-rate schedule for registration.
    ///
    /// Canonical accessor for the [`Stage::Gameplay`] plan; every system
    /// runs once per fixed update selected by the accumulator.
    pub fn fixed_schedule_mut(&mut self) -> &mut Schedule {
        self.inner.fixed_schedule_mut()
    }

    /// Returns the build-phase schedule backing a named [`Stage`].
    ///
    /// Canonical staged registration route; execution order stays
    /// `PreUpdate → Input → Gameplay(fixed) → PostFrame → Last`.
    pub fn stage_schedule_mut(&mut self, stage: Stage) -> &mut Schedule {
        self.inner.stage_schedule_mut(stage)
    }

    /// Registers one system into a named [`Stage`] plan of the
    /// build-phase engine.
    pub fn add_stage_system<S: crate::System + 'static>(
        &mut self,
        stage: Stage,
        system: S,
    ) -> &mut Self {
        self.inner.add_stage_system(stage, system);
        self
    }

    /// Registers one fixed-rate system into the build-phase engine.
    pub fn add_fixed_system<S: crate::System + 'static>(&mut self, system: S) -> &mut Self {
        self.inner.add_fixed_system(system);
        self
    }

    /// Seals the builder into a running engine with the configured fixed
    /// clock (replaces the default [`FixedTime`] published at creation).
    pub fn build(self) -> Engine<Running> {
        let Self {
            mut inner,
            fixed_delta,
            max_steps,
        } = self;
        let _ = inner
            .world_mut()
            .insert(FixedTime::new(fixed_delta.get(), max_steps.get()));
        inner.build()
    }
}

impl Default for EngineBuilder {
    /// Starts a builder with the default fixed clock.
    fn default() -> Self {
        Self::new()
    }
}

impl<State: Phase> Engine<State> {
    fn new_in_phase() -> Self {
        let mut world = World::<State>::new_in_phase();
        let _ = world.insert(Time::new());
        let _ = world.insert(FixedTime::default());
        let _ = world.insert(InputState::new());
        Self {
            world,
            schedule: Schedule::new(),
            fixed_schedule: Schedule::new(),
            pre_update: Schedule::new(),
            input: Schedule::new(),
            last: Schedule::new(),
            phase: PhantomData,
        }
    }

    /// Returns the logical world for read-only inspection.
    pub fn world(&self) -> &World<State> {
        &self.world
    }

    /// Returns the variable-rate frame schedule for read-only inspection.
    ///
    /// This is the [`Stage::PostFrame`] storage: it runs once after all
    /// fixed updates for the frame.
    pub fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// Returns the fixed-rate schedule for read-only inspection.
    ///
    /// This is the [`Stage::Gameplay`] storage.
    pub fn fixed_schedule(&self) -> &Schedule {
        &self.fixed_schedule
    }

    /// Returns the schedule backing a named [`Stage`] for inspection.
    ///
    /// `PreUpdate`/`Input` are dedicated once-per-frame schedules ahead of
    /// the fixed loop; `Gameplay` delegates to [`Engine::fixed_schedule`],
    /// `PostFrame` to [`Engine::schedule`], and `Last` to the engine's
    /// frame-final schedule (structural apply happens right before it).
    pub fn stage_schedule(&self, stage: Stage) -> &Schedule {
        match stage {
            Stage::PreUpdate => &self.pre_update,
            Stage::Input => &self.input,
            Stage::Gameplay => &self.fixed_schedule,
            Stage::PostFrame => &self.schedule,
            Stage::Last => &self.last,
        }
    }
}

impl Engine<Building> {
    /// Returns the build-phase world for setup and resource registration.
    ///
    /// Canonical registration accessor: domain resources registered here
    /// move into the sealed engine untouched. Replacing the `Time` or
    /// `FixedTime` resource is supported; the next frame recreates a
    /// missing clock with its default configuration.
    pub fn world_mut(&mut self) -> &mut World<Building> {
        &mut self.world
    }

    /// Returns the build-phase variable-rate schedule for registration.
    ///
    /// Canonical accessor for the [`Stage::PostFrame`] plan: it runs once
    /// after all fixed updates for the frame. Render extraction should
    /// normally be registered here so it observes final fixed-step poses
    /// without being repeated for each substep.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// Returns the build-phase fixed-rate schedule for registration.
    ///
    /// Canonical accessor for the [`Stage::Gameplay`] plan: every system
    /// runs once per fixed update selected by the accumulator. Systems
    /// should read [`FixedTime`] for the exact step duration and declare
    /// all resource/lane accesses normally.
    pub fn fixed_schedule_mut(&mut self) -> &mut Schedule {
        &mut self.fixed_schedule
    }

    /// Returns the build-phase schedule backing a named [`Stage`].
    ///
    /// Canonical staged registration route; execution order stays
    /// `PreUpdate → Input → Gameplay(fixed) → PostFrame → Last`.
    pub fn stage_schedule_mut(&mut self, stage: Stage) -> &mut Schedule {
        match stage {
            Stage::PreUpdate => &mut self.pre_update,
            Stage::Input => &mut self.input,
            Stage::Gameplay => &mut self.fixed_schedule,
            Stage::PostFrame => &mut self.schedule,
            Stage::Last => &mut self.last,
        }
    }

    /// Registers one system into a named [`Stage`] plan of the
    /// build-phase engine.
    ///
    /// Delegates to [`Engine::stage_schedule_mut`]; variable-rate gameplay
    /// ticks (`mutation_tick`) belong in [`Stage::PostFrame`] storage: the
    /// once-per-frame plan after the final fixed update, so one JSON-codec
    /// tick round-trip covers the whole frame instead of repeating per
    /// fixed substep. A dedicated variable-gameplay plan would only pay
    /// off with a second consumer; until then `PostFrame` is the variable
    /// plan, not a stopgap.
    pub fn add_stage_system<S: crate::System + 'static>(
        &mut self,
        stage: Stage,
        system: S,
    ) -> &mut Self {
        assert_writer_stage_matches(stage, system.name(), &system.access());
        self.stage_schedule_mut(stage).add_system(system);
        self
    }

    /// Registers one fixed-rate system into the build-phase engine.
    pub fn add_fixed_system<S: crate::System + 'static>(&mut self, system: S) -> &mut Self {
        self.fixed_schedule.add_system(system);
        self
    }
}

impl Engine<Running> {
    /// Returns the running world for operational setup between frames.
    ///
    /// Legacy-compatible accessor: the `install_*` family and scene
    /// replacement (`GameWorld::replace_scene`) take `&mut Engine<Running>`
    /// and register through this. Must only run between frames, never
    /// while systems execute. New setup code should prefer
    /// [`EngineBuilder::world_mut`]; replacing the `Time` or `FixedTime`
    /// resource stays supported and the next frame recreates a missing
    /// clock with its default configuration.
    pub fn world_mut(&mut self) -> &mut World<Running> {
        &mut self.world
    }

    /// Returns the variable-rate schedule for late system registration.
    ///
    /// Legacy escape hatch (the [`Stage::PostFrame`] plan): the `install_*`
    /// family registers here after the engine already runs, so this
    /// accessor deliberately stays on the running phase. New setup code
    /// should prefer [`EngineBuilder::schedule_mut`]; calls must happen
    /// between frames, never while systems execute.
    pub fn schedule_mut(&mut self) -> &mut Schedule {
        &mut self.schedule
    }

    /// Returns the fixed-rate schedule for late system registration.
    ///
    /// Legacy escape hatch (the [`Stage::Gameplay`] plan) for the
    /// `install_*` family; new setup code should prefer
    /// [`EngineBuilder::fixed_schedule_mut`]. Between frames only.
    pub fn fixed_schedule_mut(&mut self) -> &mut Schedule {
        &mut self.fixed_schedule
    }

    /// Returns the schedule backing a named [`Stage`] for late registration.
    ///
    /// Legacy escape hatch routing to the same storage as the build-phase
    /// accessor; execution order stays
    /// `PreUpdate → Input → Gameplay(fixed) → PostFrame → Last`. New setup code
    /// should prefer [`EngineBuilder::stage_schedule_mut`].
    pub fn stage_schedule_mut(&mut self, stage: Stage) -> &mut Schedule {
        match stage {
            Stage::PreUpdate => &mut self.pre_update,
            Stage::Input => &mut self.input,
            Stage::Gameplay => &mut self.fixed_schedule,
            Stage::PostFrame => &mut self.schedule,
            Stage::Last => &mut self.last,
        }
    }

    /// Registers one system into a named [`Stage`] plan between frames.
    ///
    /// Legacy escape hatch delegating to
    /// [`Engine::stage_schedule_mut`]; same `PostFrame`-for-variable-tick
    /// routing as the build-phase variant. New setup code should prefer
    /// [`EngineBuilder::add_stage_system`].
    pub fn add_stage_system<S: crate::System + 'static>(
        &mut self,
        stage: Stage,
        system: S,
    ) -> &mut Self {
        assert_writer_stage_matches(stage, system.name(), &system.access());
        self.stage_schedule_mut(stage).add_system(system);
        self
    }

    /// Registers one fixed-rate system between frames.
    ///
    /// Legacy escape hatch; new setup code should prefer
    /// [`EngineBuilder::add_fixed_system`].
    pub fn add_fixed_system<S: crate::System + 'static>(&mut self, system: S) -> &mut Self {
        self.fixed_schedule.add_system(system);
        self
    }

    /// Turns per-system timing on or off for every [`Stage`] schedule
    /// (debug statistics, see [`Schedule::set_timing`]). Off by default.
    pub fn set_stage_timing(&mut self, enabled: bool) {
        for schedule in [
            &mut self.pre_update,
            &mut self.input,
            &mut self.fixed_schedule,
            &mut self.schedule,
            &mut self.last,
        ] {
            schedule.set_timing(enabled);
        }
    }

    /// Drains the per-stage timings accumulated since the previous call,
    /// in execution order; empty while timing is off.
    pub fn take_stage_timings(&self) -> Vec<(Stage, crate::ScheduleTimings)> {
        [
            Stage::PreUpdate,
            Stage::Input,
            Stage::Gameplay,
            Stage::PostFrame,
            Stage::Last,
        ]
        .into_iter()
        .filter_map(|stage| Some((stage, self.stage_schedule(stage).take_timings()?)))
        .collect()
    }

    /// Runs one frame with `delta_seconds` and publishes [`Time`] first.
    ///
    /// Available only on [`Engine<Running>`]: a build-phase engine must be
    /// sealed with [`Engine::build`] (or built via [`EngineBuilder`])
    /// before entering the frame loop.
    ///
    /// Staged order is `PreUpdate → Input → Gameplay(fixed × N) →
    /// PostFrame → Last`. The fixed accumulator is advanced before the
    /// staged schedules run so
    /// the frame-level input snapshot and the [`FixedTime`] step count are
    /// already published when `PreUpdate`/`Input` execute; each selected
    /// fixed update then receives the same snapshot, and the `PostFrame`
    /// (legacy variable) schedule runs once after the fixed loop.
    ///
    /// Two structural boundaries, matching the two schedules:
    /// [`FixedStructuralQueue`](crate::mutation::FixedStructuralQueue) is
    /// drained after **every** fixed tick (so tick `k+1` sees tick `k`'s
    /// `add`/`remove`/`spawn` regardless of how the frame sliced ticks), and
    /// [`FrameStructuralQueue`](crate::mutation::FrameStructuralQueue) is
    /// drained once between `PostFrame` and `Last` so a `Last` system observes
    /// the frame's ops. Input events accumulated
    /// by a platform adapter are visible to all stages during the frame.
    /// Consumers of transient pointer/wheel deltas should normally run in a
    /// once-per-frame stage so a catch-up frame does not apply one event
    /// repeatedly. After all systems finish, transient deltas are cleared
    /// from [`InputState`]; held keys/buttons persist.
    ///
    /// `Time`/`FixedTime` publishing and `clear_frame_transients` are an
    /// intentional frame boundary, not a scheduled system: system bodies
    /// receive only shared `&Resources` (see [`crate::System::run`]) while
    /// [`crate::Resources::get_mut`] needs `&mut` (and [`World::run`]
    /// holds only `&self`), so a system cannot mutate the plain `Copy`
    /// clocks no matter what it declares — `writes::<Time>()` only orders
    /// levels and arms enforcement, it grants no `&mut`. Moving the
    /// boundary inside would require a `World`/`Resources` API change
    /// (`Mutex`-wrapped clocks or `&mut` execution, breaking the parallel
    /// level executor and every system impl). The host loop must bracket
    /// the schedules anyway: `begin_frame` before the stages, one step
    /// advance between fixed iterations, transients cleared after
    /// `Last`. The structural applies are the same kind of exclusive
    /// boundary step, taking `&mut` to the store exactly at those points.
    ///
    /// Ops queued inside `Last` are applied by the *next* frame's apply;
    /// queue frame-final structural work from `PostFrame`.
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
        let Some(fixed_steps) = self
            .world
            .resources_mut()
            .get_mut::<FixedTime>()
            .map(|fixed| fixed.begin_frame(delta_seconds))
        else {
            return;
        };

        self.world.run(&self.pre_update);
        self.world.run(&self.input);
        for _ in 0..fixed_steps {
            if let Some(fixed) = self.world.resources_mut().get_mut::<FixedTime>() {
                fixed.start_step();
            }
            self.world.run(&self.fixed_schedule);
            // Fixed boundary: tick k+1 must see tick k's structural ops, so
            // the outcome cannot depend on how the frame sliced its ticks.
            crate::mutation::apply_fixed_structural_queue(self.world.resources_mut());
        }
        self.world.run(&self.schedule);

        crate::mutation::apply_frame_structural_queue(self.world.resources_mut());
        self.world.run(&self.last);

        if let Some(input) = self.world.resources_mut().get_mut::<InputState>() {
            input.clear_frame_transients();
        }
    }

    /// Runs one frame with a [`Seconds`] delta (same contract as the `f32`
    /// path: finite and non-negative).
    pub fn run_frame_secs(&mut self, delta: Seconds) {
        self.run_frame(delta.get());
    }

    /// Advances exactly one fixed tick and returns the tick index.
    ///
    /// This is the replay/rollback entry point (`docs/networking-design.md`
    /// §6.1): it runs the [`Stage::Gameplay`] plan once and then drains the
    /// fixed structural queue, so the sequence of worlds a recorded session
    /// and a replayed one walk through is identical. Without the per-tick
    /// apply here, `#[smart_system]` `add`/`remove`/`spawn` from the fixed
    /// plan would never land in a replay and the queue would only grow.
    ///
    /// It does *not* advance frame [`Time`], run the once-per-frame stages, or
    /// touch [frame transients](InputState::clear_frame_transients) — those
    /// belong to the host frame. It also leaves the frame accumulator alone,
    /// so a host must not mix it with [`Engine::run_frame`] on one engine
    /// (the same contract the networking design states).
    ///
    /// Returns the tick index that ran.
    // qual:api — replay/rollback entry point; hosts are the callers, the in-crate tests pin its per-tick queue apply.
    pub fn run_fixed_step(&mut self) -> Tick {
        if self.world.resources().get::<FixedTime>().is_none() {
            let _ = self.world.insert(FixedTime::default());
        }
        if let Some(fixed) = self.world.resources_mut().get_mut::<FixedTime>() {
            fixed.start_replay_step();
        }
        self.world.run(&self.fixed_schedule);
        crate::mutation::apply_fixed_structural_queue(self.world.resources_mut());
        let tick = self
            .world
            .resources()
            .get::<FixedTime>()
            .map_or(0, |fixed| fixed.tick());
        Tick::new(tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutation::StructuralQueue as _;
    use crate::{InputState, KeyCode, Resources, SmartStore, System, SystemAccess};
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

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct CapturedInput {
        key_down: bool,
        pointer_position: [f32; 2],
        pointer_delta: [f32; 2],
        wheel_delta: f32,
    }

    struct CaptureInput(Arc<Mutex<Vec<CapturedInput>>>);

    impl System for CaptureInput {
        fn name(&self) -> &'static str {
            "capture_input"
        }

        fn access(&self) -> SystemAccess {
            SystemAccess::new().reads::<InputState>()
        }

        fn run(&self, resources: &Resources) {
            let input = resources.get::<InputState>().expect("input resource");
            self.0.lock().expect("capture lock").push(CapturedInput {
                key_down: input.keycode_down(KeyCode::KeyW),
                pointer_position: input.pointer_position(),
                pointer_delta: input.pointer_delta(),
                wheel_delta: input.wheel_delta(),
            });
        }
    }

    #[test]
    fn engine_publishes_input_to_systems_before_clearing_deltas() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        engine
            .schedule_mut()
            .add_system(CaptureInput(captured.clone()));
        {
            let input = engine
                .world_mut()
                .resources_mut()
                .get_mut::<InputState>()
                .expect("engine publishes input");
            input.set_keycode(KeyCode::KeyW, true);
            input.set_pointer_position([4.0, 5.0]);
            input.add_wheel_delta(-1.0);
        }

        engine.run_frame(1.0 / 60.0);

        assert_eq!(
            captured.lock().expect("capture lock").as_slice(),
            &[CapturedInput {
                key_down: true,
                pointer_position: [4.0, 5.0],
                pointer_delta: [4.0, 5.0],
                wheel_delta: -1.0,
            }]
        );
        let input = engine
            .world()
            .resources()
            .get::<InputState>()
            .expect("input resource");
        assert_eq!(input.pointer_delta(), [0.0, 0.0]);
        assert_eq!(input.wheel_delta(), 0.0);
        // Named write lands on the legacy wire codes readers still use.
        assert!(input.keycode_down(KeyCode::KeyW));
        assert!(input.key_down(17));
        assert!(input.key_down(87));
    }

    #[test]
    fn stage_timing_reports_post_frame_systems() {
        let mut engine = Engine::new();
        engine
            .schedule_mut()
            .add_system(CaptureTime(Arc::new(Mutex::new(Vec::new()))));
        assert!(engine.take_stage_timings().is_empty(), "off by default");
        engine.set_stage_timing(true);
        engine.run_frame(0.25);
        let timings = engine.take_stage_timings();
        let stages: Vec<_> = timings.iter().map(|(stage, _)| *stage).collect();
        assert_eq!(
            stages,
            [
                Stage::PreUpdate,
                Stage::Input,
                Stage::Gameplay,
                Stage::PostFrame,
                Stage::Last
            ]
        );
        assert_eq!(timings[3].1.systems[0].0, "capture_time");
        assert!(
            timings[4].1.systems.is_empty(),
            "no Last systems registered"
        );
        engine.set_stage_timing(false);
        assert!(engine.take_stage_timings().is_empty());
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
        assert!(engine.stage_schedule(Stage::PreUpdate).is_empty());
        assert!(engine.stage_schedule(Stage::Input).is_empty());
        assert!(engine.stage_schedule(Stage::Gameplay).is_empty());
        assert!(engine.stage_schedule(Stage::PostFrame).is_empty());
        assert!(engine.stage_schedule(Stage::Last).is_empty());
    }

    #[test]
    fn stage_labels_and_fixed_flag_are_stable() {
        assert_eq!(Stage::PreUpdate.name(), "pre_update");
        assert_eq!(Stage::Input.name(), "input");
        assert_eq!(Stage::Gameplay.name(), "gameplay");
        assert_eq!(Stage::PostFrame.name(), "post_frame");
        assert_eq!(Stage::Last.name(), "last");
        assert!(!Stage::PreUpdate.is_fixed());
        assert!(!Stage::Input.is_fixed());
        assert!(Stage::Gameplay.is_fixed());
        assert!(!Stage::PostFrame.is_fixed());
        assert!(!Stage::Last.is_fixed());
    }

    #[test]
    fn legacy_schedules_delegate_to_staged_plans() {
        let mut engine = Engine::new();
        engine.schedule_mut().add_system(Trace { fixed: false });
        engine
            .fixed_schedule_mut()
            .add_system(Trace { fixed: true });
        assert_eq!(engine.stage_schedule(Stage::PostFrame).len(), 1);
        assert_eq!(engine.stage_schedule(Stage::Gameplay).len(), 1);
        assert!(engine.stage_schedule(Stage::PreUpdate).is_empty());
        assert!(engine.stage_schedule(Stage::Input).is_empty());
    }

    #[test]
    fn staged_plans_run_pre_input_gameplay_post_in_order() {
        struct Tag(&'static str);
        impl System for Tag {
            fn name(&self) -> &'static str {
                self.0
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new().reads::<TraceLog>()
            }
            fn run(&self, resources: &Resources) {
                let log = resources.get::<TraceLog>().expect("trace log");
                log.0.lock().expect("trace lock").push(self.0);
            }
        }
        let trace = Arc::new(Mutex::new(Vec::new()));
        let mut engine = Engine::new();
        let _ = engine.world_mut().insert(TraceLog(trace.clone()));
        engine.add_stage_system(Stage::PreUpdate, Tag("pre_update"));
        engine.add_stage_system(Stage::Input, Tag("input"));
        engine.add_stage_system(Stage::Gameplay, Tag("gameplay"));
        engine.add_stage_system(Stage::PostFrame, Tag("post_frame"));
        engine.add_stage_system(Stage::Last, Tag("last"));

        let delta = FixedTime::default().delta_seconds();
        engine.run_frame(delta);
        assert_eq!(
            *trace.lock().expect("trace lock"),
            vec!["pre_update", "input", "gameplay", "post_frame", "last"]
        );
    }

    /// The deferred structural queue applies between `PostFrame` and
    /// `Last`, so a `Last` system observes this frame's structural ops.
    #[test]
    fn structural_queue_applies_before_last_stage_systems() {
        /// Queues one structural op (declares the queue write).
        struct QueueInsert;
        impl System for QueueInsert {
            fn name(&self) -> &'static str {
                "queue_insert"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new()
                    .reads::<SmartStore>()
                    .writes::<crate::mutation::FrameStructuralQueue>()
            }
            fn run(&self, resources: &Resources) {
                let queue = resources
                    .get::<crate::mutation::FrameStructuralQueue>()
                    .expect("queue registered by World");
                let store = resources.get::<SmartStore>().expect("store");
                let entity = store.create_entity();
                crate::mutation::StructuralQueue::enqueue(
                    queue,
                    Box::new(move |store: &mut SmartStore| {
                        store.insert(entity, 7_u32);
                    }),
                );
            }
        }
        /// Observes the applied component inside the `Last` stage.
        struct ExpectApplied(Arc<Mutex<Option<Option<u32>>>>);
        impl System for ExpectApplied {
            fn name(&self) -> &'static str {
                "expect_applied"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new()
                    .reads::<SmartStore>()
                    .reads_lane::<u32>()
            }
            fn run(&self, resources: &Resources) {
                let store = resources.get::<SmartStore>().expect("store");
                let value = store.read_lane::<u32>().and_then(|lane| {
                    let entity = lane.entities.first().copied()?;
                    lane.get(entity).copied()
                });
                *self.0.lock().expect("probe lock") = Some(value);
            }
        }

        let probe = Arc::new(Mutex::new(None));
        let mut engine = Engine::new();
        engine.schedule_mut().add_system(QueueInsert);
        engine.add_stage_system(Stage::Last, ExpectApplied(probe.clone()));

        engine.run_frame(1.0 / 60.0);

        assert_eq!(*probe.lock().expect("probe lock"), Some(Some(7)));
        let store = engine.world().store().expect("store");
        assert_eq!(store.read_lane::<u32>().map(|lane| lane.len()), Some(1));
    }

    /// Ops queued inside `Last` cannot be applied by the same frame (the one
    /// apply already ran); they land at the next frame's apply, in FIFO
    /// order — the documented placement rule for frame-final structural work.
    #[test]
    fn ops_queued_in_last_stage_apply_on_the_next_frame() {
        /// Queues a lane write with no observable side effect of its own.
        struct QueueFromLast(Arc<Mutex<usize>>);
        impl System for QueueFromLast {
            fn name(&self) -> &'static str {
                "queue_from_last"
            }
            fn access(&self) -> SystemAccess {
                SystemAccess::new()
                    .reads::<SmartStore>()
                    .writes::<crate::mutation::FrameStructuralQueue>()
            }
            fn run(&self, resources: &Resources) {
                let queue = resources
                    .get::<crate::mutation::FrameStructuralQueue>()
                    .expect("queue registered by World");
                let store = resources.get::<SmartStore>().expect("store");
                let entity = store.create_entity();
                crate::mutation::StructuralQueue::enqueue(
                    queue,
                    Box::new(move |store: &mut SmartStore| {
                        store.insert(entity, 7_u32);
                    }),
                );
                *self.0.lock().expect("count lock") += 1;
            }
        }

        let runs = Arc::new(Mutex::new(0));
        let mut engine = Engine::new();
        engine.add_stage_system(Stage::Last, QueueFromLast(runs.clone()));

        engine.run_frame(1.0 / 60.0);
        assert_eq!(
            engine
                .world()
                .store()
                .expect("store")
                .read_lane::<u32>()
                .map(|lane| lane.len()),
            None,
            "an op queued in Last waits for the next frame's apply"
        );
        assert_eq!(
            engine
                .world()
                .resources()
                .get::<crate::mutation::FrameStructuralQueue>()
                .expect("queue registered")
                .len(),
            1
        );

        engine.run_frame(1.0 / 60.0);
        assert_eq!(
            engine
                .world()
                .store()
                .expect("store")
                .read_lane::<u32>()
                .map(|lane| lane.len()),
            // Frame 1 applied the op queued in frame 0; frame 2's op is
            // still pending, so exactly one entry landed.
            Some(1),
            "the pending op applies at the next frame boundary"
        );
        assert_eq!(*runs.lock().expect("count lock"), 2);
    }

    #[test]
    fn empty_stages_preserve_legacy_frame_semantics() {
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

    #[test]
    fn secs_newtypes_mirror_raw_accessors() {
        let mut time = Time::new();
        time.advance_secs(Seconds::new(0.25));
        assert_eq!(time.delta(), Seconds::new(0.25));
        assert_eq!(time.frame_id(), Frame::new(1));

        let fixed = FixedTime::new_secs(Seconds::new(1.0 / 60.0), FixedSteps::new(4));
        assert_eq!(fixed.delta(), Seconds::new(1.0 / 60.0));
        assert_eq!(fixed.tick_id(), Tick::ZERO);
        assert_eq!(fixed.steps(), FixedSteps::ZERO);
        assert_eq!(fixed.max_steps_per_frame(), 4);
    }

    #[test]
    fn run_frame_secs_matches_raw_delta() {
        let mut engine = Engine::new();
        engine.run_frame_secs(Seconds::new(0.5));
        let time = engine
            .world()
            .resources()
            .get::<Time>()
            .expect("engine publishes Time");
        assert_eq!(time.frame_id(), Frame::new(1));
        assert_eq!(time.delta(), Seconds::new(0.5));
    }

    #[test]
    fn building_engine_seals_into_running() {
        let mut building = Engine::new_building();
        let _ = building
            .world_mut()
            .insert(TraceLog(Arc::new(Mutex::new(Vec::new()))));
        building.schedule_mut().add_system(Trace { fixed: false });
        let mut running = building.build();
        running.run_frame(FixedTime::default().delta_seconds());
        assert_eq!(running.schedule().len(), 1);
    }

    #[test]
    fn builder_registers_systems_then_seals() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut builder = EngineBuilder::new();
        builder.add_stage_system(Stage::PostFrame, CaptureTime(seen.clone()));
        builder
            .fixed_schedule_mut()
            .add_system(CaptureFixedTime(Arc::new(Mutex::new(Vec::new()))));
        let _ = builder.world_mut().insert(7_u32);
        let mut engine = builder.build();
        engine.run_frame(FixedTime::default().delta_seconds());
        let seen = seen.lock().expect("capture lock");
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].frame(), 1);
        assert_eq!(engine.world().resources().get::<u32>(), Some(&7));
    }

    #[test]
    fn builder_configures_fixed_clock_then_seals() {
        let mut engine = EngineBuilder::new()
            .fixed_delta(Seconds::new(1.0 / 30.0))
            .max_steps(FixedSteps::new(4))
            .build();
        let fixed = engine
            .world()
            .resources()
            .get::<FixedTime>()
            .expect("engine publishes FixedTime");
        assert_eq!(fixed.delta(), Seconds::new(1.0 / 30.0));
        assert_eq!(fixed.max_steps_per_frame(), 4);
        engine.run_frame(1.0 / 30.0);
        let fixed = engine
            .world()
            .resources()
            .get::<FixedTime>()
            .expect("engine publishes FixedTime");
        assert_eq!(fixed.tick_id(), Tick::new(1));
    }
}
