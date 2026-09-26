//! Typed replacements for `bool` flags across the physics pipeline.
//!
//! Each flag previously carried its meaning in the parameter name
//! (`avbd: bool`, `tick: bool`, `slow_only: bool`, ...), forcing every
//! call site to re-derive the polarity. The enums below make the polarity
//! explicit at the type level while keeping `From<bool>` conversions so
//! existing tests and call sites migrate incrementally.

/// Which solver side a split-local handle belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverSide {
    /// AVBD engine side.
    Avbd,
    /// Sequential-impulse engine side.
    SequentialImpulse,
}

impl SolverSide {
    /// `true` for the AVBD side.
    pub fn is_avbd(self) -> bool {
        matches!(self, Self::Avbd)
    }
}

impl From<bool> for SolverSide {
    /// `true` maps to [`SolverSide::Avbd`] (legacy `avbd: bool` polarity).
    fn from(avbd: bool) -> Self {
        if avbd {
            Self::Avbd
        } else {
            Self::SequentialImpulse
        }
    }
}

impl From<SolverSide> for bool {
    /// Inverse of the legacy `avbd: bool` polarity.
    fn from(side: SolverSide) -> bool {
        side.is_avbd()
    }
}

/// Routing phase for [`crate::split`](crate::split)-style ownership passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoutePhase {
    /// Cold probe (no sleep integration, e.g. rebuilds).
    Probe,
    /// Per-tick routing with sleep-state integration.
    Tick,
}

impl RoutePhase {
    /// `true` for [`RoutePhase::Tick`].
    pub fn is_tick(self) -> bool {
        matches!(self, Self::Tick)
    }
}

impl From<bool> for RoutePhase {
    /// Legacy `tick: bool` polarity.
    fn from(tick: bool) -> Self {
        if tick { Self::Tick } else { Self::Probe }
    }
}

impl From<RoutePhase> for bool {
    /// Legacy `tick: bool` polarity.
    fn from(phase: RoutePhase) -> bool {
        phase.is_tick()
    }
}

/// SAT-cache policy for box-box narrowphase queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CachePolicy {
    /// Consult and populate the SAT cache (slow-path queries only).
    Cached,
    /// Recompute unconditionally.
    Recompute,
}

impl CachePolicy {
    /// `true` for [`CachePolicy::Cached`].
    pub fn use_cache(self) -> bool {
        matches!(self, Self::Cached)
    }
}

impl From<bool> for CachePolicy {
    /// Legacy `slow_only: bool` polarity (`true` = cached).
    fn from(slow_only: bool) -> Self {
        if slow_only {
            Self::Cached
        } else {
            Self::Recompute
        }
    }
}

/// Whether the first-substep restitution bias may fire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RestitutionGate {
    /// Restitution bias enabled (first substep, new points only).
    Enabled,
    /// Restitution suppressed.
    Suppressed,
}

impl RestitutionGate {
    /// `true` for [`RestitutionGate::Enabled`].
    pub fn is_enabled(self) -> bool {
        matches!(self, Self::Enabled)
    }
}

impl From<bool> for RestitutionGate {
    /// Legacy `allow_restitution: bool` polarity.
    fn from(allow: bool) -> Self {
        if allow {
            Self::Enabled
        } else {
            Self::Suppressed
        }
    }
}

impl From<RestitutionGate> for bool {
    /// Legacy `allow_restitution: bool` polarity.
    fn from(gate: RestitutionGate) -> bool {
        gate.is_enabled()
    }
}

/// Contact-solver execution path for single-point manifolds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolvePath {
    /// Scalar Gauss-Seidel path (bit-exact baseline).
    Scalar,
    /// SIMD-wide batches for single-point manifolds.
    Wide,
}

impl SolvePath {
    /// `true` for [`SolvePath::Wide`].
    pub fn use_wide(self) -> bool {
        matches!(self, Self::Wide)
    }
}

impl From<bool> for SolvePath {
    /// Legacy `use_wide` / `wide_solver: bool` polarity.
    fn from(wide: bool) -> Self {
        if wide { Self::Wide } else { Self::Scalar }
    }
}

impl From<SolvePath> for bool {
    /// Legacy `use_wide` polarity.
    fn from(path: SolvePath) -> bool {
        path.use_wide()
    }
}

/// Collision role of a body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BodyRole {
    /// Solid body: participates in contact solving.
    Solid,
    /// Trigger volume: reports overlaps, never applies impulses.
    Trigger,
}

impl BodyRole {
    /// `true` for [`BodyRole::Trigger`].
    pub fn is_trigger(self) -> bool {
        matches!(self, Self::Trigger)
    }
}

impl From<bool> for BodyRole {
    /// Legacy `is_trigger: bool` polarity.
    fn from(is_trigger: bool) -> Self {
        if is_trigger {
            Self::Trigger
        } else {
            Self::Solid
        }
    }
}

impl From<BodyRole> for bool {
    /// Legacy `is_trigger: bool` polarity.
    fn from(role: BodyRole) -> bool {
        role.is_trigger()
    }
}

/// Joint-axis health after resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisStatus {
    /// Axes were well-formed.
    Intact,
    /// At least one axis needed its fallback value.
    Degenerate,
}

impl AxisStatus {
    /// `true` for [`AxisStatus::Degenerate`].
    pub fn is_degenerate(self) -> bool {
        matches!(self, Self::Degenerate)
    }
}

impl From<bool> for AxisStatus {
    /// Legacy `degenerate: bool` polarity.
    fn from(degenerate: bool) -> Self {
        if degenerate {
            Self::Degenerate
        } else {
            Self::Intact
        }
    }
}

impl From<AxisStatus> for bool {
    /// Legacy `degenerate: bool` polarity.
    fn from(status: AxisStatus) -> bool {
        status.is_degenerate()
    }
}

/// Origin of a continuous-collision hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HitKind {
    /// Linear sweep hit.
    Linear,
    /// Angular sweep hit.
    Angular,
}

impl HitKind {
    /// `true` for [`HitKind::Angular`].
    pub fn is_angular(self) -> bool {
        matches!(self, Self::Angular)
    }
}

impl From<bool> for HitKind {
    /// Legacy `angular: bool` polarity.
    fn from(angular: bool) -> Self {
        if angular { Self::Angular } else { Self::Linear }
    }
}

impl From<HitKind> for bool {
    /// Legacy `angular: bool` polarity.
    fn from(kind: HitKind) -> bool {
        kind.is_angular()
    }
}

/// Rolling-resistance accumulator slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RollAxis {
    /// Roll about the first tangent.
    RollU,
    /// Roll about the second tangent.
    RollV,
    /// Spin about the contact normal.
    Spin,
}

impl RollAxis {
    /// Legacy `slot: u8` encoding (0/1/2).
    pub fn as_slot(self) -> u8 {
        match self {
            Self::RollU => 0,
            Self::RollV => 1,
            Self::Spin => 2,
        }
    }

    /// Decodes the legacy `slot: u8` encoding; unknown slots map to spin.
    pub fn from_slot(slot: u8) -> Self {
        match slot {
            0 => Self::RollU,
            1 => Self::RollV,
            _ => Self::Spin,
        }
    }
}

/// Gear-coordinate periodicity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoordKind {
    /// Linear coordinate (never periodic).
    Linear,
    /// Angular coordinate (unwrapped against prior memory).
    Angular,
}

impl CoordKind {
    /// `true` for [`CoordKind::Angular`].
    pub fn is_angular(self) -> bool {
        matches!(self, Self::Angular)
    }
}

impl From<bool> for CoordKind {
    /// Legacy `angular: bool` polarity.
    fn from(angular: bool) -> Self {
        if angular { Self::Angular } else { Self::Linear }
    }
}

/// Which side of a two-sided distance query owns `point_a`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Order {
    /// The terrain/mesh side owns `point_a` (`hf_first`/`mesh_first`).
    First,
    /// The convex side owns `point_a`.
    Second,
}

impl Order {
    /// `true` for [`Order::First`].
    pub fn is_first(self) -> bool {
        matches!(self, Self::First)
    }
}

impl From<bool> for Order {
    /// Legacy `hf_first` / `mesh_first: bool` polarity.
    fn from(first: bool) -> Self {
        if first { Self::First } else { Self::Second }
    }
}

impl From<Order> for bool {
    /// Legacy first-side polarity.
    fn from(order: Order) -> bool {
        order.is_first()
    }
}

/// Island-dispatch execution mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dispatch {
    /// Sequential execution in island order.
    Sequential,
    /// Parallel execution through one scheduler level.
    Parallel,
}

impl Dispatch {
    /// `true` for [`Dispatch::Parallel`].
    pub fn is_parallel(self) -> bool {
        matches!(self, Self::Parallel)
    }
}

impl From<bool> for Dispatch {
    /// Legacy `parallel: bool` polarity.
    fn from(parallel: bool) -> Self {
        if parallel {
            Self::Parallel
        } else {
            Self::Sequential
        }
    }
}

impl From<Dispatch> for bool {
    /// Legacy `parallel: bool` polarity.
    fn from(mode: Dispatch) -> bool {
        mode.is_parallel()
    }
}

/// Structural state of the solver orchestrator registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StructuralState {
    /// Registry matches the live solvers.
    Clean,
    /// Adds/removes are pending a rebuild.
    Dirty,
}

impl StructuralState {
    /// `true` for [`StructuralState::Dirty`].
    pub fn is_dirty(self) -> bool {
        matches!(self, Self::Dirty)
    }
}

impl From<bool> for StructuralState {
    /// Legacy `structural_dirty: bool` polarity.
    fn from(dirty: bool) -> Self {
        if dirty { Self::Dirty } else { Self::Clean }
    }
}

impl From<StructuralState> for bool {
    /// Legacy `structural_dirty: bool` polarity.
    fn from(state: StructuralState) -> bool {
        state.is_dirty()
    }
}

/// Which travel bound a hinge violates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LimitSide {
    /// Lower bound violated.
    Lower,
    /// Upper bound violated.
    Upper,
}

impl LimitSide {
    /// `true` for [`LimitSide::Lower`] (legacy `Some(true)` polarity).
    pub fn is_lower(self) -> bool {
        matches!(self, Self::Lower)
    }
}

impl From<bool> for LimitSide {
    /// Legacy `Some(true) = lower` polarity.
    fn from(lower: bool) -> Self {
        if lower { Self::Lower } else { Self::Upper }
    }
}

impl From<LimitSide> for bool {
    /// Legacy lower-is-true polarity.
    fn from(side: LimitSide) -> bool {
        side.is_lower()
    }
}
