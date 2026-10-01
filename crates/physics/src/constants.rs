//! Shared numerical thresholds used across solvers.
//!
//! These are domain floors, not idiomatic factors: a bare `1e-10` in a
//! contact row and the same literal in the wide path mean the same gate
//! and must stay coupled. GPU/`#[kernel]` bodies keep literals — the WGSL
//! DSL has no spelling for a Rust `const` (see `gpu/avbd.rs`).

/// Effective-mass floor for sequential-impulse constraint rows.
///
/// Below this, `k_eff` / tangential / rolling masses are treated as inert
/// (skip the impulse, or emit a zero inverse). Shared by the island path
/// ([`crate::sequential_impulse::contacts`]) and the SIMD wide path
/// ([`crate::wide`]) so both gates stay bit-identical.
pub(crate) const MIN_EFFECTIVE_MASS: f32 = 1e-10;

/// Squared length below which a vector/axis is treated as degenerate.
///
/// Used for joint-axis admission, friction-cone length guards and similar
/// “direction has collapsed” checks. Distinct from [`MIN_EFFECTIVE_MASS`]
/// (mass domain) and from AVBD's `1e-9` near-zero used for TOI/spin.
pub(crate) const DEGENERATE_LEN2: f32 = 1e-12;

/// Absolute near-zero for soft/XPBD constraint residuals and distances.
///
/// Softer than [`DEGENERATE_LEN2`]: used when a rest length or separation
/// is compared directly (not squared) before a divide.
pub(crate) const NEAR_ZERO: f32 = 1e-9;

/// Minimum contact-normal alignment for warm-start feature matching.
///
/// Cached manifold points whose normal dots the live normal below this
/// are treated as a different feature (rolling over an edge) and do not
/// inherit warm impulses.
pub(crate) const FEATURE_NORMAL_DOT_MIN: f32 = 0.7;

/// Tetrahedron volume factor: `V = a·(b×c) / 6` (origin-based tet).
///
/// Shared by convex-hull inertia (Mirtich) and soft-body volume rows so
/// the same surface triangulation cannot disagree on signed volume.
pub(crate) const TET_VOLUME_DIVISOR: f32 = 6.0;

/// Gap (m) at which shapes count as touching for conservative advancement
/// and witness refine.
///
/// Shared by analytic `cast_shape` and the sequential-impulse kinematic
/// cast so both CA loops agree on the touch band.
pub(crate) const SHAPE_TOUCH: f32 = 1e-3;

/// Squared length treated as numerically coincident / collapsed dust.
///
/// Tighter than [`DEGENERATE_LEN2`]: used when an O(1)-meter feature
/// would have to live inside f32 rounding (coincident clamp, spin deadband,
/// zero-area EPA faces). Not a mass-domain floor.
pub(crate) const COINCIDENT_LEN2: f32 = 1e-18;

/// Squared length at which an angular/axis residual is treated as rest.
///
/// Between [`DEGENERATE_LEN2`] and [`COINCIDENT_LEN2`]: small-angle XPBD
/// locks and AVBD contact normals that have collapsed but are not yet
/// f32 dust. Keeps those gates coupled across solvers.
pub(crate) const AXIS_REST_LEN2: f32 = 1e-16;

/// Position / angle correction deadband (m or rad).
///
/// Joint NGS rows and gyroscopic-spread ratios below this skip the
/// impulse — residues smaller than solver slop.
pub(crate) const POS_CORRECTION_EPS: f32 = 1e-6;

/// Fraction of the thinnest shape feature that arms CCD / TOI casts.
///
/// Linear (and angular) sweeps shorter than this fraction of
/// `shape_min_dimension` cannot defeat the discrete phase, so both the
/// sequential-impulse and AVBD continuous paths skip them together.
pub(crate) const CCD_TRAVEL_GATE_FRACTION: f32 = 0.5;
