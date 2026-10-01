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
