//! Narrowphase caches: the substep manifold cache, the SAT separating-axis
//! cache and the pooled scheduler shard buffers.

use std::sync::Mutex;

use dashmap::DashMap;
use glam::{Quat, Vec3};
use rustc_hash::FxBuildHasher;

use crate::body::RigidBody;
use crate::engine::Manifold;

/// Pooled narrowphase shard buffers for scheduler dispatch: one uncontended
/// `Mutex<Vec>` per shard plus the cached single-group level plan. Owned by
/// the engine and passed down like the other scratch buffers, so the hot
/// loop never allocates. Resized only when the shard count moves.
#[derive(Debug, Default)]
pub struct NarrowShardPool {
    pub(crate) bufs: Vec<Mutex<Vec<Manifold>>>,
    pub(crate) level: Vec<Vec<usize>>,
}

impl NarrowShardPool {
    pub(crate) fn ensure(&mut self, shards: usize) {
        if self.bufs.len() != shards {
            self.bufs.clear();
            for _ in 0..shards {
                self.bufs.push(Mutex::new(Vec::new()));
            }
            self.level = vec![(0..shards).collect()];
        }
        for b in &self.bufs {
            b.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
    }
}

/// Narrowphase cache hit test: positions within 0.1mm, rotations within ~0.06°, margin within 0.1mm.
pub(crate) fn narrow_cache_hit(
    entry: &NarrowCacheEntry,
    a: &RigidBody,
    b: &RigidBody,
    margin: f32,
) -> bool {
    const POS_EPS_SQ: f32 = 1e-8; // (1e-4)^2
    const ROT_DOT_MIN: f32 = 0.999_999;
    const MARGIN_EPS: f32 = 1e-4;
    if (entry.margin - margin).abs() > MARGIN_EPS {
        return false;
    }
    if (entry.pos_a - a.position).length_squared() > POS_EPS_SQ {
        return false;
    }
    if (entry.pos_b - b.position).length_squared() > POS_EPS_SQ {
        return false;
    }
    if entry.rot_a.dot(a.orientation).abs() < ROT_DOT_MIN {
        return false;
    }
    if entry.rot_b.dot(b.orientation).abs() < ROT_DOT_MIN {
        return false;
    }
    true
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn sat_cache_hit(
    entry: &SatCacheEntry,
    pos_a: Vec3,
    half_a: Vec3,
    rot_a: Quat,
    pos_b: Vec3,
    half_b: Vec3,
    rot_b: Quat,
    margin: f32,
) -> bool {
    // Widened tolerances: a hit reuses only the separating axis, while
    // `box_manifold_cached` rebuilds contact points from live geometry, so
    // ~1 mm / ~0.26 deg of pose drift cannot inject stale positions.
    const POS_EPS_SQ: f32 = 1e-6;
    const HALF_EPS_SQ: f32 = 1e-6;
    const ROT_DOT_MIN: f32 = 0.999_99;
    const MARGIN_EPS: f32 = 5e-4;
    if (entry.margin - margin).abs() > MARGIN_EPS {
        return false;
    }
    if (entry.pos_a - pos_a).length_squared() > POS_EPS_SQ {
        return false;
    }
    if (entry.pos_b - pos_b).length_squared() > POS_EPS_SQ {
        return false;
    }
    if (entry.half_a - half_a).length_squared() > HALF_EPS_SQ {
        return false;
    }
    if (entry.half_b - half_b).length_squared() > HALF_EPS_SQ {
        return false;
    }
    if entry.rot_a.dot(rot_a).abs() < ROT_DOT_MIN {
        return false;
    }
    if entry.rot_b.dot(rot_b).abs() < ROT_DOT_MIN {
        return false;
    }
    true
}

/// Cached narrowphase result for one sorted body pair (first-substep only).
#[derive(Clone, Debug)]
pub(crate) struct NarrowCacheEntry {
    pub(crate) pos_a: Vec3,
    pub(crate) pos_b: Vec3,
    pub(crate) rot_a: Quat,
    pub(crate) rot_b: Quat,
    pub(crate) margin: f32,
    pub(crate) manifold: Option<Manifold>,
}

#[derive(Clone, Debug)]
/// Cached separating axis for one sorted body pair (SAT fast path).
pub struct SatCacheEntry {
    pub(crate) pos_a: Vec3,
    pub(crate) half_a: Vec3,
    pub(crate) rot_a: Quat,
    pub(crate) pos_b: Vec3,
    pub(crate) half_b: Vec3,
    pub(crate) rot_b: Quat,
    pub(crate) margin: f32,
    pub(crate) result: Option<(Vec3, f32)>,
}

/// Lock-free SAT axis cache: sorted body pair -> last separating axis.
/// `DashMap` shards internally, so the rayon narrowphase shares it without
/// `try_lock` misses or a parallel bypass. Hits reuse the axis only;
/// `box_manifold_cached` rebuilds contacts from current geometry.
pub type SatCache = DashMap<(usize, usize), SatCacheEntry, FxBuildHasher>;
