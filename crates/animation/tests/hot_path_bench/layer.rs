//! Overlay-layer shim for the frame harness (`../hot_path_bench.rs`).
//!
//! The layer install lives in this one tiny file so `hot_path_ab.sh` can
//! swap in the pre-stack variant when it copies the harness onto a base
//! that predates [`SkelLayerStack`]: only this shim is era-specific, the
//! harness itself compiles against any base that carries `SkelLayer`.

use ornis_animation::{MaskId, SkelLayer, SkelLayerStack, SkelPlayer};
use ornis_core::{Entity, SmartStore};

/// Registers the overlay layer stack lane.
pub fn register(store: &mut SmartStore) {
    store.register::<SkelLayerStack>();
}

/// Installs one overlay layer on a root.
///
/// A stack of one keeps the A/B workload of the single-layer era, so the
/// scene numbers stay comparable across the stack change.
pub fn install(store: &mut SmartStore, root: Entity, player: SkelPlayer, mask: MaskId) {
    store.insert(
        root,
        SkelLayerStack {
            layers: vec![SkelLayer { player, mask }],
        },
    );
}
