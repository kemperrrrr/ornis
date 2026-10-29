//! Era-specific animation-tree install, isolated for `hot_path_ab.sh`.
//!
//! The tree install lives in this one tiny file so `hot_path_ab.sh` can
//! swap in the flat analogue when it copies the harness onto a base that
//! predates [`SkelTree`]: only this shim is era-specific, the harness
//! itself compiles against any base that carries the flat players.

use ornis_animation::{
    MachineCursor, SkelPlayer, SkelTree, SkelTreePlayer, TreeChild, TreeCmp, TreeCond, TreeNodeDef,
    TreeParam, TreeTransition,
};
use ornis_core::{Entity, Seconds, SmartStore};

use super::FADE_SECONDS;

/// Registers the animation-tree lanes.
pub fn register(store: &mut SmartStore) {
    store.register_cold::<SkelTree>();
    store.register::<SkelTreePlayer>();
}

/// Blend tree over two clip leaves driven at `speed` — the tree workload.
pub fn install_blend(store: &mut SmartStore, root: Entity, leaves: [SkelPlayer; 2], speed: f32) {
    {
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip {
                    clip: leaves[0].clip,
                },
                TreeNodeDef::Clip {
                    clip: leaves[1].clip,
                },
                TreeNodeDef::Blend1d {
                    axis: 0,
                    children: vec![
                        TreeChild {
                            position: 0.0,
                            node: 0,
                        },
                        TreeChild {
                            position: 1.0,
                            node: 1,
                        },
                    ],
                },
            ],
            params: vec![TreeParam {
                name: "speed".to_string(),
                default: 0.0,
            }],
        };
        install_tree(store, root, tree, tree_player(leaves, speed, Vec::new()));
    }
}

/// Machine tree whose crossfade never completes — the machine workload.
pub fn install_machine(store: &mut SmartStore, root: Entity, leaves: [SkelPlayer; 2], speed: f32) {
    {
        let tree = SkelTree {
            nodes: vec![
                TreeNodeDef::Clip {
                    clip: leaves[0].clip,
                },
                TreeNodeDef::Clip {
                    clip: leaves[1].clip,
                },
                TreeNodeDef::Machine {
                    states: vec![0, 1],
                    initial: 0,
                    transitions: vec![TreeTransition {
                        from: 0,
                        to: 1,
                        when: TreeCond {
                            param: 0,
                            op: TreeCmp::Greater,
                            value: 0.25,
                        },
                        fade: Seconds::new(FADE_SECONDS),
                    }],
                },
            ],
            params: vec![TreeParam {
                name: "speed".to_string(),
                default: 0.0,
            }],
        };
        let cursor = MachineCursor {
            state: 1,
            from: Some(0),
            elapsed: Seconds::ZERO,
            duration: Seconds::new(FADE_SECONDS),
        };
        install_tree(store, root, tree, tree_player(leaves, speed, vec![cursor]));
    }
}

/// The hot player paired with a tree def: leaves and machine cursors.
fn tree_player(
    leaves: [SkelPlayer; 2],
    speed: f32,
    machines: Vec<MachineCursor>,
) -> SkelTreePlayer {
    {
        // `SkelTreePlayer` is non-exhaustive: build by field writes.
        let mut player = SkelTreePlayer::default();
        player.params = vec![speed];
        player.leaves = Vec::from(leaves);
        player.machines = machines;
        player
    }
}

/// Inserts the tree def and its paired player on `root`.
fn install_tree(store: &mut SmartStore, root: Entity, tree: SkelTree, player: SkelTreePlayer) {
    {
        store.insert_cold(root, tree);
        store.insert(root, player);
    }
}
