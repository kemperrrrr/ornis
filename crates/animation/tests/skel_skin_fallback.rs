//! `skel_skin_cpu` in GPU-fallback-only mode
//! ([`ornis_animation::SkelSkinGpuFallbackSystem`]): meshes whose palette fits the
//! GPU skinning path are left to the renderer and never CPU-skinned.

use glam::Mat4;
use ornis_animation::{JointPose, SkelSkinStats, Skeleton, SkinnedMesh, run_skel_skin_with};
use ornis_core::{Resources, SmartStore, Time};

/// Pre-run sentinel: survives only if the pass leaves the mesh untouched.
const SENTINEL: [f32; 3] = [9.0, 9.0, 9.0];

#[test]
fn gpu_fallback_only_skips_gpu_eligible_meshes() {
    // A one-joint palette fits the GPU path: the fallback-only pass
    // leaves the mesh untouched and counts neither skin nor defect.
    let mut store = SmartStore::new();
    store.register::<Skeleton>();
    store.register::<JointPose>();
    store.register::<SkinnedMesh>();
    let root = store.create_entity();
    store.insert(
        root,
        Skeleton::new(vec![None], vec![Mat4::IDENTITY], vec!["root".to_string()]),
    );
    store.insert(root, JointPose::identity(1));
    let mesh = store.create_entity();
    let mut placed = SkinnedMesh::new(
        root,
        vec![[0, 0, 0, 0]],
        vec![[1.0, 0.0, 0.0, 0.0]],
        vec![[0.0, 1.0, 0.0]],
        vec![[0.0, 0.0, 1.0]],
        vec![[0.5, 0.5]],
        vec![0, 0, 0],
    );
    placed.skinned_positions = vec![SENTINEL];
    store.insert(mesh, placed);

    let mut resources = Resources::new();
    resources.insert(store);
    resources.insert(Time::new());
    let stats = run_skel_skin_with(&resources, true);
    assert_eq!(stats, SkelSkinStats::default());
    let store = resources.get::<SmartStore>().expect("store resource kept");
    let meshes = store.read_lane::<SkinnedMesh>().expect("lane kept");
    assert_eq!(
        meshes.get(mesh).expect("mesh kept").skinned_positions,
        vec![SENTINEL]
    );
}
