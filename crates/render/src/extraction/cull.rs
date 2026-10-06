//! Frustum culling and depth sorting of extracted frame payloads.

use super::FrameUpload;
use crate::camera::Frustum;
use crate::renderer::InstanceData;
use glam::Mat4;
use glam::Vec3;

/// Per-frame cull report for [`cull_frame_upload`]: kept versus dropped
/// entries across both instance lanes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CullStats {
    /// Instances and custom entries surviving the frustum test.
    pub kept: usize,
    /// Instances and custom entries fully outside the frustum.
    pub culled: usize,
}

/// Bounding sphere of one drawn instance: the model translation as the
/// center and the longest model-matrix axis as the radius.
///
/// The shared procedural meshes are unit-bounded with the size baked
/// into the model scale (see [`extract_render_data`]), so
/// `max(scale) * 1.0` is the sphere — a conservative overestimate for
/// non-spherical primitives (never falsely culls, may keep a few extra).
pub fn instance_sphere(instance: &InstanceData) -> (Vec3, f32) {
    let model = &instance.model_matrix;
    let radius = model
        .x_axis
        .length()
        .max(model.y_axis.length())
        .max(model.z_axis.length());
    (model.w_axis.truncate(), radius)
}

/// Opt-in frustum culling of an extracted frame: drops the
/// [`FrameUpload`] entries fully outside the six planes of `view_proj`
/// in place and reports the counts.
///
/// Off by default — [`extract_render_data`] never calls this; the
/// caller applies it after extraction when a frame `view_proj` (see
/// [`crate::camera::camera_view_projection`]) is available. Degenerate
/// planes and non-finite bounds fail open (kept, never culled). Cost is
/// `O(n)` sphere tests over both lanes; measure with
/// [`cull_frame_upload_timed`] when the frame budget needs attributing.
pub fn cull_frame_upload(upload: &mut FrameUpload, view_proj: &Mat4) -> CullStats {
    let frustum = Frustum::from_view_proj(view_proj);
    let before = upload.instances.len() + upload.custom_meshes.len();
    upload.instances.retain(|instance| {
        let (center, radius) = instance_sphere(instance);
        frustum.sphere_visible(center, radius)
    });
    upload.custom_meshes.retain(|entry| {
        let (center, radius) = instance_sphere(&entry.instance);
        frustum.sphere_visible(center, radius)
    });
    let kept = upload.instances.len() + upload.custom_meshes.len();
    CullStats {
        kept,
        culled: before - kept,
    }
}

/// View-space depth of one instance: camera-forward distance of its
/// center (`-z` of the view-transformed translation, so larger means
/// farther from the eye).
pub fn instance_view_depth(instance: &InstanceData, view: &Mat4) -> f32 {
    let world = instance.model_matrix.w_axis;
    -(*view * world).z
}

/// Back-to-front depth sort of an extracted frame, in place.
///
/// Reorders both instance lanes by [`instance_view_depth`] descending
/// (farthest first, stable). Order-only: blending and depth state are
/// untouched. Non-finite depths compare equal (stable, never panics).
/// Cost is `O(n log n)` comparisons over both lanes; measure with
/// [`sort_by_depth_timed`] when the frame budget needs attributing.
///
/// Order contract: the current forward submit is a single instanced draw
/// ([`Renderer3D::render_forward`](crate::renderer::Renderer3D::render_forward))
/// and ignores instance order — this sort is a no-op today, kept for
/// future multi-draw submits where back-to-front order is honored.
pub fn sort_by_depth(upload: &mut FrameUpload, view: &Mat4) {
    upload.instances.sort_by(|a, b| {
        instance_view_depth(b, view)
            .partial_cmp(&instance_view_depth(a, view))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    upload.custom_meshes.sort_by(|a, b| {
        instance_view_depth(&b.instance, view)
            .partial_cmp(&instance_view_depth(&a.instance, view))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// [`cull_frame_upload`] plus its wall time: the [`CullStats`] report and
/// the time spent testing both lanes against the six frustum planes
/// (`O(n)` sphere tests). Use the duration to attribute cull cost inside
/// the frame budget; the culled payload is identical to the untimed call.
pub fn cull_frame_upload_timed(
    upload: &mut FrameUpload,
    view_proj: &Mat4,
) -> (CullStats, std::time::Duration) {
    let started = std::time::Instant::now();
    let stats = cull_frame_upload(upload, view_proj);
    (stats, started.elapsed())
}

/// [`sort_by_depth`] plus its wall time (`O(n log n)` comparisons).
/// Use the duration to attribute sort cost inside the frame budget; the
/// reordered payload is identical to the untimed call.
pub fn sort_by_depth_timed(upload: &mut FrameUpload, view: &Mat4) -> std::time::Duration {
    let started = std::time::Instant::now();
    sort_by_depth(upload, view);
    started.elapsed()
}
#[cfg(test)]
mod tests {
    use super::super::CustomMeshEntry;
    use super::super::FrameUpload;
    use super::super::MeshPose;
    use super::super::extract_render_data;
    use super::super::test_util::test_material;
    use super::super::test_util::test_quad;
    use super::super::test_util::test_sphere;
    use super::*;
    use ornis_animation::SkinningMode;
    use ornis_assets::scene::TransformDesc;
    use ornis_core::Engine;

    #[test]
    fn cull_frame_upload_drops_offscreen_spheres() {
        // 100 spheres: 50 near the origin (visible), 50 at x = 100 (far
        // outside the side plane). Culling is opt-in — extraction keeps
        // all 100, `cull_frame_upload` drops the offscreen half.
        let mut engine = Engine::new();
        for i in 0..100 {
            let x = if i < 50 { i as f32 * 0.05 } else { 100.0 };
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc::from_translation(glam::Vec3::new(x, 0.0, 0.0)),
            );
            store.insert(handle, test_sphere());
            store.insert(handle, test_material());
        }
        let mut extracted = extract_render_data(engine.world().store().expect("store"));
        assert_eq!(extracted.instances.len(), 100, "extraction never culls");

        let view = (
            Vec3::new(0.0, 2.5, 9.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
        );
        let (view_proj, _) = crate::camera::camera_view_projection(view, (160, 90));
        let stats = cull_frame_upload(&mut extracted, &view_proj);
        assert!(stats.culled > 0, "offscreen half culled: {stats:?}");
        assert_eq!(stats.culled, 50, "{stats:?}");
        assert_eq!(stats.kept, 50, "{stats:?}");
        assert_eq!(extracted.instances.len(), 50);
    }

    #[test]
    fn cull_frame_upload_keeps_custom_lane_in_lockstep() {
        // One visible quad plus one offscreen quad: the custom lane is
        // culled by the same sphere test as the shared batch.
        let mut upload = FrameUpload::default();
        let quad = test_quad();
        let (positions, indices) = quad.as_custom().expect("quad is custom");
        let (vertices, soup_indices) =
            crate::mesh_upload::custom_vertices(positions, indices).expect("quad valid");
        for x in [0.0, 100.0] {
            let model = Mat4::from_translation(Vec3::new(x, 0.0, 0.0));
            upload.custom_meshes.push(CustomMeshEntry {
                vertices: vertices.clone(),
                indices: soup_indices.clone(),
                instance: InstanceData {
                    model_matrix: model,
                    normal_matrix: model.inverse().transpose(),
                    material_index: crate::renderer::MaterialIdx::from_raw(0),
                },
                skinning: SkinningMode::Cpu,
                joint_palette: None,
                pose: MeshPose::Skinned,
            });
        }
        let view = (
            Vec3::new(0.0, 2.5, 9.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
        );
        let (view_proj, _) = crate::camera::camera_view_projection(view, (160, 90));
        let stats = cull_frame_upload(&mut upload, &view_proj);
        assert_eq!(stats.kept, 1, "{stats:?}");
        assert_eq!(stats.culled, 1, "{stats:?}");
    }

    #[test]
    fn timed_cull_and_sort_match_untimed_payloads() {
        // The timed wrappers report the same payload as the plain calls
        // plus a wall-time measurement for frame-budget attribution.
        let mut engine = Engine::new();
        for i in 0..10 {
            let store = engine.world_mut().store_mut().expect("store");
            let handle = store.create_entity();
            store.insert(
                handle,
                TransformDesc::from_translation(glam::Vec3::new(i as f32 * 0.05, 0.0, -(i as f32))),
            );
            store.insert(handle, test_sphere());
            store.insert(handle, test_material());
        }
        let view = (
            Vec3::new(0.0, 2.5, 9.0),
            Vec3::ZERO,
            Vec3::Y,
            60.0,
            0.1,
            100.0,
        );
        let (view_proj, _) = crate::camera::camera_view_projection(view, (160, 90));
        let mut timed = extract_render_data(engine.world().store().expect("store"));
        let mut plain = timed.clone();
        let (stats, elapsed) = cull_frame_upload_timed(&mut timed, &view_proj);
        let expected = cull_frame_upload(&mut plain, &view_proj);
        assert_eq!(stats, expected);
        assert_eq!(timed.instances.len(), plain.instances.len());
        let _ = elapsed;

        let mut timed_sort = extract_render_data(engine.world().store().expect("store"));
        let mut plain_sort = timed_sort.clone();
        let sort_elapsed = sort_by_depth_timed(&mut timed_sort, &Mat4::IDENTITY);
        sort_by_depth(&mut plain_sort, &Mat4::IDENTITY);
        assert_eq!(
            timed_sort
                .instances
                .iter()
                .map(|instance| instance.model_matrix.w_axis.z)
                .collect::<Vec<_>>(),
            plain_sort
                .instances
                .iter()
                .map(|instance| instance.model_matrix.w_axis.z)
                .collect::<Vec<_>>()
        );
        let _ = sort_elapsed;
    }

    #[test]
    fn sort_by_depth_orders_back_to_front() {
        // Identity view (camera at the origin looking down -Z): depth is
        // `-z`, so z = -1/-5/-10 sorts to -10/-5/-1. Both lanes sort.
        let mut upload = FrameUpload::default();
        for z in [-5.0, -10.0, -1.0] {
            let model = Mat4::from_translation(Vec3::new(0.0, 0.0, z));
            upload.instances.push(InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: crate::renderer::MaterialIdx::from_raw(0),
            });
        }
        sort_by_depth(&mut upload, &Mat4::IDENTITY);
        let depths: Vec<f32> = upload
            .instances
            .iter()
            .map(|instance| instance_view_depth(instance, &Mat4::IDENTITY))
            .collect();
        assert_eq!(depths, vec![10.0, 5.0, 1.0], "{depths:?}");
        let zs: Vec<f32> = upload
            .instances
            .iter()
            .map(|instance| instance.model_matrix.w_axis.z)
            .collect();
        assert_eq!(zs, vec![-10.0, -5.0, -1.0], "{zs:?}");
    }
}
