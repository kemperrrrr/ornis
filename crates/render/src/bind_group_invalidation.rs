//! Bind-group invalidation gates (review #100): the cached indirect
//! bind group must be dropped whenever a resource it binds is replaced —
//! the culler's `visible_buffer` on capacity growth and the shared
//! material / per-object storage buffers on table growth. Each test
//! renders an indirect frame, triggers the growth, renders again, and
//! asserts the pixels match the direct fallback drawn on the same
//! renderer; a stale bind group reads the old buffer and fails.
#![cfg(test)]
use super::super::read_lock;
use super::{
    InstanceData, MaterialIdx, Mesh, Renderer3D, offscreen_frame, setup_scene, surface_config,
    test_util,
};

/// One routed composite frame on an existing `renderer` (indirect when
/// armed and ready, direct otherwise) over `mesh`'s `instance_count`
/// instances; returns the composite pixels.
fn render_existing_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &Mesh,
    instance_count: u32,
) -> Vec<u8> {
    offscreen_frame(device, queue, renderer, |encoder, g| {
        renderer.render_gbuffer_with_custom(device, encoder, g, mesh, instance_count, &[]);
        renderer.render_shadows(device, encoder, mesh, instance_count);
    })
}

/// Arms the opt-in, stages `(view_proj, instances)` as the indirect
/// batch (growing or creating the culler) and prepares it — asserting
/// the frame is armed for an indirect submit.
fn stage_and_prepare(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &Mesh,
    staged: (glam::Mat4, &[InstanceData]),
) {
    let (view_proj, instances) = staged;
    let mut converted = Vec::with_capacity(instances.len());
    crate::gpu_culling::convert_instances(&mut converted, instances);
    renderer.set_indirect_draw(true);
    renderer.stage_batch_inputs(view_proj, converted);
    renderer.prepare_indirect_batch(device, queue, mesh.num_indices);
    assert!(
        renderer.indirect_batch_ready(),
        "batch must be armed after prepare"
    );
}

/// Disarms the opt-in and renders the routed frame through the direct
/// fallback on the same renderer: the reference pixels the invalidation
/// comparisons assert against.
fn direct_reference_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &Mesh,
    instance_count: u32,
) -> Vec<u8> {
    renderer.set_indirect_draw(false);
    renderer.prepare_indirect_batch(device, queue, mesh.num_indices);
    assert!(!renderer.indirect_batch_ready());
    let direct = render_existing_frame(device, queue, renderer, mesh, instance_count);
    assert!(
        direct.iter().any(|channel| *channel > 0),
        "direct reference must render something"
    );
    direct
}

/// A translation/scale instance builder for the invalidation tests.
fn inv_instance(translation: [f32; 3], scale: f32) -> InstanceData {
    InstanceData {
        model_matrix: glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(scale),
            glam::Quat::IDENTITY,
            glam::Vec3::from(translation),
        ),
        normal_matrix: glam::Mat4::IDENTITY,
        material_index: MaterialIdx::from_raw(0),
    }
}

/// The shared opening of the invalidation tests: a renderer with the
/// plain two-sphere scene, armed and staged for frame A, rendered —
/// so the first indirect draw caches the bind group the growth later
/// invalidates. Returns `(renderer, mesh, instances, view_proj, frame_a)`.
fn frame_a_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (Renderer3D, Mesh, Vec<InstanceData>, glam::Mat4, Vec<u8>) {
    let renderer = Renderer3D::new(device, &surface_config(), 1);
    let (mesh, instances, view_proj) = setup_scene(device, queue, &renderer, false, false);
    stage_and_prepare(device, queue, &renderer, &mesh, (view_proj, &instances));
    let frame_a = render_existing_frame(device, queue, &renderer, &mesh, 2);
    (renderer, mesh, instances, view_proj, frame_a)
}

/// Review #100: the cached indirect bind group must be invalidated when
/// the culler is re-created on capacity growth (`ensure_culler_capacity`).
/// Two indirect frames on one renderer with a growing scene: the second
/// frame compacts into a NEW `visible_buffer`; a stale bind group would
/// keep the old buffer and draw the first frame's content. Fails if the
/// invalidation line in `culling.rs` is removed.
#[test]
fn indirect_bind_group_invalidated_on_culler_growth() {
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (indirect_bind_group_invalidated_on_culler_growth): no wgpu adapter");
        return;
    };
    // Frame A (`frame_a_scene`): setup_scene's two visible spheres — the
    // culler is created with capacity 2 and the indirect bind group is
    // cached over its `visible_buffer`.
    let (renderer, mesh, _instances, view_proj, _frame_a) = frame_a_scene(&device, &queue);
    // Frame B: five spheres at other positions — the culler must grow
    // 2 → 4 → 8, which recreates it and drops the cached indirect bind
    // group.
    let grown = vec![
        inv_instance([-2.0, 1.0, 0.5], 0.8),
        inv_instance([1.0, -0.5, 1.5], 1.0),
        inv_instance([-1.0, 0.0, -1.0], 0.9),
        inv_instance([0.5, 1.5, 0.0], 0.7),
        inv_instance([-0.5, -1.0, 2.0], 1.1),
    ];
    renderer.upload_instances(&device, &queue, &grown);
    stage_and_prepare(&device, &queue, &renderer, &mesh, (view_proj, &grown));
    {
        let culler = read_lock(&renderer.culler);
        let culler = culler.as_ref().expect("culler built");
        assert!(
            culler.capacity >= 5,
            "culler must have grown for 5 instances (capacity={})",
            culler.capacity
        );
    }
    let indirect_b = render_existing_frame(&device, &queue, &renderer, &mesh, 5);
    let direct_b = direct_reference_frame(&device, &queue, &renderer, &mesh, 5);
    assert_eq!(
        indirect_b, direct_b,
        "indirect frame after culler growth must match the direct fallback"
    );
}

/// Review #100: the cached indirect bind group must be invalidated when
/// the shared material / per-object storage buffers grow
/// (`rebind_storage_buffers`). Two indirect frames on one renderer: the
/// second frame re-uploads a larger material table (buffer growth) with
/// a changed material-0 colour; a stale bind group would keep the old
/// material buffer and draw the old colour. Fails if the invalidation
/// line in `buffers.rs` is removed.
#[test]
fn indirect_bind_group_invalidated_on_material_growth() {
    let Some((device, queue)) = test_util::try_device() else {
        eprintln!("SKIPPED (indirect_bind_group_invalidated_on_material_growth): no wgpu adapter");
        return;
    };
    // Frame A (`frame_a_scene`): gray material 0 — culler capacity 2,
    // the indirect bind group is cached over the initial material buffer.
    let (renderer, mesh, instances, view_proj, frame_a) = frame_a_scene(&device, &queue);
    // Between frames: grow the material table past INITIAL_MAX_MATERIALS
    // (64) with material 0 now green — `rebind_storage_buffers` must drop
    // the cached indirect bind group.
    let materials: Vec<ornis_core::OpenPBRMaterial> = (0..65)
        .map(|i| {
            let mut m = ornis_core::OpenPBRMaterial::dielectric();
            m.base.color_rgb(if i == 0 {
                [0.1, 0.9, 0.1]
            } else {
                [0.5, 0.5, 0.5]
            });
            m.specular.roughness(0.5);
            m
        })
        .collect();
    renderer.upload_materials(&device, &queue, &materials);
    // Frame B: the same two spheres; the culler still fits, so only the
    // material-buffer growth path could have invalidated the bind group.
    stage_and_prepare(&device, &queue, &renderer, &mesh, (view_proj, &instances));
    {
        let culler = read_lock(&renderer.culler);
        let culler = culler.as_ref().expect("culler built");
        assert_eq!(
            culler.capacity, 2,
            "culler must not grow for the same 2 instances"
        );
    }
    let indirect_b = render_existing_frame(&device, &queue, &renderer, &mesh, 2);
    let direct_b = direct_reference_frame(&device, &queue, &renderer, &mesh, 2);
    // Premise: the material change is visible (gray frame A ≠ green
    // direct B), so the indirect/direct comparison cannot pass vacuously.
    assert_ne!(
        frame_a, direct_b,
        "material change must be visible (gray frame A vs green direct B)"
    );
    assert_eq!(
        indirect_b, direct_b,
        "indirect frame after material growth must match the direct fallback"
    );
}
