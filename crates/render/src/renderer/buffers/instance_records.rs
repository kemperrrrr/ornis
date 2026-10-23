//! Per-object record conversion phases of one frame: the build half
//! (capacity growth, per-entity previous-transform refresh, record
//! conversion) and the write half (one full-frame `queue.write_buffer`).
//!
//! Split out of `upload_instances_with_motion_flags` only so the culling
//! profile bench can time the conversion and `upload_per_objects`
//! apart — the production entry point in the parent module composes
//! exactly these two methods, so the frame still runs one conversion and
//! one upload.
use super::super::{Renderer3D, VEC4_COMPONENTS, read_lock, write_lock};
use super::Entity;
use super::InstanceData;
use super::MotionValidity;
use super::PerObjectGpu;
use super::previous_instance_state;
use super::staging_capacity_for_instances;

impl Renderer3D {
    /// Convert one frame into per-object records — the CPU half of
    /// `upload_instances_with_motion_flags`: buffer growth, the
    /// per-entity previous-transform refresh and the record conversion
    /// itself, no queue write. Returns exactly the records the paired
    /// write uploads (clamped to the grown capacity).
    // qual:allow(iosp) reason: "Keep instance identities, previous transforms, and GPU staging aligned in one frame transition"
    pub(super) fn stage_instance_records(
        &self,
        device: &wgpu::Device,
        instances: &[InstanceData],
        motion_valid: &[MotionValidity],
        entities: &[Entity],
    ) -> Vec<PerObjectGpu> {
        self.ensure_instance_capacity(device, instances.len());
        let count = instances
            .len()
            .min(self.max_objects.load(std::sync::atomic::Ordering::Relaxed) as usize);
        let identity_complete = entities.len() >= count;
        let previous_by_entity = read_lock(&self.previous_instance_models);
        let mut next_by_entity = std::collections::HashMap::with_capacity(count);
        let mut gpu_objects: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(count));
        for (index, inst) in instances.iter().take(count).enumerate() {
            let model_arr: [[f32; VEC4_COMPONENTS]; VEC4_COMPONENTS] =
                inst.model_matrix.to_cols_array_2d();
            let normal_arr: [[f32; VEC4_COMPONENTS]; VEC4_COMPONENTS] =
                inst.normal_matrix.to_cols_array_2d();
            let entity = entities.get(index).copied();
            let requested = motion_valid
                .get(index)
                .copied()
                .unwrap_or(MotionValidity::Valid);
            let (previous, validity) = previous_instance_state(
                inst,
                entity,
                &previous_by_entity,
                requested,
                identity_complete,
            );
            if identity_complete && let Some(entity) = entity {
                next_by_entity.insert(entity, inst.model_matrix);
            }
            gpu_objects.push(PerObjectGpu {
                model: model_arr,
                normal_matrix: normal_arr,
                previous_model: previous.to_cols_array_2d(),
                material_index: inst.material_index,
                motion_valid: validity.shader_flag(),
                _padding: [0; 2],
            });
        }
        drop(previous_by_entity);
        *write_lock(&self.previous_instance_models) = if identity_complete {
            next_by_entity
        } else {
            std::collections::HashMap::new()
        };
        gpu_objects
    }

    /// Push staged records into the per-object buffer — the GPU half of
    /// the upload: one full-frame `queue.write_buffer`, submitted by the
    /// frame like every other staged write. This is the
    /// `upload_per_objects` stage of the profile report.
    pub(super) fn write_instance_records(&self, queue: &wgpu::Queue, records: &[PerObjectGpu]) {
        queue.write_buffer(
            &read_lock(&self.per_object_buffer),
            0,
            bytemuck::cast_slice(records),
        );
    }
}
