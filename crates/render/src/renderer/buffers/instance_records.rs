//! Per-object record conversion phases of one frame: the build half
//! (capacity growth, per-entity previous-transform refresh, record
//! conversion) and the write half (one full-frame `queue.write_buffer`).
//!
//! Split out of `upload_instances_with_motion_flags` only so the culling
//! profile bench can time the conversion and `upload_per_objects`
//! apart — the production entry point in the parent module composes
//! exactly these two methods, so the frame still runs one conversion and
//! one upload.
//!
//! Incremental build: the record cache of the previous frame lets a frame
//! with an ECS change feed ([`ComponentChanges`]) rebuild only the records
//! an entity mutation invalidated, instead of converting the whole scene.
//! A full rebuild still happens whenever the cache cannot be trusted — a
//! different entity list, an exhaustive change set (late enable, log
//! overflow), or no change feed at all.
use std::collections::HashMap;

use ornis_core::ComponentChanges;

use super::super::{Renderer3D, read_lock, write_lock};
use super::Entity;
use super::InstanceData;
use super::MotionValidity;
use super::PerObjectGpu;
use super::previous_instance_state;
use super::staging_capacity_for_instances;

/// One staged frame's records plus the ascending record indices whose
/// content differs from the previously staged frame.
///
/// A full rebuild reports every index; an incremental one reports only
/// the rebuilt records (the changed entities and the movers that settled
/// last frame). The GPU culler turns the indices into dirty runs, so an
/// untouched static frame uploads nothing.
pub(crate) struct StagedRecords {
    /// This frame's per-object records, in dense instance order.
    pub(crate) records: Vec<PerObjectGpu>,
    /// Ascending, deduplicated record indices that changed this frame.
    pub(crate) dirty: Vec<u32>,
}

/// The per-frame inputs of a record build: the instances, their motion
/// validity and their ECS identities — three parallel slices that always
/// travel together, bundled so the build functions take one value instead
/// of growing a parameter list per input.
#[derive(Clone, Copy)]
pub(crate) struct InstanceFrame<'a> {
    /// This frame's instances, in dense instance order.
    pub(crate) instances: &'a [InstanceData],
    /// Per-instance motion validity; `[]` means "every instance is valid".
    pub(crate) motion_valid: &'a [MotionValidity],
    /// Entity per instance, same order; a shorter list leaves the tail
    /// without identity (no temporal reuse).
    pub(crate) entities: &'a [Entity],
}

impl<'a> InstanceFrame<'a> {
    /// Bundles the three per-frame slices.
    pub(crate) fn new(
        instances: &'a [InstanceData],
        motion_valid: &'a [MotionValidity],
        entities: &'a [Entity],
    ) -> Self {
        Self {
            instances,
            motion_valid,
            entities,
        }
    }
}

/// The previous frame's record build, keyed by the entity list it was
/// built for: the records themselves, the entity → record index map the
/// change feed resolves through, and the indices that moved last frame
/// (their `previous_model` must settle to their own model this frame).
#[derive(Default)]
pub(crate) struct RecordCache {
    /// Entity identities the cached records were built from, in record
    /// order. A mismatch with the frame's list forces a full rebuild.
    entities: Vec<Entity>,
    /// Cached records, one per entry of [`Self::entities`].
    records: Vec<PerObjectGpu>,
    /// Entity → record index; only the change feed and the settling
    /// refresh resolve through it, so a frame without changes touches no
    /// entry.
    index_by_entity: HashMap<Entity, u32>,
    /// Record indices that need one more rebuild: the records a change
    /// feed rebuilt last frame (their `previous_model` still holds the
    /// pre-move transform) plus the records built without any history
    /// (first appearance — their motion flag is still `Invalid`). Both
    /// settle exactly one frame later, against the then-known history.
    pending: Vec<u32>,
}

impl Renderer3D {
    /// Convert one frame into per-object records — the CPU half of
    /// `upload_instances_with_motion_flags`: buffer growth, the
    /// per-entity previous-transform refresh and the record conversion
    /// itself, no queue write. Returns exactly the records the paired
    /// write uploads (clamped to the grown capacity).
    ///
    /// Without a change feed this is a full conversion; the cache is
    /// still refreshed, so a later
    /// [`stage_instance_records_changed`](Self::stage_instance_records_changed)
    /// frame can build incrementally.
    pub(super) fn stage_instance_records(
        &self,
        device: &wgpu::Device,
        frame: InstanceFrame<'_>,
    ) -> Vec<PerObjectGpu> {
        self.stage_instance_records_with(device, frame, None)
            .records
    }

    /// [`stage_instance_records`](Self::stage_instance_records) driven by
    /// an ECS change feed: when the entity list is unchanged and the
    /// change set is not exhaustive, only the records of changed entities
    /// (plus the ones that settled last frame) are rebuilt and reported
    /// dirty. Returns the full record list either way — the GPU culler
    /// needs every record, only the dirty subset is uploaded.
    pub(super) fn stage_instance_records_changed(
        &self,
        device: &wgpu::Device,
        frame: InstanceFrame<'_>,
        changed: &ComponentChanges,
    ) -> StagedRecords {
        self.stage_instance_records_with(device, frame, Some(changed))
    }

    /// The single record build both entry points share: cache hit means
    /// patch the changed records only, anything else means convert all.
    // qual:allow(iosp) reason: "Keep instance identities, previous transforms, and GPU staging aligned in one frame transition"
    fn stage_instance_records_with(
        &self,
        device: &wgpu::Device,
        frame: InstanceFrame<'_>,
        changed: Option<&ComponentChanges>,
    ) -> StagedRecords {
        self.ensure_instance_capacity(device, frame.instances.len());
        let count = frame
            .instances
            .len()
            .min(self.max_objects.load(std::sync::atomic::Ordering::Relaxed) as usize);
        let identity_complete = frame.entities.len() >= count;
        let mut cache = write_lock(&self.record_cache);
        let reusable = identity_complete
            && cache.records.len() == count
            && cache.entities.len() == count
            && cache.entities == frame.entities[..count];
        match changed {
            Some(changed)
                if reusable && !changed.exhaustive && !names_most_of_the_frame(changed, count) =>
            {
                self.patch_instance_records(&mut cache, frame, changed)
            }
            _ => self.rebuild_instance_records(&mut cache, count, frame, identity_complete),
        }
    }

    /// Full rebuild of every record: the pre-change-detection path, the
    /// first frame, and every frame whose cache cannot be trusted. Also
    /// refreshes the whole previous-transform map.
    fn rebuild_instance_records(
        &self,
        cache: &mut RecordCache,
        count: usize,
        frame: InstanceFrame<'_>,
        identity_complete: bool,
    ) -> StagedRecords {
        let InstanceFrame {
            instances,
            motion_valid,
            entities,
        } = frame;
        let previous_by_entity = read_lock(&self.previous_instance_models);
        let mut next_by_entity = HashMap::with_capacity(count);
        let mut without_history: Vec<u32> = Vec::new();
        let mut records: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(count));
        for (index, inst) in instances.iter().take(count).enumerate() {
            let entity = entities.get(index).copied();
            let has_history = identity_complete
                && entity.is_some_and(|entity| previous_by_entity.contains_key(&entity));
            if !has_history {
                without_history.push(index as u32);
            }
            let (previous, validity) = previous_instance_state(
                inst,
                entity,
                &previous_by_entity,
                motion_valid
                    .get(index)
                    .copied()
                    .unwrap_or(MotionValidity::Valid),
                identity_complete,
            );
            if identity_complete && let Some(entity) = entity {
                next_by_entity.insert(entity, inst.model_matrix);
            }
            records.push(record_for(inst, previous, validity));
        }
        drop(previous_by_entity);
        *write_lock(&self.previous_instance_models) = if identity_complete {
            next_by_entity
        } else {
            HashMap::new()
        };
        cache.entities.clear();
        cache.index_by_entity.clear();
        cache.pending.clear();
        if identity_complete {
            cache.entities.extend_from_slice(&entities[..count]);
            cache.index_by_entity.extend(
                cache
                    .entities
                    .iter()
                    .enumerate()
                    .map(|(index, entity)| (*entity, index as u32)),
            );
        }
        let dirty = (0..count as u32).collect();
        cache.pending = without_history;
        cache.records.clone_from(&records);
        StagedRecords { records, dirty }
    }

    /// Incremental rebuild: patch the records of the entities the change
    /// feed names and the ones that settled last frame, returning the
    /// cached records with those patches applied.
    fn patch_instance_records(
        &self,
        cache: &mut RecordCache,
        frame: InstanceFrame<'_>,
        changed: &ComponentChanges,
    ) -> StagedRecords {
        let InstanceFrame {
            instances,
            motion_valid,
            entities,
        } = frame;
        let mut changed_indices: Vec<u32> = Vec::with_capacity(changed.ids.len());
        for entity in &changed.ids {
            if let Some(&index) = cache.index_by_entity.get(entity) {
                changed_indices.push(index);
            }
        }
        changed_indices.sort_unstable();
        changed_indices.dedup();
        let mut dirty: Vec<u32> = Vec::with_capacity(changed_indices.len() + cache.pending.len());
        dirty.extend_from_slice(&changed_indices);
        dirty.extend_from_slice(&cache.pending);
        dirty.sort_unstable();
        dirty.dedup();
        if dirty.is_empty() {
            return StagedRecords {
                records: cache.records.clone(),
                dirty,
            };
        }
        let previous_by_entity = read_lock(&self.previous_instance_models);
        let mut next_by_entity = HashMap::with_capacity(dirty.len());
        for &index in &dirty {
            let inst = &instances[index as usize];
            let entity = entities[index as usize];
            let (previous, validity) = previous_instance_state(
                inst,
                Some(entity),
                &previous_by_entity,
                motion_valid
                    .get(index as usize)
                    .copied()
                    .unwrap_or(MotionValidity::Valid),
                true,
            );
            next_by_entity.insert(entity, inst.model_matrix);
            cache.records[index as usize] = record_for(inst, previous, validity);
        }
        drop(previous_by_entity);
        {
            let mut previous_models = write_lock(&self.previous_instance_models);
            for (entity, model) in next_by_entity {
                previous_models.insert(entity, model);
            }
        }
        cache.pending = changed_indices;
        StagedRecords {
            records: cache.records.clone(),
            dirty,
        }
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

/// Whether the feed names so many records that resolving, sorting and
/// deduplicating their indices would cost more than rebuilding the frame
/// outright. Half the frame is the crossover on the 1M scene (a dense
/// feed is a sort of a million `u32`s, a rebuild is one linear pass).
fn names_most_of_the_frame(changed: &ComponentChanges, count: usize) -> bool {
    changed.ids.len().saturating_mul(2) >= count
}

/// One record for `inst` with its already-resolved temporal history.
fn record_for(inst: &InstanceData, previous: glam::Mat4, validity: MotionValidity) -> PerObjectGpu {
    PerObjectGpu {
        model: inst.model_matrix.to_cols_array_2d(),
        normal_matrix: inst.normal_matrix.to_cols_array_2d(),
        previous_model: previous.to_cols_array_2d(),
        material_index: inst.material_index,
        motion_valid: validity.shader_flag(),
        _padding: [0; 2],
    }
}

#[cfg(test)]
mod tests {
    use super::super::MaterialIdx;
    use super::super::test_util;
    use super::{InstanceData, InstanceFrame, MotionValidity, PerObjectGpu, StagedRecords};
    use glam::{Mat4, Vec3};
    use ornis_core::{ComponentChanges, Entity};

    /// A test frame: the device and renderer that stage it, `n` instances
    /// in slot order, and the entity list that keys them. The staging
    /// helpers below are methods so every test reads as frame steps
    /// instead of repeating the same five arguments.
    struct TestFrame {
        device: wgpu::Device,
        renderer: super::super::super::Renderer3D,
        instances: Vec<InstanceData>,
        entities: Vec<Entity>,
    }

    impl TestFrame {
        /// `None` on a host without an adapter.
        fn new(n: usize) -> Option<Self> {
            let (device, renderer) = renderer()?;
            Some(Self {
                device,
                renderer,
                instances: (0..n).map(|i| instance(i as f32)).collect(),
                entities: (0..n as u32).map(Entity::new).collect(),
            })
        }

        /// Stage without a change feed — the full-rebuild path.
        fn stage(&self) -> Vec<PerObjectGpu> {
            self.renderer.stage_instance_records(
                &self.device,
                InstanceFrame::new(&self.instances, &[], &self.entities),
            )
        }

        /// Stage driven by a change feed — the incremental path.
        fn stage_changed(&self, changed: &ComponentChanges) -> StagedRecords {
            self.renderer.stage_instance_records_changed(
                &self.device,
                InstanceFrame::new(&self.instances, &[], &self.entities),
                changed,
            )
        }

        /// A non-exhaustive feed naming the entities at `indices`.
        fn changed_at(&self, indices: &[usize]) -> ComponentChanges {
            let ids: Vec<Entity> = indices.iter().map(|&index| self.entities[index]).collect();
            changed(&ids)
        }

        /// The steady state the incremental tests start from: build, let
        /// the first-seen records settle, and return the settled records.
        fn primed(&self) -> Vec<PerObjectGpu> {
            let _ = self.stage();
            let settled = self.stage_changed(&no_changes());
            assert_eq!(
                settled.dirty.len(),
                self.instances.len(),
                "premise: every first-seen record settles once"
            );
            self.stage()
        }
    }

    /// One instance at translation `x`; record-identity only cares about
    /// the model matrix, so the normal matrix stays the identity.
    fn instance(x: f32) -> InstanceData {
        InstanceData {
            model_matrix: Mat4::from_translation(Vec3::new(x, 0.0, 0.0)),
            normal_matrix: Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        }
    }

    /// A non-exhaustive feed naming `ids`, in feed order.
    fn changed(ids: &[Entity]) -> ComponentChanges {
        ComponentChanges {
            ids: ids.to_vec(),
            exhaustive: false,
        }
    }

    /// An exhaustive feed: the id list cannot be trusted (log overflow,
    /// tracking enabled late).
    fn exhaustive() -> ComponentChanges {
        ComponentChanges {
            ids: Vec::new(),
            exhaustive: true,
        }
    }

    fn no_changes() -> ComponentChanges {
        changed(&[])
    }

    fn renderer() -> Option<(wgpu::Device, super::super::super::Renderer3D)> {
        let (device, _queue) = test_util::try_device()?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width: 64,
            height: 64,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        Some((
            device.clone(),
            super::super::super::Renderer3D::new(&device, &config, 1),
        ))
    }

    /// A feed that names half the frame or more is a rebuild, not a
    /// patch: resolving and sorting that many indices costs more than one
    /// linear pass, so every record is reported dirty.
    #[test]
    fn a_dense_feed_rebuilds_the_whole_frame() {
        let Some(frame) = TestFrame::new(4) else {
            return;
        };
        let _ = frame.primed();
        let staged = frame.stage_changed(&frame.changed_at(&[0, 1]));
        assert_eq!(
            staged.dirty,
            vec![0, 1, 2, 3],
            "half the frame clears the density crossover"
        );
    }

    /// A frame with no reported changes rebuilds nothing: every record is
    /// byte-identical to the previous build and no dirty index is
    /// reported, so the culler uploads nothing for a static scene.
    #[test]
    fn unchanged_frame_rebuilds_nothing() {
        let Some(frame) = TestFrame::new(4) else {
            return;
        };
        let first = frame.primed();
        let second = frame.stage_changed(&no_changes());
        assert!(
            second.dirty.is_empty(),
            "nothing changed: nothing to upload"
        );
        assert_eq!(second.records, first, "a static frame keeps every record");
    }

    /// A reported entity change rebuilds only that record — the others are
    /// reused verbatim, and only the changed index is reported dirty.
    #[test]
    fn changed_entity_rebuilds_only_its_record() {
        let Some(mut frame) = TestFrame::new(4) else {
            return;
        };
        let base = frame.primed();
        frame.instances[2].model_matrix = Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0));
        let moved = frame.stage_changed(&frame.changed_at(&[2]));
        assert_eq!(moved.dirty, vec![2], "only record 2 was mutated");
        assert_eq!(
            moved.records[2].model,
            frame.instances[2].model_matrix.to_cols_array_2d()
        );
        assert_eq!(moved.records[0], base[0]);
        assert_eq!(moved.records[1], base[1]);
        assert_eq!(moved.records[3], base[3]);
    }

    /// The frame after a move settles the mover: its record is rebuilt
    /// once more without a change-feed entry, so `previous_model` falls
    /// back to its own model (zero motion) instead of reporting last
    /// frame's displacement forever.
    #[test]
    fn settling_frame_zeroes_the_motion_of_the_moved_record() {
        let Some(mut frame) = TestFrame::new(3) else {
            return;
        };
        let _ = frame.primed();
        frame.instances[1].model_matrix = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let moved = frame.stage_changed(&frame.changed_at(&[1]));
        assert_eq!(moved.dirty, vec![1]);

        let settled = frame.stage_changed(&no_changes());
        assert_eq!(settled.dirty, vec![1], "the mover settles one frame later");
        assert_eq!(
            settled.records[1].previous_model, settled.records[1].model,
            "a settled record reports zero motion"
        );
        assert_eq!(
            settled.records[1].motion_valid,
            MotionValidity::Valid.shader_flag()
        );

        let after_settle = frame.stage_changed(&no_changes());
        assert!(
            after_settle.dirty.is_empty(),
            "settling happens exactly once"
        );
    }

    /// A different entity list (spawn, despawn, reorder) invalidates the
    /// cache: every record is rebuilt and reported dirty, whatever the
    /// change feed claims.
    #[test]
    fn entity_list_change_forces_a_full_rebuild() {
        let Some(frame) = TestFrame::new(3) else {
            return;
        };
        let _ = frame.stage();
        let reordered = [frame.entities[2], frame.entities[0], frame.entities[1]];
        let full = frame.renderer.stage_instance_records_changed(
            &frame.device,
            InstanceFrame::new(&frame.instances, &[], &reordered),
            &no_changes(),
        );
        assert_eq!(full.dirty, vec![0, 1, 2], "a new order rebuilds all");
        assert_eq!(
            full.records[0].model,
            frame.instances[0].model_matrix.to_cols_array_2d(),
            "records stay in instance order — the entity list only keys them"
        );
    }

    /// An exhaustive change set (log overflow, tracking enabled late) also
    /// rebuilds everything — the feed itself says the id list cannot be
    /// trusted.
    #[test]
    fn exhaustive_change_set_forces_a_full_rebuild() {
        let Some(frame) = TestFrame::new(3) else {
            return;
        };
        let _ = frame.stage();
        let full = frame.stage_changed(&exhaustive());
        assert_eq!(full.dirty, vec![0, 1, 2]);
    }

    /// A full build (no change feed) still primes the cache, so the next
    /// frame with a feed goes incremental — the feed does not have to be
    /// present from the very first frame.
    #[test]
    fn full_build_primes_the_cache_for_a_later_feed() {
        let Some(mut frame) = TestFrame::new(3) else {
            return;
        };
        let base = frame.primed();
        frame.instances[0].model_matrix = Mat4::from_translation(Vec3::new(-7.0, 0.0, 0.0));
        let patched = frame.stage_changed(&frame.changed_at(&[0]));
        assert_eq!(patched.dirty, vec![0]);
        assert_eq!(
            patched.records[0].model,
            frame.instances[0].model_matrix.to_cols_array_2d()
        );
        assert_eq!(patched.records[1], base[1]);
        assert_eq!(patched.records[2], base[2]);
    }
}
