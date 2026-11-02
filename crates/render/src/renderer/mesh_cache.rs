//! Cross-frame GPU buffer cache for per-entity custom / skinned meshes.
//!
//! [`super::Renderer3D::stage_custom_meshes`] used to allocate a fresh
//! vertex + index buffer pair for every entry on every frame. The cache
//! keeps one pair per entity alive between frames instead:
//!
//! * contents identical to last frame — no upload at all (static index
//!   lists, GPU-skinned bind-pose rows);
//! * new contents that fit — `queue.write_buffer` into the existing buffer;
//! * new contents that do not fit — the buffer is recreated (the only path
//!   that allocates; counted by the cache allocation counter).
//!
//! Entities not staged in a frame are evicted at its end. `write_buffer`
//! lands before the commands of the next `queue.submit`, so the staged
//! meshes of one staging call must be submitted before the next call
//! rewrites the same buffers (the once-per-frame contract of the caller).

use crate::mesh::Mesh;
use ornis_core::Entity;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Buffer counters of the custom-mesh cache (tests).
#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CustomMeshBufferStats {
    /// GPU buffers created since the cache was built.
    pub(crate) allocations: u64,
    /// `queue.write_buffer` uploads into an existing buffer.
    pub(crate) writes: u64,
    /// Entities currently holding a cached buffer pair.
    pub(crate) cached_entities: usize,
}

/// Which buffer of the pair a [`Slot`] is (label + usage).
#[derive(Clone, Copy)]
enum SlotKind {
    Vertex,
    Index,
}

impl SlotKind {
    fn label(self) -> &'static str {
        match self {
            Self::Vertex => "custom_mesh_vb",
            Self::Index => "custom_mesh_ib",
        }
    }

    fn usage(self) -> wgpu::BufferUsages {
        match self {
            Self::Vertex => wgpu::BufferUsages::VERTEX,
            Self::Index => wgpu::BufferUsages::INDEX,
        }
    }
}

/// Allocation / upload counters.
#[derive(Default)]
struct Counters {
    allocations: u64,
    writes: u64,
}

/// One GPU buffer plus the bytes last written to it.
struct Slot {
    kind: SlotKind,
    buffer: wgpu::Buffer,
    capacity: u64,
    bytes: Vec<u8>,
}

impl Slot {
    fn create(device: &wgpu::Device, kind: SlotKind, data: &[u8], counters: &mut Counters) -> Self {
        counters.allocations += 1;
        Self {
            kind,
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(kind.label()),
                contents: data,
                usage: kind.usage() | wgpu::BufferUsages::COPY_DST,
            }),
            capacity: data.len() as u64,
            bytes: data.to_vec(),
        }
    }

    /// Unchanged — nothing; fits — `write_buffer`; grew — recreate.
    fn update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        counters: &mut Counters,
    ) {
        let action = self.plan(data);
        action.apply(self, device, queue, data, counters);
    }

    /// Decides what an update with `data` needs; pure, touches no GPU state.
    fn plan(&self, data: &[u8]) -> SlotAction {
        if self.bytes == data {
            SlotAction::Keep
        } else if data.len() as u64 <= self.capacity {
            SlotAction::Write
        } else {
            SlotAction::Recreate
        }
    }

    fn write(&mut self, queue: &wgpu::Queue, data: &[u8], counters: &mut Counters) {
        queue.write_buffer(&self.buffer, 0, data);
        counters.writes += 1;
        self.bytes.clear();
        self.bytes.extend_from_slice(data);
    }
}

#[derive(Clone, Copy)]
enum SlotAction {
    Keep,
    Write,
    Recreate,
}

impl SlotAction {
    fn apply(
        self,
        slot: &mut Slot,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        counters: &mut Counters,
    ) {
        match self {
            Self::Keep => {}
            Self::Write => slot.write(queue, data, counters),
            Self::Recreate => *slot = Slot::create(device, slot.kind, data, counters),
        }
    }
}

struct CachedMesh {
    vertex: Slot,
    index: Slot,
    seen_frame: u64,
}

/// Per-entity buffer cache; lives behind a lock on the renderer.
#[derive(Default)]
pub(crate) struct CustomMeshCache {
    meshes: HashMap<Entity, CachedMesh>,
    frame: u64,
    counters: Counters,
}

impl CustomMeshCache {
    /// Starts a staging pass (one per frame).
    pub(crate) fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// Drops the buffers of entities not staged since [`Self::begin_frame`].
    pub(crate) fn end_frame(&mut self) {
        let frame = self.frame;
        self.meshes.retain(|_, m| m.seen_frame == frame);
    }

    /// Current counters.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> CustomMeshBufferStats {
        CustomMeshBufferStats {
            allocations: self.counters.allocations,
            writes: self.counters.writes,
            cached_entities: self.meshes.len(),
        }
    }

    /// Stages `entry`'s rows: interleaved skinned rows when `gpu_draw`,
    /// classic rows otherwise. `None` when a GPU entry has no interleaved
    /// rows; `Some(Err)` when the rows fail the `upload_*` validation
    /// (nothing is cached then).
    pub(crate) fn stage_entry(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entry: &crate::extraction::CustomMeshEntry,
        gpu_draw: bool,
    ) -> Option<Result<Mesh, crate::mesh_upload::UploadError>> {
        let indices = &entry.indices;
        if !gpu_draw {
            return Some(
                super::meshes::validate_rows(entry.vertices.len(), indices)
                    .map(|()| self.stage(device, queue, entry.entity, &entry.vertices, indices)),
            );
        }
        let rows = entry.skinned_gpu_vertices()?;
        Some(
            super::meshes::validate_rows(rows.len(), indices)
                .map(|()| self.stage(device, queue, entry.entity, &rows, indices)),
        )
    }

    /// Returns a [`Mesh`] over `entity`'s cached buffers holding `rows` /
    /// `indices` (already validated by the caller). A second request for
    /// the same entity within one frame gets fresh, uncached buffers so it
    /// cannot overwrite the first one before submit.
    pub(crate) fn stage<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entity: Entity,
        rows: &[T],
        indices: &[u32],
    ) -> Mesh {
        let vertices: &[u8] = bytemuck::cast_slice(rows);
        let index_bytes: &[u8] = bytemuck::cast_slice(indices);
        let frame = self.frame;
        let counters = &mut self.counters;
        let cached = match self.meshes.get_mut(&entity) {
            Some(m) if m.seen_frame == frame => None,
            Some(m) => {
                m.seen_frame = frame;
                m.vertex.update(device, queue, vertices, counters);
                m.index.update(device, queue, index_bytes, counters);
                Some(&*m)
            }
            None => {
                let m = CachedMesh {
                    vertex: Slot::create(device, SlotKind::Vertex, vertices, counters),
                    index: Slot::create(device, SlotKind::Index, index_bytes, counters),
                    seen_frame: frame,
                };
                Some(&*self.meshes.entry(entity).or_insert(m))
            }
        };
        let (vertex_buffer, index_buffer) = match cached {
            Some(m) => (m.vertex.buffer.clone(), m.index.buffer.clone()),
            None => (
                Slot::create(device, SlotKind::Vertex, vertices, counters).buffer,
                Slot::create(device, SlotKind::Index, index_bytes, counters).buffer,
            ),
        };
        Mesh {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            vertex_count: rows.len() as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRI: [u32; 3] = [0, 1, 2];

    fn rows(value: f32, count: usize) -> Vec<[f32; 4]> {
        vec![[value; 4]; count]
    }

    fn frame(
        cache: &mut CustomMeshCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[(Entity, &[[f32; 4]])],
    ) -> Vec<Mesh> {
        cache.begin_frame();
        let staged = meshes
            .iter()
            .map(|(entity, bytes)| cache.stage(device, queue, *entity, bytes, &TRI))
            .collect();
        cache.end_frame();
        queue.submit([]);
        staged
    }

    #[test]
    fn repeated_staging_reuses_buffers_and_allocations_stay_flat() {
        let Some((device, queue)) = super::super::test_util::try_device() else {
            return;
        };
        let mut cache = CustomMeshCache::default();
        let entity = Entity::new(7);
        let bind = rows(1.0, 3);
        let first = frame(&mut cache, &device, &queue, &[(entity, &bind)]);
        assert_eq!(cache.stats().allocations, 2);
        for _ in 0..10 {
            let again = frame(&mut cache, &device, &queue, &[(entity, &bind)]);
            assert_eq!(again[0].vertex_buffer, first[0].vertex_buffer);
            assert_eq!(again[0].index_buffer, first[0].index_buffer);
        }
        // Unchanged contents: no allocation and no upload.
        assert_eq!(cache.stats().allocations, 2);
        assert_eq!(cache.stats().writes, 0);

        // Same-size new vertices (CPU skin): write_buffer, indices untouched.
        let posed = frame(&mut cache, &device, &queue, &[(entity, &rows(2.0, 3))]);
        assert_eq!(posed[0].vertex_buffer, first[0].vertex_buffer);
        assert_eq!(cache.stats().allocations, 2);
        assert_eq!(cache.stats().writes, 1);

        // Growth reallocates only the grown buffer.
        let grown = frame(&mut cache, &device, &queue, &[(entity, &rows(2.0, 6))]);
        assert_ne!(grown[0].vertex_buffer, first[0].vertex_buffer);
        assert_eq!(grown[0].index_buffer, first[0].index_buffer);
        assert_eq!(cache.stats().allocations, 3);
        assert_eq!(grown[0].vertex_count, 6);
    }

    #[test]
    fn absent_entities_are_evicted_and_duplicates_get_own_buffers() {
        let Some((device, queue)) = super::super::test_util::try_device() else {
            return;
        };
        let mut cache = CustomMeshCache::default();
        let (a, b) = (Entity::new(1), Entity::new(2));
        let bytes = rows(1.0, 3);
        frame(&mut cache, &device, &queue, &[(a, &bytes), (b, &bytes)]);
        assert_eq!(cache.stats().cached_entities, 2);
        frame(&mut cache, &device, &queue, &[(a, &bytes)]);
        assert_eq!(cache.stats().cached_entities, 1);

        let dup = frame(&mut cache, &device, &queue, &[(a, &bytes), (a, &bytes)]);
        assert_ne!(dup[0].vertex_buffer, dup[1].vertex_buffer);
        assert_eq!(cache.stats().cached_entities, 1);
    }
}
