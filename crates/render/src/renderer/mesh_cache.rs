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
//!   that allocates; counted in [`CustomMeshBufferStats::allocations`]).
//!
//! Entities not staged in a frame are evicted at its end. `write_buffer`
//! lands before the commands of the next `queue.submit`, so the staged
//! meshes of one staging call must be submitted before the next call
//! rewrites the same buffers (the once-per-frame contract of the caller).

use crate::mesh::Mesh;
use ornis_core::Entity;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Buffer counters of the custom-mesh cache (diagnostics / tests).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CustomMeshBufferStats {
    /// GPU buffers created since the renderer was built.
    pub allocations: u64,
    /// `queue.write_buffer` uploads into an existing buffer.
    pub writes: u64,
    /// Entities currently holding a cached buffer pair.
    pub cached_entities: usize,
}

/// One GPU buffer plus the bytes last written to it.
struct Slot {
    buffer: wgpu::Buffer,
    capacity: u64,
    bytes: Vec<u8>,
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
    allocations: u64,
    writes: u64,
}

fn create_slot(
    device: &wgpu::Device,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[u8],
    allocations: &mut u64,
) -> Slot {
    *allocations += 1;
    Slot {
        buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage: usage | wgpu::BufferUsages::COPY_DST,
        }),
        capacity: data.len() as u64,
        bytes: data.to_vec(),
    }
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
    pub(crate) fn stats(&self) -> CustomMeshBufferStats {
        CustomMeshBufferStats {
            allocations: self.allocations,
            writes: self.writes,
            cached_entities: self.meshes.len(),
        }
    }

    /// Validates `rows` / `indices` like the `upload_*` helpers, then
    /// [`Self::stage`]s them.
    ///
    /// # Errors
    ///
    /// The [`super::meshes::validate_rows`] errors; nothing is cached then.
    pub(crate) fn stage_rows<T: bytemuck::Pod>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entity: Entity,
        rows: &[T],
        indices: &[u32],
    ) -> Result<Mesh, crate::mesh_upload::UploadError> {
        super::meshes::validate_rows(rows.len(), indices)?;
        Ok(self.stage(
            device,
            queue,
            entity,
            bytemuck::cast_slice(rows),
            rows.len() as u32,
            indices,
        ))
    }

    /// Returns a [`Mesh`] over `entity`'s cached buffers holding `vertices`
    /// / `indices` (already validated by the caller). A second request for
    /// the same entity within one frame gets fresh, uncached buffers so it
    /// cannot overwrite the first one before submit.
    pub(crate) fn stage(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entity: Entity,
        vertices: &[u8],
        vertex_count: u32,
        indices: &[u32],
    ) -> Mesh {
        let index_bytes: &[u8] = bytemuck::cast_slice(indices);
        let frame = self.frame;
        let (allocations, writes) = (&mut self.allocations, &mut self.writes);
        let cached = match self.meshes.get_mut(&entity) {
            Some(m) if m.seen_frame == frame => None,
            Some(m) => {
                m.seen_frame = frame;
                update_slot(
                    device,
                    queue,
                    &mut m.vertex,
                    "custom_mesh_vb",
                    wgpu::BufferUsages::VERTEX,
                    vertices,
                    allocations,
                    writes,
                );
                update_slot(
                    device,
                    queue,
                    &mut m.index,
                    "custom_mesh_ib",
                    wgpu::BufferUsages::INDEX,
                    index_bytes,
                    allocations,
                    writes,
                );
                Some(&*m)
            }
            None => {
                let m = CachedMesh {
                    vertex: create_slot(
                        device,
                        "custom_mesh_vb",
                        wgpu::BufferUsages::VERTEX,
                        vertices,
                        allocations,
                    ),
                    index: create_slot(
                        device,
                        "custom_mesh_ib",
                        wgpu::BufferUsages::INDEX,
                        index_bytes,
                        allocations,
                    ),
                    seen_frame: frame,
                };
                Some(&*self.meshes.entry(entity).or_insert(m))
            }
        };
        let (vertex_buffer, index_buffer) = match cached {
            Some(m) => (m.vertex.buffer.clone(), m.index.buffer.clone()),
            None => (
                create_slot(
                    device,
                    "custom_mesh_vb",
                    wgpu::BufferUsages::VERTEX,
                    vertices,
                    allocations,
                )
                .buffer,
                create_slot(
                    device,
                    "custom_mesh_ib",
                    wgpu::BufferUsages::INDEX,
                    index_bytes,
                    allocations,
                )
                .buffer,
            ),
        };
        Mesh {
            vertex_buffer,
            index_buffer,
            num_indices: indices.len() as u32,
            vertex_count,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn update_slot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    slot: &mut Slot,
    label: &str,
    usage: wgpu::BufferUsages,
    data: &[u8],
    allocations: &mut u64,
    writes: &mut u64,
) {
    if slot.bytes == data {
        return;
    }
    if data.len() as u64 <= slot.capacity {
        queue.write_buffer(&slot.buffer, 0, data);
        *writes += 1;
        slot.bytes.clear();
        slot.bytes.extend_from_slice(data);
    } else {
        *slot = create_slot(device, label, usage, data, allocations);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRI: [u32; 3] = [0, 1, 2];

    fn rows(value: f32, count: usize) -> Vec<u8> {
        bytemuck::cast_slice(&vec![value; count * 4]).to_vec()
    }

    fn frame(
        cache: &mut CustomMeshCache,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        meshes: &[(Entity, &[u8])],
    ) -> Vec<Mesh> {
        cache.begin_frame();
        let staged = meshes
            .iter()
            .map(|(entity, bytes)| cache.stage(device, queue, *entity, bytes, 3, &TRI))
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
        assert_eq!(grown[0].vertex_count, 3);
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
