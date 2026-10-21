// qual:allow(srp, file_length=440) reason: "Keep palette and instance buffer history together until this pinned module budget is exceeded"
//! Growable GPU stores: per-object instances, materials, and skinning palettes with their capacity management and upload entry points.
use super::*;
use crate::antialias::MotionValidity;
use ornis_core::Entity;
/// GPU per-instance record mirroring CPU [`InstanceData`] with padding to 16 bytes.
///
/// The WGSL `PerObject` declaration is generated from this layout
/// ([`PerObjectGpu::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "PerObject")]
pub struct PerObjectGpu {
    /// Local-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose model matrix (normal transform).
    pub normal_matrix: [[f32; 4]; 4],
    /// Previous frame's local-to-world matrix: paired with the previous
    /// camera and skin palette to write object motion into the g-buffer.
    pub previous_model: [[f32; 4]; 4],
    /// Index into the material buffer uploaded by `upload_materials`.
    pub material_index: MaterialIdx,
    /// Whether the previous vertex position is trustworthy for temporal reprojection.
    pub motion_valid: u32,
    /// Aligns the record to 16-byte stride (padding: not shader-visible).
    #[wgsl(skip)]
    _padding: [u32; 2],
}

/// Index into the deduplicated material table ([`FrameUpload::materials`](crate::extraction::FrameUpload)).
///
/// Newtype over `u32` so material indices never mix with texture handles
/// or entity ids at the type level. The DSL substitutes it to WGSL `u32`
/// (like the CPU compiler erases `repr(transparent)` wrappers), so GPU
/// records ([`PerObjectGpu`], shader interfaces) spell it directly.
#[repr(transparent)]
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
pub struct MaterialIdx(u32);

impl MaterialIdx {
    /// Wraps a raw `u32` material table index.
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Raw `u32` material table index.
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Material index as `usize` for table lookups.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for MaterialIdx {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for MaterialIdx {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

impl From<MaterialIdx> for u32 {
    fn from(h: MaterialIdx) -> Self {
        h.0
    }
}

impl From<MaterialIdx> for usize {
    fn from(h: MaterialIdx) -> Self {
        h.0 as usize
    }
}

/// CPU-side description of one drawn instance.
#[derive(Debug, Clone, Copy)]
pub struct InstanceData {
    /// Local-to-world matrix.
    pub model_matrix: Mat4,
    /// Normal matrix (inverse transpose of the linear part).
    pub normal_matrix: Mat4,
    /// Index into the material table.
    pub material_index: MaterialIdx,
}

/// Exact CPU staging capacity for one [`Renderer3D::upload_instances`]
/// call: the caller reserves this once up front, so the per-instance
/// conversion never reallocates mid-frame. Identity today (exact fit —
/// a repeated same-size frame reserves the same capacity, no growth);
/// kept as a named helper so the upload path reads as one reservation.
pub(crate) fn staging_capacity_for_instances(needed: usize) -> usize {
    needed
}

fn previous_instance_state(
    instance: &InstanceData,
    entity: Option<Entity>,
    previous_by_entity: &std::collections::HashMap<Entity, Mat4>,
    requested_validity: MotionValidity,
    identity_complete: bool,
) -> (Mat4, MotionValidity) {
    let previous = entity
        .and_then(|entity| previous_by_entity.get(&entity).copied())
        .unwrap_or(instance.model_matrix);
    let validity = match entity {
        Some(entity) if identity_complete && previous_by_entity.contains_key(&entity) => {
            requested_validity
        }
        _ => MotionValidity::Invalid,
    };
    (previous, validity)
}

fn previous_palette_frame(
    previous_by_entity: &std::collections::HashMap<Entity, Vec<u8>>,
    current_bytes: &[u8],
    entities: &[Entity],
    count: usize,
) -> (
    Vec<u8>,
    Vec<MotionValidity>,
    std::collections::HashMap<Entity, Vec<u8>>,
) {
    let identity_complete = entities.len() >= count;
    let mut previous_bytes = Vec::with_capacity(count * PALETTE_BYTE_SIZE);
    let mut motion_validity = Vec::with_capacity(count);
    let mut next_by_entity = std::collections::HashMap::with_capacity(count);
    for slot in 0..count {
        let start = slot * PALETTE_BYTE_SIZE;
        let end = start + PALETTE_BYTE_SIZE;
        let current = &current_bytes[start..end];
        let entity = entities.get(slot).copied();
        let previous = if identity_complete {
            entity.and_then(|entity| {
                previous_by_entity
                    .get(&entity)
                    .filter(|previous| previous.len() == PALETTE_BYTE_SIZE)
            })
        } else {
            None
        };
        if let Some(previous) = previous {
            previous_bytes.extend_from_slice(previous);
            motion_validity.push(MotionValidity::Valid);
        } else {
            previous_bytes.extend_from_slice(current);
            motion_validity.push(MotionValidity::Invalid);
        }
        if identity_complete && let Some(entity) = entity {
            next_by_entity.insert(entity, current.to_vec());
        }
    }
    (previous_bytes, motion_validity, next_by_entity)
}

impl Renderer3D {
    /// Grow a storage buffer when `needed` exceeds `capacity`, doubling
    /// until it fits (amortized O(1) across frames).
    fn grown_storage_buffer(
        device: &wgpu::Device,
        label: &str,
        old: &wgpu::Buffer,
        element_bytes: usize,
        needed: usize,
        capacity: &std::sync::atomic::AtomicU32,
    ) -> Option<wgpu::Buffer> {
        if needed <= capacity.load(std::sync::atomic::Ordering::Relaxed) as usize {
            return None;
        }
        let mut grown = capacity.load(std::sync::atomic::Ordering::Relaxed).max(1);
        while (grown as usize) < needed {
            grown = grown.saturating_mul(2);
        }
        capacity.store(grown, std::sync::atomic::Ordering::Relaxed);
        // Destroying the buffer under a live bind group would fault the
        // next submit; wgpu defers actual destruction until the GPU is
        // done, so replace-then-rebind inside the same frame is safe.
        old.destroy();
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (element_bytes * grown as usize) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    }

    /// Rebind every pass that reads the grown storage buffers: each pass
    /// holds its own bind group over the same layout, so all three are
    /// rebuilt together whenever either buffer moves. The cached
    /// indirect bind group (bound over the material table) is dropped
    /// too and rebuilt on the next indirect draw.
    fn rebind_storage_buffers(&self, device: &wgpu::Device) {
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
        *write_lock(&self.indirect_bind_group) = None;
        *write_lock(&self.gbuffer_bind_group) =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gbuffer bind group (grown)"),
                layout: &self.gbuffer_bind_group_layout,
                entries: &shaders::bind_group_entries(
                    &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                    |r| match r.name {
                        "camera" => Some(self.camera_buffer.as_entire_binding()),
                        "per_objects" => Some(per_object.as_entire_binding()),
                        "materials" => Some(material.as_entire_binding()),
                        _ => None,
                    },
                )
                .unwrap_or_default(),
            });
        *write_lock(&self.forward_pass.bind_group) =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward bind group (grown)"),
                layout: &self.forward_pass.bind_group_layout,
                entries: &shaders::bind_group_entries(
                    &shaders::pbr_generated::PBR_RESOURCES,
                    |r| match r.name {
                        "camera" => Some(self.camera_buffer.as_entire_binding()),
                        "per_objects" => Some(per_object.as_entire_binding()),
                        "materials" => Some(material.as_entire_binding()),
                        "lighting" => Some(self.lighting_buffer.as_entire_binding()),
                        "shadow_tex" => {
                            Some(wgpu::BindingResource::TextureView(&self.shadow_array_view))
                        }
                        "csm_tex" => Some(wgpu::BindingResource::TextureView(&self.csm_array_view)),
                        "shadow_sampler" => {
                            Some(wgpu::BindingResource::Sampler(&self.shadow_sampler))
                        }
                        "shadow_cube_tex" => Some(wgpu::BindingResource::TextureView(
                            &self.shadow_cube_array_view,
                        )),
                        "prefilter_cube" | "irradiance_cube" | "brdf_lut" | "ibl_sampler" => {
                            self.ibl.binding(r.name)
                        }
                        _ => self.clusters.entry(r.name),
                    },
                )
                .unwrap_or_default(),
            });
    }

    /// Bind group of the gbuffer layout with `camera` and `per_objects`
    /// supplied by the caller: the compacted culler buffer binds as
    /// `per_objects` for indirect batches (the shared material table is
    /// unchanged). Built on cache miss only — the result is cached in
    /// `indirect_bind_group` and invalidated when the culler is
    /// (re)created or the material table grows.
    pub(crate) fn gbuffer_bind_group_with(
        &self,
        device: &wgpu::Device,
        camera: &wgpu::Buffer,
        per_objects: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let material = read_lock(&self.material_buffer);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gbuffer bind group (per_objects override)"),
            layout: &self.gbuffer_bind_group_layout,
            entries: &shaders::bind_group_entries(
                &shaders::gbuffer_generated::GBUFFER_RESOURCES,
                |r| match r.name {
                    "camera" => Some(camera.as_entire_binding()),
                    "per_objects" => Some(per_objects.as_entire_binding()),
                    "materials" => Some(material.as_entire_binding()),
                    _ => None,
                },
            )
            .unwrap_or_default(),
        })
    }

    /// Ensure the per-object buffer fits `needed` instances, growing and
    /// rebinding when it does not. The lighting frame bind group is built
    /// per frame from the live buffer, so it needs no rebuild here.
    fn ensure_instance_capacity(&self, device: &wgpu::Device, needed: usize) {
        let grown = {
            let old = read_lock(&self.per_object_buffer);
            Self::grown_storage_buffer(
                device,
                "per-object buffer (grown)",
                &old,
                std::mem::size_of::<PerObjectGpu>(),
                needed,
                &self.max_objects,
            )
        };
        if let Some(buffer) = grown {
            *write_lock(&self.per_object_buffer) = buffer;
            self.rebind_storage_buffers(device);
        }
    }

    /// Ensure the material buffer fits `needed` entries, growing and
    /// rebinding when it does not.
    fn ensure_material_capacity(&self, device: &wgpu::Device, needed: usize) {
        let grown = {
            let old = read_lock(&self.material_buffer);
            Self::grown_storage_buffer(
                device,
                "material buffer (grown)",
                &old,
                OPENPBR_MATERIAL_SIZE,
                needed,
                &self.max_materials,
            )
        };
        if let Some(buffer) = grown {
            *write_lock(&self.material_buffer) = buffer;
            self.rebind_storage_buffers(device);
        }
    }

    /// Replace the GPU material table, growing the storage buffer (and
    /// the passes' bind groups) when `materials` exceeds current capacity;
    /// instances reference entries by index.
    ///
    /// The g-buffer material-id layer is `R16Uint` (the spec-guaranteed
    /// multisampleable integer format — see [`MSAA_SAMPLE_COUNT`]), so ids
    /// past `u16::MAX` would truncate: debug builds assert the table fits.
    pub fn upload_materials(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        materials: &[OpenPBRMaterial],
    ) {
        debug_assert!(
            materials.len() <= u16::MAX as usize,
            "material table of {} exceeds the R16Uint id range",
            materials.len()
        );
        self.ensure_material_capacity(device, materials.len());
        let count = materials.len().min(
            self.max_materials
                .load(std::sync::atomic::Ordering::Relaxed) as usize,
        );
        queue.write_buffer(
            &read_lock(&self.material_buffer),
            0,
            bytemuck::cast_slice(&materials[..count]),
        );
    }

    /// Convert and upload instances into the per-object buffer used by
    /// both gbuffer and forward passes, growing the buffer (and rebinding
    /// the passes) when the frame needs more than the current capacity.
    ///
    /// The CPU staging vector reserves `instances.len()` exactly up front,
    /// so the conversion never reallocates mid-frame; a repeated
    /// same-size frame reuses the GPU buffer as-is
    /// (`Self::ensure_instance_capacity` no-ops when the data fits).
    pub fn upload_instances(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceData],
    ) {
        self.upload_instances_with_motion_flags(device, queue, instances, &[], &[]);
    }

    /// Upload instances with per-instance motion-validity states and stable
    /// ECS identities. A missing identity or first-seen entity disables
    /// temporal reuse; CPU-deformed custom meshes do the same through
    /// [`MotionValidity::Invalid`].
    ///
    /// The converted records come back next to the upload, so the caller
    /// can stage the very same records for the GPU culler instead of
    /// converting the frame a second time.
    // qual:allow(iosp) reason: "Keep instance identities, previous transforms, and GPU staging aligned in one frame transition"
    pub(crate) fn upload_instances_with_motion_flags(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        queue.write_buffer(
            &read_lock(&self.per_object_buffer),
            0,
            bytemuck::cast_slice(&gpu_objects),
        );
        gpu_objects
    }

    /// Ensure the palette buffer fits `needed` slots, growing it (and only
    /// it — skinned bind groups are built per draw from the live buffer, so
    /// no pass needs rebinding here) when it does not.
    fn ensure_palette_capacity(&self, device: &wgpu::Device, needed: usize) {
        let grown = {
            let old = read_lock(&self.palette_buffer);
            Self::grown_storage_buffer(
                device,
                "skin palette buffer (grown)",
                &old,
                PALETTE_BYTE_SIZE,
                needed,
                &self.max_palettes,
            )
        };
        if let Some(buffer) = grown {
            *write_lock(&self.palette_buffer) = buffer;
            let slots = u64::from(self.max_palettes.load(std::sync::atomic::Ordering::Relaxed));
            let previous = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("previous skin palette buffer (grown)"),
                size: (PALETTE_BYTE_SIZE as u64) * slots,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            *write_lock(&self.previous_palette_buffer) = previous;
        }
    }

    /// Pack one frame's joint palettes into the palette storage buffer and
    /// return one [`PaletteHandle`] per entry, in order. This compatibility
    /// path has no stable owner identities, so it does not expose previous
    /// palettes for temporal reprojection.
    pub fn upload_skin_palettes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        palettes: &[Vec<u8>],
    ) -> Vec<PaletteHandle> {
        self.upload_skin_palettes_with_entities(device, queue, palettes, &[])
            .0
    }

    /// Keyed palette upload used by extracted GPU-skinned entities. Previous
    /// matrices are matched by entity identity, not dense palette slot, and
    /// newly seen/missing identities receive an invalid-motion state.
    // qual:allow(iosp) reason: "Stage current and previous palettes together so dense slots preserve per-entity motion history"
    pub(crate) fn upload_skin_palettes_with_entities(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        palettes: &[Vec<u8>],
        entities: &[Entity],
    ) -> (Vec<PaletteHandle>, Vec<MotionValidity>) {
        self.ensure_palette_capacity(device, palettes.len());
        let count = palettes
            .len()
            .min(self.max_palettes.load(std::sync::atomic::Ordering::Relaxed) as usize);
        let mut bytes: Vec<u8> = Vec::with_capacity(count * PALETTE_BYTE_SIZE);
        for (slot, palette) in palettes.iter().take(count).enumerate() {
            debug_assert_eq!(
                palette.len(),
                PALETTE_BYTE_SIZE,
                "palette slot {slot} must be one full upload slot"
            );
            bytes.extend_from_slice(&palette[..palette.len().min(PALETTE_BYTE_SIZE)]);
            bytes.resize((slot + 1) * PALETTE_BYTE_SIZE, 0);
        }
        let previous_palettes = read_lock(&self.previous_skin_palettes);
        let identity_complete = entities.len() >= count;
        let (previous_bytes, motion_validity, next_previous) =
            previous_palette_frame(&previous_palettes, &bytes, entities, count);
        drop(previous_palettes);
        if !bytes.is_empty() {
            queue.write_buffer(
                &read_lock(&self.previous_palette_buffer),
                0,
                &previous_bytes,
            );
            queue.write_buffer(&read_lock(&self.palette_buffer), 0, &bytes);
        }
        *write_lock(&self.previous_skin_palettes) = if identity_complete {
            next_previous
        } else {
            std::collections::HashMap::new()
        };
        self.palette_count
            .store(count as u32, std::sync::atomic::Ordering::Relaxed);
        (
            (0..count as u32).map(PaletteHandle::from_raw).collect(),
            motion_validity,
        )
    }

    /// Forget a palette that did not produce a GPU-skinned draw this frame,
    /// so the next frame cannot reproject from a pose that was not rendered.
    pub(crate) fn invalidate_skin_palette_history(&self, entity: Entity) {
        write_lock(&self.previous_skin_palettes).remove(&entity);
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::{previous_instance_state, previous_palette_frame};
    use crate::antialias::MotionValidity;
    use ornis_core::Entity;

    #[test]
    fn previous_instance_transforms_follow_entity_when_dense_order_changes() {
        let entity_a = Entity::new(4);
        let entity_b = Entity::new(9);
        let previous = std::collections::HashMap::from([
            (entity_a, Mat4::from_translation(glam::Vec3::X)),
            (entity_b, Mat4::from_translation(glam::Vec3::Y)),
        ]);
        let current_b = InstanceData {
            model_matrix: Mat4::from_translation(glam::Vec3::new(0.0, 20.0, 0.0)),
            normal_matrix: Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        };
        let (previous_b, validity_b) = previous_instance_state(
            &current_b,
            Some(entity_b),
            &previous,
            MotionValidity::Valid,
            true,
        );
        assert_eq!(
            previous_b.w_axis,
            Mat4::from_translation(glam::Vec3::Y).w_axis
        );
        assert_eq!(validity_b, MotionValidity::Valid);

        let first_seen = Entity::new(10);
        let (previous_new, validity_new) = previous_instance_state(
            &current_b,
            Some(first_seen),
            &previous,
            MotionValidity::Valid,
            true,
        );
        assert_eq!(previous_new, current_b.model_matrix);
        assert_eq!(validity_new, MotionValidity::Invalid);

        // An incomplete identity map (fewer entities than instances)
        // cannot be matched against the history: the motion stays
        // invalid even for an entity the history knows, so the temporal
        // pass never reprojects from an untrusted frame.
        let (previous_incomplete, validity_incomplete) = previous_instance_state(
            &current_b,
            Some(entity_b),
            &previous,
            MotionValidity::Valid,
            false,
        );
        assert_eq!(
            previous_incomplete,
            Mat4::from_translation(glam::Vec3::Y),
            "the stored transform stays, but the motion flag does not trust it"
        );
        assert_eq!(validity_incomplete, MotionValidity::Invalid);
    }

    #[test]
    fn previous_skin_palettes_follow_entity_when_palette_slots_reorder() {
        let entity_a = Entity::new(4);
        let entity_b = Entity::new(9);
        let previous = std::collections::HashMap::from([
            (entity_a, vec![0xAA; PALETTE_BYTE_SIZE]),
            (entity_b, vec![0xBB; PALETTE_BYTE_SIZE]),
        ]);
        let current = [vec![0x11; PALETTE_BYTE_SIZE], vec![0x22; PALETTE_BYTE_SIZE]].concat();
        let (previous_bytes, validity, next) =
            previous_palette_frame(&previous, &current, &[entity_b, entity_a], 2);
        assert_eq!(
            &previous_bytes[..PALETTE_BYTE_SIZE],
            previous[&entity_b].as_slice()
        );
        assert_eq!(
            &previous_bytes[PALETTE_BYTE_SIZE..],
            previous[&entity_a].as_slice()
        );
        assert_eq!(validity, vec![MotionValidity::Valid; 2]);
        assert_eq!(next[&entity_b], vec![0x11; PALETTE_BYTE_SIZE]);
        assert_eq!(next[&entity_a], vec![0x22; PALETTE_BYTE_SIZE]);

        let (_, validity, _) = previous_palette_frame(
            &previous,
            &current[..PALETTE_BYTE_SIZE],
            &[Entity::new(12)],
            1,
        );
        assert_eq!(validity, vec![MotionValidity::Invalid]);
    }

    #[test]
    fn per_object_material_index_substitutes_to_u32() {
        use crate::shaders::interface::{GbufferFragmentInput, GbufferVertexOutput};
        // The DSL substitutes the transparent `MaterialIdx` newtype to WGSL
        // `u32`: the declaration texts are unchanged from the raw-`u32` era.
        assert!(
            PerObjectGpu::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            PerObjectGpu::WGSL_SOURCE
        );
        assert!(
            PerObjectGpu::WGSL_SOURCE.contains("motion_valid: u32"),
            "{}",
            PerObjectGpu::WGSL_SOURCE
        );
        assert_eq!(
            PerObjectGpu::FIELD_NAMES,
            &[
                "model",
                "normal_matrix",
                "previous_model",
                "material_index",
                "motion_valid",
            ]
        );
        assert!(
            GbufferVertexOutput::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            GbufferVertexOutput::WGSL_SOURCE
        );
        assert!(
            GbufferFragmentInput::WGSL_SOURCE.contains("material_index: u32"),
            "{}",
            GbufferFragmentInput::WGSL_SOURCE
        );
        assert!(
            GbufferVertexOutput::WGSL_SOURCE.contains("previous_world_position: vec4<f32>"),
            "{}",
            GbufferVertexOutput::WGSL_SOURCE
        );
        assert!(
            GbufferFragmentInput::WGSL_SOURCE.contains("previous_world_position: vec4<f32>"),
            "{}",
            GbufferFragmentInput::WGSL_SOURCE
        );
        // The expanded record has an explicit 208-byte storage stride;
        // assert the CPU offsets against the generated WGSL declaration.
        assert_eq!(std::mem::size_of::<MaterialIdx>(), 4);
        assert_eq!(std::mem::size_of::<PerObjectGpu>(), 208);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, previous_model), 128);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, material_index), 192);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, motion_valid), 196);
        // The derived declaration validates with naga.
        let mut module = naga::Module::default();
        let handle = PerObjectGpu::naga_add_type(&mut module);
        let naga::TypeInner::Struct { members, span } = &module.types[handle].inner else {
            panic!("PerObjectGpu must lower to a naga struct");
        };
        assert_eq!(*span, 208);
        assert_eq!(members[2].offset, 128);
        assert_eq!(members[3].offset, 192);
        assert_eq!(members[4].offset, 196);
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("PerObjectGpu declaration must validate");
    }

    #[test]
    fn staging_reserve_is_exact_and_stable_at_same_size() {
        // The upload path reserves once up front: exact fit for the
        // frame, and re-reserving the same size never reallocates.
        assert_eq!(staging_capacity_for_instances(300), 300);
        let mut staging: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(300));
        assert!(staging.capacity() >= 300);
        let capacity = staging.capacity();
        staging.reserve(staging_capacity_for_instances(300).saturating_sub(staging.len()));
        assert_eq!(staging.capacity(), capacity, "same size: no realloc");
    }
}
