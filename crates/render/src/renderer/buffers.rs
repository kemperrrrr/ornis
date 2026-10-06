//! Growable GPU stores: per-object instances, materials, and skinning palettes with their capacity management and upload entry points.
use super::*;
/// GPU per-instance record mirroring CPU [`InstanceData`] with padding to 16 bytes.
///
/// The WGSL `PerObject` declaration is generated from this layout
/// ([`PerObjectGpu::WGSL_SOURCE`]); the field list here is the single source
/// of truth for the buffer layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, WgslStruct)]
#[wgsl(name = "PerObject")]
pub struct PerObjectGpu {
    /// Local-to-world matrix.
    pub model: [[f32; 4]; 4],
    /// Inverse-transpose model matrix (normal transform).
    pub normal_matrix: [[f32; 4]; 4],
    /// Index into the material buffer uploaded by `upload_materials`.
    pub material_index: MaterialIdx,
    /// Aligns the record to 16-byte stride (padding: not shader-visible).
    #[wgsl(skip)]
    _padding: [u32; 3],
}

/// Index into the deduplicated material table ([`FrameUpload::materials`]).
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
    /// rebuilt together whenever either buffer moves.
    fn rebind_storage_buffers(&self, device: &wgpu::Device) {
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
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
                        "shadow_sampler" => {
                            Some(wgpu::BindingResource::Sampler(&self.shadow_sampler))
                        }
                        "shadow_cube_tex" => Some(wgpu::BindingResource::TextureView(
                            &self.shadow_cube_array_view,
                        )),
                        "prefilter_cube" | "irradiance_cube" | "brdf_lut" | "ibl_sampler" => {
                            self.ibl.binding(r.name)
                        }
                        _ => None,
                    },
                )
                .unwrap_or_default(),
            });
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
    /// ([`Self::ensure_instance_capacity`] no-ops when the data fits).
    pub fn upload_instances(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[InstanceData],
    ) {
        self.ensure_instance_capacity(device, instances.len());
        let count = instances
            .len()
            .min(self.max_objects.load(std::sync::atomic::Ordering::Relaxed) as usize);
        let mut gpu_objects: Vec<PerObjectGpu> =
            Vec::with_capacity(staging_capacity_for_instances(count));
        for inst in instances.iter().take(count) {
            let model_arr: [[f32; VEC4_COMPONENTS]; VEC4_COMPONENTS] =
                inst.model_matrix.to_cols_array_2d();
            let normal_arr: [[f32; VEC4_COMPONENTS]; VEC4_COMPONENTS] =
                inst.normal_matrix.to_cols_array_2d();
            gpu_objects.push(PerObjectGpu {
                model: model_arr,
                normal_matrix: normal_arr,
                material_index: inst.material_index,
                _padding: [0; VEC3_COMPONENTS],
            });
        }
        queue.write_buffer(
            &read_lock(&self.per_object_buffer),
            0,
            bytemuck::cast_slice(&gpu_objects),
        );
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
        }
    }

    /// Pack one frame's joint palettes into the palette storage buffer and
    /// return one [`PaletteHandle`] per entry, in order.
    ///
    /// Each palette must be one full [`PALETTE_BYTE_SIZE`] slot (see
    /// [`crate::skinning::palette_upload_bytes`]): short slots are
    /// zero-padded (padding is never indexed), over-long slots are
    /// truncated — both defensively, producers stage exact slots. The
    /// buffer grows when the frame needs more slots than the current
    /// capacity; [`Renderer3D::palette_count`] records the staged count so
    /// stale handles record no commands instead of binding out of range.
    pub fn upload_skin_palettes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        palettes: &[Vec<u8>],
    ) -> Vec<PaletteHandle> {
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
        queue.write_buffer(&read_lock(&self.palette_buffer), 0, &bytes);
        self.palette_count
            .store(count as u32, std::sync::atomic::Ordering::Relaxed);
        (0..count as u32).map(PaletteHandle::from_raw).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
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
        assert_eq!(
            PerObjectGpu::FIELD_NAMES,
            &["model", "normal_matrix", "material_index"]
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
        // CPU layout is unchanged: transparent wrapper, same offsets/size.
        assert_eq!(std::mem::size_of::<MaterialIdx>(), 4);
        assert_eq!(std::mem::size_of::<PerObjectGpu>(), 144);
        assert_eq!(std::mem::offset_of!(PerObjectGpu, material_index), 128);
        // The derived declaration validates with naga.
        let mut module = naga::Module::default();
        let handle = PerObjectGpu::naga_add_type(&mut module);
        let naga::TypeInner::Struct { members, span } = &module.types[handle].inner else {
            panic!("PerObjectGpu must lower to a naga struct");
        };
        assert_eq!(*span, 144);
        assert_eq!(members[2].offset, 128);
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
