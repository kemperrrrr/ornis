//! Custom and skinned mesh upload, staging, and draw recording (color and shadow entries) for `Renderer3D`.
use super::gbuffer::GbufferResolveSlot;
use super::*;
/// Upload one inline `MeshDesc::Custom` soup as its own [`Mesh`].
///
/// Per-entity path: each Custom entity owns its vertex/index buffers (the
/// shared sphere mesh never stands in for custom geometry). CPU-side
/// conversion is deduplicated upstream by the extraction's
/// [`crate::mesh_upload::SoupCache`] (identical soups convert once per
/// frame — see [`crate::mesh_upload::SoupHash`] and the per-soup budget
/// on [`crate::mesh_upload::custom_vertices`]), so this function only
/// moves the already-converted arrays into buffers created exactly like
/// [`crate::mesh::create_sphere`].
/// Edge direction stays one-way (`ornis-render` depends on
/// `ornis-mesh-editor`, never the reverse).
///
/// # Errors
///
/// Returns [`crate::mesh_upload::UploadError::EmptyMesh`] when either slice
/// is empty, [`crate::mesh_upload::UploadError::InvalidMesh`] when the soup
/// fails validation — callers skip the entity, never a stub mesh.
pub fn upload_custom_mesh(
    device: &wgpu::Device,
    positions: &[[f32; 3]],
    indices: &[u32],
) -> Result<Mesh, crate::mesh_upload::UploadError> {
    let (vertices, indices) = crate::mesh_upload::custom_vertices(positions, indices)?;
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom mesh vertex buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom mesh index buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    })
}

/// Upload interleaved [`SkinnedVertex`] rows as their own [`Mesh`].
///
/// GPU-draw path for [`SkinningMode::Gpu`](ornis_animation::SkinningMode)
/// entries: the rows come from
/// [`CustomMeshEntry::skinned_gpu_vertices`](crate::extraction::CustomMeshEntry::skinned_gpu_vertices)
/// (bind-pose vertices plus staged influences) and draw through the
/// skinned pipeline, which reads locations 0–5 from this single buffer.
/// Buffers are created exactly like [`upload_custom_mesh`].
///
/// # Errors
///
/// Returns [`crate::mesh_upload::UploadError::EmptyMesh`] when either slice
/// is empty, [`crate::mesh_upload::UploadError::InvalidMesh`] when the
/// index list is not a multiple of three or points past the vertices —
/// callers skip the entry, never a stub mesh.
pub fn upload_skinned_mesh(
    device: &wgpu::Device,
    vertices: &[SkinnedVertex],
    indices: &[u32],
) -> Result<Mesh, crate::mesh_upload::UploadError> {
    use crate::mesh_upload::UploadError;
    use ornis_mesh_editor::MeshError;
    if vertices.is_empty() || indices.is_empty() {
        return Err(UploadError::EmptyMesh);
    }
    if !indices.len().is_multiple_of(TRIANGLE_VERTS) {
        return Err(UploadError::InvalidMesh(
            MeshError::IndexCountNotMultipleOfThree,
        ));
    }
    if indices
        .iter()
        .any(|index| (*index as usize) >= vertices.len())
    {
        return Err(UploadError::InvalidMesh(MeshError::IndexOutOfBounds));
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("skinned mesh vertex buffer"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("skinned mesh index buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    })
}

/// Upload already-converted classic [`Vertex`] rows as their own [`Mesh`].
///
/// Production path for [`CustomMeshEntry`](crate::extraction::CustomMeshEntry)
/// vertices: the extraction's [`SoupCache`](crate::mesh_upload::SoupCache)
/// already converted the soup (normals, box-projection UVs, fallback
/// tangents), so this only moves the rows into buffers created exactly like
/// [`upload_custom_mesh`]. Classic-row counterpart of
/// [`upload_skinned_mesh`] with the same validation shape.
///
/// # Errors
///
/// Returns [`crate::mesh_upload::UploadError::EmptyMesh`] when either slice
/// is empty, [`crate::mesh_upload::UploadError::InvalidMesh`] when the
/// index list is not a multiple of three or points past the vertices —
/// callers skip the entry, never a stub mesh.
pub fn upload_vertex_rows(
    device: &wgpu::Device,
    vertices: &[Vertex],
    indices: &[u32],
) -> Result<Mesh, crate::mesh_upload::UploadError> {
    use crate::mesh_upload::UploadError;
    use ornis_mesh_editor::MeshError;
    if vertices.is_empty() || indices.is_empty() {
        return Err(UploadError::EmptyMesh);
    }
    if !indices.len().is_multiple_of(TRIANGLE_VERTS) {
        return Err(UploadError::InvalidMesh(
            MeshError::IndexCountNotMultipleOfThree,
        ));
    }
    if indices
        .iter()
        .any(|index| (*index as usize) >= vertices.len())
    {
        return Err(UploadError::InvalidMesh(MeshError::IndexOutOfBounds));
    }
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom mesh vertex buffer"),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("custom mesh index buffer"),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Ok(Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        vertex_count: vertices.len() as u32,
    })
}

/// One staged per-entity custom mesh: its GPU buffers plus the instance and
/// the resolved color/depth draw decisions.
///
/// Produced per frame by
/// [`Renderer3D::stage_custom_meshes`] from
/// [`FrameUpload::custom_meshes`](crate::extraction::FrameUpload);
/// consumed by the `*_with_custom` draw methods through
/// [`CustomGbufferDraw`]. Entries that fail upload are skipped before this
/// is built — a staged entry always draws, never a stub.
pub struct StagedCustomMesh {
    /// Per-entity GPU buffers (classic or interleaved skinned rows).
    pub mesh: Mesh,
    /// Model/normal matrices plus the merged-material-table index.
    pub instance: InstanceData,
    /// Color-draw decision (`Gpu` binds the staged palette slot).
    pub draw: SkinnedDraw,
    /// Depth-pre-pass decision (same rule, shadow failure variant).
    pub shadow: SkinnedDraw,
}

/// One custom draw inside a shared g-buffer/shadow/forward pass: the staged
/// mesh plus its slot in the uploaded per-object buffer.
///
/// Slots are dense: sphere instances occupy `0..instance_count`, custom
/// instances follow in extraction order (see
/// [`Renderer3D::stage_custom_meshes`]), so each entry draws the
/// single-instance range `instance_slot..instance_slot + 1` — no per-draw
/// `queue.write_buffer` rewrite of slot 0, which would collapse every draw
/// onto the last-written instance under a single ordered submit.
#[derive(Clone, Copy)]
pub struct CustomGbufferDraw<'a> {
    /// Staged per-entity mesh (classic or skinned rows).
    pub mesh: &'a Mesh,
    /// Per-object slot holding this entry's instance.
    pub instance_slot: u32,
    /// Color-draw decision for this entry.
    pub draw: SkinnedDraw,
    /// Depth-pre-pass decision for this entry.
    pub shadow: SkinnedDraw,
}

/// Draw items for `staged` entries whose instances were uploaded at
/// `base_slot..base_slot + staged.len()` (spheres occupy the slots below
/// `base_slot`). Order-preserving: item `i` borrows staged entry `i`.
pub fn custom_draw_items(
    staged: &[StagedCustomMesh],
    base_slot: u32,
) -> Vec<CustomGbufferDraw<'_>> {
    staged
        .iter()
        .enumerate()
        .map(|(index, entry)| CustomGbufferDraw {
            mesh: &entry.mesh,
            instance_slot: base_slot + index as u32,
            draw: entry.draw,
            shadow: entry.shadow,
        })
        .collect()
}

impl Renderer3D {
    /// Skinned g-buffer pipeline: the palette-blend vertex stage
    /// ([`wgsl_vertex_source_skinned`]) with the classic fragment stage and
    /// the same five MRT targets. The bind-group layout carries the extra
    /// binding-3 palette (see [`GBUFFER_SKINNED_RESOURCES`]); bind groups
    /// are built per draw (the palette slot differs per entry), so only
    /// the pipeline and layout are retained.
    pub(super) fn create_skinned_pipeline(
        device: &wgpu::Device,
        camera_buffer: &wgpu::Buffer,
        per_object_buffer: &wgpu::Buffer,
        material_buffer: &wgpu::Buffer,
        palette_buffer: &wgpu::Buffer,
        sample_count: u32,
    ) -> (wgpu::RenderPipeline, wgpu::BindGroupLayout) {
        // Layout entries come from the skinned pass resource table; the
        // probe bind group below pins the name → buffer mapping (a table
        // gain without an update panics loudly here, never at draw time).
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = GBUFFER_SKINNED_RESOURCES
            .iter()
            .map(|r| shaders::bgl_entry(r, sample_count > 1))
            .collect();
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("skinned gbuffer bind group layout"),
            entries: &bgl_entries,
        });
        // Probe: every table row resolves to a live buffer (same mapping
        // the per-draw groups use).
        let _ = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned gbuffer bind group (probe)"),
            layout: &bind_group_layout,
            entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                "camera" => Some(camera_buffer.as_entire_binding()),
                "per_objects" => Some(per_object_buffer.as_entire_binding()),
                "materials" => Some(material_buffer.as_entire_binding()),
                "palette" => Some(palette_buffer.as_entire_binding()),
                _ => None,
            })
            .unwrap_or_default(),
        });

        let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned gbuffer vertex"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl_vertex_source_skinned())),
        });
        let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skinned gbuffer fragment"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shaders::gbuffer_fragment())),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("skinned gbuffer pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("skinned gbuffer pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some(skinned_entry_point()),
                buffers: &[Some(SkinnedVertex::desc())],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fs_module,
                entry_point: Some(shaders::gbuffer_generated::fs_main::entry_point()),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        // `R16Uint`: the spec-guaranteed multisampleable
                        // integer format (see the `material_id` field docs).
                        format: wgpu::TextureFormat::R16Uint,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rg16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        });

        (pipeline, bind_group_layout)
    }

    /// Stage one frame's custom entries into per-entity GPU meshes.
    ///
    /// Production upload for
    /// [`FrameUpload::custom_meshes`](crate::extraction::FrameUpload)
    /// (called from `RenderSubmit`): GPU-mode entries upload their joint
    /// palette ([`palette_upload_bytes`], one slot per entry, in order)
    /// plus interleaved rows ([`upload_skinned_mesh`]); every other entry
    /// uploads its converted rows ([`upload_vertex_rows`]). Draw decisions
    /// come from [`CustomMeshEntry::draw`](crate::extraction::CustomMeshEntry::draw)
    /// / `shadow_draw` with the uploaded handle — a `MissingPalette` /
    /// `ShadowWithoutPalette` failure degrades to [`SkinnedDraw::Cpu`],
    /// never a panic and never a stub draw (unreachable for extracted
    /// entries — the extraction and upload staging limits agree —
    /// defensive only). Entries whose buffers fail
    /// validation, or GPU entries without interleaved rows, are skipped
    /// (no sphere stub, no invented geometry).
    ///
    /// Caching: per-frame re-upload (correctness fallback). CPU-side
    /// conversion is already deduplicated upstream by the extraction's
    /// [`SoupCache`](crate::mesh_upload::SoupCache) (identical soups
    /// convert once per frame); the GPU buffers here are rebuilt every
    /// frame instead of keyed across frames. A cross-frame GPU cache
    /// (content-hash → shared buffers with eviction) is the follow-up —
    /// it needs buffer sharing plus a growth bound this change
    /// deliberately does not introduce.
    ///
    /// Deterministic: output order matches `entries` order (dense lane
    /// order); no randomness. The caller uploads
    /// `staged.iter().map(|entry| &entry.instance)` contiguously after
    /// the sphere instances and builds [`custom_draw_items`] over the
    /// result, so slots stay dense.
    pub fn stage_custom_meshes(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        entries: &[CustomMeshEntry],
    ) -> Vec<StagedCustomMesh> {
        // Palettes first, in entry order: one upload call stages every
        // GPU-mode palette, so handles come back densely (`0..count`).
        // Entries that cannot stage bytes keep `None` and resolve through
        // the `draw()` / `shadow_draw()` CPU fallback below.
        let palette_bytes: Vec<Option<Vec<u8>>> = entries
            .iter()
            .map(|entry| match entry.skinning {
                SkinningMode::Gpu => entry
                    .joint_palette
                    .as_ref()
                    .and_then(|palette| palette_upload_bytes(palette).ok()),
                SkinningMode::Cpu => None,
            })
            .collect();
        let staged_palettes: Vec<Vec<u8>> = palette_bytes
            .iter()
            .filter_map(|bytes| bytes.clone())
            .collect();
        let handles = self.upload_skin_palettes(device, queue, &staged_palettes);
        // `handles[i]` is the slot of the `i`-th staged palette; walk the
        // entries with a cursor over them so each GPU entry reclaims its
        // own handle in order.
        let mut palette_cursor = 0usize;
        let mut staged = Vec::with_capacity(entries.len());
        for (entry, bytes) in entries.iter().zip(&palette_bytes) {
            let handle = match bytes {
                Some(_) => {
                    let handle = handles.get(palette_cursor).copied();
                    palette_cursor += 1;
                    handle
                }
                None => None,
            };
            let draw = entry.draw(handle).unwrap_or(SkinnedDraw::Cpu);
            let shadow = entry.shadow_draw(handle).unwrap_or(SkinnedDraw::Cpu);
            let mesh = match draw {
                SkinnedDraw::Gpu(_) => match entry.skinned_gpu_vertices() {
                    Some(rows) => upload_skinned_mesh(device, &rows, &entry.indices),
                    None => continue,
                },
                SkinnedDraw::Cpu => upload_vertex_rows(device, &entry.vertices, &entry.indices),
            };
            let Ok(mesh) = mesh else {
                continue;
            };
            staged.push(StagedCustomMesh {
                mesh,
                instance: entry.instance,
                draw,
                shadow,
            });
        }
        staged
    }

    /// Draw one per-entity custom mesh with its own instance.
    ///
    /// The caller uploads the merged material table once (a custom entry's
    /// `instance.material_index` points into it), draws the shared sphere
    /// batch first, then calls this once per custom entry: the single
    /// instance is uploaded into slot 0 and `mesh` is drawn with count 1.
    /// Rewriting slot 0 per entry is why shared draws must come first;
    /// per-frame cost is one small upload + one draw per custom entity
    /// (no refit budget yet — see PLAN §h).
    pub fn render_custom_entry(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance: &InstanceData,
    ) {
        self.upload_instances(device, queue, std::slice::from_ref(instance));
        self.render_gbuffer(encoder, g, mesh, 1);
    }

    /// Draw one per-entity skinned mesh through the skinned pipeline.
    ///
    /// GPU-draw path for one entry: `mesh` holds interleaved
    /// [`SkinnedVertex`] rows (see [`upload_skinned_mesh`]), `instance`
    /// goes into per-object slot 0 (shared draws must come first, same as
    /// [`render_custom_entry`](Self::render_custom_entry)), and `handle`
    /// selects the palette slot uploaded by
    /// [`upload_skin_palettes`](Self::upload_skin_palettes) for binding 3.
    /// A stale handle (at or past the last staged count) records no
    /// commands — exact no-op, never an out-of-bounds bind. The fragment
    /// stage and targets match [`render_gbuffer`](Self::render_gbuffer),
    /// so Cpu and Gpu draws land in the same buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn render_skinned_entry(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        g: &GbufferTargets<'_>,
        mesh: &Mesh,
        instance: &InstanceData,
        handle: PaletteHandle,
    ) {
        if handle.index()
            >= self
                .palette_count
                .load(std::sync::atomic::Ordering::Relaxed) as usize
        {
            return;
        }
        self.upload_instances(device, queue, std::slice::from_ref(instance));
        let palette = read_lock(&self.palette_buffer);
        let per_object = read_lock(&self.per_object_buffer);
        let material = read_lock(&self.material_buffer);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skinned gbuffer bind group"),
            layout: &self.skinned_bind_group_layout,
            entries: &shaders::bind_group_entries(&GBUFFER_SKINNED_RESOURCES, |r| match r.name {
                "camera" => Some(self.camera_buffer.as_entire_binding()),
                "per_objects" => Some(per_object.as_entire_binding()),
                "materials" => Some(material.as_entire_binding()),
                "palette" => Some(wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &palette,
                    offset: handle.byte_offset(),
                    size: std::num::NonZeroU64::new(PALETTE_BYTE_SIZE as u64),
                })),
                _ => None,
            })
            .unwrap_or_default(),
        });
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("skinned gbuffer pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: g.albedo,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Albedo),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.normal,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::Normal),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_id,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.world_position,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::WorldPosition),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: g.material_params,
                    depth_slice: None,
                    resolve_target: self.gbuffer_resolve_target(GbufferResolveSlot::MaterialParams),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: g.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        rpass.set_pipeline(&self.skinned_pipeline);
        rpass.set_bind_group(0, &bind_group, &[]);
        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        rpass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }
}
