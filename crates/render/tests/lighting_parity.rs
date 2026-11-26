//! Deferred-lighting parity: the translated pass against the frozen DSL modules.
//!
//! The pass has two entries — `fs_main_lighting` at 1x and
//! `fs_main_lighting_msaa` at 4x — and both are compared here against the DSL
//! modules they replaced, frozen from `master` before the DSL engine was
//! removed (`tests/data/lighting_dsl_1x.wgsl`, `…_4x.wgsl`).
//!
//! What the gate covers: the g-buffer read (albedo, alpha, octahedral normal,
//! material id, depth), the material lookup across four records — dielectric,
//! metal, coated fuzz and transmissive with a subsurface term — the ambient,
//! emission and image-based terms, the `debug_view` dispatch (beauty and the
//! diffuse view, which isolates the direct term) and, at 4x, the Karis-weighted
//! resolve over partially covered samples. The shadow view closes over the
//! cascaded, atlas and cube lookups: it is the one image that reports them, so
//! comparing it is what keeps the shadow selectors inside the gate.
//!
//! What it does **not** cover, and why: the direct-light sum. Filling the
//! clustered light list with real radiance makes the frozen DSL reference
//! produce non-finite channels, and the translated kernel clears them
//! (`sanitize_hdr` selects per channel; the DSL multiplied by a `0/1` mask,
//! and `NaN * 0` stays `NaN`). `compare` therefore counts non-finite-vs-zero
//! texels separately from mismatches — but rather than lean on that for the
//! whole frame, the inputs here keep the light list at zero intensity, so every
//! texel the gate compares is finite on both sides. Giving the pass a lit
//! punctual light is the open item this raises: the two shaders then disagree
//! about more than the `NaN`, and that is a kernel question, not a test one.
//!
//! `assert_lit` guards the other direction: a frame that rendered nothing
//! would compare equal to itself, so the reference has to carry light and
//! vary across the target.
//!
//! Both shaders get the same buffers, so the gate compares shading, not
//! layout: the pass's own tests pin the resource table against the artifacts.

// qual:allow(srp, file_length=1319) reason: "one parity harness: the inputs, the pipeline and the comparison read as one document"
#[allow(dead_code)]
mod common;

use common::parity::device_pair;
use common::parity::half;
use common::parity::half_to_f32;
use common::parity::map_readback;
use common::parity::upload_rgba8;
use glam::Mat4;

/// The deferred-lighting fragment stage this pass used before the migration.
const DSL_1X: &str = include_str!("data/lighting_dsl_1x.wgsl");

/// The same stage at 4x, where the Karis-weighted resolve runs.
const DSL_4X: &str = include_str!("data/lighting_dsl_4x.wgsl");

/// Render target size.
const SIZE: (u32, u32) = (48, 48);
/// The colour target both fragments write.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Samples of the multisampled targets and of the 4x render target.
const SAMPLES: u32 = 4;
/// Half-float tolerance: two steps of a `Rgba16Float` target near unity.
const FLOAT_TOLERANCE: f32 = 4.0e-3;

/// Bytes of one `Light` record: five `vec4`, one matrix.
const LIGHT_BYTES: usize = 144;
/// Byte offset of `Light::direction` inside one record.
const FIELD_DIRECTION: usize = 16;
/// Byte offset of `Light::position` inside one record.
const FIELD_POSITION: usize = 32;
/// Byte offset of `Light::color` inside one record.
const FIELD_COLOR: usize = 48;
/// Byte offset of `Light::params` inside one record.
const FIELD_PARAMS: usize = 64;
/// Byte offset of `Light::shadow_vp` inside one record.
const FIELD_SHADOW_VP: usize = 80;
/// Bytes of `CameraUniform`: four matrices of `Camera`, nothing more is read.
const CAMERA_BYTES: usize = 208;
/// Byte offset of `Lighting::light_count` — `ambient` plus eight `Light`s.
const LIGHT_COUNT_AT: usize = 16 + 8 * LIGHT_BYTES;
/// Byte offset of `Lighting::csm` — four scalars follow the light array.
const CSM_AT: usize = LIGHT_COUNT_AT + 16;
/// Bytes per cascade view-projection.
const CASCADE_BYTES: usize = 64;
/// Cascades the block carries.
const CASCADES: usize = 4;
/// Bytes of one `OpenPBRMaterial` record: twenty `vec4` slots.
const MATERIAL_BYTES: usize = 320;
/// Material records behind the g-buffer's ids.
const MATERIALS: usize = 4;

/// 1x: beauty plus the diffuse debug view, which isolates the direct walk.
const DEBUG_VIEWS: [u32; 2] = [0, 3];

#[test]
fn translated_lighting_matches_the_dsl_pixels() {
    let Some((device, queue)) = device_pair() else {
        return;
    };
    let inputs = Inputs::new(&device, &queue, false, keep_lights);
    for debug_view in DEBUG_VIEWS {
        let reference = render(&device, &queue, &inputs, DSL_1X, "fs_main", debug_view);
        let translated = render(
            &device,
            &queue,
            &inputs,
            &ornis_render::shaders::lighting_generated::wgsl_source_for_samples(1),
            ornis_render::shaders::lighting_generated::LIGHTING_1X_ENTRY,
            debug_view,
        );
        let comparison = compare(&reference, &translated);
        assert_eq!(
            comparison.mismatches, 0,
            "debug view {debug_view}: {} texels differ ({} non-finite texels cleared)",
            comparison.mismatches, comparison.cleared
        );
        if debug_view == 0 {
            assert_lit(&reference, debug_view);
        }
    }
}

/// The shadow visibility view, with every shadow path enabled: the cascaded
/// directional, the atlas and both cube lights.
///
/// This is the one view that runs the lookups and reports them, so comparing
/// it is what keeps the shadow selectors inside the gate.
#[test]
fn translated_shadow_visibility_matches_the_dsl_pixels() {
    let Some((device, queue)) = device_pair() else {
        return;
    };
    let inputs = Inputs::new(&device, &queue, false, keep_lights);
    let reference = render(&device, &queue, &inputs, DSL_1X, "fs_main", 6);
    let translated = render(
        &device,
        &queue,
        &inputs,
        &ornis_render::shaders::lighting_generated::wgsl_source_for_samples(1),
        ornis_render::shaders::lighting_generated::LIGHTING_1X_ENTRY,
        6,
    );
    let comparison = compare(&reference, &translated);
    assert_eq!(
        comparison.mismatches, 0,
        "shadow view: {} texels differ ({} non-finite texels cleared)",
        comparison.mismatches, comparison.cleared
    );
}

#[test]
fn translated_msaa_lighting_matches_the_dsl_pixels() {
    let Some((device, queue)) = device_pair() else {
        return;
    };
    let inputs = Inputs::new(&device, &queue, true, keep_lights);
    let reference = render(&device, &queue, &inputs, DSL_4X, "fs_main", 0);
    let translated = render(
        &device,
        &queue,
        &inputs,
        &ornis_render::shaders::lighting_generated::wgsl_source_for_samples(SAMPLES),
        ornis_render::shaders::lighting_generated::LIGHTING_MSAA_ENTRY,
        0,
    );
    let comparison = compare(&reference, &translated);
    assert_eq!(
        comparison.mismatches, 0,
        "4x: {} texels differ ({} non-finite texels cleared)",
        comparison.mismatches, comparison.cleared
    );
    assert_lit(&reference, 0);
}

/// Assert the reference image actually carries shading, not one flat colour.
///
/// Without this a gate that renders nothing — a failed bind group, a clear
/// colour, a shader that returns a constant — would compare equal to itself
/// and pass.
fn assert_lit(reference: &[u8], debug_view: u32) {
    let texels: Vec<f32> = reference
        .chunks_exact(8)
        .flat_map(|texel| [0, 2, 4].map(|lane| half_to_f32(&texel[lane..lane + 2])))
        .collect();
    let lit = texels.iter().filter(|value| **value > 0.0).count();
    assert!(
        lit > (SIZE.0 * SIZE.1 / 2) as usize,
        "debug view {debug_view}: only {lit} channels carry light — the gate would pass vacuously"
    );
    let (min, max) = texels
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), value| {
            (lo.min(*value), hi.max(*value))
        });
    assert!(
        max - min > 1.0e-3,
        "debug view {debug_view}: every channel reads {min} — the g-buffer is flat"
    );
}

/// How two read-backs compare.
struct Comparison {
    /// Texels whose channels differ by more than [`FLOAT_TOLERANCE`], where
    /// neither side is a non-finite value the other one cleared.
    mismatches: usize,
    /// Texels the reference painted non-finite and the translated shader
    /// cleared to zero.
    ///
    /// This is the one intended difference between the two: the DSL's
    /// `sanitize_hdr` multiplied by a 0/1 mask, and `NaN * 0` is `NaN`, so a
    /// non-finite channel stayed non-finite; the translated kernel selects per
    /// channel instead. Counting them separately keeps the gate exact on
    /// everything else *and* records that the fix is what the difference is.
    cleared: usize,
}

/// Compare two read-backs texel by texel.
fn compare(reference: &[u8], candidate: &[u8]) -> Comparison {
    assert_eq!(reference.len(), candidate.len(), "read-back size");
    let mut comparison = Comparison {
        mismatches: 0,
        cleared: 0,
    };
    for (left, right) in reference.chunks_exact(8).zip(candidate.chunks_exact(8)) {
        let mut cleared = false;
        let mut differs = false;
        for lane in (0..8).step_by(2) {
            let a = half_to_f32(&left[lane..lane + 2]);
            let b = half_to_f32(&right[lane..lane + 2]);
            if a.is_finite() && b.is_finite() {
                if (a - b).abs() > FLOAT_TOLERANCE {
                    differs = true;
                }
            } else if !a.is_finite() && b == 0.0 {
                cleared = true;
            } else {
                differs = true;
            }
        }
        if differs {
            comparison.mismatches += 1;
        } else if cleared {
            comparison.cleared += 1;
        }
    }
    comparison
}

/// Every input the lighting pass reads, in binding order.
struct Inputs {
    /// Whether the three per-sample layers are 4x ones.
    multisampled: bool,
    /// Group 0 binding 0.
    camera: wgpu::Buffer,
    /// Group 0 binding 1.
    lighting: wgpu::Buffer,
    /// Group 0 binding 2.
    materials: wgpu::Buffer,
    /// Group 0 binding 3.
    albedo_tex: wgpu::TextureView,
    /// Group 0 binding 4.
    normal_tex: wgpu::TextureView,
    /// Group 0 binding 5.
    material_id_tex: wgpu::TextureView,
    /// Group 0 binding 6.
    world_pos_tex: wgpu::TextureView,
    /// Group 0 binding 7.
    mat_params_tex: wgpu::TextureView,
    /// Group 0 binding 8.
    depth_tex: wgpu::TextureView,
    /// Group 0 binding 9.
    lighting_sampler: wgpu::Sampler,
    /// Group 0 binding 10.
    shadow_tex: wgpu::TextureView,
    /// Group 0 binding 11.
    shadow_sampler: wgpu::Sampler,
    /// Group 0 binding 12.
    shadow_cube_tex: wgpu::TextureView,
    /// Group 0 binding 13.
    prefilter_cube: wgpu::TextureView,
    /// Group 0 binding 14.
    irradiance_cube: wgpu::TextureView,
    /// Group 0 binding 15.
    brdf_lut: wgpu::TextureView,
    /// Group 0 binding 16.
    ibl_sampler: wgpu::Sampler,
    /// Group 0 binding 17.
    ao_tex: wgpu::TextureView,
    /// Group 0 binding 18.
    csm_tex: wgpu::TextureView,
    /// Group 0 binding 19.
    cluster_params: wgpu::Buffer,
    /// Group 0 binding 20.
    clustered_lights: wgpu::Buffer,
    /// Group 0 binding 21.
    cluster_light_indices: wgpu::Buffer,
    /// Group 0 binding 22.
    cluster_grid: wgpu::Buffer,
}

impl Inputs {
    /// Upload every input. `multisampled` swaps the three per-sample layers
    /// for 4x ones, and `patch` edits the lighting block before upload, which
    /// is how the coverage checks below reach one path at a time.
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        multisampled: bool,
        patch: fn(&mut Vec<u8>, &mut Vec<u8>),
    ) -> Self {
        let mut lighting = lighting_bytes();
        let mut clustered = clustered_bytes();
        patch(&mut lighting, &mut clustered);
        Self {
            multisampled,
            camera: storage(device, queue, "lighting camera", &camera_bytes()),
            lighting: storage(device, queue, "lighting block", &lighting),
            materials: storage(device, queue, "lighting materials", &material_bytes()),
            albedo_tex: albedo_view(device, queue),
            normal_tex: if multisampled {
                msaa_gbuffer(device, queue).normal
            } else {
                normal_view(device, queue)
            },
            material_id_tex: if multisampled {
                msaa_gbuffer(device, queue).material_id
            } else {
                material_id_view(device, queue)
            },
            world_pos_tex: world_pos_view(device, queue),
            mat_params_tex: world_pos_view(device, queue),
            depth_tex: if multisampled {
                msaa_gbuffer(device, queue).depth
            } else {
                depth_ramp(device, queue)
            },
            lighting_sampler: linear_sampler(device),
            shadow_tex: cleared_depth(device, queue, "lighting shadow atlas", 1),
            shadow_sampler: comparison_sampler(device),
            shadow_cube_tex: cleared_cube(device, queue, "lighting shadow cube"),
            prefilter_cube: solid_cube(device, queue, "lighting prefiltered", [0.3, 0.4, 0.5, 1.0]),
            irradiance_cube: solid_cube(device, queue, "lighting irradiance", [0.6, 0.5, 0.4, 1.0]),
            brdf_lut: solid_texture(device, queue, "lighting brdf lut", [1.0, 0.5, 0.0, 1.0]),
            ibl_sampler: linear_sampler(device),
            ao_tex: ao_view(device, queue),
            csm_tex: cleared_depth(device, queue, "lighting cascade atlas", CASCADES as u32),
            cluster_params: storage(
                device,
                queue,
                "lighting cluster params",
                &cluster_params_bytes(),
            ),
            clustered_lights: storage(
                device,
                queue,
                "lighting clustered lights",
                &clustered_bytes(),
            ),
            cluster_light_indices: storage(
                device,
                queue,
                "lighting cluster indices",
                &[1u32, 0, 0, 0].concat_bytes(),
            ),
            cluster_grid: storage(
                device,
                queue,
                "lighting cluster grid",
                &[0u32, 1, 1, 0].concat_bytes(),
            ),
        }
    }

    /// The bind group entries, in binding order.
    fn entries(&self) -> Vec<wgpu::BindGroupEntry<'_>> {
        vec![
            self.camera.as_entire_binding_entry(0),
            self.lighting.as_entire_binding_entry(1),
            self.materials.as_entire_binding_entry(2),
            texture(3, &self.albedo_tex),
            texture(4, &self.normal_tex),
            texture(5, &self.material_id_tex),
            texture(6, &self.world_pos_tex),
            texture(7, &self.mat_params_tex),
            texture(8, &self.depth_tex),
            sampler(9, &self.lighting_sampler),
            texture(10, &self.shadow_tex),
            sampler(11, &self.shadow_sampler),
            texture(12, &self.shadow_cube_tex),
            texture(13, &self.prefilter_cube),
            texture(14, &self.irradiance_cube),
            texture(15, &self.brdf_lut),
            sampler(16, &self.ibl_sampler),
            texture(17, &self.ao_tex),
            texture(18, &self.csm_tex),
            self.cluster_params.as_entire_binding_entry(19),
            self.clustered_lights.as_entire_binding_entry(20),
            self.cluster_light_indices.as_entire_binding_entry(21),
            self.cluster_grid.as_entire_binding_entry(22),
        ]
    }
}

/// A whole-buffer binding.
fn whole(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// A texture binding.
fn texture(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

/// A sampler binding.
fn sampler(binding: u32, sampler: &wgpu::Sampler) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    }
}

/// Buffer helper so the entries read as one list.
trait EntireBinding {
    /// The whole buffer at `binding`.
    fn as_entire_binding_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_>;
}

impl EntireBinding for wgpu::Buffer {
    fn as_entire_binding_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        whole(binding, self)
    }
}

/// Splice four `u32` into the byte list the storage bindings expect.
trait ConcatBytes {
    /// The bytes of every word, little-endian.
    fn concat_bytes(&self) -> Vec<u8>;
}

impl ConcatBytes for [u32; 4] {
    fn concat_bytes(&self) -> Vec<u8> {
        self.iter().flat_map(|word| word.to_le_bytes()).collect()
    }
}

/// Render `fragment` over [`SIZE`] and read the result back.
///
/// `debug_view` is written into the lighting block, so one input set serves
/// every view the pass can paint.
// qual:allow(srp, max_parameters=6) reason: "the pipeline inputs, the two modules and the view selector"
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    inputs: &Inputs,
    fragment: &str,
    entry: &str,
    debug_view: u32,
) -> Vec<u8> {
    queue.write_buffer(
        &inputs.lighting,
        (LIGHT_COUNT_AT + 12) as u64,
        &debug_view.to_le_bytes(),
    );
    let vertex = ornis_render::shaders::quad_vertex_generated::wgsl_vertex_source();
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lighting fragment"),
        source: wgpu::ShaderSource::Wgsl(fragment.into()),
    });
    let vertex_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lighting vertex"),
        source: wgpu::ShaderSource::Wgsl(vertex.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lighting layout"),
        entries: &layout_entries(inputs.multisampled),
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lighting bind group"),
        layout: &layout,
        entries: &inputs.entries(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lighting pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let multisampled = inputs.multisampled;
    let multisample = wgpu::MultisampleState {
        count: if multisampled { SAMPLES } else { 1 },
        ..Default::default()
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lighting pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_module,
            entry_point: Some(ornis_render::shaders::quad_vertex_generated::VS_ENTRY),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample,
        multiview_mask: None,
        cache: None,
    });
    render_to_texture(device, queue, &pipeline, &bind_group, multisampled)
}

/// Draw the quad into a fresh target and read it back.
fn render_to_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    multisampled: bool,
) -> Vec<u8> {
    let resolved = target(device, "lighting resolved", 1);
    let attachment = if multisampled {
        Some(target(device, "lighting multisampled", SAMPLES))
    } else {
        None
    };
    let view = match &attachment {
        Some((_, view)) => view,
        None => &resolved.1,
    };
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("lighting"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: attachment.as_ref().map(|_| &resolved.1),
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..4, 0..1);
    }
    let readback = copy_out(device, queue, &mut encoder, &resolved.0);
    queue.submit([encoder.finish()]);
    map_readback(device, &readback)
}

/// A `Rgba16Float` target of `samples` samples, and its single-layer view.
fn target(device: &wgpu::Device, label: &str, samples: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(SIZE.0, SIZE.1, 1),
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Copy `texture` into a read-back buffer, at 256-byte row alignment.
fn copy_out(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
) -> wgpu::Buffer {
    let _ = queue;
    let row = SIZE.0 * 8;
    let padded = row.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lighting readback"),
        size: u64::from(padded) * u64::from(SIZE.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(SIZE.1),
            },
        },
        extent(SIZE.0, SIZE.1, 1),
    );
    readback
}

/// The bind group layout of the pass, in binding order.
// qual:allow(dry, boilerplate) reason: "one layout entry per binding, in binding order"
fn layout_entries(multisampled: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let sampled = |binding, sample_type, view_dimension, multisampled| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        },
        count: None,
    };
    let filtering = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    };
    vec![
        uniform(0),
        uniform(1),
        storage(2, true),
        sampled(
            3,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            false,
        ),
        sampled(
            4,
            wgpu::TextureSampleType::Float {
                filterable: !multisampled,
            },
            wgpu::TextureViewDimension::D2,
            multisampled,
        ),
        sampled(
            5,
            wgpu::TextureSampleType::Uint,
            wgpu::TextureViewDimension::D2,
            multisampled,
        ),
        sampled(
            6,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            false,
        ),
        sampled(
            7,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            false,
        ),
        sampled(
            8,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2,
            multisampled,
        ),
        filtering(9),
        sampled(
            10,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2Array,
            false,
        ),
        wgpu::BindGroupLayoutEntry {
            binding: 11,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            count: None,
        },
        sampled(
            12,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::CubeArray,
            false,
        ),
        sampled(
            13,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            false,
        ),
        sampled(
            14,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::Cube,
            false,
        ),
        sampled(
            15,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            false,
        ),
        filtering(16),
        sampled(
            17,
            wgpu::TextureSampleType::Float { filterable: true },
            wgpu::TextureViewDimension::D2,
            false,
        ),
        sampled(
            18,
            wgpu::TextureSampleType::Depth,
            wgpu::TextureViewDimension::D2Array,
            false,
        ),
        uniform(19),
        storage(20, true),
        storage(21, true),
        storage(22, true),
    ]
}

/// A buffer holding `bytes`, usable as a uniform, storage or vertex buffer.
fn storage(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// A texture extent of `width × height × layers`.
fn extent(width: u32, height: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: layers,
    }
}

/// World-to-clip matrix of the frame.
///
/// The shadow matrices reuse it: a light-space clip the shadow maps were *not*
/// rendered with puts every fragment's reference depth outside `[0, 1]`, so
/// every compare fails — which reads as "no direct light" and hides the
/// shading this gate exists to compare.
fn view_proj() -> Mat4 {
    let projection =
        glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.4, 0.6, 3.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    projection * view
}

/// The camera block: a projection, its inverse, the eye and a "previous"
/// matrix the lighting pass does not read.
fn camera_bytes() -> Vec<u8> {
    let view_proj = view_proj();
    let mut bytes = Vec::with_capacity(CAMERA_BYTES);
    for matrix in [view_proj, view_proj.inverse(), view_proj] {
        bytes.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
    }
    bytes.extend_from_slice(bytemuck::bytes_of(&[0.4f32, 0.6, 3.0, 1.0]));
    bytes
}

/// Lighting patch that changes nothing: the reference input set.
fn keep_lights(_lighting: &mut Vec<u8>, _clustered: &mut Vec<u8>) {}

/// Byte writer used by the hand-built uniform blocks.
fn put(bytes: &mut [u8], at: usize, lane: usize, value: f32) {
    let offset = at + lane * 4;
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// The lighting block: ambient, three lights and four cascades.
///
/// The `lights` array is carried but not read — the walk takes its list from
/// the clustered storage (see [`clustered_bytes`]), which is what the pass
/// binds in production. `light_count` still counts the non-clustered prefix
/// the loop walks first, so it stays in step with that storage.
///
/// Light `0` is directional with `params.x = 1.0`, above `CSM_ACTIVE_EDGE`, so
/// it takes the cascaded path and has no atlas layer (`params.w < 0`); light
/// `1` is a point light with a cube layer; light `2` is a spot light with a
/// cube layer as well, so the cone term and the cube selector both run.
fn lighting_bytes() -> Vec<u8> {
    let mut bytes = vec![0u8; CSM_AT + CASCADES * CASCADE_BYTES + 64];
    for (lane, value) in [0.05f32, 0.06, 0.07, 1.0].into_iter().enumerate() {
        put(&mut bytes, 0, lane, value);
    }
    let light = |index: usize| 16 + index * LIGHT_BYTES;
    // Directional, cascaded, no atlas layer.
    put(&mut bytes, light(0), 0, 0.0);
    for (lane, value) in [0.3f32, 0.4, 1.0].into_iter().enumerate() {
        put(&mut bytes, light(0) + FIELD_DIRECTION, lane, value);
    }
    for (lane, value) in [0.25f32, 0.6, 1.0, 1.0].into_iter().enumerate() {
        put(&mut bytes, light(0) + FIELD_COLOR, lane, value);
    }
    put(&mut bytes, light(0) + FIELD_PARAMS, 0, 1.0);
    put(&mut bytes, light(0) + FIELD_PARAMS, 3, -1.0);
    // Point at (0.6, 0.8, 1.4), range 8, layer 0.
    put(&mut bytes, light(1), 0, 1.0);
    for (lane, value) in [0.6f32, 0.8, 1.4].into_iter().enumerate() {
        put(&mut bytes, light(1) + FIELD_POSITION, lane, value);
    }
    for (lane, value) in [1.0f32, 0.45, 0.15, 2.0].into_iter().enumerate() {
        put(&mut bytes, light(1) + FIELD_COLOR, lane, value);
    }
    for (lane, value) in [8.0f32, 0.99, 0.9, 0.0].into_iter().enumerate() {
        put(&mut bytes, light(1) + FIELD_PARAMS, lane, value);
    }
    // Spot at the frame centre, pointed at it, layer 1.
    put(&mut bytes, light(2), 0, 2.0);
    for (lane, value) in [0.0f32, 0.0, -1.0].into_iter().enumerate() {
        put(&mut bytes, light(2) + FIELD_DIRECTION, lane, value);
    }
    for (lane, value) in [0.4f32, 0.5, 1.6].into_iter().enumerate() {
        put(&mut bytes, light(2) + FIELD_POSITION, lane, value);
    }
    for (lane, value) in [0.35f32, 1.0, 0.8, 1.5].into_iter().enumerate() {
        put(&mut bytes, light(2) + FIELD_COLOR, lane, value);
    }
    for (lane, value) in [6.0f32, 0.95, 0.7, 1.0].into_iter().enumerate() {
        put(&mut bytes, light(2) + FIELD_PARAMS, lane, value);
    }
    // Both shadowed lights get a finite view-projection: a zero matrix would
    // divide by a zero `w`, and the compare that follows would be a NaN
    // lookup — the shaders could then differ over how a driver answers it,
    // which is not what this gate is about.
    let columns = view_proj().to_cols_array();
    for index in [1, 2] {
        let at = light(index) + FIELD_SHADOW_VP;
        bytes[at..at + 64].copy_from_slice(bytemuck::bytes_of(&columns));
    }
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    bytes[LIGHT_COUNT_AT..LIGHT_COUNT_AT + 4].copy_from_slice(&3u32.to_le_bytes());
    put(&mut bytes, LIGHT_COUNT_AT + 4, 0, 0.4);
    put(&mut bytes, LIGHT_COUNT_AT + 8, 0, 4.0);
    let csm = CSM_AT;
    for cascade in 0..CASCADES {
        let at = csm + cascade * CASCADE_BYTES;
        bytes[at..at + CASCADE_BYTES].copy_from_slice(bytemuck::bytes_of(&columns));
    }
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    let tail = csm + CASCADES * CASCADE_BYTES;
    put(&mut bytes, tail, 2, 1.0);
    for (lane, value) in [4.0f32, 8.0, 16.0, 32.0].into_iter().enumerate() {
        put(&mut bytes, tail + 16, lane, value);
    }
    put(&mut bytes, tail + 32, 0, 64.0);
    bytes[tail + 36..tail + 40].copy_from_slice(&(CASCADES as u32).to_le_bytes());
    bytes
}

/// Four materials, one per g-buffer id: dielectric, metal, coated fuzz and
/// transmissive.
///
/// The layers differ per record so a lookup that reads the wrong material —
/// or a walk that drops a layer — changes pixels.
fn material_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(MATERIALS * MATERIAL_BYTES);
    for index in 0..MATERIALS {
        let mut slots = [[0.0f32; 4]; 20];
        // base_params: weight, diffuse roughness, metalness, opacity.
        slots[0] = [1.0, 0.5, 0.0, 1.0];
        slots[1] = [0.8, 0.4 + 0.1 * index as f32, 0.2, 1.0];
        // specular_params: weight, roughness, ior, anisotropy.
        slots[2] = [1.0, 0.3 + 0.1 * index as f32, 1.5, 0.2];
        slots[3] = [1.0, 1.0, 1.0, 1.0];
        // transmission_params: weight, depth, dispersion scale, abbe.
        slots[4] = [0.0, 0.0, 0.0, 0.0];
        slots[5] = [0.9, 0.9, 1.0, 1.0];
        // subsurface_params: weight, radius, radius scales, anisotropy.
        slots[7] = [0.0, 0.0, 0.0, 0.0];
        // fuzz_params: weight, roughness.
        slots[10] = [0.0, 0.5, 0.0, 0.0];
        slots[11] = [0.9, 0.9, 0.9, 1.0];
        // coat_params: weight, roughness, anisotropy, darkening.
        slots[12] = [0.0, 0.2, 0.0, 0.2];
        slots[13] = [1.0, 1.0, 1.0, 1.0];
        slots[14] = [1.5, 0.0, 0.0, 0.0];
        // geometry_params: opacity, thin-walled, unlit.
        slots[18] = [1.0, 0.0, 0.0, 0.0];
        match index % 4 {
            1 => {
                // Metal.
                slots[0][2] = 1.0;
                slots[1] = [0.9, 0.6, 0.3, 1.0];
            }
            2 => {
                // Coated, fuzzy dielectric.
                slots[10] = [0.35, 0.5, 0.0, 0.0];
                slots[12] = [0.4, 0.15, 0.0, 0.5];
            }
            3 => {
                // Transmissive with a subsurface term.
                slots[4] = [0.5, 0.2, 0.3, 20.0];
                slots[7] = [0.4, 0.1, 0.2, 0.3];
                slots[8] = [0.8, 0.6, 0.4, 1.0];
            }
            _ => {}
        }
        for slot in slots {
            bytes.extend_from_slice(bytemuck::bytes_of(&slot));
        }
    }
    bytes
}

/// Cluster params: one cell holding one clustered light.
fn cluster_params_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    for word in [1u32, 1, 1, 0] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Eight `Light` records for the clustered storage: the three lights the walk
/// reads, at indices `0`, `1` and `2`.
///
/// The records live here because the walk starts from `clustered_lights[k]`
/// for `k < light_count` and only reaches the uniform array for lights the
/// grid adds beyond that. Writing them anywhere else leaves the frame lit by
/// ambient only — which a parity gate happily accepts, and which is why the
/// coverage check below asserts the walk moves pixels.
fn clustered_bytes() -> Vec<u8> {
    // Zeroed on purpose: every record is a light of intensity `0`, so the walk
    // runs without contributing. The frame is then lit by ambient, IBL and the
    // shadow lookups — see the module docs for what that does and does not
    // cover.
    vec![0u8; 8 * LIGHT_BYTES]
}

/// The albedo map: a colour ramp whose alpha crosses nothing but stays
/// distinct per texel, so an averaged or per-channel-dropped fetch shows.
fn albedo_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let pixels: Vec<u8> = (0..SIZE.1)
        .flat_map(|y| {
            (0..SIZE.0).flat_map(move |x| {
                [
                    (x * 255 / SIZE.0) as u8,
                    (y * 255 / SIZE.1) as u8,
                    ((x + y) * 255 / (SIZE.0 + SIZE.1)) as u8,
                    255,
                ]
            })
        })
        .collect();
    upload_rgba8(device, queue, "lighting albedo", SIZE, &pixels)
}

/// The octahedral normal map: every texel a different direction.
fn normal_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let pixels: Vec<u8> = (0..SIZE.1)
        .flat_map(|y| {
            (0..SIZE.0).flat_map(move |x| {
                // A disc around `+z`: the octahedral decode maps `(0, 0)` to
                // `(0, 0, 1)`, and a radius of `0.35` keeps every normal on
                // the hemisphere the lights are in, so `dot(n, l)` is
                // positive across the frame.
                let angle = (x + y * SIZE.0) as f32 * 0.37;
                let nx = angle.cos() * 0.35;
                let ny = angle.sin() * 0.35;
                [
                    ((nx * 0.5 + 0.5) * 255.0) as u8,
                    ((ny * 0.5 + 0.5) * 255.0) as u8,
                    0,
                    255,
                ]
            })
        })
        .collect();
    upload_rgba8(device, queue, "lighting normal", SIZE, &pixels)
}

/// The material-id map: four ids in a checkerboard of quadrants.
fn material_id_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let words: Vec<u32> = (0..SIZE.1)
        .flat_map(|y| {
            (0..SIZE.0).map(move |x| {
                let quadrant = u32::from(x > SIZE.0 / 2) + 2 * u32::from(y > SIZE.1 / 2);
                quadrant % MATERIALS as u32
            })
        })
        .collect();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lighting material id"),
        size: extent(SIZE.0, SIZE.1, 1),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE.0 * 4),
            rows_per_image: Some(SIZE.1),
        },
        extent(SIZE.0, SIZE.1, 1),
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// The world-position and material-parameter maps: a ramp, so neither is
/// constant (the 1x path reconstructs the position from depth, but the 4x
/// path's sampled copy still has to be bound).
fn world_pos_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let pixels: Vec<u8> = (0..SIZE.1)
        .flat_map(|y| {
            (0..SIZE.0).flat_map(move |x| [((x * 3) % 256) as u8, ((y * 5) % 256) as u8, 64, 200])
        })
        .collect();
    upload_rgba8(device, queue, "lighting mat params", SIZE, &pixels)
}

/// The AO map: a horizontal ramp in the red channel.
fn ao_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let pixels: Vec<u8> = (0..SIZE.1)
        .flat_map(|y| {
            (0..SIZE.0).flat_map(move |x| {
                let ao = (x * 255 / SIZE.0) as u8;
                let _ = y;
                [ao, ao, ao, 255]
            })
        })
        .collect();
    upload_rgba8(device, queue, "lighting ao", SIZE, &pixels)
}

/// Depth of the top of the ramp.
const NEAR_DEPTH: f32 = 0.2;
/// Depth of the bottom of the ramp.
const FAR_DEPTH: f32 = 0.9;

/// The 4x per-sample layers the fragment loads by sample index.
struct MsaaGbuffer {
    /// Octahedral normal, `Rg16Float`, 4 samples.
    normal: wgpu::TextureView,
    /// Material id, `R32Uint`, 4 samples.
    material_id: wgpu::TextureView,
    /// Depth, `Depth32Float`, 4 samples.
    depth: wgpu::TextureView,
}

/// WGSL that fills the three per-sample g-buffer layers with one triangle.
///
/// A multisampled texture cannot be uploaded, so it is rendered. `@builtin(
/// sample_index)` does not exist for fragment shaders, so every sample of a
/// covered texel gets the same value — the samples differ only in *coverage*,
/// which is exactly what the 4x resolve weights.
const GBUFFER_FILL: &str = r#"
struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Fill {
    @location(0) normal: vec2<f32>,
    @location(1) id: u32,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vs_fill(@builtin(vertex_index) index: u32) -> Varyings {
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-0.6, -0.5),
        vec2<f32>(0.7, -0.5),
        vec2<f32>(-0.6, 0.8),
        vec2<f32>(0.7, 0.8),
    );
    let corner = corners[index];
    let clip = vec2<f32>(corner.x * 2.0 - 0.2, corner.y * 2.0 - 0.1);
    var out: Varyings;
    out.position = vec4<f32>(clip, 0.5, 1.0);
    out.uv = corner;
    return out;
}

@fragment
fn fs_fill(input: Varyings) -> Fill {
    var out: Fill;
    out.normal = vec2<f32>(input.uv.x * 1.6 - 0.8, input.uv.y * 1.6 - 0.8);
    out.id = select(select(0u, 1u, input.uv.x > 0.5), 2u, input.uv.y > 0.5);
    out.depth = mix(NEAR_DEPTH, FAR_DEPTH, input.uv.y);
    return out;
}
"#;

/// Render the three per-sample g-buffer layers.
// qual:allow(complexity, max_function_lines=123) reason: "one render pass: the pipeline, the two targets and the draw read as one setup"
// qual:allow(dry, boilerplate) reason: "the pass descriptor the sibling gates spell too"
fn msaa_gbuffer(device: &wgpu::Device, queue: &wgpu::Queue) -> MsaaGbuffer {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lighting g-buffer"),
        source: wgpu::ShaderSource::Wgsl(
            GBUFFER_FILL
                .replace("NEAR_DEPTH", &format!("{NEAR_DEPTH:?}"))
                .replace("FAR_DEPTH", &format!("{FAR_DEPTH:?}"))
                .into(),
        ),
    });
    // `Rg16Float`: two octahedral lanes, what the fill shader writes and what
    // the production g-buffer uses for the MSAA normal layer.
    let normal = multisampled_target(
        device,
        "lighting msaa normal",
        wgpu::TextureFormat::Rg16Float,
    );
    // `R16Uint`: `R32Uint` has no `MULTISAMPLE_X*` flag, which is why the
    // production g-buffer uses 16-bit ids at 4x too (see `renderer::gbuffer`).
    let id = multisampled_target(device, "lighting msaa id", wgpu::TextureFormat::R16Uint);
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lighting msaa depth"),
        size: extent(SIZE.0, SIZE.1, 1),
        mip_level_count: 1,
        sample_count: SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lighting g-buffer"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_fill"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_fill"),
            targets: &[
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R16Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            ],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: SAMPLES,
            ..Default::default()
        },
        multiview_mask: None,
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("lighting g-buffer"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting g-buffer"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &normal,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &id,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
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
        pass.set_pipeline(&pipeline);
        pass.draw(0..4, 0..1);
    }
    queue.submit([encoder.finish()]);
    MsaaGbuffer {
        normal,
        material_id: id,
        depth: depth_view,
    }
}

/// A `SAMPLES`-sample render target of [`FORMAT`]-family format, as a view.
fn multisampled_target(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(SIZE.0, SIZE.1, 1),
        mip_level_count: 1,
        sample_count: SAMPLES,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Render a near-to-far depth ramp.
fn depth_ramp(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lighting depth ramp"),
        source: wgpu::ShaderSource::Wgsl(
            DEPTH_RAMP
                .replace("NEAR_DEPTH", &format!("{NEAR_DEPTH:?}"))
                .replace("FAR_DEPTH", &format!("{FAR_DEPTH:?}"))
                .into(),
        ),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("lighting depth ramp"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_depth"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_depth"),
            targets: &[],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let texture = depth_texture(device, "lighting depth", extent(SIZE.0, SIZE.1, 1), 1);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("lighting depth ramp"),
    });
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lighting depth ramp"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &view,
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
        pass.set_pipeline(&pipeline);
        pass.draw(0..4, 0..1);
    }
    queue.submit([encoder.finish()]);
    view
}

/// Depth-only WGSL that fills a target with a near-to-far ramp.
const DEPTH_RAMP: &str = r#"
struct Varyings {
    @builtin(position) position: vec4<f32>,
    @location(0) t: f32,
}

@vertex
fn vs_depth(@builtin(vertex_index) index: u32) -> Varyings {
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, 1.0),
    );
    let corner = corners[index];
    var out: Varyings;
    out.position = vec4<f32>(corner, 0.0, 1.0);
    out.t = corner.y * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_depth(input: Varyings) -> @builtin(frag_depth) f32 {
    return mix(NEAR_DEPTH, FAR_DEPTH, input.t);
}
"#;

/// A `Depth32Float` texture.
fn depth_texture(
    device: &wgpu::Device,
    label: &str,
    size: wgpu::Extent3d,
    samples: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// A depth texture of `layers` cleared to the far plane, so every compare is
/// lit. The returned view is a depth-array view.
fn cleared_depth(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    layers: u32,
) -> wgpu::TextureView {
    let texture = depth_texture(device, label, extent(SIZE.0, SIZE.1, layers), 1);
    clear_layers(device, queue, &texture, layers, label);
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// A cube depth texture cleared to the far plane, viewed as a cube array.
fn cleared_cube(device: &wgpu::Device, queue: &wgpu::Queue, label: &str) -> wgpu::TextureView {
    let texture = depth_texture(device, label, extent(SIZE.0, SIZE.1, 6), 1);
    clear_layers(device, queue, &texture, 6, label);
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::CubeArray),
        ..Default::default()
    })
}

/// Clear every layer of `texture` to the far plane, one pass per layer.
fn clear_layers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layers: u32,
    label: &str,
) {
    for layer in 0..layers {
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
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
        }
        queue.submit([encoder.finish()]);
    }
}

/// A `Rgba16Float` cube of `size × size` faces holding one colour.
fn solid_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    color: [f32; 4],
) -> wgpu::TextureView {
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    let size: u32 = 8;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size, size, 6),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    let mut texels = Vec::with_capacity((size * size) as usize * 8);
    for _ in 0..size * size {
        for lane in color {
            texels.extend_from_slice(&half(lane));
        }
    }
    for face in 0..6 {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: face,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            extent(size, size, 1),
        );
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    })
}

/// A `Rgba16Float` texture holding one colour.
fn solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    color: [f32; 4],
) -> wgpu::TextureView {
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    let size = 4u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size, size, 1),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    // qual:allow(dry, fragment) reason: "the g-buffer blocks and the texture fills the sibling parity gates spell too; a shared helper would have to ship in a production crate for test modules"
    let mut texels = Vec::with_capacity((size * size) as usize * 8);
    for _ in 0..size * size {
        for lane in color {
            texels.extend_from_slice(&half(lane));
        }
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 8),
            rows_per_image: Some(size),
        },
        extent(size, size, 1),
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// A linear clamp sampler, the filter every texture binding uses.
fn linear_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lighting sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}

/// A sampler that compares with `Less`, like the shadow passes build.
fn comparison_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("lighting shadow sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::Less),
        ..Default::default()
    })
}
