//! Forward-PBR parity: the translated pass against the frozen DSL version.
//!
//! The gate draws one triangle with two punctual lights — a directional one
//! that takes the cascaded-shadow path and a point one that takes the cube
//! path — so the light walk, both shadow selectors and the OpenPBR layer sum
//! all run. The material has a coat and a fuzz layer, and the image-based term
//! reads a solid prefiltered cube, a solid irradiance cube and a
//! `(1, 0)` split-sum LUT, so every additive term is non-zero and a term that
//! dropped out would show.
//!
//! Both shaders get the same buffers, so the gate compares shading, not layout:
//! `pbr_generated`'s own tests pin the uniform offsets against the host records.

// qual:allow(srp, file_length=700) reason: "one parity harness: the inputs, the pipeline and the comparison read as one document"
#[allow(dead_code)]
mod common;

use common::parity::half;
use common::parity::half_to_f32;
use common::parity::map_readback;
use glam::{Mat4, Vec3};

/// The forward-PBR fragment stage this pass used before the migration.
const DSL_FRAGMENT: &str = include_str!("data/pbr_dsl.wgsl");

/// The frozen DSL assembly of the textured entry, captured before the DSL
/// path was retired for this pass.
const TEXTURED_DSL: &str = include_str!("data/textured_pbr_dsl.wgsl");

/// Render target size.
const SIZE: (u32, u32) = (64, 64);
/// The colour target both fragments write.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Half-float tolerance: two steps of a `Rgba16Float` target near unity.
const FLOAT_TOLERANCE: f32 = 4.0e-3;

/// Bytes of one `PerObjectGpu` record: four matrices, then four counters.
const OBJECT_BYTES: usize = 208;
/// Bytes of one `OpenPBRMaterial` record: twenty `vec4` slots.
const MATERIAL_BYTES: usize = 320;
/// Bytes of one `Light` record in the uniform block: five `vec4`, one matrix.
const LIGHT_BYTES: usize = 144;
/// Bytes of `CameraUniform`: two matrices, the eye, the previous matrix.
const CAMERA_BYTES: usize = 208;
/// Byte offset of `Lighting::light_count` — `ambient` plus eight `Light`s.
const LIGHT_COUNT_AT: usize = 16 + 8 * LIGHT_BYTES;
/// Byte offset of `Lighting::csm` — `light_count`, `ibl_weight`,
/// `ibl_max_mip` and `debug_view` follow the array.
const CSM_AT: usize = LIGHT_COUNT_AT + 16;
/// Bytes per cascade view-projection.
const CASCADE_BYTES: usize = 64;
/// Cascades the block carries.
const CASCADES: usize = 4;

#[test]
fn translated_pbr_matches_the_dsl_pixels() {
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let inputs = Inputs::new(&device, &queue);
    let vertex = ornis_render::shaders::pbr_generated::wgsl_vertex_source();
    let reference = render(
        &device,
        &queue,
        &vertex,
        DSL_FRAGMENT,
        "fs_main",
        &inputs,
        false,
    );
    let translated = render(
        &device,
        &queue,
        &vertex,
        &ornis_render::shaders::pbr_generated::wgsl_source(),
        "fs_main",
        &inputs,
        false,
    );
    let mismatches = float_mismatches(&reference, &translated);
    assert_eq!(mismatches, 0, "{mismatches} texels differ");
    let lit = reference
        .chunks_exact(8)
        .filter(|texel| texel.iter().any(|byte| *byte != 0))
        .count();
    assert!(
        lit > (SIZE.0 * SIZE.1 / 2) as usize,
        "only {lit} texels carry a colour — the gate would pass vacuously"
    );
}

/// Same gate for the textured entry: three role maps, one shared sampler.
///
/// The role texels are per-channel distinct on purpose — a white map would
/// hide a swapped or dropped channel the way a white light hid the radiance
/// bug.
#[test]
fn translated_textured_pbr_matches_the_dsl_pixels() {
    let Some((device, queue)) = common::parity::device_pair() else {
        return;
    };
    let inputs = Inputs::new(&device, &queue);
    let vertex = ornis_render::shaders::pbr_generated::wgsl_vertex_source();
    let reference = render(
        &device,
        &queue,
        &vertex,
        TEXTURED_DSL,
        "fs_main_textured",
        &inputs,
        true,
    );
    let translated = render(
        &device,
        &queue,
        &vertex,
        &ornis_render::shaders::material_textures::wgsl_source_textured(),
        ornis_render::shaders::material_textures::TEXTURED_FS_ENTRY,
        &inputs,
        true,
    );
    let mismatches = float_mismatches(&reference, &translated);
    assert_eq!(mismatches, 0, "{mismatches} texels differ");
    // The maps must reach the frame: without them the pass would be the
    // untextured one and the gate would compare two copies of the same thing.
    let plain = render(
        &device,
        &queue,
        &vertex,
        &ornis_render::shaders::pbr_generated::wgsl_source(),
        "fs_main",
        &inputs,
        false,
    );
    assert_ne!(
        reference, plain,
        "the role maps changed nothing — the gate compares two untextured frames"
    );
    let lit = reference
        .chunks_exact(8)
        .filter(|texel| texel.iter().any(|byte| *byte != 0))
        .count();
    assert!(
        lit > (SIZE.0 * SIZE.1 / 2) as usize,
        "only {lit} texels carry a colour — the gate would pass vacuously"
    );
}

/// Every buffer and texture the pass binds.
struct Inputs {
    /// Group 0 binding 0.
    camera: wgpu::Buffer,
    /// Group 0 binding 1: one `PerObjectGpu` for the g-buffer vertex stage.
    objects: wgpu::Buffer,
    /// Group 0 binding 2: one `OpenPBRMaterial`.
    materials: wgpu::Buffer,
    /// Group 0 binding 3.
    lighting: wgpu::Buffer,
    /// Group 0 binding 4.
    shadow_tex: wgpu::TextureView,
    /// Group 0 binding 5.
    shadow_sampler: wgpu::Sampler,
    /// Group 0 binding 6.
    shadow_cube_tex: wgpu::TextureView,
    /// Group 0 binding 7.
    prefilter_cube: wgpu::TextureView,
    /// Group 0 binding 8.
    irradiance_cube: wgpu::TextureView,
    /// Group 0 binding 9.
    brdf_lut: wgpu::TextureView,
    /// Group 0 binding 10.
    ibl_sampler: wgpu::Sampler,
    /// Group 0 binding 11.
    csm_tex: wgpu::TextureView,
    /// Group 0 binding 12.
    cluster_params: wgpu::Buffer,
    /// Group 0 binding 13.
    clustered_lights: wgpu::Buffer,
    /// Group 0 binding 14.
    cluster_light_indices: wgpu::Buffer,
    /// Group 0 binding 15.
    cluster_grid: wgpu::Buffer,
    /// Group 0 binding 16: base-colour map of the textured entry.
    base_color_tex: wgpu::TextureView,
    /// Group 0 binding 17: metallic-roughness map (G roughness, B metal).
    metallic_roughness_tex: wgpu::TextureView,
    /// Group 0 binding 18: emissive map.
    emissive_tex: wgpu::TextureView,
    /// Group 0 binding 19: the one sampler all three roles share.
    material_sampler: wgpu::Sampler,
    /// Positions, normals, uvs and tangents of one triangle.
    vertices: wgpu::Buffer,
}

impl Inputs {
    /// Upload every input, clearing the shadow maps to "fully lit".
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            camera: storage(device, queue, "pbr camera", &camera_bytes()),
            objects: storage(device, queue, "pbr objects", &object_bytes()),
            materials: storage(device, queue, "pbr materials", &material_bytes()),
            lighting: storage(device, queue, "pbr lighting", &lighting_bytes()),
            shadow_tex: cleared_depth(device, queue, "pbr shadow atlas", 1),
            shadow_sampler: comparison_sampler(device),
            shadow_cube_tex: cleared_cube(device, queue, "pbr shadow cube"),
            prefilter_cube: solid_cube(device, queue, "pbr prefiltered", [0.3, 0.4, 0.5, 1.0]),
            irradiance_cube: solid_cube(device, queue, "pbr irradiance", [0.6, 0.5, 0.4, 1.0]),
            brdf_lut: solid_texture(device, queue, "pbr brdf lut", [1.0, 0.0, 0.0, 1.0]),
            ibl_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("pbr ibl sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            csm_tex: cleared_depth(device, queue, "pbr cascade atlas", CASCADES as u32),
            cluster_params: storage(device, queue, "pbr cluster params", &[0u8; 16]),
            clustered_lights: storage(
                device,
                queue,
                "pbr clustered lights",
                &clustered_light_bytes(),
            ),
            cluster_light_indices: storage(device, queue, "pbr cluster indices", &[0u8; 4]),
            cluster_grid: storage(device, queue, "pbr cluster grid", &[0u8; 8]),
            vertices: vertex_buffer(device, queue),
            base_color_tex: solid_texture(device, queue, "pbr base color", [0.8, 0.5, 0.2, 1.0]),
            metallic_roughness_tex: solid_texture(
                device,
                queue,
                "pbr metallic roughness",
                [0.0, 0.7, 0.35, 1.0],
            ),
            emissive_tex: solid_texture(device, queue, "pbr emissive", [0.1, 0.2, 0.3, 1.0]),
            material_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("pbr material sampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
        }
    }

    /// The bind group entries, in binding order.
    fn entries(&self, textured: bool) -> Vec<wgpu::BindGroupEntry<'_>> {
        let mut entries = vec![
            self.camera.as_entire_binding_entry(0),
            self.objects.as_entire_binding_entry(1),
            self.materials.as_entire_binding_entry(2),
            self.lighting.as_entire_binding_entry(3),
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&self.shadow_tex),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&self.shadow_cube_tex),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&self.prefilter_cube),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(&self.irradiance_cube),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(&self.brdf_lut),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::Sampler(&self.ibl_sampler),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: wgpu::BindingResource::TextureView(&self.csm_tex),
            },
            self.cluster_params.as_entire_binding_entry(12),
            self.clustered_lights.as_entire_binding_entry(13),
            self.cluster_light_indices.as_entire_binding_entry(14),
            self.cluster_grid.as_entire_binding_entry(15),
        ];
        if textured {
            entries.extend([
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(&self.base_color_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(&self.metallic_roughness_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(&self.emissive_tex),
                },
                wgpu::BindGroupEntry {
                    binding: 19,
                    resource: wgpu::BindingResource::Sampler(&self.material_sampler),
                },
            ]);
        }
        entries
    }
}

/// Buffer helper: the whole buffer at one binding.
trait EntireBinding {
    /// The whole-buffer resource at `binding`.
    fn as_entire_binding_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_>;
}

impl EntireBinding for wgpu::Buffer {
    fn as_entire_binding_entry(&self, binding: u32) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.as_entire_binding(),
        }
    }
}

/// One frame of camera data: the current and the previous view-projection.
fn camera_bytes() -> Vec<u8> {
    let projection =
        glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 1.0, 0.1, 100.0);
    let view = glam::camera::rh::view::look_at_mat4(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
    let view_proj = projection * view;
    let mut bytes = Vec::with_capacity(CAMERA_BYTES);
    for matrix in [view_proj, view_proj.inverse()] {
        bytes.extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array()));
    }
    bytes.extend_from_slice(bytemuck::bytes_of(&[0.0f32, 0.0, 3.0, 1.0]));
    bytes.extend_from_slice(bytemuck::bytes_of(&view_proj.to_cols_array()));
    bytes
}

/// One object record: the identity transform, material `0`.
fn object_bytes() -> Vec<u8> {
    let model = Mat4::IDENTITY;
    let mut bytes = Vec::with_capacity(OBJECT_BYTES);
    bytes.extend_from_slice(bytemuck::bytes_of(&model.to_cols_array()));
    bytes.extend_from_slice(bytemuck::bytes_of(&model.to_cols_array()));
    bytes.extend_from_slice(bytemuck::bytes_of(&model.to_cols_array()));
    bytes.extend_from_slice(bytemuck::bytes_of(&[0u32, 0, 0, 0]));
    bytes
}

/// One `OpenPBRMaterial`: a coated, fuzzy dielectric.
fn material_bytes() -> Vec<u8> {
    let mut slots = [[0.0f32; 4]; 20];
    // base_params: weight, diffuse roughness, metalness, opacity.
    slots[0] = [1.0, 0.5, 0.0, 1.0];
    slots[1] = [0.8, 0.4, 0.2, 1.0];
    // specular_params: weight, roughness, ior, anisotropy.
    slots[2] = [1.0, 0.4, 1.5, 0.0];
    slots[3] = [1.0, 1.0, 1.0, 1.0];
    // fuzz_params: weight, roughness.
    slots[10] = [0.25, 0.5, 0.0, 0.0];
    slots[11] = [0.9, 0.9, 0.9, 1.0];
    // coat_params: weight, roughness, anisotropy, darkening.
    slots[12] = [0.3, 0.2, 0.0, 0.5];
    slots[13] = [1.0, 1.0, 1.0, 1.0];
    slots[14] = [1.5, 0.0, 0.0, 0.0];
    // geometry_params: opacity, thin-walled, unlit.
    slots[18] = [1.0, 0.0, 0.0, 0.0];
    let mut bytes = Vec::with_capacity(MATERIAL_BYTES);
    for slot in slots {
        bytes.extend_from_slice(bytemuck::bytes_of(&slot));
    }
    bytes
}

/// The lighting block: two lights, an ambient term and four cascades.
///
/// Light `0` is directional with `params.x = 1.0`, which is above
/// `CSM_ACTIVE_EDGE`, so it takes the cascaded path; `params.w = -1.0` keeps it
/// out of the single-layer atlas. Light `1` is a point light with a layer in
/// `params.w`, which takes the cube path.
// qual:allow(dry, fragment) reason: "the block spellings the sibling parity gates repeat; a shared helper would have to ship in a production crate for test modules"
fn lighting_bytes() -> Vec<u8> {
    let mut bytes = vec![0u8; CSM_AT + CASCADES * CASCADE_BYTES + 64];
    let put = |bytes: &mut Vec<u8>, at: usize, lane: usize, value: f32| {
        let offset = at + lane * 4;
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    };
    for (lane, value) in [0.05f32, 0.06, 0.07, 1.0].into_iter().enumerate() {
        put(&mut bytes, 0, lane, value);
    }
    let lights = 16;
    // Light 0: directional, unit intensity, no range cutoff. The channels
    // differ so a lobe scaled by one channel instead of the colour vector
    // shows up in the pixels: with a white light that mistake is invisible.
    const DIRECTIONAL_COLOR: [f32; 3] = [0.25, 0.6, 1.0];
    put(&mut bytes, lights, 0, 0.0);
    put(&mut bytes, lights + 4, 0, -0.4);
    put(&mut bytes, lights + 4, 1, 0.5);
    put(&mut bytes, lights + 4, 2, -1.0);
    for (lane, value) in DIRECTIONAL_COLOR.into_iter().enumerate() {
        put(&mut bytes, lights + 12, lane, value);
    }
    put(&mut bytes, lights + 12, 3, 1.0);
    put(&mut bytes, lights + 16, 0, 1.0);
    put(&mut bytes, lights + 16, 3, -1.0);
    // Light 1: point at -1 on x, range 5, a layer in `params.w`, and a
    // different colour from the directional so neither can stand in for it.
    const POINT_COLOR: [f32; 3] = [1.0, 0.45, 0.15];
    let point = lights + LIGHT_BYTES;
    put(&mut bytes, point, 0, 1.0);
    put(&mut bytes, point + 8, 0, -1.0);
    put(&mut bytes, point + 8, 1, 0.0);
    put(&mut bytes, point + 8, 2, 1.0);
    for (lane, value) in POINT_COLOR.into_iter().enumerate() {
        put(&mut bytes, point + 12, lane, value);
    }
    put(&mut bytes, point + 12, 3, 2.0);
    put(&mut bytes, point + 16, 0, 5.0);
    put(&mut bytes, point + 16, 1, 0.99);
    put(&mut bytes, point + 16, 2, 0.9);
    put(&mut bytes, point + 16, 3, 0.0);
    // light_count, ibl_weight, ibl_max_mip, debug_view.
    // qual:allow(dry, fragment) reason: "the block spellings the sibling parity gates repeat; a shared helper would have to ship in a production crate for test modules"
    bytes[LIGHT_COUNT_AT..LIGHT_COUNT_AT + 4].copy_from_slice(&2u32.to_le_bytes());
    put(&mut bytes, LIGHT_COUNT_AT + 4, 0, 0.4);
    put(&mut bytes, LIGHT_COUNT_AT + 8, 0, 4.0);
    // csm: four cascades, the view axis, the splits and the far split.
    let csm = CSM_AT;
    for cascade in 0..CASCADES {
        let at = csm + cascade * CASCADE_BYTES;
        for lane in 0..4 {
            put(&mut bytes, at + lane * 16, lane, 1.0);
        }
    }
    let tail = csm + CASCADES * CASCADE_BYTES;
    put(&mut bytes, tail, 2, 1.0);
    for (lane, value) in [4.0f32, 8.0, 16.0, 32.0].into_iter().enumerate() {
        put(&mut bytes, tail + 16, lane, value);
    }
    put(&mut bytes, tail + 32, 0, 64.0);
    bytes
}

/// Two `Light` records in the clustered storage: the point light duplicated,
/// because `clustered=0` means the fragment only reads the uniform array.
fn clustered_light_bytes() -> Vec<u8> {
    vec![0u8; 2 * LIGHT_BYTES]
}

/// One interleaved vertex in object space.
struct Vertex {
    /// Object-space position.
    position: [f32; 3],
    /// Object-space normal.
    normal: [f32; 3],
    /// Texture coordinates.
    uv: [f32; 2],
    /// Object-space tangent.
    tangent: [f32; 3],
}

/// A vertex of the front-facing triangle.
const fn vertex(position: [f32; 3], uv: [f32; 2]) -> Vertex {
    Vertex {
        position,
        normal: [0.0, 0.0, 1.0],
        uv,
        tangent: [1.0, 0.0, 0.0],
    }
}

/// One triangle in object space, large enough to cover the frame.
const TRIANGLE: [Vertex; 3] = [
    vertex([-1.2, -1.2, 0.0], [0.0, 0.0]),
    vertex([1.2, -1.2, 0.0], [1.0, 0.0]),
    vertex([0.0, 1.2, 0.0], [0.5, 1.0]),
];

/// Pack [`TRIANGLE`] into one interleaved vertex buffer.
fn vertex_buffer(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Buffer {
    let mut bytes = Vec::new();
    for vertex in TRIANGLE {
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.position));
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.normal));
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.uv));
        bytes.extend_from_slice(bytemuck::bytes_of(&vertex.tangent));
    }
    storage(device, queue, "pbr vertices", &bytes)
}

/// A texture extent of `width × height × layers`.
fn extent(width: u32, height: u32, layers: u32) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: layers,
    }
}

/// A buffer holding `bytes`, usable as a uniform, storage or vertex buffer.
fn storage(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::VERTEX
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytes);
    buffer
}

/// A sampler that compares with `Less`, like the shadow passes build.
fn comparison_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("pbr shadow sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::Less),
        ..Default::default()
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
    let texture = depth_texture(device, label, extent(SIZE.0, SIZE.1, layers));
    clear_layers(device, queue, &texture, layers, label);
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// A cube depth texture cleared to the far plane, viewed as a cube array.
fn cleared_cube(device: &wgpu::Device, queue: &wgpu::Queue, label: &str) -> wgpu::TextureView {
    let texture = depth_texture(device, label, extent(SIZE.0, SIZE.1, 6));
    clear_layers(device, queue, &texture, 6, label);
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::CubeArray),
        ..Default::default()
    })
}

/// A `Depth32Float` texture of the given size.
fn depth_texture(device: &wgpu::Device, label: &str, size: wgpu::Extent3d) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

/// Clear every layer of `texture` to the far plane.
///
/// One pass per layer: a multi-layer attachment view is a multiview pass, and
/// the adapter does not have to enable that feature to clear a shadow map.
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
        clear_depth(device, queue, &view, label);
    }
}

/// Clear a depth view to `1.0` — the far plane, which reads as "lit".
fn clear_depth(device: &wgpu::Device, queue: &wgpu::Queue, view: &wgpu::TextureView, label: &str) {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    let attachment = wgpu::RenderPassDepthStencilAttachment {
        view,
        depth_ops: Some(wgpu::Operations {
            load: wgpu::LoadOp::Clear(1.0),
            store: wgpu::StoreOp::Store,
        }),
        stencil_ops: None,
    };
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[],
        depth_stencil_attachment: Some(attachment),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    drop(_pass);
    queue.submit([encoder.finish()]);
}

/// A `Rgba16Float` cube of `size × size` faces holding one colour.
fn solid_cube(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    color: [f32; 4],
) -> wgpu::TextureView {
    let size: u32 = 8;
    let texture = cube_texture(device, label, size);
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

/// A `Rgba16Float` cube texture of `size × size` faces.
fn cube_texture(device: &wgpu::Device, label: &str, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size, size, 6),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// A `Rgba16Float` texture holding one colour.
fn solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    color: [f32; 4],
) -> wgpu::TextureView {
    let size = 4u32;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: extent(size, size, 1),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
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

/// Vertex attributes of the interleaved mesh.
const ATTRIBUTES: [wgpu::VertexAttribute; 4] = [
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 12,
        shader_location: 1,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x2,
        offset: 24,
        shader_location: 2,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x3,
        offset: 32,
        shader_location: 3,
    },
];

/// Render one frame and return the colour target.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex: &str,
    fragment: &str,
    fragment_entry: &str,
    inputs: &Inputs,
    textured: bool,
) -> Vec<u8> {
    let vs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pbr parity vertex"),
        source: wgpu::ShaderSource::Wgsl(vertex.into()),
    });
    let fs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("pbr parity fragment"),
        source: wgpu::ShaderSource::Wgsl(fragment.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pbr parity"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: Some("vs_main"),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: 44,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &ATTRIBUTES,
            })],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: Some(fragment_entry),
            targets: &[Some(wgpu::ColorTargetState {
                format: FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pbr parity"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &inputs.entries(textured),
    });
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pbr parity target"),
        size: extent(SIZE.0, SIZE.1, 1),
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("pbr parity"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_vertex_buffer(0, inputs.vertices.slice(..));
        pass.draw(0..3, 0..1);
    }
    read_target(device, queue, encoder, &target)
}

/// Copy the colour target into a buffer and unpad its rows.
fn read_target(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    target: &wgpu::Texture,
) -> Vec<u8> {
    const TEXEL: u32 = 8;
    const ROW_ALIGN: u32 = 256;
    let unpadded = SIZE.0 * TEXEL;
    let padded = unpadded.div_ceil(ROW_ALIGN) * ROW_ALIGN;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pbr parity readback"),
        size: u64::from(padded * SIZE.1),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
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
    queue.submit([encoder.finish()]);
    let data = map_readback(device, &readback);
    let mut pixels = vec![0u8; (unpadded * SIZE.1) as usize];
    for row in 0..SIZE.1 as usize {
        let start = row * padded as usize;
        pixels[row * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[start..][..unpadded as usize]);
    }
    pixels
}

/// Count the texels whose float lanes differ by more than the tolerance.
fn float_mismatches(expected: &[u8], actual: &[u8]) -> usize {
    expected
        .chunks_exact(8)
        .zip(actual.chunks_exact(8))
        .filter(|(a, b)| {
            a.chunks_exact(2)
                .zip(b.chunks_exact(2))
                .any(|(x, y)| (half_to_f32(x) - half_to_f32(y)).abs() > FLOAT_TOLERANCE)
        })
        .count()
}
