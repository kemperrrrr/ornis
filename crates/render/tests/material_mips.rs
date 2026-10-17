//! Headless material-texture gates: analytic checker golden and KTX2 mip sampling.
//! CI uses lavapipe; a compression-capable adapter also exercises its native targets.

#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_render::texture_upload::{
    preferred_texture_encoding, texture_compression_features, upload_image,
};
use ornis_render::textures::upload_texture;
use ornis_render::{
    AnisotropyLevel, CpuImage, GpuTexture, InstanceData, Ktx2Image, MaterialIdx,
    MaterialTextureSet, OpenPBRMaterial, RenderBackend, RenderBackendConfig, RenderContext,
    TextureCache, TextureEncoding, TextureImage, TextureRole, Vertex, create_render_backend,
    upload_vertex_rows,
};

const SHADER: &str = r#"
struct Vertex { @builtin(position) position: vec4f, @location(0) uv: vec2f }
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;
@vertex fn plane(@builtin(vertex_index) i: u32) -> Vertex {
    let positions = array<vec4f, 4>(vec4f(-0.875, -0.875, 0.5, 1.0),
        vec4f(0.875, -0.875, 0.5, 1.0), vec4f(-2.0, 6.0, 4.0, 8.0), vec4f(2.0, 6.0, 4.0, 8.0));
    let uvs = array<vec2f, 4>(vec2f(0.0, 0.0), vec2f(32.0, 0.0), vec2f(0.0, 32.0), vec2f(32.0, 32.0));
    return Vertex(positions[i], uvs[i]);
}
@vertex fn quad(@builtin(vertex_index) i: u32) -> Vertex {
    let positions = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    let p = positions[i];
    return Vertex(vec4f(p, 0.5, 1.0), vec2f(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5));
}
@fragment fn filtered(v: Vertex) -> @location(0) vec4f {
    return textureSample(image, image_sampler, v.uv);
}
"#;

fn sample(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &GpuTexture,
    view: &wgpu::TextureView,
    (size, level): (u32, Option<u32>),
) -> Vec<u8> {
    let source = match level {
        Some(mip) => format!(
            "{SHADER}\n@fragment fn selected(v: Vertex) -> @location(0) vec4f {{ return textureSampleLevel(image, image_sampler, v.uv, {mip}.0); }}"
        ),
        None => SHADER.to_owned(),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("material texture pixel gate"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("material texture pixel gate"),
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some(if level.is_some() { "quad" } else { "plane" }),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(if level.is_some() {
                "selected"
            } else {
                "filtered"
            }),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
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
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&image.sampler),
            },
        ],
    });
    render_and_readback(
        device,
        queue,
        &FrameSpec {
            size: (size, size),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        },
        |encoder, target| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.draw(0..4, 0..1);
        },
    )
}

#[test]
fn tilted_checker_matches_linear_light_golden_not_moire() {
    let Some((_, device, queue)) = device() else {
        return;
    };
    let pixels = (0..256 * 256)
        .flat_map(|i| {
            let c = if (i % 256 + i / 256) % 2 == 0 { 0 } else { 255 };
            [c, c, c, 255]
        })
        .collect();
    let image = CpuImage::from_rgba8(256, 256, pixels).unwrap();
    // The analytic golden is the isotropic reference: the checker averages to
    // the linear-light 188 at every level the plane samples, which pins the
    // mip chain, not the anisotropy (the footprint gate below covers that).
    let uploaded = upload_texture(
        &device,
        &queue,
        &image,
        TextureRole::BaseColor,
        AnisotropyLevel::OFF,
    )
    .unwrap();
    assert_eq!(uploaded.texture.mip_level_count(), 9);
    let pixels = sample(&device, &queue, &uploaded, &uploaded.view, (128, None));
    let golden = include_bytes!("data/tilted_checker.rgba");
    assert_eq!(pixels.len(), golden.len());
    let mismatches = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden.as_chunks::<4>().0.iter())
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(&a, &b)| a.abs_diff(b) > 2))
        .count();
    // At most a one-pixel rasterization boundary; interior must remain flat.
    assert!(
        mismatches < 128,
        "{mismatches} pixels differ from the analytic golden"
    );
    let base_only = uploaded.texture.create_view(&wgpu::TextureViewDescriptor {
        mip_level_count: Some(1),
        ..Default::default()
    });
    let control = sample(&device, &queue, &uploaded, &base_only, (128, None));
    let unstable = control
        .as_chunks::<4>()
        .0
        .iter()
        .zip(golden.as_chunks::<4>().0.iter())
        .filter(|(a, b)| b[0] == 188 && a[0].abs_diff(188) > 20)
        .count();
    assert!(
        unstable > 500,
        "base-only control must demonstrate aliasing: {unstable}"
    );
}

/// Footprint gate for the anisotropic material sampler (`#122`): the same
/// tilted plane, but with 32-texel bands along `v` instead of a checker.
///
/// The plane repeats the 256² texture 32 times, so a footprint covers tens of
/// texels horizontally while its vertical extent stays near ten; the band
/// period (64 texels) is already inside one texel from mip 6 on. That makes
/// the two samplers analytically predictable: trilinear filtering picks the
/// level of the *long* footprint axis (≥ 6.2 everywhere in the lower half) and
/// averages the plane to the flat 188 — the same level the golden gate pins.
/// Anisotropic sampling sizes the level by the *short* axis instead (3.2–4.9
/// there) and takes its extra samples along the long one, which is constant
/// inside a band, so those rows keep black/white stripes.
///
/// Both renders are measured on the same interior window (the plane's
/// silhouette is excluded: per-quad derivatives are one-sided there and no
/// sampler's level means anything — the golden gate budgets that same
/// boundary). The gate then asks the anisotropic pass to produce many more
/// band-coloured pixels than the isotropic one, which fails for a level of 1
/// (identical images) as well as for a missing mip chain (both striped).
#[test]
fn tilted_bands_stay_sharp_with_anisotropy_at_default() {
    let Some((adapter, device, queue)) = device() else {
        return;
    };
    let pixels = (0..256 * 256)
        .flat_map(|i| {
            let c = if (i / 256 / 32) % 2 == 0 { 0 } else { 255 };
            [c, c, c, 255]
        })
        .collect();
    let image = CpuImage::from_rgba8(256, 256, pixels).unwrap();
    let isotropic = upload_texture(
        &device,
        &queue,
        &image,
        TextureRole::BaseColor,
        AnisotropyLevel::OFF,
    )
    .unwrap();
    let anisotropic = upload_texture(
        &device,
        &queue,
        &image,
        TextureRole::BaseColor,
        AnisotropyLevel::DEFAULT,
    )
    .unwrap();
    let flat = sample(&device, &queue, &isotropic, &isotropic.view, (128, None));
    let sharp = sample(
        &device,
        &queue,
        &anisotropic,
        &anisotropic.view,
        (128, None),
    );
    let supported = adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::ANISOTROPIC_FILTERING);
    assert_anisotropic_contrast(&flat, &sharp, supported);
}

/// Same footprint as [`tilted_bands_stay_sharp_with_anisotropy_at_default`],
/// drawn by the textured-forward pass (`create_render_backend` + a striped
/// base-color material) instead of the upload sampler's test shader.
///
/// The texture is uploaded at [`AnisotropyLevel::OFF`]. Both frames share
/// that entry, so a pass that sampled [`GpuTexture::sampler`] would stay
/// flat at every config level and fail. `OFF` from
/// [`RenderBackendConfig::material_anisotropy`] must average the bands;
/// `DEFAULT` must keep them. After the `OFF` frame the setter raises the
/// level to 16 and the pass is drawn again: that frame has to match
/// `DEFAULT`, which fails when an already-built sampler is left at `OFF`.
#[test]
fn forward_tilted_bands_follow_backend_anisotropy() {
    // The clip match is pure CPU; a wrong plane fails before any adapter.
    assert_tilted_band_clip();
    let Some((adapter, device, queue)) = device() else {
        return;
    };
    let pixels = (0..256 * 256)
        .flat_map(|i| {
            let c = if (i / 256 / 32) % 2 == 0 { 0 } else { 255 };
            [c, c, c, 255]
        })
        .collect();
    let image = CpuImage::from_rgba8(256, 256, pixels).unwrap();
    // Upload sampler is isotropic on purpose. The frame must not use it.
    let mut cache = TextureCache::new();
    let handle = cache
        .upload_cached(
            &device,
            &queue,
            &image,
            TextureRole::BaseColor,
            AnisotropyLevel::OFF,
        )
        .unwrap();
    let mut textures = MaterialTextureSet::default();
    textures.set(TextureRole::BaseColor, handle);
    let mesh = band_plane_mesh(&device);
    let flat = render_forward_bands(
        &device,
        &queue,
        &mesh,
        &textures,
        &cache,
        AnisotropyLevel::OFF,
    );
    let mut sharp_backend = forward_backend(&device, AnisotropyLevel::DEFAULT);
    let sharp = draw_forward_bands(
        &device,
        &queue,
        &mut *sharp_backend,
        &mesh,
        &textures,
        &cache,
    );
    // The OFF backend already built its sampler. Raising the level has to
    // replace that sampler, not leave the first one in place.
    let mut raised = forward_backend(&device, AnisotropyLevel::OFF);
    let _ = draw_forward_bands(&device, &queue, &mut *raised, &mesh, &textures, &cache);
    raised.set_material_anisotropy(&device, AnisotropyLevel::DEFAULT.get());
    let raised_frame = draw_forward_bands(&device, &queue, &mut *raised, &mesh, &textures, &cache);
    let supported = adapter
        .get_downlevel_capabilities()
        .flags
        .contains(wgpu::DownlevelFlags::ANISOTROPIC_FILTERING);
    assert_anisotropic_contrast(&flat, &sharp, supported);
    if supported {
        assert_eq!(
            sharp, raised_frame,
            "raising the level after the pass exists must rebuild the material sampler"
        );
    } else {
        assert_eq!(
            flat, raised_frame,
            "unsupported anisotropy stays at 1 after the setter too"
        );
    }
}

/// Clip positions of the upload-gate plane, reproduced by one view-projection
/// so the textured-forward vertex stage rasterizes the same quad.
fn tilted_band_view_proj() -> glam::Mat4 {
    glam::Mat4::from_cols(
        glam::Vec4::new(1.75, 0.0, 0.0, 0.0),
        glam::Vec4::new(-1.125, 6.875, 3.5, 7.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(-0.875, -0.875, 0.5, 1.0),
    )
}

fn assert_tilted_band_clip() {
    let view_proj = tilted_band_view_proj();
    let samples = [
        ([0.0, 0.0, 0.0], [-0.875, -0.875, 0.5, 1.0]),
        ([1.0, 0.0, 0.0], [0.875, -0.875, 0.5, 1.0]),
        ([0.0, 1.0, 0.0], [-2.0, 6.0, 4.0, 8.0]),
        ([16.0 / 7.0, 1.0, 0.0], [2.0, 6.0, 4.0, 8.0]),
    ];
    for (position, expected) in samples {
        let clip = view_proj * glam::Vec4::new(position[0], position[1], position[2], 1.0);
        let want = glam::Vec4::from_array(expected);
        assert!(
            (clip - want).length() < 1e-4,
            "clip {clip} != upload-gate plane {want}"
        );
    }
    assert!(
        view_proj.determinant().abs() > 1e-3,
        "the camera uniform inverts this matrix"
    );
}

#[test]
fn tilted_band_clip_matches_the_upload_gate_plane() {
    assert_tilted_band_clip();
}

fn band_plane_mesh(device: &wgpu::Device) -> ornis_render::Mesh {
    let vertex = |position: [f32; 3], uv: [f32; 2]| Vertex {
        position,
        normal: [0.0, 0.0, 1.0],
        uv,
        tangent: [1.0, 0.0, 0.0],
    };
    upload_vertex_rows(
        device,
        &[
            vertex([0.0, 0.0, 0.0], [0.0, 0.0]),
            vertex([1.0, 0.0, 0.0], [32.0, 0.0]),
            vertex([0.0, 1.0, 0.0], [0.0, 32.0]),
            vertex([16.0 / 7.0, 1.0, 0.0], [32.0, 32.0]),
        ],
        &[0, 1, 2, 2, 1, 3],
    )
    .expect("band plane")
}

fn forward_backend(device: &wgpu::Device, level: AnisotropyLevel) -> Box<dyn RenderBackend> {
    let spec = FrameSpec {
        size: (128, 128),
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
    };
    create_render_backend(
        device,
        &RenderBackendConfig {
            surface_config: spec.surface_config(),
            material_anisotropy: level,
            ..RenderBackendConfig::default()
        },
    )
}

fn render_forward_bands(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &ornis_render::Mesh,
    textures: &MaterialTextureSet,
    cache: &TextureCache,
    level: AnisotropyLevel,
) -> Vec<u8> {
    let mut backend = forward_backend(device, level);
    draw_forward_bands(device, queue, &mut *backend, mesh, textures, cache)
}

fn draw_forward_bands(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    backend: &mut dyn RenderBackend,
    mesh: &ornis_render::Mesh,
    textures: &MaterialTextureSet,
    cache: &TextureCache,
) -> Vec<u8> {
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([1.0, 1.0, 1.0]);
    material.base.metalness(0.0);
    material.specular.weight(0.0);
    backend.upload_materials(device, queue, &[material]);
    let model = glam::Mat4::IDENTITY;
    backend.upload_instances(
        device,
        queue,
        &[InstanceData {
            model_matrix: model,
            normal_matrix: model,
            material_index: MaterialIdx::from(0u32),
        }],
    );
    backend.set_lights(queue, [1.0, 1.0, 1.0], &[]);
    backend.set_camera(
        queue,
        &tilted_band_view_proj().to_cols_array_2d(),
        [0.0, 0.0, 5.0],
    );
    let spec = FrameSpec {
        size: (128, 128),
        format: wgpu::TextureFormat::Rgba16Float,
    };
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("forward anisotropy depth"),
        size: wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
    let hdr = render_and_readback(device, queue, &spec, |encoder, target| {
        backend.render_textured_forward(
            RenderContext {
                device,
                queue,
                encoder,
                target,
            },
            &depth_view,
            mesh,
            1,
            textures,
            cache,
        );
    });
    hdr_to_srgb8(&hdr)
}

/// Interior of the lower (stretched) half, eroded two pixels sideways and
/// one vertically. `reference` is the isotropic render: a covered pixel
/// with covered neighbours has the plane's own footprint, not an edge
/// derivative. 128² frames only.
fn plane_interior(reference: &[u8]) -> Vec<usize> {
    const SIZE: usize = 128;
    let covered = |x: usize, y: usize| reference[(y * SIZE + x) * 4] != 0;
    let mut window = Vec::new();
    for y in 64..121 {
        for x in 2..126 {
            let inside = covered(x, y)
                && covered(x - 1, y)
                && covered(x + 1, y)
                && covered(x - 2, y)
                && covered(x + 2, y)
                && covered(x, y - 1)
                && covered(x, y + 1);
            if inside {
                window.push(y * SIZE + x);
            }
        }
    }
    window
}

fn band_pixels(render: &[u8], window: &[usize]) -> usize {
    window
        .iter()
        .filter(|&&i| !(64..=192).contains(&render[i * 4]))
        .count()
}

fn assert_anisotropic_contrast(flat: &[u8], sharp: &[u8], supported: bool) {
    let window = plane_interior(flat);
    let isotropic_bands = band_pixels(flat, &window);
    let anisotropic_bands = band_pixels(sharp, &window);
    assert!(
        window.len() > 1024,
        "the lower half of the plane must cover pixels: {} (isotropic {isotropic_bands}, anisotropic {anisotropic_bands})",
        window.len()
    );
    if !supported {
        assert_eq!(
            flat,
            sharp,
            "unsupported anisotropy must degrade to 1 (window {}, isotropic {isotropic_bands}, anisotropic {anisotropic_bands})",
            window.len()
        );
        eprintln!("SKIP: adapter has no anisotropic filtering");
        return;
    }
    assert!(
        anisotropic_bands > 256,
        "the default level must keep the stretched bands sharp: {anisotropic_bands} of {} \
         pixels are band-coloured (isotropic {isotropic_bands})",
        window.len()
    );
    assert!(
        anisotropic_bands > isotropic_bands * 4,
        "anisotropy must restore contrast the isotropic pass averaged away: \
         {anisotropic_bands} vs {isotropic_bands} band pixels (window {})",
        window.len()
    );
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = u32::from(bits & 0x3ff);
    let out = match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut significand = frac;
            let mut shift = 0u32;
            while significand & 0x400 == 0 {
                significand <<= 1;
                shift += 1;
            }
            significand &= 0x3ff;
            let exponent = (127 - 14 - shift as i32) as u32;
            sign | (exponent << 23) | (significand << 13)
        }
        31 => sign | (0xff << 23) | (frac << 13),
        _ => {
            let exponent = (i32::from(exp) - 15 + 127) as u32;
            sign | (exponent << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

fn linear_to_srgb_byte(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

fn hdr_to_srgb8(raw: &[u8]) -> Vec<u8> {
    raw.chunks_exact(8)
        .flat_map(|pixel| {
            let channel = |offset: usize| {
                let bits = u16::from_le_bytes([pixel[offset], pixel[offset + 1]]);
                linear_to_srgb_byte(f16_to_f32(bits))
            };
            let alpha = {
                let bits = u16::from_le_bytes([pixel[6], pixel[7]]);
                (f16_to_f32(bits).clamp(0.0, 1.0) * 255.0).round() as u8
            };
            [channel(0), channel(2), channel(4), alpha]
        })
        .collect()
}

#[test]
fn half_float_srgb_keeps_the_band_anchors() {
    assert_eq!(linear_to_srgb_byte(f16_to_f32(0x0000)), 0);
    assert_eq!(linear_to_srgb_byte(f16_to_f32(0x3800)), 188);
    assert_eq!(linear_to_srgb_byte(f16_to_f32(0x3c00)), 255);
    let pixel = hdr_to_srgb8(&[0x00, 0x38, 0x00, 0x38, 0x00, 0x38, 0x00, 0x3c]);
    assert_eq!(pixel, vec![188, 188, 188, 255]);
}

#[test]
fn ktx2_authored_mips_sample_on_fallback_and_native_devices() {
    let Some((adapter, device, queue)) = device() else {
        return;
    };
    for bytes in [
        include_bytes!("../../texture/tests/data/uastc.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/etc1s.ktx2").as_slice(),
    ] {
        let image = TextureImage::Ktx2(Ktx2Image::from_bytes(bytes).unwrap());
        // No compression bits were requested even if lavapipe advertises BC.
        verify_ktx_mips(&device, &queue, &image, TextureEncoding::Rgba8);
        let features = texture_compression_features(adapter.features());
        if !features.is_empty() {
            let (native_device, native_queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_features: features,
                    ..Default::default()
                }))
                .unwrap();
            verify_ktx_mips(
                &native_device,
                &native_queue,
                &image,
                preferred_texture_encoding(features),
            );
        }
    }
}

fn verify_ktx_mips(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &TextureImage,
    expected: TextureEncoding,
) {
    let uploaded = upload_image(
        device,
        queue,
        image,
        TextureRole::BaseColor,
        AnisotropyLevel::DEFAULT,
    )
    .unwrap();
    let (edge, height) = image.dimensions();
    assert_eq!(edge, height, "square fixtures");
    assert_eq!(
        uploaded.texture.format(),
        ornis_render::texture_upload::gpu_texture_format(expected, TextureRole::BaseColor)
    );
    let reference = image
        .prepare(TextureRole::BaseColor, TextureEncoding::Rgba8)
        .unwrap();
    assert_eq!(
        uploaded.texture.mip_level_count(),
        reference.levels().len() as u32
    );
    for (mip, rgba) in reference.levels().iter().enumerate() {
        let pixels = sample(
            device,
            queue,
            &uploaded,
            &uploaded.view,
            ((edge >> mip).max(1), Some(mip as u32)),
        );
        let error: u64 = pixels
            .iter()
            .zip(rgba)
            .map(|(&a, &b)| a.abs_diff(b) as u64)
            .sum();
        let mean = error as f64 / rgba.len() as f64;
        let tolerance = if expected == TextureEncoding::Rgba8 {
            1.0
        } else {
            12.0
        };
        assert!(
            mean <= tolerance,
            "{expected:?} mip {mip}: mean channel error {mean}"
        );
    }
    eprintln!(
        "KTX2 {:?}: {} uploaded payload bytes",
        uploaded.texture.format(),
        uploaded.payload_bytes
    );
    let mut cache = ornis_render::TextureCache::new();
    let level = AnisotropyLevel::DEFAULT;
    let a = cache
        .upload_image_cached(device, queue, image, TextureRole::BaseColor, level)
        .unwrap();
    let b = cache
        .upload_image_cached(device, queue, image, TextureRole::BaseColor, level)
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(cache.len(), 1);
    // The level is not part of the dedup key: the frame binds the renderer's
    // material sampler, so a second level must reuse this texture.
    let off = AnisotropyLevel::OFF;
    let other = cache
        .upload_image_cached(device, queue, image, TextureRole::BaseColor, off)
        .unwrap();
    assert_eq!(a, other);
    assert_eq!(cache.len(), 1);
}

fn device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let result = pollster::block_on(common::request_device());
    if result.is_none() {
        assert!(std::env::var_os("CI").is_none(), "CI requires lavapipe");
        eprintln!("SKIP: no wgpu adapter");
    }
    result
}

#[test]
fn native_ktx_mips_fall_back_without_compression_features() {
    let Some((_, device, queue)) = device() else {
        return;
    };
    assert!(texture_compression_features(device.features()).is_empty());
    for bytes in [
        include_bytes!("../../texture/tests/data/bc7-40.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/astc-40.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/etc2-40.ktx2").as_slice(),
    ] {
        let ktx = Ktx2Image::from_bytes(bytes).unwrap();
        assert_ne!(ktx.encoding(), Some(TextureEncoding::Rgba8));
        verify_ktx_mips(
            &device,
            &queue,
            &TextureImage::Ktx2(ktx),
            TextureEncoding::Rgba8,
        );
    }
}

#[test]
fn non_block_aligned_basis_and_native_mips_fall_back_on_compression_device() {
    let Some((adapter, _, _)) = device() else {
        return;
    };
    let features = texture_compression_features(adapter.features());
    if features.is_empty() {
        assert!(
            std::env::var_os("CI").is_none(),
            "lavapipe must exercise the compression-enabled path"
        );
        eprintln!("SKIP: no compression features");
        return;
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: features,
        ..Default::default()
    }))
    .unwrap();
    for bytes in [
        include_bytes!("../../texture/tests/data/uastc-30.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/bc7-30.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/astc-30.ktx2").as_slice(),
        include_bytes!("../../texture/tests/data/etc2-30.ktx2").as_slice(),
    ] {
        let ktx = Ktx2Image::from_bytes(bytes).unwrap();
        assert_eq!(ktx.dimensions(), (30, 30));
        verify_ktx_mips(
            &device,
            &queue,
            &TextureImage::Ktx2(ktx),
            TextureEncoding::Rgba8,
        );
    }
}
