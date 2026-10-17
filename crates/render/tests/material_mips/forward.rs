//! Textured-forward footprint of the anisotropic material sampler.
//!
//! The upload gate in the parent module builds its own samplers. These
//! tests draw the same tilted plane through `create_render_backend`, so a
//! hardcoded level on the shared `material_sampler` fails.

use super::common::sized::{FrameSpec, render_and_readback};
use ornis_render::{
    AnisotropyLevel, CpuImage, InstanceData, MaterialIdx, MaterialTextureSet, OpenPBRMaterial,
    RenderBackend, RenderBackendConfig, RenderContext, TextureCache, TextureRole, Vertex,
    create_render_backend, upload_vertex_rows,
};

/// Same footprint as `super::tilted_bands_stay_sharp_with_anisotropy_at_default`,
/// drawn by the textured-forward pass (`create_render_backend` + a striped
/// base-color material) instead of the upload sampler's test shader.
///
/// The texture is uploaded once. Its stored sampler is
/// `AnisotropyLevel::DEFAULT`, so a pass that sampled `GpuTexture::sampler`
/// would stay sharp at every config level and fail the `OFF` frame.
/// `OFF` from `RenderBackendConfig::material_anisotropy` must average the
/// bands; `DEFAULT` must keep them. After the `OFF` frame the setter raises
/// the level to 16 and the pass is drawn again: that frame has to match
/// `DEFAULT`. Lowering it back to `OFF` must match the flat frame. Either
/// step fails when an already-built sampler is left in place.
#[test]
fn forward_tilted_bands_follow_backend_anisotropy() {
    // The clip match is pure CPU; a wrong plane fails before any adapter.
    assert_tilted_band_clip();
    let Some((adapter, device, queue)) = super::device() else {
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
        .upload_cached(&device, &queue, &image, TextureRole::BaseColor)
        .unwrap();
    let mut textures = MaterialTextureSet::default();
    textures.set(TextureRole::BaseColor, handle);
    let mesh = band_plane_mesh(&device);
    let scene = BandScene {
        device: &device,
        queue: &queue,
        mesh: &mesh,
        textures: &textures,
        cache: &cache,
    };
    let flat = render_forward_bands(&scene, AnisotropyLevel::OFF);
    let mut sharp_backend = forward_backend(&device, AnisotropyLevel::DEFAULT);
    let sharp = draw_forward_bands(&scene, &mut *sharp_backend);
    // The OFF backend already built its sampler. Raising the level has to
    // replace that sampler, not leave the first one in place.
    let mut raised = forward_backend(&device, AnisotropyLevel::OFF);
    let _ = draw_forward_bands(&scene, &mut *raised);
    raised.set_material_anisotropy(&device, AnisotropyLevel::DEFAULT);
    let raised_frame = draw_forward_bands(&scene, &mut *raised);
    raised.set_material_anisotropy(&device, AnisotropyLevel::OFF);
    let lowered_frame = draw_forward_bands(&scene, &mut *raised);
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
        assert_eq!(
            flat, lowered_frame,
            "lowering the level after the pass exists must rebuild the material sampler"
        );
    } else {
        assert_eq!(
            flat, raised_frame,
            "unsupported anisotropy stays at 1 after the setter too"
        );
        assert_eq!(
            flat, lowered_frame,
            "unsupported anisotropy stays at 1 when the setter lowers the level too"
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

/// Device, mesh and material textures shared by the forward band frames.
struct BandScene<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    mesh: &'a ornis_render::Mesh,
    textures: &'a MaterialTextureSet,
    cache: &'a TextureCache,
}

fn render_forward_bands(scene: &BandScene<'_>, level: AnisotropyLevel) -> Vec<u8> {
    let mut backend = forward_backend(scene.device, level);
    draw_forward_bands(scene, &mut *backend)
}

fn draw_forward_bands(scene: &BandScene<'_>, backend: &mut dyn RenderBackend) -> Vec<u8> {
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([1.0, 1.0, 1.0]);
    material.base.metalness(0.0);
    material.specular.weight(0.0);
    backend.upload_materials(scene.device, scene.queue, &[material]);
    let model = glam::Mat4::IDENTITY;
    backend.upload_instances(
        scene.device,
        scene.queue,
        &[InstanceData {
            model_matrix: model,
            normal_matrix: model,
            material_index: MaterialIdx::from(0u32),
        }],
    );
    backend.set_lights(scene.queue, [1.0, 1.0, 1.0], &[]);
    backend.set_camera(
        scene.queue,
        &tilted_band_view_proj().to_cols_array_2d(),
        [0.0, 0.0, 5.0],
    );
    let spec = FrameSpec {
        size: (128, 128),
        format: wgpu::TextureFormat::Rgba16Float,
    };
    let depth = scene.device.create_texture(&wgpu::TextureDescriptor {
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
    let hdr = render_and_readback(scene.device, scene.queue, &spec, |encoder, target| {
        backend.render_textured_forward(
            RenderContext {
                device: scene.device,
                queue: scene.queue,
                encoder,
                target,
            },
            &depth_view,
            scene.mesh,
            1,
            scene.textures,
            scene.cache,
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

pub(super) fn assert_anisotropic_contrast(flat: &[u8], sharp: &[u8], supported: bool) {
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
