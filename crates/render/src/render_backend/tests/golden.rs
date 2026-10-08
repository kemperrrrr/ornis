//! Golden-frame probe pinning the full-scene GPU output against the checked-in snapshot.

use super::super::*;
use super::*;

/// Golden-frame: offscreen 1280×720 render of `assets/scene.ron` (5 entities)
/// via `RenderBackend` — pinning regressions of the real GPU pipeline.
///
/// Compares the current frame pixel-by-pixel against the checked-in
/// `crates/render/tests/data/golden_probe_1280x720.png` (recaptured on
/// lavapipe after the single-ACES / thin-film / coat-darkening fixes,
/// via `cargo run -p ornis-render --example render_probe`). Cross-driver
/// noise is tolerated explicitly: per-channel drift ≤4 plus a ≤1px shift
/// match against the golden neighborhood; more than 64 pixels outside
/// both windows fail the gate. Skipped when no adapter is available
/// (CI without GPU).
#[test]
fn golden_full_scene_probe_matches_snapshot() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping golden probe");
        return;
    };

    // ── Scene RON ───────────────────────────────────────────────
    let ron_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/scene.ron");
    let ron = std::fs::read_to_string(&ron_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", ron_path.display()));
    let scene = ornis_assets::scene::Scene::from_ron(&ron).expect("parse assets/scene.ron");

    // ── Gold PNG → bytes ───────────────────────────────────────
    let gold_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/golden_probe_1280x720.png");
    let gold_bytes =
        std::fs::read(&gold_path).unwrap_or_else(|e| panic!("read {}: {e}", gold_path.display()));
    let mut decoder = png::Decoder::new(std::io::Cursor::new(&gold_bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png output size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.width, 1280);
    assert_eq!(info.height, 720);
    let gold_pixels = buf[..info.buffer_size()].to_vec();

    // ── Render current frame 1280×720 offscreen ─────────────────
    const W: u32 = 1280;
    const H: u32 = 720;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("golden probe target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    };
    let backend_config = RenderBackendConfig {
        surface_config: surface_config.clone(),
        sample_count: 1,
        exposure: 1.0,
        max_objects: 256,
        max_materials: 64,
    };
    let mut backend = create_render_backend(&device, &backend_config);

    // Build mesh/materials/instances as in render_probe::build_scene_data.
    let first = scene.entities.first().expect("scene has entities");
    let mesh = match &first.mesh {
        ornis_assets::scene::MeshDesc::Sphere {
            radius,
            segments,
            rings,
        } => crate::mesh::create_sphere(&device, radius.get(), *segments, *rings),
        ornis_assets::scene::MeshDesc::Box { size } => {
            crate::mesh::create_box(&device, size.map(ornis_core::units::PositiveF32::get))
        }
        ornis_assets::scene::MeshDesc::Plane { size } => {
            crate::mesh::create_plane(&device, size.map(ornis_core::units::PositiveF32::get))
        }
        ornis_assets::scene::MeshDesc::Quad { size } => {
            crate::mesh::create_quad(&device, size.map(ornis_core::units::PositiveF32::get))
        }
        ornis_assets::scene::MeshDesc::Cylinder {
            radius,
            height,
            radial_segments,
        } => crate::mesh::create_cylinder(&device, radius.get(), height.get(), *radial_segments),
        // The golden probe renders procedural scenes; Custom soups
        // have no upload path here yet.
        ornis_assets::scene::MeshDesc::Custom { .. } => {
            panic!("Custom mesh not supported by this probe")
        }
    };
    let mut materials = Vec::new();
    let mut instances = Vec::new();
    for (i, ent) in scene.entities.iter().enumerate() {
        let mat = match &ent.material {
            ornis_assets::scene::MaterialDesc::Dielectric {
                base_color,
                roughness,
                ..
            } => {
                let mut m = ornis_core::OpenPBRMaterial::dielectric();
                m.base.color_rgb(*base_color);
                m.specular.roughness(roughness.get());
                m
            }
            ornis_assets::scene::MaterialDesc::Metal {
                base_color,
                roughness,
                ..
            } => {
                let mut m = ornis_core::OpenPBRMaterial::metal();
                m.base.color_rgb(*base_color);
                m.specular.roughness(roughness.get());
                m
            }
            ornis_assets::scene::MaterialDesc::Coat {
                base_color,
                coat_weight,
                coat_roughness,
                ..
            } => {
                let mut m = ornis_core::OpenPBRMaterial::coat();
                m.base.color_rgb(*base_color);
                m.coat.weight(coat_weight.get());
                m.coat.roughness(coat_roughness.get());
                m
            }
            ornis_assets::scene::MaterialDesc::Matte {
                base_color,
                roughness,
                ..
            } => {
                let mut m = ornis_core::OpenPBRMaterial::dielectric();
                m.base.color_rgb(*base_color);
                m.base.diffuse_roughness(roughness.get());
                m.specular.weight(0.0);
                m
            }
            ornis_assets::scene::MaterialDesc::Glass {
                base_color,
                roughness,
                ior,
                ..
            } => {
                let mut m = ornis_core::OpenPBRMaterial::glass();
                m.transmission.color_rgb(*base_color);
                m.specular.roughness(roughness.get());
                m.specular.ior(ior.get());
                m
            }
            ornis_assets::scene::MaterialDesc::Unlit { color } => {
                // Mirrors the extraction mapping (the golden scene has no
                // unlit entities; this arm is exhaustiveness only).
                let mut m = ornis_core::OpenPBRMaterial::dielectric();
                m.base.color_rgb([0.0, 0.0, 0.0]);
                m.base.weight(0.0);
                m.specular.weight(0.0);
                m.base.metalness(0.0);
                let peak = color[0].max(color[1]).max(color[2]).max(0.0);
                if peak > 0.0 {
                    m.emission.luminance(peak);
                    m.emission
                        .color_rgb([color[0] / peak, color[1] / peak, color[2] / peak]);
                }
                m.geometry
                    .set_shading(ornis_core::material::ShadingMode::Unlit);
                m
            }
        };
        materials.push(mat);
        let t = &ent.transform;
        let model =
            glam::Mat4::from_scale_rotation_translation(t.scale, t.rotation.get(), t.translation);
        instances.push(crate::renderer::InstanceData {
            model_matrix: model,
            normal_matrix: model.inverse().transpose(),
            material_index: crate::renderer::MaterialIdx::from(i as u32),
        });
    }
    backend.upload_materials(&device, &queue, &materials);
    backend.upload_instances(&device, &queue, &instances);
    backend.set_lights(&queue, scene.ambient, &scene.lights);
    let (view, proj) = {
        let cam = &scene.camera;
        let aspect = W as f32 / H as f32;
        let view = glam::camera::rh::view::look_at_mat4(cam.position, cam.target, cam.up.get());
        let proj = glam::camera::rh::proj::directx::perspective(
            cam.fov.to_radians().get(),
            aspect,
            cam.near.get(),
            cam.far.get(),
        );
        (view, proj)
    };
    let view_proj = (proj * view).to_cols_array_2d();
    backend.set_camera(&queue, &view_proj, scene.camera.position.to_array());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("golden probe encoder"),
    });
    backend.render_scene(
        RenderContext {
            device: &device,
            queue: &queue,
            encoder: &mut encoder,
            target: &target_view,
        },
        &mesh,
        instances.len() as u32,
    );

    // Read-back (same logic as in render_probe::read_back_pixels).
    let bpp = 4u32;
    let unpadded = W * bpp;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("golden readback"),
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = slice.get_mapped_range().unwrap();
    let mut pixels = vec![0u8; (unpadded * H) as usize];
    for y in 0..H as usize {
        pixels[y * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
    }
    drop(data);
    readback.unmap();

    // ── Compare ────────────────────────────────────────────────
    //
    // The golden was captured on Apple M1; CI renders on lavapipe
    // (software Vulkan). Different drivers round the sRGB transfer
    // differently (±~4 per channel) and rasterize edges up to one
    // pixel apart. Both classes are driver noise, not engine
    // regressions, so the gate tolerates them explicitly:
    //
    //   1. a pixel matches if every channel is within TOL of the
    //      golden pixel at the same position, OR of any golden
    //      pixel in its 3×3 neighborhood (≤1px shift tolerance);
    //   2. anything still unmatched is a real drift candidate —
    //      the frame fails when more than MAX_UNMATCHED remain.
    //
    // Calibrated against run 35240656602: 562 drifting bytes at
    // Δ≤3, 516 of them matching a shifted golden neighbor. A real
    // pipeline regression (wrong lighting, geometry, materials)
    // moves thousands of pixels far beyond these windows.
    const TOL: u8 = 4;
    const MAX_UNMATCHED: usize = 64;
    assert_eq!(pixels.len(), gold_pixels.len());
    let row = W as usize;
    let ch = bpp as usize;
    let off = |x: usize, y: usize| (y * row + x) * ch;
    let within_tol = |pix: &[u8], gold: &[u8]| -> bool {
        pix.iter()
            .zip(gold.iter())
            .all(|(a, b)| a.abs_diff(*b) <= TOL)
    };
    let mut max_diff: u8 = 0;
    for (a, b) in pixels.iter().zip(gold_pixels.iter()) {
        max_diff = max_diff.max(a.abs_diff(*b));
    }
    let mut shifted_matches = 0usize;
    let mut unmatched_total = 0usize;
    let mut first_unmatched: Vec<(usize, usize, Vec<u8>, Vec<u8>)> = Vec::new();
    for y in 0..H as usize {
        for x in 0..W as usize {
            let pix = &pixels[off(x, y)..off(x, y) + ch];
            if within_tol(pix, &gold_pixels[off(x, y)..off(x, y) + ch]) {
                continue;
            }
            let x_lo = x.saturating_sub(1);
            let x_hi = (x + 1).min(W as usize - 1);
            let y_lo = y.saturating_sub(1);
            let y_hi = (y + 1).min(H as usize - 1);
            let shifted = (y_lo..=y_hi).any(|yy| {
                (x_lo..=x_hi).any(|xx| within_tol(pix, &gold_pixels[off(xx, yy)..off(xx, yy) + ch]))
            });
            if shifted {
                shifted_matches += 1;
            } else {
                unmatched_total += 1;
                if first_unmatched.len() < 16 {
                    first_unmatched.push((
                        x,
                        y,
                        pix.to_vec(),
                        gold_pixels[off(x, y)..off(x, y) + ch].to_vec(),
                    ));
                }
            }
        }
    }
    eprintln!(
        "golden probe: max_diff={max_diff} shifted_matches={shifted_matches} \
         unmatched={unmatched_total}"
    );
    assert!(
        unmatched_total <= MAX_UNMATCHED,
        "golden frame drifted: {unmatched_total} pixels differ by >{TOL}/channel even after \
         1px shift tolerance (max_diff={max_diff}); first unmatched={first_unmatched:?} — \
         update tests/data/golden_probe_1280x720.png via render_probe if change is intentional"
    );
    // Sanity check: center is not black, as in probe log [52,52,186].
    let center_off = ((H / 2 * W + W / 2) * bpp) as usize;
    let center = &pixels[center_off..center_off + 4];
    assert!(center[2] > 80, "center blue must dominate: {center:?}");
}
