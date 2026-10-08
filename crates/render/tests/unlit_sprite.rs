//! Unlit sprite headless pixel gate: a `Quad` with an `Unlit` material in
//! front of the camera must render identically under wildly different
//! lighting (one directional vs eight colored point lights, shadows on and
//! off), while the same quad with a `Dielectric` material must shade
//! noticeably differently (proving the gate is light-sensitive), and the
//! quad must cover the expected central frame area.
//!
//! Skip contract: returns early with a note when no adapter exists (same as
//! the other headless gates; CI provides lavapipe).
//!
//! The scene goes through the real pipeline (`Engine` lanes +
//! [`ornis_render::extract_render_data`]), so the gate also pins the
//! `Quad` size scale and the `Unlit` GPU mapping end to end.

use ornis_assets::scene::{LightDesc, MaterialDesc, MeshDesc, ShadowCast, TransformDesc};
use ornis_core::Engine;
use ornis_core::units::{Clamped01, LinearRgb, Meters, UnitVec3};
use ornis_render::{Renderer3D, extract_render_data};

/// Offscreen frame extent (same as the MSAA probe).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: matches the composite output.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Per-channel drift tolerated for the unlit-equality gate.
const TOL: u8 = 1;
/// Brightness gate isolating quad pixels from the near-black background
/// (same window as the MSAA probe's lit fraction).
const LIT_SUM: u16 = 36;
/// Minimum fraction of differing pixels proving the dielectric control is
/// light-sensitive (per-channel swing above [`CTRL_TOL`]).
const CTRL_MIN_DIFF_FRACTION: f64 = 0.5;
/// Per-channel swing counting as "noticeably different" for the control.
const CTRL_TOL: u8 = 10;
/// Expected quad coverage band: a 2x2 quad at distance 4 under a 55-degree
/// perspective covers ~13% of the frame; the band keeps wide headroom for
/// rasterization differences across drivers.
const MIN_COVERAGE: f64 = 0.08;
/// Upper coverage bound (see [`MIN_COVERAGE`]).
const MAX_COVERAGE: f64 = 0.20;

/// Sprite under test: 2x2 unlit steel-blue.
fn unlit_material() -> MaterialDesc {
    MaterialDesc::unlit_units(LinearRgb::new([0.2, 0.4, 0.8]))
}

/// Control material: mid-gray dielectric, roughness 0.5.
fn dielectric_material() -> MaterialDesc {
    MaterialDesc::dielectric_units(LinearRgb::new([0.5, 0.5, 0.5]), Clamped01::new(0.5))
}

fn try_adapter() -> Option<(wgpu::Device, wgpu::Queue)> {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .ok()?;
        Some((device, queue))
    })
}

fn surface_config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: FORMAT,
        width: W,
        height: H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

/// One directional light, optionally shadowed.
fn directional_lights(shadow: ShadowCast) -> Vec<LightDesc> {
    vec![LightDesc::Directional {
        direction: UnitVec3::normalize(glam::Vec3::new(0.42, 0.84, 0.3))
            .expect("non-zero direction"),
        intensity: 1.5,
        color: [1.0, 1.0, 1.0],
        shadow,
    }]
}

/// Eight point lights of distinct colors on a ring facing the quad.
fn point_lights(shadow: ShadowCast) -> Vec<LightDesc> {
    const COLORS: [[f32; 3]; 8] = [
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
        [1.0, 1.0, 0.1],
        [0.1, 1.0, 1.0],
        [1.0, 0.1, 1.0],
        [1.0, 0.5, 0.1],
        [0.9, 0.9, 0.8],
    ];
    COLORS
        .iter()
        .enumerate()
        .map(|(i, color)| {
            let angle = i as f32 / 8.0 * std::f32::consts::TAU;
            LightDesc::Point {
                position: glam::Vec3::new(3.0 * angle.cos(), 3.0 * angle.sin(), 2.5),
                intensity: 3.0,
                color: *color,
                range: Meters::new(12.0),
                shadow,
            }
        })
        .collect()
}

/// Renders one frame of the `Quad` + `material` scene under `lights` and
/// reads it back (legacy hybrid path: gbuffer, deferred lighting, forward,
/// composite).
fn render_frame(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    material: MaterialDesc,
    lights: &[LightDesc],
) -> Vec<u8> {
    let renderer = Renderer3D::new(device, &surface_config(), 1);
    // Unit mesh: the primitive's size is baked into the extraction model
    // matrix (shared-unit-mesh convention, like the sphere radius), so a
    // 2x2 `Quad` desc pairs with a 1x1 mesh for a 2x2 effective sprite.
    let mesh = ornis_render::mesh::create_quad(device, [1.0, 1.0]);
    let mut engine = Engine::new();
    let store = engine.world_mut().store_mut().expect("store");
    let entity = store.create_entity();
    store.insert(entity, TransformDesc::IDENTITY);
    store.insert(
        entity,
        MeshDesc::try_quad_units([Meters::new(2.0), Meters::new(2.0)]).expect("positive size"),
    );
    store.insert(entity, material);
    let upload = extract_render_data(engine.world().store().expect("store"));
    assert_eq!(upload.instances.len(), 1, "quad entity must extract");
    renderer.upload_materials(device, queue, &upload.materials);
    renderer.upload_instances(device, queue, &upload.instances);
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::new(0.0, 0.0, 4.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        55.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        100.0,
    );
    renderer.set_camera(queue, &(proj * view).to_cols_array_2d(), [0.0, 0.0, 4.0]);
    renderer.set_lights(queue, [0.05, 0.05, 0.08], lights);

    let target_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("unlit probe target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("unlit probe encoder"),
    });
    renderer.render_scene(device, queue, &mut encoder, &target_view, &mesh, 1);
    queue.submit([encoder.finish()]);
    readback_pixels(device, queue, &target_tex)
}

/// Copies a `COPY_SRC` target back to CPU bytes (blocking), stripping the
/// 256-byte row padding.
fn readback_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_tex: &wgpu::Texture,
) -> Vec<u8> {
    const BPP: u32 = 4;
    let unpadded = W * BPP;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("unlit probe readback"),
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("unlit probe readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target_tex,
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
        .expect("poll readback");
    let data = slice.get_mapped_range().unwrap();
    let mut pixels = vec![0u8; (unpadded * H) as usize];
    for y in 0..H as usize {
        pixels[y * unpadded as usize..][..unpadded as usize]
            .copy_from_slice(&data[y * padded as usize..][..unpadded as usize]);
    }
    drop(data);
    readback.unmap();
    pixels
}

/// Per-pixel max channel spread across frames.
fn spread(frames: &[Vec<u8>], pixel: usize) -> u8 {
    let at = |f: &[u8]| [f[4 * pixel], f[4 * pixel + 1], f[4 * pixel + 2]];
    let first = at(&frames[0]);
    frames[1..]
        .iter()
        .flat_map(|f| {
            let c = at(f);
            (0..3).map(move |ch| first[ch].abs_diff(c[ch]))
        })
        .max()
        .unwrap_or(0)
}

fn is_lit(px: &[u8], pixel: usize) -> bool {
    px[4 * pixel] as u16 + px[4 * pixel + 1] as u16 + px[4 * pixel + 2] as u16 > LIT_SUM
}

#[test]
fn unlit_quad_is_light_and_shadow_independent() {
    let Some((device, queue)) = try_adapter() else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let frames = [
        render_frame(
            &device,
            &queue,
            unlit_material(),
            &directional_lights(ShadowCast::Disabled),
        ),
        render_frame(
            &device,
            &queue,
            unlit_material(),
            &directional_lights(ShadowCast::Enabled),
        ),
        render_frame(
            &device,
            &queue,
            unlit_material(),
            &point_lights(ShadowCast::Disabled),
        ),
        render_frame(
            &device,
            &queue,
            unlit_material(),
            &point_lights(ShadowCast::Enabled),
        ),
    ];
    // Quad mask: lit in the first frame and stable across all four.
    let mask: Vec<usize> = (0..(W * H) as usize)
        .filter(|p| is_lit(&frames[0], *p) && spread(&frames, *p) <= TOL)
        .collect();
    let coverage = mask.len() as f64 / (W as f64 * H as f64);
    eprintln!(
        "unlit probe: stable lit pixels = {} ({coverage:.4})",
        mask.len()
    );
    assert!(
        coverage > MIN_COVERAGE,
        "quad rendered too small to gate: {coverage:.4}"
    );
    // Every stable lit pixel must match across all four frames within TOL
    // (the mask construction already enforces this; re-assert the worst
    // spread explicitly for the report).
    let worst = (0..(W * H) as usize)
        .filter(|p| is_lit(&frames[0], *p))
        .map(|p| spread(&frames, p))
        .max()
        .unwrap_or(0);
    eprintln!("unlit probe: worst per-channel spread over lit pixels = {worst}");
    assert!(
        worst <= TOL,
        "unlit quad changed under different light/shadow: spread {worst} > {TOL}"
    );
}

#[test]
fn dielectric_control_is_light_sensitive() {
    let Some((device, queue)) = try_adapter() else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let dir = render_frame(
        &device,
        &queue,
        dielectric_material(),
        &directional_lights(ShadowCast::Disabled),
    );
    let pts = render_frame(
        &device,
        &queue,
        dielectric_material(),
        &point_lights(ShadowCast::Disabled),
    );
    // Same geometric mask as the unlit gate: pixels the quad covers.
    let mask: Vec<usize> = (0..(W * H) as usize).filter(|p| is_lit(&dir, *p)).collect();
    assert!(!mask.is_empty(), "control rendered nothing lit");
    let changed = mask
        .iter()
        .filter(|p| {
            let a = [dir[4 * *p], dir[4 * *p + 1], dir[4 * *p + 2]];
            let b = [pts[4 * *p], pts[4 * *p + 1], pts[4 * *p + 2]];
            (0..3).any(|ch| a[ch].abs_diff(b[ch]) > CTRL_TOL)
        })
        .count();
    let fraction = changed as f64 / mask.len() as f64;
    eprintln!(
        "unlit probe control: {changed}/{} quad pixels differ > {CTRL_TOL} ({fraction:.4})",
        mask.len()
    );
    assert!(
        fraction > CTRL_MIN_DIFF_FRACTION,
        "control is not light-sensitive: only {fraction:.4} of quad pixels changed"
    );
}

#[test]
fn unlit_quad_covers_the_expected_frame_area() {
    let Some((device, queue)) = try_adapter() else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let px = render_frame(
        &device,
        &queue,
        unlit_material(),
        &directional_lights(ShadowCast::Disabled),
    );
    let lit: Vec<(u32, u32)> = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|(x, y)| is_lit(&px, (*y * W + *x) as usize))
        .collect();
    let coverage = lit.len() as f64 / (W as f64 * H as f64);
    eprintln!("unlit probe: quad coverage = {coverage:.4}");
    assert!(
        (MIN_COVERAGE..=MAX_COVERAGE).contains(&coverage),
        "quad covers {coverage:.4}, expected {MIN_COVERAGE:.2}..={MAX_COVERAGE:.2}"
    );
    // The quad is centered: its bounding-box center must sit near the
    // frame center.
    let (min_x, max_x) = lit
        .iter()
        .map(|(x, _)| x)
        .fold((W, 0), |(a, b), x| (a.min(*x), b.max(*x)));
    let (min_y, max_y) = lit
        .iter()
        .map(|(_, y)| y)
        .fold((H, 0), |(a, b), y| (a.min(*y), b.max(*y)));
    let cx = (f64::from(min_x) + f64::from(max_x)) / 2.0 / f64::from(W);
    let cy = (f64::from(min_y) + f64::from(max_y)) / 2.0 / f64::from(H);
    eprintln!("unlit probe: bbox center = ({cx:.3}, {cy:.3})");
    assert!(
        (cx - 0.5).abs() < 0.1 && (cy - 0.5).abs() < 0.1,
        "quad is off-center: ({cx:.3}, {cy:.3})"
    );
}
