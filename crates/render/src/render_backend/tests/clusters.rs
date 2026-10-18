//! Clustered-lighting GPU gates (#130): the flag plumbing, the
//! assignment-pass parity against the CPU reference, the 100+-light
//! golden scene and the ≤8-light bit-identity window. Skipped when no
//! adapter is available (CI without GPU/lavapipe).

use super::super::*;
use super::*;
use crate::renderer::cpu_assign_lights_to_clusters;
use crate::renderer::{CLUSTER_COUNT, ClusterBuffer, MAX_CLUSTERED_LIGHTS};
use crate::renderer::{
    CameraUniform, ClusterParams, ClusterSpan, ClusteredLights, GpuLight, InstanceData,
    LIGHT_KIND_DIRECTIONAL, LIGHT_KIND_POINT, LIGHT_KIND_SPOT, MaterialIdx, Renderer3D,
};
use ornis_assets::scene::ShadowCast;

/// Probe size: 320×180 keeps every gate under a second on lavapipe.
const PROBE_W: u32 = 320;
const PROBE_H: u32 = 180;
/// Light-grid columns/rows (16×8 = 128 point lights, 120 beyond the
/// uniform window).
const GRID_COLS: u32 = 16;
const GRID_ROWS: u32 = 8;
/// Light-grid spacing in meters (x and z).
const GRID_SPACING: f32 = 3.5;
/// Light height above the ground plane.
const LIGHT_Y: f32 = 1.2;
/// Light cutoff radius: covers the ground below, clears the cell
/// midpoints.
const LIGHT_RANGE: f32 = 2.4;
/// Light radiometric intensity.
const LIGHT_INTENSITY: f32 = 6.0;
/// Three-color cycle so every light has a dominant channel.
const LIGHT_COLORS: [[f32; 3]; 3] = [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]];
/// Channel-dominance margin (post-tonemap sRGB bytes) for the
/// under-light pixel check.
const DOMINANCE_MARGIN: i32 = 40;
/// Brightness floor for the under-light pixel check.
const BRIGHT_FLOOR: u8 = 110;
/// Ceiling for a cell-midpoint pixel (ambient only — no light reaches).
const DARK_CEILING: u8 = 90;
/// Camera frame of the golden scene.
const EYE: glam::Vec3 = glam::Vec3::new(0.0, 24.0, 18.0);
/// Camera field of view.
const FOV: f32 = 60.0;
/// Camera near plane.
const NEAR: f32 = 0.5;
/// Camera far plane.
const FAR: f32 = 200.0;
/// Half of the NDC ↔ UV mapping.
const NDC_HALF: f32 = 0.5;
/// Search radius (pixels) around a projected point.
const SAMPLE_RADIUS: i32 = 2;
/// Camera eye of the assignment-parity scene.
const PARITY_EYE: glam::Vec3 = glam::Vec3::new(0.0, 2.5, 9.0);

/// The 128-light grid: colored point lights over the origin plane.
fn light_grid() -> Vec<LightDesc> {
    let mut lights = Vec::new();
    for row in 0..GRID_ROWS {
        for col in 0..GRID_COLS {
            let x = grid_axis(col, GRID_COLS);
            let z = grid_axis(row, GRID_ROWS);
            let index = (row * GRID_COLS + col) % LIGHT_COLORS.len() as u32;
            lights.push(LightDesc::Point {
                position: glam::Vec3::new(x, LIGHT_Y, z),
                intensity: LIGHT_INTENSITY,
                color: LIGHT_COLORS[index as usize],
                range: ornis_core::units::Meters::new(LIGHT_RANGE),
                shadow: ShadowCast::Disabled,
            });
        }
    }
    lights
}

/// World coordinate of grid slot `i` of `count` (centered, `spacing`
/// apart).
fn grid_axis(i: u32, count: u32) -> f32 {
    (i as f32 - (count - 1) as f32 * NDC_HALF) * GRID_SPACING
}

/// Camera matrices of the golden scene (view, projection, eye).
fn scene_camera() -> (glam::Mat4, glam::Mat4, [f32; 3]) {
    let view = glam::camera::rh::view::look_at_mat4(EYE, glam::Vec3::ZERO, glam::Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        FOV.to_radians(),
        PROBE_W as f32 / PROBE_H as f32,
        NEAR,
        FAR,
    );
    (view, proj, EYE.to_array())
}

/// World position → probe pixel.
fn project_pixel(view_proj: &glam::Mat4, p: glam::Vec3) -> (i32, i32) {
    let clip = *view_proj * p.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = (ndc.x * NDC_HALF + NDC_HALF) * PROBE_W as f32;
    let y = (NDC_HALF - ndc.y * NDC_HALF) * PROBE_H as f32;
    (x as i32, y as i32)
}

/// Offscreen config matching the probe target.
fn probe_config() -> wgpu::SurfaceConfiguration {
    wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        width: PROBE_W,
        height: PROBE_H,
        present_mode: wgpu::PresentMode::AutoNoVsync,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Auto,
    }
}

/// Render the plane scene with `lights` through `renderer` and read
/// the RGBA8 pixels back.
fn render_scene_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    lights: &[LightDesc],
) -> Vec<u8> {
    let (target, target_view, _) =
        probe::probe_color_target(device, "cluster probe", PROBE_W, PROBE_H);
    let mesh = crate::mesh::create_plane(device, [120.0, 120.0]);
    let material = {
        let mut m = ornis_core::OpenPBRMaterial::dielectric();
        m.base.color_rgb([0.8, 0.8, 0.8]);
        m.specular.roughness(0.8);
        m
    };
    renderer.upload_materials(device, queue, &[material]);
    renderer.upload_instances(
        device,
        queue,
        &[InstanceData {
            model_matrix: glam::Mat4::IDENTITY,
            normal_matrix: glam::Mat4::IDENTITY,
            material_index: MaterialIdx::from_raw(0),
        }],
    );
    renderer.set_lights(queue, [0.02, 0.02, 0.03], lights);
    let (view, proj, eye) = scene_camera();
    let view_proj = (proj * view).to_cols_array_2d();
    renderer.set_camera(queue, &view_proj, eye);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cluster probe encoder"),
    });
    renderer.render_scene(device, queue, &mut encoder, &target_view, &mesh, 1);
    probe::read_color_target(device, queue, encoder, &target, (PROBE_W, PROBE_H))
}

/// Brightest pixel of the window around (`px`, `py`) as RGB.
fn dominant_in_window(pixels: &[u8], px: i32, py: i32) -> (u8, u8, u8) {
    let mut best = (0u8, 0u8, 0u8);
    let mut best_sum = 0u16;
    for dy in -SAMPLE_RADIUS..=SAMPLE_RADIUS {
        for dx in -SAMPLE_RADIUS..=SAMPLE_RADIUS {
            let x = (px + dx).clamp(0, PROBE_W as i32 - 1) as usize;
            let y = (py + dy).clamp(0, PROBE_H as i32 - 1) as usize;
            let offset = (y * PROBE_W as usize + x) * 4;
            let candidate = (pixels[offset], pixels[offset + 1], pixels[offset + 2]);
            let sum = candidate.0 as u16 + candidate.1 as u16 + candidate.2 as u16;
            if sum > best_sum {
                best_sum = sum;
                best = candidate;
            }
        }
    }
    best
}

/// The flag is off by default and round-trips through the setter.
#[test]
fn clustered_flag_roundtrips() {
    assert!(!RenderBackendConfig::default().clustered_lights);
    let Some((device, _queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping clustered flag roundtrip");
        return;
    };
    let renderer = Renderer3D::new(&device, &probe_config(), 1);
    assert!(!renderer.clustered_lights_enabled());
    renderer.set_clustered_lights(true);
    assert!(renderer.clustered_lights_enabled());
    renderer.set_clustered_lights(false);
    assert!(!renderer.clustered_lights_enabled());
}

/// The assignment compute pass reproduces the CPU reference: same
/// per-cluster counts and index lists, no overflow, cursor equal to
/// the total.
#[test]
fn gpu_assignment_matches_cpu_reference() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping cluster assignment parity");
        return;
    };
    let view = glam::camera::rh::view::look_at_mat4(PARITY_EYE, glam::Vec3::ZERO, glam::Vec3::Y);
    let proj =
        glam::camera::rh::proj::directx::perspective(FOV.to_radians(), 16.0 / 9.0, 0.1, 100.0);
    let view_proj = proj * view;
    let mut records = Vec::new();
    // Two head directionals (kind only — the assignment never reads
    // their color), twelve ranged points along x, one far offscreen
    // point and one downward spot.
    for _ in 0..2u32 {
        records.push(gpu_record(
            LIGHT_KIND_DIRECTIONAL,
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
        ));
    }
    for i in 0..12u32 {
        records.push(gpu_record(
            LIGHT_KIND_POINT,
            [i as f32 * 0.3, 0.5, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.5, 0.0, 0.0, -1.0],
        ));
    }
    records.push(gpu_record(
        LIGHT_KIND_POINT,
        [500.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0, -1.0],
    ));
    records.push(gpu_record(
        LIGHT_KIND_SPOT,
        [0.0, 5.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 0.0],
        [6.0, 0.9, 0.8, -1.0],
    ));

    let camera = CameraUniform {
        view_proj: view_proj.to_cols_array_2d(),
        inv_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        camera_pos: [PARITY_EYE.x, PARITY_EYE.y, PARITY_EYE.z, 1.0],
    };
    let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cluster test camera"),
        size: std::mem::size_of::<CameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&camera_buffer, 0, bytemuck::bytes_of(&camera));
    let clusters = ClusteredLights::new(&device, &camera_buffer);
    let head_count = 2u32;
    let params = ClusterParams {
        light_count: records.len() as u32,
        head_count,
        clustered: 1,
        _pad: 0,
    };
    clusters.write_frame(
        &queue,
        &params,
        &records[..head_count as usize],
        &records[head_count as usize..],
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("cluster test encoder"),
    });
    clusters.dispatch(&mut encoder);
    queue.submit(std::iter::once(encoder.finish()));

    let reference = cpu_assign_lights_to_clusters(&records, head_count, &view_proj);
    let grid_bytes = clusters
        .read_back(
            &device,
            &queue,
            ClusterBuffer::Grid,
            CLUSTER_COUNT as u64 * std::mem::size_of::<ClusterSpan>() as u64,
        )
        .expect("grid readback");
    let grid: Vec<ClusterSpan> = grid_bytes
        .chunks_exact(std::mem::size_of::<ClusterSpan>())
        .map(|chunk| bytemuck::pod_read_unaligned(chunk))
        .collect();
    let counter_bytes = clusters
        .read_back(&device, &queue, ClusterBuffer::Counters, 8)
        .expect("counters");
    let counters: [u32; 2] = bytemuck::pod_read_unaligned(&counter_bytes[..8]);
    let total: u32 = reference.lists.iter().map(|list| list.len() as u32).sum();
    assert_eq!(counters[0], total, "cursor must claim the reference total");
    assert_eq!(counters[1], reference.overflow, "no overflow expected");
    let list_bytes = clusters
        .read_back(
            &device,
            &queue,
            ClusterBuffer::Indices,
            total as u64 * std::mem::size_of::<u32>() as u64,
        )
        .expect("index-list readback");
    let list: Vec<u32> = list_bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("u32 word")))
        .collect();
    for (cluster, span) in grid.iter().enumerate() {
        let want = &reference.lists[cluster];
        assert_eq!(span.count as usize, want.len(), "cluster {cluster} count");
        let got = &list[span.offset as usize..span.offset as usize + span.count as usize];
        assert_eq!(got, &want[..], "cluster {cluster} list");
    }
}

/// 128 point lights over a plane: the uniform window keeps 8 and drops
/// 120, the cluster storage carries all 120 beyond it, and every light
/// tints the ground below with its own color — the gate that the
/// clustered path actually lights the frame (#130).
#[test]
fn hundred_point_lights_reach_the_frame() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping cluster golden scene");
        return;
    };
    let renderer = Renderer3D::new(&device, &probe_config(), 1);
    renderer.set_clustered_lights(true);
    let pixels = render_scene_pixels(&device, &queue, &renderer, &light_grid());
    // Upload accounting: the window dropped 120 lights, the cluster
    // storage uploaded exactly those.
    assert_eq!(renderer.light_upload_stats().dropped_lights, 120);
    let stats = renderer.cluster_light_stats();
    assert_eq!(stats.clustered, 120);
    assert_eq!(stats.dropped, 0);
    assert!(stats.clustered < MAX_CLUSTERED_LIGHTS);

    let (view, proj, _) = scene_camera();
    let view_proj = proj * view;
    for row in 0..GRID_ROWS {
        for col in 0..GRID_COLS {
            let index = (row * GRID_COLS + col) % LIGHT_COLORS.len() as u32;
            let ground = glam::Vec3::new(grid_axis(col, GRID_COLS), 0.0, grid_axis(row, GRID_ROWS));
            let (px, py) = project_pixel(&view_proj, ground);
            let (r, g, b) = dominant_in_window(&pixels, px, py);
            let color = LIGHT_COLORS[index as usize];
            let (strong, weak_a, weak_b) = if color[0] > color[1].max(color[2]) {
                (r, g, b)
            } else if color[1] > color[2] {
                (g, r, b)
            } else {
                (b, r, g)
            };
            assert!(
                i32::from(strong) > i32::from(weak_a) + DOMINANCE_MARGIN
                    && i32::from(strong) > i32::from(weak_b) + DOMINANCE_MARGIN,
                "light ({col},{row}) must tint the ground: {strong} vs {weak_a}/{weak_b}"
            );
            assert!(
                strong > BRIGHT_FLOOR,
                "light ({col},{row}) must brighten the ground: {strong}"
            );
        }
    }
    // A cell midpoint sits beyond every light's range: ambient only.
    let mid = glam::Vec3::new(GRID_SPACING * NDC_HALF, 0.0, GRID_SPACING * NDC_HALF);
    let (mx, my) = project_pixel(&view_proj, mid);
    let (r, g, b) = dominant_in_window(&pixels, mx, my);
    assert!(
        r.max(g).max(b) < DARK_CEILING,
        "cell midpoint must stay dark: {r}/{g}/{b}"
    );
}

/// With ≤8 lights every light fits the uniform window, the cluster
/// span is zero, and the clustered-on frame is byte-identical to the
/// clustered-off frame (the same shader binary, the same storage
/// records — only data differs).
#[test]
fn clustered_window_is_bit_identical_below_eight_lights() {
    let Some((device, queue)) = try_device() else {
        eprintln!("no GPU adapter; skipping cluster parity window");
        return;
    };
    let lights: Vec<LightDesc> = (0..6)
        .map(|i| LightDesc::Point {
            position: glam::Vec3::new(grid_axis(i, 6), LIGHT_Y, 0.0),
            intensity: LIGHT_INTENSITY,
            color: LIGHT_COLORS[i as usize % LIGHT_COLORS.len()],
            range: ornis_core::units::Meters::new(LIGHT_RANGE),
            shadow: ShadowCast::Disabled,
        })
        .collect();
    let config = probe_config();
    let off = Renderer3D::new(&device, &config, 1);
    let on = Renderer3D::new(&device, &config, 1);
    on.set_clustered_lights(true);
    let pixels_off = render_scene_pixels(&device, &queue, &off, &lights);
    let pixels_on = render_scene_pixels(&device, &queue, &on, &lights);
    assert_eq!(
        off.light_upload_stats().dropped_lights,
        0,
        "the parity scene must fit the uniform window"
    );
    assert_eq!(on.cluster_light_stats().clustered, 0);
    assert_eq!(pixels_off, pixels_on, "≤8 lights: bit-identical frames");
}

/// One packed assignment-gate record from the raw fields (#130): color
/// is irrelevant to the assignment (never read), the shadow slot stays
/// zeroed.
fn gpu_record(kind: f32, position: [f32; 4], direction: [f32; 4], params: [f32; 4]) -> GpuLight {
    GpuLight {
        kind: [kind, 0.0, 0.0, 0.0],
        direction,
        position,
        color: [1.0, 1.0, 1.0, 1.0],
        params,
        shadow_vp: [[0.0; 4]; 4],
    }
}
