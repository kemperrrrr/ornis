//! AO corner gate (#124): a cube in a three-plane corner under ambient-only
//! light, plus a direct-light control and a depth-step blur probe.
//!
//! Acceptance: disabled AO is bit-identical to the AO-free frame; the
//! cube/floor contact and the wall joint darken while the open floor does
//! not; checked-in goldens pin off/SSAO/GTAO; a directional light with no
//! ambient is untouched; a red fin at ~1 m in front of a blue wall at ~5 m
//! keeps its occlusion on the near side of one bilateral tap.
//!
//! Skip contract: returns early with a note when no adapter exists (same
//! as the other headless gates; CI provides lavapipe).
//!
//! Adapter acquisition and the render-and-read-back flow live in the shared
//! `common` harness (no per-gate copies).

#[allow(dead_code)]
mod common;

use common::sized::{FrameSpec, render_and_readback};
use ornis_render::render_backend::RenderContext;
use ornis_render::{
    AoMethod, AoSettings, AoWiring, Bloom, DirectionalLight, InstanceData, MaterialIdx,
    OpenPBRMaterial, RenderFrame3D, Renderer3D, Technique,
};

/// Offscreen frame extent (MSAA-gate precedent).
const W: u32 = 320;
/// Offscreen frame height.
const H: u32 = 180;
/// Read-back format: matches the composite output.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Per-channel drift tolerated before a pixel counts as changed (same
/// window as the golden probe: driver rounding, not engine regressions).
const TOL: u8 = 4;
/// Channel delta that counts as real occlusion (clearly above noise).
const OCCLUSION_DELTA: u8 = 6;
/// Minimum fraction of darkened pixels: the corner junctions must cover
/// this share of the frame at radius 0.75 m.
const MIN_DARK_FRACTION: f64 = 0.005;
/// Minimum fraction of non-black pixels: the corner must be visible.
const MIN_LIT_FRACTION: f64 = 0.2;
/// How many golden pixels may exceed [`TOL`] (driver edges, not a new kernel).
const GOLDEN_BUDGET: usize = 64;
/// Camera of the corner scene (shared by the upload and the region probe).
const EYE: [f32; 3] = [3.0, 2.2, 3.0];
/// Look-at target of the corner scene.
const LOOK: [f32; 3] = [-0.8, 0.8, -0.8];
/// Vertical field of view of the corner scene, degrees.
const FOV_Y: f32 = 55.0;
/// Near plane of the corner scene, meters.
const NEAR_M: f32 = 0.1;
/// Far plane of the corner scene, meters.
const FAR_M: f32 = 100.0;
/// Instanced boxes: floor, two walls, cube.
const CORNER_INSTANCES: u32 = 4;

/// Offscreen frame spec (shared harness builds targets from it).
fn frame_spec() -> FrameSpec {
    FrameSpec {
        size: (W, H),
        format: FORMAT,
    }
}

/// View-projection of the corner camera.
fn corner_view_proj() -> glam::Mat4 {
    let view = glam::camera::rh::view::look_at_mat4(
        glam::Vec3::from_array(EYE),
        glam::Vec3::from_array(LOOK),
        glam::Vec3::Y,
    );
    let proj = glam::camera::rh::proj::directx::perspective(
        FOV_Y.to_radians(),
        W as f32 / H as f32,
        NEAR_M,
        FAR_M,
    );
    proj * view
}

/// One renderer over the corner scene: unit-box mesh instanced into a
/// floor, two walls meeting at (-2, 0, -2), and a cube on the floor in
/// that joint. Rough white dielectric, ambient-only light (no direct
/// term for AO to touch by mistake).
fn build_corner(device: &wgpu::Device, queue: &wgpu::Queue) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::create_box(device, [1.0, 1.0, 1.0]);
    let mut material = OpenPBRMaterial::dielectric();
    material.base.color_rgb([0.9, 0.9, 0.9]);
    material.specular.roughness(1.0);
    renderer.upload_materials(device, queue, &[material]);
    // (translation, scale): floor, wall x=-2, wall z=-2, cube in the joint.
    let boxes = [
        ([0.0, -0.1, 0.0], [10.0, 0.2, 10.0]),
        ([-2.1, 3.0, 0.0], [0.2, 6.0, 10.0]),
        ([0.0, 3.0, -2.1], [10.0, 6.0, 0.2]),
        ([-1.73, 0.25, -1.73], [0.5, 0.5, 0.5]),
    ];
    let instances: Vec<InstanceData> = boxes
        .iter()
        .map(|(t, s)| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(*s),
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(*t),
            );
            InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: MaterialIdx::from_raw(0),
            }
        })
        .collect();
    renderer.upload_instances(device, queue, &instances);
    let eye = glam::Vec3::from_array(EYE);
    renderer.set_camera(
        queue,
        &corner_view_proj().to_cols_array_2d(),
        [eye.x, eye.y, eye.z],
    );
    renderer.set_lights(queue, [0.55, 0.55, 0.6], &[]);
    (renderer, mesh)
}

/// Renders one plan-path frame (Deferred, no bloom) with the given AO
/// wiring (`None` = AO-free) into a fresh target and reads it back.
fn render_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &ornis_render::Mesh,
    ao: Option<AoWiring>,
) -> Vec<u8> {
    render_count(device, queue, renderer, mesh, CORNER_INSTANCES, ao)
}

fn render_count(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &Renderer3D,
    mesh: &ornis_render::Mesh,
    instances: u32,
    ao: Option<AoWiring>,
) -> Vec<u8> {
    render_and_readback(device, queue, &frame_spec(), |encoder, view| {
        let mut plan = match ao {
            Some(wiring) => {
                RenderFrame3D::new_with_ao(FORMAT, (W, H), Technique::Deferred, Bloom::Off, wiring)
            }
            None => RenderFrame3D::new_with(FORMAT, (W, H), Technique::Deferred, Bloom::Off),
        };
        plan.render(
            RenderContext {
                device,
                queue,
                encoder,
                target: view,
            },
            renderer,
            mesh,
            instances,
        );
    })
}

fn wiring(method: AoMethod, intensity: f32) -> AoWiring {
    AoWiring::new(AoSettings::new(method, 0.75, intensity, 0.025).expect("valid AO settings"))
}

/// Adapter, corner scene, and the AO-free reference frame. `None` when no
/// adapter exists (the gate skips, same contract as the other headless tests).
fn prepared_corner() -> Option<(
    wgpu::Device,
    wgpu::Queue,
    Renderer3D,
    ornis_render::Mesh,
    Vec<u8>,
)> {
    let Some((_adapter, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return None;
    };
    let (renderer, mesh) = build_corner(&device, &queue);
    let reference = render_pixels(&device, &queue, &renderer, &mesh, None);
    Some((device, queue, renderer, mesh, reference))
}

/// Fraction of texels whose RGB sum clears the near-black cutoff.
fn lit_fraction(pixels: &[u8]) -> f64 {
    let lit = pixels
        .chunks_exact(4)
        .filter(|c| c[0] as u16 + c[1] as u16 + c[2] as u16 > 36)
        .count() as f64;
    lit / (pixels.len() / 4) as f64
}

#[test]
fn ao_disabled_is_bit_identical_to_no_ao() {
    let Some((device, queue, renderer, mesh, reference)) = prepared_corner() else {
        return;
    };
    // Zero intensity collapses both kernels to the white factor on the
    // GPU, exactly like the unbound fallback.
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 0.0)));
        assert_eq!(
            pixels, reference,
            "{method:?} at zero intensity must be bit-identical to AO-free"
        );
    }
}

#[test]
fn ao_darkens_the_corner_and_never_brightens() {
    let Some((device, queue, renderer, mesh, reference)) = prepared_corner() else {
        return;
    };
    let lit = lit_fraction(&reference);
    eprintln!("ao corner probe: lit fraction = {lit:.3}");
    assert!(
        lit > MIN_LIT_FRACTION,
        "corner scene rendered dark: {lit:.3}"
    );

    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        // Determinism: the hash is per-pixel, with no temporal component.
        let again = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        assert_eq!(pixels, again, "{method:?} AO frame is not deterministic");
        let mut dark = 0usize;
        let mut bright = 0usize;
        for (a, b) in reference.chunks_exact(4).zip(pixels.chunks_exact(4)) {
            let darker = [0, 1, 2]
                .iter()
                .any(|&i| a[i].saturating_sub(b[i]) > OCCLUSION_DELTA);
            let brighter = [0, 1, 2].iter().any(|&i| b[i].saturating_sub(a[i]) > TOL);
            dark += darker as usize;
            bright += brighter as usize;
        }
        let total = (W * H) as f64;
        let fraction = dark as f64 / total;
        eprintln!(
            "ao corner probe: {method:?} darkened pixels = {dark} ({fraction:.4}), brightened = {bright}"
        );
        assert_eq!(
            bright, 0,
            "{method:?} brightened {bright} pixels (AO must never add light)"
        );
        assert!(
            fraction >= MIN_DARK_FRACTION,
            "{method:?} darkened too little: {fraction:.4} < {MIN_DARK_FRACTION}"
        );
        assert_contacts(method, &reference, &pixels);
        assert_golden(method, &pixels);
    }
    assert_golden_off(&reference);
}

/// Pixel of a world point under the corner camera.
fn project(world: glam::Vec3) -> (i32, i32) {
    let clip = corner_view_proj() * world.extend(1.0);
    let ndc = clip.truncate() / clip.w;
    let x = (ndc.x * 0.5 + 0.5) * W as f32;
    let y = (0.5 - ndc.y * 0.5) * H as f32;
    (x as i32, y as i32)
}

/// Mean RGB inside a square window (the contact is a few pixels wide).
fn window_mean(pixels: &[u8], x: i32, y: i32) -> f32 {
    let mut sum = 0u32;
    let mut n = 0u32;
    for dy in -2..=2 {
        for dx in -2..=2 {
            let px = x + dx;
            let py = y + dy;
            if px < 0 || py < 0 || px >= W as i32 || py >= H as i32 {
                continue;
            }
            let i = ((py as u32 * W + px as u32) * 4) as usize;
            sum += pixels[i] as u32 + pixels[i + 1] as u32 + pixels[i + 2] as u32;
            n += 1;
        }
    }
    sum as f32 / n.max(1) as f32
}

/// Cube/floor contact and the wall joint darken; the open floor does not.
fn assert_contacts(method: AoMethod, reference: &[u8], pixels: &[u8]) {
    let contact = project(glam::Vec3::new(-1.73, 0.02, -1.35));
    let joint = project(glam::Vec3::new(-1.98, 0.8, -1.98));
    let open = project(glam::Vec3::new(1.6, 0.02, 1.6));
    let contact_drop =
        window_mean(reference, contact.0, contact.1) - window_mean(pixels, contact.0, contact.1);
    let joint_drop =
        window_mean(reference, joint.0, joint.1) - window_mean(pixels, joint.0, joint.1);
    let open_drop = window_mean(reference, open.0, open.1) - window_mean(pixels, open.0, open.1);
    eprintln!(
        "ao regions {method:?}: contact {contact:?} drop {contact_drop:.1}, joint {joint:?} drop {joint_drop:.1}, floor {open:?} drop {open_drop:.1}"
    );
    assert!(
        contact_drop > f32::from(OCCLUSION_DELTA),
        "{method:?} cube/floor contact not darkened ({contact_drop:.1} at {contact:?})"
    );
    assert!(
        joint_drop > f32::from(OCCLUSION_DELTA),
        "{method:?} wall joint not darkened ({joint_drop:.1} at {joint:?})"
    );
    assert!(
        open_drop < f32::from(TOL),
        "{method:?} open floor darkened ({open_drop:.1} at {open:?})"
    );
}

fn golden_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load_golden(name: &str) -> Vec<u8> {
    let path = golden_path(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!((info.width, info.height), (W, H), "{name}");
    buf[..info.buffer_size()].to_vec()
}

fn write_golden(name: &str, pixels: &[u8]) {
    let path = golden_path(name);
    std::fs::create_dir_all(path.parent().expect("data dir")).expect("mkdir");
    let file =
        std::fs::File::create(&path).unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    let mut encoder = png::Encoder::new(file, W, H);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("png header")
        .write_image_data(pixels)
        .expect("png data");
    eprintln!("wrote {}", path.display());
}

fn wants_golden_write() -> bool {
    std::env::var_os("ORNIS_WRITE_AO_GOLDEN").is_some()
}

fn assert_matches_golden(name: &str, pixels: &[u8]) {
    if wants_golden_write() {
        write_golden(name, pixels);
        return;
    }
    let golden = load_golden(name);
    assert_eq!(pixels.len(), golden.len(), "{name}");
    let mut bad = 0usize;
    let mut worst = 0u8;
    for (a, b) in pixels.chunks_exact(4).zip(golden.chunks_exact(4)) {
        let delta = (0..3).map(|i| a[i].abs_diff(b[i])).max().unwrap_or(0);
        worst = worst.max(delta);
        bad += usize::from(delta > TOL);
    }
    assert!(
        bad <= GOLDEN_BUDGET && worst <= 16,
        "{name}: {bad} pixels differ by >{TOL} (budget {GOLDEN_BUDGET}), worst {worst}"
    );
}

fn assert_golden(method: AoMethod, pixels: &[u8]) {
    let name = match method {
        AoMethod::Ssao => "ao_corner_ssao.png",
        AoMethod::Gtao => "ao_corner_gtao.png",
    };
    assert_matches_golden(name, pixels);
}

fn assert_golden_off(pixels: &[u8]) {
    assert_matches_golden("ao_corner_off.png", pixels);
}

#[test]
fn ao_leaves_direct_light_bit_identical() {
    let Some((device, queue, renderer, mesh, _)) = prepared_corner() else {
        return;
    };
    let light = DirectionalLight::default();
    renderer.set_lights(&queue, [0.0, 0.0, 0.0], &[light.to_light_desc()]);
    let reference = render_pixels(&device, &queue, &renderer, &mesh, None);
    let lit = lit_fraction(&reference);
    assert!(
        lit > MIN_LIT_FRACTION,
        "direct-only frame is black: {lit:.3}"
    );
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        assert_eq!(
            pixels, reference,
            "{method:?} changed a direct-only frame (AO must not touch direct light)"
        );
    }
}

/// Red fin at ~1 m in front of a blue wall at ~5 m. The crease is the
/// silhouette the bilateral tap can reach; the two albedos name the sides.
fn build_depth_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (Renderer3D, ornis_render::Mesh) {
    let renderer = Renderer3D::new(device, &frame_spec().surface_config(), 1);
    let mesh = ornis_render::create_box(device, [1.0, 1.0, 1.0]);
    let mut near_mat = OpenPBRMaterial::dielectric();
    near_mat.base.color_rgb([0.9, 0.25, 0.25]);
    near_mat.specular.roughness(1.0);
    let mut far_mat = OpenPBRMaterial::dielectric();
    far_mat.base.color_rgb([0.25, 0.25, 0.9]);
    far_mat.specular.roughness(1.0);
    renderer.upload_materials(device, queue, &[near_mat, far_mat]);
    // A facing wall plus a thin fin, ~1 m away, with the blue wall ~5 m
    // behind the fin. The crease sits on the silhouette the blur can reach.
    let boxes = [
        ([0.0, -0.02, 0.0], [2.0, 0.04, 2.0], 0u32),
        ([-0.15, 0.28, 0.0], [0.55, 0.56, 0.06], 0),
        ([0.38, 0.28, 0.18], [0.03, 0.56, 0.4], 0),
        ([0.0, 1.0, -4.0], [12.0, 5.0, 0.1], 1),
    ];
    let instances: Vec<InstanceData> = boxes
        .iter()
        .map(|(t, s, material)| {
            let model = glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::from_array(*s),
                glam::Quat::IDENTITY,
                glam::Vec3::from_array(*t),
            );
            InstanceData {
                model_matrix: model,
                normal_matrix: model.inverse().transpose(),
                material_index: MaterialIdx::from_raw(*material),
            }
        })
        .collect();
    renderer.upload_instances(device, queue, &instances);
    let eye = glam::Vec3::new(0.15, 0.4, 1.15);
    let view =
        glam::camera::rh::view::look_at_mat4(eye, glam::Vec3::new(0.05, 0.28, 0.0), glam::Vec3::Y);
    let proj = glam::camera::rh::proj::directx::perspective(
        50.0f32.to_radians(),
        W as f32 / H as f32,
        0.1,
        30.0,
    );
    renderer.set_camera(
        queue,
        &(proj * view).to_cols_array_2d(),
        [eye.x, eye.y, eye.z],
    );
    renderer.set_lights(queue, [0.55, 0.55, 0.6], &[]);
    (renderer, mesh)
}

/// Full-res pixels covered by one bilateral tap: 1.5 half-res texels times
/// the AO divisor. A weights=1 blur mixes exactly this far.
const BLUR_TAP_PX: i32 = 3;
/// Sum-of-RGB drop on the red side. One weights=1 tap out of five samples
/// would move the blue side by at least this divided by five.
const STEP_NEAR_DROP: i32 = 36;
/// Sum-of-RGB drop still accepted on the blue side of the tap.
const STEP_FAR_DROP: i32 = 6;
/// Silhouette pairs required, so one driver-noisy pixel cannot pass it.
const STEP_PAIR_MIN: usize = 8;

#[test]
fn ao_blur_does_not_cross_a_depth_step() {
    let Some((_adapter, device, queue)) = pollster::block_on(common::request_device()) else {
        eprintln!("SKIP: no wgpu adapter (CI runs this on lavapipe)");
        return;
    };
    let (renderer, mesh) = build_depth_step(&device, &queue);
    let reference = render_count(&device, &queue, &renderer, &mesh, 4, None);
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let wiring = AoWiring::new(AoSettings::new(method, 0.6, 1.0, 0.02).expect("valid"));
        let pixels = render_count(&device, &queue, &renderer, &mesh, 4, Some(wiring));
        let pairs = depth_step_pairs(&reference, &pixels);
        assert!(
            pairs >= STEP_PAIR_MIN,
            "{method:?} depth step lost its separation ({pairs} pairs)"
        );
    }
}

/// Red/blue pairs one bilateral tap apart: the red pixel is occluded and the
/// blue pixel is not. Returns how many such pairs the frame contains.
fn depth_step_pairs(reference: &[u8], pixels: &[u8]) -> usize {
    let mut pairs = 0usize;
    for y in 0..H as i32 {
        pairs += depth_step_row(reference, pixels, y);
    }
    pairs
}

fn depth_step_row(reference: &[u8], pixels: &[u8], y: i32) -> usize {
    let mut pairs = 0usize;
    for x in 0..W as i32 {
        pairs += depth_step_at(reference, pixels, x, y);
    }
    pairs
}

fn depth_step_at(reference: &[u8], pixels: &[u8], x: i32, y: i32) -> usize {
    if !is_red(reference, x, y) {
        return 0;
    }
    let near = rgb_sum(reference, x, y) - rgb_sum(pixels, x, y);
    if near < STEP_NEAR_DROP {
        return 0;
    }
    blur_taps_onto_blue(reference, pixels, x, y, near)
}

/// Cardinal taps at [`BLUR_TAP_PX`]. A weights=1 blur (center plus four
/// taps, one of them this red pixel) would push `near / 5` onto the blue
/// side; that has to clear [`STEP_FAR_DROP`] while the real tap does not.
fn blur_taps_onto_blue(reference: &[u8], pixels: &[u8], x: i32, y: i32, near: i32) -> usize {
    let mut pairs = 0usize;
    for (dx, dy) in [
        (BLUR_TAP_PX, 0),
        (-BLUR_TAP_PX, 0),
        (0, BLUR_TAP_PX),
        (0, -BLUR_TAP_PX),
    ] {
        if tap_stays_on_blue(reference, pixels, x + dx, y + dy, near) {
            pairs += 1;
        }
    }
    pairs
}

fn tap_stays_on_blue(reference: &[u8], pixels: &[u8], x: i32, y: i32, near: i32) -> bool {
    if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 || !is_blue(reference, x, y) {
        return false;
    }
    let far = rgb_sum(reference, x, y) - rgb_sum(pixels, x, y);
    far <= STEP_FAR_DROP && near / 5 > STEP_FAR_DROP
}

fn rgb_sum(pixels: &[u8], x: i32, y: i32) -> i32 {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    i32::from(pixels[i]) + i32::from(pixels[i + 1]) + i32::from(pixels[i + 2])
}

/// Near fin: red albedo, clearly above the blue backdrop.
fn is_red(pixels: &[u8], x: i32, y: i32) -> bool {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    i32::from(pixels[i]) > i32::from(pixels[i + 2]) + i32::from(TOL) * 5
}

/// Backdrop: blue albedo, clearly above the red fin.
fn is_blue(pixels: &[u8], x: i32, y: i32) -> bool {
    let i = ((y as u32 * W + x as u32) * 4) as usize;
    i32::from(pixels[i + 2]) > i32::from(pixels[i]) + i32::from(TOL) * 5
}
