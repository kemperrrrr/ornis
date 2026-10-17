//! AO corner gate (#124): a cube in a three-plane corner under ambient-only
//! light, plus a direct-light control and a depth-step blur probe.
//!
//! Acceptance: disabled AO is bit-identical to the AO-free frame; the
//! cube/floor contact and the wall joint darken while the open floor does
//! not; checked-in goldens pin off/SSAO/GTAO; a directional light with no
//! ambient is untouched, including when cascaded shadows are on; with
//! ambient, AO still darkens under those cascades; a red fin at ~1 m in
//! front of a blue wall at ~5 m keeps its occlusion on the near side of
//! one bilateral tap.
//!
//! Skip contract: returns early with a note when no adapter exists (same
//! as the other headless gates; CI provides lavapipe). `ORNIS_REQUIRE_GPU=1`
//! panics instead of skipping, so a CI shard cannot pass this gate by
//! rendering nothing.
//!
//! Adapter acquisition and the render-and-read-back flow live in the shared
//! `common` harness (no per-gate copies).

#[allow(dead_code)]
mod common;

/// Red fin against a blue wall: the bilateral blur must not cross the step.
#[path = "ao_corner/depth_step.rs"]
mod depth_step;

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
/// (translation, scale) of those boxes. The mesh is a unit box, so the
/// world AABB is each translation expanded by half the scale.
const CORNER_BOXES: [([f32; 3], [f32; 3]); 4] = [
    ([0.0, -0.1, 0.0], [10.0, 0.2, 10.0]),
    ([-2.1, 3.0, 0.0], [0.2, 6.0, 10.0]),
    ([0.0, 3.0, -2.1], [10.0, 6.0, 0.2]),
    ([-1.73, 0.25, -1.73], [0.5, 0.5, 0.5]),
];

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
    let instances: Vec<InstanceData> = CORNER_BOXES
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
    let (_adapter, device, queue) = common::adapter_or_skip()?;
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
        let (dark, bright) = occlusion_counts(&reference, &pixels);
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

/// Pixels AO darkened (any channel drops by more than [`OCCLUSION_DELTA`])
/// and pixels it brightened (any channel rises by more than [`TOL`]).
fn occlusion_counts(reference: &[u8], pixels: &[u8]) -> (usize, usize) {
    let mut dark = 0usize;
    let mut bright = 0usize;
    for (a, b) in reference.chunks_exact(4).zip(pixels.chunks_exact(4)) {
        let darker = [0, 1, 2]
            .iter()
            .any(|&i| a[i].saturating_sub(b[i]) > OCCLUSION_DELTA);
        let brighter = [0, 1, 2].iter().any(|&i| b[i].saturating_sub(a[i]) > TOL);
        dark += usize::from(darker);
        bright += usize::from(brighter);
    }
    (dark, bright)
}

/// World AABB of [`CORNER_BOXES`]: each unit box is centered on its translation.
fn corner_caster_aabb() -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for (translation, scale) in CORNER_BOXES {
        for axis in 0..translation.len() {
            let half = scale[axis] / 2.0;
            min[axis] = min[axis].min(translation[axis] - half);
            max[axis] = max[axis].max(translation[axis] + half);
        }
    }
    (min, max)
}

/// Shadowed directional over the corner, with CSM engaged for `ambient`.
fn engage_corner_csm(queue: &wgpu::Queue, renderer: &Renderer3D, ambient: [f32; 3]) {
    let eye = glam::Vec3::from_array(EYE);
    let forward = (glam::Vec3::from_array(LOOK) - eye).normalize();
    renderer.set_csm_camera(NEAR_M, FAR_M, forward.to_array());
    let (min, max) = corner_caster_aabb();
    renderer.set_csm_casters(min, max);
    let direction = ornis_core::UnitVec3::normalize(glam::Vec3::new(-1.0, 0.8, 0.3))
        .expect("shadowed directional");
    let light = DirectionalLight {
        direction,
        ..DirectionalLight::default()
    };
    renderer.set_lights(queue, ambient, &[light.to_light_desc()]);
    assert_eq!(renderer.csm_cascade_count(), 4, "CSM did not engage");
}

/// Pixel of a world point under the corner camera.
fn project(world: glam::Vec3) -> (i32, i32) {
    let clip = corner_view_proj() * world.extend(1.0);
    let (x, y) = common::sized::clip_to_pixel(clip, W, H);
    (x as i32, y as i32)
}

/// Mean RGB inside a square window (the contact is a few pixels wide).
///
/// `n > 0` is load-bearing: a center past the frame used to skip every
/// pixel and return 0 for both images, so the open-floor check passed
/// without reading one.
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
    assert!(n > 0, "AO window ({x}, {y}) is outside the {W}x{H} frame");
    sum as f32 / n as f32
}

/// Cube/floor contact and the wall joint darken; the open floor does not.
fn assert_contacts(method: AoMethod, reference: &[u8], pixels: &[u8]) {
    let contact = project(glam::Vec3::new(-1.73, 0.02, -1.35));
    let joint = project(glam::Vec3::new(-1.98, 0.8, -1.98));
    // In frame (about pixel 160, 163) and metres from both walls and the cube.
    let open = project(glam::Vec3::new(1.0, 0.02, 1.0));
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

/// CSM on: AO stays off a direct-only frame and still darkens ambient.
///
/// A mutant that forces the AO factor to 1 whenever `csm.split_far` is
/// live keeps the direct-only frames identical and leaves the ambient
/// frames undarkened, so the darkening assert is what kills it.
#[test]
fn ao_with_csm_darkens_ambient_and_leaves_direct() {
    let Some((device, queue, renderer, mesh, _)) = prepared_corner() else {
        return;
    };
    engage_corner_csm(&queue, &renderer, [0.0, 0.0, 0.0]);
    let direct = render_pixels(&device, &queue, &renderer, &mesh, None);
    let lit = lit_fraction(&direct);
    eprintln!("ao+csm probe: direct-only lit fraction = {lit:.3}");
    assert!(lit > 0.0, "direct-only CSM frame is black: {lit:.3}");
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        assert_eq!(
            pixels, direct,
            "{method:?} changed a direct-only CSM frame (AO must not touch direct light)"
        );
    }

    engage_corner_csm(&queue, &renderer, [0.55, 0.55, 0.6]);
    let ambient = render_pixels(&device, &queue, &renderer, &mesh, None);
    let lit = lit_fraction(&ambient);
    assert!(
        lit > MIN_LIT_FRACTION,
        "ambient CSM frame is black: {lit:.3}"
    );
    for method in [AoMethod::Ssao, AoMethod::Gtao] {
        let pixels = render_pixels(&device, &queue, &renderer, &mesh, Some(wiring(method, 1.0)));
        let (dark, bright) = occlusion_counts(&ambient, &pixels);
        let fraction = dark as f64 / f64::from(W * H);
        eprintln!(
            "ao+csm probe: {method:?} darkened pixels = {dark} ({fraction:.4}), brightened = {bright}"
        );
        assert_eq!(
            bright, 0,
            "{method:?} brightened {bright} pixels under CSM (AO must never add light)"
        );
        assert!(
            fraction >= MIN_DARK_FRACTION,
            "{method:?} AO does not darken under CSM ({dark})"
        );
    }
}
