//! GPU parity of the storage-buffer guest against both production passes.

use crate::auto_exposure::AutoExposure;
use crate::auto_exposure::AutoExposureSettings;
use crate::auto_exposure::ExposureFrame;
use crate::auto_exposure::wasm2gpu::GuestFrame;
use crate::auto_exposure::wasm2gpu::GuestGpu;
use crate::auto_exposure::wasm2gpu::GuestInputs;
use crate::auto_exposure::wasm2gpu::GuestOutput;
use crate::auto_exposure::wasm2gpu::lower_guest;

use super::build_histogram;
use super::hdr_texture_from_rgba;
use super::histogram_bin;
use super::log_luminance;
use super::read_exposure;
use super::read_histogram;
use super::try_device;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 4;
const PREVIOUS: f32 = 0.75;
const DT: f32 = 1.0 / 30.0;

struct Scene {
    name: &'static str,
    mode: u32,
    deferred: Vec<[f32; 4]>,
    forward: Vec<[f32; 4]>,
}

struct ProductionInputs<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    hdr: &'a wgpu::TextureView,
    hdr_fwd: &'a wgpu::TextureView,
    mode: u32,
    settings: AutoExposureSettings,
}

#[test]
fn wasm_guest_matches_production_histogram_and_exposure_for_scene_modes() {
    let lowered = lower_guest();
    // The shared bootstrap already panics under the CI gate; keep the
    // requirement explicit at this parity test's skip boundary as well.
    let Some((device, queue)) = try_device() else {
        assert!(
            !crate::gpu_required(),
            "ORNIS_REQUIRE_GPU is set but auto-exposure parity has no wgpu device"
        );
        return;
    };
    let scenes = scenes();
    let guest = GuestGpu::new(&device, &lowered, WIDTH, HEIGHT);
    let settings = settings();
    for scene in scenes {
        check_scene(&device, &queue, &guest, scene, settings);
    }
}

fn check_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    guest: &GuestGpu,
    scene: Scene,
    settings: AutoExposureSettings,
) {
    let hdr = hdr_texture_from_rgba(device, queue, WIDTH, HEIGHT, &scene.deferred);
    let hdr_fwd = hdr_texture_from_rgba(device, queue, WIDTH, HEIGHT, &scene.forward);
    let inputs = GuestInputs {
        width: WIDTH,
        height: HEIGHT,
        mode: scene.mode,
        deferred: &scene.deferred,
        forward: &scene.forward,
    };
    let production = ProductionInputs {
        device,
        queue,
        hdr: &hdr,
        hdr_fwd: &hdr_fwd,
        mode: scene.mode,
        settings,
    };
    let production_histogram = production_histogram(&production);
    let guest_result: GuestOutput = guest
        .run(
            device,
            queue,
            GuestFrame {
                inputs,
                settings,
                dt: DT,
                previous_exposure: PREVIOUS,
            },
        )
        .expect("guest readback should match its allocation");
    assert_eq!(
        production_histogram, guest_result.histogram,
        "{} histogram: production vs wasm guest",
        scene.name,
    );
    assert_eq!(
        production_histogram,
        build_histogram(&selected_rgb(&scene), &settings),
        "{} histogram: production vs CPU reference",
        scene.name,
    );
    let production_exposure = production_exposure(&production);
    let error = (production_exposure - guest_result.exposure).abs();
    assert!(
        error <= 5e-4,
        "{} exposure: production={production_exposure} wasm={} error={error}",
        scene.name,
        guest_result.exposure,
    );
}

fn production_histogram(inputs: &ProductionInputs<'_>) -> [u32; super::HISTOGRAM_BINS] {
    let pass = AutoExposure::new(inputs.device);
    let mut encoder = inputs
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure production histogram parity"),
        });
    pass.encode_histogram_only(
        inputs.device,
        inputs.queue,
        &mut encoder,
        ExposureFrame {
            hdr: inputs.hdr,
            hdr_fwd: inputs.hdr_fwd,
            mode: inputs.mode,
            width: WIDTH,
            height: HEIGHT,
            settings: inputs.settings,
            dt_seconds: DT,
        },
    );
    inputs.queue.submit(std::iter::once(encoder.finish()));
    read_histogram(inputs.device, inputs.queue, &pass)
}

fn production_exposure(inputs: &ProductionInputs<'_>) -> f32 {
    let pass = AutoExposure::new(inputs.device);
    pass.write_manual(inputs.queue, PREVIOUS);
    let mut encoder = inputs
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure production exposure parity"),
        });
    pass.encode(
        inputs.device,
        inputs.queue,
        &mut encoder,
        ExposureFrame {
            hdr: inputs.hdr,
            hdr_fwd: inputs.hdr_fwd,
            mode: inputs.mode,
            width: WIDTH,
            height: HEIGHT,
            settings: inputs.settings,
            dt_seconds: DT,
        },
    );
    inputs.queue.submit(std::iter::once(encoder.finish()));
    read_exposure(inputs.device, inputs.queue, &pass)
}

fn selected_rgb(scene: &Scene) -> Vec<[f32; 3]> {
    scene
        .deferred
        .iter()
        .zip(&scene.forward)
        .map(|(deferred, forward)| {
            let scale = forward[3];
            match scene.mode {
                1 => [forward[0] * scale, forward[1] * scale, forward[2] * scale],
                2 => [
                    deferred[0] + forward[0] * scale,
                    deferred[1] + forward[1] * scale,
                    deferred[2] + forward[2] * scale,
                ],
                _ => [deferred[0], deferred[1], deferred[2]],
            }
        })
        .collect()
}

fn scenes() -> Vec<Scene> {
    let levels = [0.0, 0.005, 0.02, 0.04, 0.12, 0.18, 0.7, 1.3, 8.0, 80.0];
    let deferred: Vec<[f32; 4]> = (0..(WIDTH * HEIGHT) as usize)
        .map(|index| {
            let value = levels[index % levels.len()];
            [value, value, value, 1.0]
        })
        .collect();
    let forward: Vec<[f32; 4]> = (0..(WIDTH * HEIGHT) as usize)
        .map(|index| {
            let value = levels[(index * 3 + 4) % levels.len()];
            let alpha = [0.0, 0.2, 0.5, 0.85][index % 4];
            [value, value * 0.8, value * 0.6, alpha]
        })
        .collect();
    let mut non_finite = deferred.clone();
    non_finite[0] = [f32::NAN, 0.0, 0.0, 1.0];
    non_finite[1] = [f32::INFINITY, 0.0, 0.0, 1.0];
    non_finite[2] = [f32::NEG_INFINITY, 0.0, 0.0, 1.0];
    // Non-finite samples in the layer the mode does *not* select must not
    // reach the chosen texel: multiply-masking (`NaN * 0.0`) would poison it
    // and silently drop a valid sample into bin 0.
    // Pixel 3 has non-black deferred RGB and non-zero forward RGB/alpha.
    // Pixel 0 does not: NaN poisoning there would still produce bin 0,
    // allowing a broken multiply-mask to pass the histogram comparison.
    let mut bad_unselected_deferred = non_finite.clone();
    bad_unselected_deferred[0] = deferred[0];
    bad_unselected_deferred[3] = [f32::NAN, 0.0, 0.0, 1.0];
    let mut bad_unselected_forward = forward.clone();
    bad_unselected_forward[3] = [f32::NAN, 0.0, 0.0, 1.0];
    bad_unselected_forward[1] = [f32::INFINITY, 0.0, 0.0, 1.0];
    bad_unselected_forward[2] = [f32::NEG_INFINITY, 0.0, 0.0, 1.0];
    vec![
        Scene {
            name: "deferred HDR with non-finite guards",
            mode: 0,
            deferred: non_finite.clone(),
            forward: forward.clone(),
        },
        Scene {
            name: "forward-only alpha coverage",
            mode: 1,
            deferred: deferred.clone(),
            forward: forward.clone(),
        },
        Scene {
            name: "hybrid clipped HDR highlights",
            mode: 2,
            deferred: deferred.clone(),
            forward: forward.clone(),
        },
        Scene {
            name: "forward mode ignores non-finite deferred",
            mode: 1,
            deferred: bad_unselected_deferred,
            forward: forward.clone(),
        },
        Scene {
            name: "deferred mode ignores non-finite forward",
            mode: 0,
            deferred,
            forward: bad_unselected_forward,
        },
    ]
}

fn settings() -> AutoExposureSettings {
    AutoExposureSettings {
        target_luminance: 0.22,
        ev_compensation: 0.35,
        min_ev: -6.0,
        max_ev: 6.0,
        min_log_lum: -7.0,
        max_log_lum: 7.0,
        low_percentile: 0.12,
        high_percentile: 0.88,
        brighten_speed: 2.5,
        darken_speed: 1.25,
        exponential_transition_distance: 1.25,
    }
    .sanitized()
}

/// Keep the NaN regression independent of the infinity guards: poisoning the
/// selected sample must change its histogram bin in *both* selection modes.
#[test]
fn unselected_nan_faces_a_nonzero_histogram_bin() {
    let settings = settings();
    let mut modes = Vec::new();
    for scene in scenes() {
        let selected = selected_rgb(&scene);
        let unselected = match scene.mode {
            0 => &scene.forward,
            1 => &scene.deferred,
            _ => continue,
        };
        for (index, texel) in unselected.iter().enumerate() {
            if texel[..3].iter().any(|value| value.is_nan()) {
                assert!(
                    histogram_bin(log_luminance(selected[index]), &settings) > 0,
                    "{} pixel {index}: NaN poisoning must not also land in bin 0",
                    scene.name
                );
                modes.push(scene.mode);
            }
        }
    }
    modes.sort_unstable();
    assert_eq!(modes, [0, 1], "each selection mode needs a NaN regression");
}
