//! Unit + GPU tests for the auto-exposure pass.
//!
//! Compiled only under `cfg(test)`: the `#[cfg(test)]` marker below keeps the
//! module's production-line count at zero for the size/SRP gate, exactly like
//! an inline `#[cfg(test)] mod tests`.

#[cfg(test)]
use super::*;

/// Adapter handle, or `None` on headless hosts without a GPU (the skip is
/// logged, never silent).
///
/// Shared bootstrap [`crate::try_headless_device`]: when
/// [`crate::gpu_required`] holds (`ORNIS_REQUIRE_GPU` — non-empty, not
/// `"0"`; the CI `test` shard sets `=1`) a missing adapter panics instead
/// of skipping, so a missing device can never turn a GPU gate into a pass.
fn try_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let pair = crate::try_headless_device();
    if pair.is_none() {
        eprintln!("SKIPPED (auto_exposure): no wgpu device");
    }
    pair
}

#[test]
fn exposure_params_layout_is_a_16_byte_uniform_block() {
    assert_eq!(std::mem::size_of::<ExposureParams>(), 16);
    assert_eq!(std::mem::offset_of!(ExposureParams, exposure), 0);
    let packed = ExposureParams::pack(2.0);
    assert_eq!(packed.exposure, 2.0);
    assert_eq!(packed.pad, [0.0; 3]);
    assert_eq!(bytemuck::bytes_of(&packed).len(), 16);
    assert!(EXPOSURE_PARAMS_WGSL.contains("exposure: f32"));
    assert!(EXPOSURE_PARAMS_WGSL.contains(EXPOSURE_WGSL_NAME));
}

#[test]
fn params_layouts_match_their_wgsl() {
    assert_eq!(std::mem::size_of::<HistogramParams>(), 16);
    assert_eq!(std::mem::size_of::<AdaptParams>(), 48);
    assert_eq!(std::mem::offset_of!(AdaptParams, min_log_lum), 0);
    assert_eq!(std::mem::offset_of!(AdaptParams, transition_distance), 36);
    assert_eq!(std::mem::offset_of!(AdaptParams, dt), 40);
    let hist = histogram_wgsl();
    assert!(hist.contains("mode: u32"));
    assert!(hist.contains("array<atomic<u32>, 256u>"));
    assert!(hist.contains("var<workgroup> histogram_shared"));
    assert!(hist.contains("forward_tex"));
    let adapt = adapt_wgsl();
    assert!(adapt.contains("brighten_speed: f32"));
    assert!(adapt.contains("darken_speed: f32"));
    assert!(adapt.contains("array<u32, 256u>"));
    assert!(adapt.contains("exposure_out.exposure"));
    // Bin 0 is excluded from the reduction.
    assert!(adapt.contains("var i = 1u; i < HISTOGRAM_BINS"));
}

#[test]
fn shaders_validate_with_naga() {
    for (name, source) in [("histogram", histogram_wgsl()), ("adapt", adapt_wgsl())] {
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{name} compute shader must parse: {e}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} compute shader must validate: {e}"));
    }
}

#[test]
fn manual_mode_defaults_to_the_pixel_identity_no_op() {
    assert_eq!(
        ExposureMode::default(),
        ExposureMode::Manual { multiplier: 1.0 }
    );
    assert_eq!(ExposureParams::pack(1.0).exposure, 1.0);
}

#[test]
fn settings_sanitize_replaces_nan_and_orders_the_knobs() {
    let broken = AutoExposureSettings {
        target_luminance: f32::NAN,
        ev_compensation: f32::INFINITY,
        min_ev: f32::NAN,
        max_ev: -100.0,
        min_log_lum: f32::NAN,
        max_log_lum: f32::NAN,
        low_percentile: 0.9,
        high_percentile: 0.1,
        brighten_speed: f32::NAN,
        darken_speed: -5.0,
        exponential_transition_distance: f32::NAN,
    }
    .sanitized();
    let default = AutoExposureSettings::default();
    assert_eq!(broken.target_luminance, default.target_luminance);
    assert_eq!(broken.ev_compensation, 0.0);
    assert_eq!(broken.min_ev, default.min_ev);
    assert!(broken.max_ev >= broken.min_ev);
    assert_eq!(broken.min_log_lum, default.min_log_lum);
    assert!(broken.max_log_lum >= broken.min_log_lum);
    assert!(broken.low_percentile < broken.high_percentile);
    assert_eq!(broken.brighten_speed, default.brighten_speed);
    assert!(broken.darken_speed >= MIN_ADAPTATION_SPEED);
    assert_eq!(
        broken.exponential_transition_distance,
        default.exponential_transition_distance
    );
}

#[test]
fn histogram_reserves_bin_zero_for_below_range_texels() {
    let settings = AutoExposureSettings::default();
    // Black is far below the window: bin 0.
    assert_eq!(histogram_bin(log_luminance([0.0, 0.0, 0.0]), &settings), 0);
    // In-range luminance never lands in bin 0.
    let mid = log_luminance([0.5, 0.5, 0.5]);
    let bin = histogram_bin(mid, &settings);
    assert!((1..HISTOGRAM_BINS).contains(&bin), "bin {bin}");
    // `2^20` is far above the window: the last bin.
    assert_eq!(
        histogram_bin(log_luminance([1.0e6, 1.0e6, 1.0e6]), &settings),
        HISTOGRAM_BINS - 1
    );
}

/// A non-finite texel is metered as black on both sides: the CPU floors it
/// to the luminance floor, and the WGSL guard
/// (`luma != luma || abs(luma) > 3.4e38` — the `isFinite` replacement this
/// WGSL frontend lacks) does the same, so neither a NaN nor a ±Inf can land
/// in an in-range bin and diverge from the host reference. Metering `+Inf`
/// as black is deliberate: it is treated like any other non-finite sample,
/// not as an over-range hit, keeping GPU and CPU bit-identical.
#[test]
fn non_finite_texels_are_metered_as_black() {
    let settings = AutoExposureSettings::default();
    let black = log_luminance([0.0, 0.0, 0.0]);
    for bad in [
        [f32::NAN, 0.0, 0.0],
        [0.0, f32::NAN, 0.0],
        [0.0, 0.0, f32::NAN],
        [f32::NAN, f32::NAN, f32::NAN],
        [f32::INFINITY, 0.0, 0.0],
        [0.0, f32::NEG_INFINITY, 0.0],
    ] {
        assert_eq!(
            log_luminance(bad),
            black,
            "non-finite texel {bad:?} must meter as black"
        );
        assert_eq!(histogram_bin(log_luminance(bad), &settings), 0);
    }
    // The GPU histogram agrees: the NaN and the ±Inf texels land in bin 0
    // next to the mid-grey texel's own bin. The infinities matter most —
    // without the magnitude guard `+Inf` would slip into the last bin as a
    // phantom over-range sample.
    if let Some((device, queue)) = try_device() {
        const W: u32 = 4;
        const H: u32 = 1;
        let texels = [
            [f32::NAN; 3],
            [f32::INFINITY; 3],
            [f32::NEG_INFINITY; 3],
            [0.5; 3],
        ];
        let view = hdr_texture_from_texels(&device, &queue, W, H, &texels);
        let pass = AutoExposure::new(&device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure non-finite histogram encoder"),
        });
        pass.encode_histogram_only(
            &device,
            &queue,
            &mut encoder,
            ExposureFrame {
                hdr: &view,
                hdr_fwd: &view,
                mode: 0,
                width: W,
                height: H,
                settings,
                dt_seconds: DEFAULT_FRAME_DT_SECONDS,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        let gpu = read_histogram(&device, &queue, &pass);
        let mid_bin = histogram_bin(log_luminance([0.5; 3]), &settings);
        assert_eq!(gpu, build_histogram(&texels, &settings));
        assert_eq!(gpu[0], 3, "the NaN/+Inf/-Inf texels must land in bin 0");
        assert_eq!(gpu[mid_bin], 1, "the mid-grey texel must land in its bin");
    }
}

#[test]
fn bin_log_luminance_matches_the_analytic_bin_center() {
    let settings = AutoExposureSettings::default();
    for index in [1_usize, 8, 100, 200, 255] {
        let reconstructed = bin_log_luminance(index, &settings);
        // Invert the WGSL mapping: `bin = floor(t*(BINS-2)) + 1`.
        let t_lo = (index - 1) as f32 / IN_RANGE_BINS;
        let t_hi = index as f32 / IN_RANGE_BINS;
        let lo = settings.min_log_lum + t_lo * settings.log_range();
        let hi = settings.min_log_lum + t_hi.min(1.0) * settings.log_range();
        assert!(
            reconstructed >= lo - 1e-3 && reconstructed <= hi + 1e-3,
            "bin {index}: {reconstructed} not in [{lo}, {hi}]"
        );
    }
}

#[test]
fn average_log_luminance_recovers_a_uniform_scene() {
    let settings = AutoExposureSettings::default();
    let lum = 0.5_f32;
    let texels = vec![[lum; 3]; 4096];
    let hist = build_histogram(&texels, &settings);
    let avg = average_log_luminance(&hist, &settings).expect("non-empty histogram");
    assert!(
        (avg - lum.log2()).abs() < 0.05,
        "avg={avg} expected≈{}",
        lum.log2()
    );
}

#[test]
fn below_range_texels_are_excluded_from_the_average() {
    let settings = AutoExposureSettings::default();
    // A huge black field plus a few mid-grey texels: the average must track
    // the grey texels, not be dragged toward the bin-0 edge.
    let mut texels = vec![[0.0_f32; 3]; 100_000];
    texels.extend(std::iter::repeat_n([0.5_f32; 3], 50));
    let hist = build_histogram(&texels, &settings);
    let avg = average_log_luminance(&hist, &settings).expect("non-empty histogram");
    assert!(
        (avg - 0.5_f32.log2()).abs() < 0.05,
        "avg={avg} should track the grey texels"
    );
}

#[test]
fn an_all_below_range_histogram_has_no_average() {
    let settings = AutoExposureSettings::default();
    let texels = vec![[0.0_f32; 3]; 64];
    let hist = build_histogram(&texels, &settings);
    assert_eq!(average_log_luminance(&hist, &settings), None);
}

#[test]
fn tail_clipping_ignores_a_bright_minority() {
    let settings = AutoExposureSettings::default();
    // A dark field with a bright minority that lies inside the clipped tail
    // (< the top 10% of samples), so clipping removes it entirely.
    let mut texels = vec![[0.01_f32; 3]; 1000];
    texels.extend(std::iter::repeat_n([1000.0_f32; 3], 50));
    let hist = build_histogram(&texels, &settings);
    let avg = average_log_luminance(&hist, &settings).expect("non-empty histogram");
    assert!((avg - 0.01_f32.log2()).abs() < 0.2, "avg={avg}");

    // Without clipping the bright texels pull the average up sharply;
    // clipping must remove that gain.
    let unclipped = AutoExposureSettings {
        low_percentile: 0.0,
        high_percentile: 1.0,
        ..settings
    };
    let avg_unclipped = average_log_luminance(&hist, &unclipped).expect("non-empty histogram");
    assert!(
        avg_unclipped > avg + 0.5,
        "clipping barely changed the average: clipped {avg} vs unclipped {avg_unclipped}"
    );
}

#[test]
fn desired_exposure_ev_targets_the_key_luminance() {
    let settings = AutoExposureSettings::default();
    // A scene at the target luminance needs exposure 1.0 (0 EV).
    let ev = desired_exposure_ev(settings.target_luminance.log2(), &settings);
    assert!(ev.abs() < 1e-3, "{ev}");
    // A ten-times darker scene needs ~+3.3 EV.
    let darker = desired_exposure_ev((settings.target_luminance / 10.0).log2(), &settings);
    assert!((darker - 10.0_f32.log2()).abs() < 1e-3, "{darker}");
}

#[test]
fn desired_exposure_respects_the_ev_clamp() {
    let settings = AutoExposureSettings {
        min_ev: -2.0,
        max_ev: 2.0,
        ..Default::default()
    };
    // An almost black scene cannot push exposure past 2^2.
    assert_eq!(desired_exposure_ev(-20.0, &settings), 2.0);
    // A blown-out scene cannot push it below 2^-2.
    assert_eq!(desired_exposure_ev(20.0, &settings), -2.0);
    assert_eq!(desired_exposure(-20.0, &settings), settings.max_exposure());
}

#[test]
fn adaptation_step_is_clamped_to_speed_times_dt() {
    let settings = AutoExposureSettings::default();
    // A large delta saturates at `speed * dt` (linear regime is
    // `|delta| * speed * dt / transition_distance`).
    let step = adapt_exposure_ev(0.0, 20.0, 1.0 / 60.0, &settings);
    let expected = settings.brighten_speed / 60.0;
    assert!((step - expected).abs() < 1e-4, "step {step} vs {expected}");

    // A small delta stays in the linear regime and moves by less than the
    // saturated step.
    let small = adapt_exposure_ev(0.0, 0.1, 1.0 / 60.0, &settings);
    assert!(small > 0.0 && small < expected, "small step {small}");

    // Darkening uses `darken_speed`.
    let dark = adapt_exposure_ev(2.0, -2.0, 1.0 / 60.0, &settings);
    assert!(
        (dark - (2.0 - settings.darken_speed / 60.0)).abs() < 1e-4,
        "dark step {dark}"
    );

    // Repeated steps converge to the target.
    let mut ev = 0.0;
    for _ in 0..6000 {
        ev = adapt_exposure_ev(ev, 4.0, 1.0 / 60.0, &settings);
    }
    assert!((ev - 4.0).abs() < 1e-2, "{ev}");
}

/// Both ends of the smoothing step clamp into the EV window, mirroring the
/// `prev_ev`/`target_ev` clamps of the WGSL adapt pass: a previous exposure
/// or a target outside the window behaves exactly like the clamped value,
/// and every clamped pair stays inside the window (no final clamp needed).
#[test]
fn adapt_exposure_ev_clamps_both_ends_into_the_ev_window() {
    let settings = AutoExposureSettings {
        min_ev: -1.0,
        max_ev: 1.0,
        ..AutoExposureSettings::default()
    };
    let dt = 1.0 / 60.0;
    // A previous exposure far above the window behaves as if it started at
    // the window's top edge (e.g. a switch out of a Manual multiplier of
    // 100.0, or a window narrowed under a running Auto).
    let from_outside = adapt_exposure_ev(6.64, -0.4, dt, &settings);
    let from_edge = adapt_exposure_ev(1.0, -0.4, dt, &settings);
    assert_eq!(from_outside, from_edge, "prev_ev must be clamped");
    // A target far below the window is clamped to the bottom edge before
    // the step is taken, so the step follows the clamped target.
    let along_raw = adapt_exposure_ev(0.0, -9.1, dt, &settings);
    let along_clamped = adapt_exposure_ev(0.0, -1.0, dt, &settings);
    assert_eq!(along_raw, along_clamped, "target_ev must be clamped");
    // Whatever the ends, the result is inside the window.
    for ev in [from_outside, along_raw] {
        assert!(
            (settings.min_ev..=settings.max_ev).contains(&ev),
            "adapted {ev} escaped the EV window"
        );
    }
}

/// Uploads an `Rgba32Float` texture holding `texels` (opaque alpha).
fn hdr_texture_from_texels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    texels: &[[f32; 3]],
) -> wgpu::TextureView {
    let opaque: Vec<[f32; 4]> = texels.iter().map(|t| [t[0], t[1], t[2], 1.0]).collect();
    hdr_texture_from_rgba(device, queue, width, height, &opaque)
}

/// Uploads an `Rgba32Float` texture holding `texels` with their own alpha
/// (the forward layer carries coverage: the histogram meters `rgb * a`).
fn hdr_texture_from_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    texels: &[[f32; 4]],
) -> wgpu::TextureView {
    assert_eq!(texels.len(), (width * height) as usize);
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("auto-exposure test hdr"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
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
        bytemuck::cast_slice(texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4 * 4),
            rows_per_image: Some(height),
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// A constant-luminance HDR view.
fn constant_hdr(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    rgb: [f32; 3],
) -> wgpu::TextureView {
    let texels = vec![rgb; (width * height) as usize];
    hdr_texture_from_texels(device, queue, width, height, &texels)
}

/// Reads the exposure multiplier back from the shared uniform.
fn read_exposure(device: &wgpu::Device, queue: &wgpu::Queue, pass: &AutoExposure) -> f32 {
    let value = read_buffer(device, queue, pass.exposure_buffer(), 16);
    f32::from_le_bytes([value[0], value[1], value[2], value[3]])
}

/// Reads the histogram counts back from the pass buffer.
fn read_histogram(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &AutoExposure,
) -> [u32; HISTOGRAM_BINS] {
    let size = (HISTOGRAM_BINS * std::mem::size_of::<u32>()) as u64;
    let bytes = read_buffer(device, queue, &pass.histogram_buffer, size);
    let mut hist = [0_u32; HISTOGRAM_BINS];
    for (i, slot) in hist.iter_mut().enumerate() {
        let o = i * 4;
        *slot = u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
    }
    hist
}

/// Copies `size` bytes from the start of `src` into a mappable buffer and
/// returns them.
fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("auto-exposure test readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("auto-exposure test readback encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let bytes = slice.get_mapped_range().expect("mapped range").to_vec();
    readback.unmap();
    bytes
}

/// One auto-exposure update: the two HDR layers, the shader mix mode, the
/// dispatch size, the settings and the frame delta.
#[derive(Clone, Copy)]
struct Meter<'a> {
    hdr: &'a wgpu::TextureView,
    hdr_fwd: &'a wgpu::TextureView,
    mode: u32,
    width: u32,
    height: u32,
    settings: &'a AutoExposureSettings,
    dt: f32,
}

impl<'a> Meter<'a> {
    /// Meters `view` as both layers (deferred mode) `dt` seconds apart.
    fn deferred(
        view: &'a wgpu::TextureView,
        settings: &'a AutoExposureSettings,
        width: u32,
        height: u32,
        dt: f32,
    ) -> Self {
        Self {
            hdr: view,
            hdr_fwd: view,
            mode: 0,
            width,
            height,
            settings,
            dt,
        }
    }
}

/// Runs one auto-exposure update and returns the exposure it wrote.
fn advance(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &AutoExposure,
    meter: Meter<'_>,
) -> f32 {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("auto-exposure test encoder"),
    });
    pass.encode(
        device,
        queue,
        &mut encoder,
        ExposureFrame {
            hdr: meter.hdr,
            hdr_fwd: meter.hdr_fwd,
            mode: meter.mode,
            width: meter.width,
            height: meter.height,
            settings: *meter.settings,
            dt_seconds: meter.dt,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    read_exposure(device, queue, pass)
}

/// Runs `frames` identical updates and returns the exposure after the last.
fn advance_n(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pass: &AutoExposure,
    meter: Meter<'_>,
    frames: u32,
) -> f32 {
    let mut exposure = 1.0;
    for _ in 0..frames {
        exposure = advance(device, queue, pass, meter);
    }
    exposure
}

/// The exposure the CPU reference maps the histogram of `texels` onto.
fn cpu_target(texels: &[[f32; 3]], settings: &AutoExposureSettings) -> f32 {
    let hist = build_histogram(texels, settings);
    let avg = average_log_luminance(&hist, settings).expect("texels sit inside the window");
    desired_exposure(avg, settings)
}

/// Uploads `texels` as one HDR layer, advances the exposure one saturated
/// step, and returns `(exposure, relative_error_vs_cpu_target)`. The shared
/// body of every GPU-vs-CPU conformance check.
fn gpu_saturated_exposure(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    size: (u32, u32),
    texels: &[[f32; 3]],
    settings: &AutoExposureSettings,
) -> (f32, f32) {
    // A huge dt saturates the step, so one update lands exactly on the
    // target without temporal dependence.
    const SATURATED_DT: f32 = 100.0;
    let (width, height) = size;
    let view = hdr_texture_from_texels(device, queue, width, height, texels);
    let gpu = advance(
        device,
        queue,
        &AutoExposure::new(device),
        Meter::deferred(&view, settings, width, height, SATURATED_DT),
    );
    let expected = cpu_target(texels, settings);
    (gpu, (gpu - expected).abs() / expected)
}

/// Two constant-luminance HDR scenes (a metering pair).
fn scenes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    lums: [[f32; 3]; 2],
) -> (wgpu::TextureView, wgpu::TextureView) {
    (
        constant_hdr(device, queue, width, height, lums[0]),
        constant_hdr(device, queue, width, height, lums[1]),
    )
}

/// Builds a `width × height` gradient whose texels sit at bin centres, with a
/// mix of grey and coloured texels (so the luma weights matter), spanning
/// below-range, in-range and above-range luminance.
fn gradient_texels(width: u32, height: u32, settings: &AutoExposureSettings) -> Vec<[f32; 3]> {
    let directions = [
        [1.0_f32, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.5, 0.25],
        [0.2, 0.8, 0.5],
    ];
    let count = (width * height) as usize;
    let mut texels = Vec::with_capacity(count);
    for i in 0..count {
        let direction = directions[i % directions.len()];
        let weight: f32 = direction
            .iter()
            .zip(REC709_LUMA.iter())
            .map(|(d, w)| d * w)
            .sum();
        // Sweep the whole window, plus below-range and above-range samples.
        let bin = 1 + (i % (HISTOGRAM_BINS - 2));
        let lum = if i % 7 == 0 {
            2.0_f32.powf(-20.0)
        } else if i % 11 == 0 {
            2.0_f32.powf(20.0)
        } else {
            2.0_f32.powf(bin_log_luminance(bin, settings))
        };
        let scale = lum / weight;
        texels.push([
            direction[0] * scale,
            direction[1] * scale,
            direction[2] * scale,
        ]);
    }
    texels
}

#[test]
fn gpu_and_cpu_histograms_match_bin_for_bin() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    // Non-square, not a multiple of the workgroup size (8), coloured texels.
    const W: u32 = 19;
    const H: u32 = 13;
    let settings = AutoExposureSettings::default();
    let texels = gradient_texels(W, H, &settings);
    let view = hdr_texture_from_texels(&device, &queue, W, H, &texels);
    let cpu = build_histogram(&texels, &settings);

    let pass = AutoExposure::new(&device);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("auto-exposure histogram test encoder"),
    });
    pass.encode_histogram_only(
        &device,
        &queue,
        &mut encoder,
        ExposureFrame {
            hdr: &view,
            hdr_fwd: &view,
            mode: 0,
            width: W,
            height: H,
            settings,
            dt_seconds: DEFAULT_FRAME_DT_SECONDS,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    let gpu = read_histogram(&device, &queue, &pass);

    assert_eq!(
        gpu.iter().sum::<u32>(),
        (W * H),
        "every texel must be counted exactly once"
    );
    assert_eq!(gpu, cpu, "GPU histogram must match the CPU reference");
}

#[test]
fn gpu_reduction_matches_the_cpu_reduction_on_a_gradient() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 19;
    const H: u32 = 13;
    let settings = AutoExposureSettings::default();
    let texels = gradient_texels(W, H, &settings);

    // A huge dt saturates the step, so one update lands exactly on the target
    // without temporal dependence.
    let (gpu, relative) = gpu_saturated_exposure(&device, &queue, (W, H), &texels, &settings);
    assert!(
        relative < 0.01,
        "gpu exposure {gpu} vs cpu-derived {}",
        cpu_target(&texels, &settings)
    );
}

#[test]
fn gpu_holds_the_previous_exposure_on_a_black_frame() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    let black = constant_hdr(&device, &queue, W, H, [0.0; 3]);
    let pass = AutoExposure::new(&device);
    pass.write_manual(&queue, 2.0);
    let exposure = advance_n(
        &device,
        &queue,
        &pass,
        Meter::deferred(&black, &settings, W, H, 1.0 / 60.0),
        8,
    );
    assert_eq!(
        exposure, 2.0,
        "a black frame must hold the previous exposure"
    );
}

#[test]
fn gpu_meters_a_gray_object_on_a_black_background() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 32;
    const H: u32 = 32;
    let settings = AutoExposureSettings::default();
    // Mostly black, a small mid-grey object: the exposure must follow the
    // object, not the black background.
    let mut texels = vec![[0.0_f32; 3]; (W * H) as usize];
    for texel in texels.iter_mut().take(200) {
        *texel = [0.5, 0.5, 0.5];
    }
    let view = hdr_texture_from_texels(&device, &queue, W, H, &texels);
    let expected = cpu_target(&texels, &settings);
    let pass = AutoExposure::new(&device);
    let exposure = advance(
        &device,
        &queue,
        &pass,
        Meter::deferred(&view, &settings, W, H, 100.0),
    );
    // The mid-grey object is brighter than the `0.18` key, so the exposure
    // must fall well below `1.0` instead of being dragged toward the black
    // background (which, if bin 0 leaked in, would push it far above `1.0`).
    assert!(exposure < 1.0, "object must darken the frame: {exposure}");
    let relative = (exposure - expected).abs() / expected;
    assert!(
        relative < 0.02,
        "exposure {exposure} vs expected {expected}"
    );
}

#[test]
fn gpu_clears_the_histogram_between_frames() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    let bright = constant_hdr(&device, &queue, W, H, [4.0; 3]);
    let dark = constant_hdr(&device, &queue, W, H, [0.01; 3]);

    // Sequence: bright then dark.
    let sequenced = AutoExposure::new(&device);
    advance(
        &device,
        &queue,
        &sequenced,
        Meter::deferred(&bright, &settings, W, H, 100.0),
    );
    let after = advance(
        &device,
        &queue,
        &sequenced,
        Meter::deferred(&dark, &settings, W, H, 100.0),
    );

    // Reference: the dark frame alone on a fresh pass.
    let fresh = AutoExposure::new(&device);
    let reference = advance_n(
        &device,
        &queue,
        &fresh,
        Meter::deferred(&dark, &settings, W, H, 100.0),
        1,
    );

    let relative = (after - reference).abs() / reference;
    assert!(
        relative < 0.02,
        "stale counts leaked into the next frame: {after} vs {reference}"
    );
}

#[test]
fn gpu_hybrid_mode_meters_the_forward_layer() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    let (black, forward) = scenes(&device, &queue, W, H, [[0.0; 3], [4.0; 3]]);

    // Deferred-only on a black layer: the exposure is held.
    let deferred_only = AutoExposure::new(&device);
    let held = advance(
        &device,
        &queue,
        &deferred_only,
        Meter::deferred(&black, &settings, W, H, 100.0),
    );
    assert_eq!(held, 1.0);

    // Hybrid: a bright forward layer over an empty deferred layer must be
    // metered (the frame darkens).
    let hybrid = AutoExposure::new(&device);
    let exposure = advance(
        &device,
        &queue,
        &hybrid,
        Meter {
            hdr: &black,
            hdr_fwd: &forward,
            mode: 2,
            width: W,
            height: H,
            settings: &settings,
            dt: 100.0,
        },
    );
    assert!(
        exposure < 0.5,
        "bright forward geometry must be metered: {exposure}"
    );
}

#[test]
fn dark_and_bright_scenes_converge_to_the_target_luminance() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 32;
    const H: u32 = 32;
    let settings = AutoExposureSettings::default();
    for luminance in [0.01_f32, 2.0_f32] {
        let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);
        let pass = AutoExposure::new(&device);
        let exposure = advance_n(
            &device,
            &queue,
            &pass,
            Meter::deferred(&view, &settings, W, H, DEFAULT_FRAME_DT_SECONDS),
            600,
        );
        let expected = settings.target_luminance / luminance;
        let relative = (exposure - expected).abs() / expected;
        assert!(
            relative < 0.15,
            "luminance {luminance}: exposure {exposure} vs expected {expected}"
        );
    }
}

#[test]
fn adaptation_speed_scales_with_dt() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    // A scene slightly below the target (darker than the `0.18` key, so the
    // exposure rises) with a small enough delta to stay in the linear
    // regime (`|delta| < transition_distance`).
    let luminance = 0.127_f32;
    let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);

    let step_ev = |dt: f32| {
        let pass = AutoExposure::new(&device);
        advance(
            &device,
            &queue,
            &pass,
            Meter::deferred(&view, &settings, W, H, dt),
        )
        .log2()
    };
    let slow = step_ev(1.0 / 60.0);
    let fast = step_ev(1.0 / 30.0);
    assert!(slow > 0.0, "scene is darker than the key: {slow}");
    let ratio = fast / slow;
    assert!(
        (ratio - 2.0).abs() < 0.1,
        "doubling dt must double the EV step: {slow} -> {fast} (ratio {ratio})"
    );
}

/// The metered quantile window clips *partial* bins: with 1000 dark and 200
/// blown-out texels (1200 in-range samples, window `[120, 1080]`) only 880 of
/// the 1000 dark samples and 80 of the 200 bright ones are metered. A
/// whole-bin reduction would weight both bins by their full counts and land
/// more than a stop away.
#[test]
fn tail_clipping_uses_partial_bin_weights() {
    let settings = AutoExposureSettings::default();
    assert_eq!(
        (settings.low_percentile, settings.high_percentile),
        (0.10, 0.90)
    );
    let mut texels = vec![[0.01_f32; 3]; 1000];
    texels.extend(std::iter::repeat_n([1000.0_f32; 3], 200));
    let dark_bin = histogram_bin(log_luminance([0.01; 3]), &settings);
    let bright_bin = histogram_bin(log_luminance([1000.0; 3]), &settings);
    assert!((1..HISTOGRAM_BINS).contains(&dark_bin), "bin {dark_bin}");
    assert!(
        (1..HISTOGRAM_BINS).contains(&bright_bin),
        "bin {bright_bin}"
    );
    assert_ne!(dark_bin, bright_bin, "the two scenes must not merge");

    let hist = build_histogram(&texels, &settings);
    let avg = average_log_luminance(&hist, &settings).expect("both bins are in range");
    let exact = (880.0 * bin_log_luminance(dark_bin, &settings)
        + 80.0 * bin_log_luminance(bright_bin, &settings))
        / 960.0;
    assert!((avg - exact).abs() < 1e-3, "avg {avg} vs exact {exact}");

    let whole = (1000.0 * bin_log_luminance(dark_bin, &settings)
        + 200.0 * bin_log_luminance(bright_bin, &settings))
        / 1200.0;
    assert!(
        (avg - whole).abs() > 1.0,
        "avg {avg} must not be the whole-bin mean {whole}"
    );
}

/// GPU counterpart: one saturated step (`dt = 100 s`) lands exactly on the
/// partial-weight reference, and nowhere near the whole-bin mean.
#[test]
fn gpu_partial_bin_weights_match_the_clipped_reference() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    // 1200 texels — exactly the sample count the `[p10, p90]` window of
    // `[120, 1080]` above was derived for.
    const W: u32 = 60;
    const H: u32 = 20;
    let settings = AutoExposureSettings::default();
    let mut texels = vec![[0.01_f32; 3]; 1000];
    texels.extend(std::iter::repeat_n([1000.0_f32; 3], 200));
    let dark_bin = histogram_bin(log_luminance([0.01; 3]), &settings);
    let bright_bin = histogram_bin(log_luminance([1000.0; 3]), &settings);
    let exact = (880.0 * bin_log_luminance(dark_bin, &settings)
        + 80.0 * bin_log_luminance(bright_bin, &settings))
        / 960.0;
    let expected = desired_exposure(exact, &settings);

    let (gpu, relative) = gpu_saturated_exposure(&device, &queue, (W, H), &texels, &settings);
    assert!(
        relative < 0.01,
        "gpu exposure {gpu} vs partial-weight reference {expected}"
    );

    let whole = (1000.0 * bin_log_luminance(dark_bin, &settings)
        + 200.0 * bin_log_luminance(bright_bin, &settings))
        / 1200.0;
    let whole_ev = desired_exposure_ev(whole, &settings);
    assert!(
        (gpu.log2() - whole_ev).abs() > 1.0,
        "gpu {gpu} must sit over a stop from the whole-bin mean (ev {whole_ev})"
    );
}

/// Mode 1 (forward-only technique): the deferred layer is ignored and the
/// forward layer alone is metered — a black deferred layer plus a bright
/// forward layer must still adapt instead of holding the previous exposure.
#[test]
fn gpu_forward_only_mode_meters_the_forward_layer() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    let (black, forward) = scenes(&device, &queue, W, H, [[0.0; 3], [4.0; 3]]);
    let expected = cpu_target(&vec![[4.0_f32; 3]; (W * H) as usize], &settings);

    let exposure = advance(
        &device,
        &queue,
        &AutoExposure::new(&device),
        Meter {
            hdr: &black,
            hdr_fwd: &forward,
            mode: 1,
            width: W,
            height: H,
            settings: &settings,
            dt: 100.0,
        },
    );
    assert!(
        exposure < 0.5,
        "the forward layer must be metered on its own: {exposure}"
    );
    let relative = (exposure - expected).abs() / expected;
    assert!(
        relative < 0.02,
        "exposure {exposure} vs CPU reference {expected}"
    );
}

/// The forward layer carries coverage: the histogram must meter the
/// premultiplied `fwd.rgb * fwd.a` in both plans that sample it —
/// `mode == 1` (forward-only, the deferred layer is ignored) and
/// `mode == 2` (hybrid, deferred + forward). Dropping the alpha weight
/// moves the metered luminance and must fail.
#[test]
fn gpu_forward_layer_is_metered_through_its_alpha() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    let samples = (W * H) as usize;
    let forward = hdr_texture_from_rgba(
        &device,
        &queue,
        W,
        H,
        &vec![[4.0_f32, 4.0, 4.0, 0.5]; samples],
    );

    // A fresh pass per measurement: the build pass accumulates into the
    // histogram buffer, and only the adapt pass clears it.
    let meter = |hdr: &wgpu::TextureView, mode: u32| -> [u32; HISTOGRAM_BINS] {
        let pass = AutoExposure::new(&device);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("auto-exposure alpha histogram encoder"),
        });
        pass.encode_histogram_only(
            &device,
            &queue,
            &mut encoder,
            ExposureFrame {
                hdr,
                hdr_fwd: &forward,
                mode,
                width: W,
                height: H,
                settings,
                dt_seconds: DEFAULT_FRAME_DT_SECONDS,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));
        read_histogram(&device, &queue, &pass)
    };

    // `mode == 1`: forward-only plans meter the forward layer alone.
    let black = constant_hdr(&device, &queue, W, H, [0.0; 3]);
    let gpu = meter(&black, 1);
    let premultiplied = vec![[2.0_f32; 3]; samples];
    assert_eq!(
        gpu,
        build_histogram(&premultiplied, &settings),
        "forward-only must meter fwd.rgb * fwd.a"
    );
    let straight = vec![[4.0_f32; 3]; samples];
    assert_ne!(
        gpu,
        build_histogram(&straight, &settings),
        "the alpha weight was dropped from the forward-only branch"
    );

    // `mode == 2`: the hybrid composite adds `deferred + fwd.rgb * fwd.a`.
    let deferred = constant_hdr(&device, &queue, W, H, [1.0; 3]);
    let gpu = meter(&deferred, 2);
    let hybrid = vec![[3.0_f32; 3]; samples];
    assert_eq!(
        gpu,
        build_histogram(&hybrid, &settings),
        "hybrid must meter deferred + fwd.rgb * fwd.a"
    );
    let straight_hybrid = vec![[5.0_f32; 3]; samples];
    assert_ne!(
        gpu,
        build_histogram(&straight_hybrid, &settings),
        "the alpha weight was dropped from the hybrid branch"
    );
}

/// One GPU step of `1/60 s` in each direction must land exactly where the
/// CPU reference lands: `brighten_speed` while the exposure rises,
/// `darken_speed` while it falls (the two rates must not be swapped).
#[test]
fn gpu_one_step_matches_the_cpu_speed_in_both_directions() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings::default();
    // Uniform scenes (a single bin, so the reduction is exact) off-key in
    // opposite directions; a fresh pass starts at `1.0` (EV 0).
    for (luminance, rises) in [(0.05_f32, true), (0.6_f32, false)] {
        let texels = vec![[luminance; 3]; (W * H) as usize];
        let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);
        let gpu = advance(
            &device,
            &queue,
            &AutoExposure::new(&device),
            Meter::deferred(&view, &settings, W, H, 1.0 / 60.0),
        );
        let avg = average_log_luminance(&build_histogram(&texels, &settings), &settings)
            .expect("uniform in-range scene");
        let target = desired_exposure_ev(avg, &settings);
        let expected = exp2(adapt_exposure_ev(0.0, target, 1.0 / 60.0, &settings));
        let relative = (gpu - expected).abs() / expected;
        assert!(
            relative < 0.005,
            "luminance {luminance}: gpu {gpu} vs cpu {expected}"
        );
        let direction = if rises { "rise" } else { "fall" };
        assert_eq!(
            gpu > 1.0,
            rises,
            "luminance {luminance}: the exposure must {direction} ({gpu})"
        );
    }
}

/// A manual exposure above the EV window must be adapted from the window's
/// top edge, not from the raw value: the WGSL `prev_ev` clamp and the CPU
/// mirror in `adapt_exposure_ev` are both load-bearing.
#[test]
fn gpu_one_step_from_a_manual_exposure_outside_the_window_matches_the_cpu() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings {
        min_ev: -1.0,
        max_ev: 1.0,
        ..AutoExposureSettings::default()
    };
    // Manual 100.0 is EV ≈ 6.64, far above the window; the scene below wants
    // an in-window target, so the step is dictated by the clamped previous
    // EV (exactly one stop of headroom above the target).
    let luminance = 0.25_f32;
    let texels = vec![[luminance; 3]; (W * H) as usize];
    let avg = average_log_luminance(&build_histogram(&texels, &settings), &settings)
        .expect("uniform in-range scene");
    let target = settings.target_ev() - avg;
    assert!(
        target > settings.min_ev && target < settings.max_ev,
        "the target {target} must sit inside the window"
    );
    let pass = AutoExposure::new(&device);
    pass.write_manual(&queue, 100.0);
    let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);
    let gpu = advance(
        &device,
        &queue,
        &pass,
        Meter::deferred(&view, &settings, W, H, 1.0 / 60.0),
    );
    // The CPU reference receives the raw previous EV (log2 of the manual
    // multiplier) and clamps it internally, like the shader does.
    let expected = exp2(adapt_exposure_ev(
        100.0_f32.log2(),
        target,
        1.0 / 60.0,
        &settings,
    ));
    let relative = (gpu - expected).abs() / expected;
    assert!(
        relative < 0.005,
        "gpu {gpu} vs cpu {expected} (clamped prev EV)"
    );
    // The step moved down from EV 1.0 (exposure 2.0), not from the raw
    // manual EV — an unclamped prev would still sit near 98.
    assert!(gpu < 2.0, "must leave the window's top edge, got {gpu}");
}

/// A target far below the EV window is clamped before the step is taken:
/// with the previous EV at the window's top edge and the clamped target one
/// stop down, the linear regime moves `|delta| / transition_distance` of a
/// `speed * dt` — the raw (unclamped) target would saturate at the full
/// `speed * dt` and land further.
#[test]
fn gpu_step_along_a_clamped_target_matches_the_cpu() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings {
        min_ev: -1.0,
        max_ev: 0.0,
        ..AutoExposureSettings::default()
    };
    // Bright scene: target ≈ log2(0.18) − log2(100) ≈ −9.1, deep below the
    // window; the clamped target is −1.0, exactly one stop from the EV-0
    // start (a fresh pass sits at exposure 1.0).
    let luminance = 100.0_f32;
    let texels = vec![[luminance; 3]; (W * H) as usize];
    let avg = average_log_luminance(&build_histogram(&texels, &settings), &settings)
        .expect("uniform in-range scene");
    let target_raw = settings.target_ev() - avg;
    assert!(
        target_raw < settings.min_ev,
        "the raw target {target_raw} must sit below the window"
    );
    let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);
    let gpu = advance(
        &device,
        &queue,
        &AutoExposure::new(&device),
        Meter::deferred(&view, &settings, W, H, 1.0 / 60.0),
    );
    let expected = exp2(adapt_exposure_ev(0.0, target_raw, 1.0 / 60.0, &settings));
    // The clamped delta is exactly 1 EV, so the step is (1/60) / 1.5 EV —
    // the deterministic value the GPU must reproduce.
    let reference = exp2(-1.0 / 60.0 / settings.exponential_transition_distance);
    assert!(
        (expected - reference).abs() / reference < 1e-4,
        "cpu {expected} vs analytic {reference}"
    );
    let relative = (gpu - expected).abs() / expected;
    assert!(
        relative < 1e-4,
        "gpu {gpu} vs cpu {expected} (clamped target)"
    );
}

/// One stop of EV compensation doubles the exposure the target maps onto
/// (`target_ev = log2(key) + ev_compensation`), the other direction halves it.
#[test]
fn ev_compensation_shifts_the_target_ev_by_a_stop() {
    let settings = AutoExposureSettings::default();
    let up = AutoExposureSettings {
        ev_compensation: 1.0,
        ..settings
    };
    let down = AutoExposureSettings {
        ev_compensation: -1.0,
        ..settings
    };
    for avg in [-6.0_f32, -1.0, 0.0, 3.0] {
        let base = desired_exposure_ev(avg, &settings);
        assert!(
            (desired_exposure_ev(avg, &up) - (base + 1.0)).abs() < 1e-6,
            "+1 EV at avg {avg}"
        );
        assert!(
            (desired_exposure_ev(avg, &down) - (base - 1.0)).abs() < 1e-6,
            "-1 EV at avg {avg}"
        );
        let base_exposure = desired_exposure(avg, &settings);
        assert!(
            (desired_exposure(avg, &up) / base_exposure - 2.0).abs() < 1e-3,
            "+1 EV must double the exposure"
        );
        assert!(
            (desired_exposure(avg, &down) / base_exposure - 0.5).abs() < 1e-3,
            "-1 EV must halve the exposure"
        );
    }
}

/// The GPU target carries the same bias: a compensated metering lands at
/// twice the exposure of the unbiased one.
#[test]
fn gpu_ev_compensation_doubles_the_exposure() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let unbiased = AutoExposureSettings::default();
    let settings = AutoExposureSettings {
        ev_compensation: 1.0,
        ..unbiased
    };
    let luminance = 0.5_f32;
    let texels = vec![[luminance; 3]; (W * H) as usize];
    let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);
    // A huge `dt` saturates the step, so one update lands on the target.
    let gpu = advance(
        &device,
        &queue,
        &AutoExposure::new(&device),
        Meter::deferred(&view, &settings, W, H, 100.0),
    );
    let expected = cpu_target(&texels, &settings);
    let relative = (gpu - expected).abs() / expected;
    assert!(
        relative < 0.02,
        "gpu exposure {gpu} vs cpu reference {expected}"
    );
    let plain = cpu_target(&texels, &unbiased);
    assert!(
        (gpu / plain - 2.0).abs() < 0.05,
        "+1 EV must double the exposure: {gpu} vs {plain}"
    );
}

/// The EV window clamps the adapted exposure on the GPU: scenes that want an
/// EV far outside `[-2, 2]` land exactly on `2^2` / `2^-2` after one
/// saturated step (both scenes sit inside the histogram window, so neither
/// clamps as "below range").
#[test]
fn gpu_ev_window_clamps_the_adapted_exposure() {
    let Some((device, queue)) = try_device() else {
        return;
    };
    const W: u32 = 16;
    const H: u32 = 16;
    let settings = AutoExposureSettings {
        min_ev: -2.0,
        max_ev: 2.0,
        ..AutoExposureSettings::default()
    };
    for (luminance, clamped) in [(0.01_f32, 4.0_f32), (100.0_f32, 0.25)] {
        let texels = vec![[luminance; 3]; (W * H) as usize];
        let avg = average_log_luminance(&build_histogram(&texels, &settings), &settings)
            .expect("both scenes sit inside the histogram window");
        let unclamped = settings.target_ev() - avg;
        assert!(
            unclamped.abs() > 3.0,
            "luminance {luminance} must want an EV outside the window: {unclamped}"
        );

        let view = constant_hdr(&device, &queue, W, H, [luminance; 3]);
        let gpu = advance(
            &device,
            &queue,
            &AutoExposure::new(&device),
            Meter::deferred(&view, &settings, W, H, 100.0),
        );
        let relative = (gpu - clamped).abs() / clamped;
        assert!(
            relative < 1e-4,
            "luminance {luminance}: exposure {gpu} vs clamp {clamped}"
        );
    }
}
