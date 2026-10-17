//! Reference and shader-contract regressions for AA.
use super::*;

#[test]
fn geometry_samples_and_shader_modes() {
    for (mode, samples, shader) in [
        (AntiAliasing::Off, 1, 0),
        (AntiAliasing::Msaa, 4, 1),
        (AntiAliasing::Fxaa, 1, 2),
        (AntiAliasing::Smaa, 1, 3),
        (AntiAliasing::Taa, 1, 4),
    ] {
        assert_eq!(mode.sample_count(), samples);
        assert_eq!(mode.shader_mode(), shader);
    }
}

#[test]
fn jitter_is_bounded_periodic_and_not_the_zero_index_corner() {
    assert!((temporal_jitter(0) - Vec2::new(0.0, -1.0 / 6.0)).length() < 1e-6);
    for frame in 0..64 {
        let jitter = temporal_jitter(frame);
        assert!(jitter.abs().max_element() <= 0.5);
        assert_eq!(jitter, temporal_jitter(frame + 8));
    }
    let projection = glam::camera::rh::proj::directx::perspective(1.0, 1.5, 0.1, 100.0);
    let point = Vec4::new(1.0, 2.0, -10.0, 1.0);
    let original = projection * point;
    let shifted = jitter_projection(projection, Vec2::new(0.25, -0.25), [100, 100]) * point;
    assert!((shifted.x / shifted.w - original.x / original.w - 0.005).abs() < 1e-6);
    assert!((shifted.y / shifted.w - original.y / original.w - 0.005).abs() < 1e-6);
    assert_eq!(shifted.z, original.z);
    assert_eq!(shifted.w, original.w);
    assert_eq!(jitter_projection(projection, Vec2::ONE, [0, 1]), projection);
}

#[test]
fn motion_includes_skin_deformation_and_uv_y_flip() {
    let current = Vec4::new(0.4, 0.2, 0.5, 1.0);
    let previous = Vec4::new(0.0, -0.2, 0.5, 1.0);
    let motion = motion_vector(current, previous).unwrap();
    assert!((motion - Vec2::new(-0.2, 0.2)).length() < 1e-6);
    assert_eq!(motion_vector(current, current), Some(Vec2::ZERO));
    assert_eq!(motion_vector(current, Vec4::ZERO), None);
    assert_eq!(motion_vector(Vec4::splat(f32::NAN), previous), None);
}

#[test]
fn static_history_converges_after_64_frames() {
    let target = Vec3::new(0.2, 0.4, 0.8);
    let mut history = Vec3::ZERO;
    for _ in 0..64 {
        history = temporal_blend(target, history, &[Vec3::ZERO, Vec3::ONE], 0.9, true);
    }
    assert!((history - target).abs().max_element() < 0.001);
}

#[test]
fn moving_foreground_does_not_leave_a_bright_trail() {
    let background = Vec3::splat(0.02);
    let foreground = Vec3::splat(16.0);
    // Even an erroneously accepted reprojection cannot escape a flat neighbourhood.
    assert_eq!(
        temporal_blend(background, foreground, &[background; 9], 0.9, true),
        background
    );
    // Correct disocclusion rejection bypasses accumulation entirely.
    assert_eq!(
        temporal_blend(background, foreground, &[foreground; 9], 0.9, false),
        background
    );
    assert_eq!(
        temporal_blend(background, Vec3::NAN, &[], 0.9, true),
        background
    );
}

#[test]
fn shader_validates_with_webgpu_capabilities() {
    let source = include_str!("resolve.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 4);
}
