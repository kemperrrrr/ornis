//! Shader-only parity of the llvm2gpu AO kernels against master's WGSL.
//!
//! The scene is a 64×64 depth image with an interior step and a far last
//! row. GTAO's radius puts pixel (0, 0) step 1 on UV x = 1, where master
//! clamps the texel to `width - 1` and an unclamped cast is out of bounds.
//! `ORNIS_REQUIRE_GPU=1` panics when no adapter exists.

#[allow(dead_code)]
mod common;

#[path = "ao_master_parity/pass.rs"]
mod pass;
#[path = "ao_master_parity/scene.rs"]
mod scene;

use ornis_render::AoMethod;
use ornis_render::shaders::ao_generated::{wgsl_source_blur, wgsl_source_raw_for_samples};
use scene::LevelDiff;

fn master(name: &str, multisampled: bool) -> String {
    let source = match name {
        "gtao" => include_str!("ao_master_parity/master_gtao.wgsl"),
        "ssao" => include_str!("ao_master_parity/master_ssao.wgsl"),
        "blur" => include_str!("ao_master_parity/master_blur.wgsl"),
        other => panic!("unknown master shader {other}"),
    };
    if multisampled {
        source
            .replace("texture_depth_2d", "texture_depth_multisampled_2d")
            .replace("texture_2d<f32>", "texture_multisampled_2d<f32>")
    } else {
        source.to_string()
    }
}

fn report(label: &str, diff: &LevelDiff) {
    eprintln!(
        "{label}: max {}/255  over5 {}/{}  mae {:.6}",
        diff.max_level, diff.over_5, diff.pixels, diff.mae
    );
}

fn assert_identical(label: &str, diff: &LevelDiff) {
    report(label, diff);
    assert_eq!(
        diff.max_level, 0,
        "{label}: max {}/255, {} px > 5 levels, mae {:.6}",
        diff.max_level, diff.over_5, diff.mae
    );
}

fn raw_pair(shot: &scene::Shot, method: AoMethod, multisampled: bool) -> LevelDiff {
    let Some((_adapter, device, queue)) = common::adapter_or_skip() else {
        return LevelDiff {
            max_level: 0,
            over_5: 0,
            mae: 0.0,
            pixels: 0,
        };
    };
    let name = if method.is_ssao() { "ssao" } else { "gtao" };
    let samples = if multisampled { 4 } else { 1 };
    let master_px = pass::draw_raw(
        &device,
        &queue,
        shot,
        &master(name, multisampled),
        multisampled,
    );
    let generated = wgsl_source_raw_for_samples(method, samples);
    let new_px = pass::draw_raw(&device, &queue, shot, &generated, multisampled);
    scene::level_diff(&master_px, &new_px)
}

#[test]
fn gtao_border_matches_master_wgsl() {
    let diff = raw_pair(&scene::border_shot(), AoMethod::Gtao, false);
    if diff.pixels == 0 {
        return;
    }
    assert_identical("gtao 1x border", &diff);
}

#[test]
fn gtao_border_4x_matches_master_wgsl() {
    let diff = raw_pair(&scene::border_shot(), AoMethod::Gtao, true);
    if diff.pixels == 0 {
        return;
    }
    assert_identical("gtao 4x border", &diff);
}

#[test]
fn ssao_border_matches_master_wgsl() {
    let diff = raw_pair(&scene::border_shot(), AoMethod::Ssao, false);
    if diff.pixels == 0 {
        return;
    }
    assert_identical("ssao 1x border", &diff);
}

#[test]
fn ssao_border_4x_matches_master_wgsl() {
    let diff = raw_pair(&scene::border_shot(), AoMethod::Ssao, true);
    if diff.pixels == 0 {
        return;
    }
    assert_identical("ssao 4x border", &diff);
}

#[test]
fn gtao_local_matches_master_wgsl() {
    let diff = raw_pair(&scene::local_shot(), AoMethod::Gtao, false);
    if diff.pixels == 0 {
        return;
    }
    assert_identical("gtao 1x local", &diff);
}

#[test]
fn ssao_local_matches_master_wgsl() {
    let diff = raw_pair(&scene::local_shot(), AoMethod::Ssao, false);
    if diff.pixels == 0 {
        return;
    }
    assert_identical("ssao 1x local", &diff);
}

#[test]
fn blur_border_matches_master_wgsl() {
    let Some((_adapter, device, queue)) = common::adapter_or_skip() else {
        return;
    };
    let shot = scene::border_shot();
    let raw = scene::ao_raw_rgba();
    let master_px = pass::draw_blur(&device, &queue, &shot, &master("blur", false), &raw);
    let new_px = pass::draw_blur(&device, &queue, &shot, &wgsl_source_blur(), &raw);
    assert_identical("blur", &scene::level_diff(&master_px, &new_px));
}
