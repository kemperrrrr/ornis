//! Warmup, the recorded series, and the median/IQR lines.

use super::super::scene::Scene;
use super::super::scene::kind_name;
use super::super::scene::visible_ids;
use super::super::stats::Summary;
use super::super::stats::summarize;
use super::dispatch::Sample;

/// Unmeasured dispatches before the recorded series.
pub(super) const WARMUP: u32 = 5;
/// Recorded dispatches per shader per scene.
pub(super) const REPEATS: u32 = 30;
/// Nanoseconds in one microsecond. The table prints microseconds.
const US_IN_NS: f64 = 1_000.0;

pub(super) fn measure<F>(dispatch: F) -> Vec<Sample>
where
    F: Fn() -> Sample,
{
    warm(&dispatch);
    (0..REPEATS).map(|_| dispatch()).collect()
}

fn warm<F>(dispatch: &F)
where
    F: Fn() -> Sample,
{
    (0..WARMUP).for_each(|_| {
        let _ = dispatch();
    });
}

pub(super) fn scene_report(
    scene: &Scene,
    prod: &[Sample],
    wasm: &[Sample],
    twin: &[Sample],
) -> String {
    format!(
        "{kind}\tn={n}\tvisible={visible}\n  production\t{prod}\n  wasm2gpu\t{wasm}\n  hand_twin\t{twin}\n  gpu wasm/prod {ratio_wp}  gpu wasm/twin {ratio_wt}",
        kind = kind_name(scene.kind),
        n = scene.spheres.len(),
        visible = visible_ids(scene).len(),
        prod = clock_line(prod, scene.spheres.len()),
        wasm = clock_line(wasm, scene.spheres.len()),
        twin = clock_line(twin, scene.spheres.len()),
        ratio_wp = clock_ratio(wasm, prod),
        ratio_wt = clock_ratio(wasm, twin),
    )
}

fn clock_line(samples: &[Sample], count: usize) -> String {
    let wall = summarize(&walls(samples));
    format!(
        "gpu {}  wall {}  gpu_ns/obj {}",
        gpu_line(samples),
        format_summary_us(&wall),
        per_object(samples, count),
    )
}

fn gpu_line(samples: &[Sample]) -> String {
    match gpus(samples) {
        Some(values) => format_summary_us(&summarize(&values)),
        None => "unmeasured".to_string(),
    }
}

fn per_object(samples: &[Sample], count: usize) -> String {
    match gpus(samples) {
        Some(values) if count > 0 => format!("{:.3}", summarize(&values).median / count as f64),
        _ => "n/a".to_string(),
    }
}

fn clock_ratio(left: &[Sample], right: &[Sample]) -> String {
    match (gpus(left), gpus(right)) {
        (Some(lhs), Some(rhs)) => ratio_text(summarize(&lhs).median, summarize(&rhs).median),
        _ => "n/a".to_string(),
    }
}

fn ratio_text(left: f64, right: f64) -> String {
    if right == 0.0 {
        "n/a".to_string()
    } else {
        format!("{:.3}", left / right)
    }
}

fn format_summary_us(summary: &Summary) -> String {
    format!(
        "{:.3} [{:.3}–{:.3}] iqr {:.3}",
        ns_to_us(summary.median),
        ns_to_us(summary.q1),
        ns_to_us(summary.q3),
        ns_to_us(summary.iqr),
    )
}

fn ns_to_us(ns: f64) -> f64 {
    ns / US_IN_NS
}

fn walls(samples: &[Sample]) -> Vec<f64> {
    samples.iter().map(|sample| sample.wall_ns).collect()
}

fn gpus(samples: &[Sample]) -> Option<Vec<f64>> {
    let mut out = Vec::new();
    for sample in samples {
        out.push(sample.gpu_ns?);
    }
    Some(out)
}
