//! #113 gate: autovectorized slice loop vs explicit SIMD over SoA lanes
//! (IDEAS §31, decision rule: close #113 as won't-do when autovec reaches at
//! least 80% of the explicit path).
//!
//! The kernel is the linear integration of `smart_system_linear`
//! (`v += G*DT; p += v*DT`) over the plain `f32` columns the SoA lanes reduce
//! to. Explicit variants use `core::arch` intrinsics in the same operation
//! order, so they are checked bit-for-bit against the scalar loop before any
//! timing. The AVX2 variant exists only when the build enables AVX2
//! (`-C target-cpu=x86-64-v3`).

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const DT: f32 = 1.0 / 60.0;
const GRAVITY: f32 = -9.81;
const ACCEL: f32 = GRAVITY * DT;
const ENTITY_COUNTS: [usize; 2] = [10_000, 100_000];
const SAMPLE_SIZE: usize = 20;
const MEASURE_SECS: u64 = 4;
const WARMUP_SECS: u64 = 1;

fn integrate_scalar(position: &mut [f32], velocity: &mut [f32]) {
    for (p, v) in position.iter_mut().zip(velocity.iter_mut()) {
        *v += ACCEL;
        *p += *v * DT;
    }
}

#[cfg(target_arch = "x86_64")]
fn integrate_sse(position: &mut [f32], velocity: &mut [f32]) {
    use std::arch::x86_64::{_mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps};
    let len = position.len();
    // SAFETY: SSE is part of the x86_64 baseline.
    let (accel, dt) = unsafe { (_mm_set1_ps(ACCEL), _mm_set1_ps(DT)) };
    let mut i = 0;
    while i + 4 <= len {
        // SAFETY: `i + 4 <= len` and both slices have `len` elements, so the
        // four-lane loads and stores stay in bounds; SSE is baseline on x86_64.
        unsafe {
            let p = _mm_loadu_ps(position.as_ptr().add(i));
            let v = _mm_add_ps(_mm_loadu_ps(velocity.as_ptr().add(i)), accel);
            let p = _mm_add_ps(p, _mm_mul_ps(v, dt));
            _mm_storeu_ps(velocity.as_mut_ptr().add(i), v);
            _mm_storeu_ps(position.as_mut_ptr().add(i), p);
        }
        i += 4;
    }
    integrate_scalar(&mut position[i..], &mut velocity[i..]);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
fn integrate_avx2(position: &mut [f32], velocity: &mut [f32]) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
    };
    let len = position.len();
    // SAFETY: AVX2 is enabled at compile time by the cfg gate above.
    let (accel, dt) = unsafe { (_mm256_set1_ps(ACCEL), _mm256_set1_ps(DT)) };
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: `i + 8 <= len` and both slices have `len` elements, so the
        // eight-lane loads and stores stay in bounds; AVX2 is enabled at
        // compile time by the cfg gate above.
        unsafe {
            let p = _mm256_loadu_ps(position.as_ptr().add(i));
            let v = _mm256_add_ps(_mm256_loadu_ps(velocity.as_ptr().add(i)), accel);
            let p = _mm256_add_ps(p, _mm256_mul_ps(v, dt));
            _mm256_storeu_ps(velocity.as_mut_ptr().add(i), v);
            _mm256_storeu_ps(position.as_mut_ptr().add(i), p);
        }
        i += 8;
    }
    integrate_scalar(&mut position[i..], &mut velocity[i..]);
}

fn columns(count: usize) -> (Vec<f32>, Vec<f32>) {
    let position = (0..count).map(|i| i as f32 * 0.25).collect();
    let velocity = (0..count).map(|i| (i % 7) as f32 - 3.0).collect();
    (position, velocity)
}

fn assert_bit_identical(name: &str, kernel: fn(&mut [f32], &mut [f32])) {
    let (mut p_ref, mut v_ref) = columns(1_003);
    let (mut p, mut v) = columns(1_003);
    for _ in 0..16 {
        integrate_scalar(&mut p_ref, &mut v_ref);
        kernel(&mut p, &mut v);
    }
    assert!(
        p.iter()
            .zip(&p_ref)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && v.iter()
                .zip(&v_ref)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{name} is not bit-identical to the scalar loop"
    );
}

fn bench_gate(c: &mut Criterion) {
    #[cfg(target_arch = "x86_64")]
    assert_bit_identical("sse", integrate_sse);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    assert_bit_identical("avx2", integrate_avx2);

    let mut group = c.benchmark_group("simd_gate_integrate");
    group
        .sample_size(SAMPLE_SIZE)
        .measurement_time(Duration::from_secs(MEASURE_SECS))
        .warm_up_time(Duration::from_secs(WARMUP_SECS));
    for count in ENTITY_COUNTS {
        group.throughput(Throughput::Elements(count as u64));
        let (mut p, mut v) = columns(count);
        group.bench_with_input(BenchmarkId::new("autovec_slices", count), &count, |b, _| {
            b.iter(|| integrate_scalar(black_box(&mut p), black_box(&mut v)))
        });
        #[cfg(target_arch = "x86_64")]
        {
            let (mut p, mut v) = columns(count);
            group.bench_with_input(BenchmarkId::new("explicit_sse", count), &count, |b, _| {
                b.iter(|| integrate_sse(black_box(&mut p), black_box(&mut v)))
            });
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let (mut p, mut v) = columns(count);
            group.bench_with_input(BenchmarkId::new("explicit_avx2", count), &count, |b, _| {
                b.iter(|| integrate_avx2(black_box(&mut p), black_box(&mut v)))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_gate);
criterion_main!(benches);
