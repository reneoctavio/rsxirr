//! Isolates the SIMD NPV kernels from the root-finding above them: `npv` at a fixed rate,
//! over lengths that straddle the SIMD threshold and the 8-lane block size.
//!
//! Also where the `*_MIN_LEN` thresholds are measured: run this with `ENABLE_NEON=0` (or
//! `ENABLE_AVX2=0 ENABLE_AVX=0`) to time the auto-vectorized fallback at each length, then
//! with the kernels on, and take the crossover.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pyxirr::{irr, npv};

fn cash_flow(len: usize) -> Vec<f64> {
    (0..len)
        .map(|i| {
            if i == 0 {
                -1.0e6
            } else {
                12_500.0 + (i % 11) as f64 * 37.5
            }
        })
        .collect()
}

/// Lengths straddling every threshold that matters: the SIMD cutoffs (`SIMD256_MIN_LEN`,
/// `NEON_MIN_LEN`), the 8-lane block size, and the remainders past a whole block.
const LENGTHS: [usize; 16] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 64, 100, 256, 1024];

fn bench_npv_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("NPV kernel");

    for len in LENGTHS {
        let values = cash_flow(len);
        group.bench_with_input(BenchmarkId::new("npv", len), &values, |b, values| {
            b.iter(|| black_box(npv(black_box(0.07), black_box(values), Some(true))))
        });
    }

    group.finish();
}

/// `irr` drives the NPV+derivative kernel through Newton, which the `npv`-only group above
/// never touches. Lengths below 4 take the analytical fast paths, so they start at 4.
fn bench_irr_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("IRR kernel");

    for len in LENGTHS.into_iter().filter(|&len| len >= 4) {
        let values = cash_flow(len);
        group.bench_with_input(BenchmarkId::new("irr", len), &values, |b, values| {
            b.iter(|| black_box(irr(black_box(values), None)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_npv_by_size, bench_irr_by_size);
criterion_main!(benches);
