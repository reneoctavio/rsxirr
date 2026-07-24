//! Isolates the SIMD NPV kernels from the root-finding above them: `npv` at a fixed rate,
//! over lengths that straddle the SIMD threshold and the 8-lane block size.
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pyxirr::npv;

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

fn bench_npv_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("NPV kernel");

    // 9 is the first length that takes the SIMD path; 11 and 100 leave a remainder
    for len in [4usize, 5, 6, 7, 8, 9, 11, 12, 16, 64, 100, 256, 1024] {
        let values = cash_flow(len);
        group.bench_with_input(BenchmarkId::new("npv", len), &values, |b, values| {
            b.iter(|| black_box(npv(black_box(0.07), black_box(values), Some(true))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_npv_by_size);
criterion_main!(benches);
