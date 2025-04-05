use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pyxirr::{irr, mirr};

#[path = "../tests/common/mod.rs"]
mod common;

static B_1: &[i32] = &[-100, 39, 59, 55, 20];
static B_2: &[f64] = &[
    -217500.0,
    -217500.0,
    108466.80462450592,
    101129.96439328062,
    93793.12416205535,
    86456.28393083003,
    79119.44369960476,
    71782.60346837944,
    64445.76323715414,
    57108.92300592884,
    49772.08277470355,
    42435.24254347826,
    35098.40231225296,
    27761.56208102766,
    20424.721849802358,
    13087.88161857707,
    5751.041387351768,
    -1585.7988438735192,
    -8922.639075098821,
    -16259.479306324123,
    -23596.31953754941,
    -30933.159768774713,
    -38270.0,
    -45606.8402312253,
    -52943.680462450604,
    -60280.520693675906,
    -67617.36092490121,
];
static B_3: &[f64] = &[10.0, 1.0, 2.0, -3.0, 4.0];
static B_4: &[i32] = &[-1000, 100, 250, 500, 500];

fn bench_irr(c: &mut Criterion) {
    let mut group = c.benchmark_group("IRR Benchmarks");

    // Benchmark 1: Basic IRR
    let payments: Vec<f64> = B_1.iter().map(|&x| x as f64).collect();
    group.bench_function("irr_simple", |b| b.iter(|| black_box(irr(&payments, None).unwrap())));

    // Benchmark 2: Complex IRR case
    group.bench_function("irr_complex", |b| b.iter(|| black_box(irr(B_2, None).unwrap())));

    // Benchmark 3: Special case
    group.bench_function("irr_special_case", |b| b.iter(|| black_box(irr(B_3, None).unwrap())));

    // Benchmark 4: With starting guess
    group.bench_function("irr_with_guess", |b| b.iter(|| black_box(irr(B_2, Some(0.12)).unwrap())));

    // Benchmark 5: High precision
    group.bench_function("irr_high_precision", |b| {
        b.iter(|| black_box(irr(B_2, Some(1e-10)).unwrap()))
    });

    group.finish();
}

fn bench_mirr(c: &mut Criterion) {
    let values: Vec<f64> = B_4.iter().map(|&x| x as f64).collect();
    c.bench_function("mirr_basic", |b| b.iter(|| black_box(mirr(&values, 0.1, 0.1).unwrap())));
}

criterion_group!(benches, bench_irr, bench_mirr);
criterion_main!(benches);
