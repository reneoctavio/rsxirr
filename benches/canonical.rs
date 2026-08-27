//! `canonical_irr` against `irr`, on the workload that motivates it: an IRR
//! series, where the rate is solved once per prefix of a cash flow.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use pyxirr::{canonical_irr, irr};

#[path = "../tests/common/mod.rs"]
mod common;

/// A conventional flow: one outlay, then returns. One sign change, one root.
fn conventional(periods: usize) -> Vec<f64> {
    let mut flow = vec![-1_000_000.0];
    flow.extend((0..periods).map(|period| 12_000.0 + period as f64 * 40.0));
    flow
}

/// A flow that changes sign repeatedly, which is where the two disagree and
/// where the sweep has the most roots to pass over.
fn alternating(periods: usize) -> Vec<f64> {
    (0..periods)
        .map(|period| {
            let magnitude = 100_000.0 + (period as f64) * 1_000.0;
            if period == 0 || period % 4 == 0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect()
}

fn bench_single_flow(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canonical IRR");

    for periods in [12usize, 60, 240] {
        for (shape, flow) in
            [("conventional", conventional(periods)), ("alternating", alternating(periods))]
        {
            group.bench_with_input(
                BenchmarkId::new(format!("canonical_{shape}"), periods),
                &flow,
                |b, flow| b.iter(|| black_box(canonical_irr(black_box(flow)))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("cascade_{shape}"), periods),
                &flow,
                |b, flow| b.iter(|| black_box(irr(black_box(flow), None))),
            );
        }
    }

    group.finish();
}

fn bench_prefix_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("Canonical IRR Series");

    let files =
        [("rw_100", "tests/samples/rw-100.csv"), ("orbit_001", "tests/samples/orbit-001.csv")];

    for (name, file) in files {
        let payments = common::load_payments_from_csv(file).unwrap();
        let (_, amounts) = common::split_payments(&payments);

        group.bench_with_input(BenchmarkId::new("canonical", name), &amounts, |b, amounts| {
            b.iter(|| {
                for length in 2..=amounts.len() {
                    black_box(canonical_irr(black_box(&amounts[..length]))).ok();
                }
            })
        });
        group.bench_with_input(BenchmarkId::new("cascade", name), &amounts, |b, amounts| {
            b.iter(|| {
                for length in 2..=amounts.len() {
                    black_box(irr(black_box(&amounts[..length]), None)).ok();
                }
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_single_flow, bench_prefix_series);
criterion_main!(benches);
