//! Mirrors the consumer workload that motivated this work: an IRR series, where
//! `irr` is called once per prefix of a cash flow. Consecutive prefixes have very
//! similar rates, so the previous result is a near-perfect guess for the next one.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use pyxirr::irr;

#[path = "../tests/common/mod.rs"]
mod common;

/// Every prefix solved from scratch — what the consumer does today.
fn prefix_irr_cold(amounts: &[f64]) -> f64 {
    let mut last = f64::NAN;
    for k in 2..=amounts.len() {
        if let Ok(rate) = irr(&amounts[..k], None) {
            last = rate;
        }
    }
    last
}

/// Every prefix seeded with the previous prefix's rate.
fn prefix_irr_warm(amounts: &[f64]) -> f64 {
    let mut guess = None;
    let mut last = f64::NAN;
    for k in 2..=amounts.len() {
        if let Ok(rate) = irr(&amounts[..k], guess) {
            if rate.is_finite() {
                guess = Some(rate);
            }
            last = rate;
        }
    }
    last
}

fn bench_prefix_irr(c: &mut Criterion) {
    let mut group = c.benchmark_group("Prefix IRR Series");

    let files =
        [("rw_100", "tests/samples/rw-100.csv"), ("orbit_001", "tests/samples/orbit-001.csv")];

    for (name, file) in files.iter() {
        let payments = common::load_payments_from_csv(file).unwrap();
        let (_, amounts) = common::split_payments(&payments);

        group.bench_with_input(BenchmarkId::new("cold", name), &amounts, |b, amounts| {
            b.iter(|| black_box(prefix_irr_cold(black_box(amounts))))
        });

        group.bench_with_input(BenchmarkId::new("warm", name), &amounts, |b, amounts| {
            b.iter(|| black_box(prefix_irr_warm(black_box(amounts))))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_prefix_irr);
criterion_main!(benches);
