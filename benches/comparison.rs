use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pyxirr::{irr, npv, xirr, xnpv};

#[path = "../tests/common/mod.rs"]
mod common;

fn bench_xirr_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("BXIRR by Dataset Size");

    let files = [
        ("xirr_50", "tests/samples/rw-50.csv"),
        ("xirr_100", "tests/samples/rw-100.csv"),
        ("xirr_500", "tests/samples/rw-500.csv"),
        ("xirr_1000", "tests/samples/rw-1000.csv"),
    ];

    for (name, file) in files.iter() {
        let payments = common::load_payments_from_csv(file).unwrap();
        let (dates, amounts) = common::split_payments(&payments);

        group.bench_with_input(
            BenchmarkId::new("xirr", name),
            &(dates, amounts),
            |b, (dates, amounts)| {
                b.iter(|| {
                    black_box(xirr(black_box(dates), black_box(amounts), None, None).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_irr_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("BIRR by Dataset Size");

    let files = [
        ("irr_50", "tests/samples/rw-50.csv"),
        ("irr_100", "tests/samples/rw-100.csv"),
        ("irr_500", "tests/samples/rw-500.csv"),
        ("irr_1000", "tests/samples/rw-1000.csv"),
    ];

    for (name, file) in files.iter() {
        let payments = common::load_payments_from_csv(file).unwrap();
        let (_, amounts) = common::split_payments(&payments);

        group.bench_with_input(BenchmarkId::new("irr", name), &amounts, |b, amounts| {
            b.iter(|| black_box(irr(black_box(amounts), None).unwrap()))
        });
    }

    group.finish();
}

fn bench_xnpv_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("XNPV by Dataset Size");
    let rate = 0.1;

    let files = [
        ("xnpv_50", "tests/samples/rw-50.csv"),
        ("xnpv_100", "tests/samples/rw-100.csv"),
        ("xnpv_500", "tests/samples/rw-500.csv"),
        ("xnpv_1000", "tests/samples/rw-1000.csv"),
    ];

    for (name, file) in files.iter() {
        let payments = common::load_payments_from_csv(file).unwrap();
        let (dates, amounts) = common::split_payments(&payments);

        group.bench_with_input(
            BenchmarkId::new("xnpv", name),
            &(dates, amounts),
            |b, (dates, amounts)| {
                b.iter(|| {
                    black_box(
                        xnpv(black_box(rate), black_box(dates), black_box(amounts), None).unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_special_cases(c: &mut Criterion) {
    let mut group = c.benchmark_group("Special Cases");

    // High volatility case
    {
        let payments = common::load_payments_from_csv("tests/samples/30-3.csv").unwrap();
        let (dates, amounts) = common::split_payments(&payments);

        group.bench_function("high_volatility", |b| {
            b.iter(|| black_box(xirr(black_box(&dates), black_box(&amounts), None, None).unwrap()))
        });
    }

    // Simple NPV case
    {
        let values = vec![-40_000.0, 5_000.0, 8_000.0, 12_000.0, 30_000.0];
        group.bench_function("npv_simple", |b| {
            b.iter(|| black_box(npv(black_box(0.08), black_box(&values), black_box(Some(true)))))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_xirr_by_size,
    bench_irr_by_size,
    bench_xnpv_by_size,
    bench_special_cases
);
criterion_main!(benches);
