use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pyxirr::xirr;
use time::macros::date;

#[path = "../tests/common/mod.rs"]
mod common;

fn bench_input_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("XIRR Input Methods");

    // Benchmark 1: From CSV Direct
    {
        let input = "tests/samples/random_100.csv";
        let payments = common::load_payments_from_csv(input).unwrap();
        let (dates, amounts) = common::split_payments(&payments);

        group.bench_function("from_csv_direct", |b| {
            b.iter(|| {
                let result = xirr(black_box(&dates), black_box(&amounts), None, None).unwrap();
                black_box(result)
            })
        });
    }

    // Benchmark 2: From Pre-loaded Vectors
    {
        let input = "tests/samples/random_100.csv";
        let payments = common::load_payments_from_csv(input).unwrap();
        let (dates, amounts) = common::split_payments(&payments);

        group.bench_function("from_vectors", |b| {
            b.iter(|| black_box(xirr(black_box(&dates), black_box(&amounts), None, None).unwrap()))
        });
    }

    // Benchmark 3: Small Dataset
    {
        let dates = vec![date!(2020 - 01 - 01), date!(2020 - 06 - 01), date!(2021 - 01 - 01)]
            .into_iter()
            .map(|x| x.into())
            .collect::<Vec<_>>();
        let amounts = vec![-1000.0, 500.0, 600.0];

        group.bench_function("small_dataset", |b| {
            b.iter(|| black_box(xirr(black_box(&dates), black_box(&amounts), None, None).unwrap()))
        });
    }

    // Benchmark 4: Large Dataset
    {
        let input = "tests/samples/random_1000.csv";
        let payments = common::load_payments_from_csv(input).unwrap();
        let (dates, amounts) = common::split_payments(&payments);

        group.bench_function("large_dataset", |b| {
            b.iter(|| black_box(xirr(black_box(&dates), black_box(&amounts), None, None).unwrap()))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_input_methods);
criterion_main!(benches);
