//! Cost of the last-resort `brentq_grid_search` in `irr`.
//!
//! Both inputs are constructed to fall through every earlier stage of the cascade — the
//! bracket searches find no sign change and Newton does not converge — so the grid search
//! runs to exhaustion. Both have a real IRR just below -99.9%, outside the `[-0.999, 100]`
//! bracket, which is precisely the region the current breakpoints cannot reach.
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use pyxirr::irr;

/// n=4, root at -0.99950. The tail term is negligible at rate -0.999 but dominates as the
/// rate approaches -1, which is what puts the root outside the earlier bracket.
fn short_cash_flow() -> Vec<f64> {
    vec![-4002.0, 0.001, -0.001, 1e-6]
}

/// n=20, root at -0.99955. Same shape, but npv reaches ~1e206 at the -0.99999999999999
/// endpoint, so brentq's interpolation degenerates and it bisects the whole way down.
fn long_cash_flow() -> Vec<f64> {
    let mut values = vec![-4002.0];
    values.extend(std::iter::repeat_n(0.0, 18));
    values.push(1e-60);
    values
}

fn bench_grid_fallback(c: &mut Criterion) {
    let mut group = c.benchmark_group("Grid fallback");

    for (name, values) in [("short_n4", short_cash_flow()), ("long_n20", long_cash_flow())] {
        // record what the cascade actually returns under the current breakpoints
        eprintln!("  irr({name}) = {:?}", irr(&values, None));

        group.bench_with_input(BenchmarkId::new("irr", name), &values, |b, values| {
            b.iter(|| black_box(irr(black_box(values), None)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_grid_fallback);
criterion_main!(benches);
