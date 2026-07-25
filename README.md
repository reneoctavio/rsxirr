[![rust-lang.org](https://img.shields.io/badge/Made%20with-Rust-red)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-Unlicense-blue)](LICENSE)

# pyxirr

Rust-powered collection of financial functions: IRR, XIRR, NPV, XNPV, FV, PV, MIRR and the
rest of the numpy-financial set, plus day count conventions.

This is a fork of [Anexen/pyxirr](https://github.com/Anexen/pyxirr) with the PyO3 bindings
removed. Upstream ships a Python extension module; this fork is a plain Rust crate and
keeps the crate name `pyxirr` so existing git dependencies keep resolving. The financial
functions and their semantics come from upstream; the SIMD kernels and the IRR solver have
been reworked here.

Features:

- correct
- supports different day count conventions (e.g. ACT/360, 30E/360, etc.)
- one dependency (`time`)
- SIMD-accelerated NPV on x86-64 (AVX2/AVX) and aarch64 (NEON), with a scalar fallback

# Installation

```toml
[dependencies]
pyxirr = { git = "https://github.com/reneoctavio/rsxirr", rev = "<commit>" }
```

# Usage

### Periodic cash flows

```rust
use pyxirr::{irr, npv};

let values = [-100.0, 39.0, 59.0, 55.0, 20.0];

let rate = irr(&values, None).unwrap();
assert!((rate - 0.28094842115).abs() < 1e-9);

let value = npv(0.08, &values, Some(true));
```

`irr` takes an optional guess. Supplying one matters: it is used to seed a Newton
iteration before the bracket search, which is substantially faster when you already know
roughly where the root is — computing an IRR series over successive prefixes of a cash
flow, for instance, where each period's rate is close to the previous one.

```rust
use pyxirr::irr;

let mut guess = None;
for k in 2..=values.len() {
    if let Ok(rate) = irr(&values[..k], guess) {
        guess = Some(rate);
    }
}
```

### Dated cash flows

```rust
use pyxirr::{xirr, xnpv, DateLike, DayCount};

let dates: Vec<DateLike> = ["2020-01-01", "2021-01-01", "2022-01-01"]
    .iter()
    .map(|s| s.parse().unwrap())
    .collect();
let amounts = [-1000.0, 750.0, 500.0];

let rate = xirr(&dates, &amounts, None, None).unwrap();
let value = xnpv(0.1, &dates, &amounts, None).unwrap();
```

`DateLike` parses `%Y-%m-%d` and `%m/%d/%Y`, and converts from `time::Date`.

### Day count conventions

```rust
use pyxirr::{xirr, DayCount};

// ACT_365F is the default
xirr(&dates, &amounts, None, Some(DayCount::ACT_360)).unwrap();

// or parse from a string
let day_count: DayCount = "30E/360".parse().unwrap();
xirr(&dates, &amounts, None, Some(day_count)).unwrap();
```

### Other functions

```rust
use pyxirr::{fv, mirr, nfv, npv, pmt, pv, rate};

fv(0.05 / 12.0, 10.0 * 12.0, -100.0, -100.0, false);
pv(0.05 / 12.0, 10.0 * 12.0, -100.0, 15692.93, false);
pmt(0.05, 10.0, 100_000.0, 0.0, false);
rate(10.0, -100.0, 1000.0, 0.0, false, None);
mirr(&[-1000.0, 100.0, 250.0, 500.0, 500.0], 0.1, 0.1).unwrap();
nfv(0.03, 6.0, &[1050.0, 1350.0, 1350.0, 1450.0]);
```

Also available: `ipmt`, `ppmt`, `nper`, `cumipmt`, `cumprinc`, `xfv`, `xnfv`,
`year_fraction`, `days_between`, `zero_crossing_points`.

# Multiple IRR problem

The multiple IRR problem occurs when the signs of the cash flows change more than once, so
the project has non-conventional cash flows and may have several IRRs or none. `irr` looks
for a root by bracket search, falling back to Newton and then a grid search across
`[-0.99999999999999, 1e6]`. `zero_crossing_points` is useful for finding the intervals
where an NPV profile changes sign, so you can target each root with a guess.

# SIMD dispatch

NPV runs on hand-written kernels selected once per process from the CPU's features:
AVX2+FMA, then plain AVX, then a scalar auto-vectorized fallback on x86-64; NEON on
aarch64. Short slices always take the fallback, at a threshold measured per architecture.

The tier can be forced down with environment variables, which is what the benchmarks use
to compare kernels against the fallback. They are read once, at the first NPV call:

| Variable | Effect |
| --- | --- |
| `ENABLE_AVX2=0` | skip the AVX2+FMA tier |
| `ENABLE_AVX=0` | skip the plain-AVX tier |
| `ENABLE_NEON=0` | skip the NEON kernels |

# Development

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The kernels are dispatched at runtime, so a plain `cargo test` only exercises the best
tier your CPU supports. Cover the others by forcing the dispatch down:

```bash
ENABLE_AVX2=0 cargo test               # plain AVX
ENABLE_AVX2=0 ENABLE_AVX=0 cargo test  # auto-vectorized fallback
```

The NEON kernels are behind `cfg(target_arch = "aarch64")` and are not compiled by an x86
build at all. Check them with a cross target:

```bash
rustup target add aarch64-unknown-linux-gnu
cargo clippy --target aarch64-unknown-linux-gnu --all-targets -- -D warnings
```

### Benchmarks

```bash
cargo bench --bench npv_kernel    # NPV/IRR kernels isolated by slice length
cargo bench --bench prefix_irr    # IRR series, cold vs warm-started
cargo bench --bench grid_fallback # cost of the last-resort grid search
cargo bench --bench comparison    # XIRR/IRR/XNPV by dataset size
cargo bench --bench npf           # IRR and MIRR
cargo bench --bench input         # input handling
```

Criterion supports baselines, which is how the kernel changes were measured:

```bash
cargo bench --bench npv_kernel -- --save-baseline before
# ... make a change ...
cargo bench --bench npv_kernel -- --baseline before
```

# License

Unlicense, as upstream. See [LICENSE](LICENSE).
