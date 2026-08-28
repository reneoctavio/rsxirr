//! Behaviour of [`canonical_irr`].

use pyxirr::{canonical_irr, irr, npv};

mod common;

/// The present value of `flow` at `rate` and its slope, as a polynomial in
/// `x = 1/(1+rate)`.
fn present_value(rate: f64, flow: &[f64]) -> (f64, f64) {
    let x = 1.0 / (1.0 + rate);
    let (mut value, mut slope) = (0.0, 0.0);

    for coefficient in flow.iter().rev() {
        slope = slope * x + value;
        value = value * x + coefficient;
    }

    (value, slope)
}

/// `rate` locates a root of `flow`.
///
/// Measured as the displacement `|P| / |P'|`, which is how far the rate sits
/// from the root, and not as a residue. On a long flow at a negative rate the
/// discounted terms outgrow the flow by tens of orders of magnitude, so a root
/// converged to the last bit `f64` carries still leaves a residue far above
/// any amount in the flow. A residue test rejects those.
fn assert_is_a_root(rate: f64, flow: &[f64]) {
    let x = 1.0 / (1.0 + rate);
    let (value, slope) = present_value(rate, flow);
    let displacement = value.abs() / slope.abs();

    assert!(displacement <= 1e-6 * x.abs(), "{rate} sits {displacement} from a root");
}

/// One sign change means one rate, so both functions must agree.
#[test]
fn agrees_with_the_cascade_on_conventional_flows() {
    let conventional: [&[f64]; 4] = [
        &[-100.0, 39.0, 59.0, 55.0, 20.0],
        &[-1_000.0, 100.0, 200.0, 300.0, 400.0, 500.0],
        &[-100.0, 0.0, 0.0, 74.0],
        &[-250_000.0, 30_000.0, 30_000.0, 30_000.0, 30_000.0, 200_000.0],
    ];

    for flow in conventional {
        let canonical = canonical_irr(flow).unwrap().expect("a conventional flow has a rate");
        let cascade = irr(flow, None).unwrap();

        assert!(
            (canonical - cascade).abs() < 1e-7,
            "canonical {canonical} against cascade {cascade}"
        );
        assert_is_a_root(canonical, flow);
    }
}

/// With several rates, the one the present value falls through.
#[test]
fn takes_the_rate_the_present_value_falls_through() {
    // Satisfied at 10% and at 20%; the value climbs through the first.
    let two_roots = [-100.0, 230.0, -132.0];

    let rate = canonical_irr(&two_roots).unwrap().expect("two roots is still a rate");

    assert!((rate - 0.2).abs() < 1e-7, "{rate}");
    assert_is_a_root(rate, &two_roots);
    // 10% is equally a root; it is the one the value climbs through.
    assert_is_a_root(0.1, &two_roots);
}

/// At the rate reported, a higher discount rate lowers the present value.
#[test]
fn the_rate_it_reports_falls_with_the_discount_rate() {
    let flows: [&[f64]; 3] = [
        &[-100.0, 230.0, -132.0],
        &[-100.0, 39.0, 59.0, 55.0, 20.0],
        &[-50.0, 200.0, -180.0, 40.0],
    ];

    for flow in flows {
        let Some(rate) = canonical_irr(flow).unwrap() else {
            continue;
        };

        let step = 1e-4;
        assert!(
            npv(rate + step, flow, Some(true)) < npv(rate - step, flow, Some(true)),
            "present value rises with the rate at {rate}"
        );
    }
}

/// A rate past the band is not reported.
#[test]
fn refuses_rates_past_the_ceiling() {
    // A token outlay returned a thousandfold breaks even in the hundreds of
    // percent.
    let token_outlay = [-1.0, 0.0, 0.0, 1_000.0];

    let cascade = irr(&token_outlay, Some(0.01)).unwrap();
    assert!(cascade > 1.0, "the cascade answers {cascade}");

    assert_eq!(canonical_irr(&token_outlay).unwrap(), None);
}

/// A flow that only breaks even below the band has no rate here.
#[test]
fn a_flow_that_only_loses_has_no_rate() {
    let never_recovers = [-100.0, -50.0, 10.0, 5.0];
    assert_eq!(canonical_irr(&never_recovers).unwrap(), None);
}

/// One amount of each sign is required, as in [`irr`].
#[test]
fn rejects_a_flow_of_one_sign() {
    assert!(canonical_irr(&[-100.0, -50.0, -25.0]).is_err());
    assert!(canonical_irr(&[100.0, 50.0, 25.0]).is_err());
}

/// Scaling a flow does not change its rate.
#[test]
fn is_invariant_under_scaling() {
    let flow = [-172_545.848_122_807, 787.735_232_518, 900.0, 1_100.0, 250_000.0];

    let base = canonical_irr(&flow).unwrap().expect("a rate");
    for factor in [1e-6, 1e-3, 1e3, 1e6] {
        let scaled: Vec<f64> = flow.iter().map(|value| value * factor).collect();
        let rate = canonical_irr(&scaled).unwrap().expect("a rate");
        assert!(
            (rate - base).abs() < 1e-7,
            "scaling by {factor} moved the rate from {base} to {rate}"
        );
    }
}

/// Over real monthly flows, every rate returned is a root inside the band.
#[test]
fn holds_over_real_project_flows() {
    let raw = include_str!("spreadsheet/production_flows.json");
    let flows: Vec<Vec<f64>> = raw
        .split("],")
        .map(|chunk| {
            chunk
                .trim_matches(|c: char| c == '[' || c == ']' || c.is_whitespace())
                .split(',')
                .filter_map(|value| value.trim().parse::<f64>().ok())
                .collect()
        })
        .filter(|flow: &Vec<f64>| flow.len() > 4)
        .collect();

    let mut answered = 0;
    for flow in &flows {
        for length in 5..=flow.len() {
            let prefix = &flow[..length];
            let Ok(Some(rate)) = canonical_irr(prefix) else {
                continue;
            };
            answered += 1;
            assert!(rate <= 1.0, "{rate} is past the ceiling");
            assert_is_a_root(rate, prefix);
        }
    }
    assert!(answered > 1_000, "only {answered} prefixes answered");
}

/// Deterministic flows, so a failure reproduces.
struct Flows(u64);

impl Flows {
    fn next_unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A flow of 40 to 240 monthly amounts that changes sign often. The length
    /// matters: the residue only outgrows the flow once the discounting has
    /// enough periods to compound over.
    fn next_flow(&mut self) -> Vec<f64> {
        let months = 40 + (self.next_unit() * 200.0) as usize;

        (0..months)
            .map(|month| {
                let amount = 1e4 + self.next_unit() * 1e5;
                if month == 0 || self.next_unit() < 0.35 {
                    -amount
                } else {
                    amount
                }
            })
            .collect()
    }
}

/// The smallest rate the present value falls through, by dense scan.
fn scan_for_the_rate(flow: &[f64]) -> Option<f64> {
    const STEPS: u32 = 20_000;
    let (floor, ceiling) = (0.5_f64, 2.0_f64);

    let mut high = ceiling;
    let mut above = present_value(1.0 / ceiling - 1.0, flow).0;

    for step in 1..=STEPS {
        let low = ceiling - (ceiling - floor) * f64::from(step) / f64::from(STEPS);
        let below = present_value(1.0 / low - 1.0, flow).0;

        if above * below <= 0.0 && above != below {
            let (mut lo, mut hi) = (low, high);
            for _ in 0..80 {
                let middle = (lo + hi) / 2.0;
                let at = |x: f64| present_value(1.0 / x - 1.0, flow).0;
                if at(lo) * at(middle) <= 0.0 {
                    hi = middle
                } else {
                    lo = middle
                }
            }

            let root = (lo + hi) / 2.0;
            let rate = 1.0 / root - 1.0;
            if present_value(rate, flow).1 > 0.0 {
                return Some(rate);
            }
        }

        high = low;
        above = below;
    }

    None
}

/// On a flow long enough for the discounted terms to dwarf it, the rate is
/// still the one a dense scan finds.
///
/// The acceptance check measures how far the rate sits from the root, not what
/// the present value there evaluates to. It has to: past `x = 1` the terms
/// outgrow the flow by tens of orders of magnitude, so a root located to the
/// last bit `f64` carries still leaves a residue far above any amount in the
/// flow. Judging those by their residue dropped or displaced one flow in fifty.
#[test]
fn agrees_with_a_dense_scan_on_long_flows() {
    let mut flows = Flows(0x2545_F491_4F6C_DD1D);
    let mut residue_beyond_the_flow = 0;

    for _ in 0..120 {
        let flow = flows.next_flow();
        if !flow.iter().any(|a| *a > 0.0) || !flow.iter().any(|a| *a < 0.0) {
            continue;
        }

        let scanned = scan_for_the_rate(&flow);
        let reported = canonical_irr(&flow).unwrap();

        match (scanned, reported) {
            (Some(scanned), Some(reported)) => {
                assert!((scanned - reported).abs() < 1e-6, "scan {scanned}, reported {reported}");
                assert_is_a_root(reported, &flow);

                let scale = flow.iter().fold(0.0_f64, |acc, a| acc.max(a.abs()));
                if npv(reported, &flow, Some(true)).abs() > scale {
                    residue_beyond_the_flow += 1;
                }
            }
            (None, None) => {}
            (scanned, reported) => panic!("scan {scanned:?}, reported {reported:?}"),
        }
    }

    // Without these the test would pass on a residue check too, and prove
    // nothing about the reason it was replaced.
    assert!(residue_beyond_the_flow > 0, "no flow reached the range this guards");
}
