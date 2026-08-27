//! Behaviour of [`canonical_irr`].

use pyxirr::{canonical_irr, irr, npv};

mod common;

/// The present value at `rate` is zero, on the scale of the flow.
fn assert_is_a_root(rate: f64, flow: &[f64]) {
    let scale = flow.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let residue = npv(rate, flow, Some(true)).abs();
    assert!(residue <= scale * 1e-6, "npv at {rate} is {residue}, not zero");
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
