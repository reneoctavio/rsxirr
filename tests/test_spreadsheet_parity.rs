//! What a spreadsheet answers, and what the answer has to satisfy.
//!
//! People check IRR against Excel, so agreement with a spreadsheet is part of
//! the contract. `tests/spreadsheet/goldens.csv` holds cash flows and the rate
//! a spreadsheet computed for each — 12 hand-written shapes and 60 real
//! monthly project flows, scaled so the largest is 1 (IRR does not change
//! under a positive scaling, so the rate is untouched and the amounts are
//! gone). Regenerate with `tests/spreadsheet/generate_goldens.py`.
//!
//! Agreement alone would be a weak test: it says nothing about the cases the
//! spreadsheet gives up on, and it cannot tell a right root from a wrong one
//! when a flow has several. So the goldens are checked alongside the property
//! that defines the answer — the NPV at the returned rate is zero.

use pyxirr::{irr, npv};

/// The spreadsheet's own convergence tolerance is 0.00001%.
const SPREADSHEET_TOLERANCE: f64 = 1e-7;

struct Case {
    name: String,
    guess: Option<f64>,
    expected: Option<f64>,
    flow: Vec<f64>,
}

fn cases() -> Vec<Case> {
    let raw = include_str!("spreadsheet/goldens.csv");
    raw.lines()
        .skip(1)
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            // The flow is the last field and holds spaces, so split from the left
            // exactly three times.
            let mut fields = line.splitn(4, ',');
            let name = fields.next().expect("name").to_owned();
            let guess = fields.next().expect("guess");
            let expected = fields.next().expect("expected");
            let flow = fields.next().expect("flow");

            Case {
                name,
                guess: guess.parse().ok(),
                expected: expected.parse().ok(),
                flow: flow
                    .split_whitespace()
                    .map(|value| value.parse().expect("cash flow value"))
                    .collect(),
            }
        })
        .collect()
}

/// Every rate we return agrees with the one the spreadsheet computed.
#[test]
fn matches_the_spreadsheet() {
    let mut disagreements = Vec::new();

    for case in cases() {
        let Some(expected) = case.expected else {
            continue; // the spreadsheet gave up; `solves_what_the_spreadsheet_cannot` covers it
        };
        let Ok(rate) = irr(&case.flow, case.guess) else {
            disagreements.push(format!("{}: no rate, spreadsheet said {expected}", case.name));
            continue;
        };
        if (rate - expected).abs() > SPREADSHEET_TOLERANCE {
            disagreements.push(format!("{}: {rate} vs spreadsheet {expected}", case.name));
        }
    }

    assert!(disagreements.is_empty(), "{}", disagreements.join("\n"));
}

/// The rate we return is the root, to the precision a spreadsheet claims.
///
/// This is what makes the goldens safe to trust: a rate can match a
/// spreadsheet and still be the wrong root of a flow that has several, and a
/// flow the spreadsheet could not solve still has an answer that must hold up.
///
/// The check is on the rate, not on the NPV at it. NPV is steep on a long
/// flow — a 146-month one moves by 246 per unit of rate — so a residue says
/// more about the slope than about the answer. Bisection is slow and cannot
/// be wrong, which is exactly what a reference needs to be.
#[test]
fn every_rate_is_accurate() {
    let mut wrong = Vec::new();

    for case in cases() {
        let Ok(rate) = irr(&case.flow, case.guess) else {
            wrong.push(format!("{}: no rate at all", case.name));
            continue;
        };

        let Some(refined) = bisect_near(rate, &case.flow) else {
            wrong.push(format!("{}: {rate} has no sign change around it", case.name));
            continue;
        };
        if (rate - refined).abs() > SPREADSHEET_TOLERANCE {
            wrong.push(format!("{}: {rate}, refined to {refined}", case.name));
        }
    }

    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The root nearest `rate`, found by halving a bracket around it.
///
/// Widens until the ends disagree in sign, staying above -100% where a rate
/// stops meaning anything.
fn bisect_near(rate: f64, flow: &[f64]) -> Option<f64> {
    let at = |r: f64| npv(r, flow, Some(true));

    let (mut low, mut high) = (rate, rate);
    for _ in 0..40 {
        low = (low - 1e-4).max(-0.999_999);
        high += 1e-4;
        if at(low) * at(high) <= 0.0 {
            break;
        }
    }
    if at(low) * at(high) > 0.0 {
        return None;
    }

    for _ in 0..200 {
        let middle = (low + high) / 2.0;
        if at(low) * at(middle) <= 0.0 {
            high = middle;
        } else {
            low = middle;
        }
    }
    Some((low + high) / 2.0)
}

/// Where the spreadsheet gives up, we do not.
///
/// It stops after 20 iterations, so a monthly flow whose rate sits near 1%
/// fails from its default 10% guess. Pinning this keeps someone from later
/// "fixing" us back down to the spreadsheet's reach.
#[test]
fn solves_what_the_spreadsheet_cannot() {
    for case in cases() {
        if case.expected.is_some() {
            continue;
        }
        assert!(
            irr(&case.flow, case.guess).is_ok(),
            "{}: the spreadsheet could not solve this and neither could we",
            case.name
        );
    }
}

/// The guess decides which root comes back, as it does in a spreadsheet.
///
/// A flow that changes sign more than once has more than one rate that
/// satisfies it, and nothing in the arithmetic prefers one. For longer flows
/// Newton starts at the guess and converges to the root beside it; for the
/// three-movement flow below the quadratic formula answers directly, and the
/// guess is what picks between its two roots. Comparing how near each root's
/// NPV falls to zero would decide it on rounding residue instead, which is
/// how this used to answer 20% to every guess.
#[test]
fn the_guess_decides_which_root() {
    let alternating = [-100.0, 230.0, -132.0];

    for (guess, expected) in [(None, 0.1), (Some(0.05), 0.1), (Some(0.5), 0.2)] {
        let rate = irr(&alternating, guess).unwrap();
        assert!((rate - expected).abs() < 1e-7, "guess {guess:?} returned {rate}, not {expected}");
        assert!(npv(rate, &alternating, Some(true)).abs() < 1e-7);
    }
}
