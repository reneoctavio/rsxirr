//! Internal rate of return, selected by definition rather than by search.

use crate::core::{
    models::{InvalidPaymentsError, validate},
    utils,
};

/// Widest rate per period reported: doubling.
const MAX_RATE: f64 = 1.0;

/// Narrowest rate per period reported: losing half.
const MIN_RATE: f64 = -0.5;

/// The band as a window in `x`, where `x = 1/(1+r)` and rises as the rate
/// falls. Every bound in the search is taken from these two.
const X_FLOOR: f64 = 1.0 / (1.0 + MAX_RATE);
const X_CEILING: f64 = 1.0 / (1.0 + MIN_RATE);

/// Steps of the downward sweep that brackets a root, where a flow changes sign
/// often enough that a root can hide anywhere in the band.
const SWEEP_STEPS: u32 = 96;

/// How many times a sweep cell may be split at a turn of the slope. One split
/// separates a pair of roots hiding in the same cell; the rest is headroom for
/// a cell holding more than one pair.
const SPLIT_DEPTH: u32 = 3;

/// Bisection steps used to locate a turn of the slope.
const TURN_STEPS: u32 = 60;

/// Refinement steps once a root is bracketed. Newton reaches `f64` in about
/// eight; the rest is headroom for a bracket that only bisects.
const REFINEMENT_STEPS: u32 = 60;

/// Relative tolerance for accepting a root, as a displacement in `x`.
const ROOT_TOLERANCE: f64 = 1e-9;

/// Whether every amount is a number.
///
/// [`validate`] admits `NaN` and infinities. They compare false against every
/// bound in the search, so without this a corrupt flow is indistinguishable
/// from one with no rate in the band.
fn validate_finite(values: &[f64]) -> Result<(), InvalidPaymentsError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(InvalidPaymentsError::new("every amount must be a finite number"))
    }
}

/// `P(x) = Σ vᵢ·xⁱ`, by Horner.
fn poly(values: &[f64], x: f64) -> f64 {
    values.iter().rev().fold(0.0, |value, coefficient| value * x + coefficient)
}

/// `P(x)` and `P'(x)`, by Horner.
fn poly_and_slope(values: &[f64], x: f64) -> (f64, f64) {
    let (mut value, mut derivative) = (0.0, 0.0);

    for coefficient in values.iter().rev() {
        derivative = derivative * x + value;
        value = value * x + coefficient;
    }

    (value, derivative)
}

/// Sign changes in the coefficients. By Descartes' rule of signs one of them
/// means exactly one positive root, so there is nothing for a sweep to find.
fn sign_changes(values: &[f64]) -> usize {
    values
        .iter()
        .filter(|coefficient| **coefficient != 0.0)
        .fold((0, None), |(changes, previous), coefficient| {
            let negative = *coefficient < 0.0;
            let changed = previous.is_some_and(|was| was != negative);
            (changes + usize::from(changed), Some(negative))
        })
        .0
}

/// Cauchy's bound: every positive root of `P` lies in `(0, bound]`.
fn root_bound(values: &[f64]) -> f64 {
    let Some(degree) = values.iter().rposition(|coefficient| *coefficient != 0.0) else {
        return 1.0;
    };

    // The ratio is against the other coefficients: including the leading one
    // would floor it at 1, and the bound at 2, whatever the flow looks like.
    let leading = values[degree].abs();
    let rest = values[..degree].iter().map(|coefficient| coefficient.abs()).fold(0.0_f64, f64::max);

    1.0 + rest / leading
}

/// The root of `P` in `[low, high]`, which must hold a sign change.
///
/// Takes a Newton step where it stays inside the bracket and keeps shrinking
/// it, and bisects where it does not. Newton alone stalls on a long flow: the
/// polynomial has the degree of the flow, so far from the root its step is
/// about one part in that degree.
fn refine(values: &[f64], low: f64, at_low: f64, high: f64, at_high: f64) -> f64 {
    if at_low == 0.0 {
        return low;
    }
    if at_high == 0.0 {
        return high;
    }

    // Orient the bracket so `P` runs negative to positive across it.
    let (mut below, mut above) = if at_low < 0.0 {
        (low, high)
    } else {
        (high, low)
    };

    let mut x = (low + high) / 2.0;
    let mut span = (high - low).abs();
    let mut previous_span = span;
    let (mut value, mut slope) = poly_and_slope(values, x);

    for _ in 0..REFINEMENT_STEPS {
        let escapes = ((x - above) * slope - value) * ((x - below) * slope - value) > 0.0;
        let stalls = (2.0 * value).abs() > (previous_span * slope).abs();

        previous_span = span;

        if escapes || stalls {
            span = (above - below) / 2.0;
            x = below + span;
        } else {
            span = value / slope;
            x -= span;
        }

        if span.abs() <= f64::EPSILON * x.abs() {
            break;
        }

        (value, slope) = poly_and_slope(values, x);

        if value < 0.0 {
            below = x;
        } else {
            above = x;
        }
    }

    x
}

/// Where the slope of `P` turns inside `[low, high]`, which must hold a change
/// of slope sign.
fn turning_point(values: &[f64], mut low: f64, mut at_low: f64, mut high: f64) -> f64 {
    for _ in 0..TURN_STEPS {
        let middle = (low + high) / 2.0;
        let slope = poly_and_slope(values, middle).1;

        if (slope < 0.0) == (at_low < 0.0) {
            low = middle;
            at_low = slope;
        } else {
            high = middle;
        }

        if high - low <= f64::EPSILON * high {
            break;
        }
    }

    (low + high) / 2.0
}

/// The rate at the largest `x` in `[low, high]` that is a return, if there is
/// one. Each endpoint is passed as its value and slope.
///
/// A sign change across the cell brackets a root. An even number of roots
/// inside one cell shows no sign change at all, so where the slope turns the
/// cell is split at that turn first: on either side of a turn `P` is monotone,
/// and a monotone piece hides nothing a sign change would not find.
fn scan(
    values: &[f64],
    low: f64,
    at_low: (f64, f64),
    high: f64,
    at_high: (f64, f64),
    depth: u32,
) -> Option<f64> {
    if depth > 0 && at_low.1 * at_high.1 < 0.0 {
        let turn = turning_point(values, low, at_low.1, high);
        let at_turn = poly_and_slope(values, turn);

        // The upper half first: a larger `x` is a smaller rate.
        return scan(values, turn, at_turn, high, at_high, depth - 1)
            .or_else(|| scan(values, low, at_low, turn, at_turn, depth - 1));
    }

    if at_low.0 * at_high.0 > 0.0 {
        return None;
    }

    accept(values, refine(values, low, at_low.0, high, at_high.0))
}

/// The rate at `root`, where it is a return that the present value confirms.
fn accept(values: &[f64], root: f64) -> Option<f64> {
    let (value, slope) = poly_and_slope(values, root);

    // Crossing upward in `x` is crossing downward in `r`.
    let falls = slope > 0.0;

    // `|P| / |P'|` is how far `root` sits from the root of `P`, and that is
    // what confirms it. The residue on its own cannot: on a long flow at a
    // negative rate the terms outgrow the flow by tens of orders of magnitude,
    // so cancellation leaves a residue far above any amount in the flow while
    // the root itself is located to the last bit `f64` carries.
    let located = value.abs() <= ROOT_TOLERANCE * slope.abs() * root.abs();

    (falls && located).then(|| 1.0 / root - 1.0)
}

/// The smallest rate per period at which a cash flow breaks even.
///
/// A flow that changes sign more than once is satisfied by several rates. This
/// returns the smallest one the present value *falls* through — where a higher
/// discount rate lowers the value, the only direction that behaves like a
/// return. Rates the value climbs through solve the same equation but are not
/// returns.
///
/// Returns `None` when no such rate lies between -50% and 100% per period, or
/// when the rate found cannot be confirmed to sit on a root.
///
/// Prefer this where the rate decides something, such as a threshold crossing
/// or a payment date. Prefer [`irr`](super::irr) where the rate is reported,
/// which is the one that agrees with a spreadsheet.
///
/// # Errors
///
/// [`InvalidPaymentsError`] if `values` holds no positive or no negative
/// amount.
///
/// # Examples
///
/// ```
/// use pyxirr::canonical_irr;
///
/// let rate = canonical_irr(&[-100.0, 39.0, 59.0, 55.0, 20.0])?.unwrap();
/// assert!((rate - 0.28094842).abs() < 1e-7);
///
/// // Satisfied at 10% and at 20%; only the second is a return.
/// let rate = canonical_irr(&[-100.0, 230.0, -132.0])?.unwrap();
/// assert!((rate - 0.2).abs() < 1e-7);
///
/// // A token outlay returned a thousandfold breaks even past any rate worth
/// // reporting.
/// assert_eq!(canonical_irr(&[-1.0, 0.0, 0.0, 1000.0])?, None);
/// # Ok::<(), pyxirr::InvalidPaymentsError>(())
/// ```
pub fn canonical_irr(values: &[f64]) -> Result<Option<f64>, InvalidPaymentsError> {
    let values = utils::trim_zeros(values);
    validate_finite(values)?;
    validate(values, None)?;

    // Under `x = 1/(1+r)` the present value is a polynomial, which stays
    // evaluable where dividing by `(1+r)ⁱ` does not.
    let bound = root_bound(values).min(X_CEILING);
    if bound <= X_FLOOR {
        return Ok(None);
    }

    // One sign change admits one positive root, by Descartes' rule, so the
    // band either holds that root or no sweep would have found it.
    if sign_changes(values) == 1 {
        let (at_floor, at_bound) = (poly(values, X_FLOOR), poly(values, bound));

        return Ok(if at_floor * at_bound <= 0.0 {
            accept(values, refine(values, X_FLOOR, at_floor, bound, at_bound))
        } else {
            None
        });
    }

    let (mut high, mut at_high) = (bound, poly_and_slope(values, bound));

    for step in 1..=SWEEP_STEPS {
        let low = bound - (bound - X_FLOOR) * f64::from(step) / f64::from(SWEEP_STEPS);
        let at_low = poly_and_slope(values, low);

        if let Some(rate) = scan(values, low, at_low, high, at_high, SPLIT_DEPTH) {
            return Ok(Some(rate));
        }

        high = low;
        at_high = at_low;
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound narrows the window it is there to narrow.
    ///
    /// Taking the maximum over every coefficient rather than over the others
    /// floors the ratio at 1 and the bound at 2, which is the band's own
    /// ceiling — the window then never narrows for any flow at all.
    #[test]
    fn root_bound_narrows_when_the_last_amount_dominates() {
        let ends_on_a_large_inflow = [-100.0, 10.0, 10.0, 5_000.0];

        assert!(root_bound(&ends_on_a_large_inflow) < 2.0);
    }

    /// Nothing above the bound is a root.
    #[test]
    fn root_bound_holds_every_positive_root() {
        let flows: [&[f64]; 4] = [
            &[-100.0, 39.0, 59.0, 55.0, 20.0],
            &[-100.0, 230.0, -132.0],
            &[-100.0, 10.0, 10.0, 5_000.0],
            &[-5_000.0, 10.0, 10.0, 100.0],
        ];

        for flow in flows {
            let bound = root_bound(flow);
            let mut previous = poly(flow, bound);

            for step in 1..=1_000 {
                let x = bound + f64::from(step) * bound / 100.0;
                let current = poly(flow, x);
                assert!(previous * current > 0.0, "a root past the bound {bound} of {flow:?}");
                previous = current;
            }
        }
    }

    /// A flow with no amount at all has no root to bound.
    #[test]
    fn root_bound_survives_an_empty_flow() {
        assert_eq!(root_bound(&[]), 1.0);
        assert_eq!(root_bound(&[0.0, 0.0]), 1.0);
    }
}
