//! Internal rate of return, selected by definition rather than by search.

use crate::core::{
    models::{InvalidPaymentsError, validate},
    utils,
};

/// Widest rate per period reported: doubling.
const MAX_RATE: f64 = 1.0;

/// Narrowest rate per period reported: losing half.
const MIN_RATE: f64 = -0.5;

/// Steps of the downward sweep that brackets a root, where a flow changes sign
/// often enough that a root can hide anywhere in the band.
const SWEEP_STEPS: u32 = 96;

/// Refinement steps once a root is bracketed. Newton reaches `f64` in about
/// eight; the rest is headroom for a bracket that only bisects.
const REFINEMENT_STEPS: u32 = 60;

/// Relative tolerance for accepting a root, on the scale of the flow.
const ROOT_TOLERANCE: f64 = 1e-8;

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
    let leading = values
        .iter()
        .rposition(|coefficient| *coefficient != 0.0)
        .map_or(1.0, |index| values[index].abs());

    if leading == 0.0 {
        return 1.0;
    }

    1.0 + values.iter().map(|coefficient| coefficient.abs()).fold(0.0_f64, f64::max) / leading
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

/// The rate at `root`, where it is a return that the present value confirms.
fn accept(values: &[f64], root: f64) -> Option<f64> {
    let rate = 1.0 / root - 1.0;
    let scale = values.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);

    // Crossing upward in `x` is crossing downward in `r`.
    let falls = poly_and_slope(values, root).1 > 0.0;
    let zeroes = super::npv(rate, values, Some(true)).abs() <= scale * ROOT_TOLERANCE;

    (falls && zeroes).then_some(rate)
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
/// when the rate found cannot be confirmed to zero the present value.
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
    validate(values, None)?;

    // Under `x = 1/(1+r)` the present value is a polynomial, which stays
    // evaluable where dividing by `(1+r)ⁱ` does not. The rate band becomes an
    // `x` window, and `x` falls as the rate rises.
    let floor = 1.0 / (1.0 + MAX_RATE);
    let bound = root_bound(values).min(1.0 / (1.0 + MIN_RATE));
    if bound <= floor {
        return Ok(None);
    }

    let at_bound = poly(values, bound);

    // One sign change admits one positive root, so the band either holds it or
    // no sweep would have found it.
    if sign_changes(values) == 1 {
        let at_floor = poly(values, floor);

        return Ok(if at_floor * at_bound <= 0.0 {
            accept(values, refine(values, floor, at_floor, bound, at_bound))
        } else {
            None
        });
    }

    let (mut high, mut at_high) = (bound, at_bound);

    for step in 1..=SWEEP_STEPS {
        let low = bound - (bound - floor) * f64::from(step) / f64::from(SWEEP_STEPS);
        let at_low = poly(values, low);

        if at_low * at_high <= 0.0
            && let Some(rate) = accept(values, refine(values, low, at_low, high, at_high))
        {
            return Ok(Some(rate));
        }

        high = low;
        at_high = at_low;
    }

    Ok(None)
}
