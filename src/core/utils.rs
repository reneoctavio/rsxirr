use std::ops::Range;

pub(crate) fn non_zero_range(p: &[f64]) -> Range<usize> {
    let n = p.len();
    let first_non_zero_index = p.iter().position(|&x| x != 0.0).unwrap_or(n);
    let last_non_zero_index = n - p.iter().rev().position(|&x| x != 0.0).unwrap_or(n);

    // Handle the case where all elements are zero
    if first_non_zero_index >= last_non_zero_index {
        0..0 // Return empty range
    } else {
        first_non_zero_index..last_non_zero_index
    }
}

pub(crate) fn trim_zeros(p: &[f64]) -> &[f64] {
    &p[non_zero_range(p)]
}

#[inline(always)]
pub(crate) fn fast_pow(a: f64, b: f64) -> f64 {
    // works only if a is positive
    (a.log2() * b).exp2()
}

pub(crate) fn sum_negatives_positives(values: &[f64]) -> (f64, f64) {
    values.iter().fold((0., 0.), |acc, x| {
        if x.is_sign_negative() {
            (acc.0 + x, acc.1)
        } else {
            (acc.0, acc.1 + x)
        }
    })
}

pub(crate) fn initial_guess(values: &[f64]) -> f64 {
    let (outflows, inflows) = sum_negatives_positives(values);
    let guess = inflows / -outflows - 1.0;
    guess.clamp(-0.9, 0.1)
}

pub(crate) fn is_a_good_rate_within<F>(rate: f64, tolerance: f64, f: F) -> bool
where
    F: Fn(f64) -> f64,
{
    rate.is_finite() && f(rate).abs() <= tolerance
}

/// Magnitude of a cash flow, used to judge how close to zero an NPV really is.
///
/// The npv terms sum to something of this order, so f64 cancellation puts a floor of
/// roughly `n * eps * scale` on `|npv|` at the true root. A fixed `|npv| < 1e-3` is
/// therefore both far too lax for flows in the hundreds and unreachable for flows in
/// the billions — a 1e8 cash flow cannot get `|npv|` below ~1e-8 * 1e8 = 1 no matter
/// how exact the rate is.
pub(crate) fn npv_scale(values: &[f64]) -> f64 {
    values.iter().map(|v| v.abs()).sum::<f64>().max(1.0)
}

/// Tolerance for accepting a rate a solver converged to. Sits ~5 orders of magnitude
/// above the cancellation floor, so a genuine root always passes.
pub(crate) fn converged_rate_tolerance(scale: f64) -> f64 {
    1e-9 * scale
}

/// Tolerance for accepting a rate a solver only got *near*. Keeps the historical 1e-3
/// floor for small cash flows and widens to the relative one for large ones.
pub(crate) fn approximate_rate_tolerance(scale: f64) -> f64 {
    converged_rate_tolerance(scale).max(1e-3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_zeros_all_zeros() {
        let values = vec![0.0, 0.0, 0.0];
        let result = trim_zeros(&values);
        assert!(result.is_empty(), "Expected empty slice for all zeros");
    }
}
