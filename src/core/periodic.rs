mod irr;
mod npv;

use super::{
    models::{InvalidPaymentsError, validate},
    optimize::{
        brentq, brentq_grid_search, newton_raphson_2, newton_raphson_warm,
        newton_raphson_with_default_deriv,
    },
    utils::{self},
};

/// Powers iterator - generates powers efficiently with vectorization hints
struct PowersIterator {
    base: f64,
    current_idx: usize,
    total_count: usize,
    base_powers: [f64; 4], // Cache for frequently used powers
}

impl PowersIterator {
    fn new(base: f64, start_power: usize, total_count: usize) -> Self {
        // Precompute powers for efficiency
        let mut powers = [1.0, base, base * base, base * base * base];
        if start_power > 0 {
            // Adjust starting powers if not beginning from base^0
            let start_base = base.powi(start_power as i32);
            for power in &mut powers {
                *power *= start_base;
            }
        }

        Self {
            base,
            current_idx: 0,
            total_count,
            base_powers: powers,
        }
    }
}

impl Iterator for PowersIterator {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.total_count {
            return None;
        }

        let result = if self.current_idx < 4 {
            // Use precomputed values for first few iterations
            self.base_powers[self.current_idx]
        } else {
            // For later values, use efficient power calculation strategy
            let idx = self.current_idx;
            let base_pow = self.base_powers[idx % 4] * self.base.powi(4 * (idx / 4) as i32);
            self.base_powers[idx % 4] = base_pow; // Update cache
            base_pow
        };

        self.current_idx += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_count - self.current_idx;
        (remaining, Some(remaining))
    }
}

/// SIMD-accelerated powers function
#[inline]
fn powers_simd(base: f64, n: usize, start_from_zero: bool) -> impl Iterator<Item = f64> {
    let start_power = if start_from_zero {
        0
    } else {
        1
    };
    let total_count = if start_from_zero {
        n + 1
    } else {
        n
    };

    PowersIterator::new(base, start_power, total_count)
}

// Replacement functions that use the new SIMD implementations
#[inline(always)]
fn powers(base: f64, n: usize, start_from_zero: bool) -> impl Iterator<Item = f64> {
    powers_simd(base, n, start_from_zero)
}

/// Calculates the net present value of a series of cash flows
///
/// # Arguments
///
/// * `rate` - Discount rate per period
/// * `values` - Array of cash flows, starting from time 0
/// * `start_from_zero` - If true, the first value is interpreted as occurring at time 0.
///   If false, the first value is interpreted as occurring at time 1. Default is true.
///
/// # Returns
///
/// The net present value of the cash flows.
///
/// # Example
///
/// ```
/// use pyxirr::npv;
/// let values = vec![-40_000.0, 5_000.0, 8_000.0, 12_000.0, 30_000.0];
/// let result = npv(0.08, &values, Some(true));
/// assert_eq!(result, 3065.2226681790715);
/// ```
#[inline(always)]
pub fn npv(rate: f64, values: &[f64], start_from_zero: Option<bool>) -> f64 {
    npv::npv_simd(rate, values, start_from_zero)
}

fn convert_pmt_at_beginning(pmt_at_beginning: bool) -> f64 {
    if pmt_at_beginning {
        1.
    } else {
        0.
    }
}

/// Calculates the future value of an investment based on periodic payments and a constant interest rate
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pmt` - Payment made each period
/// * `pv` - Present value of the investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The future value of the investment.
///
/// # Example
///
/// ```
/// use pyxirr::fv;
/// let result = fv(0.05 / 12.0, 10.0 * 12.0, -100.0, -100.0, false);
/// assert_eq!(result, 15692.92889433575);
/// ```
pub fn fv(rate: f64, nper: f64, pmt: f64, pv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(pv + pmt * nper);
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);
    let factor = f64::powf(1.0 + rate, nper);

    -pv * factor - pmt * (1.0 + rate * pmt_at_beginning) / rate * (factor - 1.0)
}

/// Calculates the present value of an investment based on future value, periodic payments, and constant interest rate
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pmt` - Payment made each period
/// * `fv` - Future value of the investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The present value of the investment.
///
/// # Example
///
/// ```
/// use pyxirr::pv;
/// let result = pv(0.05 / 12.0, 10.0 * 12.0, -100.0, 15692.93, false);
/// assert_eq!(result, -100.0006713162);
/// ```
pub fn pv(rate: f64, nper: f64, pmt: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(fv + pmt * nper);
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);
    let exp = f64::powf(1. + rate, nper);
    let factor = (1. + rate * pmt_at_beginning) * (exp - 1.) / rate;
    -(fv + pmt * factor) / exp
}

/// Calculates the periodic payment for a loan or investment with constant interest rate
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value of the loan or investment
/// * `fv` - Future value of the loan or investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The payment amount per period.
///
/// # Example
///
/// ```
/// use pyxirr::pmt;
/// let result = pmt(0.05, 10.0, 100_000.0, 0.0, false);
/// assert_eq!(result, -12950.45749654561);
/// ```
pub fn pmt(rate: f64, nper: f64, pv: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(fv + pv) / nper;
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);

    let exp = f64::powf(1.0 + rate, nper);
    let factor = (1. + rate * pmt_at_beginning) * (exp - 1.) / rate;

    -(fv + pv * exp) / factor
}

/// Calculates the interest payment for a specific period of an investment based on constant payments and interest rate
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `per` - Period for which to calculate the interest, must be between 1 and nper
/// * `nper` - Total number of payment periods
/// * `pv` - Present value of the investment
/// * `fv` - Future value of the investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The interest payment for the specified period, or NaN if per is out of bounds.
///
/// # Example
///
/// ```
/// use pyxirr::ipmt;
/// let result = ipmt(0.05, 2.0, 10.0, -50_000.0, 0.0, false);
/// assert_eq!(result, 2301.2385625860004);
/// ```
pub fn ipmt(rate: f64, per: f64, nper: f64, pv: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    // let total_pmt = self::pmt(rate, nper, pv, fv, pmt_at_beginning);
    // let result = rate * self::fv(rate, per - 1.0, total_pmt, pv, pmt_at_beginning);
    //
    // simplify r*(-P*(1+r)**(p-1)-(-(F+P*(1+r)**n)*r/((1+r*t)*((1+r)**n-1)))*(1+r*t)/r*((1+r)**(p-1)-1))

    // payments before first period don't make any sense.
    if per < 1.0 || per > nper {
        return f64::NAN;
    }

    // no interest if payment occurs at the beginning
    // of a period and this is the first period
    if per == 1.0 && pmt_at_beginning {
        return 0.0;
    }

    // no interest if rate == 0
    if rate == 0.0 {
        return 0.0;
    }

    let f1 = (rate + 1.0).powf(per);
    let f2 = (rate + 1.0).powf(nper);

    let result = (rate * (pv + fv) * f1 - rate * (rate + 1.0) * (fv + pv * f2))
        / ((rate + 1.0) * (f2 - 1.0));

    if pmt_at_beginning {
        // if paying at the beginning we need to discount by one period.
        result / (1.0 + rate)
    } else {
        result
    }
}

/// Calculates the principal payment for a specific period of an investment based on constant payments and interest rate
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `per` - Period for which to calculate the principal payment, must be between 1 and nper
/// * `nper` - Total number of payment periods
/// * `pv` - Present value of the investment
/// * `fv` - Future value of the investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The principal payment for the specified period, or NaN if per is out of bounds.
///
/// # Example
///
/// ```
/// use pyxirr::ppmt;
/// let result = ppmt(0.05, 2.0, 10.0, -50_000.0, 0.0, false);
/// assert_eq!(result, 4173.9901856864);
/// ```
pub fn ppmt(rate: f64, per: f64, nper: f64, pv: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    // assuming type = 1 if pmt_at_beginning else 0
    // assuming P=pv;F=fv;r=rate;n=nper;p=per;t=type, type in {1;0}
    // ppmt = fv(r,p-1,pmt(r,n,P,F,t),P,t) - fv(r,p,pmt(r,n,P,F,t),P,t)
    // after substitution:
    // simplify (-P*(1+r)^(p-1)-(-(F+P*(1+r)^n)*r/((1+r)^n-1)/(1+r*t))*(1+r*t)/r*((1+r)^(p-1)-1)) - (-P*(1+r)^p-(-(F+P*(1+r)^n)*r/((1+r)^n-1)/(1+r*t))*(1+r*t)/r*((1+r)^p-1))
    // shorter formula: -r*(F+P)*(r+1)^(per-1)/((r+1)^n - 1)
    // type correction: result /= r + 1 if type = 1
    // denominator => 1/((r+1)^p-1) => 1/(((r+1)^p-1)*(r+1)) =>
    // => 1/((r+1)^(p+1) - (r+1)) => 1/((r+1)^(p+t) -r*t + 1)
    //
    // if rate == 0:
    // simplify (-P-(-(F+P)/n) *(p-1) - (-P-(-(F+P)/n)*p))
    // shorter: -(F + P) / n;

    if per < 1.0 || per > nper {
        return f64::NAN;
    }

    if rate == 0.0 {
        return -(fv + pv) / nper;
    }

    let when = convert_pmt_at_beginning(pmt_at_beginning);
    -rate * (fv + pv) * (rate + 1.).powf(per - 1.)
        / ((rate + 1.).powf(nper + when) - rate * when - 1.)
}

/// Calculates the number of periods required for an investment to reach a specified future value
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `pmt` - Payment made each period
/// * `pv` - Present value of the investment
/// * `fv` - Future value of the investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The number of periods needed.
///
/// # Example
///
/// ```
/// use pyxirr::nper;
/// let result = nper(0.075, -2000.0, 0.0, 100_000.0, false);
/// assert_eq!(result, 21.544944197323336);
/// ```
pub fn nper(rate: f64, pmt: f64, pv: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(fv + pv) / pmt;
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);

    let z = pmt * (1. + rate * pmt_at_beginning) / rate;
    f64::log10((-fv + z) / (pv + z)) / f64::log10(1. + rate)
}

/// Calculates the interest rate per period of an investment
///
/// # Arguments
///
/// * `nper` - Total number of payment periods
/// * `pmt` - Payment made each period
/// * `pv` - Present value of the investment
/// * `fv` - Future value of the investment
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
/// * `guess` - Initial guess for the rate (default is 0.1)
///
/// # Returns
///
/// The interest rate per period.
///
/// # Example
///
/// ```
/// use pyxirr::rate;
/// let result = rate(10.0, -12950.46, 100_000.0, 0.0, false, None);
/// assert_eq!(result, 0.05);
/// ```
pub fn rate(
    nper: f64,
    pmt: f64,
    pv: f64,
    fv: f64,
    pmt_at_beginning: bool,
    guess: Option<f64>,
) -> f64 {
    newton_raphson_with_default_deriv(guess.unwrap_or(0.1), |rate| {
        fv - self::fv(rate, nper, pmt, pv, pmt_at_beginning)
    })
}

/// Calculates the net future value of a series of cash flows
///
/// http://westclintech.com/SQL-Server-Financial-Functions/SQL-Server-NFV-function
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Number of periods to project the NFV
/// * `amounts` - Array of cash flows
///
/// # Returns
///
/// The net future value of the cash flows.
///
/// # Example
///
/// ```
/// use pyxirr::nfv;
/// let amounts = vec![1050.0, 1350.0, 1350.0, 1450.0];
/// let result = nfv(0.03, 6.0, &amounts);
/// assert!((result - 5750.16).abs() < 0.01);
/// ```
pub fn nfv(rate: f64, nper: f64, amounts: &[f64]) -> f64 {
    let pv = npv(rate, amounts, Some(false));
    fv(rate, nper, 0.0, -pv, false)
}

/// Calculates the Internal Rate of Return (IRR) for a series of cash flows
///
/// # Arguments
///
/// * `values` - Array of cash flows where the first value is the initial investment (negative value)
///   and subsequent values are returns (positive values)
/// * `guess` - Initial guess for the rate (default is 0.1)
///
/// # Returns
///
/// A Result containing either the IRR as a decimal value, or an error if the calculation fails.
///
/// # Example
///
/// ```
/// use pyxirr::irr;
/// let values = vec![-100.0, 39.0, 59.0, 55.0, 20.0];
/// let result = irr(&values, None).unwrap();
/// assert!((result - 0.28094842116).abs() < 1e-7);
/// ```
pub fn irr(values: &[f64], guess: Option<f64>) -> Result<f64, InvalidPaymentsError> {
    let values = utils::trim_zeros(values);
    validate(values, None)?;

    // Fast path for small datasets
    if values.len() == 2 {
        return Ok(irr::irr_analytical_2(values));
    }

    if values.len() == 3 {
        return Ok(irr::irr_analytical_3(values));
    }

    // A caller-supplied guess means "the root is near here". The dominant case is an IRR
    // series, where each period's rate is close to the previous period's, so Newton from
    // that start converges in a couple of iterations. The bracket search below ignores the
    // guess entirely and always sweeps [0, 100], so without this the guess only ever took
    // effect when brentq failed outright.
    //
    // Capped low: a guess that turns out to be bad costs a few NPV evaluations before
    // falling through to the unchanged cascade.
    // `npv_scale` and `initial_guess` are both O(n), so they stay off the path taken when
    // the bracket search below succeeds on its own — which is the common case.
    if let Some(guess) = guess {
        const WARM_START_ITERATIONS: u32 = 6;

        let rate = newton_raphson_warm(
            guess,
            &|r| npv::npv_with_deriv_simd(r, values),
            WARM_START_ITERATIONS,
        );

        let tolerance = utils::converged_rate_tolerance(utils::npv_scale(values));
        if utils::is_a_good_rate_within(rate, tolerance, |r| npv(r, values, Some(true))) {
            return Ok(rate);
        }
    }

    // Try Brent with positive brackets
    let rate = brentq(&|r| npv(r, values, Some(true)), 0.0, 100.0, 100);
    if rate.is_finite() {
        return Ok(rate);
    }

    // Fallback to Newton-Raphson. The seed is derived from the cash flow rather than a flat
    // 0.1, which starts on the wrong side of zero for the loss-making prefixes of an IRR
    // series. `xirr` already seeds this way.
    let initial_guess = guess.unwrap_or_else(|| {
        // npv(0) is just the sum of the cash flow, and npv decreases in the rate, so a
        // negative sum puts the root below zero — where a flat 0.1 seed starts on the wrong
        // side and Newton has to walk back across it. That is the loss-making prefix of an
        // IRR series. Seeding from the data only in that case leaves the (much more common)
        // positive-rate cash flows on their original, well-tuned 0.1 start.
        if values.iter().sum::<f64>() < 0.0 {
            utils::initial_guess(values)
        } else {
            0.1
        }
    });
    let rate = newton_raphson_2(initial_guess, &|r| npv::npv_with_deriv_simd(r, values));

    // Scale-relative: with a flat 1e-3 this rejected perfectly good rates for large cash
    // flows and fell through to the expensive [-0.999, 100] sweep below.
    let tolerance = utils::approximate_rate_tolerance(utils::npv_scale(values));
    if utils::is_a_good_rate_within(rate, tolerance, |r| npv(r, values, Some(true))) {
        return Ok(rate);
    }

    // Expended search for negative rates
    let rate = brentq(&|r| npv(r, values, Some(true)), -0.999, 100.0, 100);
    if rate.is_finite() {
        return Ok(rate);
    }

    // Final fallback with minimal iterations (to avoid the catastrophic slowdown)
    let breakpoints = &[-0.99999999999999, -0.75, -0.5, -0.25, 0., 0.25, 0.5, 1.0, 1e6];
    let f = |r| npv(r, values, Some(true));
    let rate = brentq_grid_search(&[breakpoints], &f).next();

    Ok(rate.unwrap_or(f64::NAN))
}

/// Calculates the Modified Internal Rate of Return (MIRR) for a series of cash flows
///
/// # Arguments
///
/// * `values` - Array of cash flows where the first value is the initial investment (negative value)
///   and subsequent values are returns (positive values)
/// * `finance_rate` - Interest rate paid on the funds invested
/// * `reinvest_rate` - Interest rate received on reinvestment of cash flows
///
/// # Returns
///
/// A Result containing either the MIRR as a decimal value, or an error if the calculation fails.
///
/// # Example
///
/// ```
/// use pyxirr::mirr;
/// let values = vec![-1000.0, 100.0, 250.0, 500.0, 500.0];
/// let result = mirr(&values, 0.1, 0.1).unwrap();
/// assert!((result - 0.10401626745).abs() < 1e-7);
/// ```
pub fn mirr(
    values: &[f64],
    finance_rate: f64,
    reinvest_rate: f64,
) -> Result<f64, InvalidPaymentsError> {
    // must contain at least one positive and one negative value
    validate(values, None)?;

    let positive: f64 = powers(1. + reinvest_rate, values.len(), true)
        .zip(values.iter().rev())
        .filter(|&(_, v)| *v > 0.0)
        .map(|(r, v)| v * r)
        .sum();

    let negative: f64 = powers(1. + finance_rate, values.len(), true)
        .zip(values.iter())
        .filter(|&(_, v)| *v < 0.0)
        .map(|(r, &v)| v / r)
        .sum();

    Ok((positive / -negative).powf(1.0 / (values.len() - 1) as f64) - 1.0)
}

/// Calculates the cumulative principal payment between start_period and end_period
///
/// https://wiki.documentfoundation.org/Documentation/Calc_Functions/CUMPRINC
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value of the investment
/// * `start_period` - First period in the calculation
/// * `end_period` - Last period in the calculation
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The cumulative principal payment.
///
/// # Example
///
/// ```
/// use pyxirr::cumprinc;
/// let result = cumprinc(0.09 / 12.0, 30.0 * 12.0, 125_000.0, 13.0, 24.0, false);
/// assert!((result + 934.10712).abs() < 1e-5);
/// ```
pub fn cumprinc(
    rate: f64,
    nper: f64,
    pv: f64,
    start_period: f64,
    end_period: f64,
    pmt_at_beginning: bool,
) -> f64 {
    (start_period.trunc() as u64..=end_period.trunc() as u64)
        .map(|per| ppmt(rate, per as f64, nper, pv, 0.0, pmt_at_beginning))
        .sum()
}

/// Calculates the cumulative interest payment between start_period and end_period
///
/// https://wiki.documentfoundation.org/Documentation/Calc_Functions/CUMIPMT
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value of the investment
/// * `start_period` - First period in the calculation
/// * `end_period` - Last period in the calculation
/// * `pmt_at_beginning` - If true, payments are made at the beginning of each period;
///   if false, at the end of each period
///
/// # Returns
///
/// The cumulative interest payment.
///
/// # Example
///
/// ```
/// use pyxirr::cumipmt;
/// let result = cumipmt(0.09 / 12.0, 30.0 * 12.0, 125_000.0, 13.0, 24.0, false);
/// assert!((result + 11135.23213).abs() < 1e-5);
/// ```
pub fn cumipmt(
    rate: f64,
    nper: f64,
    pv: f64,
    start_period: f64,
    end_period: f64,
    pmt_at_beginning: bool,
) -> f64 {
    (start_period.trunc() as u64..=end_period.trunc() as u64)
        .map(|per| ipmt(rate, per as f64, nper, pv, 0.0, pmt_at_beginning))
        .sum()
}
