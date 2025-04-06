use std::cmp::Ordering;

use super::{
    models::{validate, InvalidPaymentsError},
    optimize::{brentq, brentq_grid_search, newton_raphson_2, newton_raphson_with_default_deriv},
    utils::{self},
};

/// Powers iterator - generates powers efficiently with vectorization hints
pub struct PowersIterator {
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
pub fn powers_simd(base: f64, n: usize, start_from_zero: bool) -> impl Iterator<Item = f64> {
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

/// SIMD-accelerated NPV calculation
#[inline]
pub fn npv_simd(rate: f64, values: &[f64], start_from_zero: Option<bool>) -> f64 {
    if rate == 0.0 {
        return values.iter().sum();
    }

    let start_from_zero = start_from_zero.unwrap_or(true);
    let base = 1.0 + rate;

    #[cfg(target_arch = "x86_64")]
    {
        // Skip AVX if environment variable is set
        if std::env::var("ENABLE_AVX").map(|x| x == "1").unwrap_or(true)
            && is_x86_feature_detected!("avx")
            && values.len() > 8
        {
            return unsafe { npv_simd_avx(base, values, start_from_zero) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::env::var("ENABLE_NEON").map(|x| x == "1").unwrap_or(true) && values.len() > 8 {
            return unsafe { npv_simd_neon(base, values, start_from_zero) };
        }
    }

    // Fallback optimized for auto-vectorization
    npv_autovec(base, values, start_from_zero)
}

/// Auto-vectorizable NPV implementation
#[inline]
fn npv_autovec(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
    // Use multiple accumulators to help compiler auto-vectorize
    const UNROLL: usize = 4;
    let mut power = if start_from_zero {
        1.0
    } else {
        base
    };

    // Create multiple accumulators
    let mut sums = [0.0; UNROLL];
    let n = values.len();
    let main_part = n - (n % UNROLL);

    // Main loop with multiple accumulators
    for chunk in 0..(main_part / UNROLL) {
        let mut local_powers = [power; UNROLL];

        // Calculate powers for this chunk
        for i in 1..UNROLL {
            local_powers[i] = local_powers[i - 1] * base;
        }

        // Update accumulators
        for i in 0..UNROLL {
            let idx = chunk * UNROLL + i;
            sums[i] += values[idx] / local_powers[i];
        }

        // Update power for next chunk
        power = local_powers[UNROLL - 1] * base;
    }

    // Combine accumulators into a sum
    let mut sum = sums.iter().sum();

    // Process remaining elements
    for &value in &values[main_part..n] {
        sum += value / power;
        power *= base;
    }

    sum
}

/// AVX implementation for NPV
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn npv_simd_avx(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = 0.0;
    let mut power = if start_from_zero {
        1.0
    } else {
        base
    };

    let vec_base = _mm256_set1_pd(base);
    let chunk_size = 4;

    // Process 4 elements at a time
    let simd_limit = (values.len() / chunk_size) * chunk_size;

    for i in (0..simd_limit).step_by(chunk_size) {
        // Create initial vector with same power
        let vec_power_base = _mm256_set1_pd(power);

        // Create multipliers [1, base, base^2, base^3]
        let vec_mult = _mm256_set_pd(base * base * base, base * base, base, 1.0);

        // Multiply to get [power, power*base, power*base^2, power*base^3]
        let vec_power = _mm256_mul_pd(vec_power_base, vec_mult);

        // Load values
        let vec_values = _mm256_loadu_pd(&values[i]);

        // Divide values by powers
        let vec_result = _mm256_div_pd(vec_values, vec_power);

        // Sum the results
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), vec_result);
        sum += result_array.iter().sum::<f64>();

        // Update power for next chunk - power *= base^4
        // Use SIMD to compute base^4
        let base4 =
            _mm256_mul_pd(_mm256_mul_pd(vec_base, vec_base), _mm256_mul_pd(vec_base, vec_base));
        // Extract the scalar value
        power *= _mm256_cvtsd_f64(base4);
    }

    // Process remaining elements
    for &value in &values[simd_limit..] {
        sum += value / power;
        power *= base;
    }

    sum
}

/// NEON implementation for NPV (Apple Silicon, AWS Graviton)
#[cfg(target_arch = "aarch64")]
unsafe fn npv_simd_neon(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
    use std::arch::aarch64::*;

    let mut sum = 0.0;
    let mut power = if start_from_zero {
        1.0
    } else {
        base
    };

    let vec_base = vdupq_n_f64(base);
    let chunk_size = 2; // NEON processes 2 doubles at a time

    // Process 2 elements at a time
    let simd_limit = (values.len() / chunk_size) * chunk_size;

    for i in (0..simd_limit).step_by(chunk_size) {
        // Create initial vector with same power
        let vec_power_base = vdupq_n_f64(power);

        // Create multipliers [1, base]
        let vec_mult = vcombine_f64(vdup_n_f64(1.0), vdup_n_f64(base));

        // Multiply to get [power, power*base]
        let vec_power = vmulq_f64(vec_power_base, vec_mult);

        // Load values
        let vec_values = vld1q_f64(&values[i]);

        // Divide values by powers
        let vec_result = vdivq_f64(vec_values, vec_power);

        // Sum the results
        let mut result_array = [0.0; 2];
        vst1q_f64(result_array.as_mut_ptr(), vec_result);
        sum += result_array.iter().sum::<f64>();

        // Update power for next chunk - power *= base²
        let base2 = vmulq_f64(vec_base, vec_base);
        power *= vgetq_lane_f64(base2, 0);
    }

    // Process remaining elements
    for i in simd_limit..values.len() {
        sum += values[i] / power;
        power *= base;
    }

    sum
}

/// SIMD-accelerated NPV with derivative calculation
#[inline]
pub fn npv_with_deriv_simd(rate: f64, values: &[f64]) -> (f64, f64) {
    if rate <= -1.0 {
        return (f64::INFINITY, f64::INFINITY);
    }

    if values.is_empty() {
        return (0.0, 0.0);
    }

    // Process first value separately
    let sum = values[0];
    let deriv = 0.0;

    if values.len() <= 1 {
        return (sum, deriv);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::env::var("ENABLE_AVX").map(|x| x == "1").unwrap_or(true)
            && is_x86_feature_detected!("avx")
            && values.len() > 8
        {
            let (simd_sum, simd_deriv) = unsafe { npv_with_deriv_avx(rate, &values[1..], 1) };
            return (sum + simd_sum, deriv + simd_deriv);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::env::var("ENABLE_NEON").map(|x| x == "1").unwrap_or(true) && values.len() > 8 {
            let (simd_sum, simd_deriv) = unsafe { npv_with_deriv_neon(rate, &values[1..], 1) };
            return (sum + simd_sum, deriv + simd_deriv);
        }
    }

    // Fall back to auto-vectorized version
    let (auto_sum, auto_deriv) = npv_with_deriv_autovec(rate, &values[1..], 1);
    (sum + auto_sum, deriv + auto_deriv)
}

/// Auto-vectorizable implementation of NPV with derivative
#[inline]
fn npv_with_deriv_autovec(rate: f64, values: &[f64], start_index: usize) -> (f64, f64) {
    let base = 1.0 + rate;
    let inv_base = 1.0 / base;

    // Use multiple accumulators to help compiler auto-vectorize
    const UNROLL: usize = 4;
    let mut sums = [0.0; UNROLL];
    let mut derivs = [0.0; UNROLL];

    let mut power = base;
    let n = values.len();
    let main_part = n - (n % UNROLL);

    // Process main part with multiple accumulators
    for chunk in 0..(main_part / UNROLL) {
        let mut powers = [power; UNROLL];

        // Calculate powers for this chunk
        for i in 1..UNROLL {
            powers[i] = powers[i - 1] * base;
        }

        // Update accumulators
        for i in 0..UNROLL {
            let idx = chunk * UNROLL + i;
            let term = values[idx] / powers[i];
            sums[i] += term;

            let index = start_index + idx;
            derivs[i] -= (index as f64) * term * inv_base;
        }

        // Update power for next chunk
        power = powers[UNROLL - 1] * base;
    }

    // Combine accumulators
    let mut sum = sums.iter().sum();
    let mut deriv = derivs.iter().sum();

    // Process remaining elements
    for (i, &value) in values[main_part..n].iter().enumerate() {
        let term = value / power;
        sum += term;

        let index = start_index + main_part + i;
        deriv -= (index as f64) * term * inv_base;

        power *= base;
    }

    (sum, deriv)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn npv_with_deriv_avx(rate: f64, values: &[f64], start_index: usize) -> (f64, f64) {
    use std::arch::x86_64::*;

    let base = 1.0 + rate;
    let inv_base = 1.0 / base;

    let mut sum = 0.0;
    let mut deriv = 0.0;
    let mut power = base;

    let vec_base = _mm256_set1_pd(base);
    let vec_inv_base = _mm256_set1_pd(inv_base);
    let vec_neg_one = _mm256_set1_pd(-1.0);
    let chunk_size = 4;

    // Process chunks of 4 elements
    let simd_limit = (values.len() / chunk_size) * chunk_size;

    // Pre-compute base^2 and base^3 using SIMD operations
    let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
    let vec_base_cubed = _mm256_mul_pd(vec_base_squared, vec_base);
    // Pre-compute base^4 for power updates
    let vec_base4 = _mm256_mul_pd(vec_base_squared, vec_base_squared);
    let base4 = _mm256_cvtsd_f64(vec_base4);

    for chunk_idx in 0..(simd_limit / chunk_size) {
        let idx = chunk_idx * chunk_size;

        // Create initial vector with same power
        let vec_power_base = _mm256_set1_pd(power);

        // Create multipliers vector using pre-computed SIMD values
        // [1.0, base, base^2, base^3]
        let vec_mult = _mm256_set_pd(
            _mm256_cvtsd_f64(vec_base_cubed),   // base^3
            _mm256_cvtsd_f64(vec_base_squared), // base^2
            base,                               // base
            1.0,                                // 1.0
        );

        // Multiply to get [power, power*base, power*base^2, power*base^3]
        let vec_power = _mm256_mul_pd(vec_power_base, vec_mult);

        // Load values
        let vec_values = _mm256_loadu_pd(&values[idx]);

        // Create index vector for derivative
        let vec_indices = _mm256_set_pd(
            (start_index + idx + 3) as f64,
            (start_index + idx + 2) as f64,
            (start_index + idx + 1) as f64,
            (start_index + idx) as f64,
        );

        // Calculate terms
        let vec_term = _mm256_div_pd(vec_values, vec_power);

        // Calculate derivative terms
        let vec_deriv = _mm256_mul_pd(vec_indices, vec_term);
        let vec_deriv = _mm256_mul_pd(vec_deriv, vec_inv_base);
        let vec_deriv = _mm256_mul_pd(vec_deriv, vec_neg_one);

        // Store results
        let mut term_array = [0.0; 4];
        let mut deriv_array = [0.0; 4];
        _mm256_storeu_pd(term_array.as_mut_ptr(), vec_term);
        _mm256_storeu_pd(deriv_array.as_mut_ptr(), vec_deriv);

        sum += term_array.iter().sum::<f64>();
        deriv += deriv_array.iter().sum::<f64>();

        // Update power for next chunk using pre-computed base^4
        power *= base4;
    }

    // Process remaining elements
    for (i, &value) in values[simd_limit..].iter().enumerate() {
        let term = value / power;
        sum += term;

        // Adjust index since enumerate starts from 0 for the slice
        let index = start_index + simd_limit + i;
        deriv -= (index as f64) * term * inv_base;

        power *= base;
    }

    (sum, deriv)
}

#[cfg(target_arch = "aarch64")]
unsafe fn npv_with_deriv_neon(rate: f64, values: &[f64], start_index: usize) -> (f64, f64) {
    use std::arch::aarch64::*;

    let base = 1.0 + rate;
    let inv_base = 1.0 / base;

    let mut sum = 0.0;
    let mut deriv = 0.0;
    let mut power = base;

    let vec_base = vdupq_n_f64(base);
    let vec_inv_base = vdupq_n_f64(inv_base);
    let vec_neg_one = vdupq_n_f64(-1.0);
    let chunk_size = 2;

    // Process chunks of 2 elements
    let simd_limit = (values.len() / chunk_size) * chunk_size;

    // Pre-compute base^2 for power updates
    let vec_base_squared = vmulq_f64(vec_base, vec_base);
    let base2 = vgetq_lane_f64(vec_base_squared, 0);

    for chunk_idx in 0..(simd_limit / chunk_size) {
        let idx = chunk_idx * chunk_size;

        // Create initial vector with same power
        let vec_power_base = vdupq_n_f64(power);

        // Create multipliers [1.0, base]
        let vec_mult = vcombine_f64(vdup_n_f64(1.0), vdup_n_f64(base));

        // Multiply to get [power, power*base]
        let vec_power = vmulq_f64(vec_power_base, vec_mult);

        // Load values
        let vec_values = vld1q_f64(&values[idx]);

        // Create index vector for derivative
        let vec_indices = vcombine_f64(
            vdup_n_f64((start_index + idx) as f64),
            vdup_n_f64((start_index + idx + 1) as f64),
        );

        // Calculate terms
        let vec_term = vdivq_f64(vec_values, vec_power);

        // Calculate derivative terms
        let vec_deriv = vmulq_f64(vec_indices, vec_term);
        let vec_deriv = vmulq_f64(vec_deriv, vec_inv_base);
        let vec_deriv = vmulq_f64(vec_deriv, vec_neg_one);

        // Store results
        let mut term_array = [0.0; 2];
        let mut deriv_array = [0.0; 2];
        vst1q_f64(term_array.as_mut_ptr(), vec_term);
        vst1q_f64(deriv_array.as_mut_ptr(), vec_deriv);

        sum += term_array.iter().sum::<f64>();
        deriv += deriv_array.iter().sum::<f64>();

        // Update power for next chunk
        power *= base2;
    }

    // Process remaining elements
    for i in simd_limit..values.len() {
        let term = values[i] / power;
        sum += term;
        deriv -= ((start_index + i) as f64) * term * inv_base;
        power *= base;
    }

    (sum, deriv)
}

// Replacement functions that use the new SIMD implementations
#[inline(always)]
pub fn powers(base: f64, n: usize, start_from_zero: bool) -> impl Iterator<Item = f64> {
    powers_simd(base, n, start_from_zero)
}

#[inline(always)]
pub fn npv(rate: f64, values: &[f64], start_from_zero: Option<bool>) -> f64 {
    npv_simd(rate, values, start_from_zero)
}

#[inline(always)]
fn npv_with_deriv(rate: f64, values: &[f64]) -> (f64, f64) {
    npv_with_deriv_simd(rate, values)
}

fn convert_pmt_at_beginning(pmt_at_beginning: bool) -> f64 {
    if pmt_at_beginning {
        1.
    } else {
        0.
    }
}

pub fn fv(rate: f64, nper: f64, pmt: f64, pv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(pv + pmt * nper);
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);
    let factor = f64::powf(1.0 + rate, nper);

    -pv * factor - pmt * (1.0 + rate * pmt_at_beginning) / rate * (factor - 1.0)
}

pub fn pv(rate: f64, nper: f64, pmt: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(fv + pmt * nper);
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);
    let exp = f64::powf(1. + rate, nper);
    let factor = (1. + rate * pmt_at_beginning) * (exp - 1.) / rate;
    -(fv + pmt * factor) / exp
}

pub fn pmt(rate: f64, nper: f64, pv: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(fv + pv) / nper;
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);

    let exp = f64::powf(1.0 + rate, nper);
    let factor = (1. + rate * pmt_at_beginning) * (exp - 1.) / rate;

    -(fv + pv * exp) / factor
}

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

pub fn nper(rate: f64, pmt: f64, pv: f64, fv: f64, pmt_at_beginning: bool) -> f64 {
    if rate == 0.0 {
        return -(fv + pv) / pmt;
    }

    let pmt_at_beginning = convert_pmt_at_beginning(pmt_at_beginning);

    let z = pmt * (1. + rate * pmt_at_beginning) / rate;
    f64::log10((-fv + z) / (pv + z)) / f64::log10(1. + rate)
}

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

// http://westclintech.com/SQL-Server-Financial-Functions/SQL-Server-NFV-function
pub fn nfv(rate: f64, nper: f64, amounts: &[f64]) -> f64 {
    let pv = npv(rate, amounts, Some(false));
    fv(rate, nper, 0.0, -pv, false)
}

// #[inline(always)]
// pub fn npv(rate: f64, values: &[f64], start_from_zero: Option<bool>) -> f64 {
//     if rate == 0.0 {
//         return values.iter().sum();
//     }
//
//     let start_from_zero = start_from_zero.unwrap_or(true);
//     let base = 1.0 + rate;
//
//     // Manual loop unrolling and avoiding Vec allocation for powers
//     let mut sum = 0.0;
//     let mut power = if start_from_zero {
//         1.0
//     } else {
//         base
//     };
//
//     for &v in values {
//         sum += v / power;
//         power *= base;
//     }
//
//     sum
// }
//
// #[cfg(target_arch = "x86_64")]
// #[inline(always)]
// fn npv_with_deriv(rate: f64, values: &[f64]) -> (f64, f64) {
//     use std::arch::x86_64::*;
//
//     if rate <= -1.0 {
//         return (f64::INFINITY, f64::INFINITY);
//     }
//
//     // Process first value separately (not discounted)
//     let mut sum = values[0];
//     let mut deriv = 0.0;
//
//     if values.len() <= 1 {
//         return (sum, deriv);
//     }
//
//     let base = 1.0 + rate;
//     let inv_base = 1.0 / base;
//
//     // Decide whether to use SIMD or scalar version
//     if is_x86_feature_detected!("avx2") && values.len() > 8 {
//         unsafe {
//             // Set up SIMD registers
//             let vec_base = _mm256_set1_pd(base);
//             let vec_inv_base = _mm256_set1_pd(inv_base);
//
//             // Start with the second value
//             let mut power = base;
//
//             // Process 4 elements at a time
//             let chunk_size = 4;
//             let chunks = (values.len() - 1) / chunk_size;
//
//             for chunk_idx in 0..chunks {
//                 let idx = 1 + chunk_idx * chunk_size;
//
//                 // Compute power vector more efficiently using vec_base
//                 let vec_p0 = _mm256_set1_pd(power);
//                 let vec_p1 = _mm256_mul_pd(vec_p0, vec_base);
//                 let vec_p2 = _mm256_mul_pd(vec_p1, vec_base);
//                 let vec_p3 = _mm256_mul_pd(vec_p2, vec_base);
//                 let vec_power = _mm256_set_pd(
//                     _mm256_cvtsd_f64(vec_p3), // power * base^3
//                     _mm256_cvtsd_f64(vec_p2), // power * base^2
//                     _mm256_cvtsd_f64(vec_p1), // power * base^1
//                     _mm256_cvtsd_f64(vec_p0), // power
//                 );
//
//                 // Load 4 values
//                 let vec_values = _mm256_loadu_pd(&values[idx]);
//
//                 // Calculate indices for derivative
//                 let vec_indices = _mm256_set_pd(
//                     (idx + 3) as f64,
//                     (idx + 2) as f64,
//                     (idx + 1) as f64,
//                     (idx) as f64,
//                 );
//
//                 // Calculate v / power
//                 let vec_term = _mm256_div_pd(vec_values, vec_power);
//
//                 // Calculate horizontal sum of terms
//                 let mut term_array = [0.0; 4];
//                 _mm256_storeu_pd(term_array.as_mut_ptr(), vec_term);
//                 sum += term_array.iter().sum::<f64>();
//
//                 // Calculate derivative: -i * term / base
//                 let vec_deriv = _mm256_mul_pd(vec_indices, vec_term);
//                 let vec_deriv = _mm256_mul_pd(vec_deriv, vec_inv_base);
//
//                 // Calculate horizontal sum of derivative terms
//                 let mut deriv_array = [0.0; 4];
//                 _mm256_storeu_pd(deriv_array.as_mut_ptr(), vec_deriv);
//                 deriv -= deriv_array.iter().sum::<f64>();
//
//                 // Update power for next chunk - use vec_base for multiplies
//                 power *= _mm256_cvtsd_f64(_mm256_mul_pd(
//                     _mm256_mul_pd(_mm256_mul_pd(vec_base, vec_base), vec_base),
//                     vec_base,
//                 ));
//             }
//
//             // Process remaining elements
//             let start = 1 + chunks * chunk_size;
//             for i in start..values.len() {
//                 let term = values[i] / power;
//                 sum += term;
//                 deriv -= (i as f64) * term * inv_base;
//                 power *= base;
//             }
//         }
//     } else {
//         // Scalar fallback version
//         let mut power = base;
//         for i in 1..values.len() {
//             let term = values[i] / power;
//             sum += term;
//             deriv -= (i as f64) * term * inv_base;
//             power *= base;
//         }
//     }
//
//     (sum, deriv)
// }

// #[inline(always)]
// fn npv_with_deriv(rate: f64, values: &[f64]) -> (f64, f64) {
//     if rate <= -1.0 {
//         return (f64::INFINITY, f64::INFINITY);
//     }

//     let base = 1.0 + rate;
//     let inv_base = 1.0 / base;

//     // Process first value separately
//     let mut sum = values[0];
//     let mut deriv = 0.0;

//     if values.len() <= 1 {
//         return (sum, deriv);
//     }

//     // Use multiple accumulators to enable compiler auto-vectorization
//     let mut power = base;

//     // Process 4 elements at a time using independent accumulators
//     let chunks = (values.len() - 1) / 4;
//     let mut sum1 = 0.0;
//     let mut sum2 = 0.0;
//     let mut sum3 = 0.0;
//     let mut sum4 = 0.0;
//     let mut deriv1 = 0.0;
//     let mut deriv2 = 0.0;
//     let mut deriv3 = 0.0;
//     let mut deriv4 = 0.0;

//     for chunk in 0..chunks {
//         let i1 = 1 + chunk * 4;
//         let i2 = i1 + 1;
//         let i3 = i1 + 2;
//         let i4 = i1 + 3;

//         let p1 = power;
//         let p2 = p1 * base;
//         let p3 = p2 * base;
//         let p4 = p3 * base;

//         let term1 = values[i1] / p1;
//         let term2 = values[i2] / p2;
//         let term3 = values[i3] / p3;
//         let term4 = values[i4] / p4;

//         sum1 += term1;
//         sum2 += term2;
//         sum3 += term3;
//         sum4 += term4;

//         deriv1 -= (i1 as f64) * term1 * inv_base;
//         deriv2 -= (i2 as f64) * term2 * inv_base;
//         deriv3 -= (i3 as f64) * term3 * inv_base;
//         deriv4 -= (i4 as f64) * term4 * inv_base;

//         power = p4 * base;
//     }

//     // Combine accumulators
//     sum += sum1 + sum2 + sum3 + sum4;
//     deriv += deriv1 + deriv2 + deriv3 + deriv4;

//     // Process remaining elements
//     for i in (1 + chunks * 4)..values.len() {
//         let term = values[i] / power;
//         sum += term;
//         deriv -= (i as f64) * term * inv_base;
//         power *= base;
//     }

//     (sum, deriv)
// }

pub fn irr(values: &[f64], guess: Option<f64>) -> Result<f64, InvalidPaymentsError> {
    let values = utils::trim_zeros(values);
    validate(values, None)?;

    // Fast path for small datasets
    if values.len() == 2 {
        return Ok(irr_analytical_2(values));
    }

    if values.len() == 3 {
        return Ok(irr_analytical_3(values));
    }

    let initial_guess = guess.unwrap_or(0.1);

    // Try Brent with positive brackets
    let rate = brentq(&|r| npv(r, values, Some(true)), 0.0, 100.0, 100);
    if rate.is_finite() {
        return Ok(rate);
    }

    // Fallback to Newton-Raphson
    let rate = newton_raphson_2(initial_guess, &|r| npv_with_deriv(r, values));

    if utils::is_a_good_rate(rate, |r| npv(r, values, Some(true))) {
        return Ok(rate);
    }

    // Expended search for negative rates
    let rate = brentq(&|r| npv(r, values, Some(true)), -0.999, 100.0, 100);
    if rate.is_finite() {
        return Ok(rate);
    }

    // Final fallback with minimal iterations (to avoid the catastrophic slowdown)
    let breakpoints = &[-0.9, -0.5, 0.0, 0.5, 1.0];
    let f = |r| npv(r, values, Some(true));
    let rate = brentq_grid_search(&[breakpoints], &f).next();

    Ok(rate.unwrap_or(f64::NAN))
}

fn irr_analytical_2(values: &[f64]) -> f64 {
    // cf[0]/(1+r)^0 + cf[1]/(1+r)^1 = 0  => multiply by (1 + r)
    // cf[0]*(1+r) + cf[1] = 0  => divide by cf[0] and move tho the right
    // lets x = 1+r, a = cf[0], b = cf[1]
    // solve a*x + b = 0
    // x = -b/a, r = x - 1
    -values[1] / values[0] - 1.0
}

fn irr_analytical_3(values: &[f64]) -> f64 {
    // cf[0]/(1+r)^0 + cf[1]/(1+r)^1 + cf[2]/(1+r)^2 = 0  => multiply by (1+r)^2
    // cf[0]*(1+r)^2 + cf[1]*(1+r) + cf[2] = 0  => quadratic equation
    // lets x = 1+r, a = cf[0], b = cf[1], c = cf[2]
    // solve a*x^2 + b*x + c = 0
    // x = 1 + r => r = x - 1
    let (a, b, c) = (values[0], values[1], values[2]);

    if a == 0.0 {
        // 0*x^2 + bx + c = 0 =>
        // x = -c/b
        let x = -c / b;
        return x - 1.0;
    };

    // x = (-b ± sqrt(b^2-4ac))/2a, a != 0
    let d = b.powf(2.) - 4. * a * c; // discriminant

    match d.total_cmp(&0.0) {
        Ordering::Less => {
            // no solutions
            f64::NAN
        }
        Ordering::Equal => {
            // exactly one solution
            let x = -b / (2. * a);
            x - 1.0
        }
        Ordering::Greater => {
            // two solutions
            let x1 = (-b + d.sqrt()) / (2. * a);
            let x2 = (-b - d.sqrt()) / (2. * a);
            // x = 1 + r => r = x - 1
            let (r1, r2) = (x1 - 1.0, x2 - 1.0);

            // rate < -1 doesn't make sense
            match (r1.total_cmp(&-1.), r2.total_cmp(&-1.)) {
                (Ordering::Less, Ordering::Less) => f64::NAN,
                (Ordering::Equal | Ordering::Less, Ordering::Equal | Ordering::Less) => -1.0,
                (Ordering::Greater, Ordering::Less | Ordering::Equal) => r1,
                (Ordering::Less | Ordering::Equal, Ordering::Greater) => r2,
                (Ordering::Greater, Ordering::Greater) => {
                    // if both roots are non-negative,
                    // choose the one that best approximates npv to zero
                    let p1 = npv(r1, values, Some(true));
                    let p2 = npv(r2, values, Some(true));

                    if p1.abs() < p2.abs() {
                        r1
                    } else {
                        r2
                    }
                }
            }
        }
    }
}

pub fn mirr(
    values: &[f64],
    finance_rate: f64,
    reinvest_rate: f64,
) -> Result<f64, InvalidPaymentsError> {
    // must contain at least one positive and one negative value
    validate(values, None)?;

    let positive: f64 = powers(1. + reinvest_rate, values.len(), true)
        .zip(values.iter().rev())
        .filter(|(_r, &v)| v > 0.0)
        .map(|(r, v)| v * r)
        .sum();

    let negative: f64 = powers(1. + finance_rate, values.len(), true)
        .zip(values.iter())
        .filter(|(_r, &v)| v < 0.0)
        .map(|(r, &v)| v / r)
        .sum();

    Ok((positive / -negative).powf(1.0 / (values.len() - 1) as f64) - 1.0)
}

/// Calculates the cumulative principal payment between start_period and end_period
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value
/// * `start_period` - First period in the calculation
/// * `end_period` - Last period in the calculation
/// * `pmt_at_beginning` - When payments are made (beginning or end of period)
///
/// # Returns
///
/// * `Option<f64>` - The cumulative principal payment, or None if calculation fails
pub fn cumprinc(
    rate: f64,
    nper: f64,
    pv: f64,
    start_period: f64,
    end_period: f64,
    pmt_at_beginning: bool,
) -> f64 {
    // https://wiki.documentfoundation.org/Documentation/Calc_Functions/CUMPRINC
    (start_period.trunc() as u64..=end_period.trunc() as u64)
        .map(|per| ppmt(rate, per as f64, nper, pv, 0.0, pmt_at_beginning))
        .sum()
}

/// Calculates the cumulative interest payment between start_period and end_period
///
/// # Arguments
///
/// * `rate` - Interest rate per period
/// * `nper` - Total number of payment periods
/// * `pv` - Present value
/// * `start_period` - First period in the calculation
/// * `end_period` - Last period in the calculation
/// * `pmt_at_beginning` - When payments are made (beginning or end of period)
///
/// # Returns
///
/// * `f64` - The cumulative interest payment
pub fn cumipmt(
    rate: f64,
    nper: f64,
    pv: f64,
    start_period: f64,
    end_period: f64,
    pmt_at_beginning: bool,
) -> f64 {
    // https://wiki.documentfoundation.org/Documentation/Calc_Functions/CUMIPMT
    (start_period.trunc() as u64..=end_period.trunc() as u64)
        .map(|per| ipmt(rate, per as f64, nper, pv, 0.0, pmt_at_beginning))
        .sum()
}
