/// Trait to abstract SIMD operations for different architectures
trait SimdOps {
    /// Process a chunk of data for NPV calculation
    unsafe fn process_npv_chunk(
        base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
    ) -> (f64, f64); // returns (sum, new_power)

    /// Process a chunk for NPV with derivative calculation
    unsafe fn process_npv_deriv_chunk(
        base: f64,
        inv_base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
        start_index: usize,
    ) -> (f64, f64, f64); // returns (sum, deriv, new_power)

    /// Get chunk size for processing
    fn chunk_size() -> usize;

    /// Get precomputed power for updating between chunks
    unsafe fn precompute_power_factor(base: f64) -> f64;

    /// Check if this implementation is supported on current CPU
    fn is_supported() -> bool;
}

/// AVX implementation
#[cfg(target_arch = "x86_64")]
struct AvxOps;

#[cfg(target_arch = "x86_64")]
impl SimdOps for AvxOps {
    #[target_feature(enable = "avx")]
    unsafe fn process_npv_chunk(
        base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
    ) -> (f64, f64) {
        use std::arch::x86_64::*;

        // Prefetch next chunk for better memory performance
        if start_idx + 8 < values.len() {
            _mm_prefetch::<_MM_HINT_T0>(values.as_ptr().add(start_idx + 8) as *const i8);
        }

        let vec_base = _mm256_set1_pd(base);
        let vec_power_base = _mm256_set1_pd(power);

        // Create multipliers [1, base, base^2, base^3] more efficiently
        let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
        let vec_base_cubed = _mm256_mul_pd(vec_base_squared, vec_base);
        let vec_mult = _mm256_set_pd(
            _mm256_cvtsd_f64(vec_base_cubed),
            _mm256_cvtsd_f64(vec_base_squared),
            base,
            1.0,
        );

        // Multiply to get powers
        let vec_power = _mm256_mul_pd(vec_power_base, vec_mult);

        // Alignment-aware load
        let vec_values = if (values.as_ptr() as usize + start_idx * 8) % 32 == 0 {
            _mm256_load_pd(&values[start_idx])
        } else {
            _mm256_loadu_pd(&values[start_idx])
        };

        // Divide values by powers
        let vec_result = _mm256_div_pd(vec_values, vec_power);

        // // Use horizontal add to sum up the 4 packed doubles:
        // let hadd = _mm256_hadd_pd(vec_result, vec_result);
        // let low128 = _mm256_castpd256_pd128(hadd);
        // let high128 = _mm256_extractf128_pd(hadd, 1);
        // let sum128 = _mm_add_pd(low128, high128);
        // let sum = _mm_cvtsd_f64(sum128);

        // Sum the results
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), vec_result);
        let sum = result_array.iter().sum::<f64>();

        // Return sum and updated power
        (sum, power * Self::precompute_power_factor(base))
    }

    #[target_feature(enable = "avx")]
    unsafe fn process_npv_deriv_chunk(
        base: f64,
        inv_base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
        start_index: usize,
    ) -> (f64, f64, f64) {
        use std::arch::x86_64::*;

        // Prefetch next chunk
        if start_idx + 8 < values.len() {
            _mm_prefetch::<_MM_HINT_T0>(values.as_ptr().add(start_idx + 8) as *const i8);
        }

        let vec_base = _mm256_set1_pd(base);
        let vec_power_base = _mm256_set1_pd(power);
        let vec_inv_base = _mm256_set1_pd(inv_base);
        let vec_neg_one = _mm256_set1_pd(-1.0);

        // More efficient power calculation
        let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
        let vec_base_cubed = _mm256_mul_pd(vec_base_squared, vec_base);
        let vec_mult = _mm256_set_pd(
            _mm256_cvtsd_f64(vec_base_cubed),
            _mm256_cvtsd_f64(vec_base_squared),
            base,
            1.0,
        );

        // Multiply to get powers
        let vec_power = _mm256_mul_pd(vec_power_base, vec_mult);

        // Alignment-aware load
        let vec_values = if (values.as_ptr() as usize + start_idx * 8) % 32 == 0 {
            _mm256_load_pd(&values[start_idx])
        } else {
            _mm256_loadu_pd(&values[start_idx])
        };

        // More efficient index vector creation
        let offset = start_index + start_idx;
        let vec_indices = _mm256_set_pd(
            (offset + 3) as f64,
            (offset + 2) as f64,
            (offset + 1) as f64,
            offset as f64,
        );

        // Calculate terms
        let vec_term = _mm256_div_pd(vec_values, vec_power);

        // Combined calculation for derivative terms
        let vec_deriv = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_mul_pd(vec_indices, vec_term), vec_inv_base),
            vec_neg_one,
        );

        // Store results
        let mut term_array = [0.0; 4];
        let mut deriv_array = [0.0; 4];
        _mm256_storeu_pd(term_array.as_mut_ptr(), vec_term);
        _mm256_storeu_pd(deriv_array.as_mut_ptr(), vec_deriv);

        let sum = term_array.iter().sum::<f64>();
        let deriv = deriv_array.iter().sum::<f64>();

        (sum, deriv, power * Self::precompute_power_factor(base))
    }

    fn chunk_size() -> usize {
        4
    }

    #[target_feature(enable = "avx")]
    unsafe fn precompute_power_factor(base: f64) -> f64 {
        use std::arch::x86_64::*;
        let vec_base = _mm256_set1_pd(base);
        let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
        let vec_base4 = _mm256_mul_pd(vec_base_squared, vec_base_squared);
        _mm256_cvtsd_f64(vec_base4)
    }

    fn is_supported() -> bool {
        is_x86_feature_detected!("avx")
    }
}

/// AVX2+FMA implementation
#[cfg(target_arch = "x86_64")]
struct Avx2Ops;

#[cfg(target_arch = "x86_64")]
impl SimdOps for Avx2Ops {
    #[target_feature(enable = "avx2,fma")]
    unsafe fn process_npv_chunk(
        base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
    ) -> (f64, f64) {
        use std::arch::x86_64::*;

        // Prefetch next chunk for better memory performance
        if start_idx + 8 < values.len() {
            _mm_prefetch::<_MM_HINT_T0>(values.as_ptr().add(start_idx + 8) as *const i8);
        }

        let vec_base = _mm256_set1_pd(base);
        let vec_power_base = _mm256_set1_pd(power);

        // More efficient power calculation with FMA
        let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
        let vec_base_cubed = _mm256_fmadd_pd(vec_base, vec_base_squared, _mm256_setzero_pd());

        // Set up multipliers more efficiently
        let vec_mult = _mm256_set_pd(
            _mm_cvtsd_f64(_mm256_castpd256_pd128(vec_base_cubed)),
            _mm_cvtsd_f64(_mm256_castpd256_pd128(vec_base_squared)),
            base,
            1.0,
        );

        // Use FMA for power calculation
        let vec_power = _mm256_mul_pd(vec_power_base, vec_mult);

        // Load values - check if we can use aligned load
        let vec_values = if (values.as_ptr() as usize + start_idx * 8) % 32 == 0 {
            _mm256_load_pd(&values[start_idx])
        } else {
            _mm256_loadu_pd(&values[start_idx])
        };

        let vec_result = _mm256_div_pd(vec_values, vec_power);

        // Sum the results
        let mut result_array = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), vec_result);
        let sum = result_array.iter().sum::<f64>();

        (sum, power * Self::precompute_power_factor(base))
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn process_npv_deriv_chunk(
        base: f64,
        inv_base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
        start_index: usize,
    ) -> (f64, f64, f64) {
        use std::arch::x86_64::*;

        // Prefetch next chunk
        if start_idx + 8 < values.len() {
            _mm_prefetch::<_MM_HINT_T0>(values.as_ptr().add(start_idx + 8) as *const i8);
        }

        let vec_base = _mm256_set1_pd(base);
        let vec_power_base = _mm256_set1_pd(power);
        let vec_inv_base = _mm256_set1_pd(inv_base);

        // More efficient power calculation with FMA
        let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
        let vec_base_cubed = _mm256_fmadd_pd(vec_base, vec_base_squared, _mm256_setzero_pd());

        // Set up multipliers
        let vec_mult = _mm256_set_pd(
            _mm_cvtsd_f64(_mm256_castpd256_pd128(vec_base_cubed)),
            _mm_cvtsd_f64(_mm256_castpd256_pd128(vec_base_squared)),
            base,
            1.0,
        );

        // Use FMA where applicable
        let vec_power = _mm256_mul_pd(vec_power_base, vec_mult);

        // Optimized value loading
        let vec_values = if (values.as_ptr() as usize + start_idx * 8) % 32 == 0 {
            _mm256_load_pd(&values[start_idx])
        } else {
            _mm256_loadu_pd(&values[start_idx])
        };

        // Create index vector for derivative - using direct set rather than repeated conversions
        let offset = start_index + start_idx;
        let vec_indices = _mm256_set_pd(
            (offset + 3) as f64,
            (offset + 2) as f64,
            (offset + 1) as f64,
            offset as f64,
        );

        // Calculate NPV terms
        let vec_term = _mm256_div_pd(vec_values, vec_power);

        // Use FMA to optimize derivative calculation
        let vec_deriv = _mm256_mul_pd(vec_indices, vec_term);
        let vec_deriv = _mm256_fnmadd_pd(vec_deriv, vec_inv_base, _mm256_setzero_pd());

        // Store results
        let mut term_array = [0.0; 4];
        let mut deriv_array = [0.0; 4];
        _mm256_storeu_pd(term_array.as_mut_ptr(), vec_term);
        _mm256_storeu_pd(deriv_array.as_mut_ptr(), vec_deriv);

        let sum = term_array.iter().sum::<f64>();
        let deriv = deriv_array.iter().sum::<f64>();

        (sum, deriv, power * Self::precompute_power_factor(base))
    }

    fn chunk_size() -> usize {
        4
    }

    #[target_feature(enable = "avx2,fma")]
    unsafe fn precompute_power_factor(base: f64) -> f64 {
        use std::arch::x86_64::*;
        let vec_base = _mm256_set1_pd(base);
        let vec_base_squared = _mm256_mul_pd(vec_base, vec_base);
        let vec_base4 = _mm256_fmadd_pd(vec_base_squared, vec_base_squared, _mm256_setzero_pd());
        _mm_cvtsd_f64(_mm256_castpd256_pd128(vec_base4))
    }

    fn is_supported() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
}

/// Auto-vectorized implementation
struct AutoVecOps;

impl SimdOps for AutoVecOps {
    unsafe fn process_npv_chunk(
        base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
    ) -> (f64, f64) {
        // Using a pattern that's friendly for auto-vectorization
        const CHUNK: usize = 4;
        let mut accum = 0.0;
        let mut powers = [0.0; CHUNK];

        // Initialize powers for this chunk
        powers[0] = power;
        for i in 1..CHUNK {
            powers[i] = powers[i - 1] * base;
        }

        // Process values
        for i in 0..CHUNK {
            accum += values[start_idx + i] / powers[i];
        }

        // Return sum and updated power for next chunk
        (accum, powers[CHUNK - 1] * base)
    }

    unsafe fn process_npv_deriv_chunk(
        base: f64,
        inv_base: f64,
        values: &[f64],
        start_idx: usize,
        power: f64,
        start_index: usize,
    ) -> (f64, f64, f64) {
        const CHUNK: usize = 4;
        let mut sum_accum = 0.0;
        let mut deriv_accum = 0.0;
        let mut powers = [0.0; CHUNK];

        // Initialize powers for this chunk
        powers[0] = power;
        for i in 1..CHUNK {
            powers[i] = powers[i - 1] * base;
        }

        // Process values - compute sum and derivative
        for i in 0..CHUNK {
            let idx = start_idx + i;
            let term = values[idx] / powers[i];
            sum_accum += term;
            deriv_accum -= (start_index + idx) as f64 * term * inv_base;
        }

        // Return sum, derivative and updated power for next chunk
        (sum_accum, deriv_accum, powers[CHUNK - 1] * base)
    }

    fn chunk_size() -> usize {
        4
    }

    unsafe fn precompute_power_factor(base: f64) -> f64 {
        base * base * base * base // base^4
    }

    fn is_supported() -> bool {
        true
    }
}

/// Generic NPV calculation using SIMD operations
#[inline]
unsafe fn npv_simd_generic<S: SimdOps>(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
    let mut sum = 0.0;
    let mut power = if start_from_zero {
        1.0
    } else {
        base
    };

    let chunk_size = S::chunk_size();
    let simd_limit = (values.len() / chunk_size) * chunk_size;

    // Process chunks
    for i in (0..simd_limit).step_by(chunk_size) {
        let (chunk_sum, new_power) = S::process_npv_chunk(base, values, i, power);
        sum += chunk_sum;
        power = new_power;
    }

    // Process remaining elements
    for &value in &values[simd_limit..] {
        sum += value / power;
        power *= base;
    }

    sum
}

/// Generic NPV with derivative calculation
#[inline]
unsafe fn npv_with_deriv_generic<S: SimdOps>(
    rate: f64,
    values: &[f64],
    start_index: usize,
) -> (f64, f64) {
    let base = 1.0 + rate;
    let inv_base = 1.0 / base;

    let mut sum = 0.0;
    let mut deriv = 0.0;
    let mut power = base;

    let chunk_size = S::chunk_size();
    let simd_limit = (values.len() / chunk_size) * chunk_size;

    // Process chunks
    for chunk_idx in 0..(simd_limit / chunk_size) {
        let idx = chunk_idx * chunk_size;
        let (chunk_sum, chunk_deriv, new_power) =
            S::process_npv_deriv_chunk(base, inv_base, values, idx, power, start_index);

        sum += chunk_sum;
        deriv += chunk_deriv;
        power = new_power;
    }

    // Process remaining elements
    for (i, &value) in values[simd_limit..].iter().enumerate() {
        let term = value / power;
        sum += term;
        let index = start_index + simd_limit + i;
        deriv -= (index as f64) * term * inv_base;
        power *= base;
    }

    (sum, deriv)
}

/// Lanes consumed per iteration of the 256-bit kernels: two 4-wide vectors, so the discount
/// factor and accumulator chains are independent and the loop is not latency-bound.
#[cfg(target_arch = "x86_64")]
const SIMD256_BLOCK: usize = 8;

/// Shortest slice worth entering a 256-bit kernel for, rather than the auto-vectorized
/// fallback. Measured, not assumed — see `benches/npv_kernel.rs`.
#[cfg(target_arch = "x86_64")]
const SIMD256_MIN_LEN: usize = 6;

/// Horizontal sum of a 4-wide vector. Done once at the end of a kernel, never per chunk.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn hsum_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let lo = _mm256_castpd256_pd128(v);
    let hi = _mm256_extractf128_pd(v, 1);
    let s = _mm_add_pd(lo, hi);
    _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)))
}

/// `acc + a * b` for the AVX2+FMA tier: a single fused instruction.
macro_rules! madd_fma {
    ($a:expr, $b:expr, $acc:expr) => {
        _mm256_fmadd_pd($a, $b, $acc)
    };
}

/// `acc + a * b` for the plain AVX tier, which has no FMA. Rounds twice rather than
/// once, so results can differ from the AVX2 tier in the last ulp — as they already did.
macro_rules! madd_mul_add {
    ($a:expr, $b:expr, $acc:expr) => {
        _mm256_add_pd(_mm256_mul_pd($a, $b), $acc)
    };
}

/// Emits the 256-bit NPV kernels for one x86 feature tier.
///
/// AVX and AVX2+FMA differ only in how a multiply-accumulate is spelled, so both tiers
/// are generated from this one definition instead of two copies that can drift apart.
/// Every other intrinsic used here is AVX-level.
macro_rules! define_simd256_kernels {
    ($npv:ident, $npv_deriv:ident, $feature:literal, $madd:ident) => {
        /// NPV over the whole slice in one pass.
        ///
        /// Replaces the per-chunk `SimdOps` path, which recomputed `[1, b, b², b³]` and `b⁴` for
        /// every four elements, divided by the discount factor, and round-tripped the result
        /// through the stack to sum it. Here the powers of `1/base` are carried in registers, the
        /// division becomes a multiply, and there is exactly one horizontal reduction at the end.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feature)]
        unsafe fn $npv(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
            use std::arch::x86_64::*;

            let inv_base = 1.0 / base;
            let ib2 = inv_base * inv_base;
            let ib3 = ib2 * inv_base;
            let ib4 = ib2 * ib2;

            // discount factor of the first element: base^0 or base^-1
            let p0 = if start_from_zero {
                1.0
            } else {
                inv_base
            };

            let mut pow_lo = _mm256_set_pd(p0 * ib3, p0 * ib2, p0 * inv_base, p0);
            let mut pow_hi = _mm256_mul_pd(pow_lo, _mm256_set1_pd(ib4));
            let step = _mm256_set1_pd(ib4 * ib4);

            let mut acc_lo = _mm256_setzero_pd();
            let mut acc_hi = _mm256_setzero_pd();

            let ptr = values.as_ptr();
            let blocks = values.len() / SIMD256_BLOCK;

            for k in 0..blocks {
                let i = k * SIMD256_BLOCK;
                acc_lo = $madd!(_mm256_loadu_pd(ptr.add(i)), pow_lo, acc_lo);
                acc_hi = $madd!(_mm256_loadu_pd(ptr.add(i + 4)), pow_hi, acc_hi);
                pow_lo = _mm256_mul_pd(pow_lo, step);
                pow_hi = _mm256_mul_pd(pow_hi, step);
            }

            // A remainder of 4 or more is still worth a vector step; only the last 0-3 go scalar.
            let mut i = blocks * SIMD256_BLOCK;
            if values.len() - i >= 4 {
                acc_lo = $madd!(_mm256_loadu_pd(ptr.add(i)), pow_lo, acc_lo);
                pow_lo = _mm256_mul_pd(pow_lo, _mm256_set1_pd(ib4));
                i += 4;
            }

            let mut sum = hsum_pd(_mm256_add_pd(acc_lo, acc_hi));

            // lane 0 of pow_lo is the discount factor of the next unprocessed element
            let mut power = _mm256_cvtsd_f64(pow_lo);
            while i < values.len() {
                sum += *ptr.add(i) * power;
                power *= inv_base;
                i += 1;
            }

            sum
        }

        /// NPV and its derivative over the whole slice in one pass.
        ///
        /// Same treatment as [`npv_simd_avx2`], plus: the index vector is carried and incremented
        /// rather than rebuilt from four `usize -> f64` conversions per chunk, and the `-1/base`
        /// factor common to every derivative term is applied once at the end instead of per element.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = $feature)]
        unsafe fn $npv_deriv(rate: f64, values: &[f64], start_index: usize) -> (f64, f64) {
            use std::arch::x86_64::*;

            let base = 1.0 + rate;
            let inv_base = 1.0 / base;
            let ib2 = inv_base * inv_base;
            let ib3 = ib2 * inv_base;
            let ib4 = ib2 * ib2;

            // Matches `npv_with_deriv_generic`: the discount exponent starts at 1 whatever
            // `start_index` is — the caller has already handled element 0 — while `start_index`
            // only feeds the derivative's index weights. The two coincide for the production
            // call, which passes `&values[1..]` with `start_index == 1`.
            let p0 = inv_base;

            let mut pow_lo = _mm256_set_pd(p0 * ib3, p0 * ib2, p0 * inv_base, p0);
            let mut pow_hi = _mm256_mul_pd(pow_lo, _mm256_set1_pd(ib4));
            let step = _mm256_set1_pd(ib4 * ib4);

            let si = start_index as f64;
            let mut idx_lo = _mm256_set_pd(si + 3.0, si + 2.0, si + 1.0, si);
            let mut idx_hi = _mm256_add_pd(idx_lo, _mm256_set1_pd(4.0));
            let idx_step = _mm256_set1_pd(SIMD256_BLOCK as f64);

            let mut sum_lo = _mm256_setzero_pd();
            let mut sum_hi = _mm256_setzero_pd();
            // accumulates sum(i * v_i / base^i); scaled by -1/base once the loop is done
            let mut deriv_lo = _mm256_setzero_pd();
            let mut deriv_hi = _mm256_setzero_pd();

            let ptr = values.as_ptr();
            let blocks = values.len() / SIMD256_BLOCK;

            for k in 0..blocks {
                let i = k * SIMD256_BLOCK;

                let term_lo = _mm256_mul_pd(_mm256_loadu_pd(ptr.add(i)), pow_lo);
                let term_hi = _mm256_mul_pd(_mm256_loadu_pd(ptr.add(i + 4)), pow_hi);

                sum_lo = _mm256_add_pd(sum_lo, term_lo);
                sum_hi = _mm256_add_pd(sum_hi, term_hi);

                deriv_lo = $madd!(term_lo, idx_lo, deriv_lo);
                deriv_hi = $madd!(term_hi, idx_hi, deriv_hi);

                pow_lo = _mm256_mul_pd(pow_lo, step);
                pow_hi = _mm256_mul_pd(pow_hi, step);
                idx_lo = _mm256_add_pd(idx_lo, idx_step);
                idx_hi = _mm256_add_pd(idx_hi, idx_step);
            }

            let mut i = blocks * SIMD256_BLOCK;
            if values.len() - i >= 4 {
                let term = _mm256_mul_pd(_mm256_loadu_pd(ptr.add(i)), pow_lo);
                sum_lo = _mm256_add_pd(sum_lo, term);
                deriv_lo = $madd!(term, idx_lo, deriv_lo);
                pow_lo = _mm256_mul_pd(pow_lo, _mm256_set1_pd(ib4));
                i += 4;
            }

            let mut sum = hsum_pd(_mm256_add_pd(sum_lo, sum_hi));
            let mut weighted = hsum_pd(_mm256_add_pd(deriv_lo, deriv_hi));

            let mut power = _mm256_cvtsd_f64(pow_lo);
            while i < values.len() {
                let term = *ptr.add(i) * power;
                sum += term;
                weighted += (start_index + i) as f64 * term;
                power *= inv_base;
                i += 1;
            }

            (sum, -weighted * inv_base)
        }
    };
}

define_simd256_kernels!(npv_simd_avx2, npv_with_deriv_avx2, "avx2,fma", madd_fma);
define_simd256_kernels!(npv_simd_avx, npv_with_deriv_avx, "avx", madd_mul_add);

/// Lanes consumed per iteration of the NEON kernels. A NEON vector holds two doubles, so
/// four of them are unrolled per iteration: that gives the same four independent
/// accumulator chains as the 256-bit kernels, which is what keeps the loop throughput-bound
/// rather than stalled on the ~4-cycle FMA latency.
#[cfg(target_arch = "aarch64")]
const NEON_BLOCK: usize = 8;

/// Shortest slice worth entering [`npv_simd_neon`] for, rather than the auto-vectorized
/// fallback. Measured on Apple M2, not assumed — see `benches/npv_kernel.rs`.
#[cfg(target_arch = "aarch64")]
const NEON_MIN_LEN: usize = 3;

/// The same threshold for [`npv_with_deriv_neon`], which is higher: that kernel sets up
/// eight vectors (four discount-factor chains, four index chains) before its first
/// multiply, and only a whole 8-element block pays that back. Below this, `irr` measured
/// slower with the kernel than without it. The kernel is handed `&values[1..]`, so a slice
/// of this length gives it exactly one block.
#[cfg(target_arch = "aarch64")]
const NEON_DERIV_MIN_LEN: usize = NEON_BLOCK + 1;

/// NPV over the whole slice in one pass.
///
/// Replaces the per-chunk `SimdOps` path, which rebuilt `[1, base]` and `base²` for every
/// two elements, divided by the discount factor, and round-tripped the pair through the
/// stack to sum it — two lanes of work per call, with a serial dependency between calls.
/// Here the powers of `1/base` are carried in registers across four independent chains, the
/// division becomes a multiply-accumulate, and there is exactly one horizontal reduction at
/// the end.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn npv_simd_neon(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
    use std::arch::aarch64::*;

    let inv_base = 1.0 / base;
    let ib2 = inv_base * inv_base;
    let ib4 = ib2 * ib2;
    let ib8 = ib4 * ib4;

    // discount factor of the first element: base^0 or base^-1
    let p0 = if start_from_zero {
        1.0
    } else {
        inv_base
    };

    // pow_j carries the factors of elements i + 2j and i + 2j + 1
    let mut pow0 = vcombine_f64(vdup_n_f64(p0), vdup_n_f64(p0 * inv_base));
    let mut pow1 = vmulq_n_f64(pow0, ib2);
    let mut pow2 = vmulq_n_f64(pow0, ib4);
    let mut pow3 = vmulq_n_f64(pow1, ib4);
    let step = vdupq_n_f64(ib8);

    let mut acc0 = vdupq_n_f64(0.0);
    let mut acc1 = vdupq_n_f64(0.0);
    let mut acc2 = vdupq_n_f64(0.0);
    let mut acc3 = vdupq_n_f64(0.0);

    let ptr = values.as_ptr();
    let len = values.len();
    let blocks = len / NEON_BLOCK;

    for k in 0..blocks {
        let i = k * NEON_BLOCK;
        acc0 = vfmaq_f64(acc0, vld1q_f64(ptr.add(i)), pow0);
        acc1 = vfmaq_f64(acc1, vld1q_f64(ptr.add(i + 2)), pow1);
        acc2 = vfmaq_f64(acc2, vld1q_f64(ptr.add(i + 4)), pow2);
        acc3 = vfmaq_f64(acc3, vld1q_f64(ptr.add(i + 6)), pow3);
        pow0 = vmulq_f64(pow0, step);
        pow1 = vmulq_f64(pow1, step);
        pow2 = vmulq_f64(pow2, step);
        pow3 = vmulq_f64(pow3, step);
    }

    // Up to three whole pairs are left over; each already has its factors in pow_j, so the
    // remainder costs no extra power arithmetic. `tail_pow` tracks the vector whose lane 0
    // is the factor of the next unprocessed element.
    let mut i = blocks * NEON_BLOCK;
    let mut tail_pow = pow0;
    if len - i >= 2 {
        acc0 = vfmaq_f64(acc0, vld1q_f64(ptr.add(i)), pow0);
        i += 2;
        tail_pow = pow1;
    }
    if len - i >= 2 {
        acc1 = vfmaq_f64(acc1, vld1q_f64(ptr.add(i)), pow1);
        i += 2;
        tail_pow = pow2;
    }
    if len - i >= 2 {
        acc2 = vfmaq_f64(acc2, vld1q_f64(ptr.add(i)), pow2);
        i += 2;
        tail_pow = pow3;
    }

    let acc = vaddq_f64(vaddq_f64(acc0, acc1), vaddq_f64(acc2, acc3));
    let mut sum = vaddvq_f64(acc);

    // at most one element left
    if i < len {
        sum += *ptr.add(i) * vgetq_lane_f64::<0>(tail_pow);
    }

    sum
}

fn npv_autovec(base: f64, values: &[f64], start_from_zero: bool) -> f64 {
    unsafe { npv_simd_generic::<AutoVecOps>(base, values, start_from_zero) }
}

/// NPV and its derivative over the whole slice in one pass.
///
/// Same treatment as [`npv_simd_neon`], plus: the index vector is carried and incremented
/// rather than rebuilt from two `usize -> f64` conversions per chunk, and the `-1/base`
/// factor common to every derivative term is applied once at the end instead of per element.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn npv_with_deriv_neon(rate: f64, values: &[f64], start_index: usize) -> (f64, f64) {
    use std::arch::aarch64::*;

    let base = 1.0 + rate;
    let inv_base = 1.0 / base;
    let ib2 = inv_base * inv_base;
    let ib4 = ib2 * ib2;
    let ib8 = ib4 * ib4;

    // Matches `npv_with_deriv_generic`: the discount exponent starts at 1 whatever
    // `start_index` is — the caller has already handled element 0 — while `start_index`
    // only feeds the derivative's index weights. The two coincide for the production
    // call, which passes `&values[1..]` with `start_index == 1`.
    let p0 = inv_base;

    let mut pow0 = vcombine_f64(vdup_n_f64(p0), vdup_n_f64(p0 * inv_base));
    let mut pow1 = vmulq_n_f64(pow0, ib2);
    let mut pow2 = vmulq_n_f64(pow0, ib4);
    let mut pow3 = vmulq_n_f64(pow1, ib4);
    let step = vdupq_n_f64(ib8);

    let si = start_index as f64;
    let mut idx0 = vcombine_f64(vdup_n_f64(si), vdup_n_f64(si + 1.0));
    let mut idx1 = vaddq_f64(idx0, vdupq_n_f64(2.0));
    let mut idx2 = vaddq_f64(idx0, vdupq_n_f64(4.0));
    let mut idx3 = vaddq_f64(idx0, vdupq_n_f64(6.0));
    let idx_step = vdupq_n_f64(NEON_BLOCK as f64);

    let mut sum0 = vdupq_n_f64(0.0);
    let mut sum1 = vdupq_n_f64(0.0);
    let mut sum2 = vdupq_n_f64(0.0);
    let mut sum3 = vdupq_n_f64(0.0);
    // accumulate sum(i * v_i / base^i); scaled by -1/base once the loop is done
    let mut wgt0 = vdupq_n_f64(0.0);
    let mut wgt1 = vdupq_n_f64(0.0);
    let mut wgt2 = vdupq_n_f64(0.0);
    let mut wgt3 = vdupq_n_f64(0.0);

    let ptr = values.as_ptr();
    let len = values.len();
    let blocks = len / NEON_BLOCK;

    for k in 0..blocks {
        let i = k * NEON_BLOCK;

        let term0 = vmulq_f64(vld1q_f64(ptr.add(i)), pow0);
        let term1 = vmulq_f64(vld1q_f64(ptr.add(i + 2)), pow1);
        let term2 = vmulq_f64(vld1q_f64(ptr.add(i + 4)), pow2);
        let term3 = vmulq_f64(vld1q_f64(ptr.add(i + 6)), pow3);

        sum0 = vaddq_f64(sum0, term0);
        sum1 = vaddq_f64(sum1, term1);
        sum2 = vaddq_f64(sum2, term2);
        sum3 = vaddq_f64(sum3, term3);

        wgt0 = vfmaq_f64(wgt0, term0, idx0);
        wgt1 = vfmaq_f64(wgt1, term1, idx1);
        wgt2 = vfmaq_f64(wgt2, term2, idx2);
        wgt3 = vfmaq_f64(wgt3, term3, idx3);

        pow0 = vmulq_f64(pow0, step);
        pow1 = vmulq_f64(pow1, step);
        pow2 = vmulq_f64(pow2, step);
        pow3 = vmulq_f64(pow3, step);

        idx0 = vaddq_f64(idx0, idx_step);
        idx1 = vaddq_f64(idx1, idx_step);
        idx2 = vaddq_f64(idx2, idx_step);
        idx3 = vaddq_f64(idx3, idx_step);
    }

    let mut i = blocks * NEON_BLOCK;
    let mut tail_pow = pow0;
    if len - i >= 2 {
        let term = vmulq_f64(vld1q_f64(ptr.add(i)), pow0);
        sum0 = vaddq_f64(sum0, term);
        wgt0 = vfmaq_f64(wgt0, term, idx0);
        i += 2;
        tail_pow = pow1;
    }
    if len - i >= 2 {
        let term = vmulq_f64(vld1q_f64(ptr.add(i)), pow1);
        sum1 = vaddq_f64(sum1, term);
        wgt1 = vfmaq_f64(wgt1, term, idx1);
        i += 2;
        tail_pow = pow2;
    }
    if len - i >= 2 {
        let term = vmulq_f64(vld1q_f64(ptr.add(i)), pow2);
        sum2 = vaddq_f64(sum2, term);
        wgt2 = vfmaq_f64(wgt2, term, idx2);
        i += 2;
        tail_pow = pow3;
    }

    let mut sum = vaddvq_f64(vaddq_f64(vaddq_f64(sum0, sum1), vaddq_f64(sum2, sum3)));
    let mut weighted = vaddvq_f64(vaddq_f64(vaddq_f64(wgt0, wgt1), vaddq_f64(wgt2, wgt3)));

    // at most one element left
    if i < len {
        let term = *ptr.add(i) * vgetq_lane_f64::<0>(tail_pow);
        sum += term;
        weighted += (start_index + i) as f64 * term;
    }

    (sum, -weighted * inv_base)
}

fn npv_with_deriv_autovec(rate: f64, values: &[f64], start_index: usize) -> (f64, f64) {
    unsafe { npv_with_deriv_generic::<AutoVecOps>(rate, values, start_index) }
}

/// SIMD tier selected once per process from CPU support + the `ENABLE_*` overrides.
///
/// Neither the CPUID probe nor the environment changes during a run, so resolving this
/// per call meant taking the process-wide environment `RwLock` inside the brentq inner
/// loop. The override vars are read exactly once (at the first NPV call) and are frozen
/// for the rest of the process.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum SimdTier {
    Avx2,
    Avx,
    None,
}

#[cfg(target_arch = "x86_64")]
fn simd_tier() -> SimdTier {
    static TIER: std::sync::LazyLock<SimdTier> = std::sync::LazyLock::new(|| {
        if std::env::var("ENABLE_AVX2").map(|x| x == "1").unwrap_or(true) && Avx2Ops::is_supported()
        {
            SimdTier::Avx2
        } else if std::env::var("ENABLE_AVX").map(|x| x == "1").unwrap_or(true)
            && AvxOps::is_supported()
        {
            SimdTier::Avx
        } else {
            SimdTier::None
        }
    });
    *TIER
}

/// NEON enablement, cached once. There is no CPU probe to pair it with: NEON is mandatory
/// on aarch64, so the env var is the only thing that can turn these kernels off — which is
/// what `benches/npv_kernel.rs` uses to time the fallback.
#[cfg(target_arch = "aarch64")]
fn neon_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("ENABLE_NEON").map(|x| x == "1").unwrap_or(true));
    *ENABLED
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
        if values.len() >= SIMD256_MIN_LEN {
            match simd_tier() {
                SimdTier::Avx2 => return unsafe { npv_simd_avx2(base, values, start_from_zero) },
                SimdTier::Avx => return unsafe { npv_simd_avx(base, values, start_from_zero) },
                SimdTier::None => {}
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() && values.len() >= NEON_MIN_LEN {
            return unsafe { npv_simd_neon(base, values, start_from_zero) };
        }
    }

    // Fallback optimized for auto-vectorization
    npv_autovec(base, values, start_from_zero)
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
        if values.len() >= SIMD256_MIN_LEN {
            match simd_tier() {
                SimdTier::Avx2 => {
                    let (simd_sum, simd_deriv) =
                        unsafe { npv_with_deriv_avx2(rate, &values[1..], 1) };
                    return (sum + simd_sum, deriv + simd_deriv);
                }
                SimdTier::Avx => {
                    let (simd_sum, simd_deriv) =
                        unsafe { npv_with_deriv_avx(rate, &values[1..], 1) };
                    return (sum + simd_sum, deriv + simd_deriv);
                }
                SimdTier::None => {}
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() && values.len() >= NEON_DERIV_MIN_LEN {
            let (simd_sum, simd_deriv) = unsafe { npv_with_deriv_neon(rate, &values[1..], 1) };
            return (sum + simd_sum, deriv + simd_deriv);
        }
    }

    // Fall back to auto-vectorized version
    let (auto_sum, auto_deriv) = npv_with_deriv_autovec(rate, &values[1..], 1);
    (sum + auto_sum, deriv + auto_deriv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_npv_implementation<S: SimdOps>(values: &[f64], rate: f64) -> f64 {
        unsafe {
            let base = 1.0 + rate;
            npv_simd_generic::<S>(base, values, true)
        }
    }

    fn test_npv_deriv_implementation<S: SimdOps>(values: &[f64], rate: f64) -> (f64, f64) {
        unsafe { npv_with_deriv_generic::<S>(rate, values, 0) }
    }

    fn get_reference_npv(values: &[f64], rate: f64) -> f64 {
        let base = 1.0 + rate;
        let mut sum = 0.0;
        let mut power = 1.0;
        for &val in values {
            sum += val / power;
            power *= base;
        }
        sum
    }

    fn get_reference_npv_deriv(values: &[f64], rate: f64, start_index: usize) -> (f64, f64) {
        let base = 1.0 + rate;
        let inv_base = 1.0 / base;
        let mut sum = 0.0;
        let mut deriv = 0.0;
        let mut power = base;

        for (i, &val) in values.iter().enumerate() {
            let term = val / power;
            sum += term;
            // Use the provided start_index
            deriv -= ((i + start_index) as f64) * term * inv_base;
            power *= base;
        }

        (sum, deriv)
    }

    #[test]
    fn test_simd_implementations_boundary_cases() {
        let test_cases = vec![
            vec![],                                                      // empty
            vec![100.0],                                                 // single value
            vec![100.0, -30.0],                                          // less than chunk size
            vec![100.0, -30.0, 20.0],                                    // less than chunk size
            vec![100.0, -30.0, 20.0, 15.0],                              // exact chunk size (4)
            vec![100.0, -30.0, 20.0, 15.0, 5.0],                         // chunk size + 1
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0],       // exactly 8
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0, 12.0], // > 8 (SIMD threshold)
        ];

        let rate = 0.05;

        for values in &test_cases {
            let ref_npv = get_reference_npv(values, rate);
            let (ref_sum, ref_deriv) = get_reference_npv_deriv(values, rate, 0);

            // Test AutoVecOps (always available)
            {
                let auto_npv = test_npv_implementation::<AutoVecOps>(values, rate);
                assert!(
                    (auto_npv - ref_npv).abs() < 1e-6,
                    "AutoVecOps NPV failed for len {}: got {}, expected {}",
                    values.len(),
                    auto_npv,
                    ref_npv
                );

                let (auto_sum, auto_deriv) =
                    test_npv_deriv_implementation::<AutoVecOps>(values, rate);
                assert!(
                    (auto_sum - ref_sum).abs() < 1e-6 && (auto_deriv - ref_deriv).abs() < 1e-6,
                    "AutoVecOps NPV+deriv failed for len {}",
                    values.len()
                );
            }

            // Test AVX implementation if supported
            #[cfg(target_arch = "x86_64")]
            if AvxOps::is_supported() {
                let avx_npv = test_npv_implementation::<AvxOps>(values, rate);
                assert!(
                    (avx_npv - ref_npv).abs() < 1e-6,
                    "AVX NPV failed for len {}: got {}, expected {}",
                    values.len(),
                    avx_npv,
                    ref_npv
                );

                let (avx_sum, avx_deriv) = test_npv_deriv_implementation::<AvxOps>(values, rate);
                assert!(
                    (avx_sum - ref_sum).abs() < 1e-6 && (avx_deriv - ref_deriv).abs() < 1e-6,
                    "AVX NPV+deriv failed for len {}",
                    values.len()
                );
            }

            // Test AVX2+FMA implementation if supported
            #[cfg(target_arch = "x86_64")]
            if Avx2Ops::is_supported() {
                let avx2_npv = test_npv_implementation::<Avx2Ops>(values, rate);
                assert!(
                    (avx2_npv - ref_npv).abs() < 1e-6,
                    "AVX2 NPV failed for len {}: got {}, expected {}",
                    values.len(),
                    avx2_npv,
                    ref_npv
                );

                let (avx2_sum, avx2_deriv) = test_npv_deriv_implementation::<Avx2Ops>(values, rate);
                assert!(
                    (avx2_sum - ref_sum).abs() < 1e-6 && (avx2_deriv - ref_deriv).abs() < 1e-6,
                    "AVX2 NPV+deriv failed for len {}",
                    values.len()
                );
            }

            // Test NEON implementation on ARM
            #[cfg(target_arch = "aarch64")]
            {
                let neon_npv = unsafe { npv_simd_neon(1.0 + rate, values, true) };
                assert!(
                    (neon_npv - ref_npv).abs() < 1e-6,
                    "NEON NPV failed for len {}: got {}, expected {}",
                    values.len(),
                    neon_npv,
                    ref_npv
                );

                let (neon_sum, neon_deriv) = unsafe { npv_with_deriv_neon(rate, values, 0) };
                assert!(
                    (neon_sum - ref_sum).abs() < 1e-6 && (neon_deriv - ref_deriv).abs() < 1e-6,
                    "NEON NPV+deriv failed for len {}",
                    values.len()
                );
            }
        }
    }

    #[test]
    fn test_simd_implementations_non_zero_start() {
        let test_cases = vec![
            vec![],                                                      // empty
            vec![100.0],                                                 // single value
            vec![100.0, -30.0],                                          // less than chunk size
            vec![100.0, -30.0, 20.0],                                    // less than chunk size
            vec![100.0, -30.0, 20.0, 15.0],                              // exact chunk size (4)
            vec![100.0, -30.0, 20.0, 15.0, 5.0],                         // chunk size + 1
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0],       // exactly 8
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0, 12.0], // > 8 (SIMD threshold)
        ];

        let rate = 0.05;

        for values in &test_cases {
            // Test with start_from_zero = false for each implementation
            let base = 1.0 + rate;
            let mut sum = 0.0;
            let mut power = base; // Start from base instead of 1.0
            for &val in values {
                sum += val / power;
                power *= base;
            }
            let ref_npv = sum;

            // Test AutoVecOps
            let auto_npv = unsafe { npv_simd_generic::<AutoVecOps>(base, values, false) };
            assert!(
                (auto_npv - ref_npv).abs() < 1e-6,
                "AutoVecOps NPV(non-zero start) failed for len {}: got {}, expected {}",
                values.len(),
                auto_npv,
                ref_npv
            );

            // Test other implementations...
            #[cfg(target_arch = "x86_64")]
            if AvxOps::is_supported() {
                let avx_npv = unsafe { npv_simd_generic::<AvxOps>(base, values, false) };
                assert!(
                    (avx_npv - ref_npv).abs() < 1e-6,
                    "AVX NPV(non-zero start) failed for len {}: got {}, expected {}",
                    values.len(),
                    avx_npv,
                    ref_npv
                );
            }

            #[cfg(target_arch = "x86_64")]
            if Avx2Ops::is_supported() {
                let avx2_npv = unsafe { npv_simd_generic::<Avx2Ops>(base, values, false) };
                assert!(
                    (avx2_npv - ref_npv).abs() < 1e-6,
                    "AVX2 NPV(non-zero start) failed for len {}: got {}, expected {}",
                    values.len(),
                    avx2_npv,
                    ref_npv
                );
            }

            #[cfg(target_arch = "aarch64")]
            {
                let neon_npv = unsafe { npv_simd_neon(base, values, false) };
                assert!(
                    (neon_npv - ref_npv).abs() < 1e-6,
                    "NEON NPV(non-zero start) failed for len {}: got {}, expected {}",
                    values.len(),
                    neon_npv,
                    ref_npv
                );
            }
        }
    }

    #[test]
    fn test_simd_implementations_with_start_index() {
        let test_cases = vec![
            vec![],                                                      // empty
            vec![100.0],                                                 // single value
            vec![100.0, -30.0],                                          // less than chunk size
            vec![100.0, -30.0, 20.0],                                    // less than chunk size
            vec![100.0, -30.0, 20.0, 15.0],                              // exact chunk size (4)
            vec![100.0, -30.0, 20.0, 15.0, 5.0],                         // chunk size + 1
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0],       // exactly 8
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0, 12.0], // > 8 (SIMD threshold)
        ];

        let rate = 0.05;
        let start_index = 1;

        for values in &test_cases {
            let (ref_sum, ref_deriv) = get_reference_npv_deriv(values, rate, start_index);

            // Test all implementations with start_index = 1
            let (auto_sum, auto_deriv) =
                unsafe { npv_with_deriv_generic::<AutoVecOps>(rate, values, start_index) };
            assert!(
                (auto_sum - ref_sum).abs() < 1e-6 && (auto_deriv - ref_deriv).abs() < 1e-6,
                "AutoVecOps NPV+deriv with start_index failed for len {}",
                values.len()
            );

            // Test other implementations...
            #[cfg(target_arch = "x86_64")]
            if AvxOps::is_supported() {
                let (avx_sum, avx_deriv) =
                    unsafe { npv_with_deriv_generic::<AvxOps>(rate, values, start_index) };
                assert!(
                    (avx_sum - ref_sum).abs() < 1e-6 && (avx_deriv - ref_deriv).abs() < 1e-6,
                    "AVX NPV+deriv with start_index failed for len {}",
                    values.len()
                );
            }

            #[cfg(target_arch = "x86_64")]
            if Avx2Ops::is_supported() {
                let (avx2_sum, avx2_deriv) =
                    unsafe { npv_with_deriv_generic::<Avx2Ops>(rate, values, start_index) };
                assert!(
                    (avx2_sum - ref_sum).abs() < 1e-6 && (avx2_deriv - ref_deriv).abs() < 1e-6,
                    "AVX2 NPV+deriv with start_index failed for len {}",
                    values.len(),
                );
            }

            #[cfg(target_arch = "aarch64")]
            {
                let (neon_sum, neon_deriv) =
                    unsafe { npv_with_deriv_neon(rate, values, start_index) };
                assert!(
                    (neon_sum - ref_sum).abs() < 1e-6 && (neon_deriv - ref_deriv).abs() < 1e-6,
                    "NEON NPV+deriv with start_index failed for len {}",
                    values.len(),
                );
            }
        }
    }

    #[test]
    fn test_public_npv_functions() {
        let test_cases = vec![
            vec![],                                                      // empty
            vec![100.0],                                                 // single value
            vec![100.0, -30.0],                                          // less than chunk size
            vec![100.0, -30.0, 20.0],                                    // less than chunk size
            vec![100.0, -30.0, 20.0, 15.0],                              // exact chunk size (4)
            vec![100.0, -30.0, 20.0, 15.0, 5.0],                         // chunk size + 1
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0],       // exactly 8
            vec![100.0, -30.0, 20.0, 15.0, 5.0, -10.0, 25.0, 8.0, 12.0], // > 8 (SIMD threshold)
        ];

        let rate = 0.05;

        for values in &test_cases {
            // Test public NPV function
            let ref_npv = get_reference_npv(values, rate);
            let npv = npv_simd(rate, values, Some(true));
            assert!(
                (npv - ref_npv).abs() < 1e-6,
                "npv_simd failed for len {}: got {}, expected {}",
                values.len(),
                npv,
                ref_npv
            );

            // Test public NPV with derivative function
            if !values.is_empty() {
                let (sum, deriv) = npv_with_deriv_simd(rate, values);
                let first = values[0];
                let (ref_sum, ref_deriv) = if values.len() > 1 {
                    let (s, d) = get_reference_npv_deriv(&values[1..], rate, 1);
                    (first + s, d)
                } else {
                    (first, 0.0)
                };

                assert!(
                    (sum - ref_sum).abs() < 1e-6 && (deriv - ref_deriv).abs() < 1e-6,
                    "npv_with_deriv_simd failed for len {}: got ({}, {}), expected ({}, {})",
                    values.len(),
                    sum,
                    deriv,
                    ref_sum,
                    ref_deriv
                );
            }
        }
    }

    #[test]
    fn test_npv_with_deriv_simd_empty_array() {
        let empty: Vec<f64> = vec![];
        let rate = 0.05;

        // Test with empty array
        let (sum, deriv) = npv_with_deriv_simd(rate, &empty);
        assert_eq!(sum, 0.0, "NPV of empty array should be 0.0");
        assert_eq!(deriv, 0.0, "Derivative of empty array should be 0.0");

        // Test with rate <= -1.0 and empty array
        let (sum, deriv) = npv_with_deriv_simd(-1.0, &empty);
        assert!(
            sum.is_infinite() && sum.is_sign_positive(),
            "NPV should be positive infinity for rate <= -1.0"
        );
        assert!(
            deriv.is_infinite() && deriv.is_sign_positive(),
            "Derivative should be positive infinity for rate <= -1.0"
        );

        // Test with rate = 0.0 and empty array
        let (sum, deriv) = npv_with_deriv_simd(0.0, &empty);
        assert_eq!(sum, 0.0, "NPV of empty array should be 0.0 when rate = 0.0");
        assert_eq!(deriv, 0.0, "Derivative of empty array should be 0.0 when rate = 0.0");
    }

    /// The rewritten AVX2 kernels must agree with the scalar reference at every length:
    /// below the block size, exactly on a block boundary, and at each of the seven
    /// possible remainders past one.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_kernels_match_reference() {
        if !Avx2Ops::is_supported() {
            eprintln!("AVX2 not supported on this CPU, skipping");
            return;
        }

        for &rate in &[0.05, 0.12, -0.35, 0.9, 2.5, 1e-6, -0.999] {
            let base = 1.0 + rate;

            for len in 0..40usize {
                // varied signs and magnitudes, deterministic
                let values: Vec<f64> = (0..len)
                    .map(|i| {
                        let x = (i as f64 + 1.0) * 137.0357;
                        if i % 3 == 0 {
                            -x * 1000.0
                        } else {
                            x
                        }
                    })
                    .collect();

                for &start_from_zero in &[true, false] {
                    let got = unsafe { npv_simd_avx2(base, &values, start_from_zero) };
                    let want =
                        unsafe { npv_simd_generic::<AutoVecOps>(base, &values, start_from_zero) };
                    let tol = 1e-9 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tol,
                        "npv mismatch: rate {rate}, len {len}, start_from_zero {start_from_zero}: got {got}, want {want}"
                    );
                }

                for &start_index in &[0usize, 1, 5] {
                    let (got_sum, got_deriv) =
                        unsafe { npv_with_deriv_avx2(rate, &values, start_index) };
                    let (want_sum, want_deriv) =
                        unsafe { npv_with_deriv_generic::<AutoVecOps>(rate, &values, start_index) };

                    let sum_tol = 1e-9 * want_sum.abs().max(1.0);
                    let deriv_tol = 1e-9 * want_deriv.abs().max(1.0);
                    assert!(
                        (got_sum - want_sum).abs() <= sum_tol,
                        "deriv-sum mismatch: rate {rate}, len {len}, start_index {start_index}: got {got_sum}, want {want_sum}"
                    );
                    assert!(
                        (got_deriv - want_deriv).abs() <= deriv_tol,
                        "deriv mismatch: rate {rate}, len {len}, start_index {start_index}: got {got_deriv}, want {want_deriv}"
                    );
                }
            }
        }
    }

    /// The rewritten plain-AVX kernels must agree with the scalar reference at every length:
    /// below the block size, exactly on a block boundary, and at each of the seven
    /// possible remainders past one.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx_kernels_match_reference() {
        if !AvxOps::is_supported() {
            eprintln!("AVX not supported on this CPU, skipping");
            return;
        }

        for &rate in &[0.05, 0.12, -0.35, 0.9, 2.5, 1e-6, -0.999] {
            let base = 1.0 + rate;

            for len in 0..40usize {
                // varied signs and magnitudes, deterministic
                let values: Vec<f64> = (0..len)
                    .map(|i| {
                        let x = (i as f64 + 1.0) * 137.0357;
                        if i % 3 == 0 {
                            -x * 1000.0
                        } else {
                            x
                        }
                    })
                    .collect();

                for &start_from_zero in &[true, false] {
                    let got = unsafe { npv_simd_avx(base, &values, start_from_zero) };
                    let want =
                        unsafe { npv_simd_generic::<AutoVecOps>(base, &values, start_from_zero) };
                    let tol = 1e-9 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tol,
                        "npv mismatch: rate {rate}, len {len}, start_from_zero {start_from_zero}: got {got}, want {want}"
                    );
                }

                for &start_index in &[0usize, 1, 5] {
                    let (got_sum, got_deriv) =
                        unsafe { npv_with_deriv_avx(rate, &values, start_index) };
                    let (want_sum, want_deriv) =
                        unsafe { npv_with_deriv_generic::<AutoVecOps>(rate, &values, start_index) };

                    let sum_tol = 1e-9 * want_sum.abs().max(1.0);
                    let deriv_tol = 1e-9 * want_deriv.abs().max(1.0);
                    assert!(
                        (got_sum - want_sum).abs() <= sum_tol,
                        "deriv-sum mismatch: rate {rate}, len {len}, start_index {start_index}: got {got_sum}, want {want_sum}"
                    );
                    assert!(
                        (got_deriv - want_deriv).abs() <= deriv_tol,
                        "deriv mismatch: rate {rate}, len {len}, start_index {start_index}: got {got_deriv}, want {want_deriv}"
                    );
                }
            }
        }
    }

    /// The rewritten NEON kernels must agree with the scalar reference at every length:
    /// below the 8-lane block, exactly on a block boundary, and at each of the seven
    /// possible remainders — which on NEON split into up to three whole pairs plus a
    /// single scalar element.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_kernels_match_reference() {
        for &rate in &[0.05, 0.12, -0.35, 0.9, 2.5, 1e-6, -0.999] {
            let base = 1.0 + rate;

            for len in 0..40usize {
                // varied signs and magnitudes, deterministic
                let values: Vec<f64> = (0..len)
                    .map(|i| {
                        let x = (i as f64 + 1.0) * 137.0357;
                        if i % 3 == 0 {
                            -x * 1000.0
                        } else {
                            x
                        }
                    })
                    .collect();

                for &start_from_zero in &[true, false] {
                    let got = unsafe { npv_simd_neon(base, &values, start_from_zero) };
                    let want =
                        unsafe { npv_simd_generic::<AutoVecOps>(base, &values, start_from_zero) };
                    let tol = 1e-9 * want.abs().max(1.0);
                    assert!(
                        (got - want).abs() <= tol,
                        "npv mismatch: rate {rate}, len {len}, start_from_zero {start_from_zero}: got {got}, want {want}"
                    );
                }

                for &start_index in &[0usize, 1, 5] {
                    let (got_sum, got_deriv) =
                        unsafe { npv_with_deriv_neon(rate, &values, start_index) };
                    let (want_sum, want_deriv) =
                        unsafe { npv_with_deriv_generic::<AutoVecOps>(rate, &values, start_index) };

                    let sum_tol = 1e-9 * want_sum.abs().max(1.0);
                    let deriv_tol = 1e-9 * want_deriv.abs().max(1.0);
                    assert!(
                        (got_sum - want_sum).abs() <= sum_tol,
                        "deriv-sum mismatch: rate {rate}, len {len}, start_index {start_index}: got {got_sum}, want {want_sum}"
                    );
                    assert!(
                        (got_deriv - want_deriv).abs() <= deriv_tol,
                        "deriv mismatch: rate {rate}, len {len}, start_index {start_index}: got {got_deriv}, want {want_deriv}"
                    );
                }
            }
        }
    }

    /// Long series: the reciprocal-power chain accumulates error over the whole slice, so
    /// check it stays negligible rather than assuming it does.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn test_neon_kernels_long_series_accuracy() {
        let values: Vec<f64> = (0..5000)
            .map(|i| {
                if i == 0 {
                    -1.0e8
                } else {
                    25_000.0 + (i % 7) as f64 * 13.0
                }
            })
            .collect();

        for &rate in &[0.004, 0.05, 0.35] {
            let base = 1.0 + rate;

            let got = unsafe { npv_simd_neon(base, &values, true) };
            let want = get_reference_npv(&values, rate);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "long npv drift: rate {rate}: got {got}, want {want}"
            );

            let (got_sum, got_deriv) = unsafe { npv_with_deriv_neon(rate, &values[1..], 1) };
            let (want_sum, want_deriv) = get_reference_npv_deriv(&values[1..], rate, 1);
            assert!(
                (got_sum - want_sum).abs() <= 1e-9 * want_sum.abs().max(1.0),
                "long deriv-sum drift: rate {rate}: got {got_sum}, want {want_sum}"
            );
            assert!(
                (got_deriv - want_deriv).abs() <= 1e-9 * want_deriv.abs().max(1.0),
                "long deriv drift: rate {rate}: got {got_deriv}, want {want_deriv}"
            );
        }
    }

    /// Long series: the reciprocal-power chain accumulates error over the whole slice, so
    /// check it stays negligible rather than assuming it does.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_kernels_long_series_accuracy() {
        if !Avx2Ops::is_supported() {
            return;
        }

        let values: Vec<f64> = (0..5000)
            .map(|i| {
                if i == 0 {
                    -1.0e8
                } else {
                    25_000.0 + (i % 7) as f64 * 13.0
                }
            })
            .collect();

        for &rate in &[0.004, 0.05, 0.35] {
            let base = 1.0 + rate;

            let got = unsafe { npv_simd_avx2(base, &values, true) };
            let want = get_reference_npv(&values, rate);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "long npv drift: rate {rate}: got {got}, want {want}"
            );

            let (got_sum, got_deriv) = unsafe { npv_with_deriv_avx2(rate, &values[1..], 1) };
            let (want_sum, want_deriv) = get_reference_npv_deriv(&values[1..], rate, 1);
            assert!(
                (got_sum - want_sum).abs() <= 1e-9 * want_sum.abs().max(1.0),
                "long deriv-sum drift: rate {rate}: got {got_sum}, want {want_sum}"
            );
            assert!(
                (got_deriv - want_deriv).abs() <= 1e-9 * want_deriv.abs().max(1.0),
                "long deriv drift: rate {rate}: got {got_deriv}, want {want_deriv}"
            );
        }
    }
}
