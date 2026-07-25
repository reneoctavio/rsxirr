//! `unsafe` in this crate is confined to the SIMD kernels in `core::periodic::npv`, where
//! it covers two obligations: the `#[target_feature]` preconditions discharged by the
//! cached CPU probe in `simd_tier`, and the raw loads whose bounds the kernels establish
//! themselves. Denying this lint keeps those obligations written down rather than implied
//! by the `unsafe fn` signature -- and it applies on every target, which matters here
//! because half the kernels are behind `cfg(target_arch)` and never compiled by CI.
#![deny(unsafe_op_in_unsafe_fn)]

mod conversions;
mod core;

pub use core::{
    DateLike, DayCount, cumipmt, cumprinc, days_between, fv, ipmt, irr, mirr, nfv, nper, npv, pmt,
    ppmt, pv, rate, xfv, xirr, xnfv, xnpv, year_fraction, zero_crossing_points,
};
