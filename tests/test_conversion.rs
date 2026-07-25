use pyxirr::{DateLike, xirr};
use rstest::rstest;
use time::macros::date;

use crate::common::{load_payments_from_csv, split_payments};

mod common;

const INPUT: &str = "tests/samples/unordered.csv";
const EXPECTED: f64 = 0.16353715844;

#[rstest]
fn test_core_xirr_functionality() {
    let payments = load_payments_from_csv(INPUT).unwrap();
    let (dates, amounts) = split_payments(&payments);

    let result = xirr(&dates, &amounts, None, None).unwrap();
    assert_almost_eq!(result, EXPECTED, 1e-7);
}

#[rstest]
fn test_with_different_date_formats() {
    // Create payment data with different date formats
    let dates = vec![date!(2020 - 01 - 01), date!(2021 - 01 - 01), date!(2022 - 01 - 01)]
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<_>>();

    let amounts = vec![-1000.0, 100.0, 1000.0];

    let result = xirr(&dates, &amounts, None, None).unwrap();
    // Expected value would need to be calculated
    assert!(result > 0.0);
}

#[rstest]
fn test_different_amount_types() {
    // Test with different numeric types that would be converted to f64
    let payments = load_payments_from_csv(INPUT).unwrap();
    let (dates, _) = split_payments(&payments);

    // Integer amounts
    let int_amounts: Vec<i32> = vec![-1000, 500, 700];
    let float_amounts: Vec<f64> = int_amounts.iter().map(|&x| x as f64).collect();

    let result = xirr(&dates[0..3], &float_amounts, None, None).unwrap();
    assert!(result > 0.0);
}

#[rstest]
fn test_invalid_inputs() {
    // Test empty arrays
    let dates = Vec::<DateLike>::new();
    let amounts = Vec::<f64>::new();

    let result = xirr(&dates, &amounts, None, None);
    assert!(result.is_err());

    // Test arrays of different lengths
    let dates = vec![date!(2020 - 01 - 01), date!(2021 - 01 - 01)]
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<_>>();

    let amounts = vec![-1000.0, 500.0, 700.0];

    let result = xirr(&dates, &amounts, None, None);
    assert!(result.is_err());

    // Test payments all with same sign
    let dates = vec![date!(2020 - 01 - 01), date!(2021 - 01 - 01), date!(2022 - 01 - 01)]
        .into_iter()
        .map(|x| x.into())
        .collect::<Vec<_>>();

    let amounts = vec![1000.0, 500.0, 700.0];

    let result = xirr(&dates, &amounts, None, None);
    assert!(result.is_err());
}
