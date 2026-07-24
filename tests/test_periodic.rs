use pyxirr::{cumipmt, cumprinc, fv, ipmt, irr, mirr, nfv, nper, npv, pmt, ppmt, pv, rate};
use rstest::rstest;

mod common;

const INTEREST_RATE: f64 = 0.05;
const PERIODS: f64 = 10.0;
const PAYMENT: f64 = -50_000.0;
const PV: f64 = 100_000.0;
const FV: f64 = 110_000.0;

#[rstest]
fn test_fv_macro_working() {
    assert_future_value!(INTEREST_RATE, PERIODS, -12950.4574965456, PV, None, None);
    assert_future_value!(INTEREST_RATE, PERIODS, -12333.7690443292, PV, None, Some(true));
    assert_future_value!(INTEREST_RATE, PERIODS, -21695.9607427458, PV, Some(FV), None);
    assert_future_value!(INTEREST_RATE, PERIODS, -20662.8197549960, PV, Some(FV), Some(true));
}

// ------------ FV ----------------

#[rstest]
fn test_fv_pmt_at_end() {
    let result = fv(0.05 / 12.0, 10.0 * 12.0, -100.0, -100.0, false);
    assert_almost_eq!(result, 15692.9288943357);
}

#[rstest]
fn test_fv_pmt_at_beginning() {
    let result = fv(0.05 / 12.0, 10.0 * 12.0, -100.0, -100.0, true);
    assert_almost_eq!(result, 15757.6298441047);
}

#[rstest]
fn test_fv_zero_rate() {
    let result = fv(0.0, 10.0 * 12.0, -100.0, -100.0, false);
    assert_almost_eq!(result, 12100.0);
}

#[rstest]
fn test_fv_vectorized() {
    let rates = [[0.05 / 12.0, 0.06 / 12.0], [0.07 / 12.0, 0.0]];
    let mut result = vec![vec![0.0; 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            result[i][j] = fv(rates[i][j], 10.0 * 12.0, -100.0, -100.0, false);
        }
    }

    assert_almost_eq!(result[0][0], 15692.928894335748);
    assert_almost_eq!(result[0][1], 16569.874354049032);
    assert_almost_eq!(result[1][0], 17509.446881023265);
    assert_almost_eq!(result[1][1], 12100.0);
}

#[rstest]
fn test_fv_vectorized_multi() {
    let rates = [0.05 / 12.0, 0.06 / 12.0, 0.07 / 12.0];
    let nper = [5.0 * 12.0, 10.0 * 12.0, 12.0 * 12.0];
    let pv = [-100.0, -150.0, -200.0];
    let pmt_at_beginning = [false, false, true];

    let mut result = vec![0.0; 3];

    for i in 0..3 {
        result[i] = fv(rates[i], nper[i], -100.0, pv[i], pmt_at_beginning[i]);
    }

    assert_almost_eq!(result[0], 6928.944151934635);
    assert_almost_eq!(result[1], 16660.844190750646);
    assert_almost_eq!(result[2], 23062.71469294612);
}

#[rstest]
fn test_fv_vectorized_iterable() {
    let pmt_values = vec![-100.0, -200.0, -300.0];
    let mut actual = vec![0.0; 3];

    for (i, &pmt) in pmt_values.iter().enumerate() {
        actual[i] = fv(0.05 / 12.0, 10.0 * 12.0, pmt, -100.0, false);
    }

    let expected = [15692.92889433575, 31221.15683890247, 46749.38478346919];
    for i in 0..actual.len() {
        assert_almost_eq!(actual[i], expected[i]);
    }
}

// ------------ PV ----------------

#[rstest]
fn test_pv_pmt_at_end() {
    let result = pv(0.05 / 12.0, 10.0 * 12.0, -100.0, 15692.93, false);
    assert_almost_eq!(result, -100.0006713162);
}

#[rstest]
fn test_pv_pmt_at_beginning() {
    let result = pv(0.05 / 12.0, 10.0 * 12.0, -100.0, 15692.93, true);
    assert_almost_eq!(result, -60.71677534615);
}

#[rstest]
fn test_pv_zero_rate() {
    let result = pv(0.0, 10.0 * 12.0, -100.0, 15692.93, false);
    assert_almost_eq!(result, -3692.93);
}

#[rstest]
fn test_pv_default_pv() {
    let result = pv(0.05 / 12.0, 10.0 * 12.0, -100.0, 0.0, false);
    assert_almost_eq!(result, 9428.1350328234);
}

#[rstest]
fn test_pv_vectorized() {
    let rates = [[0.05 / 12.0, 0.06 / 12.0], [0.07 / 12.0, 0.0]];
    let mut result = vec![vec![0.0; 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            result[i][j] = pv(rates[i][j], 10.0 * 12.0, -100.0, 0.0, false);
        }
    }

    assert_almost_eq!(result[0][0], 9428.135032823473);
    assert_almost_eq!(result[0][1], 9007.345332716726);
    assert_almost_eq!(result[1][0], 8612.635414137785);
    assert_almost_eq!(result[1][1], 12000.0);
}

// ------------ NPV ----------------

#[rstest]
fn test_npv_works() {
    let values = vec![-40_000.0, 5_000.0, 8_000.0, 12_000.0, 30_000.0];
    let result = npv(0.08, &values, Some(true));
    assert_almost_eq!(result, 3065.222668179);
}

#[rstest]
fn test_npv_start_from_zero() {
    let values = vec![-40_000.0, 5_000.0, 8_000.0, 12_000.0, 30_000.0];
    let result = npv(0.08, &values, Some(false));
    assert_almost_eq!(result, 2838.169137203);
}

#[rstest]
fn test_npv_zero_rate() {
    let values = vec![-40_000.0, 5_000.0, 8_000.0, 12_000.0, 30_000.0];
    let result = npv(0.0, &values, Some(false));
    assert_almost_eq!(result, 15_000.0);
}

// ------------ PMT ----------------

#[rstest]
fn test_pmt_pmt_at_end() {
    let pmt_result = pmt(INTEREST_RATE, PERIODS, PV, 0.0, false);
    assert_future_value!(INTEREST_RATE, PERIODS, pmt_result, PV, None, None);
}

#[rstest]
fn test_pmt_pmt_at_beginning() {
    let pmt_result = pmt(INTEREST_RATE, PERIODS, PV, 0.0, true);
    assert_future_value!(INTEREST_RATE, PERIODS, pmt_result, PV, None, Some(true));
}

#[rstest]
fn test_pmt_non_zero_fv() {
    let pmt_result = pmt(INTEREST_RATE, PERIODS, PV, FV, false);
    assert_future_value!(INTEREST_RATE, PERIODS, pmt_result, PV, Some(FV), None);
}

#[rstest]
fn test_pmt_zero_rate() {
    let pmt_result = pmt(0.0, PERIODS, PV, FV, false);
    assert_future_value!(0.0, PERIODS, pmt_result, PV, Some(FV), None);
}

#[rstest]
fn test_pmt_vec() {
    let rates = [[0.075 / 12.0, 0.01 / 12.0], [0.0, 0.5 / 12.0]];
    let mut result = vec![vec![0.0; 2]; 2];

    for i in 0..2 {
        for j in 0..2 {
            result[i][j] = pmt(rates[i][j], 12.0 * 15.0, 200_000.0, 0.0, false);
        }
    }

    assert_almost_eq!(result[0][0], -1854.0247200054619);
    assert_almost_eq!(result[0][1], -1196.9890290366611);
    assert_almost_eq!(result[1][0], -1111.111111111111);
    assert_almost_eq!(result[1][1], -8338.702667524864);
}

// ------------ IPMT ----------------

#[rstest]
fn test_ipmt_works() {
    let result = ipmt(INTEREST_RATE, 2.0, PERIODS, PAYMENT, 0.0, false);
    assert_almost_eq!(result, 2301.238562586);
}

#[rstest]
fn test_ipmt_pmt_at_beginning() {
    let result = ipmt(INTEREST_RATE, 2.0, PERIODS, PAYMENT, 0.0, true);
    assert_almost_eq!(result, 2191.6557738917);
}

#[rstest]
fn test_ipmt_non_zero_fv() {
    let result = ipmt(INTEREST_RATE, 2.0, PERIODS, PAYMENT, FV, true);
    assert_almost_eq!(result, 2608.108309425);
}

#[rstest]
fn test_ipmt_first_period() {
    let result = ipmt(INTEREST_RATE, 1.0, PERIODS, PAYMENT, 0.0, false);
    assert_almost_eq!(result, -PAYMENT * INTEREST_RATE);
}

#[rstest]
fn test_ipmt_zero_period() {
    let result = ipmt(INTEREST_RATE, 0.0, PERIODS, PAYMENT, 0.0, false);
    assert!(result.is_nan());
}

#[rstest]
fn test_ipmt_per_greater_than_nper() {
    let result = ipmt(INTEREST_RATE, PERIODS + 2.0, PERIODS, PAYMENT, 0.0, false);
    assert!(result.is_nan());
}

#[rstest]
fn test_ipmt_large_power() {
    let result = ipmt(0.1479, 297.0, 300.0, -270.51, 0.0, false);
    assert_almost_eq!(result, 16.9656277018672);

    let result = ipmt(0.1479, 297.0, 300.0, -270.51, -100.0, false);
    assert_almost_eq!(result, 8.447346936597);

    let result = ipmt(0.1479, 297.0, 300.0, -270.51, 0.0, true);
    assert_almost_eq!(result, 14.7797087741678);

    let result = ipmt(0.1479, 297.0, 300.0, -270.51, -100.0, true);
    assert_almost_eq!(result, 7.358957171005);
}

#[rstest]
fn test_ipmt_vec() {
    let per: Vec<f64> = (0..=13).map(|x| x as f64).collect();
    let n = per.len();
    let mut result = Vec::with_capacity(n);

    for &p in &per {
        result.push(ipmt(0.0824 / 12.0, p, 12.0, 25_000.0, 0.0, false));
    }

    let expected = [
        f64::NAN,
        -171.66666666666666,
        -157.89337457350777,
        -144.0255058746426,
        -130.06241114404526,
        -116.00343649629737,
        -101.84792355596869,
        -87.59520942678299,
        -73.2446266605768,
        -58.79550322604296,
        -44.24716247725825,
        -29.598923121998908,
        -14.850099189833006,
        f64::NAN,
    ];

    for i in 0..n {
        if expected[i].is_nan() {
            assert!(result[i].is_nan(), "Expected NaN at index {}", i);
        } else {
            assert_almost_eq!(result[i], expected[i]);
        }
    }
}

#[rstest]
fn test_ipmt_vec_large_power() {
    let rates = [0.0, 0.1479, 0.1479, 0.1479];
    let final_values = [0.0, 0.0, -100.0, 0.0];
    let pmt_at_beginning = [false, false, false, true];

    let mut result = Vec::with_capacity(4);

    for i in 0..4 {
        let res = ipmt(rates[i], 297.0, 300.0, -270.51, final_values[i], pmt_at_beginning[i]);
        result.push(res);
    }

    let expected = [0.0, 16.9656277018672, 8.447346936597, 14.7797087741678];

    for i in 0..expected.len() {
        assert_almost_eq!(result[i], expected[i]);
    }
}

// ------------ PPMT ----------------

#[rstest]
fn test_ppmt_works() {
    let result = ppmt(INTEREST_RATE, 2.0, PERIODS, PAYMENT, 0.0, false);
    assert_almost_eq!(result, 4173.9901856864);

    let result = ppmt(INTEREST_RATE, 2.0, PERIODS, PAYMENT, 0.0, true);
    assert_almost_eq!(result, 3975.2287482728307);

    let result = ppmt(INTEREST_RATE, 0.0, 10.0, PAYMENT, 0.0, false);
    assert!(result.is_nan());

    let result = ppmt(INTEREST_RATE, 11.0, 10.0, PAYMENT, 0.0, false);
    assert!(result.is_nan());
}

#[rstest]
fn test_ppmt_zero_rate() {
    let result = ppmt(0.0, 2.0, PERIODS, PAYMENT, 0.0, false);
    assert_almost_eq!(result, 5000.0);

    let result = ppmt(0.0, 2.0, PERIODS, PAYMENT, 0.0, true);
    assert_almost_eq!(result, 5000.0);
}

#[rstest]
fn test_ppmt_large_power() {
    // https://github.com/numpy/numpy-financial/issues/35
    let result = ppmt(0.1479, 297.0, 300.0, -270.51, 0.0, false);
    assert_almost_eq!(result, 23.0428012981328);

    let result = ppmt(0.1479, 297.0, 300.0, -270.51, 0.0, true);
    assert_almost_eq!(result, 20.0738751617151);

    let result = ppmt(0.0, 297.0, 300.0, -270.51, 0.0, false);
    assert_almost_eq!(result, 0.9017);
}

#[rstest]
fn test_ppmt_vec() {
    let per: Vec<f64> = (1..6).map(|x| x as f64).collect();
    let mut result = Vec::with_capacity(per.len());

    for &p in &per {
        result.push(ppmt(0.1 / 12.0, p, 24.0, 2000.0, 0.0, false));
    }

    let expected = [
        -75.62318600836664,
        -76.25337922510303,
        -76.88882405197889,
        -77.52956425241204,
        -78.17564395451548,
    ];

    for i in 0..expected.len() {
        assert_almost_eq!(result[i], expected[i])
    }

    let rates = [0.0, 0.0, 0.05, 0.05];
    let pmt_at_beginning = [true, false, true, false];
    let mut result = Vec::with_capacity(4);

    for i in 0..4 {
        result.push(ppmt(rates[i], 2.0, 10.0, -50_000.0, 0.0, pmt_at_beginning[i]));
    }

    let expected = [5000.0, 5000.0, 3975.2287482728307, 4173.9901856864];
    for i in 0..expected.len() {
        assert_almost_eq!(result[i], expected[i])
    }

    let per_invalid = [0.0, 11.0];
    let results =
        per_invalid.iter().map(|&p| ppmt(0.05, p, 10.0, -100.0, 0.0, false)).collect::<Vec<_>>();
    for r in results {
        assert!(r.is_nan());
    }
}

// ------------ NPER ----------------

#[rstest]
fn test_nper_pmt_at_end() {
    let nper_result = nper(INTEREST_RATE, PAYMENT, PV, 0.0, false);
    assert_future_value!(INTEREST_RATE, nper_result, PAYMENT, PV, None, None);
}

#[rstest]
fn test_nper_pmt_at_beginning() {
    let nper_result = nper(INTEREST_RATE, PAYMENT, PV, 0.0, true);
    assert_future_value!(INTEREST_RATE, nper_result, PAYMENT, PV, None, Some(true));
}

#[rstest]
fn test_nper_non_zero_fv() {
    let nper_result = nper(INTEREST_RATE, PAYMENT, PV, FV, false);
    assert_future_value!(INTEREST_RATE, nper_result, PAYMENT, PV, Some(FV), None);
}

#[rstest]
fn test_nper_zero_rate() {
    let nper_result = nper(0.0, PAYMENT, PV, FV, false);
    assert_future_value!(0.0, nper_result, PAYMENT, PV, Some(FV), None);
}

#[rstest]
fn test_nper_vec() {
    let rates = [0.0, 0.075];
    let mut result = Vec::with_capacity(2);

    for &rate in &rates {
        result.push(nper(rate, -2000.0, 0.0, 100_000.0, false));
    }

    assert_almost_eq!(result[0], 50.0);
    assert_almost_eq!(result[1], 21.544944197323336);
}

// ------------ RATE ----------------

#[rstest]
fn test_rate_works() {
    let rate_result = rate(PERIODS, PAYMENT, PV, 0.0, false, None);
    assert_future_value!(rate_result, PERIODS, PAYMENT, PV, None, None);
}

#[rstest]
fn test_rate_non_zero_fv() {
    let rate_result = rate(PERIODS, PAYMENT, PV, FV, false, None);
    assert_future_value!(rate_result, PERIODS, PAYMENT, PV, Some(FV), None);
}

#[rstest]
fn test_rate_pmt_at_beginning() {
    let rate_result = rate(PERIODS, PAYMENT, PV, FV, true, None);
    assert_future_value!(rate_result, PERIODS, PAYMENT, PV, Some(FV), Some(true));
}

#[rstest]
fn test_rate_vec() {
    let pv_values = [-593.06, -4725.38, -662.05, -428.78, -13.65];
    let fv_values = [214.07, 4509.97, 224.11, 686.29, -329.67];
    let mut actual = Vec::with_capacity(pv_values.len());

    for i in 0..pv_values.len() {
        actual.push(rate(2.0, 0.0, pv_values[i], fv_values[i], false, None));
    }

    let expected = [-0.39920185, -0.02305873, -0.41818459, 0.26513414, f64::NAN];

    for i in 0..actual.len() {
        if expected[i].is_nan() {
            assert!(actual[i].is_nan(), "Expected NaN at index {}", i);
        } else {
            assert_almost_eq!(actual[i], expected[i], 1e-8);
        }
    }
}

// ------------ NFV ----------------

#[rstest]
fn test_nfv() {
    // example from https://www.youtube.com/watch?v=775ljhriB8U
    let amounts = vec![1050.0, 1350.0, 1350.0, 1450.0];
    let result = nfv(0.03, 6.0, &amounts);
    assert_almost_eq!(result, 5750.16, 0.01);
}

// ------------ IRR ----------------

#[rstest]
#[case(&[-100.0, 39.0, 59.0, 55.0, 20.0], 0.28094842116)]
#[case(&[-100.0, 0.0, 0.0, 74.0], -0.09549583034)]
#[case(&[-100.0, 100.0, 0.0, -7.0], -0.08329966618)]
#[case(&[-161445.03, 2113.73, 7626.73, 8619.84, 8612.92], -0.43658134635)]
#[case(&[-150000.0, 15000.0, 25000.0, 35000.0, 45000.0, 60000.0], 0.05243288885)]
#[case(&[-100.0, 0.0, 0.0, 74.0], -0.09549583034)]
#[case(&[-100.0, 39.0, 59.0, 55.0, 20.0], 0.28094842115)]
#[case(&[-100.0, 100.0, 0.0, -7.0], -0.08329966618)]
#[case(&[-100.0, 100.0, 0.0, 7.0], 0.06205848562)]
#[case(&[-5.0, 10.5, 1.0, -8.0, 1.0], 0.08859833852)]
#[case(&[-5.0, 10.5, 1.0, -8.0, 1.0, 0.0, 0.0, 0.0], 0.08859833852)]
#[case(&[-40000.0, 5000.0, 8000.0, 12000.0, 30000.0], 0.10582259840)]
#[case(&[-10.0, 2.0, 2.0, 2.0, 2.0], -0.08364541746615073)]
#[case(&[
    -5099701.25, -22503.796875, -22503.79296875, -22503.79296875, -20907.26171875,
    -17899.7421875, -17899.7421875, -17899.7421875, -14660.69140625, -12447.80078125,
    -12447.796875, -12018.1640625, -5991.81640625, -5991.81640625, -5991.81640625,
    -2885.875, 1653.125, 1653.125, 1653.125, 8307.328125, 11110408.45703125
], 0.038039605693757084)]
fn test_irr_works(#[case] input: &[f64], #[case] expected: f64) {
    let result = irr(input, None).unwrap();
    assert_almost_eq!(result, expected, 1e-7);
}

#[rstest]
#[case(&[87.17; 5], &[-86.43], -0.49367042606)]
#[case(&[-87.17; 180], &[5809.3], -0.01352676905)]
#[case(&[-172545.848122807], &[787.735232517999; 480], 0.0038401048)]
#[case(&[
    -12138.436076306429, 576.2077573369947, 576.2077573369947, 576.2077573369947,
    576.2077573369947, 576.2077573369947, 576.2077573369947, 576.2077573369947,
    576.2077573369947, 576.2077573369947, 576.2077573369947, 576.2077573369947,
    576.2077573369947, 576.2077573369947, 576.2077573369947, 576.2077573369947,
    576.2077573369947, 576.2077573369947, 576.2077573369947, 576.2077573369947,
], &[-528.8894576218179], -0.01562626238348752)]
#[case(&[
    -10351.121144852736, 450.71546738230256, 450.71546738230256, 450.71546738230256,
    450.71546738230256, 450.71546738230256, 450.71546738230256, 450.71546738230256,
    450.71546738230256, 450.71546738230256, 450.71546738230256, 450.71546738230256,
    450.71546738230256, 450.71546738230256, 450.71546738230256, 450.71546738230256,
    450.71546738230256, 450.71546738230256, 450.71546738230256, 450.71546738230256,
], &[-654.3817475765102], -0.02792064231450042)]
#[case(&[
    -15634.416942708685, 800.1218253156637, 800.1218253156637, 800.1218253156637,
    800.1218253156637, 800.1218253156637, 800.1218253156637, 800.1218253156637,
    800.1218253156637, 800.1218253156637, 800.1218253156637, 800.1218253156637,
    800.1218253156637, 800.1218253156637, 800.1218253156637, 800.1218253156637,
    800.1218253156637, 800.1218253156637, 800.1218253156637, 800.1218253156637
], &[-304.9753896431489], -0.004883289820554082)]
fn test_irr_equal_payments(#[case] first: &[f64], #[case] other: &[f64], #[case] expected: f64) {
    let input: Vec<f64> = first.iter().chain(other.iter()).cloned().collect();
    let result = irr(&input, None).unwrap();
    assert_almost_eq!(result, expected, 1e-7);
}

#[rstest]
// https://github.com/numpy/numpy-financial/issues/44
#[case(&[-1678.87, 771.96, 1814.05, 3520.30, 3552.95, 3584.99, -1.0], 0.9688775470209261)]
#[case(&[-1678.87, 771.96, 1814.05, 3520.30, 3552.95, 3584.99, 4789.91, -1.0], 1.0042698487205577)]
// https://github.com/numpy/numpy-financial/issues/39
#[case(&[
    -217500.0, -217500.0, 108466.80462450592, 101129.96439328062, 93793.12416205535,
    86456.28393083003, 79119.44369960476, 71782.60346837944, 64445.76323715414,
    57108.92300592884, 49772.08277470355, 42435.24254347826, 35098.40231225296,
    27761.56208102766, 20424.721849802358, 13087.88161857707, 5751.041387351768,
    -1585.7988438735192, -8922.639075098821, -16259.479306324123, -23596.31953754941,
    -30933.159768774713, -38270.0, -45606.8402312253, -52943.680462450604,
    -60280.520693675906, -67617.36092490121
], 0.12)]
// https://github.com/numpy/numpy-financial/issues/28
#[case(&[-50.0, -100.0, 600.0, 300.0, -100.0], 1.8544178284461061)]
// https://github.com/Anexen/pyxirr/issues/56
#[case(&[
    0.0, -54163.55222425675, -15411.724067521238, 11824.611799779348, 13831.220768857136,
    24713.445277451923, 42399.405170720645, 32779.24733697434, 29832.522397937253,
    21750.50072725094, 20140.886499523196, 18357.799360554745, 10074.662845544659
], 0.235461374465902)]
fn test_irr_special_cases(#[case] input: &[f64], #[case] expected: f64) {
    let rate = irr(input, None).unwrap();
    assert_almost_eq!(rate, expected, 1e-6);

    // test net present value of all cash flows equal to zero
    let npv_result = npv(rate, input, Some(true));
    assert_almost_eq!(npv_result, 0.0, 1e-4);
}

#[rstest]
// https://github.com/Anexen/pyxirr/issues/46
#[case(&[
    -1.44852555e+08,  1.28859998e+06,  1.27305118e+06,  1.25407349e+06,
    1.24199669e+06,  1.22647792e+06,  1.21095552e+06,  1.19206955e+06,
    1.17989821e+06,  1.16436524e+06,  1.14883185e+06,  1.12945217e+06,
    1.11780102e+06,  1.10228427e+06,  1.08671783e+06,  1.06755759e+06,
    1.05502327e+06,  1.03885451e+06,  1.02247003e+06,  1.00227444e+06,
    9.88024873e+05
], -0.138541274008)]
#[case(&[
    -1.44852555e+08,  1.41881733e+06,  1.40267049e+06,  1.38296290e+06,
    1.37042160e+06,  1.35430596e+06,  1.33818655e+06,  1.31857419e+06,
    1.30593472e+06,  1.28980433e+06,  1.27367350e+06,  1.25354845e+06,
    1.24144917e+06,  1.22533563e+06,  1.20917048e+06,  1.18927331e+06,
    1.17625690e+06,  1.15946627e+06,  1.14245161e+06,  1.12147927e+06,
    1.10668164e+06
] , -0.13209372260468)]
fn test_gh_46(#[case] input: &[f64], #[case] expected: f64) {
    let rate = irr(input, None).unwrap();
    assert_almost_eq!(rate, expected);

    // test net present value of all cash flows equal to zero
    let npv_result = npv(rate, input, Some(true));
    assert_almost_eq!(npv_result, 0.0, 1e-4);
}

#[rstest]
#[case("tests/samples/unordered.csv", 0.7039842300)]
#[case("tests/samples/random_100.csv", 2.3320600601)]
#[case("tests/samples/random_1000.csv", 0.8607558299)]
#[case("tests/samples/minus_0_993.csv", -0.995697224362268)]
fn test_irr_samples(#[case] input: &str, #[case] expected: f64) {
    let payments = common::load_payments_from_csv(input).unwrap();
    let (_, amounts) = common::split_payments(&payments);

    let rate = irr(&amounts, None).unwrap();
    assert_almost_eq!(rate, expected);

    // test net present value of all cash flows equal to zero
    let npv_result = npv(rate, &amounts, Some(true));
    assert_almost_eq!(npv_result, 0.0, 1e-4);
}

// ------------ MIRR ----------------

#[rstest]
fn test_mirr_works() {
    let values = vec![-1000.0, 100.0, 250.0, 500.0, 500.0];
    let result = mirr(&values, 0.1, 0.1).unwrap();
    assert_almost_eq!(result, 0.10401626745);
}

#[rstest]
fn test_mirr_same_sign() {
    let values_positive = vec![100_000.0, 50_000.0, 25_000.0];
    let result = mirr(&values_positive, 0.1, 0.1);
    assert!(result.is_err());

    let values_negative = vec![-100_000.0, -50_000.0, -25_000.0];
    let result = mirr(&values_negative, 0.1, 0.1);
    assert!(result.is_err());
}

// ------------ CUMPRINC ----------------

#[rstest]
fn test_cumprinc_works() {
    let result = cumprinc(0.09 / 12.0, 30.0 * 12.0, 125_000.0, 13.0, 24.0, false);
    assert_almost_eq!(result, -934.1071234, 1e-7);

    let result = cumprinc(0.09 / 12.0, 30.0 * 12.0, 125_000.0, 1.0, 1.0, false);
    assert_almost_eq!(result, -68.27827118, 1e-7);
}

// ------------ CUMIPMT ----------------

#[rstest]
fn test_cumipmt_works() {
    let result = cumipmt(0.09 / 12.0, 30.0 * 12.0, 125_000.0, 13.0, 24.0, false);
    assert_almost_eq!(result, -11135.23213075);

    let result = cumipmt(0.09 / 12.0, 30.0 * 12.0, 125_000.0, 1.0, 1.0, false);
    assert_almost_eq!(result, -937.5, 1e-7);
}

// ------------ IRR with a guess ----------------

/// A guess must not change *which* root is reported when it points at the root the
/// guess-free call finds. This is the path an IRR series takes: each period seeds the
/// next, so a drifting answer would compound across the series.
#[rstest]
#[case(&[-100.0, 39.0, 59.0, 55.0, 20.0])]
#[case(&[-100.0, 100.0, 0.0, -7.0])]
#[case(&[-40000.0, 5000.0, 8000.0, 12000.0, 30000.0])]
#[case(&[-10.0, 2.0, 2.0, 2.0, 2.0])]
// large magnitudes: |npv| < 1e-3 is unreachable here regardless of how exact the rate is
#[case(&[
    -1.44852555e+08,  1.28859998e+06,  1.27305118e+06,  1.25407349e+06,
    1.24199669e+06,  1.22647792e+06,  1.21095552e+06,  1.19206955e+06,
    1.17989821e+06,  1.16436524e+06,  1.14883185e+06,  1.12945217e+06,
    1.11780102e+06,  1.10228427e+06,  1.08671783e+06,  1.06755759e+06,
    1.05502327e+06,  1.03885451e+06,  1.02247003e+06,  1.00227444e+06,
    9.88024873e+05
])]
#[case(&[
    -5099701.25, -22503.796875, -22503.79296875, -22503.79296875, -20907.26171875,
    -17899.7421875, -17899.7421875, -17899.7421875, -14660.69140625, -12447.80078125,
    -12447.796875, -12018.1640625, -5991.81640625, -5991.81640625, -5991.81640625,
    -2885.875, 1653.125, 1653.125, 1653.125, 8307.328125, 11110408.45703125
])]
fn test_irr_guess_agrees_with_no_guess(#[case] input: &[f64]) {
    let expected = irr(input, None).unwrap();

    // exact seed, and seeds off by a plausible period-over-period drift
    for offset in [0.0, 1e-9, 0.005, -0.005, 0.05, -0.05] {
        let rate = irr(input, Some(expected + offset)).unwrap();
        assert_almost_eq!(rate, expected, 1e-7);
    }
}

/// A guess nowhere near any root must still produce a genuine root, not NaN and not a
/// rate that merely satisfies a loose absolute tolerance.
#[rstest]
#[case(&[-100.0, 39.0, 59.0, 55.0, 20.0])]
#[case(&[-40000.0, 5000.0, 8000.0, 12000.0, 30000.0])]
#[case(&[-10.0, 2.0, 2.0, 2.0, 2.0])]
#[case(&[
    -1.44852555e+08,  1.28859998e+06,  1.27305118e+06,  1.25407349e+06,
    1.24199669e+06,  1.22647792e+06,  1.21095552e+06,  1.19206955e+06,
    1.17989821e+06,  1.16436524e+06,  1.14883185e+06,  1.12945217e+06,
    1.11780102e+06,  1.10228427e+06,  1.08671783e+06,  1.06755759e+06,
    1.05502327e+06,  1.03885451e+06,  1.02247003e+06,  1.00227444e+06,
    9.88024873e+05
])]
fn test_irr_absurd_guess_still_finds_a_root(#[case] input: &[f64]) {
    let scale: f64 = input.iter().map(|v| v.abs()).sum();

    for guess in [-0.999, -0.9, 0.0, 5.0, 100.0] {
        let rate = irr(input, Some(guess)).unwrap();
        assert!(rate.is_finite(), "guess {guess} produced {rate}");
        // npv must vanish relative to the size of the cash flow
        assert!(
            npv(rate, input, Some(true)).abs() <= 1e-6 * scale,
            "guess {guess} produced rate {rate}, npv {}",
            npv(rate, input, Some(true))
        );
    }
}

/// Upstream https://github.com/Anexen/pyxirr/issues/69: an all-zero cash flow used to
/// panic. Our `non_zero_range` guard fixes it differently than upstream's `unwrap_or(0)`,
/// so pin the behavior here too.
#[rstest]
#[case(&[0.0, 0.0, 0.0, 0.0])]
#[case(&[-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
#[case(&[0.0])]
#[case(&[])]
fn test_irr_all_zeros(#[case] input: &[f64]) {
    assert!(irr(input, None).is_err(), "all-zero cash flow must be rejected, not panic");
    assert!(irr(input, Some(0.1)).is_err(), "same with a guess");
}

/// Roots below -99.9% fall outside the `[-0.999, 100]` bracket, so they can only be found
/// by the last-resort grid search. With the previous breakpoints (which started at -0.9)
/// these returned NaN. See `benches/grid_fallback.rs` for the cost of the wider grid.
#[rstest]
#[case(&[-4002.0, 0.001, -0.001, 1e-6], -0.9995)]
#[case(&[
    -4002.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-60,
], -0.9995507255384063)]
fn test_irr_root_below_minus_999(#[case] input: &[f64], #[case] expected: f64) {
    let rate = irr(input, None).unwrap();
    assert!(rate.is_finite(), "expected a root near {expected}, got {rate}");
    assert_almost_eq!(rate, expected, 1e-9);

    // and it really is a root, relative to the size of the cash flow
    let scale: f64 = input.iter().map(|v| v.abs()).sum();
    assert!(npv(rate, input, Some(true)).abs() <= 1e-6 * scale);
}
