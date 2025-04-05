use std::{fs::File, io::BufReader, path::Path};

use csv::ReaderBuilder;
use time::{macros::format_description, Date};

/// Loads payments from a CSV file
pub fn load_payments_from_csv<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<(Date, f64)>, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut csv_reader = ReaderBuilder::new()
        .has_headers(false)
        .trim(csv::Trim::All) // Trim whitespace from all fields
        .flexible(true) // Be flexible with the format
        .double_quote(true) // Handle quoted fields properly
        .from_reader(reader);

    // Create a format for parsing dates in YYYY-MM-DD format
    let format = format_description!("[year]-[month]-[day]");

    let mut payments = Vec::new();
    for result in csv_reader.records() {
        let record = result?;

        // Skip empty lines
        if record.is_empty() || record.iter().all(|field| field.trim().is_empty()) {
            continue;
        }

        // Skip lines with too few fields
        if record.len() < 2 {
            continue;
        }

        let date_str = record.get(0).unwrap().trim();
        let amount_str = record.get(1).unwrap().trim();

        // Skip if either field is empty after trimming
        if date_str.is_empty() || amount_str.is_empty() {
            continue;
        }

        // Handle quoted date strings by removing quotes if present
        let date_str = date_str.trim_matches('"');

        let date = Date::parse(date_str, &format)?;
        let amount = amount_str.parse::<f64>()?;

        payments.push((date, amount));
    }

    Ok(payments)
}

/// Splits a payment sequence into dates and amounts
pub fn split_payments(payments: &[(Date, f64)]) -> (Vec<pyxirr::DateLike>, Vec<f64>) {
    let dates = payments.iter().map(|(date, _)| (*date).into()).collect();
    let amounts = payments.iter().map(|(_, amount)| *amount).collect();

    (dates, amounts)
}

/// Assert that two f64 values are almost equal
#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b, eps) = (&$a, &$b, $eps);
        assert!((*a - *b).abs() < eps, "assertion failed: `({} !~= {})`", *a, *b);
    }};
    ($a:expr, $b:expr) => {{
        let (a, b) = (&$a, &$b);
        let eps: f64 = 1e-9;
        assert!((*a - *b).abs() < eps, "assertion failed: `({} !~= {})`", *a, *b);
    }};
}

/// Assert the future value equation balances
#[macro_export]
macro_rules! assert_future_value {
    ($rate:expr, $nper:expr, $pmt:expr, $pv:expr, $fv:expr, $pmt_at_beginning:expr) => {{
        let (rate, nper, pmt, pv, fv, pmt_at_beginning) =
            ($rate, $nper, $pmt, $pv, $fv, $pmt_at_beginning);

        let fv = fv.unwrap_or(0.0);

        if rate == 0.0 {
            assert_almost_eq!(fv + pv + pmt * nper, 0.0);
            return;
        }

        let pmt_at_beginning = if pmt_at_beginning.unwrap_or(false) {
            1.0
        } else {
            0.0
        };

        let result = fv
            + pv * f64::powf(1.0 + rate, nper)
            + pmt * (1.0 + rate * pmt_at_beginning) / rate * (f64::powf(1.0 + rate, nper) - 1.0);

        assert_almost_eq!(result, 0.0, 1e-6);
    }};
}
