use std::cmp::Ordering;

use super::npv;

#[inline(always)]
pub(super) fn irr_analytical_2(values: &[f64]) -> f64 {
    // cf[0]/(1+r)^0 + cf[1]/(1+r)^1 = 0  => multiply by (1 + r)
    // cf[0]*(1+r) + cf[1] = 0  => divide by cf[0] and move tho the right
    // lets x = 1+r, a = cf[0], b = cf[1]
    // solve a*x + b = 0
    // x = -b/a, r = x - 1
    -values[1] / values[0] - 1.0
}

#[inline(always)]
pub(super) fn irr_analytical_3(values: &[f64]) -> f64 {
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
