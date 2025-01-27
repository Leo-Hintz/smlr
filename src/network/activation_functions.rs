#[derive(Clone)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    LeakyReLU,
    None,
}

impl ActivationFunction {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid =>  sigmoid_naive(x),
            ActivationFunction::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::LeakyReLU => if x > 0.0 { x } else { 0.01 * x },
            ActivationFunction::None => x,
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => x * (1.0 - x),
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Tanh => 1.0 - x.powi(2),
            ActivationFunction::LeakyReLU => if x > 0.0 { 1.0 } else { 0.01 },
            ActivationFunction::None => 1.0,
        }
    }
}

fn sigmoid_naive(x : f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

//https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
fn sigmoid_numerically_stable(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn sigmoid_naive_taylor(x: f64) -> f64 {
    0.5 + x * 0.25 - x.powi(3) / 48.0 + x.powi(5) / 480.0 - x.powi(7) / 5376.0
}

#[test]
fn test_sigmoid_naive_accuracy() {
    use crate::test_utils;

    let errors = test_utils::test_math_fn_accuracy(sigmoid_naive, "lookup_tables/sigmoid_lookup.csv");
    println!("Sum of errors for sigmoid_naive: {:?}", errors.iter().map(|(_,y)| y).sum::<f64>());
    println!("Mean squared error sigmoid_naive: {:?}", errors.iter().map(|(_,y)| y.powi(2)).sum::<f64>());
    println!("Max error for sigmoid_naive: {:?}", errors.iter().map(|(_,y)| y).cloned().fold(f64::NEG_INFINITY, f64::max));

    let x_min = errors.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
    let x_max = errors.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let y_max = errors.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);

    let normalized_errors = errors.iter().map(|&(x, y)| (x, y / y_max)).collect::<Vec<(f64, f64)>>();    

    test_utils::plot("sigmoid_numerically_stable_accuracy.svg", vec![normalized_errors], x_min, x_max, -0.001, 1.1);
}
#[test]
fn test_sigmoid_naive_speed() {
    use crate::test_utils;
    let iterations = 10000000;
    let duration = test_utils::test_fn_speed(sigmoid_naive, iterations);
    println!("sigmoid_naive took: {}s for {} iterations", duration, iterations);
}

#[test]
fn test_sigmoid_numerically_stable_accuracy() {
    use crate::test_utils;

    let errors = test_utils::test_math_fn_accuracy(sigmoid_numerically_stable, "lookup_tables/sigmoid_lookup.csv");
    println!("Sum of errors for sigmoid_numerically_stable: {:?}", errors.iter().map(|(_,y)| y).sum::<f64>());
    println!("Mean squared error for sigmoid_numerically_stable: {:?}", errors.iter().map(|(_,y)| y.powi(2)).sum::<f64>());
    println!("Max error {:?} for sigmoid_numerically_stable:", errors.iter().map(|(_,y)| y).cloned().fold(f64::NEG_INFINITY, f64::max));

    let x_min = errors.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
    let x_max = errors.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);
    let y_max = errors.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, f64::max);

    let normalized_errors = errors.iter().map(|&(x, y)| (x, y / y_max)).collect::<Vec<(f64, f64)>>();    

    test_utils::plot("sigmoid_numerically_stable_accuracy.svg", vec![normalized_errors], x_min, x_max, -0.001, 1.1);
}

#[test]
fn test_sigmoid_numerically_stable_speed() {
    use crate::test_utils;
    let iterations = 10000000;
    let duration = test_utils::test_fn_speed(sigmoid_numerically_stable, 10000000);
    println!("sigmoid_numerically_stable took: {}s for {} iterations", duration, iterations);
}

#[test]
fn test_sigmoid_naive_taylor_accuracy() {
    use crate::test_utils;
    let errors = test_utils::test_math_fn_accuracy(sigmoid_naive_taylor, "lookup_tables/sigmoid_lookup.csv");

    println!("Sum of errors for sigmoid_naive_taylor: {:?}", errors.iter().map(|(_,y)| y).sum::<f64>());
    println!("Mean squared error for sigmoid_naive_taylor: {:?}", errors.iter().map(|(_,y)| y.powi(2)).sum::<f64>());
    println!("Max error {:?} for sigmoid_naive_taylor:", errors.iter().map(|(_,y)| y).cloned().fold(f64::NEG_INFINITY, f64::max));
}
