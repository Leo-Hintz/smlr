use ndarray::Array2;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
pub fn re_lu(x: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
pub fn mse(labels: &Array2<f64>, outputs: &Array2<f64>) -> f64 {
    let diff = outputs - labels;

    let squared_diff = diff.mapv(|x| x.powi(2));
    
    let mse = squared_diff.sum() / squared_diff.len() as f64;

    mse
}