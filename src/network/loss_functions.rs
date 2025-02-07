use ndarray::{Array2, Axis};

pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
}

impl LossFunction {
    pub fn calculate_loss(&self, labels: &Array2<f64>, outputs: &Array2<f64>) -> f64 {
        match self {
            LossFunction::MeanSquaredError => {
                let diff = outputs - labels;
                let squared_diff = diff.mapv(|x| x.powi(2));
                let mse = squared_diff.sum() / ((squared_diff.len_of(Axis(0)) * squared_diff.len_of(Axis(1))) as f64);
                mse
            }

            LossFunction::CrossEntropy => {
                //Guide for stable cross entropy: https://jaykmody.com/blog/stable-softmax/
                let c = outputs.iter().fold(f64::NEG_INFINITY, |a, x| if x > &a { *x } else { a });
                let divisors = outputs.map_axis(Axis(0), |x| x.mapv(|x| (x - c).exp()).sum());
                let log_softmax = outputs - c - divisors.mapv(|x| x.ln());
                let loss = (labels * log_softmax).sum();
                -loss / labels.dim().0 as f64
            }
        }
    }
    
    pub fn derivative(&self, labels: &Array2<f64>, outputs: &Array2<f64>) -> Array2<f64> {
        match self {
            LossFunction::MeanSquaredError => outputs - labels,
            LossFunction::CrossEntropy => {
                //Guide for stable cross entropy: https://jaykmody.com/blog/stable-softmax/
                let c = outputs.iter().fold(f64::NEG_INFINITY, |a, x| if x > &a { *x } else { a });
                let divisors = outputs.map_axis(Axis(0), |x| x.mapv(|x| (x - c).exp()).sum());
                let probabilities = (outputs - c).mapv(|x| x.exp()) / divisors;
                probabilities - labels
            }
        }
    }
}

#[test]
fn test_cross_entropy() {
    use rand::{rngs, SeedableRng};

    fn test_three_classes(labels: Array2<f64>, outputs: Array2<f64>) -> () {
        let c = outputs.iter().fold(f64::NEG_INFINITY, |a, x| if x > &a { *x } else { a });
        let divisor = outputs.mapv(|x: f64| (x - c).exp()).sum();
    
        let true_loss = -(
            (labels[[0, 0]] * (outputs[[0, 0]] - c - divisor.ln()) + 
            labels[[1, 0]] * (outputs[[1, 0]] - c - divisor.ln()) + 
            labels[[2, 0]] * (outputs[[2, 0]] - c - divisor.ln())) / 3.0);
    
        
            let loss = LossFunction::CrossEntropy.calculate_loss(&labels, &outputs);
            println!("labels: {:?}", &labels);
            println!("outputs: {:?}", &outputs);
            println!("true loss: {}", true_loss);
            println!("loss: {}", loss);
            assert!((true_loss - loss).abs() < 1e-15);
            assert!(loss.is_finite());
            assert!(!loss.is_nan());
            assert!(loss.is_sign_positive());
    }

    for i in 0..4000 {
        let max_val = 1000000.0;
        rngs::StdRng::seed_from_u64(i);
        let labels = Array2::from_shape_vec((3, 1), (0..3).map(|_| rand::random::<f64>() * max_val).collect()).unwrap();
        let outputs = Array2::from_shape_vec((3, 1), (0..3).map(|_| rand::random::<f64>() * max_val).collect() ).unwrap();
        test_three_classes(labels, outputs);
    }
    
}

#[test]
fn test_mean_squared_error() {
    use rand::{rngs, SeedableRng};

    fn test_three_classes(labels: Array2<f64>, outputs: Array2<f64>) -> () {
    
        let true_loss = ((labels[[0, 0]] - outputs[[0, 0]]).powi(2) + 
                        (labels[[1, 0]] - outputs[[1, 0]]).powi(2) + 
                        (labels[[2, 0]] - outputs[[2, 0]]).powi(2)) / 3.0;
        
            let loss = LossFunction::MeanSquaredError.calculate_loss(&labels, &outputs);
            println!("labels: {:?}", &labels);
            println!("outputs: {:?}", &outputs);
            println!("true loss: {}", true_loss);
            println!("loss: {}", loss);
            assert!((true_loss - loss).abs() < 1e-15);
            assert!(loss.is_finite());
            assert!(!loss.is_nan());
            assert!(loss.is_sign_positive());
    }
    
    for i in 0..4000 {
        let max_val = 1000000.0;
        rngs::StdRng::seed_from_u64(i);
        let labels = Array2::from_shape_vec((3, 1), (0..3).map(|_| rand::random::<f64>() * max_val).collect()).unwrap();
        let outputs = Array2::from_shape_vec((3, 1), (0..3).map(|_| rand::random::<f64>() * max_val).collect() ).unwrap();
        test_three_classes(labels, outputs);
    }
}
