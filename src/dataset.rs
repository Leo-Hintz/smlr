#![allow(dead_code)]
use ndarray::{Array1, Array2, s, Axis};

#[derive(Clone)]
pub enum Normalizer {
    MinMax,
    ZScore,
    None,
}
#[derive(Clone)]
pub struct NormalizerParams {
    param1: Array1<f64>,
    param2: Array1<f64>,
}

#[derive(Clone)]
pub struct Dataset {
    inputs: Array2<f64>,
    labels: Array2<f64>,
    batch_size: usize,
    iter_count: usize,
}

impl Dataset {
    pub fn new(inputs: Array2<f64>, labels: Array2<f64>, batch_size: usize) -> Dataset {
        Dataset {
            inputs,
            labels,
            batch_size,
            iter_count: 0,
        }
    }

    pub fn get_all(&self) -> (Array2<f64>, Array2<f64>) {
        (self.inputs.clone(), self.labels.clone())
    }

    pub fn normalize_inputs(&mut self) -> (Array1<f64>, Array1<f64>) {
        Self::normalize(&mut self.inputs)
    }

    pub fn normalize_labels(&mut self) -> (Array1<f64>, Array1<f64>) {
        Self::normalize(&mut self.labels)
    }

    fn normalize(data: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let maxs = data.map_axis(Axis(1), |x| x.into_iter().fold(f64::NEG_INFINITY, |a, x| x.max(a)));
        let mins = data.map_axis(Axis(1), |x| x.into_iter().fold(f64::INFINITY, |a, x| x.min(a)));

        for (i, mut row) in data.axis_iter_mut(Axis(0)).enumerate() {
            let max = maxs[i];
            let min = mins[i];
            if max != min {
                row.map_inplace(|val| *val = (*val - min) / (max - min));
            }
        }
        (maxs, mins)
    }

    pub fn standardize_inputs(&mut self) -> (Array1<f64>, Array1<f64>) {
        let means = &self.inputs.sum_axis(Axis(1)) / self.inputs.dim().1 as f64;
        let stds = &self.inputs.map_axis(Axis(1), |x| x.std(0.0));
        self.inputs = (&self.inputs - &means) / stds;
        (means.clone(), stds.clone())
    }

    pub fn standardize_labels(&mut self) -> (Array1<f64>, Array1<f64>) {
        let means = &self.labels.sum_axis(Axis(1)) / self.labels.dim().1 as f64;
        let stds = &self.labels.map_axis(Axis(1), |x| x.std(0.0));
        self.labels = (&self.labels - &means) / stds;
        (means.clone(), stds.clone())
    }

}

impl Iterator for Dataset {
    type Item = (Array2<f64>, Array2<f64>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_count >= self.inputs.dim().1 {
            return None;
        }
        let batch_size = self.batch_size;
        let inputs = self.inputs.slice(s![.., self.iter_count..self.iter_count + batch_size]).to_owned();
        let outputs = self.labels.slice(s![.., self.iter_count..self.iter_count + batch_size]).to_owned();
        self.iter_count += batch_size;
        Some((inputs, outputs))
    }
}

#[test]
fn test_standardization() {
    use ndarray::Array2;
            
    let inputs = Array2::from_shape_vec((1, 3), vec![-1.0, 0.0, 1.0]).unwrap();
    let labels = Array2::from_shape_vec((1, 3), vec![-1.0, 1.0, 3.0]).unwrap();
    let mut dataset = Dataset::new(inputs.clone(), labels.clone(), 3);
    dataset.standardize_inputs();
    dataset.standardize_labels();

    let (new_inputs, new_labels) = dataset.get_all();

    println!("standardized inputs: {}", new_inputs);
    println!("standardized labels: {}", new_labels);

    assert_eq!(new_inputs, new_labels);     
}

#[test]
fn test_normalization() {
    use ndarray::Array2;

    use crate::dataset::Dataset;
    let inputs = Array2::from_shape_vec((1, 3), vec![-1.0, 0.0, 1.0]).unwrap();
    let labels = Array2::from_shape_vec((1, 3), vec![-1.0, 1.0, 3.0]).unwrap();
    let mut dataset = Dataset::new(inputs.clone(), labels.clone(), 3);
    dataset.normalize_inputs();
    dataset.normalize_labels();

    let (new_inputs, new_labels) = dataset.get_all();

    println!("normalized inputs: {}", new_inputs);
    println!("normalized labels: {}", new_labels);
    
    assert_eq!(new_inputs, new_labels);
}