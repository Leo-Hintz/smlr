#![allow(dead_code)]
use ndarray::{Array1, Array2, s, Axis};
use rand::prelude::SliceRandom;

#[derive(Clone)]
pub struct Dataset {
    inputs: Array2<f64>,
    labels: Array2<f64>,
    batch_size: usize,
    iter_count: usize,
    order: Vec<usize>,
}

impl Dataset {
    pub fn new(inputs: Array2<f64>, labels: Array2<f64>, batch_size: usize) -> Dataset {
        Dataset {
            inputs: inputs.clone(),
            labels,
            batch_size,
            iter_count: 0,
            order: (0..inputs.dim().1).collect(),
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

    pub fn shuffle(&mut self) {
        self.order.shuffle(&mut rand::thread_rng());
        let mut new_inputs = Array2::zeros(self.inputs.dim());
        let mut new_labels = Array2::zeros(self.labels.dim());
        for (i, &j) in self.order.iter().enumerate() {
            new_inputs.slice_mut(s![.., i]).assign(&self.inputs.slice(s![.., j]));
            new_labels.slice_mut(s![.., i]).assign(&self.labels.slice(s![.., j]));
        }
        self.inputs = new_inputs;
        self.labels = new_labels;
    }
}

impl Iterator for Dataset {
    type Item = (Array2<f64>, Array2<f64>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_count >= self.inputs.dim().1 {
            return None;
        }
        let batch_size = self.batch_size;
        let start_idx = self.iter_count;

        let end_idx = std::cmp::min(self.iter_count + batch_size, self.inputs.dim().1);

        let inputs = self.inputs.slice(s![.., start_idx..end_idx]).to_owned();
        let outputs = self.labels.slice(s![.., start_idx..end_idx]).to_owned();
        self.iter_count = end_idx;
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

#[test]
fn test_iterator_dataset_is_multiple_of_batch_size() {
    for batch_size in 1..100 {
        for multiple in 1..10 {
            println!("Testing: batch_size: {}, dataset_size: {}", batch_size, batch_size * multiple);
            test_iterator(batch_size, batch_size * multiple, false);
        }
    }
}

#[test]
fn test_iterator_dataset_is_not_multiple_of_batch_size() {
    for batch_size in 1..=20 {
        for dataset_size in 1..=200 {
            if dataset_size % batch_size == 0 {
                continue;
            }
            println!("Testing: batch_size: {}, dataset_size: {}", batch_size, dataset_size);
            test_iterator(batch_size, dataset_size, false);
        }
    }}

fn test_iterator(batch_size: usize, dataset_size: usize, shuffle: bool) {
    use ndarray::Array2;

    let inputs = Array2::from_shape_vec((1, dataset_size), vec![0.0; dataset_size]).unwrap();
    let labels = Array2::from_shape_vec((1, dataset_size), vec![0.0; dataset_size]).unwrap();
    let mut dataset = Dataset::new(inputs.clone(), labels.clone(), batch_size);
    
    if shuffle {
        dataset.shuffle();
    }

    for (i, data) in dataset.enumerate() {
        let (inputs, labels) = data;
        println!("inputs: {}", inputs.len_of(Axis(1)));
        println!("labels: {}", labels.len_of(Axis(1)));
        let next_batch_size = batch_size.min(dataset_size - i * batch_size);
        assert_eq!(inputs.len_of(Axis(1)), next_batch_size);
    }
}

#[test]
fn test_iterator_dataset_is_multiple_of_batch_size_after_shuffle() {
    for batch_size in 1..100 {
        for multiple in 1..10 {
            println!("Testing: batch_size: {}, dataset_size: {}", batch_size, batch_size * multiple);
            test_iterator(batch_size, batch_size * multiple, true);
        }
    }
}

#[test]
fn test_iterator_dataset_is_not_multiple_of_batch_size_after_shuffle() {
    for batch_size in 1..=20 {
        for dataset_size in 1..=200 {
            if dataset_size % batch_size == 0 {
                continue;
            }
            println!("Testing: batch_size: {}, dataset_size: {}", batch_size, dataset_size);
            test_iterator(batch_size, dataset_size, true);
        }
    }}

#[test]
fn test_shuffle_multi_dimensional() {
    
    let inputs = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 
                                                              4.0, 5.0, 6.0]).unwrap();

    let labels = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 
                                                              4.0, 5.0, 6.0]).unwrap();
    
    let mut dataset = Dataset::new(inputs.clone(), labels.clone(), 3);
    dataset.shuffle();
    let (new_inputs, new_labels) = dataset.get_all();
    println!("new inputs: {}", new_inputs);
    println!("new labels: {}", new_labels);
}