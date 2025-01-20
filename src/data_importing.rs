#![allow(dead_code)]
extern crate mnist;
extern crate ndarray;

use mnist::*;
use ndarray::{Array1, Array2};

const TRAINING_SIZE: usize = 50_000;
//const VALIDATION_SET_SIZE: usize = 10_000;
const TEST_SET_SIZE: usize = 10_000;

pub fn read_mnist_data() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(50_000)
        .test_set_length(10_000)
        .finalize();
    
    let training_images = Array2::from_shape_vec((784, TRAINING_SIZE), mnist.trn_img.iter().map(|&x| x as f64 / 255.0).collect()).unwrap();    
    let mut training_labels = Array2::zeros((10, TRAINING_SIZE));
    for i in 0..mnist.trn_lbl.len() {
        let next_label = match mnist.trn_lbl[i] {
            0=> Array1::from_shape_vec(10, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1=> Array1::from_shape_vec(10, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            2=> Array1::from_shape_vec(10, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            3=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            4=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            5=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            6=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            7=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            8=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            9=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            _=> panic!("Invalid label")
        };
        training_labels.column_mut(i).assign(&next_label.unwrap());
    }

    let testing_images = Array2::from_shape_vec((784, TEST_SET_SIZE), mnist.tst_img.iter().map(|&x| x as f64 / 255.0).collect()).unwrap();

    let mut testing_labels = Array2::zeros((10, TEST_SET_SIZE));
    for i in 0..mnist.tst_lbl.len() {
        let next_label = match mnist.tst_lbl[i] {
            0=> Array1::from_shape_vec(10, vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1=> Array1::from_shape_vec(10, vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            2=> Array1::from_shape_vec(10, vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            3=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            4=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            5=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            6=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            7=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            8=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            9=> Array1::from_shape_vec(10, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            _=> panic!("Invalid label")
        };
        testing_labels.column_mut(i).assign(&next_label.unwrap());
    }

    (training_images, training_labels, testing_images, testing_labels)
}
