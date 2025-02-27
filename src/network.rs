pub mod activation_functions;
pub mod loss_functions;
use activation_functions::ActivationFunction;


extern crate rand;
use core::f64;
use crate::dataset::Dataset;

use rand::{rngs, Rng, SeedableRng};
use ndarray::{Array1, Array2, Axis};


use loss_functions::LossFunction;


pub struct Network {
    layers: Vec<Layer>,
    lossfunction: LossFunction,
    seed: u64,
}

impl Network {

    pub fn new(seed: u64, lossfunction: LossFunction, layers: Vec<Layer>) -> Network {
        
        let mut network = Network { 
            layers,
            lossfunction,
            seed,
        };
        let mut rng = rngs::StdRng::seed_from_u64(seed);
        network.initialize_weights(&mut rng);
        network
    }
 
    pub fn get_layers(&self) -> Vec<Layer> {
        self.layers.clone()
    }

    pub fn get_seed(&self) -> u64 {
        self.seed
    }

    pub fn run(&mut self, input_batch: &Array2<f64>) -> Array2<f64> {
        let mut to_compute = input_batch.to_owned();
        for layer in self.layers.iter() {
            to_compute = layer.calculate_output(&to_compute);
        }
        to_compute
    }

    fn run_and_save_layer_outputs(&mut self, input_batch: &Array2<f64>) -> Vec<Array2<f64>> {
        let mut to_compute = input_batch.to_owned();
        let mut layer_outputs = vec![Array2::zeros((0, 0)); self.layers.len()];
        for (layer, layer_output) in self.layers.iter().zip(layer_outputs.iter_mut()) {
            to_compute = layer.calculate_output(&to_compute);
            *layer_output = to_compute.clone();
        }
        layer_outputs
    }

    pub fn fit(&mut self, dataset: Dataset, epochs: usize, initial_learning_rate: f64, decay_rate: f64) {
        
        
        for i in 0..epochs {
            let learning_rate = initial_learning_rate / (1.0 + decay_rate * i as f64);
            println!("epoch: {}", i);
            for batch in dataset.clone() {
                let (input_batch, output_batch) = batch;
                let layer_outputs = self.run_and_save_layer_outputs(&input_batch);
                self.backpropagation(&input_batch, &output_batch, layer_outputs, learning_rate);
            }
            
            let (inputs, labels) = dataset.get_all();
            let outputs = self.run(&inputs);

            println!("Error is: {}", self.lossfunction.calculate_loss(&labels, &outputs));
        }
    }

    pub fn backpropagation(&mut self, input_data: &Array2<f64>, labels: &Array2<f64>, layer_outputs: Vec<Array2<f64>>, learning_rate: f64) { 
        
        let batch_size: f64 = input_data.dim().1 as f64;
        let mut right_layer_derivatives = self.lossfunction.derivative(labels, layer_outputs.last().unwrap());

        for i in (0..self.layers.len()).rev() {
            let current_layer_outputs = &layer_outputs[i];
            //sigmoid derivatives in R^(w x p) where w is the number of outputs and p is the number of training points
            let activation_derivative = current_layer_outputs.mapv(|x| self.layers[i].activation_function.derivative(x));
            
            //calculate base gradients in R^(w x p) where w is the number of l + 1 neurons and p is the number of training points
            let base_gradients = &activation_derivative * &right_layer_derivatives;
            
            //weight derivatives in R^(n x p) where n is the number of l neurons and p is the number of training points
            let weight_derivatives = if i > 0 { &layer_outputs[i - 1] } else { &input_data };

            //update right layer derivatives in R^(n x p) where n is the number of l layer neurons and p is the number of training points

            //(This is because the number of current weights is the number of next iteration neurons)
            right_layer_derivatives = self.layers[i].weights.t().dot(&base_gradients);
            
            //update weights where layers[i].weights is in R^(w x n) where n is the number of neurons and w is the number of weights per neuron
            let weight_gradients = base_gradients.dot(&weight_derivatives.t());

            let bias_gradients = base_gradients.sum_axis(Axis(1));
            
            self.layers[i].weights = &self.layers[i].weights - (learning_rate * weight_gradients / batch_size);

            //update biases where layers[i].bias is in R^(w x 1) where n is the number of neurons
            self.layers[i].biases = &self.layers[i].biases - (learning_rate * bias_gradients / batch_size);
        }
    }

    fn initialize_weights(&mut self, rng: &mut rngs::StdRng) {
        for layer in self.layers.iter_mut() {
            let (rows, cols) = layer.weights.dim();
            let scale = (6.0 / ((rows + cols) as f64)).sqrt();
            for weight in layer.weights.iter_mut() {
                *weight = rng.gen_range(-scale..scale);
            }
        }
    }
}

#[derive(Clone)]
pub struct Layer {
    weights: Array2<f64>, // One row per (l + 1) neuron, one column per l neuron
    biases: Array1<f64>,
    activation_function: ActivationFunction,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, activation_function: ActivationFunction) -> Layer {  
        Layer {
            weights: Array2::from_shape_vec((output_size, input_size), vec![1.0; output_size * input_size]).unwrap(),
            biases: Array1::from_shape_vec(output_size, vec![1.0; output_size]).unwrap(),
            activation_function,
        }
    }

    pub fn get_weights(&self) -> Array2<f64> {
        self.weights.clone()
    }

    pub fn get_biases(&self) -> Array1<f64> {
        self.biases.clone()
    }

    fn calculate_output(&self, batch: &Array2<f64>) -> Array2<f64> {
        let mut output = self.weights.dot(batch);
        for mut col in output.columns_mut().into_iter() {
            col += &self.biases;
        }
        output.map(|&x| self.activation_function.activate(x))
    }
}


#[test]
fn test_forward_pass() {
    use rand::{rngs, SeedableRng, Rng};
    use ndarray::{Array1, Array2};

    let dataset_size = 10;

    let inputs = Array2::from_shape_vec((1, dataset_size), (1..=dataset_size).map(|x| x as f64).collect()).unwrap();

    let mut network = Network::new(123,
        LossFunction::MeanSquaredError, 
        vec![
        Layer::new(1, 2, ActivationFunction::Tanh),
        Layer::new(2, 1, ActivationFunction::None)
    ]);
    
    network.run(&inputs);
    println!("outputs: {:?}", network.run(&inputs));

    let mut rng = rngs::StdRng::seed_from_u64(123);
    let range: f64 = (6.0 / 3 as f64).sqrt();
    let initial_weights = Array1::from_shape_vec(4, (0..4).map(|_| rng.gen_range(-range..range)).collect::<Vec<f64>>()).unwrap();
    let initial_biases = Array1::from_shape_vec(3, vec![1.0, 1.0, 1.0]).unwrap();
    
    let h1 = |x : f64| (initial_weights[0] * x + initial_biases[0]).tanh();
    let h2 = |x : f64| (initial_weights[1] * x + initial_biases[1]).tanh();
    let o = |x : f64| initial_weights[2] * h1(x) + initial_weights[3] * h2(x) + initial_biases[2];

    let manual_outputs = Array2::from_shape_vec((1, dataset_size), inputs.iter().map(|&x| o(x)).collect()).unwrap();
    println!("manual outputs: {:?}", manual_outputs);
    assert!(manual_outputs.iter().zip(network.run(&inputs).iter()).all(|(a, b)| (a - b).abs() < 1e-15));
}
