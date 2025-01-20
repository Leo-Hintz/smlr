pub mod network;
mod utils;
mod data_importing;
pub mod dataset;

//TODO:
    // Optimzations:
        // - (GPU support)

    // New features: 
        // - Support new loss functions (custom)
        // - Support Datasets x
        // - Support different types of normalizations
        // - Support custom activation functions
        // - Support stochastic gradient descent
        // - (Convolution)
        

#[cfg(test)]
mod network_tests {
    use dataset::Dataset;

    use super::*;
    #[test]
    fn test_forward_pass() {
        use rand::{rngs::StdRng, SeedableRng, Rng};
        use ndarray::{Array1, Array2};
        use network::{Network, Layer, ActivationFunction, LossFunction};

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

        let mut rng = StdRng::seed_from_u64(123);
        let range: f64 = (6.0 / 3 as f64).sqrt();
        let initial_weights = Array1::from_shape_vec(4, (0..4).map(|_| rng.gen_range(-range..range)).collect::<Vec<f64>>()).unwrap();
        let initial_biases = Array1::from_shape_vec(3, vec![1.0, 1.0, 1.0]).unwrap();
        
        //perform gradient descent manually for one step for model specifed above:
        let h1 = |x : f64| (initial_weights[0] * x + initial_biases[0]).tanh();
        let h2 = |x : f64| (initial_weights[1] * x + initial_biases[1]).tanh();
        let o = |x : f64| initial_weights[2] * h1(x) + initial_weights[3] * h2(x) + initial_biases[2];

        let manual_outputs = Array2::from_shape_vec((1, dataset_size), inputs.iter().map(|&x| o(x)).collect()).unwrap();
        println!("manual outputs: {:?}", manual_outputs);
        assert!(manual_outputs.iter().zip(network.run(&inputs).iter()).all(|(a, b)| (a - b).abs() < 1e-1));
        
    }
    #[test]
    fn test_one_training_step() {
        use rand::{rngs::StdRng, SeedableRng, Rng};
        use ndarray::{Array1, Array2};
        use network::{Network, Layer, ActivationFunction, LossFunction};
        use dataset::Dataset;

        let learning_rate: f64 = 1.0;
        let dataset_size = 10;

        //Generate dataset
        let inputs = Array2::from_shape_vec((1, dataset_size), (1..=dataset_size).map(|x| x as f64).collect()).unwrap();
        let f = |x : &f64| x * x;
        
        let labels: Vec<f64> = inputs.iter().map(|x| f(x)).collect();
        let labels = Array2::from_shape_vec((1, dataset_size), labels).unwrap();
        
        let dataset = Dataset::new(
            inputs.clone(), 
            labels.clone(), 
            dataset_size
        );

        let mut network = Network::new(123,
            LossFunction::MeanSquaredError, 
            vec![
            Layer::new(1, 2, ActivationFunction::Tanh),
            Layer::new(2, 1, ActivationFunction::None)
        ]);

        //train network using gradient descent with one training step
        network.fit(dataset, 1, learning_rate, 0.0);
        
        let mut rng = StdRng::seed_from_u64(123);
        let range: f64 = (6.0 / 3 as f64).sqrt();
        let initial_weights = Array1::from_shape_vec(4, (0..4).map(|_| rng.gen_range(-range..range)).collect::<Vec<f64>>()).unwrap();
        let initial_biases = Array1::from_shape_vec(3, vec![1.0, 1.0, 1.0]).unwrap();
        
        //perform gradient descent manually for one step for model specifed above:
        let h1 = |x : f64| (initial_weights[0] * x + initial_biases[0]).tanh();
        let h2 = |x : f64| (initial_weights[1] * x + initial_biases[1]).tanh();
        let o = |x : f64| initial_weights[2] * h1(x) + initial_weights[3] * h2(x) + initial_biases[2];
        
        let dmse = |x : f64| (o(x) - f(&x));

        let dweights = |x : f64| {
            let h1 = h1(x);
            let h2 = h2(x);
            let dmse = dmse(x);
            let dweights = Array1::from_shape_vec(4, vec![
                dmse * initial_weights[2] * (1.0 - h1 * h1) * x,
                dmse * initial_weights[3] * (1.0 - h2 * h2) * x,
                dmse * h1,
                dmse * h2
            ]);
            dweights.unwrap()
        };
        let dbiases = |x : f64| {
            let h1 = h1(x);
            let h2 = h2(x);
            let dmse = dmse(x);
            let dbiases = Array1::from_shape_vec(3, vec![
                dmse * initial_weights[2] * (1.0 - h1 * h1),
                dmse * initial_weights[3] * (1.0 - h2 * h2),
                dmse
            ]);
            dbiases.unwrap()
        };

        let weight_gradients = inputs.iter().map(|&x| dweights(x)).fold(Array1::zeros(4),|x: Array1<f64>, a| x + a) / (dataset_size as f64);
        let bias_gradients = inputs.iter().map(|&x| dbiases(x)).fold(Array1::zeros(3),|x: Array1<f64>, a| x + a) / (dataset_size as f64);
        
        let new_manual_biases = &initial_biases - (learning_rate * &bias_gradients);
        let new_manual_weights = &initial_weights - (learning_rate * &weight_gradients);

        //compare new weights after one training step
        let auto_weights = Array1::from_vec(network.get_layers().iter().flat_map(|layer| layer.get_weights()).collect::<Vec<f64>>());
        println!("weight diffs: {:?}", &new_manual_weights - &auto_weights);
        println!("manual weights: {:?}", new_manual_weights);
        println!("auto weights: {:?}", auto_weights);
        
        assert!(new_manual_weights.iter().zip(network.get_layers().iter().flat_map(|layer| layer.get_weights())).all(|(a, b)| (a - b).abs() < 1e-14));
        assert!(new_manual_biases.iter().zip(network.get_layers().iter().flat_map(|layer| layer.get_biases())).all(|(a, b)| (a - b).abs() < 1e-14));
    }
   #[test]
    fn fit_one_dimensional_function() {
        use ndarray::Array2;
        
        use plotlib::page::Page;
        use plotlib::repr::Plot;
        use plotlib::style::{LineStyle, LineJoin};
        use plotlib::view::ContinuousView;

        use network::{Network, Layer, ActivationFunction, LossFunction};

        const DATASET_SIZE: usize = 100;
        const STEP_SIZE: f64 = 0.01;
        
        let learning_rate: f64 = 0.1;
        let decay_rate: f64 = 0.001;
        let batch_size = 10;
        let epochs = 5000;
        println!("epochs: {}", epochs);
        
        //Initialize network
        let mut network = Network::new(
            123,
            LossFunction::MeanSquaredError,
            vec![
            Layer::new(1, 10, ActivationFunction::Tanh),
            Layer::new(10, 10, ActivationFunction::Tanh),
            Layer::new(10, 1, ActivationFunction::None)
        ]);

        let f = |x : &f64| x * x;

        let inputs = Array2::from_shape_vec((1, DATASET_SIZE), (0..DATASET_SIZE).map(|x| (x as f64) * STEP_SIZE).collect::<Vec<f64>>()).unwrap();
        
        let outputs = network.run(&inputs);
        
        let labels: Vec<f64> = inputs.iter().map(|x| f(x)).collect();
        let labels = Array2::from_shape_vec((1, DATASET_SIZE), labels).unwrap();
        
        println!("before training");
        println!("mean squared error is: {}", utils::mse(&labels, &outputs));
        
        let true_data = (0..DATASET_SIZE).map(|x| x as f64).map(|x| (x, f(&x))).collect();
        let predictions = inputs.iter().zip(outputs.iter()).map(|(&input, &output)| (input, output)).collect();
        
        let s1: Plot = Plot::new(true_data).line_style(
            LineStyle::new()
                .colour("burlywood")
                .linejoin(LineJoin::Round),
        );
        let s2: Plot = Plot::new(predictions).line_style(
            LineStyle::new()
                .colour("darkolivegreen")
                .linejoin(LineJoin::Round),
        );
        let view = ContinuousView::new().add(s1).add(s2);
        Page::single(&view).save("before_training.svg").expect("saving svg");

        let dataset = Dataset::new(
            inputs.clone(), 
            labels.clone(), 
            batch_size
        );

        //train network
        network.fit(dataset, epochs, learning_rate, decay_rate);
        
        //test network after training
        let outputs = network.run(&inputs);
        
        println!("after training");
        println!("mean squared error is: {}", utils::mse(&labels, &outputs));

        let true_data = (0..DATASET_SIZE).map(|x| (x as f64) * STEP_SIZE).map(|x| (x, f(&x))).collect();
        let predictions = inputs.iter().zip(outputs.iter()).map(|(&input, &output)| (input, output)).collect();

        let s1: Plot = Plot::new(true_data).line_style(
            LineStyle::new()
                .colour("burlywood")
                .linejoin(LineJoin::Round),
        );
        let s2: Plot = Plot::new(predictions).line_style(
            LineStyle::new()
                .colour("darkolivegreen")
                .linejoin(LineJoin::Round),
        );
        let view = ContinuousView::new().add(s1).add(s2);
        Page::single(&view).save("after_training.svg").expect("saving svg");
    }
}
