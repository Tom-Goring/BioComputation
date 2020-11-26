use rand::Rng;

#[derive(Clone)]
pub struct Network {
    pub layers: Vec<Vec<Neuron>>,
    layer_sizes: Vec<u32>,
}

impl Network {
    pub fn new(layer_sizes: &[u32]) -> Network {
        let mut layers = Vec::new();

        let mut layer_iterator = layer_sizes.iter();
        let first_layer_size = *layer_iterator.next().unwrap();
        let mut prev_layer_size = first_layer_size;

        for &layer_size in layer_iterator {
            let mut layer = Vec::new();
            for _ in 0..layer_size {
                layer.push(Neuron::from_rng(prev_layer_size + 1)); // +1 for the bias neuron
            }
            layer.shrink_to_fit();
            layers.push(layer);
            prev_layer_size = layer_size;
        }
        layers.shrink_to_fit();
        Self {
            layers,
            layer_sizes: layer_sizes.to_vec(),
        }
    }

    pub fn from_layers(layers: Vec<Vec<Neuron>>) -> Network {
        let layer_sizes = layers.iter().map(|layer| layer.len() as u32).collect();
        Self {
            layers,
            layer_sizes,
        }
    }

    pub fn train(&mut self) {}

    pub fn predict(&self, inputs: &[f64]) -> f64 {
        *self.run(inputs).last().unwrap().last().unwrap()
    }

    pub fn run(&self, inputs: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();
        results.push(inputs.to_vec());
        for layer in &self.layers {
            let mut layer_results = Vec::new();
            for node in layer {
                layer_results.push(node.process_inputs(results.last().unwrap()))
            }
            results.push(layer_results);
        }
        results
    }
}

#[derive(Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
}

impl Neuron {
    pub fn new(weights: &[f64]) -> Self {
        Self {
            weights: weights.to_vec(),
        }
    }

    pub fn from_rng(num_weights: u32) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..num_weights)
            .into_iter()
            .map(|_| rng.gen_range(-0.5, 0.5))
            .collect();
        Self { weights }
    }

    pub fn process_inputs(&self, inputs: &[f64]) -> f64 {
        sigmoid(process_inputs_with_bias(&self.weights, inputs))
    }

    pub fn weights(&self) -> &Vec<f64> {
        &self.weights
    }

    pub fn update_weights(&mut self, weights: &[f64]) {
        self.weights = weights.to_vec();
    }
}

#[inline]
fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + (-f).exp())
}

#[inline]
fn _step(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        1.0
    } else {
        0.0
    }
}

#[inline]
fn process_inputs_with_bias(node_weights: &[f64], inputs: &[f64]) -> f64 {
    let mut iterator = node_weights.iter();
    let mut total = *iterator.next().unwrap();
    for (weight, value) in iterator.zip(inputs.iter()) {
        total += weight * value;
    }
    total
}
