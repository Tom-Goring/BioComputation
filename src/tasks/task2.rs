use crate::nn::network::{Network, Neuron};
use crate::Individual;
use gnuplot::PlotOption::Color;
use gnuplot::{AxesCommon, Figure};
use lazy_static::lazy_static;
use num::traits::Pow;
use rand::Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};

lazy_static! {
    static ref TRAINING_DATA: Vec<(Vec<f64>, f64)> = {
        let file = File::open("data1.txt").expect("File not found");
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        for line in reader.lines().take(28) {
            let tokens = line
                .unwrap()
                .split(' ')
                .map(|token| token.parse().unwrap())
                .collect::<Vec<f64>>();

            let len = tokens.len();

            data.push((Vec::from(&tokens[0..len - 1]), tokens[len - 1]));
        }
        data
    };
    static ref TEST_DATA: Vec<(Vec<f64>, f64)> = {
        let file = File::open("data1.txt").expect("File not found");
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        for line in reader.lines().skip(28) {
            let tokens = line
                .unwrap()
                .split(' ')
                .map(|token| token.parse().unwrap())
                .collect::<Vec<f64>>();

            let len = tokens.len();

            data.push((Vec::from(&tokens[0..len - 1]), tokens[len - 1]));
        }
        data
    };
}

pub fn run_data_science_task() {
    let config = crate::AlgorithmConfig {
        population_size: 100,
        epochs: 2000,
        tournament_size: 2,
        mutation_rate: 20.0,
        mutation_size: 0.5,
    };

    let stats = crate::run::<Network>(config);

    let solution = stats.solution;

    println!(
        "Training Len: {}\nTest Len: {}",
        TRAINING_DATA.len(),
        TEST_DATA.len()
    );

    let percent_correct = TEST_DATA
        .iter()
        .map(|(inputs, output)| {
            let x = (
                solution.predict(inputs).round() as i32,
                output.round() as i32,
            );
            if x.0 != x.1 {
                println!(
                    "Wrong prediction for {:?}, wrongly predicted {} vs actual answer of {}",
                    inputs, x.1, x.0
                );
            }
            x
        })
        .filter(|(obtained_answer, ideal_answer)| (*obtained_answer == *ideal_answer))
        .count() as f64
        / TEST_DATA.len() as f64;

    println!(
        "{} percent accuracy when solution is run on sample data.",
        percent_correct * 100.0
    );

    let mut fg = Figure::new();
    let axes = fg.axes2d();
    axes.lines(
        0..config.epochs,
        &stats.average_generational_fitness,
        &[Color("green")],
    );
    axes.set_x_label("Generation", &[])
        .set_y_label("Average Generation Fitness", &[]);

    fg.show().unwrap();
}

impl Individual for Network {
    fn new() -> Self {
        Self::new(&[6, 2, 1])
    }

    fn crossover(&self, _partner: &Self) -> Self {
        self.clone()
    }

    fn mutate(&self, mutation_rate: f64, mutation_size: f64) -> Self {
        let mut rng = rand::thread_rng();

        let mut mut_mod = 1.0;

        if self.fitness() > 0.5 {
            mut_mod = 2.0;
        }

        let mut layers = Vec::new();
        // construct new network using old one
        for layer in &self.layers {
            let mut new_layer = Vec::new();
            for node in layer {
                let current_weights = node.weights();
                let new_weights = current_weights
                    .iter()
                    .map(|weight| {
                        if rng.gen_range(0.0, 100.0) < mutation_rate * mut_mod {
                            let mut_size = rng.gen_range(0.0, mutation_size);
                            if rng.gen_bool(0.5) {
                                weight + mut_size
                            } else {
                                weight - mut_size
                            }
                        } else {
                            *weight
                        }
                    })
                    .collect::<Vec<f64>>();
                new_layer.push(Neuron::new(&new_weights));
            }
            layers.push(new_layer);
        }

        Self::from_layers(layers)
    }

    fn fitness(&self) -> f64 {
        let ideal_results: Vec<f64> = TRAINING_DATA.iter().map(|(_, output)| *output).collect();
        let actual_results: Vec<f64> = TRAINING_DATA
            .iter()
            .map(|(input, _)| self.predict(input))
            .collect();

        0.0 - mean_squared_error(&ideal_results, &actual_results)
    }
}

#[inline]
pub fn mean_squared_error(ideal: &[f64], actual: &[f64]) -> f64 {
    ideal
        .iter()
        .zip(actual)
        .map(|(ideal, actual)| (ideal - actual).pow(2))
        .sum::<f64>()
        / ideal.len() as f64
}
