use crate::nn::network::{Network, Neuron};
use crate::Individual;
use gnuplot::PlotOption::Color;
use gnuplot::{AxesCommon, Figure};
use lazy_static::lazy_static;
use num::traits::Pow;
use rand::Rng;
use std::fs::File;
use std::io::{BufRead, BufReader};

const TRAINING_SET_LEN: usize = 1400;

lazy_static! {
    static ref TRAINING_DATA: Vec<(Vec<f64>, f64)> = {
        let file = File::open("data2.txt").expect("File not found");
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        for line in reader.lines().take(TRAINING_SET_LEN) {
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
        let file = File::open("data2.txt").expect("File not found");
        let reader = BufReader::new(file);
        let mut data = Vec::new();
        for line in reader.lines().skip(TRAINING_SET_LEN) {
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
        population_size: 10000,
        epochs: 200,
        tournament_size: 2,
        mutation_rate: 2.0,
        mutation_size: 1.0,
    };

    log::info!(
        "Training Len: {} | Test Len: {}",
        TRAINING_DATA.len(),
        TEST_DATA.len()
    );

    let stats = crate::run::<Network>(config);
    let solution = &stats.solution;

    let percent_correct = (TEST_DATA
        .iter()
        .map(|(inputs, output)| {
            (
                solution.predict(inputs).round() as i32,
                output.round() as i32,
            )
        })
        .filter(|(obtained_answer, ideal_answer)| (*obtained_answer == *ideal_answer))
        .count() as f64
        / TEST_DATA.len() as f64)
        * 100.0 as f64;

    log::info!("Accuracy of {}%", percent_correct.round() as u32);

    let mut fg = Figure::new();
    let axes = fg.axes2d();
    axes.lines(
        0..config.epochs,
        &stats.average_generational_fitness,
        &[Color("green")],
    );
    axes.set_x_label("Generation", &[])
        .set_y_label("Average percent accuracy", &[]);

    fg.show().unwrap();
}

impl Individual for Network {
    fn new() -> Self {
        Self::new(&[6, 3, 3, 3, 3, 3, 3, 3, 1])
    }

    fn crossover(&self, partner: &Self) -> Self {
        let mut rng = rand::thread_rng();
        let mut layers = Vec::new();

        for (self_layer, partner_layer) in self.layers.iter().zip(&partner.layers) {
            let crossover_point = rng.gen_range(0, self_layer.len());
            layers.push(
                self_layer[..crossover_point]
                    .iter()
                    .chain(partner_layer[crossover_point..].iter())
                    .cloned()
                    .collect(),
            );
        }

        Self::from_layers(layers)
    }

    fn mutate(&self, mutation_rate: f64, mutation_size: f64) -> Self {
        let mut rng = rand::thread_rng();

        let mut layers = Vec::new();
        for layer in &self.layers {
            let mut new_layer = Vec::new();
            for node in layer {
                let current_weights = node.weights();
                let new_weights = current_weights
                    .iter()
                    .map(|weight| {
                        if rng.gen_range(0.0, 100.0) < (mutation_rate) {
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
        self.current_fitness
    }

    fn calculate_fitness(&self) -> f64 {
        let ideal_results: Vec<f64> = TRAINING_DATA.iter().map(|(_, output)| *output).collect();
        let actual_results: Vec<f64> = TRAINING_DATA
            .iter()
            .map(|(input, _)| self.predict(input))
            .collect();

        // let mut correct = 0;
        // for (ideal_result, actual_result) in ideal_results.iter().zip(&actual_results) {
        //     if *ideal_result as u32 == *actual_result as u32 {
        //         correct += 1;
        //     }
        // }

        // let percent_correct = ((correct as f64 / TRAINING_SET_LEN as f64) * 100.0) as u32;
        //
        // percent_correct as f64

        1.0 / mean_squared_error(&ideal_results, &actual_results)
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
