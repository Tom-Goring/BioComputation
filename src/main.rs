#![allow(dead_code, unused_variables, unused_mut)]

use gnuplot::PlotOption::Color;
use gnuplot::{AxesCommon, Figure};
use rand::Rng;

fn main() {
    let config = bio_computation::AlgorithmConfig {
        population_size: 50,
        epochs: 100,
        tournament_size: 4,
        mutation_rate: 1.0,
        mutation_size: 2.0,
    };

    let stats = bio_computation::run::<Individual>(config);

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

const GENOME_SIZE: usize = 50;

#[derive(Clone)]
pub struct Individual {
    genes: Vec<f64>,
}

impl bio_computation::Individual for Individual {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            genes: (0..GENOME_SIZE).map(|_| rng.gen_range(-0.5, 0.5)).collect(),
        }
    }

    fn crossover(&self, partner: &Self) -> Self {
        Self {
            genes: crossover(&self.genes, &partner.genes),
        }
    }

    fn mutate(&self, mutation_rate: f64, mutation_size: f64) -> Self {
        Self {
            genes: self
                .genes
                .iter()
                .map(|gene| mutate(*gene, mutation_rate, mutation_size))
                .collect(),
        }
    }

    fn fitness(&self) -> f64 {
        self.genes.iter().sum()
    }
}

#[inline]
fn mutate(gene: f64, mutation_rate: f64, mutation_size: f64) -> f64 {
    let mut rng = rand::thread_rng();
    if rng.gen_range(0.0, 100.0) < mutation_rate {
        let mut_size = rng.gen_range(0.0, mutation_size);
        if rng.gen_bool(0.5) {
            if gene + mut_size > 1.0 {
                1.0
            } else {
                gene + mut_size
            }
        } else if gene - mut_size < 0.0 {
            0.0
        } else {
            gene - mut_size
        }
    } else {
        gene
    }
}

#[inline]
fn crossover(genome1: &[f64], genome2: &[f64]) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let crossover_point = rng.gen_range(0, genome1.len());
    genome1[..crossover_point]
        .iter()
        .chain(&genome2[crossover_point..])
        .copied()
        .collect()
}

mod tests {
    #[test]
    fn crossover() {
        use super::crossover;

        let genome1 = [0.0; 8];
        let genome2 = [1.0; 8];

        println!("{:?}", crossover(&genome1, &genome2));
    }
}
