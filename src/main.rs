#![allow(dead_code)]

use rand::Rng;
use rand::seq::SliceRandom;

const NUM_GENES: usize = 10;
const POP_COUNT: u32 = 16;
const SELECTION_SIZE: u32 = 4;
const SELECTION_ROUNDS: u32 = 16;

#[derive(Clone)]
struct Individual {
    genes: Vec<u32>,
    fitness: u32,
}

impl Individual {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        let genes: Vec<u32> = (0..NUM_GENES).map(|_| rng.gen_range(0, 2)).collect();
        let fitness = genes.iter().filter(|&n| *n == 1).count() as u32;

        Self {
            genes,
            fitness
        }
    }
}

fn calculate_fitness(individual: &Individual) -> u32 {
    individual.genes.iter().filter(|&n| *n == 1).count() as u32
}

fn calculate_total_fitness(pop: &Vec<Individual>) -> u32 {
    pop.iter().map(|i| i.fitness).sum()
}

fn create_single_offspring_from_pop(pop: &Vec<Individual>) -> Individual {
    let mut rng = rand::thread_rng();
    let dad = pop.choose(&mut rng).unwrap();
    let mum = pop.choose(&mut rng).unwrap();

    let dad_slice: &[u32] = &dad.genes[..];
    let mum_slice: &[u32] = &mum.genes[..];

    let mut genes: Vec<u32> = Vec::new();

    if dad.fitness > mum.fitness {
        genes.extend_from_slice(dad_slice);
    } else {
        genes.extend_from_slice(mum_slice);
    }

    let fitness = genes.iter().filter(|&n| *n == 1).count() as u32 ;

    Individual {
        genes,
        fitness
    }
}

fn main() {
    let mut population: Vec<Individual> = Vec::new();

    for _ in 0..POP_COUNT {
        population.push(Individual::new());
    }

    let mut new_pop = Vec::new();

    for _ in 0..POP_COUNT {
        new_pop.push(create_single_offspring_from_pop(&population));
    }

    let old_fitness = calculate_total_fitness(&population);
    let new_fitness = calculate_total_fitness(&new_pop);

    println!("{}", old_fitness);
    println!("{}", new_fitness);

    assert!(new_fitness > old_fitness);
    println!("Fitness of population increased!");
}
