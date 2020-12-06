use rand::seq::SliceRandom;
use rayon::prelude::*;

pub trait Individual: Clone + Send + Sync {
    fn new() -> Self;
    fn crossover(&self, partner: &Self) -> Self;
    fn mutate(&self, mutation_rate: f64, mutation_size: f64) -> Self;
    fn fitness(&self) -> f64;
    fn calculate_fitness(&self) -> f64;
}

#[derive(Copy, Clone)]
pub struct AlgorithmConfig {
    pub population_size: usize,
    pub epochs: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub mutation_size: f64,
}

#[derive(Clone)]
pub struct AlgorithmStats<I: Individual> {
    pub total_generational_fitness: Vec<f64>,
    pub average_generational_fitness: Vec<f64>,
    pub solution: I,
}

pub fn run<I: Individual>(config: AlgorithmConfig) -> AlgorithmStats<I> {
    let mut total_generational_fitness = Vec::new();
    let mut average_generational_fitness = Vec::new();
    let mut population: Vec<I> = Vec::new();

    for _ in 0..config.population_size {
        population.push(I::new());
    }

    let total_fitness: f64 = population
        .par_iter()
        .map(|individual| individual.fitness())
        .sum();
    total_generational_fitness.push(total_fitness);
    average_generational_fitness.push(total_fitness / population.len() as f64);

    let mut best_candidate: I = I::new();

    for _ in 0..config.epochs {
        let breeding_population: Vec<I> = (0..config.population_size)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let mut tournament: Vec<I> = population
                    .choose_multiple(&mut rng, config.tournament_size)
                    .cloned()
                    .collect();

                tournament.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
                tournament.last().unwrap().clone()
            })
            .collect();

        let mut new_population: Vec<I> = (1..=25)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng();
                let parent1 = breeding_population.choose(&mut rng).unwrap();
                let parent2 = breeding_population.choose(&mut rng).unwrap();

                let child1 = parent1
                    .crossover(parent2)
                    .mutate(config.mutation_rate, config.mutation_size);
                let child2 = parent2
                    .crossover(parent1)
                    .mutate(config.mutation_rate, config.mutation_size);

                [child1, child2]
            })
            .collect::<Vec<[I; 2]>>()
            .iter()
            .flatten()
            .cloned()
            .collect();

        new_population.par_sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());

        let new_population: Vec<I> = new_population
            .iter()
            .par_bridge()
            .map(|individual| individual.mutate(config.mutation_rate, config.mutation_rate))
            .collect();

        if new_population.last().unwrap().fitness() > best_candidate.fitness() {
            best_candidate = new_population.last().unwrap().clone();
        }

        population = new_population;
        let total_fitness: f64 = population
            .par_iter()
            .map(|individual| individual.fitness())
            .sum();
        total_generational_fitness.push(total_fitness);
        average_generational_fitness.push(total_fitness / population.len() as f64);
    }

    AlgorithmStats {
        total_generational_fitness,
        average_generational_fitness,
        solution: best_candidate,
    }
}
