use rand::seq::SliceRandom;

pub trait Individual: Clone {
    fn new() -> Self;
    fn crossover(&self, partner: &Self) -> Self;
    fn mutate(&self, mutation_rate: f64, mutation_size: f64) -> Self;
    fn fitness(&self) -> f64;
}

#[derive(Copy, Clone)]
pub struct AlgorithmConfig {
    pub population_size: usize,
    pub epochs: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub mutation_size: f64,
}

pub struct AlgorithmStats<I: Individual> {
    pub total_generational_fitness: Vec<f64>,
    pub average_generational_fitness: Vec<f64>,
    pub solution: I,
}

pub fn run<I: Individual>(config: AlgorithmConfig) -> AlgorithmStats<I> {
    let mut total_generational_fitness = Vec::new();
    let mut average_generational_fitness = Vec::new();
    let mut population: Vec<I> = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..config.population_size {
        population.push(I::new());
    }

    let total_fitness: f64 = population
        .iter()
        .map(|individual| individual.fitness())
        .sum();
    total_generational_fitness.push(total_fitness);
    average_generational_fitness.push(total_fitness / population.len() as f64);

    let mut best_candidate: I = I::new();

    for _ in 0..config.epochs {
        let mut breeding_population = Vec::new();

        for _ in 0..config.population_size {
            let mut tournament: Vec<I> = population
                .choose_multiple(&mut rng, config.tournament_size)
                .cloned()
                .collect();

            tournament.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
            let best_candidate = tournament.last().unwrap();
            breeding_population.push(best_candidate.clone());
        }

        let mut new_population = Vec::new();
        while new_population.len() != config.population_size {
            let parent1 = breeding_population.choose(&mut rng).unwrap();
            let parent2 = breeding_population.choose(&mut rng).unwrap();

            let child1 = parent1
                .crossover(parent2)
                .mutate(config.mutation_rate, config.mutation_size);
            let child2 = parent2
                .crossover(parent1)
                .mutate(config.mutation_rate, config.mutation_size);

            new_population.push(child1);
            new_population.push(child2);
        }

        new_population.sort_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap());
        if new_population.last().unwrap().fitness() > best_candidate.fitness() {
            best_candidate = new_population.last().unwrap().clone();
        }

        population = new_population;
        let total_fitness: f64 = population
            .iter()
            .map(|individual| individual.fitness())
            .sum();
        total_generational_fitness.push(total_fitness);
        average_generational_fitness.push(total_fitness / population.len() as f64)
    }

    AlgorithmStats {
        total_generational_fitness,
        average_generational_fitness,
        solution: best_candidate,
    }
}
