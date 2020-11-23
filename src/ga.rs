use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp::Ordering::Equal;

pub struct Population {
    pub population: Vec<Individual>,
    pub total_fitness: f32,
    pub current_generation: usize,

    population_size: usize,
    mutation_probability: usize,
    mutation_step: f32,
    tournament_size: usize,
}

impl Population {
    pub fn new(
        genome_size: usize,
        population_size: usize,
        mutation_probability: usize,
        mutation_step: f32,
        tournament_size: usize,
    ) -> Self {
        let mut population: Vec<Individual> = Vec::new();

        for _ in 0..population_size {
            population.push(Individual::new(genome_size));
        }

        let total_fitness = population.iter().map(|i| i.fitness).sum();
        let current_generation = 1;

        Self {
            population,
            total_fitness,
            current_generation,

            population_size,
            mutation_probability,
            mutation_step,
            tournament_size,
        }
    }

    pub fn advance_generation(&mut self) {
        let mut rng = rand::thread_rng();
        let mut tmp_pop = Vec::new();
        let mut new_pop = Vec::new();

        for _ in 0..self.population_size {
            let mut tournament_pops = Vec::new();

            for _ in 0..self.tournament_size {
                tournament_pops.push(self.population.choose(&mut rng).unwrap());
            }

            tournament_pops.sort_by(|&a, &b| a.fitness.partial_cmp(&b.fitness).unwrap_or(Equal));

            let best_candidate = *tournament_pops.last().unwrap();

            tmp_pop.push(best_candidate);
        }

        for _ in 0..self.population_size {
            let dad = tmp_pop.choose(&mut rng).unwrap();
            let mum = tmp_pop.choose(&mut rng).unwrap();
            new_pop.push(dad.breed_with(mum, self.mutation_probability, self.mutation_step));
        }

        self.population = new_pop;
        self.current_generation += 1;
        self.update_fitness();
    }

    pub fn advance_generations(&mut self, generations: usize) {
        (0..generations)
            .into_iter()
            .for_each(|_| self.advance_generation());
    }

    fn update_fitness(&mut self) {
        self.total_fitness = self.population.iter().map(|i| i.fitness).sum();
    }

    pub fn average_fitness(&self) -> f32 {
        self.population.iter().map(|i| i.fitness).sum::<f32>() / self.population_size as f32
    }
}

#[derive(Clone)]
pub struct Individual {
    genes: Vec<f32>,
    genome_size: usize,
    fitness: f32,
}

impl Individual {
    fn new(genome_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let genes: Vec<f32> = (0..genome_size).map(|_| rng.gen_range(0.0, 1.01)).collect();
        let fitness = genes.iter().sum::<f32>();

        Self {
            genes,
            genome_size,
            fitness,
        }
    }

    pub fn breed_with(
        &self,
        partner: &Self,
        mutation_probability: usize,
        mutation_step: f32,
    ) -> Self {
        let self_slice: &[f32] = &self.genes[..];
        let partner_slice: &[f32] = &partner.genes[..];

        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(0, self.genome_size);

        let mut genes: Vec<f32> = (&self_slice[0..crossover_point]).to_vec();
        genes.extend_from_slice(&partner_slice[crossover_point..self.genome_size]);

        #[rustfmt::skip]
            let genes: Vec<f32> = genes
            .iter()
            .map(|&gene| {
                if rng.gen_range(0, 100) < mutation_probability {
                    let amount = rng.gen_range(0.0, mutation_step);
                    if rng.gen_bool(1.0 / 2.0) {
                        if gene + amount > 1.0 { 1.0 } else { gene + amount }
                    } else {
                        if gene - amount < 0.0 { 0.0 } else { gene - amount }
                    }
                } else {
                    gene
                }
            })
            .collect();

        let fitness = genes.iter().sum::<f32>() as f32;

        Individual {
            genome_size: self.genome_size,
            genes,
            fitness,
        }
    }
}
