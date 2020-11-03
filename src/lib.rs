use rand::seq::SliceRandom;
use rand::Rng;

pub struct Population {
    pub population: Vec<Individual>,
    pub total_fitness: u32,
    pub current_generation: usize,

    genome_size: usize,
    population_size: usize,
    mutation_probability: usize,
}

impl Population {
    pub fn new(genome_size: usize, population_size: usize, mutation_probability: usize) -> Self {
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

            genome_size,
            population_size,
            mutation_probability,
        }
    }

    pub fn advance_generation(&mut self) {
        let mut rng = rand::thread_rng();
        let mut tmp_pop = Vec::new();
        let mut new_pop = Vec::new();

        for _ in 0..self.population_size {
            let candidate_a = self.population.choose(&mut rng).unwrap();
            let candidate_b = self.population.choose(&mut rng).unwrap();
            let best_candidate = if candidate_a.fitness > candidate_b.fitness {
                candidate_a
            } else {
                candidate_b
            };

            tmp_pop.push(best_candidate.clone());
        }

        for _ in 0..self.population_size {
            let dad = tmp_pop.choose(&mut rng).unwrap();
            let mum = tmp_pop.choose(&mut rng).unwrap();
            new_pop.push(dad.breed_with(mum, self.mutation_probability));
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
}

#[derive(Clone)]
pub struct Individual {
    genes: Vec<u32>,
    genome_size: usize,
    fitness: u32,
}

impl Individual {
    fn new(genome_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let genes: Vec<u32> = (0..genome_size).map(|_| rng.gen_range(0, 2)).collect();
        let fitness = genes.iter().filter(|&n| *n == 1).count() as u32;

        Self {
            genes,
            genome_size,
            fitness,
        }
    }

    fn update_fitness(&mut self) {
        self.fitness = self.genes.iter().filter(|&n| *n == 1).count() as u32
    }

    pub fn breed_with(&self, partner: &Self, mutation_probability: usize) -> Self {
        let self_slice: &[u32] = &self.genes[..];
        let partner_slice: &[u32] = &partner.genes[..];

        let mut rng = rand::thread_rng();
        let crossover_point = rng.gen_range(0, self.genome_size);

        let mut genes: Vec<u32> = (&self_slice[0..crossover_point]).to_vec();
        genes.extend_from_slice(&partner_slice[crossover_point..self.genome_size]);

        let genes: Vec<u32> = genes
            .iter()
            .map(|&gene| {
                if rng.gen_range(0, 100) < mutation_probability {
                    if gene == 0 {
                        1
                    } else {
                        0
                    }
                } else {
                    gene
                }
            })
            .collect();

        let fitness = genes.iter().filter(|&n| *n == 1).count() as u32;

        Individual {
            genome_size: self.genome_size,
            genes,
            fitness,
        }
    }
}
