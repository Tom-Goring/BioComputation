use bio_computation::Population;
use gnuplot::{AxesCommon, Caption, Color, Figure};

fn main() {
    let mut pop = Population::new(50, 50, 1);
    let mut results: Vec<u32> = Vec::new();
    let mut generations: Vec<usize> = Vec::new();

    results.push(pop.total_fitness);
    generations.push(pop.current_generation);

    for _ in 0..110 {
        pop.advance_generations(1);
        results.push(pop.total_fitness);
        generations.push(pop.current_generation);
    }

    let mut fg = Figure::new();
    fg.axes2d()
        .lines(&generations, &results, &[Caption(""), Color("black")])
        .set_x_label("Generation", &[])
        .set_y_label("Total Fitness", &[]);
    fg.show().unwrap();
}
