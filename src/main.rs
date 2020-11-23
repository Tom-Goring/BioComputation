use bio_computation::Population;
use gnuplot::{AxesCommon, Color, Figure};
use rayon::prelude::*;

fn main() {
    let runs = (0..50)
        .into_par_iter()
        .map(|_| {
            let mut pop = Population::new(50, 50, 2, 1.0, 2);
            let mut results: Vec<f32> = Vec::new();
            let mut generations: Vec<usize> = Vec::new();

            results.push(pop.average_fitness());
            generations.push(pop.current_generation);

            for _ in 0..50 {
                pop.advance_generation();
                results.push(pop.average_fitness());
                generations.push(pop.current_generation);
            }

            results
        })
        .collect::<Vec<Vec<f32>>>();

    let colors = [
        "black", "red", "blue", "green", "orange", "purple", "gray", "pink", "brown", "yellow",
    ];

    let mut fg = Figure::new();

    let axes = fg.axes2d();

    for i in 0..runs.len() {
        axes.lines(0..50, &runs[i], &[Color(colors[i % colors.len()])]);
    }

    // fg.axes2d()
    //     .lines(0..100, &runs[0], &[Color(colors[0])])
    //     .lines(0..100, &runs[1], &[Color(colors[1])])
    //     .lines(0..100, &runs[2], &[Color(colors[2])])
    //     .lines(0..100, &runs[3], &[Color(colors[3])])
    //     .lines(0..100, &runs[4], &[Color(colors[4])])
    //     .lines(0..100, &runs[5], &[Color(colors[5])])
    //     .lines(0..100, &runs[6], &[Color(colors[6])])
    //     .lines(0..100, &runs[7], &[Color(colors[7])])
    //     .lines(0..100, &runs[8], &[Color(colors[8])])
    //     .lines(0..100, &runs[9], &[Color(colors[9])])
    axes.set_x_label("Generation", &[])
        .set_y_label("Generational Average Fitness", &[]);

    fg.show().unwrap();
}
