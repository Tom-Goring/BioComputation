use bio_computation::run_data_science_task;
use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().init().unwrap();
    run_data_science_task();
}
