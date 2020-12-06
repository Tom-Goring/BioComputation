# Biocomputation 2020

## Installation

### Windows

Navigate to https://www.rust-lang.org/tools/install, download the .exe and execute it. You will need to have the corresponding C++ redistributables installed.

### Linux

```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## How to Run

Run the following command to clone the repo to your local machine:
```
git clone git@github.com:Tom-Goring/BioComputation.git
```

cd into the directory and run:
```
cargo run --release
```
The program should compile in release mode and run. It may take a long while on weak machines - the outputs in my project report were run on a machine with 32 threads, so use that as a benchmark.