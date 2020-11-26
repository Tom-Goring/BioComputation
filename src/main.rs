use bio_computation::network::Network;

fn main() {
    let nn = Network::new(&[2, 1]);
    println!("{:?}", nn.predict(&[0.0, 0.0]));
    println!("{:?}", nn.predict(&[0.0, 1.0]));
    println!("{:?}", nn.predict(&[1.0, 0.0]));
    println!("{:?}", nn.predict(&[1.0, 1.0]));
}
