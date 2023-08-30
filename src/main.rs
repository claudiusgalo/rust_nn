use activations::SIGMOID;
use network::Network;

pub mod activations;
pub mod matrix;
pub mod network;

fn main() {
  //At some point I will attempt to pass in vectorized database inputs and target values for training.
  let inputs = vec![
    vec![0.0, 0.0],
    vec![1.0, 0.0],
    vec![0.0, 1.0],
    vec![1.0, 1.0],
  ];

  let targets = vec![
    vec![0.0],
    vec![1.0],
    vec![1.0],
    vec![0.0],
  ];

  //Here you can define the layers [(input), (hidden), (output)], learning rate, activation method, and generations.
  let mut network = Network::new(vec![2, 4, 1 ], 0.9, SIGMOID);
  network.train(inputs, targets, 18000);


  println!("0 and 0: {:?}", network.feed_forward(vec![0.0, 0.0]));
  println!("0 and 1: {:?}", network.feed_forward(vec![0.0, 1.0]));
  println!("1 and 0: {:?}", network.feed_forward(vec![1.0, 0.0]));
  println!("1 and 1: {:?}", network.feed_forward(vec![1.0, 1.0]));
}
