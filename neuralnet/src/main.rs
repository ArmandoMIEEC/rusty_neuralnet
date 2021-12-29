use neuralnet::tools;

fn main() {
    let mut inputs = Vec::new();
    let mut s1 = String::new();
    let mut s2 = String::new();
    let mut x1: bool;
    let mut x2: bool;

    let w1 = vec![1.0, 1.0];
    let w2 = vec![-1.0, -1.0];
    let w3 = vec![1.0, 1.0];
    let b1 = -0.5;
    let b2 = 1.5;
    let b3 = -1.5;
    let n1 = tools::Neuron::new(w1, b1, tools::ActivFunc::Fuzzy);
    let n2 = tools::Neuron::new(w2, b2, tools::ActivFunc::Fuzzy);
    let n3 = tools::Neuron::new(w3, b3, tools::ActivFunc::Fuzzy);

    let l1_neurons = vec![n1, n2];
    let l2_neurons = vec![n3];
    let l1 = tools::Layer::new(l1_neurons);
    let l2 = tools::Layer::new(l2_neurons);

    let net_layers = vec![l1, l2];
    let mut net = tools::Network::new(0, net_layers);

    for _ in 0..24 {
        println!();
    }
    println!(" --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---");
    println!("|  Semester Project - Fuzzy Sets and Neural Networks                |");
    println!("|  Implementation of a 2 layer feedforward neural network in Rust   |");
    println!("|  Implemented Neural Network: x1 XOR x2                            |");
    println!("|  Author: J. Armando Rodrigues                                     |");
    println!(" --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---\n");

    loop {
        loop {
            println!("Enter x1:");
            std::io::stdin()
                .read_line(&mut s1)
                .expect("Error reading your input");

            x1 = match s1.trim().parse() {
                Ok(bool) => bool,
                Err(_error) => {
                    println!(
                        "Could not parse your input as boolean. Accepted values: true, false."
                    );
                    for _ in 0..s1.len() {
                        s1.pop();
                    }
                    continue;
                }
            };

            inputs.push(x1);
            break;
        }

        loop {
            println!("Enter x2:");
            std::io::stdin()
                .read_line(&mut s2)
                .expect("Error reading your input");

            x2 = match s2.trim().parse() {
                Ok(bool) => bool,
                Err(_error) => {
                    println!(
                        "Could not parse your input as boolean. Accepted values: true, false."
                    );
                    for _ in 0..s2.len() {
                        s2.pop();
                    }
                    continue;
                }
            };

            inputs.push(x2);
            break;
        }

        println!("\n --- --- --- --- --- ---");
        println!("    * Network Output *");
        println!(" --- --- --- --- --- ---");
        println!(
            "    x1 XOR x2: {}      ",
            net.calc_output(&inputs).unwrap()[0]
        );
        println!("    Inputs: {}, {}    ", x1, x2);
        println!(" --- --- --- --- --- ---\n");

        for _ in 0..s1.len() {
            s1.pop();
        }
        for _ in 0..s2.len() {
            s2.pop();
        }
        for _ in 0..2 {
            inputs.pop();
        }
    }
}
