use neuralnet::tools;

fn main() {
    let inputs = vec![true, false];

    let w1 = vec![1.0, 1.0];
    let w2 = vec![-1.0, -1.0];
    let w3 = vec![1.0, 1.0];
    let b1 = -0.5;
    let b2 = 1.5;
    let b3 = -1.5;
    let n1 = tools::Neuron::new(w1, b1, tools::ActivFunc::Fuzzy);
    let n2 = tools::Neuron::new(w2, b2, tools::ActivFunc::Fuzzy);
    let n3 = tools::Neuron::new(w3, b3, tools::ActivFunc::Fuzzy);

    let l1_v = vec![n1, n2];
    let l2_v = vec![n3];
    let l1 = tools::Layer::new(l1_v);
    let l2 = tools::Layer::new(l2_v);

    let net_v = vec![l1, l2];
    let mut net = tools::Network::new(0, net_v);

    println!("{:?}", net.calc_output(&inputs).unwrap());
}
