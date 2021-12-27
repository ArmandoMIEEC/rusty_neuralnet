use neuralnet::tools;

fn main() {
    let inputs = vec![true, true];
    let w = vec![1.0, 1.0];
    let b = -1.5;
    let n = tools::Neuron::new(w, b, tools::ActivFunc::Fuzzy);
    let n_vec = vec![n];
    let mut l = tools::Layer::new(n_vec);

    let o = l.calc_output(&inputs).unwrap();

    println!("{:?}", o);
}
