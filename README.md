# rusty_neuralnet
## Introduction

## Code Structure
This rust package includes two crates. The neuralnet library crate provides a set of tools that allow for the creation of multi layered feedforward neural networks.
The binary crate consists of the semester project itself, where I use the library I created to develop a 2 layer feedforward neural network that implements the XOR logic function based on 2 user provided inputs.

## Library Crate - Creating an object-oriented toolset for feedforward neural networks

```rust
#[derive(Debug)]
    pub enum NeuralNetError {
        Mismatch,
        NotImplemented,
        InvalidActivFuncRes,
        NoOutput,
    }
    
    pub enum ActivFunc {
        Sigmoid,
        Tanh,
        Linear,
        Fuzzy,
    }
```

```rust
    pub struct Neuron {
        id: u8,
        weights: Vec<f32>,
        bias: f32,
        activ_func: ActivFunc,
        output: Option<bool>,
    }

    pub struct Layer {
        id: u8,
        neurons: Vec<Neuron>,
    }

    #[allow(dead_code)]
    pub struct Network {
        id: u8,
        layers: Vec<Layer>,
    }
```

```rust
   impl Neuron {
        pub fn new(weights: Vec<f32>, bias: f32, activ_func: ActivFunc) -> Self {
            Self {
                id: 0,
                weights,
                bias,
                activ_func,
                output: None,
            }
        }
        pub fn calc_output(&mut self, inputs: &Vec<bool>) -> Result<bool, NeuralNetError> {
            if self.weights.len() != inputs.len() {
                return Err(NeuralNetError::Mismatch);
            }

            let mut sum: f32 = 0.0;
            for pair in self.weights.iter().zip(inputs.iter()) {
                let (weight, input) = pair;
                let input_float: f32 = match *input {
                    true => 1.0,
                    false => 0.0,
                };
                sum += input_float * *weight;
            }

            let z_float = sum + self.bias;

            match self.activ_func {
                ActivFunc::Fuzzy => {
                    self.output = Some(fuzzy(z_float)?);
                }
                _ => return Err(NeuralNetError::NotImplemented),
            }

            match self.output {
                Some(output) => return Ok(output),
                None => return Err(NeuralNetError::NoOutput),
            }
        }
    }
```

```rust
pub fn calc_output(&mut self, inputs: &Vec<bool>) -> Result<Vec<bool>, NeuralNetError> {
            let mut outputs = Vec::new();
            let mut cur_output: bool;
            for neuron in self.neurons.iter_mut() {
                cur_output = neuron.calc_output(inputs)?;
                outputs.push(cur_output);
            }
```

```rust
pub fn calc_output(&mut self, inputs: &Vec<bool>) -> Result<Vec<bool>, NeuralNetError> {
            let mut cur_input = inputs.clone();

            for layer in self.layers.iter_mut() {
                match layer.calc_output(&cur_input) {
                    Ok(thing) => cur_input = thing,
                    _ => return Err(NeuralNetError::NoOutput),
                }
            }
```

## Binary Crate - Implementation of the XOR logic function

```rust
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
```

## Testing and Results

## Conclusion
