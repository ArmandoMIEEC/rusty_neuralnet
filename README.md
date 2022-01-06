# rusty_neuralnet
## Table of Contents
1. [Introduction](#intro)
2. [Code Structure](#code)
3. [Library Crate](#lib)
4. [Bynary Crate](#main)
5. [Testing and Results](#test)
6. [Conclusion](#end)

<a name="intro"></a>
## Introduction
This README file contains a very brief overview of my semester project. For this project I developed a 2 layer feedforward neural network in Rust that implements the XOR logic function between two inputs given by the user. The project includes a toolset of functions to create any feedforward neural network and a cli interface for testing the implemented network. All the code in this project was written by me. 

<a name="code"></a>
## Code Structure
This Rust package includes two crates. The neuralnet library crate provides a set of tools that allow for the creation of multi layered feedforward neural networks.
The binary crate consists of the semester project itself, where I use the library I created to develop a 2 layer feedforward neural network that implements the XOR logic function based on 2 user provided inputs.

<a name="lib"></a>
## Library Crate - Creating an object-oriented toolset for feedforward neural networks
We use two enums. The NeuralNetError enum implements the Debug and Display traits for the errors that the functions in this library can output in case of failure.
The ActivFunc enum is used to express the different activation functions that can be used by each neuron. Although this enum includes several different activation functions I have only implemented the behavior for the "Fuzzy" variant which corresponds to ((sign(z)+1)/2). Trying to use any of the other activation functions will output a NotImplemented error.

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
When it comes to data representation, there are structs representing Neurons, Layers and Networks. So, my library allows for the creation of multiple multi layer networks. The Neuron struct includes a vector of 32 bit floats representing its weights, a bias field, an activation function field (which is an ActivFunc enum as shown earlier), an id field and an output boolean wrapped in an Option type to allow for None outputs (if the output has not been calculated yet). The Layer struct contains a vector of neurons and the Network struct contains a vector of layers. 

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
Each of these structs implements a constructor function called new (as per Rust standards) and a calc_output function. Here is an example of the implementation block for the Neuron struct:

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
For the Layer struct, the calc_output function invokes the calc_output implementation for every neuron in the layer and stores the outputs in a vector of booleans.

```rust
pub fn calc_output(&mut self, inputs: &Vec<bool>) -> Result<Vec<bool>, NeuralNetError> {
            let mut outputs = Vec::new();
            let mut cur_output: bool;
            for neuron in self.neurons.iter_mut() {
                cur_output = neuron.calc_output(inputs)?;
                outputs.push(cur_output);
            }
```
The Network struct implements a similar function, but it uses the outputs of one layer as the inputs of the next.

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
<a name="main"></a>
## Binary Crate - Implementation of the XOR logic function
The binary crate is relatively simple. I started by creating the XOR network. We create 3 neurons: 1 OR neuron, 1 NAND neuron and 1 AND neuron. I added the first 2 neurons to the first layer and the remaining neuron to the second layer. Having created the layers I just added them to a  a new network. The rest of the code in the binary crate enters a loop to constantly ask the user for new inputs for the network and calculate the final network output.

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
<a name="test"></a>
## Testing and Results
The network is working as expected. I did not develop a series of hardcoded results to test the network against. Instead, I created a simple cli interface to allow the user to test the network by himself in a more interactive way.

<a name="end"></a>
## Conclusion
I think that I was able to complete the objectives of the semester project. If this was a bigger project and worth more points some of the improvements that could be made include:
* Implementing a destructor function for each struct implementation block,
* Wrapping the id field of every struct in an Option type (to allow for None ids before they are set),
* Implementing a functional programming aproach instead of an object-oriented one (for example using closures for the activation functions and higher order fucntions for calculating neuron outputs),
* Implementing an automated testing crate for the network (to ditch the user controled cli-based testing entirely and to better test every possible failure condition.

