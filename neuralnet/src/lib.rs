pub mod tools {
    use std::error;
    use std::io::{Error, ErrorKind};

    pub struct Neuron {
        id: u8,
        layer: u8,
        weights: Vec<f32>,
        bias: f32,
        output: bool,
    }

    pub struct Layer {
        id: u8,
        neurons: Vec<Neuron>,
    }

    pub struct Network {
        id: u8,
        layers: Vec<Layer>,
    }

    impl Neuron {
        pub fn create_neuron(weights: Vec<f32>, bias: f32) -> Neuron {
            Neuron {
                id: 0,
                layer: 0,
                weights,
                bias,
                output: false,
            }
        }
        pub fn calc_output(
            inputs: &Vec<bool>,
            neuron: &Neuron,
        ) -> Result<(), Box<dyn error::Error>> {
            Ok(())
        }
    }
    impl Layer {
        pub fn create_layer(neurons: Vec<Neuron>) -> Layer {
            Layer { id: 0, neurons }
        }
    }
    impl Network {
        pub fn create_network(id: u8, layers: Vec<Layer>) -> Network {
            Network { id, layers }
        }
    }
}
