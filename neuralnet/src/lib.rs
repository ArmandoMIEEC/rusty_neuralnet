pub mod tools {
    use std::error;
    use std::fmt;

    #[derive(Debug)]
    pub enum NeuralNetError {
        Mismatch,
        NotImplemented,
        InvalidActivFuncRes,
    }

    impl std::error::Error for NeuralNetError {}

    impl fmt::Display for NeuralNetError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            match self {
                NeuralNetError::Mismatch => write!(f, "Mismatch Error"),
                NeuralNetError::NotImplemented => write!(f, "Not Implemented Error"),
                NeuralNetError::InvalidActivFuncRes => {
                    write!(f, "Invalid Active Function Result Error")
                }
            }
        }
    }

    pub enum ActivFunc {
        Sigmoid,
        Tanh,
        Linear,
        Fuzzy,
    }

    pub struct Neuron {
        id: u8,
        weights: Vec<f32>,
        bias: f32,
        activ_func: ActivFunc,
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
        pub fn new(weights: Vec<f32>, bias: f32, activ_func: ActivFunc) -> Self {
            Self {
                id: 0,
                weights,
                bias,
                activ_func,
                output: false,
            }
        }
        pub fn calc_output(&mut self, inputs: &Vec<bool>) -> Result<(), NeuralNetError> {
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
                    self.output = fuzzy(z_float).unwrap();
                }
                _ => return Err(NeuralNetError::NotImplemented),
            }

            Ok(())
        }
    }
    impl Layer {
        pub fn new(neurons: Vec<Neuron>) -> Self {
            Self { id: 0, neurons }
        }
    }
    impl Network {
        pub fn new(id: u8, layers: Vec<Layer>) -> Self {
            Self { id, layers }
        }
    }

    pub fn fuzzy(z_float: f32) -> Result<bool, NeuralNetError> {
        let return_val = (sign(z_float) + 1) / 2 as i32;

        match return_val {
            1 => Ok(true),
            0 => Ok(false),
            _ => Err(NeuralNetError::InvalidActivFuncRes),
        }
    }

    pub fn sign(z_float: f32) -> i32 {
        if z_float > 0.0 {
            return 1;
        } else if z_float < 0.0 {
            return -1;
        } else {
            return 0;
        }
    }
}
