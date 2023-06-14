use tch::{Tensor, Kind, Device, nn::{self, Linear, Path}};

pub struct SimpleRNN {
    hidden_size: i64,
    i2h: Linear,
    i2o: Linear,
}

impl SimpleRNN {
    pub fn new(vs: &Path<'_>, input_size: i64, hidden_size: i64, output_size: i64) -> Self {
        Self {
            hidden_size,
            i2h: nn::linear(vs, input_size + hidden_size, hidden_size, Default::default()),
            i2o: nn::linear(vs, input_size + hidden_size, output_size, Default::default()),
        }
    }

    pub fn init_hidden(&self, device: Device) -> Tensor {
        Tensor::zeros(self.hidden_size, (Kind::Float, device))
    }

    pub fn forward_rnn(&self, input: &Tensor, hidden: &Tensor) -> (Tensor, Tensor) {
        let combined = Tensor::cat(&[input, hidden], 1);
        let hidden = combined.apply(&self.i2h).tanh();
        let output = combined.apply(&self.i2o).softmax(-1, None);
        (hidden, output)
    }
}