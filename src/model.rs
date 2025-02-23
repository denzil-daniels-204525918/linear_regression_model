use burn::module::Module;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use burn::nn::{Linear, MSELoss};

#[derive(Module, Debug)]
pub struct LinearRegression<B: AutodiffBackend> {
    linear: Linear<B>,
}

impl<B: AutodiffBackend> LinearRegression<B> {
    // Initialize the model
    pub fn new() -> Self {
        Self {
            linear: Linear::new(1, 1), // 1 input (x), 1 output (y)
        }
    }

    // Forward pass
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}

struct MSELoss();

// Loss function (Mean Squared Error)
pub fn compute_loss<B: AutodiffBackend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 2> {
    MSELoss::new().forward(predictions, targets)
}
