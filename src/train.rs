use burn::{
    optim::{Adam, AdamConfig,Optimizer,},
    tensor::{backend::AutodiffBackend, Tensor},
};
use crate::model::{LinearRegression, compute_loss};

pub fn train<B: AutodiffBackend>(
    model: &mut LinearRegression<B>,
    x_train: Tensor<B, 2>,
    y_train: Tensor<B, 2>,
    epochs: usize
) {
    let _optim_config = AdamConfig::new(); // Create optimizer config
    let optim_config = AdamConfig::new();
    let mut optimizer = Adam::new(&optim_config);


    for epoch in 0..epochs {
        let predictions = model.forward(x_train.clone());
        let loss = compute_loss(predictions, y_train.clone());

        optimizer.step(loss.clone()); // Use `step()` to update model parameters

        println!("Epoch {}: Loss = {:?}", epoch + 1, loss.into_scalar());
    }
}