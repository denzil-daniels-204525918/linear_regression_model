# Steps to set up the project

Install Rust:
• Download and install Rust from  https://www.rust-lang.org/tools/install our run curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
In your terminal and follow the instructions.
• Verify and test the installation by running rustc --version in your terminal. 

Install Rust Rover IDE(GUI):
• Download Rust Rover from https://www.jetbrains.com/rust/.
• Follow the installation instructions for your operating system.

Create a new Rust Project:
• Open Rust Rover.
• Run in terminal: 
cargo new linear_regression_model 
cd linear_regression_model
• Replace the contents of the Cargo.toml file with the provided TOML configuration:
[dependencies]
burn = { version = "0.16.0", features = ["wgpu", "train"] }
ndarray = "0.16.0"
rand = "0.9.0"
rgb = "0.8.50"
textplots = "0.8.6"

Connect Rust Rover to GitHub:
• Install Git from https://git-scm.com/.
• In Rust Rover, go to VCS > Enable Version Control Integration and select Git.
• Create a new repository on GitHub and link it to your project. You can use command line: git init     
git remote add origin <your - github - repo - u r l > 
git add .  
git commit -m ”Initial commit” 
git push -u origin main   • Alternatively, navigate to file>New>Project from Version Control. On the Clone repository tab that pops up, select Git for Version Control, Add your dedicated repository to URL: and select your RustRover directory under Directory




Creating the Linear Regression Model  Generating synthetic data


Create a new file to load the dataset called dataset.rs and execute the following code:  use rand::Rng;
use std::fs::File;
use std::io::Write;

fn generate_dataset(filename: &str, num_samples: usize) {
    let mut rng= rand::rng();
    let noise_factor = 1.0; // Adjust noise level
    let mut file = File::create(filename).expect("Unable to create file");

    // Write CSV header
    writeln!(file, "x,y").expect("Unable to write header");

    for _ in 0..num_samples {
        let x: f64 = rng.random_range(-10.0..10.0); // Random x values in range [-10, 10] for a range of 100, this can be changed as needed
        let noise: f64 = rng.random_range(-noise_factor..noise_factor); // Small noise
        let y = 2.0 * x + 1.0 + noise; // Apply equation with noise

        writeln!(file, "{},{}", x, y).expect("Unable to write data");
    }

    println!("Dataset saved to {}", filename);
}

fn main() {
    generate_dataset("dataset.csv", 100); // Generate 100 samples
}

*Note that rand will be seen as an unresolved import unless the dependencies were created in cargo.toml.  Creates a dataset where: it generates 100(x,y) points and ads noise to y=2x+1 for realism. The dataset then gets saved to dataset.csv for training. This will simulate real-world data with some randomness.

• Run the program in Rust Rover’s terminal using sh cargo run. This creates a CSV dataset in the root.


Defining the model
Use the burn library to define a simple linear regression model where: y = wx + b
where w (weight) and b (bias) are learnable parameters.  • Using the following code we will implement linear regression model with one layer. Create a new file called model.rs and add the following code:

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::AutodiffBackend, Tensor},
};

#[derive(Module, Debug)]
pub struct LinearRegression<B: AutodiffBackend> {
    layer: Linear<B>,
}

impl<B: AutodiffBackend> LinearRegression<B> {
    pub fn new() -> Self {
        Self {
            layer: LinearConfig::new(1, 1).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.layer.forward(x)
    }
}

// Compute Mean Squared Error (MSE) loss
pub fn compute_loss<B: AutodiffBackend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    (predictions - targets).powf(2.0).mean()
}



Training the model
Create a file called train.rs an add the following code:  use burn::{
    optim::{Adam, AdamConfig},
    tensor::{backend::AutodiffBackend, Tensor},
};
use crate::model::{LinearRegression, compute_loss};

pub fn train<B: AutodiffBackend>(
    model: &mut LinearRegression<B>, 
    x_train: Tensor<B, 2>, 
    y_train: Tensor<B, 2>, 
    epochs: usize
) {
    let optim_config = AdamConfig::new(); // Create optimizer config
    let mut optimizer = Adam::new(&optim_config); // Initialize optimizer

    for epoch in 0..epochs {
        let predictions = model.forward(x_train.clone()); // Forward pass
        let loss = compute_loss(predictions.clone(), y_train.clone()); // Compute loss

        optimizer.step(loss.clone()); // Update model parameters

        println!("Epoch {}: Loss = {:?}", epoch + 1, loss.into_scalar());
    }
}

Trains for epochs iterations.
Uses Adam optimizer for gradient descent.
Prints loss per epoch to monitor training.

Evaluate Model

The code added to the main.rs file test the model on the new unseen data and prints the information required.  mod dataset;
mod model;
mod train;

use burn::{
    tensor::{backend::AutodiffBackend, Data, Tensor},
};
use dataset::generate_data;
use model::LinearRegression;
use train::train;

fn main() {
    let data = generate_data(100);
    
    // Convert dataset to tensors
    let x_values: Vec<f32> = data.iter().map(|(x, _)| *x).collect();
    let y_values: Vec<f32> = data.iter().map(|(_, y)| *y).collect();

    let x_train = Tensor::<AutodiffBackend, 2>::from_data(Data::from(x_values));
    let y_train = Tensor::<AutodiffBackend, 2>::from_data(Data::from(y_values));

    let mut model = LinearRegression::new();
    
    println!("🚀 Training model...");
    train(&mut model, x_train, y_train, 100);

    println!("✅ Training complete!");
}

Run the Code
Compile the build by executing the following commands in Rust’s terminal: Compile the project: sh cargo build
Run the training script: sh cargo run

Expected output:
✅ Dataset saved to dataset.csv
🚀 Training model...
Epoch 1: Loss = 4.82
Epoch 2: Loss = 3.72
...
Epoch 100: Loss = 0.01
✅ Training complete!



