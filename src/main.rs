
mod dataset;
mod model;
mod train;
mod generate_dataset;

use dataset::load_dataset;
use model::LinearRegression;
use train::train;

fn main() {
    // ✅ Generate dataset
    crate::generate_dataset::generate_dataset("dataset.csv", 100); // 100 samples

    // ✅ Load dataset
    let (x_train, y_train) = load_dataset("dataset.csv");

    // ✅ Initialize and train the model
    let mut model = LinearRegression::new();
    train(&mut model, x_train, y_train, 100); // Train for 100 epochs
}