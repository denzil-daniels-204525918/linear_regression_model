use burn::tensor::{TensorData, Tensor, backend::AutodiffBackend};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_dataset<B: AutodiffBackend>(filename: &str) -> (Tensor<B,2>, Tensor<B,2>) {
    let file = File::open(filename).expect("Failed to open dataset");
    let reader = BufReader::new(file);

    let mut x_values = Vec::new();
    let mut y_values = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        if i == 0 { continue; } // Skip CSV header

        let line = line.expect("Error reading line");
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 2 { continue; }

        let x: f32 = parts[0].parse().expect("Invalid x value");
        let y: f32 = parts[1].parse().expect("Invalid y value");

        x_values.push(vec![x]);
        y_values.push(vec![y]);
    }

    let x_tensor = Tensor::from_data(TensorData::from(x_values));
    let y_tensor = Tensor::from_data(TensorData::from(y_values));

    (x_tensor, y_tensor)
}
