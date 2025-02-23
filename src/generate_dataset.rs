use rand::Rng;
use std::fs::File;
use std::io::Write;

pub fn generate_dataset(filename: &str, num_samples: usize) {
    let mut rng = rand::rng();
    let mut file = File::create(filename).expect("Failed to create dataset file");

    // Write CSV header
    writeln!(file, "x,y").expect("Failed to write header");

    for _ in 0..num_samples {
        let x: f32 = rng.random_range(-10.0..10.0); // Random x values
        let noise: f32 = rng.random_range(-1.0..1.0); // Small noise
        let y = 2.0 * x + 1.0 + noise; // y = 2x + 1 + noise

        writeln!(file, "{},{}", x, y).expect("Failed to write data");
    }

    println!("✅ Dataset saved to {}", filename);
}
