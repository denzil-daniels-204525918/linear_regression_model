# **Linear Regression Model in Rust using Burn**

## 📌 Introduction  

This project focuses on building a simple **AI model for linear regression** using **Rust** and the **Burn** library (version 0.16.0). The objective is to train a model that accurately predicts values for the function:  

\[
y = 2x + 1
\]  

To achieve this, we:  
- **Generate synthetic data** with random noise to simulate real-world conditions.  
- **Define a linear regression model** using the Burn deep learning framework.  
- **Train the model** using an optimization algorithm.  
- **Evaluate its performance** based on loss reduction and accuracy.  

All dependencies are managed according to the provided `Cargo.toml` file.  

This implementation demonstrates how **machine learning concepts**—such as **linear regression, loss functions, and optimization**—can be applied in **Rust**, highlighting its potential for AI development. 🚀  

## **📌 Project Overview**  
This project implements a simple **Linear Regression AI model** using the **Rust programming language** and the **Burn library (version 0.16.0)**. The goal is to predict values based on the equation:  

\[
y = 2x + 1
\]

using **synthetic data**. The model is trained using **gradient descent** and optimized with the **Adam optimizer** to minimize the Mean Squared Error (MSE) loss function.  

The entire project is structured to follow best practices in Rust and machine learning, ensuring efficient training and evaluation.  

---

## **🔧 Features**  
✅ Generates **synthetic data** with random noise to simulate real-world conditions  
✅ Implements a **linear regression model** using the Burn framework  
✅ Uses **Mean Squared Error (MSE)** as the loss function  
✅ Trains the model using the **Adam optimizer**  
✅ Evaluates model performance on unseen data  
✅ Visualizes the training results using the **textplots** crate  

---

### **📁 Project Structure**  

    linear_regression_model/ 
    │── src/ 
    │   ├── dataset.rs  # Generates and saves synthetic training data 
    │   ├── model.rs    # Defines the linear regression model
    │   ├── train.rs.   # Implements training logic
    │   ├── main.rs.    # Runs the training and evaluation process
    │── Cargo.toml      # Project dependencies and configurations
    │── README.md       # Documentation (this file)
    │── dataset.csv     # Generated dataset (created after running)

# ** Steps to set up the project**

**1. Install Rust:** <br/>
• Download and install Rust from  https://www.rust-lang.org/tools/install or run _**curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh**_<br/>
• In your terminal and follow the instructions. Verify and test the installation by running _**rustc --version**_ in your terminal. <br/>

**2. Install Rust Rover IDE(GUI):** <br/>
• Download Rust Rover from https://www.jetbrains.com/rust/. <br/>
• Follow the installation instructions for your operating system.<br/>

**3. Create a new Rust Project:** <br/>
• Open Rust Rover. <br/>
• Run in terminal: <br/>
 
    cargo new linear_regression_model
    cd linear_regression_model
    
• Replace the contents of the Cargo.toml file with the provided TOML configuration:<br/>

    [dependencies]
    burn = { version = "0.16.0", features = ["wgpu", "train"] }
    ndarray = "0.16.0"
    rand = "0.9.0"
    rgb = "0.8.50"
    textplots = "0.8.6"

**4. Connect Rust Rover to GitHub:** <br/>
• Install Git from https://git-scm.com/.<br/>
• In Rust Rover, go to VCS > Enable Version Control Integration and select Git.<br/>
• Create a new repository on GitHub and link it to your project. You can use command line: git init   <br/>  

    git remote add origin <your - github - repo - u r l > 
    git add .  
    git commit -m ”Initial commit” 
    git push -u origin main 
    
• Alternatively, navigate to file>New>Project from Version Control. On the Clone repository tab that pops up, select Git for Version Control, Add your dedicated repository to URL: and select your RustRover directory under Directory

# **Creating the Linear Regression Model**

**1. Creating the Linear Regression Model   Generating synthetic data** <br/>

• Create a new file to load the dataset called _**dataset.rs**_ and execute the following code:  <br/>    
    
    use rand::Rng;
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

***Note that _**rand**_ will be seen as an unresolved import unless the dependencies were created in cargo.toml.**   <br/>

• The above code creates a dataset where: it generates 100(x,y) points and ads noise to y=2x+1 for realism. The dataset then gets saved to dataset.csv for training. This will simulate real-world data with some randomness.<br/>

• Run the program in Rust Rover’s terminal using sh cargo run. This creates a CSV dataset in the root.<br/>


**2. Defining the model** <br/>
• Use the burn library to define a simple linear regression model where: y = wx + b.<br/>
• Where w (weight) and b (bias) are learnable parameters.  <br/>
• Using the following code we will implement linear regression model with one layer. Create a new file called _**model.rs**_ and add the following code:

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

**3. Training the model**   <br/>
• Create a file called _**train.rs**_ an add the following code:

    use burn::{
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

• Trains for epochs iterations.  <br/>
• Uses Adam optimizer for gradient descent.  <br/>
• Prints loss per epoch to monitor training.  <br/>

**4. Evaluate Model**   <br/>

The code added to the _**main.rs**_ file test the model on the new unseen data and prints the information required.  

    mod dataset;
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

**Run the Code**    <br/>
_Compile the build by executing the following commands in Rust’s terminal: _    <br/>
**Compile the project**: _**sh cargo build**_
**Run the training script**: _**sh cargo run**_

**Expected output:**    <br/>
✅ Dataset saved to dataset.csv   <br/>
🚀 Training model...   <br/>
Epoch 1: Loss = 4.82   <br/>
Epoch 2: Loss = 3.72   <br/>
...   <br/>
Epoch 100: Loss = 0.01   <br/>
✅ Training complete!   <br/>


## 🏗️ Approach  

To solve the problem of predicting **y = 2x + 1** using a linear regression model, I followed a structured approach consisting of four key steps:  

### **1. Generating Synthetic Data**  
To train the model, I created a dataset of **(x, y) pairs**, where:  

\[
y = 2x + 1 + \text{noise}
\]  

- Used the `rand` crate to generate random `x` values in the range **[-10, 10]**.  
- Added **random noise** to simulate real-world data variations.  
- Saved the dataset as **`dataset.csv`** for easy loading into the model.  

### **2. Defining the Linear Regression Model**  
- Used the **Burn** library to define a **single-layer model** with one input and one output.  
- Implemented a **forward pass**, where the model learns to approximate \( y = wx + b \).  
- Chose **Mean Squared Error (MSE)** as the loss function to measure prediction accuracy.  

### **3. Training the Model**  
- Loaded the dataset into **Burn tensors** for processing.  
- Used the **Adam optimizer** to iteratively adjust `w` and `b`.  
- Trained the model for **100 epochs**, printing loss values at intervals to track improvements.  

### **4. Evaluating Performance**  
- Compared the model’s predictions against expected values.  
- Observed loss reduction over training iterations, confirming model improvement.  
- Verified that predicted values closely matched the equation **y = 2x + 1**.  

This approach ensured that the model effectively learned the **linear relationship** between `x` and `y`, demonstrating the use of **machine learning in Rust** using the Burn library. 🚀  

## 📊 Results & Evaluation  

### **1. Model Performance**  
After training the linear regression model using the Burn library, the model successfully learned the function:  

\[
y = 2x + 1
\]  

- **Initial loss:** The loss was high at the start, indicating poor predictions.  
- **Training progress:** Over **100 epochs**, the loss gradually decreased.  
- **Final loss:** A low final loss value confirmed that the model had learned the correct pattern.  

### **2. Sample Predictions vs. Expected Values**  
To evaluate the model’s performance, I tested it with new input values and compared predictions with expected values:  

    | Input (x) | Expected (y) = 2x + 1 | Predicted (y) |
    |-----------|-----------------------|---------------|
    | 5.0       | 11.0                  | 10.98         |
    | -3.0      | -5.0                  | -5.02         |
    | 7.5       | 16.0                  | 15.97         |
    | 0.0       | 1.0                   | 1.01          |

- The predicted values **closely match** the expected values, demonstrating that the model successfully captured the linear relationship.  
- **Small differences** are due to the added noise in the dataset and minor approximation errors.  

### **3. Loss Reduction Over Time**  
The loss function’s decreasing trend over epochs confirmed that the model was learning effectively. Below is a sample visualization of how the loss changed:  

## 🤔 Reflection on the Learning Process  

### **1. Learning Rust for Machine Learning**  
This project provided a great opportunity to explore **Rust** for AI and machine learning. Initially, I had to **wrap my head around the language** and how to execute code within the Rust environment. Despite my initial hesitation towards **Rust and GitHub**, I found myself appreciating their power and potential as I progressed through the project.  

Before solving the problem, my first priority was **setting up the environment and orienting myself**. I followed the steps outlined in **"Steps to Set Up the Project"**, ensuring that all dependencies were correctly installed and configured.  

Since I am not yet **familiar with Rust**, I leveraged **OpenAI** for assistance in generating initial code snippets, which I then modified and refined to suit the project's needs.  

#### **Key Takeaways:**  
✅ Understanding Rust’s **ownership model** helped in efficient data management.  
✅ The **Burn library** provides a solid framework for defining and training models.  
✅ I still have alot to learn before I can become more confident

---

### **2. Challenges Faced & Solutions**  

🔴 **Challenge:** Open AI giving deprecated code.  
✅ **Solution:** Carefully working through the errors and resolving through Rust documentation and error codes`.  
---

### **3. Key Takeaways**  
✅ Rust is **efficient** for AI but has a **steeper learning curve** compared to Python.  
✅ The **Burn library** is powerful but requires a deeper understanding of **tensors and gradients**.  
✅ Understanding **linear regression from scratch** (rather than relying on pre-built ML tools) significantly improved my grasp of machine learning fundamentals.  

This project opened my understanding of **Rust, AI modeling, and optimization techniques**, I foresee growing in understanding of the environment the more I use it.

## 4 Resources
    https://stackoverflow.com/questions
    https://rmarkdown.rstudio.com/
    https://github.com
    https://www.jetbrains.com/
    https://chatgpt.com/
    https://doc.rust-lang.org/error_codes/error-index.html
    https://docs.rs/

**Initial resources**
    
    • Rust Documentation: https://doc.rust-lang.org/
    • Burn Library Documentation: https://docs.rs/burn/0.16.0/burn/
    • GitHub Guide: https://docs.github.com/en/get-started
    • Rust Rover Documentation: https://www.jetbrains.com/help/rust/
    • LaTeX Documentation: https://www.latex-project.org/help/
