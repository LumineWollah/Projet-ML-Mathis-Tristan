mod linear_perceptron;
mod naive_multi_layer_perceptron;

use linear_perceptron::LinearPerceptron;
use naive_multi_layer_perceptron::MyMLP;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{SystemTime, UNIX_EPOCH};
use std::env;

fn mod_0_5_python_style(x: f64) -> f64 {
    let m = x % 0.5;
    if m < 0.0 { m + 0.5 } else { m }
}

fn generate_test_2_dataset() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Time-based seed → different dataset each run
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut rng = StdRng::seed_from_u64(seed);

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    // First cluster: around (1,1), label +1
    for _ in 0..50 {
        let x = rng.gen::<f64>() * 0.9 + 1.0;
        let y = rng.gen::<f64>() * 0.9 + 1.0;
        inputs.push(vec![x, y]);
        outputs.push(vec![1.0]);
    }

    // Second cluster: around (2,2), label -1
    for _ in 0..50 {
        let x = rng.gen::<f64>() * 0.9 + 2.0;
        let y = rng.gen::<f64>() * 0.9 + 2.0;
        inputs.push(vec![x, y]);
        outputs.push(vec![-1.0]);
    }

    (inputs, outputs)
}

fn generate_test_4_dataset() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Time-based seed → new dataset every run
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut rng = StdRng::seed_from_u64(seed);

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for _ in 0..500 {
        // X in [-1, +1]
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        let y = rng.gen::<f64>() * 2.0 - 1.0;

        let inside_vertical_stripe = x.abs() <= 0.3;
        let inside_horizontal_stripe = y.abs() <= 0.3;

        let label = if inside_vertical_stripe || inside_horizontal_stripe {
            1.0
        } else {
            -1.0
        };

        inputs.push(vec![x, y]);
        outputs.push(vec![label]);
    }

    (inputs, outputs)
}

fn generate_test_5_dataset() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Time-based seed → new dataset every run
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut rng = StdRng::seed_from_u64(seed);

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for _ in 0..500 {
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        let y = rng.gen::<f64>() * 2.0 - 1.0;

        let v1 = -x - y - 0.5;
        let v2 =  x - y - 0.5;

        let label = if v1 > 0.0 && y < 0.0 && v2 < 0.0 {
            Some(vec![ 1.0, -1.0, -1.0])
        } else if v1 < 0.0 && y > 0.0 && v2 < 0.0 {
            Some(vec![-1.0,  1.0, -1.0])
        } else if v1 < 0.0 && y < 0.0 && v2 > 0.0 {
            Some(vec![-1.0, -1.0,  1.0])
        } else {
            None
        };

        if let Some(lbl) = label {
            inputs.push(vec![x, y]);
            outputs.push(lbl);
        }
    }

    (inputs, outputs)
}

fn generate_test_6_dataset() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    // Time-based seed → new dataset each run
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    let mut rng = StdRng::seed_from_u64(seed);

    let mut inputs = Vec::with_capacity(1000);
    let mut outputs = Vec::with_capacity(1000);

    for _ in 0..1000 {
        // X in [-1, 1]^2
        let x = rng.gen::<f64>() * 2.0 - 1.0;
        let y = rng.gen::<f64>() * 2.0 - 1.0;

        // Python-style remainder in [0, 0.5)
        let xm = mod_0_5_python_style(x).abs();
        let ym = mod_0_5_python_style(y).abs();

        let label = if xm <= 0.25 && ym > 0.25 {
            vec![ 1.0, -1.0, -1.0]
        } else if xm > 0.25 && ym <= 0.25 {
            vec![-1.0,  1.0, -1.0]
        } else {
            vec![-1.0, -1.0,  1.0]
        };

        inputs.push(vec![x, y]);
        outputs.push(label);
    }

    (inputs, outputs)
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    if (args.contains(&"-c".to_string()) || args.contains(&"--classification".to_string())) && args.contains(&"--linear".to_string()) {
        println!("Running linear classification tests...");
        run_linear_classification_tests();
        return;
    }

    if (args.contains(&"-c".to_string()) || args.contains(&"--classification".to_string())) && args.contains(&"--mlp".to_string()) {
        println!("Running MLP classification tests...");
        run_mlp_classification_tests();
        return;
    }

    if (args.contains(&"-r".to_string()) || args.contains(&"--regression".to_string())) && args.contains(&"--linear".to_string()) {
        println!("Running linear regression tests...");
        run_linear_regression_tests();
        return;
    }

    if (args.contains(&"-r".to_string()) || args.contains(&"--regression".to_string())) && args.contains(&"--mlp".to_string()) {
        println!("Running MLP regression tests...");
        run_mlp_regression_tests();
        return;
    }

    // Default behaviour
    println!("No mode specified. Use --classification or --regression.");
}

fn run_linear_classification_tests() {
    use linear_perceptron::LinearPerceptron;

    // #### LINEAR CLASSIFICATION ####

    // ## Test 1: Linear Simple (OK)
    println!("\n=== Linear Test 1: Linear Simple ===\n");
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 3.0],
        vec![3.0, 3.0],
    ];

    let outputs = vec![
        vec![1.0],
        vec![-1.0],
        vec![-1.0],
    ];

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("\nTraining...");
    lin.train_classification(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let raw = lin.predict_raw(x);
        let p = lin.predict_class(x);
        println!("x={:?}, y={}, raw={:.2}, pred_sign={:.2}", x, y[0], raw, p);
    }

    // ## Test 2: Linear Multiple (OK)
    println!("\n=== Linear Test 2: Linear Multiple ===\n");
    let (inputs, outputs) = generate_test_2_dataset();

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("Training...");
    lin.train_classification(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for i in 0..10 {
        let idx = i * 10;
        let raw = lin.predict_raw(&inputs[idx]);
        let p = lin.predict_class(&inputs[idx]);
        println!(
            "x={:?}, y={}, raw={:.2}, pred_sign={:.2}",
            inputs[idx], outputs[idx][0], raw, p
        );
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..100 {
    //     let raw = lin.predict_raw(&inputs[i]);
    //     let p = lin.predict_class(&inputs[i]);
    //     if (p >= 0.0 && outputs[i][0] == 1.0) || (p < 0.0 && outputs[i][0] == -1.0) {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);

    // ## Test 3: XOR (KO)
    println!("\n=== Linear Test 3: XOR ===\n");
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let outputs = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 500_000;
    let alpha = 0.1;

    println!("\nTraining...");
    lin.train_classification(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let raw = lin.predict_raw(x);
        let p = lin.predict_class(x);
        println!("x={:?}, y={}, raw={:.2}", x, y[0], raw);
    }

    // ## Test 4: Cross (KO)
    println!("\n=== Linear Test 4: Cross ===\n");
    let (inputs, outputs) = generate_test_4_dataset();

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 500_000;
    let alpha = 0.05;

    println!("Training...");
    lin.train_classification(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for i in 0..50 {
        let idx = i * 10;
        let raw = lin.predict_raw(&inputs[idx]);
        let p = lin.predict_class(&inputs[idx]);
        println!(
            "x={:?}, y={}, raw={:.2}, pred_sign={:.2}",
            inputs[idx], outputs[idx][0], raw, p
        );
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..500 {
    //     let raw = lin.predict_raw(&inputs[i]);
    //     let p = lin.predict_class(&inputs[i]);
    //     if (p >= 0.0 && outputs[i][0] == 1.0) || (p < 0.0 && outputs[i][0] == -1.0) {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);

    // ## Test 5: Three Classes (OK with one-vs-all linear perceptrons)
    println!("\n=== Linear Test 5: Three Classes ===\n");
    let (inputs, outputs) = generate_test_5_dataset();

    // One-vs-all: 3 separate linear perceptrons, each sees +/-1 for its own class
    let mut lin1 = LinearPerceptron::new(2);
    let mut lin2 = LinearPerceptron::new(2);
    let mut lin3 = LinearPerceptron::new(2);

    // Build outputs for each classifier
    let mut outputs1 = Vec::with_capacity(outputs.len());
    let mut outputs2 = Vec::with_capacity(outputs.len());
    let mut outputs3 = Vec::with_capacity(outputs.len());
    for y in &outputs {
        outputs1.push(vec![y[0]]);
        outputs2.push(vec![y[1]]);
        outputs3.push(vec![y[2]]);
    }

    let num_iter = 500_000;
    let alpha = 0.05;

    println!("Training...");
    lin1.train_classification(&inputs, &outputs1, num_iter, alpha);
    lin2.train_classification(&inputs, &outputs2, num_iter, alpha);
    lin3.train_classification(&inputs, &outputs3, num_iter, alpha);

    println!("\nResults:");
    for i in 0..30 {
        let idx = i * 10;
        let p = vec![
            lin1.predict_class(&inputs[idx]),
            lin2.predict_class(&inputs[idx]),
            lin3.predict_class(&inputs[idx]),
        ];
        println!(
            "x={:?}, y={:?}, pred={:.2?}",
            inputs[idx], outputs[idx], p
        );
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..inputs.len() {
    //     let p = vec![
    //         lin1.predict_class(&inputs[i]),
    //         lin2.predict_class(&inputs[i]),
    //         lin3.predict_class(&inputs[i]),
    //     ];

    //     let predicted_class = if p[0] > p[1] && p[0] > p[2] {
    //         vec![1.0, -1.0, -1.0]
    //     } else if p[1] > p[0] && p[1] > p[2] {
    //         vec![-1.0, 1.0, -1.0]
    //     } else {
    //         vec![-1.0, -1.0, 1.0]
    //     };

    //     if predicted_class == outputs[i] {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);

    // ## Test 6: Multi Cross (KO)
    println!("\n=== Linear Test 6: Multi Cross ===\n");
    let (inputs, outputs) = generate_test_6_dataset();

    let mut lin1 = LinearPerceptron::new(2);
    let mut lin2 = LinearPerceptron::new(2);
    let mut lin3 = LinearPerceptron::new(2);

    let mut outputs1 = Vec::with_capacity(outputs.len());
    let mut outputs2 = Vec::with_capacity(outputs.len());
    let mut outputs3 = Vec::with_capacity(outputs.len());
    for y in &outputs {
        outputs1.push(vec![y[0]]);
        outputs2.push(vec![y[1]]);
        outputs3.push(vec![y[2]]);
    }

    let num_iter = 10_000_000;
    let alpha = 0.005;

    println!("Training...");
    lin1.train_classification(&inputs, &outputs1, num_iter, alpha);
    lin2.train_classification(&inputs, &outputs2, num_iter, alpha);
    lin3.train_classification(&inputs, &outputs3, num_iter, alpha);

    println!("\nResults:");
    for i in 0..30 {
        let idx = i * 10;
        let p = vec![
            lin1.predict_class(&inputs[idx]),
            lin2.predict_class(&inputs[idx]),
            lin3.predict_class(&inputs[idx]),
        ];
        println!(
            "x={:?}, y={:?}, pred={:.2?}",
            inputs[idx], outputs[idx], p
        );
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..inputs.len() {
    //     let p = vec![
    //         lin1.predict_class(&inputs[i]),
    //         lin2.predict_class(&inputs[i]),
    //         lin3.predict_class(&inputs[i]),
    //     ];

    //     let predicted_class = if p[0] > p[1] && p[0] > p[2] {
    //         vec![1.0, -1.0, -1.0]
    //     } else if p[1] > p[0] && p[1] > p[2] {
    //         vec![-1.0, 1.0, -1.0]
    //     } else {
    //         vec![-1.0, -1.0, 1.0]
    //     };

    //     if predicted_class == outputs[i] {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);
}

fn run_linear_regression_tests() {
    use linear_perceptron::LinearPerceptron;

    // #### LINEAR REGRESSION ####

    // Test 1: Linear Simple 2D (OK)
    println!("\n=== Test 1: Linear Simple 2D ===\n");
    let inputs = vec![
        vec![1.0],
        vec![2.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![4.0],
    ];

    let mut lin = LinearPerceptron::new(1);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("\nTraining...");
    lin.train_regression(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = lin.predict_regression(x);
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 2: Non-Linear Simple 2D (KO)
    println!("\n=== Test 2: Non-Linear Simple 2D ===\n");
    let inputs = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![3.0],
        vec![2.5],
    ];

    let mut lin = LinearPerceptron::new(1);

    let num_iter = 100_000;
    let alpha = 0.05;

    println!("\nTraining...");
    lin.train_regression(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = lin.predict_regression(x);
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 3: Linear Simple 3D (OK)
    println!("\n=== Test 3: Linear Simple 3D ===\n");
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![3.0],
        vec![2.5],
    ];

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("\nTraining...");
    lin.train_regression(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = lin.predict_regression(x);
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 4: Linear Tricky 3D (OK)
    println!("\n=== Test 4: Linear Tricky 3D ===\n");
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];

    let outputs = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 100_000;
    let alpha = 0.1;

    println!("\nTraining...");
    lin.train_regression(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = lin.predict_regression(x);
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 5: Non-Linear Simple 3D (KO)
    println!("\n=== Test 5: Non-Linear Simple 3D ===\n");
    let inputs = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![1.0],
        vec![-2.0],
        vec![-1.0],
    ];

    let mut lin = LinearPerceptron::new(2);

    let num_iter = 200_000;
    let alpha = 0.01;

    println!("\nTraining...");
    lin.train_regression(&inputs, &outputs, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = lin.predict_regression(x);
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }
}

fn run_mlp_classification_tests() {
    // #### CLASSIFICATION ####

    // ## Test 1: Linear Simple
    println!("\n=== Test 1: Linear Simple ===\n");
    let inputs = vec![
            vec![1.0, 1.0],
            vec![2.0, 3.0],
            vec![3.0, 3.0],
    ];
    
    let outputs = vec![
        vec![1.0],
        vec![-1.0],
        vec![-1.0],
    ];

    let mut mlp = MyMLP::new(&[2, 1]); // input, hidden, output

    let num_iter = 50_000; // iterations
    let alpha = 0.1; // learning rate

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, true, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, true)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // ## Test 2: Linear Multiple
    println!("\n=== Test 2: Linear Multiple ===\n");
    let (inputs, outputs) = generate_test_2_dataset();

    let mut mlp = MyMLP::new(&[2, 1]);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("Training...");
    mlp.train(&inputs, &outputs, true, num_iter, alpha);

    println!("\nResults:");
    for i in 0..10 {
        let p = mlp.predict(&inputs[i*10], true)[0];
        println!("x={:?}, y={}, pred={:.2}", inputs[i*10], outputs[i*10][0], p);
    }

    // ## Test 3: XOR
    println!("\n=== Test 3: XOR ===\n");
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let outputs = vec![
        vec![0.0],
        vec![1.0],
        vec![1.0],
        vec![0.0],
    ];

    let mut mlp = MyMLP::new(&[2, 2, 1]);

    let num_iter = 500_000;
    let alpha = 0.1;

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, true, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, true)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // ## Test 4: Cross
    println!("\n=== Test 4: Cross ===\n");
    let (inputs, outputs) = generate_test_4_dataset();

    let mut mlp = MyMLP::new(&[2, 4, 1]);

    let num_iter = 500_000;
    let alpha = 0.05;

    println!("Training...");
    mlp.train(&inputs, &outputs, true, num_iter, alpha);

    println!("\nResults:");
    for i in 0..50 {
        let p = mlp.predict(&inputs[i*10], true)[0];
        println!("x={:?}, y={}, pred={:.2}", inputs[i*10], outputs[i*10][0], p);
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..500 {
    //     let p = mlp.predict(&inputs[i], true)[0];
    //     if (p >= 0.0 && outputs[i][0] == 1.0) || (p < 0.0 && outputs[i][0] == -1.0) {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);

    // ## Test 5: Three Classes
    println!("\n=== Test 5: Three Classes ===\n");
    let (inputs, outputs) = generate_test_5_dataset();
    let mut mlp = MyMLP::new(&[2, 3]);

    let num_iter = 500_000;
    let alpha = 0.05;

    println!("Training...");
    mlp.train(&inputs, &outputs, true, num_iter, alpha);

    println!("\nResults:");
    for i in 0..30 {
        let p = mlp.predict(&inputs[i*10], true);
        println!("x={:?}, y={:?}, pred={:.2?}", inputs[i*10], outputs[i*10], p);
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..inputs.len() {
    //     let p = mlp.predict(&inputs[i], true);
    //     let predicted_class = if p[0] > p[1] && p[0] > p[2] {
    //         vec![1.0, -1.0, -1.0]
    //     } else if p[1] > p[0] && p[1] > p[2] {
    //         vec![-1.0, 1.0, -1.0]
    //     } else {
    //         vec![-1.0, -1.0, 1.0]
    //     };

    //     if predicted_class == outputs[i] {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);

    // ## Test 6: Multi Cross
    println!("\n=== Test 6: Multi Cross ===\n");
    let (inputs, outputs) = generate_test_6_dataset();
    let mut mlp = MyMLP::new(&[2, 16, 16, 3]);

    let num_iter = 10_000_000;
    let alpha = 0.005;

    println!("Training...");
    mlp.train(&inputs, &outputs, true, num_iter, alpha);

    println!("\nResults:");
    for i in 0..30 {
        let p = mlp.predict(&inputs[i*10], true);
        println!("x={:?}, y={:?}, pred={:.2?}", inputs[i*10], outputs[i*10], p);
    }

    // println!("\nPrediction comparison:");
    // let mut correct = 0;
    // let mut incorrect = 0;
    // for i in 0..inputs.len() {
    //     let p = mlp.predict(&inputs[i], true);
    //     let predicted_class = if p[0] > p[1] && p[0] > p[2] {
    //         vec![1.0, -1.0, -1.0]
    //     } else if p[1] > p[0] && p[1] > p[2] {
    //         vec![-1.0, 1.0, -1.0]
    //     } else {
    //         vec![-1.0, -1.0, 1.0]
    //     };

    //     if predicted_class == outputs[i] {
    //         correct += 1;
    //     } else {
    //         incorrect += 1;
    //     }
    // }
    // println!("Correct: {}, Incorrect: {}", correct, incorrect);
}

fn run_mlp_regression_tests() {
    // #### REGRESSION ####
    // Test 1: Linear Simple 2D
    println!("\n=== Regression Test 1: Linear Simple 2D ===\n");
    let inputs = vec![
        vec![1.0],
        vec![2.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![4.0],
    ];

    let mut mlp = MyMLP::new(&[1, 1]);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, false, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, false)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 2: Non-Linear Simple 2D
    println!("\n=== Regression Test 2: Non-Linear Simple 2D ===\n");
    let inputs = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![3.0],
        vec![2.5],
    ];

    let mut mlp = MyMLP::new(&[1, 3, 1]);

    let num_iter = 100_000;
    let alpha = 0.05;

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, false, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, false)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 3: Linear Simple 3D
    println!("\n=== Regression Test 3: Linear Simple 3D ===\n");
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 1.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![3.0],
        vec![2.5],
    ];

    let mut mlp = MyMLP::new(&[2, 1]);

    let num_iter = 50_000;
    let alpha = 0.1;

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, false, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, false)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 4: Linear Tricky 3D
    println!("\n=== Regression Test 4: Linear Tricky 3D ===\n");
    let inputs = vec![
        vec![1.0, 1.0],
        vec![2.0, 2.0],
        vec![3.0, 3.0],
    ];

    let outputs = vec![
        vec![1.0],
        vec![2.0],
        vec![3.0],
    ];

    let mut mlp = MyMLP::new(&[2, 1]);

    let num_iter = 100_000;
    let alpha = 0.1;

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, false, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, false)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }

    // Test 5: Non-Linear Simple 3D
    println!("\n=== Regression Test 5: Non-Linear Simple 3D ===\n");
    let inputs = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
    ];

    let outputs = vec![
        vec![2.0],
        vec![1.0],
        vec![-2.0],
        vec![-1.0],
    ];

    let mut mlp = MyMLP::new(&[2, 2, 1]);

    let num_iter = 200_000;
    let alpha = 0.01;

    println!("\nTraining...");
    mlp.train(&inputs, &outputs, false, num_iter, alpha);

    println!("\nResults:");
    for (x, y) in inputs.iter().zip(outputs.iter()) {
        let p = mlp.predict(x, false)[0];
        println!("x={:?}, y={}, pred={:.2}", x, y[0], p);
    }
}
