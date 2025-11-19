pub mod naive_multi_layer_perceptron;
pub mod linear_perceptron;

use naive_multi_layer_perceptron::MyMLP;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::sync::{Mutex, OnceLock};

/// Global model, trained lazily the first time `predict_move` is called.
static MODEL: OnceLock<Mutex<MyMLP>> = OnceLock::new();

/// Path to your dataset file (relative to the process working directory).
const DATASET_PATH: &str = "dataset.csv";

/// Number of input features:
///  - 6 rows * 7 columns * 3 (one-hot per cell)
///  - +3 for current player one-hot
const INPUT_DIM: usize = 6 * 7 * 3 + 3;

/// Number of possible moves (columns)
const OUTPUT_DIM: usize = 7;

/// Load the Connect 4 dataset from CSV.
///
/// Each sample is 7 non-empty lines:
///  - 6 lines for the board (each line: 7 cells * 3 values)
///  - 1 line: [p0, p1, p2, col]
/// Where (p0,p1,p2) is one-hot for the player who played the move,
/// and `col` is a column index 0..6.
///
/// We build:
///   X: board (6x7x3) flatté + [p0,p1,p2]
///   Y: one-hot sur 7 colonnes, dans {-1, +1}
fn load_connect4_dataset(path: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let file = File::open(path)
        .unwrap_or_else(|e| panic!("Cannot open dataset file '{}': {}", path, e));
    let reader = BufReader::new(file);

    let mut inputs = Vec::<Vec<f64>>::new();
    let mut outputs = Vec::<Vec<f64>>::new();

    let mut block: Vec<String> = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line from dataset");
        let trimmed = line.trim().to_string();

        if trimmed.is_empty() {
            if !block.is_empty() {
                parse_sample_block(&block, &mut inputs, &mut outputs);
                block.clear();
            }
        } else {
            block.push(trimmed);
        }
    }

    if !block.is_empty() {
        parse_sample_block(&block, &mut inputs, &mut outputs);
    }

    (inputs, outputs)
}

/// Parse one sample block (7 lines) and push into inputs/outputs.
/// Silently ignores malformed blocks.
fn parse_sample_block(lines: &[String], inputs: &mut Vec<Vec<f64>>, outputs: &mut Vec<Vec<f64>>) {
    if lines.len() != 7 {
        // Expect 6 board rows + 1 metadata row
        return;
    }

    // --- Board (6 rows) ---
    let mut board_vals: Vec<f64> = Vec::with_capacity(6 * 7 * 3);
    for row in &lines[0..6] {
        for tok in row.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let v: f64 = match t.parse() {
                Ok(v) => v,
                Err(_) => return, // malformed, skip sample
            };
            board_vals.push(v);
        }
    }

    // Sanity: we expect exactly 6*7*3 = 126 values
    if board_vals.len() != 6 * 7 * 3 {
        return;
    }

    // --- Metadata row ---
    let meta = &lines[6];
    let meta_vals: Vec<f64> = meta
        .split(',')
        .filter_map(|t| t.trim().parse::<f64>().ok())
        .collect();

    if meta_vals.len() != 4 {
        return;
    }

    let p0 = meta_vals[0];
    let p1 = meta_vals[1];
    let p2 = meta_vals[2];
    let col_idx = meta_vals[3] as usize;

    if col_idx >= OUTPUT_DIM {
        return;
    }

    // Build input: board + player one-hot
    let mut x = Vec::with_capacity(INPUT_DIM);
    x.extend(board_vals);
    x.push(p0);
    x.push(p1);
    x.push(p2);

    // Build output: one-hot on 7 columns, in {-1, +1}
    let mut y = vec![-1.0; OUTPUT_DIM];
    y[col_idx] = 1.0;

    inputs.push(x);
    outputs.push(y);
}

/// Train the MLP for Connect 4 move prediction.
fn init_model() -> MyMLP {
    println!("Loading Connect 4 dataset from '{}'", DATASET_PATH);
    let (inputs, outputs) = load_connect4_dataset(DATASET_PATH);
    println!("Loaded {} samples", inputs.len());

    if inputs.is_empty() {
        panic!("Dataset is empty; cannot train model");
    }

    // Simple architecture: 129 -> 64 -> 64 -> 7
    let mut mlp = MyMLP::new(&[INPUT_DIM, 64, 64, OUTPUT_DIM]);

    // Hyperparamètres à ajuster si besoin
    let num_iter = 200_000;
    let alpha = 0.01;

    println!("Training MLP for {} iterations...", num_iter);
    mlp.train(&inputs, &outputs, true, num_iter, alpha);
    println!("Training finished.");

    mlp
}

/// Get a reference to the global trained model (lazy init).
fn get_model() -> &'static Mutex<MyMLP> {
    MODEL.get_or_init(|| Mutex::new(init_model()))
}

/// FFI entry point called from C#.
///
/// Signature C#:
///   [DllImport("ml_rs", CallingConvention = CallingConvention.Cdecl)]
///   static extern int predict_move(double[] input, int input_len);
///
/// - `input` is a flattened feature vector of length INPUT_DIM
///   (exactement comme construit dans EncodeBoard côté C#).
/// - Return value is a column index in [0, 6].
#[no_mangle]
pub extern "C" fn predict_move(input_ptr: *const f64, len: usize) -> i32 {
    // Basic safety checks
    if input_ptr.is_null() || len != INPUT_DIM {
        // Valeur de repli : colonne 0
        return 0;
    }

    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, len) };

    let model_mutex = get_model();
    let mut mlp = model_mutex.lock().expect("Failed to lock model mutex");

    // Forward pass
    let output = mlp.predict(input_slice, true); // Vec<f64> de taille 7

    // Argmax over outputs
    let mut best_idx = 0;
    let mut best_val = output[0];
    for (i, &v) in output.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }

    best_idx as i32
}
