pub mod naive_multi_layer_perceptron;
pub mod linear_perceptron;

use naive_multi_layer_perceptron::MyMLP;
use std::fs::File;
use std::io::{BufRead, BufReader};

const DATASET_PATH: &str = "dataset.csv";

pub const INPUT_DIM: usize = 6 * 7 * 3 + 3;
pub const OUTPUT_DIM: usize = 7;

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

fn parse_sample_block(
    lines: &[String],
    inputs: &mut Vec<Vec<f64>>,
    outputs: &mut Vec<Vec<f64>>,
) {
    if lines.len() != 7 {
        return;
    }

    let mut board_vals: Vec<f64> = Vec::with_capacity(6 * 7 * 3);
    for row in &lines[0..6] {
        for tok in row.split(',') {
            let t = tok.trim();
            if t.is_empty() {
                continue;
            }
            let v: f64 = match t.parse() {
                Ok(v) => v,
                Err(_) => return,
            };
            board_vals.push(v);
        }
    }

    if board_vals.len() != 6 * 7 * 3 {
        return;
    }

    let meta = &lines[6];
    let mut meta_vals = Vec::<f64>::new();
    for tok in meta.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let v: f64 = match t.parse() {
            Ok(v) => v,
            Err(_) => return,
        };
        meta_vals.push(v);
    }

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

    let mut x = Vec::with_capacity(INPUT_DIM);
    x.extend(board_vals);
    x.push(p0);
    x.push(p1);
    x.push(p2);

    let mut y = vec![-1.0; OUTPUT_DIM];
    y[col_idx] = 1.0;

    inputs.push(x);
    outputs.push(y);
}

fn init_model() -> MyMLP {
    println!("Loading Connect 4 dataset from '{}'", DATASET_PATH);
    let (inputs, outputs) = load_connect4_dataset(DATASET_PATH);
    println!("Loaded {} samples", inputs.len());

    if inputs.is_empty() {
        panic!("Dataset is empty; cannot train model");
    }

    let mut mlp = MyMLP::new(&[INPUT_DIM, 64, 64, OUTPUT_DIM]);

    let num_iter = 1_000_000;
    let alpha = 0.01;

    println!("Training MLP for {} iterations...", num_iter);
    mlp.train(&inputs, &outputs, true, num_iter, alpha);
    println!("Training finished.");

    mlp
}

#[no_mangle]
pub extern "C" fn create_ai() -> *mut MyMLP {
    let mlp = init_model();
    Box::into_raw(Box::new(mlp))
}

#[no_mangle]
pub extern "C" fn destroy_ai(ptr: *mut MyMLP) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        Box::from_raw(ptr);
    }
}

/// Fill `out_scores` (length OUTPUT_DIM) with the 7 raw output values.
///
/// Returns 1 on success, 0 on error.
#[no_mangle]
pub extern "C" fn predict_scores(
    ai: *mut MyMLP,
    input_ptr: *const f64,
    input_len: usize,
    out_scores_ptr: *mut f64,
    out_scores_len: usize,
) -> i32 {
    if ai.is_null()
        || input_ptr.is_null()
        || out_scores_ptr.is_null()
        || input_len != INPUT_DIM
        || out_scores_len != OUTPUT_DIM
    {
        return 0;
    }

    let mlp: &mut MyMLP = unsafe { &mut *ai };
    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, input_len) };
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_scores_ptr, out_scores_len) };

    let output = mlp.predict(input_slice, true); // Vec<f64> size 7

    if output.len() != OUTPUT_DIM {
        return 0;
    }

    out_slice.copy_from_slice(&output);
    1
}

/// Kept for compatibility: returns argmax column (may be full).
#[no_mangle]
pub extern "C" fn predict_move(ai: *mut MyMLP, input_ptr: *const f64, len: usize) -> i32 {
    if ai.is_null() || input_ptr.is_null() || len != INPUT_DIM {
        return 0;
    }

    let mlp: &mut MyMLP = unsafe { &mut *ai };
    let input_slice = unsafe { std::slice::from_raw_parts(input_ptr, len) };

    let output = mlp.predict(input_slice, true);

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
