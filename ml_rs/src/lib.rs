pub mod naive_multi_layer_perceptron;
pub mod linear_perceptron;

use naive_multi_layer_perceptron::MyMLP;

use std::ffi::CStr;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::raw::c_char;

// -------------------- Training dataset --------------------

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

fn parse_sample_block(lines: &[String], inputs: &mut Vec<Vec<f64>>, outputs: &mut Vec<Vec<f64>>) {
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

    let num_iter = 200_000;
    let alpha = 0.01;

    println!("Training MLP for {} iterations...", num_iter);
    mlp.train(&inputs, &outputs, true, num_iter, alpha);
    println!("Training finished.");

    mlp
}

// -------------------- Text model format --------------------
//
// File format (UTF-8):
//   Line 1: MLRSMLP_TXT1
//   Line 2: d0,d1,d2,...,dL   (layer sizes, input included)
//   Remaining lines: weights as f64, one per line, in this order:
//     for l=1..=L:
//       for i=0..=d[l-1]:
//         for j=0..=d[l]:
//           W[l][i][j]
//
// This format is intentionally simple and stable.

const TXT_MAGIC: &str = "MLRSMLP_TXT1";

fn expected_weight_count(d: &[usize]) -> usize {
    if d.len() < 2 {
        return 0;
    }
    let L = d.len() - 1;
    let mut count = 0usize;
    for l in 1..=L {
        count += (d[l - 1] + 1) * (d[l] + 1);
    }
    count
}

fn serialize_mlp_text(mlp: &MyMLP) -> String {
    let mut s = String::new();
    s.push_str(TXT_MAGIC);
    s.push('\n');

    // layer sizes
    for (k, sz) in mlp.d.iter().enumerate() {
        if k > 0 {
            s.push(',');
        }
        s.push_str(&sz.to_string());
    }
    s.push('\n');

    // weights
    for l in 1..=mlp.L {
        for i in 0..=mlp.d[l - 1] {
            for j in 0..=mlp.d[l] {
                s.push_str(&format!("{:.17}\n", mlp.W[l][i][j]));
            }
        }
    }

    s
}

fn deserialize_mlp_text(text: &str) -> Option<MyMLP> {
    let mut lines = text.lines();

    let magic = lines.next()?.trim();
    if magic != TXT_MAGIC {
        return None;
    }

    let dims_line = lines.next()?.trim();
    if dims_line.is_empty() {
        return None;
    }

    let mut d: Vec<usize> = Vec::new();
    for tok in dims_line.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            return None;
        }
        let sz: usize = t.parse().ok()?;
        if sz == 0 {
            return None;
        }
        d.push(sz);
    }
    if d.len() < 2 {
        return None;
    }

    let expected = expected_weight_count(&d);

    let mut weights: Vec<f64> = Vec::with_capacity(expected);
    for line in lines {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: f64 = t.parse().ok()?;
        weights.push(v);
        if weights.len() == expected {
            break;
        }
    }

    if weights.len() != expected {
        return None;
    }

    let mut mlp = MyMLP::new(&d);

    let mut idx = 0usize;
    for l in 1..=mlp.L {
        for i in 0..=mlp.d[l - 1] {
            for j in 0..=mlp.d[l] {
                mlp.W[l][i][j] = weights[idx];
                idx += 1;
            }
        }
    }

    Some(mlp)
}

fn cstr_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(ptr) }.to_str().ok()?;
    Some(s.to_string())
}

// -------------------- FFI: create / load / save / destroy --------------------

#[no_mangle]
pub extern "C" fn create_ai() -> *mut MyMLP {
    let mlp = init_model();
    Box::into_raw(Box::new(mlp))
}

/// Load a model from a TEXT file.
/// Returns null on failure.
#[no_mangle]
pub extern "C" fn load_ai_text(path: *const c_char) -> *mut MyMLP {
    let path = match cstr_to_string(path) {
        Some(p) => p,
        None => return std::ptr::null_mut(),
    };

    let mut f = match File::open(&path) {
        Ok(f) => f,
        Err(_) => return std::ptr::null_mut(),
    };

    let mut text = String::new();
    if f.read_to_string(&mut text).is_err() {
        return std::ptr::null_mut();
    }

    let mlp = match deserialize_mlp_text(&text) {
        Some(m) => m,
        None => return std::ptr::null_mut(),
    };

    Box::into_raw(Box::new(mlp))
}

/// Save a model to a TEXT file.
/// Returns 1 on success, 0 on failure.
#[no_mangle]
pub extern "C" fn save_ai_text(ai: *mut MyMLP, path: *const c_char) -> i32 {
    if ai.is_null() {
        return 0;
    }

    let path = match cstr_to_string(path) {
        Some(p) => p,
        None => return 0,
    };

    let mlp: &MyMLP = unsafe { &*ai };
    let text = serialize_mlp_text(mlp);

    let mut f = match File::create(&path) {
        Ok(f) => f,
        Err(_) => return 0,
    };

    match f.write_all(text.as_bytes()) {
        Ok(_) => 1,
        Err(_) => 0,
    }
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

// -------------------- FFI: inference --------------------

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

    let output = mlp.predict(input_slice, true);
    if output.len() != OUTPUT_DIM {
        return 0;
    }

    out_slice.copy_from_slice(&output);
    1
}
