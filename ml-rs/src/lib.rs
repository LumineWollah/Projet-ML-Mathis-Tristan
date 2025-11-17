//! Minimal ML library in Rust (snake_case).
//! Implements:
//! - Perceptron (Rosenblatt)
//! - Linear regression (delta rule)
//! Handles bias term automatically by adding 1.0 at start of feature vector.

pub mod perceptron;
pub mod linear_regressor;

pub type VecF = Vec<f64>;
pub type MatF = Vec<Vec<f64>>;

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "dot: size mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn axpy(alpha: f64, x_vec: &[f64], y_vec: &mut [f64]) {
    assert_eq!(x_vec.len(), y_vec.len(), "axpy: size mismatch");
    for (y_i, x_i) in y_vec.iter_mut().zip(x_vec.iter()) {
        *y_i += alpha * x_i;
    }
}

pub fn add_bias_column(x_data: &MatF) -> MatF {
    x_data
        .iter()
        .map(|row| {
            let mut new_row = Vec::with_capacity(row.len() + 1);
            new_row.push(1.0);
            new_row.extend_from_slice(row);
            new_row
        })
        .collect()
}

pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    assert_eq!(y_true.len(), y_pred.len(), "mse: size mismatch");
    let n = y_true.len() as f64;
    let sum_sq: f64 = y_true
        .iter()
        .zip(y_pred)
        .map(|(t, p)| {
            let e = p - t;
            e * e
        })
        .sum();
    sum_sq / n
}
