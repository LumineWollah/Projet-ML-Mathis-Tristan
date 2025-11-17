use crate::{MatF, VecF, dot, axpy, add_bias_column};
use rand::{seq::SliceRandom, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub struct LinearRegressorOptions {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for LinearRegressorOptions {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_epochs: 1000,
            shuffle: true,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinearRegressor {
    options: LinearRegressorOptions,
    weights: VecF,
}

impl LinearRegressor {
    pub fn new(options: LinearRegressorOptions) -> Self {
        Self { options, weights: Vec::new() }
    }

    pub fn fit(&mut self, x_data_raw: &MatF, y_data: &[f64]) {
        assert!(!x_data_raw.is_empty(), "LinearRegressor::fit: x_data empty");
        let sample_count = x_data_raw.len();
        let feature_count = x_data_raw[0].len();
        assert_eq!(y_data.len(), sample_count, "LinearRegressor::fit: label size mismatch");

        let x_data = add_bias_column(x_data_raw);
        self.weights = vec![0.0; feature_count + 1];

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.options.seed);

        for _epoch in 0..self.options.max_epochs {
            let mut order: Vec<usize> = (0..sample_count).collect();
            if self.options.shuffle {
                order.shuffle(&mut rng);
            }

            for &idx in &order {
                let x_i = &x_data[idx];
                let y_i = y_data[idx];
                let y_pred = dot(&self.weights, x_i);
                let error = y_i - y_pred;
                let alpha = self.options.learning_rate * error;
                axpy(alpha, x_i, &mut self.weights);
            }
        }
    }

    /// Train and record mean squared error after each epoch.
    pub fn fit_with_history(&mut self, x_data_raw: &MatF, y_data: &[f64]) -> Vec<f64> {
        assert!(!x_data_raw.is_empty(), "LinearRegressor::fit_with_history: x_data empty");
        let sample_count = x_data_raw.len();
        let feature_count = x_data_raw[0].len();
        assert_eq!(y_data.len(), sample_count, "LinearRegressor::fit_with_history: label size mismatch");

        let x_data = add_bias_column(x_data_raw);
        self.weights = vec![0.0; feature_count + 1];

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.options.seed);
        let mut history = Vec::with_capacity(self.options.max_epochs);

        for _epoch in 0..self.options.max_epochs {
            let mut order: Vec<usize> = (0..sample_count).collect();
            if self.options.shuffle {
                order.shuffle(&mut rng);
            }

            for &idx in &order {
                let x_i = &x_data[idx];
                let y_i = y_data[idx];
                let y_pred = dot(&self.weights, x_i);
                let error = y_i - y_pred;
                let alpha = self.options.learning_rate * error;
                axpy(alpha, x_i, &mut self.weights);
            }

            let preds: Vec<f64> = x_data.iter().map(|r| dot(&self.weights, r)).collect();
            let mse = crate::mse(y_data, &preds);
            history.push(mse);
        }

        history
    }

    pub fn predict(&self, x_data_raw: &MatF) -> Vec<f64> {
        let x_data = add_bias_column(x_data_raw);
        x_data.iter().map(|x| dot(&self.weights, x)).collect()
    }

    pub fn weights_with_bias(&self) -> &[f64] {
        &self.weights
    }
}
