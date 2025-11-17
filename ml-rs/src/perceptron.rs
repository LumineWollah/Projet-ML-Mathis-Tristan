use crate::{MatF, VecF, dot, axpy, add_bias_column};
use rand::{seq::SliceRandom, SeedableRng};

#[derive(Clone, Copy, Debug)]
pub struct PerceptronOptions {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub shuffle: bool,
    pub seed: u64,
}

impl Default for PerceptronOptions {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            max_epochs: 100,
            shuffle: true,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Perceptron {
    options: PerceptronOptions,
    weights: VecF,
}

impl Perceptron {
    pub fn new(options: PerceptronOptions) -> Self {
        Self { options, weights: Vec::new() }
    }

    pub fn fit(&mut self, x_data_raw: &MatF, y_data: &[f64]) {
        assert!(!x_data_raw.is_empty(), "Perceptron::fit: x_data empty");
        let sample_count = x_data_raw.len();
        let feature_count = x_data_raw[0].len();
        assert_eq!(y_data.len(), sample_count, "Perceptron::fit: label size mismatch");

        let x_data = add_bias_column(x_data_raw);
        self.weights = vec![0.0; feature_count + 1];

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.options.seed);

        for _epoch in 0..self.options.max_epochs {
            let mut order: Vec<usize> = (0..sample_count).collect();
            if self.options.shuffle {
                order.shuffle(&mut rng);
            }

            let mut error_count = 0;
            for &idx in &order {
                let x_i = &x_data[idx];
                let y_i = y_data[idx];
                let activation = dot(&self.weights, x_i);
                let prediction = if activation >= 0.0 { 1.0 } else { -1.0 };
                if (prediction - y_i).abs() > f64::EPSILON {
                    let alpha = self.options.learning_rate * y_i;
                    axpy(alpha, x_i, &mut self.weights);
                    error_count += 1;
                }
            }

            if error_count == 0 {
                break;
            }
        }
    }

    /// Same as `fit`, but returns a vector of misclassifications per epoch.
    pub fn fit_with_history(&mut self, x_data_raw: &MatF, y_data: &[f64]) -> Vec<usize> {
        assert!(!x_data_raw.is_empty(), "Perceptron::fit_with_history: x_data empty");
        let sample_count = x_data_raw.len();
        let feature_count = x_data_raw[0].len();
        assert_eq!(y_data.len(), sample_count, "Perceptron::fit_with_history: label size mismatch");

        let x_data = add_bias_column(x_data_raw);
        self.weights = vec![0.0; feature_count + 1];

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.options.seed);
        let mut history = Vec::with_capacity(self.options.max_epochs);

        for _epoch in 0..self.options.max_epochs {
            let mut order: Vec<usize> = (0..sample_count).collect();
            if self.options.shuffle {
                order.shuffle(&mut rng);
            }

            let mut error_count = 0usize;
            for &idx in &order {
                let x_i = &x_data[idx];
                let y_i = y_data[idx];
                let activation = dot(&self.weights, x_i);
                let prediction = if activation >= 0.0 { 1.0 } else { -1.0 };
                if (prediction - y_i).abs() > f64::EPSILON {
                    let alpha = self.options.learning_rate * y_i;
                    axpy(alpha, x_i, &mut self.weights);
                    error_count += 1;
                }
            }
            history.push(error_count);
            if error_count == 0 {
                break;
            }
        }

        history
    }

    pub fn predict(&self, x_data_raw: &MatF) -> Vec<f64> {
        let x_data = add_bias_column(x_data_raw);
        x_data
            .iter()
            .map(|x| if dot(&self.weights, x) >= 0.0 { 1.0 } else { -1.0 })
            .collect()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn weights_with_bias(&self) -> &[f64] {
        &self.weights
    }
}
