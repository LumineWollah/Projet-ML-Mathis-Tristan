use rand::Rng;

/// Simple linear model / perceptron:
/// y_hat = w · x + b
///
/// - For classification: interpret `y_hat.signum()` as the predicted class
/// - For regression: use `y_hat` directly
pub struct LinearPerceptron {
    weights: Vec<f64>,
    bias: f64,
    input_dim: usize,
}

impl LinearPerceptron {
    /// Create a new linear perceptron with `input_dim` inputs
    /// Weights are initialized randomly in [-0.5, 0.5], bias = 0
    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            weights.push(rng.gen_range(-0.5..0.5));
        }

        LinearPerceptron {
            weights,
            bias: 0.0,
            input_dim,
        }
    }

    /// Raw linear output: w · x + b
    pub fn predict_raw(&self, input: &[f64]) -> f64 {
        assert_eq!(input.len(), self.input_dim);
        let mut sum = self.bias;
        for (w, &x) in self.weights.iter().zip(input.iter()) {
            sum += w * x;
        }
        sum
    }

    /// For classification: return sign of the raw output (in {-1.0, 0.0, 1.0})
    pub fn predict_class(&self, input: &[f64]) -> f64 {
        self.predict_raw(input).signum()
    }

    /// For regression: just return the raw linear output.
    pub fn predict_regression(&self, input: &[f64]) -> f64 {
        self.predict_raw(input)
    }

    /// Train using simple SGD on squared error for *classification* targets
    ///
    /// - `inputs`: Vec of samples, each sample is a Vec<f64> of length `input_dim`
    /// - `outputs`: Vec of targets, each is Vec<f64> but only `outputs[k][0]` is used
    /// - `num_iter`: number of SGD steps
    /// - `alpha`: learning rate
    ///
    /// Targets can be in {-1, 1}, {0, 1}, or any real values; we just minimize MSE
    pub fn train_classification(
        &mut self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        num_iter: usize,
        alpha: f64,
    ) {
        assert_eq!(inputs.len(), outputs.len());
        let mut rng = rand::thread_rng();

        for _ in 0..num_iter {
            let k = rng.gen_range(0..inputs.len());
            let x = &inputs[k];
            let y = outputs[k][0];

            // prediction
            let y_hat = self.predict_raw(x);

            // gradient of 0.5 * (y_hat - y)^2 wrt w_i: (y_hat - y) * x_i
            let error = y_hat - y;

            // update weights and bias
            for i in 0..self.input_dim {
                self.weights[i] -= alpha * error * x[i];
            }
            self.bias -= alpha * error;
        }
    }

    /// Train using simple SGD on squared error for *regression* targets
    pub fn train_regression(
        &mut self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        num_iter: usize,
        alpha: f64,
    ) {
        assert_eq!(inputs.len(), outputs.len());
        let mut rng = rand::thread_rng();

        for _ in 0..num_iter {
            let k = rng.gen_range(0..inputs.len());
            let x = &inputs[k];
            let y = outputs[k][0];

            let y_hat = self.predict_raw(x);
            let error = y_hat - y;

            for i in 0..self.input_dim {
                self.weights[i] -= alpha * error * x[i];
            }
            self.bias -= alpha * error;
        }
    }
}
