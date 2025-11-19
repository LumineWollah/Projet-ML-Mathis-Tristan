use rand::Rng;

pub struct MyMLP {
    /// neurons per layer (input included)
    d: Vec<usize>,
    /// number of weight layers (L = d.len() - 1)
    L: usize,
    /// W[l][i][j]
    /// - l = layer index (1..=L), l=0 unused
    /// - i = neuron index in previous layer (0..=d[l-1])  (0 = bias)
    /// - j = neuron index in this layer (0..=d[l])         (0 = bias, always 0 weight)
    pub(crate) W: Vec<Vec<Vec<f64>>>,
    /// X[l][j] = activation of neuron j in layer l (j=0 is bias = 1.0)
    pub(crate) X: Vec<Vec<f64>>,
    /// deltas[l][j] = backpropagation error for neuron j in layer l
    pub(crate) deltas: Vec<Vec<f64>>,
}

impl MyMLP {
    /// npl: neurons per layer (input included)
    pub fn new(npl: &[usize]) -> Self {
        assert!(npl.len() >= 2, "Need at least input and output layers");
        let d = npl.to_vec();
        let L = d.len() - 1;

        let mut rng = rand::thread_rng();

        // Initialize weights
        let mut W: Vec<Vec<Vec<f64>>> = Vec::with_capacity(d.len());
        for l in 0..d.len() {
            if l == 0 {
                // No weights going *into* layer 0
                W.push(Vec::new());
                continue;
            }

            let mut layer_w: Vec<Vec<f64>> = Vec::with_capacity(d[l - 1] + 1);
            for _i in 0..=d[l - 1] {
                let mut row: Vec<f64> = Vec::with_capacity(d[l] + 1);
                for j in 0..=d[l] {
                    let val = if j == 0 {
                        0.0
                    } else {
                        rng.gen_range(-1.0..1.0)
                    };
                    row.push(val);
                }
                layer_w.push(row);
            }
            W.push(layer_w);
        }

        // Initialize activations X and deltas
        let mut X: Vec<Vec<f64>> = Vec::with_capacity(d.len());
        let mut deltas: Vec<Vec<f64>> = Vec::with_capacity(d.len());

        for l in 0..d.len() {
            let mut x_layer = Vec::with_capacity(d[l] + 1);
            let mut delta_layer = Vec::with_capacity(d[l] + 1);

            for j in 0..=d[l] {
                x_layer.push(if j == 0 { 1.0 } else { 0.0 });
                delta_layer.push(0.0);
            }

            X.push(x_layer);
            deltas.push(delta_layer);
        }

        MyMLP { d, L, W, X, deltas }
    }

    fn propagate(&mut self, inputs: &[f64], is_classification: bool) {
        assert_eq!(
            inputs.len(),
            self.d[0],
            "Input size must match number of input neurons"
        );

        // Copy inputs into X[0][1..]
        for (j, &val) in inputs.iter().enumerate() {
            self.X[0][j + 1] = val;
        }

        // Forward pass
        for l in 1..=self.L {
            for j in 1..=self.d[l] {
                let mut signal = 0.0;
                for i in 0..=self.d[l - 1] {
                    signal += self.W[l][i][j] * self.X[l - 1][i];
                }

                let mut x = signal;
                if is_classification || l != self.L {
                    x = signal.tanh();
                }
                self.X[l][j] = x;
            }
        }
    }

    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.X[self.L][1..=self.d[self.L]].to_vec()
    }

    pub fn train(
        &mut self,
        all_samples_inputs: &[Vec<f64>],
        all_samples_expected_outputs: &[Vec<f64>],
        is_classification: bool,
        num_iter: usize,
        alpha: f64,
    ) {
        assert_eq!(
            all_samples_inputs.len(),
            all_samples_expected_outputs.len()
        );

        let mut rng = rand::thread_rng();

        for it in 0..num_iter {
            let k = rng.gen_range(0..all_samples_inputs.len());
            let inputs_k = &all_samples_inputs[k];
            let expected_outputs_k = &all_samples_expected_outputs[k];

            self.propagate(inputs_k, is_classification);

            // Output layer deltas
            for j in 1..=self.d[self.L] {
                let mut delta = self.X[self.L][j] - expected_outputs_k[j - 1];
                if is_classification {
                    delta *= 1.0 - self.X[self.L][j].powi(2);
                }
                self.deltas[self.L][j] = delta;
            }

            // Hidden layers
            if self.L >= 2 {
                for l in (2..=self.L).rev() {
                    for i in 1..=self.d[l - 1] {
                        let mut total = 0.0;
                        for j in 1..=self.d[l] {
                            total += self.W[l][i][j] * self.deltas[l][j];
                        }
                        total *= 1.0 - self.X[l - 1][i].powi(2);
                        self.deltas[l - 1][i] = total;
                    }
                }
            }

            // Update weights
            for l in 1..=self.L {
                for i in 0..=self.d[l - 1] {
                    for j in 1..=self.d[l] {
                        self.W[l][i][j] -= alpha * self.X[l - 1][i] * self.deltas[l][j];
                    }
                }
            }

            if (it + 1) % (num_iter / 10).max(1) == 0 {
                println!("Iteration {}/{}", it + 1, num_iter);
            }
        }
    }
}
