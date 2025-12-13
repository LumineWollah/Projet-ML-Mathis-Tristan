use rand::{seq::SliceRandom, thread_rng};

pub struct RbfNetwork {
    pub centers: Vec<Vec<f64>>, // μ_k
    pub weights: Vec<f64>,      // w_k
    pub gamma: f64,
}

impl RbfNetwork {
    pub fn new(centers: Vec<Vec<f64>>, gamma: f64) -> Self {
        let k = centers.len();
        Self {
            centers,
            weights: vec![0.0; k],
            gamma,
        }
    }

    fn rbf(&self, x: &[f64], c: &[f64]) -> f64 {
        let mut dist2 = 0.0;
        for i in 0..x.len() {
            let d = x[i] - c[i];
            dist2 += d * d;
        }
        (-self.gamma * dist2).exp()
    }

    fn phi(&self, x: &[f64]) -> Vec<f64> {
        self.centers.iter().map(|c| self.rbf(x, c)).collect()
    }

    pub fn predict_regression(&self, x: &[f64]) -> f64 {
        let phi = self.phi(x);
        phi.iter()
            .zip(self.weights.iter())
            .map(|(p, w)| p * w)
            .sum()
    }

    pub fn predict_classification(&self, x: &[f64]) -> i32 {
        let y = self.predict_regression(x);
        if y >= 0.0 { 1 } else { -1 }
    }

    /// Train using closed-form solution:
    /// W = (ΦᵀΦ)^-1 ΦᵀY
    pub fn train(&mut self, xs: &[Vec<f64>], ys: &[f64]) {
        let n = xs.len();
        let k = self.centers.len();

        // Φ matrix (n x k)
        let mut phi = vec![vec![0.0; k]; n];
        for i in 0..n {
            phi[i] = self.phi(&xs[i]);
        }

        // Compute ΦᵀΦ
        let mut a = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for t in 0..n {
                    a[i][j] += phi[t][i] * phi[t][j];
                }
            }
        }

        // Compute ΦᵀY
        let mut b = vec![0.0; k];
        for i in 0..k {
            for t in 0..n {
                b[i] += phi[t][i] * ys[t];
            }
        }

        self.weights = solve_linear_system(a, b);
    }
}

/// Simple Gaussian elimination (small K)
fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();

    for i in 0..n {
        // pivot
        let mut max_row = i;
        for r in i + 1..n {
            if a[r][i].abs() > a[max_row][i].abs() {
                max_row = r;
            }
        }
        a.swap(i, max_row);
        b.swap(i, max_row);

        let diag = a[i][i];
        if diag.abs() < 1e-12 {
            continue;
        }

        for j in i..n {
            a[i][j] /= diag;
        }
        b[i] /= diag;

        for r in 0..n {
            if r != i {
                let factor = a[r][i];
                for c in i..n {
                    a[r][c] -= factor * a[i][c];
                }
                b[r] -= factor * b[i];
            }
        }
    }

    b
}

/// Lloyd k-means (used for center selection)
pub fn kmeans(xs: &[Vec<f64>], k: usize, iters: usize) -> Vec<Vec<f64>> {
    let mut rng = thread_rng();
    let mut centers = xs.choose_multiple(&mut rng, k).cloned().collect::<Vec<_>>();

    for _ in 0..iters {
        let mut groups = vec![Vec::<&Vec<f64>>::new(); k];

        for x in xs {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for (i, c) in centers.iter().enumerate() {
                let mut d = 0.0;
                for j in 0..x.len() {
                    let dx = x[j] - c[j];
                    d += dx * dx;
                }
                if d < best_dist {
                    best_dist = d;
                    best = i;
                }
            }
            groups[best].push(x);
        }

        for i in 0..k {
            if groups[i].is_empty() {
                continue;
            }
            for d in 0..centers[i].len() {
                centers[i][d] =
                    groups[i].iter().map(|x| x[d]).sum::<f64>() / groups[i].len() as f64;
            }
        }
    }

    centers
}
