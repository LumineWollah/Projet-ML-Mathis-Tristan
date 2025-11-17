use ml_rs::linear_regressor::{LinearRegressor, LinearRegressorOptions};
use ml_rs::{MatF, mse};

fn main() {
    let x_data: MatF = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 2.0],
    ];
    let y_data = vec![1.0, 3.0, -2.0, -2.0, 2.0, 1.0];

    let mut regressor = LinearRegressor::new(LinearRegressorOptions {
        learning_rate: 0.05,
        max_epochs: 500,
        shuffle: true,
        seed: 123,
    });

    regressor.fit(&x_data, &y_data);
    println!("Weights (bias, w1, w2): {:?}", regressor.weights_with_bias());
    let y_pred = regressor.predict(&x_data);
    println!("MSE: {}", mse(&y_data, &y_pred));
    println!("Pred(1.0, 0.5) = {}", regressor.predict_one(&[1.0, 0.5]));
}
