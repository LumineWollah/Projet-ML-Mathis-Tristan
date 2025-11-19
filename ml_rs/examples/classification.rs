use ml_rs::perceptron::{Perceptron, PerceptronOptions};
use ml_rs::MatF;

fn main() {
    let x_data: MatF = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y_data = vec![-1.0, 1.0, 1.0, 1.0];

    let mut classifier = Perceptron::new(PerceptronOptions {
        learning_rate: 1.0,
        max_epochs: 50,
        shuffle: true,
        seed: 123,
    });

    classifier.fit(&x_data, &y_data);
    println!("Weights (bias first): {:?}", classifier.weights_with_bias());
    let predictions = classifier.predict(&x_data);
    println!("Predictions: {:?}", predictions);
}
