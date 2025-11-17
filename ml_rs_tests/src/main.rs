use plotters::prelude::*;
use ml_rs::perceptron::{Perceptron, PerceptronOptions};
use ml_rs::linear_regressor::{LinearRegressor, LinearRegressorOptions};
use ml_rs::{MatF, mse};
use std::fs;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::time::{SystemTime, UNIX_EPOCH};

fn to_matf_2d(d: &[[f64; 2]]) -> MatF { d.iter().map(|r| r.to_vec()).collect() }

/// Time-based seed 
fn now_seed() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64
}

/// Fixed-grid curve:
/// - X axis: 0..max_epochs (full training span, even if converged earlier)
/// - Y axis: provided as [y_min, y_max] (0..1 for perceptron error rate; 0..initial MSE for regression)
fn draw_curve_fixed(
    path: &str,
    title: &str,
    ys: &[f64],
    max_epochs: usize,
    y_min: f64,
    y_max: f64,
    xlab: &str,
    ylab: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..max_epochs as f64, y_min..y_max)?;

    chart.configure_mesh().x_desc(xlab).y_desc(ylab).draw()?;

    // Plot points for epochs 0..=ys.len()
    let xs: Vec<f64> = (0..=ys.len()).map(|i| i as f64).collect();
    chart.draw_series(LineSeries::new(xs.into_iter().zip(ys.iter().cloned()), &BLACK))?;

    root.present()?;
    Ok(())
}

fn draw_data(path: &str, x: &MatF, y: &[f64], w: &[f64], title: &str, _is_class: bool) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800,600)).into_drawing_area();
    root.fill(&WHITE)?;
    let (minx, maxx) = (x.iter().map(|r| r[0]).fold(f64::INFINITY,f64::min), x.iter().map(|r| r[0]).fold(f64::NEG_INFINITY,f64::max));
    let (miny, maxy) = (x.iter().map(|r| r[1]).fold(f64::INFINITY,f64::min), x.iter().map(|r| r[1]).fold(f64::NEG_INFINITY,f64::max));
    let mut chart = ChartBuilder::on(&root).caption(title,("sans-serif",24))
        .margin(20).x_label_area_size(40).y_label_area_size(40)
        .build_cartesian_2d((minx-0.5)..(maxx+0.5),(miny-0.5)..(maxy+0.5))?;
    chart.configure_mesh().x_desc("x1").y_desc("x2").draw()?;
    for (i,row) in x.iter().enumerate() {
        let pt = (row[0], row[1]);
        if y[i]>0.0 { chart.draw_series(std::iter::once(Circle::new(pt,6,BLUE.filled())))?; }
        else { chart.draw_series(std::iter::once(TriangleMarker::new(pt,8,RED.filled())))?; } }
    if w.len()>=3 && w[2].abs()>1e-12 {
        let w0=w[0]; let w1=w[1]; let w2=w[2];
        let xa=minx-0.5; let xb=maxx+0.5;
        let ya=-(w0+w1*xa)/w2; let yb=-(w0+w1*xb)/w2;
        chart.draw_series(LineSeries::new(vec![(xa,ya),(xb,yb)],&BLACK))?; }
    root.present()?; Ok(()) 
}

fn draw_regression_1d(
    path: &str,
    x: &MatF,         
    y: &[f64],
    w: &[f64],        // [bias, w1]
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 500)).into_drawing_area();
    root.fill(&WHITE)?;

    let (xmin, xmax) = (
        x.iter().map(|r| r[0]).fold(f64::INFINITY, f64::min),
        x.iter().map(|r| r[0]).fold(f64::NEG_INFINITY, f64::max),
    );
    let (ymin_pts, ymax_pts) = (
        y.iter().cloned().fold(f64::INFINITY, f64::min),
        y.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );

    let pad_x = (xmax - xmin).abs().max(1e-6) * 0.1;
    let xa = xmin - pad_x;
    let xb = xmax + pad_x;

    let (b, w1) = (w[0], w[1]);
    let ya = b + w1 * xa;
    let yb = b + w1 * xb;

    let ymin = ymin_pts.min(ya.min(yb));
    let ymax = ymax_pts.max(ya.max(yb));
    let pad_y = (ymax - ymin).abs().max(1e-6) * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d((xa)..(xb), (ymin - pad_y)..(ymax + pad_y))?;

    chart.configure_mesh().x_desc("x").y_desc("y / ŷ").draw()?;

    for (i, row) in x.iter().enumerate() {
        chart.draw_series(std::iter::once(Circle::new((row[0], y[i]), 5, BLUE.filled())))?;
    }

    chart.draw_series(LineSeries::new(vec![(xa, ya), (xb, yb)], &BLACK))?;

    root.present()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure images directory exists
    let _ = fs::create_dir_all("images");

    // === test_1_linear_simple ===
    println!("\n[TEST test_1_linear_simple] Perceptron");
    let x = to_matf_2d(&[[1.0,1.0],[2.0,3.0],[3.0,3.0]]);
    let y = vec![1.0,-1.0,-1.0];
    let max_epochs = 10;
    let mut model = Perceptron::new(PerceptronOptions{learning_rate:0.1,max_epochs:max_epochs,shuffle:true,seed: now_seed()});
    let history = model.fit_with_history(&x,&y);
    let preds = model.predict(&x);
    println!("epochs={}, weights={:?}, preds={:?}", history.len(), model.weights_with_bias(), preds);
    draw_data("images/test_1_linear_simple_data.png", &x, &y, model.weights_with_bias(), "test_1_linear_simple", true)?;
    let n = y.len() as f64;
    let ys_rate: Vec<f64> = history.iter().map(|&e| e as f64 / n).collect();
    draw_curve_fixed("images/test_1_linear_simple_curve.png","test_1_linear_simple — errors/epoch",&ys_rate,max_epochs,0.0,1.0,"epoch","error rate")?;


    // === test_2_linear_multiple ===
    println!("\n[TEST test_2_linear_multiple] Perceptron");
    let mut rng = StdRng::seed_from_u64(now_seed());
    let mut x: MatF = Vec::with_capacity(100);

    for _ in 0..50 {
        let x1 = rng.gen::<f64>() * 0.9 + 1.0;
        let x2 = rng.gen::<f64>() * 0.9 + 1.0;
        x.push(vec![x1, x2]);
    }
    for _ in 0..50 {
        let x1 = rng.gen::<f64>() * 0.9 + 2.0;
        let x2 = rng.gen::<f64>() * 0.9 + 2.0;
        x.push(vec![x1, x2]);
    }

    let mut y = vec![0.0; 100];
    for i in 0..50 { y[i] =  1.0; }
    for i in 50..100 { y[i] = -1.0; }

    let max_epochs = 50;
    let mut model = Perceptron::new(PerceptronOptions {
        learning_rate: 0.2,
        max_epochs,
        shuffle: true,
        seed: now_seed(),
    });

    let history = model.fit_with_history(&x, &y);
    let preds = model.predict(&x);
    println!("epochs={}, weights={:?}", history.len(), model.weights_with_bias());

    draw_data(
        "images/test_2_linear_multiple_data.png",
        &x,
        &y,
        model.weights_with_bias(),
        "test_2_linear_multiple",
        true,
    )?;

    let n = y.len() as f64;
    let ys_rate: Vec<f64> = history.iter().map(|&e| e as f64 / n).collect();

    draw_curve_fixed(
        "images/test_2_linear_multiple_curve.png","test_2_linear_multiple — error rate/epoch",
        &ys_rate,
        max_epochs,
        0.0,
        1.0,
        "epoch",
        "error rate",
    )?;

    // === test_3_xor ===
    println!("\n[TEST test_3_xor] Perceptron");
    let x = to_matf_2d(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]);
    let y = vec![1.0, 1.0, -1.0, -1.0];

    let max_epochs = 100;
    let mut model = Perceptron::new(PerceptronOptions {
        learning_rate: 0.2,
        max_epochs,
        shuffle: true,
        seed: now_seed(),
    });

    let history = model.fit_with_history(&x, &y);
    let preds = model.predict(&x);
    println!("epochs={}, weights={:?}, preds={:?}", history.len(), model.weights_with_bias(), preds);

    draw_data(
        "images/test_3_xor_data.png",
        &x,
        &y,
        model.weights_with_bias(),
        "test_3_xor",
        true,
    )?;

    let n = y.len() as f64;
    let ys_rate: Vec<f64> = history.iter().map(|&e| e as f64 / n).collect();
    draw_curve_fixed(
        "images/test_3_xor_curve.png",
        "test_3_xor — error rate/epoch",
        &ys_rate,
        max_epochs,
        0.0,
        1.0,
        "epoch",
        "error rate",
    )?;

    // === test_4_cross ===
    println!("\n[TEST test_4_cross] Perceptron");
    let mut rng = StdRng::seed_from_u64(now_seed());
    let mut x: MatF = Vec::with_capacity(500);
    let mut y: Vec<f64> = Vec::with_capacity(500);

    for _ in 0..500 {
        let x1 = rng.gen::<f64>() * 2.0 - 1.0;
        let x2 = rng.gen::<f64>() * 2.0 - 1.0;
        let label = if x1.abs() <= 0.3 || x2.abs() <= 0.3 { 1.0 } else { -1.0 };
        x.push(vec![x1, x2]);
        y.push(label);
    }

    let max_epochs = 200;
    let mut model = Perceptron::new(PerceptronOptions {
        learning_rate: 0.2,
        max_epochs,
        shuffle: true,
        seed: now_seed(),
    });

    let history = model.fit_with_history(&x, &y);
    let preds = model.predict(&x);
    println!("epochs={}, weights={:?}, preds(first 10)={:?}", history.len(), model.weights_with_bias(), &preds[..10]);

    draw_data(
        "images/test_4_cross_data.png",
        &x,
        &y,
        model.weights_with_bias(),
        "test_4_cross",
        true,
    )?;

    let n = y.len() as f64;
    let ys_rate: Vec<f64> = history.iter().map(|&e| e as f64 / n).collect();
    draw_curve_fixed(
        "images/test_4_cross_curve.png",
        "test_4_cross — error rate/epoch",
        &ys_rate,
        max_epochs,
        0.0,
        1.0,
        "epoch",
        "error rate",
    )?;

    // === test_5_multi_linear ===
    println!("\n[TEST test_5_multi_linear] Perceptron (multi-class one-vs-rest)");
    let mut rng = StdRng::seed_from_u64(now_seed());

    let mut x: MatF = Vec::with_capacity(500);
    let mut y_multi: Vec<[f64; 3]> = Vec::with_capacity(500);

    for _ in 0..500 {
        let x1 = rng.gen::<f64>() * 2.0 - 1.0;
        let x2 = rng.gen::<f64>() * 2.0 - 1.0;
        let cond_a = -x1 - x2 - 0.5 > 0.0 && x2 < 0.0 && x1 - x2 - 0.5 < 0.0;
        let cond_b = -x1 - x2 - 0.5 < 0.0 && x2 > 0.0 && x1 - x2 - 0.5 < 0.0;
        let cond_c = -x1 - x2 - 0.5 < 0.0 && x2 < 0.0 && x1 - x2 - 0.5 > 0.0;

        let label = if cond_a {
            [1.0, -1.0, -1.0]
        } else if cond_b {
            [-1.0, 1.0, -1.0]
        } else if cond_c {
            [-1.0, -1.0, 1.0]
        } else {
            [-1.0, -1.0, -1.0]
        };

        x.push(vec![x1, x2]);
        y_multi.push(label);
    }

    let mut x_filtered: MatF = Vec::new();
    let mut y_filtered: Vec<[f64; 3]> = Vec::new();
    for (i, label) in y_multi.iter().enumerate() {
        if *label != [-1.0, -1.0, -1.0] {
            x_filtered.push(x[i].clone());
            y_filtered.push(*label);
        }
    }

    let max_epochs = 100;
    let mut models: Vec<Perceptron> = Vec::new();
    let mut histories: Vec<Vec<usize>> = Vec::new();

    for class_idx in 0..3 {
        let y_class: Vec<f64> = y_filtered.iter().map(|lbl| lbl[class_idx]).collect();
        let mut model = Perceptron::new(PerceptronOptions {
            learning_rate: 0.2,
            max_epochs,
            shuffle: true,
            seed: now_seed() + class_idx as u64,
        });
        let history = model.fit_with_history(&x_filtered, &y_class);
        histories.push(history);
        models.push(model);
    }

    // Visualize data and class boundaries
    let root = BitMapBackend::new("images/test_5_multi_linear_data.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let (minx, maxx) = (
        x_filtered.iter().map(|r| r[0]).fold(f64::INFINITY, f64::min),
        x_filtered.iter().map(|r| r[0]).fold(f64::NEG_INFINITY, f64::max),
    );
    let (miny, maxy) = (
        x_filtered.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min),
        x_filtered.iter().map(|r| r[1]).fold(f64::NEG_INFINITY, f64::max),
    );
    let mut chart = ChartBuilder::on(&root)
        .caption("test_5_multi_linear", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d((minx - 0.5)..(maxx + 0.5), (miny - 0.5)..(maxy + 0.5))?;
    chart.configure_mesh().x_desc("x1").y_desc("x2").draw()?;

    for (i, row) in x_filtered.iter().enumerate() {
        let pt = (row[0], row[1]);
        let label = y_filtered[i];
        let color = if label[0] == 1.0 {
            &BLUE
        } else if label[1] == 1.0 {
            &RED
        } else {
            &GREEN
        };
        chart.draw_series(std::iter::once(Circle::new(pt, 4, color.filled())))?;
    }

    // Draw one line per perceptron (colored by class)
    let mut model_number = 0;
    for model in &models {
        let w = model.weights_with_bias();
        if w.len() >= 3 && w[2].abs() > 1e-12 {
            let xa = minx - 0.5;
            let xb = maxx + 0.5;
            let ya = -(w[0] + w[1] * xa) / w[2];
            let yb = -(w[0] + w[1] * xb) / w[2];
            if model_number == 0 {
                chart.draw_series(LineSeries::new(vec![(xa, ya), (xb, yb)], &BLUE))?;
            } else if model_number == 1 {
                chart.draw_series(LineSeries::new(vec![(xa, ya), (xb, yb)], &RED))?;
            } else {
                chart.draw_series(LineSeries::new(vec![(xa, ya), (xb, yb)], &GREEN))?;
            }
        }
        model_number += 1;
    }
    root.present()?;

    for (i, history) in histories.iter().enumerate() {
        let ys_rate: Vec<f64> = history.iter().map(|&e| e as f64 / y_filtered.len() as f64).collect();
        let path = format!("images/test_5_multi_linear_curve_class{}.png", i + 1);
        let title = format!("test_5_multi_linear — class {} error/epoch", i + 1);
        draw_curve_fixed(&path, &title, &ys_rate, max_epochs, 0.0, 1.0, "epoch", "error rate")?;
    }

    println!("Trained {} perceptrons (1 per class).", models.len());
        
    // === test_6_multi_cross ===
    // X = np.random.random((1000, 2)) * 2.0 - 1.0
    // Y = class by modulo stripes with period 0.5
    println!("\n[TEST test_6_multi_cross] Perceptron (multi-class one-vs-rest)");
    let mut rng = StdRng::seed_from_u64(now_seed());

    // --- generate data ---
    let mut x: MatF = Vec::with_capacity(1000);
    let mut y_multi: Vec<[f64; 3]> = Vec::with_capacity(1000);
    for _ in 0..1000 {
        let x1 = rng.gen::<f64>() * 2.0 - 1.0;
        let x2 = rng.gen::<f64>() * 2.0 - 1.0;

        let xm = (x1 % 0.5).abs();
        let ym = (x2 % 0.5).abs();

        // class rules:
        // [1,-1,-1] if |x%0.5| <= 0.25 and |y%0.5| > 0.25
        // [-1,1,-1] if |x%0.5| > 0.25 and |y%0.5| <= 0.25
        // [-1,-1,1] otherwise
        let label = if xm <= 0.25 && ym > 0.25 {
            [1.0, -1.0, -1.0]
        } else if xm > 0.25 && ym <= 0.25 {
            [-1.0, 1.0, -1.0]
        } else {
            [-1.0, -1.0, 1.0]
        };

        x.push(vec![x1, x2]);
        y_multi.push(label);
    }

    let x_filtered = x;
    let y_filtered = y_multi;

    let max_epochs = 1000;
    let mut models: Vec<Perceptron> = Vec::new();
    let mut histories: Vec<Vec<usize>> = Vec::new();

    for class_idx in 0..3 {
        let y_class: Vec<f64> = y_filtered.iter().map(|lbl| lbl[class_idx]).collect();
        let mut model = Perceptron::new(PerceptronOptions {
            learning_rate: 0.2,
            max_epochs,
            shuffle: true,
            seed: now_seed() + class_idx as u64,
        });
        let history = model.fit_with_history(&x_filtered, &y_class);
        histories.push(history);
        models.push(model);
    }

    let root = BitMapBackend::new("images/test_6_multi_cross_data.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let (minx, maxx) = (
        x_filtered.iter().map(|r| r[0]).fold(f64::INFINITY, f64::min),
        x_filtered.iter().map(|r| r[0]).fold(f64::NEG_INFINITY, f64::max),
    );
    let (miny, maxy) = (
        x_filtered.iter().map(|r| r[1]).fold(f64::INFINITY, f64::min),
        x_filtered.iter().map(|r| r[1]).fold(f64::NEG_INFINITY, f64::max),
    );
    let mut chart = ChartBuilder::on(&root)
        .caption("test_6_multi_cross", ("sans-serif", 24))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d((minx - 0.5)..(maxx + 0.5), (miny - 0.5)..(maxy + 0.5))?;
    chart.configure_mesh().x_desc("x1").y_desc("x2").draw()?;

    for (i, row) in x_filtered.iter().enumerate() {
        let pt = (row[0], row[1]);
        let label = y_filtered[i];
        let color = if label[0] == 1.0 { &BLUE } else if label[1] == 1.0 { &RED } else { &GREEN };
        chart.draw_series(std::iter::once(Circle::new(pt, 3, color.filled())))?;
    }

    for (k, model) in models.iter().enumerate() {
        let w = model.weights_with_bias();
        if w.len() >= 3 && w[2].abs() > 1e-12 {
            let xa = minx - 0.5;
            let xb = maxx + 0.5;
            let ya = -(w[0] + w[1] * xa) / w[2];
            let yb = -(w[0] + w[1] * xb) / w[2];
            let stroke = match k { 0 => &BLUE, 1 => &RED, _ => &GREEN };
            chart.draw_series(LineSeries::new(vec![(xa, ya), (xb, yb)], stroke))?;
        }
    }
    root.present()?;

    for (i, history) in histories.iter().enumerate() {
        let ys_rate: Vec<f64> = history.iter().map(|&e| e as f64 / y_filtered.len() as f64).collect();
        let path = format!("images/test_6_multi_cross_curve_class{}.png", i + 1);
        let title = format!("test_6_multi_cross — class {} error/epoch", i + 1);
        draw_curve_fixed(&path, &title, &ys_rate, max_epochs, 0.0, 1.0, "epoch", "error rate")?;
    }
    println!("Trained {} perceptrons (OVA) for test_6_multi_cross.", models.len());

    // === test_7_regression_1d ===
    // X = [[1],[2]]
    // Y = [2, 3]
    println!("\n[TEST test_7_regression_1d] Linear regression (1D)");
    let x: MatF = vec![ vec![1.0], vec![2.0] ];
    let y: Vec<f64> = vec![2.0, 3.0];

    let max_epochs = 100;
    let mut model = LinearRegressor::new(LinearRegressorOptions{
        learning_rate: 0.3,
        max_epochs,
        shuffle: true,
        seed: now_seed(),
    });
    let history = model.fit_with_history(&x, &y);
    let w = model.weights_with_bias(); // [bias, w1]
    println!("epochs={}, final MSE={:.6}, weights(bias,w1)={:?}",
            history.len(), *history.last().unwrap_or(&f64::NAN), w);

    draw_regression_1d(
        "images/test_7_regression_1d_data.png",
        &x, &y, &w, "test_7_regression_1d"
    )?;

    let init_mse = *history.first().unwrap_or(&1.0);
    let y_max = init_mse.max(1.0e-6);
    draw_curve_fixed(
        "images/test_7_regression_1d_curve.png",
        "test_7_regression_1d — MSE/epoch",
        &history,
        max_epochs,
        0.0,
        y_max,
        "epoch",
        "MSE",
    )?;

    println!("\nAll tests finished."); Ok(()) 
}
