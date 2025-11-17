
use plotters::prelude::*;
use ml_rs::MatF;

pub fn draw_curve_fixed(
    path: &str,
    title: &str,
    ys: &[f64],
    max_epochs: usize,
    y_min: f64,
    y_max: f64,
    x_label: &str,
    y_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (900, 420)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 22))
        .margin(18)
        .x_label_area_size(40)
        .y_label_area_size(54)
        .build_cartesian_2d(0f64..max_epochs as f64, y_min..y_max)?;

    chart.configure_mesh()
        .x_desc(x_label).y_desc(y_label)
        .x_labels(10)
        .y_labels(6)
        .draw()?;

    let xs: Vec<f64> = (1..=ys.len()).map(|i| i as f64).collect();
    chart.draw_series(LineSeries::new(xs.into_iter().zip(ys.iter().cloned()), &BLACK))?;

    root.present()?;
    Ok(())
}

pub fn draw_classification_data(
    path: &str,
    x: &MatF,
    y: &[f64],
    weights: &[f64],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (900, 620)).into_drawing_area();
    root.fill(&WHITE)?;

    let (min_x, max_x) = extrema(x.iter().map(|r| r[0]));
    let (min_y, max_y) = extrema(x.iter().map(|r| r[1]));
    let (x0, x1) = pad(min_x, max_x, 0.2);
    let (y0, y1) = pad(min_y, max_y, 0.2);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(18)
        .x_label_area_size(40)
        .y_label_area_size(54)
        .build_cartesian_2d(x0..x1, y0..y1)?;

    chart.configure_mesh().x_desc("x1").y_desc("x2").draw()?;

    for (i, row) in x.iter().enumerate() {
        let pt = (row[0], row[1]);
        if y[i] > 0.0 {
            chart.draw_series(std::iter::once(Circle::new(pt, 6, BLUE.filled())))?;
        } else {
            chart.draw_series(std::iter::once(TriangleMarker::new(pt, 8, RED.filled())))?;
        }
    }

    if weights.len() >= 3 && weights[2].abs() > 1e-12 {
        let w0 = weights[0]; let w1 = weights[1]; let w2 = weights[2];
        let ya = -(w0 + w1 * x0) / w2;
        let yb = -(w0 + w1 * x1) / w2;
        chart.draw_series(LineSeries::new(vec![(x0, ya), (x1, yb)], &BLACK))?;
    }

    root.present()?;
    Ok(())
}

pub fn draw_regression_data(
    path: &str,
    x: &MatF,
    y_true: &[f64],
    weights: &[f64],
    title: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (900, 620)).into_drawing_area();
    root.fill(&WHITE)?;

    let x1s: Vec<f64> = x.iter().map(|r| r[0]).collect();
    let x2_mean: f64 = if x.is_empty() { 0.0 } else { x.iter().map(|r| r[1]).sum::<f64>() / (x.len() as f64) };

    let (min_x, max_x) = extrema(x1s.iter().copied());
    let (min_y, max_y) = extrema(y_true.iter().copied());
    let (x0, x1) = pad(min_x, max_x, 0.2);
    let (y0, y1) = pad(min_y, max_y, 0.2);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 24))
        .margin(18)
        .x_label_area_size(40)
        .y_label_area_size(54)
        .build_cartesian_2d(x0..x1, y0..y1)?;

    chart.configure_mesh().x_desc("x1").y_desc("y").draw()?;

    chart.draw_series(
        x.iter().zip(y_true.iter()).map(|(row, &yt)| Circle::new((row[0], yt), 5, GREEN.filled()))
    )?;

    if weights.len() >= 3 {
        let w0 = weights[0]; let w1 = weights[1]; let w2 = weights[2];
        let f = |x1: f64| w0 + w1 * x1 + w2 * x2_mean;
        chart.draw_series(LineSeries::new(vec![(x0, f(x0)), (x1, f(x1))], &BLACK))?;
    }

    root.present()?;
    Ok(())
}

fn extrema(mut it: impl Iterator<Item=f64>) -> (f64, f64) {
    let first = it.next().unwrap_or(0.0);
    let mut lo = first; let mut hi = first;
    for v in it { if v < lo { lo = v } if v > hi { hi = v } }
    if (hi - lo).abs() < 1e-12 { (lo - 1.0, hi + 1.0) } else { (lo, hi) }
}

fn pad(lo: f64, hi: f64, pct: f64) -> (f64, f64) {
    let w = hi - lo;
    (lo - pct*w, hi + pct*w)
}
