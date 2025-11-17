# ml_rs_viz_tests

Runs every test extracted from your original notebook `[Notebook] Cas de tests.ipynb`.

Each test:
- trains the corresponding model from your `ml-rs` library,
- prints a console summary,
- generates two PNGs:
  - `<test>_data.png` — data + model boundary/line
  - `<test>_curve.png` — training curve (errors or MSE)

## Usage
```bash
cd ml_rs_viz_tests
cargo run --release
```
