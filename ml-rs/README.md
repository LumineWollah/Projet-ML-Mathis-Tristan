# ml-rs (snake_case version)

Rust ML library implementing:
- Perceptron (Rosenblatt)
- Linear regression (delta rule / Widrow–Hoff)

All variables are in `snake_case` for idiomatic Rust style.

## Build
```bash
cargo build --release
```

## Run examples
```bash
cargo run --release --example classification
cargo run --release --example regression
```
