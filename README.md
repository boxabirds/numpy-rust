# NumPy Rust

A comprehensive Rust port of NumPy, built on the powerful `ndarray` ecosystem. This library provides a familiar NumPy-like interface for numerical computing in Rust, with strong type safety and blazing fast performance.

## Features

- **Array Creation**: Comprehensive array creation routines (`zeros`, `ones`, `arange`, `linspace`, `eye`, etc.)
- **Mathematical Functions**: Element-wise operations (trigonometry, exponentials, logarithms, etc.)
- **Linear Algebra**: Matrix operations, decompositions (SVD, QR, Cholesky), eigenvalues, and more
- **Statistics**: Mean, median, variance, standard deviation, percentiles, correlation, and more
- **Random Number Generation**: Multiple distributions (uniform, normal, exponential, beta, gamma, etc.)
- **FFT**: Fast Fourier Transform operations using `rustfft`
- **Sorting & Searching**: Efficient sorting, searching, and unique operations
- **Type Safety**: Leverage Rust's type system for safer numerical code
- **Performance**: Parallel operations using Rayon and optimized BLAS routines

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
numpy_rust = "0.1.0"
```

## Quick Start

```rust
use numpy_rust::prelude::*;

fn main() {
    // Create arrays
    let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let b = zeros::<f64>(vec![3, 3].into());
    let c = linspace(0.0, 10.0, 100).unwrap();

    // Mathematical operations
    let sin_values = sin(&a);
    let exp_values = exp(&a);

    // Statistics
    let mean_val = stats::mean(&a).unwrap();
    let std_val = stats::std(&a, 1).unwrap();

    println!("Mean: {}, Std: {}", mean_val, std_val);

    // Linear algebra
    let matrix = array![[1.0, 2.0], [3.0, 4.0]];
    let det_val = linalg::det(&matrix).unwrap();
    let inv_matrix = linalg::inv(&matrix).unwrap();

    println!("Determinant: {}", det_val);

    // Random numbers
    let random_normal = random::randn::<f64>(vec![5, 5].into());
    let random_uniform = random::uniform(0.0, 1.0, vec![10].into()).unwrap();

    // FFT
    let signal = array![1.0, 2.0, 1.0, -1.0, 1.5];
    let spectrum = fft::fft(&signal).unwrap();

    // Sorting
    let unsorted = array![3, 1, 4, 1, 5, 9, 2, 6];
    let sorted_arr = sorting::sorted(&unsorted);
    let indices = sorting::argsort(&unsorted);
}
```

## Examples

### Array Creation

```rust
use numpy_rust::prelude::*;

// Create arrays with specific values
let zeros = zeros::<f64>(vec![3, 4].into());
let ones = ones::<f64>(vec![2, 2].into());
let full = full(vec![3, 3].into(), 7.0);

// Create sequences
let range = arange(0.0, 10.0, 0.5).unwrap();
let linear = linspace(0.0, 1.0, 11).unwrap();
let log = logspace(0.0, 2.0, 10, 10.0).unwrap();

// Create special matrices
let identity = eye::<f64>(5);
let diagonal = diag(&array![1.0, 2.0, 3.0]);
```

### Mathematical Operations

```rust
use numpy_rust::prelude::*;

let x = linspace(0.0, 2.0 * std::f64::consts::PI, 100).unwrap();

// Trigonometric functions
let sin_x = sin(&x);
let cos_x = cos(&x);
let tan_x = tan(&x);

// Exponential and logarithmic
let exp_x = exp(&x);
let log_x = log(&x.mapv(|v| v + 1.0)); // log of (x + 1)

// Element-wise operations
let sqrt_x = sqrt(&x);
let squared = power(&x, &x);

// Aggregations
let sum_val = sum(&x);
let product_val = prod(&x);
```

### Linear Algebra

```rust
use numpy_rust::prelude::*;

// Matrix operations
let a = array![[1.0, 2.0], [3.0, 4.0]];
let b = array![[5.0, 6.0], [7.0, 8.0]];
let c = linalg::matmul(&a, &b).unwrap();

// Solve linear system Ax = b
let a = array![[3.0, 1.0], [1.0, 2.0]];
let b_vec = array![9.0, 8.0];
let x = linalg::solve(&a, &b_vec).unwrap();

// Eigenvalues and eigenvectors
let matrix = array![[1.0, 2.0], [2.0, 1.0]];
let (eigenvalues, eigenvectors) = linalg::eig(&matrix).unwrap();

// SVD decomposition
let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
let (u, s, vt) = linalg::svd(&a).unwrap();

// Matrix properties
let det_val = linalg::det(&matrix).unwrap();
let trace_val = linalg::trace(&matrix);
let rank = linalg::matrix_rank(&matrix, None).unwrap();
```

### Statistics

```rust
use numpy_rust::prelude::*;

let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

// Basic statistics
let mean_val = stats::mean(&data).unwrap();
let median_val = stats::median(&data).unwrap();
let variance = stats::var(&data, 1).unwrap();
let std_dev = stats::std(&data, 1).unwrap();

// Percentiles
let p25 = stats::percentile(&data, 25.0).unwrap();
let p50 = stats::percentile(&data, 50.0).unwrap();
let p75 = stats::percentile(&data, 75.0).unwrap();

// Min/Max
let min_val = stats::min(&data).unwrap();
let max_val = stats::max(&data).unwrap();
let (min_v, max_v) = stats::range(&data).unwrap();

// Correlation
let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
let corr = stats::corrcoef(&x, &y).unwrap();
```

### Random Number Generation

```rust
use numpy_rust::prelude::*;

// Uniform distribution [0, 1)
let uniform = random::rand::<f64>(vec![5, 5].into());

// Standard normal distribution
let normal = random::randn::<f64>(vec![100].into());

// Custom distributions
let custom_uniform = random::uniform(10.0, 20.0, vec![50].into()).unwrap();
let custom_normal = random::normal(100.0, 15.0, vec![1000].into()).unwrap();

// Other distributions
let exponential = random::exponential(1.5, vec![100].into()).unwrap();
let beta = random::beta(2.0, 5.0, vec![100].into()).unwrap();
let gamma = random::gamma(2.0, 2.0, vec![100].into()).unwrap();

// Integer random numbers
let randints = random::randint(0, 100, vec![20].into()).unwrap();

// Sampling
let population = array![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let sample = random::choice(&population, 5, false).unwrap();

// Permutations
let perm = random::permutation(10);
```

### FFT Operations

```rust
use numpy_rust::prelude::*;

// Create a signal
let t = linspace(0.0, 1.0, 100).unwrap();
let signal = t.mapv(|x| (2.0 * std::f64::consts::PI * 5.0 * x).sin());

// Compute FFT
let spectrum = fft::fft(&signal).unwrap();

// Compute inverse FFT
let recovered = fft::ifft(&spectrum).unwrap();

// Real FFT (more efficient for real signals)
let spectrum_real = fft::rfft(&signal).unwrap();

// Frequency bins
let freqs = fft::fftfreq(signal.len(), 1.0 / 100.0);

// Power spectral density
let psd = fft::psd(&signal).unwrap();

// FFT shift
let shifted = fft::fftshift(&spectrum);
```

### Sorting and Searching

```rust
use numpy_rust::prelude::*;

let data = array![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];

// Sort
let sorted_data = sorting::sorted(&data);

// Get sorting indices
let indices = sorting::argsort(&data);

// Unique values
let unique_vals = sorting::unique(&data);
let (values, counts) = sorting::unique_counts(&data);

// Find largest/smallest k elements
let top_3 = sorting::largest(&data.mapv(|x| x as f64), 3).unwrap();
let bottom_3 = sorting::smallest(&data.mapv(|x| x as f64), 3).unwrap();

// Search
let sorted = array![1.0, 2.0, 3.0, 5.0, 8.0];
let idx = sorting::searchsorted(&sorted, 4.0, true).unwrap();

// Find non-zero indices
let sparse = array![0.0, 1.0, 0.0, 3.0, 0.0, 5.0];
let nonzero_indices = sorting::nonzero(&sparse);

// Conditional indexing
let data_f64 = array![1.0, 2.0, 3.0, 4.0, 5.0];
let indices = sorting::where_cond(&data_f64, |&x| x > 3.0);
```

## Performance

NumPy Rust leverages several optimizations:

- **BLAS/LAPACK**: Uses optimized linear algebra routines via `ndarray-linalg`
- **Parallel Operations**: Automatic parallelization with Rayon for large arrays
- **Zero-Cost Abstractions**: Rust's zero-cost abstractions ensure minimal overhead
- **SIMD**: Automatic vectorization where possible

## Comparison with NumPy

| Feature | NumPy (Python) | NumPy Rust |
|---------|----------------|------------|
| Type Safety | Runtime | Compile-time |
| Performance | Fast (C backend) | Faster (native Rust + BLAS) |
| Memory Safety | Depends on C code | Guaranteed by Rust |
| Parallelism | GIL limitations | Native threads |
| Package Size | Large | Smaller binary |

## Architecture

NumPy Rust is built on top of these excellent crates:

- **ndarray**: Core N-dimensional array functionality
- **ndarray-linalg**: Linear algebra operations with BLAS/LAPACK bindings
- **ndarray-rand**: Random number generation for arrays
- **ndarray-stats**: Statistical operations
- **rustfft**: Fast Fourier Transform implementation
- **rand**: Random number generation
- **num-traits**: Numeric trait abstractions

## Documentation

Full API documentation is available at [docs.rs/numpy_rust](https://docs.rs/numpy_rust).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Roadmap

- [ ] Additional linear algebra operations
- [ ] Polynomial operations
- [ ] Signal processing functions
- [ ] Image processing utilities
- [ ] Integration with Python via PyO3
- [ ] GPU acceleration support
- [ ] More comprehensive benchmarks

## Acknowledgments

This project builds on the excellent work of the Rust scientific computing community, particularly the `ndarray` ecosystem maintainers.
