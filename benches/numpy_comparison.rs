use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use numpy_rust::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyModule;

fn bench_array_creation_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation_comparison");
    pyo3::prepare_freethreaded_python();

    for size in [100, 1000, 10000].iter() {
        // Our implementation - zeros
        group.bench_with_input(BenchmarkId::new("rust_zeros", size), size, |b, &size| {
            b.iter(|| {
                zeros::<f64>(black_box(IxDyn(&[size])))
            });
        });

        // Python NumPy - zeros
        group.bench_with_input(BenchmarkId::new("numpy_zeros", size), size, |b, &size| {
            b.iter(|| {
                Python::with_gil(|py| {
                    let np = PyModule::import_bound(py, "numpy").unwrap();
                    let _ = np.getattr("zeros").unwrap().call1((size,)).unwrap();
                })
            });
        });

        // Our implementation - ones
        group.bench_with_input(BenchmarkId::new("rust_ones", size), size, |b, &size| {
            b.iter(|| {
                ones::<f64>(black_box(IxDyn(&[size])))
            });
        });

        // Python NumPy - ones
        group.bench_with_input(BenchmarkId::new("numpy_ones", size), size, |b, &size| {
            b.iter(|| {
                Python::with_gil(|py| {
                    let np = PyModule::import_bound(py, "numpy").unwrap();
                    let _ = np.getattr("ones").unwrap().call1((size,)).unwrap();
                })
            });
        });

        // Our implementation - linspace
        group.bench_with_input(BenchmarkId::new("rust_linspace", size), size, |b, &size| {
            b.iter(|| {
                linspace(black_box(0.0), black_box(100.0), black_box(size)).unwrap()
            });
        });

        // Python NumPy - linspace
        group.bench_with_input(BenchmarkId::new("numpy_linspace", size), size, |b, &size| {
            b.iter(|| {
                Python::with_gil(|py| {
                    let np = PyModule::import_bound(py, "numpy").unwrap();
                    let _ = np.getattr("linspace").unwrap().call1((0.0, 100.0, size)).unwrap();
                })
            });
        });
    }

    group.finish();
}

fn bench_math_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_operations_comparison");
    pyo3::prepare_freethreaded_python();

    let size = 10000;
    let arr = Array1::linspace(0.1, 100.0, size);

    // Sin comparison
    group.bench_function("rust_sin", |b| {
        b.iter(|| {
            sin(black_box(&arr))
        });
    });

    group.bench_function("numpy_sin", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let np = PyModule::import_bound(py, "numpy").unwrap();
                let arr_py = np.getattr("linspace").unwrap().call1((0.1, 100.0, size)).unwrap();
                let _ = np.getattr("sin").unwrap().call1((arr_py,)).unwrap();
            })
        });
    });

    // Sum comparison
    group.bench_function("rust_sum", |b| {
        b.iter(|| {
            sum(black_box(&arr))
        });
    });

    group.bench_function("numpy_sum", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let np = PyModule::import_bound(py, "numpy").unwrap();
                let arr_py = np.getattr("linspace").unwrap().call1((0.1, 100.0, size)).unwrap();
                let _ = np.getattr("sum").unwrap().call1((arr_py,)).unwrap();
            })
        });
    });

    group.finish();
}

fn bench_stats_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_operations_comparison");
    pyo3::prepare_freethreaded_python();

    let size = 10000;
    let arr = Array1::linspace(0.0, 100.0, size);

    // Mean comparison
    group.bench_function("rust_mean", |b| {
        b.iter(|| {
            stats::mean(black_box(&arr)).unwrap()
        });
    });

    group.bench_function("numpy_mean", |b| {
        b.iter(|| {
            Python::with_gil(|py| {
                let np = PyModule::import_bound(py, "numpy").unwrap();
                let arr_py = np.getattr("linspace").unwrap().call1((0.0, 100.0, size)).unwrap();
                let _ = np.getattr("mean").unwrap().call1((arr_py,)).unwrap();
            })
        });
    });

    group.finish();
}

fn bench_linalg_operations_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_operations_comparison");
    pyo3::prepare_freethreaded_python();

    // Matrix multiplication comparison (small matrices where we have implementation)
    let size = 50;
    let a = Array2::from_elem((size, size), 1.0);
    let b_arr = Array2::from_elem((size, size), 2.0);

    group.bench_function("rust_matmul_50x50", |bencher| {
        bencher.iter(|| {
            linalg::matmul(black_box(&a), black_box(&b_arr)).unwrap()
        });
    });

    group.bench_function("numpy_matmul_50x50", |bencher| {
        bencher.iter(|| {
            Python::with_gil(|py| {
                let np = PyModule::import_bound(py, "numpy").unwrap();
                let a_py = np.getattr("ones").unwrap().call1(((size, size),)).unwrap();
                let b_py = np.getattr("ones").unwrap().call1(((size, size),)).unwrap();
                let b_py = np.getattr("multiply").unwrap().call1((b_py, 2.0)).unwrap();
                let _ = np.getattr("matmul").unwrap().call1((a_py, b_py)).unwrap();
            })
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation_comparison,
    bench_math_operations_comparison,
    bench_stats_operations_comparison,
    bench_linalg_operations_comparison
);
criterion_main!(benches);
