use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use numpy_rust::prelude::*;

fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("zeros", size), size, |b, &size| {
            b.iter(|| {
                zeros::<f64>(IxDyn(&[size]))
            });
        });

        group.bench_with_input(BenchmarkId::new("ones", size), size, |b, &size| {
            b.iter(|| {
                ones::<f64>(IxDyn(&[size]))
            });
        });

        group.bench_with_input(BenchmarkId::new("linspace", size), size, |b, &size| {
            b.iter(|| {
                linspace(0.0, 100.0, size).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_math_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_operations");

    let arr = Array1::linspace(0.0, 100.0, 10000);

    group.bench_function("sin", |b| {
        b.iter(|| {
            sin(black_box(&arr))
        });
    });

    group.bench_function("cos", |b| {
        b.iter(|| {
            cos(black_box(&arr))
        });
    });

    group.bench_function("exp", |b| {
        b.iter(|| {
            exp(black_box(&arr))
        });
    });

    group.bench_function("log", |b| {
        b.iter(|| {
            log(black_box(&arr))
        });
    });

    group.bench_function("sum", |b| {
        b.iter(|| {
            sum(black_box(&arr))
        });
    });

    group.bench_function("cumsum", |b| {
        b.iter(|| {
            cumsum(black_box(&arr))
        });
    });

    group.finish();
}

fn bench_stats_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_operations");

    let arr = Array1::linspace(0.0, 100.0, 10000);

    group.bench_function("mean", |b| {
        b.iter(|| {
            stats::mean(black_box(&arr)).unwrap()
        });
    });

    group.bench_function("median", |b| {
        b.iter(|| {
            stats::median(black_box(&arr)).unwrap()
        });
    });

    group.bench_function("var", |b| {
        b.iter(|| {
            stats::var(black_box(&arr), 1).unwrap()
        });
    });

    group.bench_function("std", |b| {
        b.iter(|| {
            stats::std(black_box(&arr), 1).unwrap()
        });
    });

    group.bench_function("percentile", |b| {
        b.iter(|| {
            stats::percentile(black_box(&arr), 50.0).unwrap()
        });
    });

    group.finish();
}

fn bench_linalg_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("linalg_operations");

    // Matrix multiplication
    for size in [10, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bencher, &size| {
            let a = Array2::from_elem((size, size), 1.0);
            let b = Array2::from_elem((size, size), 2.0);
            bencher.iter(|| {
                linalg::matmul(black_box(&a), black_box(&b)).unwrap()
            });
        });
    }

    // 2x2 operations
    let m2x2 = array![[1.0, 2.0], [3.0, 4.0]];

    group.bench_function("det_2x2", |b| {
        b.iter(|| {
            linalg::det(black_box(&m2x2)).unwrap()
        });
    });

    group.bench_function("inv_2x2", |b| {
        b.iter(|| {
            linalg::inv(black_box(&m2x2)).unwrap()
        });
    });

    group.bench_function("transpose", |b| {
        b.iter(|| {
            linalg::transpose(black_box(&m2x2))
        });
    });

    group.finish();
}

fn bench_fft_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_operations");

    for size in [128, 512, 1024].iter() {
        group.bench_with_input(BenchmarkId::new("fft", size), size, |b, &size| {
            let arr = Array1::linspace(0.0, 100.0, size);
            b.iter(|| {
                fft::fft(black_box(&arr)).unwrap()
            });
        });

        group.bench_with_input(BenchmarkId::new("rfft", size), size, |b, &size| {
            let arr = Array1::linspace(0.0, 100.0, size);
            b.iter(|| {
                fft::rfft(black_box(&arr)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_sorting_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting_operations");

    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(BenchmarkId::new("sort", size), size, |b, &size| {
            let arr = Array1::linspace(0.0, 100.0, size);
            b.iter(|| {
                sorting::sorted(black_box(&arr))
            });
        });

        group.bench_with_input(BenchmarkId::new("argsort", size), size, |b, &size| {
            let arr = Array1::linspace(0.0, 100.0, size);
            b.iter(|| {
                sorting::argsort(black_box(&arr))
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_math_operations,
    bench_stats_operations,
    bench_linalg_operations,
    bench_fft_operations,
    bench_sorting_operations
);
criterion_main!(benches);
