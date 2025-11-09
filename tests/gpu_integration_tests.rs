#![cfg(feature = "gpu")]

use numpy_rust::prelude::*;
use numpy_rust::linalg;
use numpy_rust::gpu::GpuContext;
use approx::assert_abs_diff_eq;

#[test]
fn test_gpu_matmul_correctness_512() {
    if GpuContext::get_or_init().is_none() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    // 512×512 matrices (GPU threshold)
    let n = 512;
    let a = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f32 * 0.1);
    let b = Array2::from_shape_fn((n, n), |(i, j)| (i * 2 + j) as f32 * 0.05);

    // This should use GPU (≥512 threshold)
    let result = linalg::matmul(&a, &b).unwrap();

    // Compute expected result manually using ndarray
    let expected = a.dot(&b);

    // Verify correctness
    assert_eq!(result.dim(), (n, n));
    for i in 0..n.min(100) {  // Check subset for performance
        for j in 0..n.min(100) {
            assert_abs_diff_eq!(result[[i, j]], expected[[i, j]], epsilon = 0.01);
        }
    }
}

#[test]
fn test_gpu_matmul_correctness_1024() {
    if GpuContext::get_or_init().is_none() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    // 1024×1024 matrices
    let n = 1024;
    let a = Array2::<f32>::ones((n, n));
    let b = Array2::<f32>::ones((n, n));

    // GPU matmul
    let result = linalg::matmul(&a, &b).unwrap();

    // All elements should be n (sum of n ones)
    assert_eq!(result.dim(), (n, n));
    for i in 0..10 {
        for j in 0..10 {
            assert_abs_diff_eq!(result[[i, j]], n as f32, epsilon = 0.1);
        }
    }
}

#[test]
fn test_gpu_vs_cpu_identity() {
    if GpuContext::get_or_init().is_none() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let n = 512;
    let a = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f32);
    let identity = Array2::from_shape_fn((n, n), |(i, j)| {
        if i == j { 1.0 } else { 0.0 }
    });

    // A × I = A
    let result = linalg::matmul(&a, &identity).unwrap();

    for i in 0..n.min(100) {
        for j in 0..n.min(100) {
            assert_abs_diff_eq!(result[[i, j]], a[[i, j]], epsilon = 1e-4);
        }
    }
}

#[test]
fn test_gpu_non_square_matrices() {
    if GpuContext::get_or_init().is_none() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    // 512×768 × 768×1024 = 512×1024
    let a = Array2::<f32>::ones((512, 768));
    let b = Array2::<f32>::ones((768, 1024));

    let result = linalg::matmul(&a, &b).unwrap();

    assert_eq!(result.dim(), (512, 1024));

    // All elements should be 768 (sum of 768 ones)
    for i in 0..10 {
        for j in 0..10 {
            assert_abs_diff_eq!(result[[i, j]], 768.0, epsilon = 0.1);
        }
    }
}
