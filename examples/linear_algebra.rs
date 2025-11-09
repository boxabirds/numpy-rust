//! Linear algebra operations example

use numpy_rust::prelude::*;

fn main() {
    println!("=== NumPy Rust - Linear Algebra ===\n");

    // Matrix multiplication
    println!("1. Matrix Multiplication:");
    let a = array![[1.0, 2.0], [3.0, 4.0]];
    let b = array![[5.0, 6.0], [7.0, 8.0]];
    let c = linalg::matmul(&a, &b).unwrap();
    println!("   A:\n{:?}", a);
    println!("   B:\n{:?}", b);
    println!("   A × B:\n{:?}", c);

    // Solving linear systems
    println!("\n2. Solving Linear System Ax = b:");
    let a_sys = array![[3.0, 1.0], [1.0, 2.0]];
    let b_sys = array![9.0, 8.0];
    let x = linalg::solve(&a_sys, &b_sys).unwrap();
    println!("   A:\n{:?}", a_sys);
    println!("   b: {:?}", b_sys);
    println!("   Solution x: {:?}", x);
    println!("   Verification Ax: {:?}", a_sys.dot(&x));

    // Matrix inverse
    println!("\n3. Matrix Inverse:");
    let matrix = array![[4.0, 7.0], [2.0, 6.0]];
    let inv = linalg::inv(&matrix).unwrap();
    println!("   Original matrix:\n{:?}", matrix);
    println!("   Inverse:\n{:?}", inv);
    println!("   Product (should be identity):\n{:?}", matrix.dot(&inv));

    // Determinant
    println!("\n4. Determinant:");
    let det_matrix = array![[1.0, 2.0], [3.0, 4.0]];
    let det_val = linalg::det(&det_matrix).unwrap();
    println!("   Matrix:\n{:?}", det_matrix);
    println!("   Determinant: {}", det_val);

    // Transpose
    println!("\n5. Matrix Transpose:");
    let orig = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let transposed = linalg::transpose(&orig);
    println!("   Original:\n{:?}", orig);
    println!("   Transposed:\n{:?}", transposed);

    // Trace
    println!("\n6. Matrix Trace:");
    let trace_matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let trace_val = linalg::trace(&trace_matrix);
    println!("   Matrix:\n{:?}", trace_matrix);
    println!("   Trace (sum of diagonal): {}", trace_val);

    println!("\n=== Demo Complete ===");
}
