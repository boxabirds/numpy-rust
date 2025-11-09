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

    // Eigenvalues and eigenvectors
    println!("\n4. Eigenvalues and Eigenvectors:");
    let symmetric = array![[2.0, 1.0], [1.0, 2.0]];
    let (eigenvalues, eigenvectors) = linalg::eig(&symmetric).unwrap();
    println!("   Symmetric matrix:\n{:?}", symmetric);
    println!("   Eigenvalues: {:?}", eigenvalues);
    println!("   Eigenvectors:\n{:?}", eigenvectors);

    // SVD decomposition
    println!("\n5. Singular Value Decomposition:");
    let m = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let (u, s, vt) = linalg::svd(&m).unwrap();
    println!("   Matrix M:\n{:?}", m);
    println!("   U:\n{:?}", u);
    println!("   Singular values: {:?}", s);
    println!("   V^T:\n{:?}", vt);

    // QR decomposition
    println!("\n6. QR Decomposition:");
    let matrix_qr = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let (q, r) = linalg::qr(&matrix_qr).unwrap();
    println!("   Matrix:\n{:?}", matrix_qr);
    println!("   Q (orthogonal):\n{:?}", q);
    println!("   R (upper triangular):\n{:?}", r);

    println!("\n=== Demo Complete ===");
}
