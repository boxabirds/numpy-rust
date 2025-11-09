//! Basic NumPy Rust operations example

use numpy_rust::prelude::*;

fn main() {
    println!("=== NumPy Rust - Basic Operations ===\n");

    // Array creation
    println!("1. Array Creation:");
    let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("   Array a: {:?}", a);

    let b = zeros::<f64>(vec![3, 3].into());
    println!("   Zeros (3x3):\n{:?}", b);

    let c = ones::<f64>(vec![2, 4].into());
    println!("   Ones (2x4):\n{:?}", c);

    let d = linspace(0.0, 10.0, 11).unwrap();
    println!("   Linspace [0, 10] with 11 points: {:?}", d);

    // Mathematical operations
    println!("\n2. Mathematical Operations:");
    let x = array![0.0, std::f64::consts::PI / 2.0, std::f64::consts::PI];
    println!("   x: {:?}", x);
    println!("   sin(x): {:?}", sin(&x));
    println!("   cos(x): {:?}", cos(&x));
    println!("   exp(x): {:?}", exp(&x));

    // Array operations
    println!("\n3. Array Operations:");
    let arr = array![1.0, 2.0, 3.0, 4.0, 5.0];
    println!("   Array: {:?}", arr);
    println!("   Sum: {}", sum(&arr));
    println!("   Product: {}", prod(&arr));
    println!("   Mean: {}", stats::mean(&arr).unwrap());
    println!("   Std Dev: {}", stats::std(&arr, 1).unwrap());

    // Matrix operations
    println!("\n4. Matrix Operations:");
    let matrix = array![[1.0, 2.0], [3.0, 4.0]];
    println!("   Matrix:\n{:?}", matrix);
    println!("   Determinant: {}", linalg::det(&matrix).unwrap());
    println!("   Trace: {}", linalg::trace(&matrix));

    println!("\n=== Demo Complete ===");
}
