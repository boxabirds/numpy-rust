//! Statistics operations example

use numpy_rust::prelude::*;

fn main() {
    println!("=== NumPy Rust - Statistics ===\n");

    // Generate sample data
    let data = array![23.0, 45.0, 12.0, 67.0, 34.0, 56.0, 78.0, 90.0, 11.0, 44.0];

    println!("1. Basic Statistics:");
    println!("   Data: {:?}", data);
    println!("   Mean: {:.2}", stats::mean(&data).unwrap());
    println!("   Median: {:.2}", stats::median(&data).unwrap());
    println!("   Variance: {:.2}", stats::var(&data, 1).unwrap());
    println!("   Standard Deviation: {:.2}", stats::std(&data, 1).unwrap());

    println!("\n2. Min/Max Statistics:");
    println!("   Min: {:.2}", stats::min(&data).unwrap());
    println!("   Max: {:.2}", stats::max(&data).unwrap());
    println!("   Range: {:.2}", stats::ptp(&data).unwrap());

    println!("\n3. Percentiles:");
    println!("   25th percentile: {:.2}", stats::percentile(&data, 25.0).unwrap());
    println!("   50th percentile (median): {:.2}", stats::percentile(&data, 50.0).unwrap());
    println!("   75th percentile: {:.2}", stats::percentile(&data, 75.0).unwrap());

    println!("\n4. Correlation:");
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = array![2.0, 4.0, 5.0, 4.0, 5.0];
    println!("   x: {:?}", x);
    println!("   y: {:?}", y);
    println!("   Correlation coefficient: {:.4}", stats::corrcoef(&x, &y).unwrap());

    println!("\n5. Histogram:");
    let data_hist = array![1.0, 2.0, 1.0, 3.0, 4.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let (counts, edges) = stats::histogram(&data_hist, 5).unwrap();
    println!("   Data: {:?}", data_hist);
    println!("   Bin counts: {:?}", counts);
    println!("   Bin edges: {:?}", edges);

    println!("\n6. Unique Values:");
    let repeated = array![1, 2, 2, 3, 1, 4, 3, 3, 5, 1];
    let (unique_vals, counts) = sorting::unique_counts(&repeated);
    println!("   Data: {:?}", repeated);
    println!("   Unique values: {:?}", unique_vals);
    println!("   Counts: {:?}", counts);

    println!("\n=== Demo Complete ===");
}
