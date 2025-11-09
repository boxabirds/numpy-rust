//! Signal processing with FFT example

use numpy_rust::prelude::*;
use std::f64::consts::PI;

fn main() {
    println!("=== NumPy Rust - Signal Processing ===\n");

    // Create a composite signal: sum of two sine waves
    let sample_rate = 100.0; // Hz
    let duration = 1.0; // seconds
    let n_samples = (sample_rate * duration) as usize;

    let t = linspace(0.0, duration, n_samples).unwrap();

    println!("1. Creating Composite Signal:");
    println!("   Sample rate: {} Hz", sample_rate);
    println!("   Duration: {} seconds", duration);
    println!("   Number of samples: {}", n_samples);

    // Signal = 5 Hz sine wave + 10 Hz sine wave
    let freq1 = 5.0;
    let freq2 = 10.0;
    let signal: Array1<f64> = t.mapv(|time| {
        (2.0 * PI * freq1 * time).sin() + 0.5 * (2.0 * PI * freq2 * time).sin()
    });

    println!("\n2. Signal Components:");
    println!("   Frequency 1: {} Hz (amplitude: 1.0)", freq1);
    println!("   Frequency 2: {} Hz (amplitude: 0.5)", freq2);

    // Compute FFT
    println!("\n3. Computing FFT:");
    let spectrum = fft::fft(&signal).unwrap();
    println!("   FFT computed with {} frequency bins", spectrum.len());

    // Compute power spectral density
    let psd = fft::psd(&signal).unwrap();

    // Find frequency bins
    let freqs = fft::fftfreq(n_samples, 1.0 / sample_rate);

    // Find dominant frequencies (first half of spectrum due to symmetry)
    let half_n = n_samples / 2;
    let mut peak_indices = Vec::new();
    let threshold = psd.slice(ndarray::s![..half_n])
        .iter()
        .cloned()
        .fold(0.0f64, f64::max) * 0.1;

    for i in 1..half_n-1 {
        if psd[i] > threshold && psd[i] > psd[i-1] && psd[i] > psd[i+1] {
            peak_indices.push(i);
        }
    }

    println!("\n4. Detected Frequency Peaks:");
    for idx in peak_indices.iter() {
        println!("   Frequency: {:.2} Hz, Power: {:.4}", freqs[*idx].abs(), psd[*idx]);
    }

    // Compute inverse FFT to verify
    let reconstructed = fft::ifft(&spectrum).unwrap();
    let reconstruction_error: f64 = reconstructed.iter()
        .zip(signal.iter())
        .map(|(r, s)| (r.re - s).abs())
        .sum::<f64>() / n_samples as f64;

    println!("\n5. FFT/IFFT Verification:");
    println!("   Average reconstruction error: {:.2e}", reconstruction_error);

    // Demonstrate other signal processing operations
    println!("\n6. Additional Signal Operations:");

    // Compute gradient (derivative)
    let derivative = gradient(&signal);
    println!("   Signal derivative computed ({} points)", derivative.len());

    // Compute cumulative sum (integral approximation)
    let integral = cumsum(&signal);
    println!("   Signal integral computed ({} points)", integral.len());

    // Convolve with a simple moving average kernel
    let kernel = array![0.2, 0.2, 0.2, 0.2, 0.2]; // 5-point moving average
    let smoothed = convolve(&signal, &kernel);
    println!("   Smoothed signal computed ({} points)", smoothed.len());

    println!("\n=== Demo Complete ===");
}
