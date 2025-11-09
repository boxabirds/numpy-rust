//! Fast Fourier Transform operations
//!
//! This module provides FFT functions using the rustfft crate.
//! Currently supports f64 only.

use ndarray::Array1;
use num_complex::Complex;
use rustfft::{FftPlanner, num_complex::Complex as RustFftComplex};
use crate::error::{NumpyError, Result};

/// Compute the one-dimensional discrete Fourier Transform
pub fn fft(arr: &Array1<f64>) -> Result<Array1<Complex<f64>>> {
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute FFT of empty array".to_string()));
    }

    let mut buffer: Vec<RustFftComplex<f64>> = arr
        .iter()
        .map(|&x| RustFftComplex::new(x, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft_transform = planner.plan_fft_forward(buffer.len());
    fft_transform.process(&mut buffer);

    let result: Vec<Complex<f64>> = buffer
        .iter()
        .map(|c| Complex::new(c.re, c.im))
        .collect();

    Ok(Array1::from_vec(result))
}

/// Compute the one-dimensional inverse discrete Fourier Transform
pub fn ifft(arr: &Array1<Complex<f64>>) -> Result<Array1<Complex<f64>>> {
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute IFFT of empty array".to_string()));
    }

    let mut buffer: Vec<RustFftComplex<f64>> = arr
        .iter()
        .map(|c| RustFftComplex::new(c.re, c.im))
        .collect();

    let mut planner = FftPlanner::new();
    let ifft_transform = planner.plan_fft_inverse(buffer.len());
    ifft_transform.process(&mut buffer);

    let n = buffer.len() as f64;
    let result: Vec<Complex<f64>> = buffer
        .iter()
        .map(|c| Complex::new(c.re / n, c.im / n))
        .collect();

    Ok(Array1::from_vec(result))
}

/// Compute the one-dimensional FFT of real input
pub fn rfft(arr: &Array1<f64>) -> Result<Array1<Complex<f64>>> {
    let full_fft = fft(arr)?;
    let n = arr.len();
    let rfft_len = n / 2 + 1;

    Ok(full_fft.slice(ndarray::s![..rfft_len]).to_owned())
}

/// Compute the inverse of rfft
pub fn irfft(arr: &Array1<Complex<f64>>, n: Option<usize>) -> Result<Array1<f64>> {
    let output_len = n.unwrap_or((arr.len() - 1) * 2);

    let mut full_spectrum = Vec::with_capacity(output_len);

    for &val in arr.iter() {
        full_spectrum.push(val);
    }

    let start_idx = if output_len % 2 == 0 { arr.len() - 2 } else { arr.len() - 1 };
    for i in (1..=start_idx).rev() {
        let val = arr[i];
        full_spectrum.push(val.conj());
    }

    let spectrum_array = Array1::from_vec(full_spectrum);
    let result = ifft(&spectrum_array)?;

    let real_result: Vec<f64> = result.iter().map(|c| c.re).collect();
    Ok(Array1::from_vec(real_result))
}

/// Compute the frequency bins for FFT output
pub fn fftfreq(n: usize, d: f64) -> Array1<f64> {
    let mut freqs = Vec::with_capacity(n);
    let val = 1.0 / (n as f64 * d);

    let n_half = (n - 1) / 2 + 1;

    for i in 0..n_half {
        freqs.push(i as f64 * val);
    }

    for i in n_half..n {
        freqs.push((i as i64 - n as i64) as f64 * val);
    }

    Array1::from_vec(freqs)
}

/// Compute the frequency bins for rfft output
pub fn rfftfreq(n: usize, d: f64) -> Array1<f64> {
    let val = 1.0 / (n as f64 * d);
    let n_out = n / 2 + 1;

    let freqs: Vec<f64> = (0..n_out)
        .map(|i| i as f64 * val)
        .collect();

    Array1::from_vec(freqs)
}

/// Shift the zero-frequency component to the center of the spectrum
pub fn fftshift<T: Clone>(arr: &Array1<T>) -> Array1<T> {
    let n = arr.len();
    let mid = (n + 1) / 2;

    let mut result = Vec::with_capacity(n);

    for i in mid..n {
        result.push(arr[i].clone());
    }

    for i in 0..mid {
        result.push(arr[i].clone());
    }

    Array1::from_vec(result)
}

/// Inverse of fftshift
pub fn ifftshift<T: Clone>(arr: &Array1<T>) -> Array1<T> {
    let n = arr.len();
    let mid = n / 2;

    let mut result = Vec::with_capacity(n);

    for i in mid..n {
        result.push(arr[i].clone());
    }

    for i in 0..mid {
        result.push(arr[i].clone());
    }

    Array1::from_vec(result)
}

/// Compute the power spectral density
pub fn psd(arr: &Array1<f64>) -> Result<Array1<f64>> {
    let spectrum = fft(arr)?;
    let n = arr.len() as f64;

    let psd: Vec<f64> = spectrum
        .iter()
        .map(|c| (c.norm_sqr()) / n)
        .collect();

    Ok(Array1::from_vec(psd))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft_ifft_roundtrip() {
        let signal = array![1.0, 2.0, 1.0, -1.0, 1.5, 1.0, 0.5, -0.5];
        let spectrum = fft(&signal).unwrap();
        let recovered = ifft(&spectrum).unwrap();

        for i in 0..signal.len() {
            assert_relative_eq!(recovered[i].re, signal[i], epsilon = 1e-10);
            assert_relative_eq!(recovered[i].im, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fftfreq() {
        let freqs = fftfreq(8, 1.0);
        assert_eq!(freqs.len(), 8);
        assert_relative_eq!(freqs[0], 0.0);
    }

    #[test]
    fn test_fftshift() {
        let arr = array![1, 2, 3, 4, 5];
        let shifted = fftshift(&arr);
        assert_eq!(shifted, array![4, 5, 1, 2, 3]);
    }

    #[test]
    fn test_ifftshift() {
        let arr = array![1, 2, 3, 4, 5];
        let shifted = fftshift(&arr);
        let recovered = ifftshift(&shifted);
        assert_eq!(recovered, arr);
    }
}
