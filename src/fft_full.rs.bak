//! Fast Fourier Transform operations
//!
//! This module provides FFT functions similar to NumPy's fft module,
//! using the rustfft crate.

use ndarray::{Array1, ArrayBase, Data, Ix1};
use num_complex::Complex;
use num_traits::Float;
use rustfft::{FftPlanner, num_complex::Complex as RustFftComplex};
use crate::error::{NumpyError, Result};

/// Compute the one-dimensional discrete Fourier Transform
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::fft;
/// use ndarray::array;
///
/// let signal = array![1.0, 2.0, 1.0, -1.0, 1.5];
/// let spectrum = fft(&signal).unwrap();
/// ```
pub fn fft<S>(arr: &ArrayBase<S, Ix1>) -> Result<Array1<Complex<S::Elem>>>
where
    S: Data,
    S::Elem: Float,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute FFT of empty array".to_string()));
    }

    // Convert to complex numbers
    let mut buffer: Vec<RustFftComplex<S::Elem>> = arr
        .iter()
        .map(|&x| RustFftComplex::new(x, S::Elem::zero()))
        .collect();

    // Create FFT planner and compute FFT
    let mut planner = FftPlanner::new();
    let fft_transform = planner.plan_fft_forward(buffer.len());
    fft_transform.process(&mut buffer);

    // Convert back to num_complex::Complex
    let result: Vec<Complex<S::Elem>> = buffer
        .iter()
        .map(|c| Complex::new(c.re, c.im))
        .collect();

    Ok(Array1::from_vec(result))
}

/// Compute the one-dimensional inverse discrete Fourier Transform
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::{fft, ifft};
/// use ndarray::array;
///
/// let signal = array![1.0, 2.0, 1.0, -1.0, 1.5];
/// let spectrum = fft(&signal).unwrap();
/// let recovered = ifft(&spectrum).unwrap();
/// ```
pub fn ifft<S>(arr: &ArrayBase<S, Ix1>) -> Result<Array1<Complex<<S::Elem as num_complex::ComplexFloat>::Real>>>
where
    S: Data,
    S::Elem: num_complex::ComplexFloat,
    <S::Elem as num_complex::ComplexFloat>::Real: Float,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute IFFT of empty array".to_string()));
    }

    type F = <S::Elem as num_complex::ComplexFloat>::Real;

    // Convert to rustfft complex
    let mut buffer: Vec<RustFftComplex<F>> = arr
        .iter()
        .map(|c| RustFftComplex::new(c.re(), c.im()))
        .collect();

    // Create FFT planner and compute inverse FFT
    let mut planner = FftPlanner::new();
    let ifft_transform = planner.plan_fft_inverse(buffer.len());
    ifft_transform.process(&mut buffer);

    // Normalize and convert back
    let n = F::from(buffer.len()).unwrap();
    let result: Vec<Complex<F>> = buffer
        .iter()
        .map(|c| Complex::new(c.re / n, c.im / n))
        .collect();

    Ok(Array1::from_vec(result))
}

/// Compute the one-dimensional FFT of real input
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::rfft;
/// use ndarray::array;
///
/// let signal = array![1.0, 2.0, 1.0, -1.0, 1.5, 1.0];
/// let spectrum = rfft(&signal).unwrap();
/// ```
pub fn rfft<S>(arr: &ArrayBase<S, Ix1>) -> Result<Array1<Complex<S::Elem>>>
where
    S: Data,
    S::Elem: Float,
{
    // For real FFT, we only need to compute half the spectrum
    // due to symmetry
    let full_fft = fft(arr)?;
    let n = arr.len();
    let rfft_len = n / 2 + 1;

    Ok(full_fft.slice(ndarray::s![..rfft_len]).to_owned())
}

/// Compute the inverse of rfft
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::{rfft, irfft};
/// use ndarray::array;
///
/// let signal = array![1.0, 2.0, 1.0, -1.0, 1.5, 1.0];
/// let spectrum = rfft(&signal).unwrap();
/// let recovered = irfft(&spectrum, Some(signal.len())).unwrap();
/// ```
pub fn irfft<S>(arr: &ArrayBase<S, Ix1>, n: Option<usize>) -> Result<Array1<<S::Elem as num_complex::ComplexFloat>::Real>>
where
    S: Data,
    S::Elem: num_complex::ComplexFloat,
    <S::Elem as num_complex::ComplexFloat>::Real: Float,
{
    type F = <S::Elem as num_complex::ComplexFloat>::Real;

    // Determine output size
    let output_len = n.unwrap_or((arr.len() - 1) * 2);

    // Reconstruct full spectrum from rfft output
    let mut full_spectrum = Vec::with_capacity(output_len);

    // Add positive frequencies
    for &val in arr.iter() {
        full_spectrum.push(val);
    }

    // Add negative frequencies (conjugate symmetry)
    let start_idx = if output_len % 2 == 0 { arr.len() - 2 } else { arr.len() - 1 };
    for i in (1..=start_idx).rev() {
        let val = arr[i];
        full_spectrum.push(val.conj());
    }

    // Convert to array and compute IFFT
    let spectrum_array = Array1::from_vec(full_spectrum);
    let result = ifft(&spectrum_array)?;

    // Extract real part
    let real_result: Vec<F> = result.iter().map(|c| c.re()).collect();
    Ok(Array1::from_vec(real_result))
}

/// Compute the frequency bins for FFT output
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::fftfreq;
///
/// let freqs = fftfreq(8, 0.125);
/// ```
pub fn fftfreq<T: Float>(n: usize, d: T) -> Array1<T> {
    let mut freqs = Vec::with_capacity(n);
    let val = T::one() / (T::from(n).unwrap() * d);

    let n_half = (n - 1) / 2 + 1;

    // Positive frequencies
    for i in 0..n_half {
        freqs.push(T::from(i).unwrap() * val);
    }

    // Negative frequencies
    for i in (n_half..n) {
        freqs.push(T::from(i as i64 - n as i64).unwrap() * val);
    }

    Array1::from_vec(freqs)
}

/// Compute the frequency bins for rfft output
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::rfftfreq;
///
/// let freqs = rfftfreq(8, 0.125);
/// ```
pub fn rfftfreq<T: Float>(n: usize, d: T) -> Array1<T> {
    let val = T::one() / (T::from(n).unwrap() * d);
    let n_out = n / 2 + 1;

    let freqs: Vec<T> = (0..n_out)
        .map(|i| T::from(i).unwrap() * val)
        .collect();

    Array1::from_vec(freqs)
}

/// Shift the zero-frequency component to the center of the spectrum
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::{fft, fftshift};
/// use ndarray::array;
///
/// let signal = array![1.0, 2.0, 1.0, -1.0];
/// let spectrum = fft(&signal).unwrap();
/// let shifted = fftshift(&spectrum);
/// ```
pub fn fftshift<S, T>(arr: &ArrayBase<S, Ix1>) -> Array1<T>
where
    S: Data<Elem = T>,
    T: Clone,
{
    let n = arr.len();
    let mid = (n + 1) / 2;

    let mut result = Vec::with_capacity(n);

    // Add second half
    for i in mid..n {
        result.push(arr[i].clone());
    }

    // Add first half
    for i in 0..mid {
        result.push(arr[i].clone());
    }

    Array1::from_vec(result)
}

/// Inverse of fftshift
///
/// # Examples
///
/// ```
/// use numpy_rust::fft::{fftshift, ifftshift};
/// use ndarray::array;
///
/// let arr = array![1, 2, 3, 4, 5];
/// let shifted = fftshift(&arr);
/// let recovered = ifftshift(&shifted);
/// ```
pub fn ifftshift<S, T>(arr: &ArrayBase<S, Ix1>) -> Array1<T>
where
    S: Data<Elem = T>,
    T: Clone,
{
    let n = arr.len();
    let mid = n / 2;

    let mut result = Vec::with_capacity(n);

    // Add second half
    for i in mid..n {
        result.push(arr[i].clone());
    }

    // Add first half
    for i in 0..mid {
        result.push(arr[i].clone());
    }

    Array1::from_vec(result)
}

/// Compute the power spectral density
pub fn psd<S>(arr: &ArrayBase<S, Ix1>) -> Result<Array1<S::Elem>>
where
    S: Data,
    S::Elem: Float,
{
    let spectrum = fft(arr)?;
    let n = S::Elem::from(arr.len()).unwrap();

    let psd: Vec<S::Elem> = spectrum
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
