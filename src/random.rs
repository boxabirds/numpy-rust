//! Random number generation
//!
//! This module provides random number generation functions similar to NumPy's
//! random module, using the rand crate ecosystem.

use ndarray::{Array, Array1, ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Normal, Uniform, StandardNormal, Exp, Beta, Gamma};
use rand::Rng;
use crate::error::{NumpyError, Result};

/// Generate random floats in the half-open interval [0.0, 1.0)
pub fn rand(shape: impl Into<IxDyn>) -> ArrayD<f64> {
    Array::random(shape.into(), Uniform::new(0.0, 1.0))
}

/// Generate random floats from a standard normal distribution
pub fn randn(shape: impl Into<IxDyn>) -> ArrayD<f64> {
    Array::random(shape.into(), StandardNormal)
}

/// Generate random integers from low (inclusive) to high (exclusive)
pub fn randint(low: i64, high: i64, shape: impl Into<IxDyn>) -> Result<ArrayD<i64>> {
    if low >= high {
        return Err(NumpyError::ValueError(
            format!("low ({}) must be less than high ({})", low, high)
        ));
    }

    let dist = Uniform::new(low, high);
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a uniform distribution over [low, high)
pub fn uniform(low: f64, high: f64, shape: impl Into<IxDyn>) -> Result<ArrayD<f64>> {
    if low >= high {
        return Err(NumpyError::ValueError(
            "low must be less than high".to_string()
        ));
    }

    let dist = Uniform::new(low, high);
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a normal (Gaussian) distribution
pub fn normal(mean: f64, std_dev: f64, shape: impl Into<IxDyn>) -> Result<ArrayD<f64>> {
    if std_dev < 0.0 {
        return Err(NumpyError::ValueError(
            "Standard deviation must be non-negative".to_string()
        ));
    }

    let dist = Normal::new(mean, std_dev)
        .map_err(|e| NumpyError::ValueError(format!("Invalid normal distribution parameters: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from an exponential distribution
pub fn exponential(scale: f64, shape: impl Into<IxDyn>) -> Result<ArrayD<f64>> {
    if scale <= 0.0 {
        return Err(NumpyError::ValueError(
            "Scale must be positive".to_string()
        ));
    }

    let lambda = 1.0 / scale;
    let dist = Exp::new(lambda)
        .map_err(|e| NumpyError::ValueError(format!("Invalid exponential distribution: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a beta distribution
pub fn beta(alpha: f64, beta_param: f64, shape: impl Into<IxDyn>) -> Result<ArrayD<f64>> {
    let dist = Beta::new(alpha, beta_param)
        .map_err(|e| NumpyError::ValueError(format!("Invalid beta distribution parameters: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a gamma distribution
pub fn gamma(shape_param: f64, scale: f64, shape: impl Into<IxDyn>) -> Result<ArrayD<f64>> {
    let dist = Gamma::new(shape_param, scale)
        .map_err(|e| NumpyError::ValueError(format!("Invalid gamma distribution parameters: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate a random permutation of integers from 0 to n-1
pub fn permutation(n: usize) -> Array1<usize> {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    Array1::from_vec(indices)
}

/// Randomly shuffle an array in-place
pub fn shuffle<T: Clone>(arr: &mut Array1<T>) {
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    if let Some(slice) = arr.as_slice_mut() {
        slice.shuffle(&mut rng);
    }
}

/// Generate a random sample from a given 1-D array
pub fn choice<T: Clone>(arr: &Array1<T>, size: usize, replace: bool) -> Result<Array1<T>> {
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot choose from empty array".to_string()));
    }

    if !replace && size > arr.len() {
        return Err(NumpyError::ValueError(
            format!("Cannot take {} samples without replacement from array of size {}",
                    size, arr.len())
        ));
    }

    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(size);

    if replace {
        for _ in 0..size {
            let idx = rng.gen_range(0..arr.len());
            result.push(arr[idx].clone());
        }
    } else {
        use rand::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..arr.len()).collect();
        indices.shuffle(&mut rng);
        for i in 0..size {
            result.push(arr[indices[i]].clone());
        }
    }

    Ok(Array1::from_vec(result))
}

/// Generate random bytes
pub fn bytes(length: usize) -> Array1<u8> {
    let mut rng = rand::thread_rng();
    let bytes: Vec<u8> = (0..length).map(|_| rng.gen()).collect();
    Array1::from_vec(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rand() {
        let r = rand(IxDyn(&[3, 4]));
        assert_eq!(r.shape(), &[3, 4]);
        assert!(r.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_randn() {
        let r = randn(IxDyn(&[100]));
        assert_eq!(r.len(), 100);
    }

    #[test]
    fn test_randint() {
        let r = randint(0, 10, IxDyn(&[100])).unwrap();
        assert!(r.iter().all(|&x| x >= 0 && x < 10));
    }

    #[test]
    fn test_permutation() {
        let p = permutation(10);
        assert_eq!(p.len(), 10);
        let mut sorted = p.to_vec();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_choice() {
        let a = Array1::from_vec(vec![1, 2, 3, 4, 5]);
        let sample = choice(&a, 3, true).unwrap();
        assert_eq!(sample.len(), 3);
        assert!(sample.iter().all(|&x| a.iter().any(|&y| y == x)));
    }
}
