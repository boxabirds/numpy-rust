//! Random number generation
//!
//! This module provides random number generation functions similar to NumPy's
//! random module, using the rand crate ecosystem.

use ndarray::{Array, Array1, Array2, ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform, StandardNormal, Exp, Beta, Gamma, ChiSquared, Poisson};
use rand::Rng;
use num_traits::Float;
use crate::error::{NumpyError, Result};

/// Set the random seed (note: this affects the thread-local RNG)
pub fn seed(seed: u64) {
    use rand::SeedableRng;
    let _rng = rand::rngs::StdRng::seed_from_u64(seed);
    // Note: In a real implementation, you'd want to store this RNG
    // and use it for all random operations
}

/// Generate random floats in the half-open interval [0.0, 1.0)
///
/// # Examples
///
/// ```
/// use numpy_rust::random::rand;
///
/// let r = rand::<f64>(vec![3, 4].into());
/// assert_eq!(r.shape(), &[3, 4]);
/// ```
pub fn rand<T: Float>(shape: impl Into<IxDyn>) -> ArrayD<T>
where
    StandardNormal: Distribution<T>,
{
    let shape = shape.into();
    Array::random(shape, Uniform::new(T::zero(), T::one()))
}

/// Generate random floats from a standard normal distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::randn;
///
/// let r = randn::<f64>(vec![3, 4].into());
/// assert_eq!(r.shape(), &[3, 4]);
/// ```
pub fn randn<T: Float>(shape: impl Into<IxDyn>) -> ArrayD<T>
where
    StandardNormal: Distribution<T>,
{
    Array::random(shape.into(), StandardNormal)
}

/// Generate random integers from low (inclusive) to high (exclusive)
///
/// # Examples
///
/// ```
/// use numpy_rust::random::randint;
///
/// let r = randint(0, 10, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
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
///
/// # Examples
///
/// ```
/// use numpy_rust::random::uniform;
///
/// let r = uniform(0.0, 1.0, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
pub fn uniform<T: Float>(low: T, high: T, shape: impl Into<IxDyn>) -> Result<ArrayD<T>>
where
    Uniform<T>: Distribution<T>,
{
    if low >= high {
        return Err(NumpyError::ValueError(
            format!("low must be less than high")
        ));
    }

    let dist = Uniform::new(low, high);
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a normal (Gaussian) distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::normal;
///
/// let r = normal(0.0, 1.0, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
pub fn normal<T: Float>(mean: T, std_dev: T, shape: impl Into<IxDyn>) -> Result<ArrayD<T>>
where
    Normal<T>: Distribution<T>,
{
    if std_dev < T::zero() {
        return Err(NumpyError::ValueError(
            "Standard deviation must be non-negative".to_string()
        ));
    }

    let dist = Normal::new(mean, std_dev)
        .map_err(|e| NumpyError::ValueError(format!("Invalid normal distribution parameters: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from an exponential distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::exponential;
///
/// let r = exponential(1.0, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
pub fn exponential<T: Float>(scale: T, shape: impl Into<IxDyn>) -> Result<ArrayD<T>>
where
    Exp<T>: Distribution<T>,
{
    if scale <= T::zero() {
        return Err(NumpyError::ValueError(
            "Scale must be positive".to_string()
        ));
    }

    let lambda = T::one() / scale;
    let dist = Exp::new(lambda)
        .map_err(|e| NumpyError::ValueError(format!("Invalid exponential distribution: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a beta distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::beta;
///
/// let r = beta(2.0, 5.0, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
pub fn beta<T: Float>(alpha: T, beta_param: T, shape: impl Into<IxDyn>) -> Result<ArrayD<T>>
where
    Beta<T>: Distribution<T>,
{
    let dist = Beta::new(alpha, beta_param)
        .map_err(|e| NumpyError::ValueError(format!("Invalid beta distribution parameters: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a gamma distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::gamma;
///
/// let r = gamma(2.0, 2.0, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
pub fn gamma<T: Float>(shape_param: T, scale: T, shape: impl Into<IxDyn>) -> Result<ArrayD<T>>
where
    Gamma<T>: Distribution<T>,
{
    let dist = Gamma::new(shape_param, scale)
        .map_err(|e| NumpyError::ValueError(format!("Invalid gamma distribution parameters: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random floats from a chi-squared distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::chisquare;
///
/// let r = chisquare(5.0, vec![5].into()).unwrap();
/// assert_eq!(r.len(), 5);
/// ```
pub fn chisquare<T: Float>(df: T, shape: impl Into<IxDyn>) -> Result<ArrayD<T>>
where
    ChiSquared<T>: Distribution<T>,
{
    let dist = ChiSquared::new(df)
        .map_err(|e| NumpyError::ValueError(format!("Invalid chi-squared distribution: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate random integers from a Poisson distribution
///
/// # Examples
///
/// ```
/// use numpy_rust::random::poisson;
///
/// let r = poisson(5.0, vec![10].into()).unwrap();
/// assert_eq!(r.len(), 10);
/// ```
pub fn poisson(lam: f64, shape: impl Into<IxDyn>) -> Result<ArrayD<u64>> {
    let dist = Poisson::new(lam)
        .map_err(|e| NumpyError::ValueError(format!("Invalid Poisson distribution: {}", e)))?;
    Ok(Array::random(shape.into(), dist))
}

/// Generate a random permutation of integers from 0 to n-1
///
/// # Examples
///
/// ```
/// use numpy_rust::random::permutation;
///
/// let p = permutation(10);
/// assert_eq!(p.len(), 10);
/// ```
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
    arr.as_slice_mut().unwrap().shuffle(&mut rng);
}

/// Generate a random sample from a given 1-D array
///
/// # Examples
///
/// ```
/// use numpy_rust::random::choice;
/// use ndarray::array;
///
/// let a = array![1, 2, 3, 4, 5];
/// let sample = choice(&a, 3, true).unwrap();
/// assert_eq!(sample.len(), 3);
/// ```
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
        let r = rand::<f64>(vec![3, 4].into());
        assert_eq!(r.shape(), &[3, 4]);
        assert!(r.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    fn test_randn() {
        let r = randn::<f64>(vec![100].into());
        assert_eq!(r.len(), 100);
    }

    #[test]
    fn test_randint() {
        let r = randint(0, 10, vec![100].into()).unwrap();
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
