//! Array creation routines similar to NumPy
//!
//! This module provides functions for creating arrays with various patterns
//! and initialization strategies.

use ndarray::{Array1, Array2, ArrayD, IxDyn};
use num_traits::{Float, Num, One, Zero};
use crate::error::{NumpyError, Result};

/// Create an array filled with zeros
///
/// # Examples
///
/// ```
/// use numpy_rust::array::zeros;
///
/// let a = zeros::<f64>(5);
/// assert_eq!(a.len(), 5);
/// ```
pub fn zeros<T: Zero + Clone>(shape: impl Into<IxDyn>) -> ArrayD<T> {
    ArrayD::zeros(shape.into())
}

/// Create a 1D array filled with zeros
pub fn zeros_1d<T: Zero + Clone>(n: usize) -> Array1<T> {
    Array1::zeros(n)
}

/// Create a 2D array filled with zeros
pub fn zeros_2d<T: Zero + Clone>(rows: usize, cols: usize) -> Array2<T> {
    Array2::zeros((rows, cols))
}

/// Create an array filled with ones
///
/// # Examples
///
/// ```
/// use numpy_rust::array::ones;
///
/// let a = ones::<f64>(5);
/// assert_eq!(a.len(), 5);
/// ```
pub fn ones<T: One + Clone>(shape: impl Into<IxDyn>) -> ArrayD<T> {
    ArrayD::ones(shape.into())
}

/// Create a 1D array filled with ones
pub fn ones_1d<T: One + Clone>(n: usize) -> Array1<T> {
    Array1::ones(n)
}

/// Create a 2D array filled with ones
pub fn ones_2d<T: One + Clone>(rows: usize, cols: usize) -> Array2<T> {
    Array2::ones((rows, cols))
}

/// Create an array filled with a specific value
///
/// # Examples
///
/// ```
/// use numpy_rust::array::full;
///
/// let a = full(5, 3.14);
/// assert_eq!(a.len(), 5);
/// ```
pub fn full<T: Clone>(shape: impl Into<IxDyn>, value: T) -> ArrayD<T> {
    ArrayD::from_elem(shape.into(), value)
}

/// Create a 1D array filled with a specific value
pub fn full_1d<T: Clone>(n: usize, value: T) -> Array1<T> {
    Array1::from_elem(n, value)
}

/// Create a 2D array filled with a specific value
pub fn full_2d<T: Clone>(rows: usize, cols: usize, value: T) -> Array2<T> {
    Array2::from_elem((rows, cols), value)
}

/// Create an identity matrix
///
/// # Examples
///
/// ```
/// use numpy_rust::array::eye;
///
/// let a = eye::<f64>(3);
/// assert_eq!(a.shape(), &[3, 3]);
/// ```
pub fn eye<T: Zero + One + Clone>(n: usize) -> Array2<T> {
    Array2::eye(n)
}

/// Create an identity matrix with specified rows and columns
pub fn eye_rect<T: Zero + One + Clone>(rows: usize, cols: usize) -> Array2<T> {
    let mut arr = Array2::zeros((rows, cols));
    let min_dim = rows.min(cols);
    for i in 0..min_dim {
        arr[[i, i]] = T::one();
    }
    arr
}

/// Create a diagonal matrix from a 1D array
///
/// # Examples
///
/// ```
/// use numpy_rust::array::diag;
/// use ndarray::array;
///
/// let v = array![1.0, 2.0, 3.0];
/// let d = diag(&v);
/// assert_eq!(d.shape(), &[3, 3]);
/// ```
pub fn diag<T: Zero + Clone>(v: &Array1<T>) -> Array2<T> {
    let n = v.len();
    let mut arr = Array2::zeros((n, n));
    for i in 0..n {
        arr[[i, i]] = v[i].clone();
    }
    arr
}

/// Create an array with evenly spaced values within a given interval
///
/// # Examples
///
/// ```
/// use numpy_rust::array::arange;
///
/// let a = arange(0.0, 10.0, 0.5);
/// assert_eq!(a.len(), 20);
/// ```
pub fn arange<T>(start: T, stop: T, step: T) -> Result<Array1<T>>
where
    T: Num + PartialOrd + Clone + Copy,
{
    if step == T::zero() {
        return Err(NumpyError::ValueError("step cannot be zero".to_string()));
    }

    let mut values = Vec::new();
    let mut current = start;

    if step > T::zero() {
        while current < stop {
            values.push(current);
            current = current + step;
        }
    } else {
        while current > stop {
            values.push(current);
            current = current + step;
        }
    }

    Ok(Array1::from_vec(values))
}

/// Create an array with evenly spaced numbers over a specified interval
///
/// # Examples
///
/// ```
/// use numpy_rust::array::linspace;
///
/// let a = linspace(0.0, 1.0, 11).unwrap();
/// assert_eq!(a.len(), 11);
/// ```
pub fn linspace<T: Float>(start: T, stop: T, num: usize) -> Result<Array1<T>> {
    if num == 0 {
        return Err(NumpyError::ValueError(
            "num must be at least 1".to_string(),
        ));
    }

    if num == 1 {
        return Ok(Array1::from_vec(vec![start]));
    }

    let step = (stop - start) / T::from(num - 1).unwrap();
    let values: Vec<T> = (0..num)
        .map(|i| start + T::from(i).unwrap() * step)
        .collect();

    Ok(Array1::from_vec(values))
}

/// Create an array with evenly spaced numbers on a log scale
///
/// # Examples
///
/// ```
/// use numpy_rust::array::logspace;
///
/// let a = logspace(0.0, 2.0, 3, 10.0).unwrap();
/// assert_eq!(a.len(), 3);
/// ```
pub fn logspace<T: Float>(start: T, stop: T, num: usize, base: T) -> Result<Array1<T>> {
    let linear = linspace(start, stop, num)?;
    Ok(linear.mapv(|x| base.powf(x)))
}

/// Create an array with values spaced evenly on a log scale (geometric progression)
///
/// # Examples
///
/// ```
/// use numpy_rust::array::geomspace;
///
/// let a = geomspace(1.0, 1000.0, 4).unwrap();
/// assert_eq!(a.len(), 4);
/// ```
pub fn geomspace<T: Float>(start: T, stop: T, num: usize) -> Result<Array1<T>> {
    if start <= T::zero() || stop <= T::zero() {
        return Err(NumpyError::ValueError(
            "start and stop must be positive".to_string(),
        ));
    }

    let log_start = start.ln();
    let log_stop = stop.ln();
    let log_values = linspace(log_start, log_stop, num)?;
    Ok(log_values.mapv(|x| x.exp()))
}

/// Create a triangular matrix (lower or upper)
pub fn tri<T: Zero + One + Clone>(n: usize, m: Option<usize>, k: isize, upper: bool) -> Array2<T> {
    let m = m.unwrap_or(n);
    let mut arr = Array2::zeros((n, m));

    for i in 0..n {
        for j in 0..m {
            let diff = j as isize - i as isize;
            let should_fill = if upper { diff >= k } else { diff <= k };
            if should_fill {
                arr[[i, j]] = T::one();
            }
        }
    }

    arr
}

/// Create a lower triangular matrix
pub fn tril<T: Zero + One + Clone>(n: usize, m: Option<usize>, k: isize) -> Array2<T> {
    tri(n, m, k, false)
}

/// Create an upper triangular matrix
pub fn triu<T: Zero + One + Clone>(n: usize, m: Option<usize>, k: isize) -> Array2<T> {
    tri(n, m, k, true)
}

/// Create a mesh grid from coordinate vectors
///
/// # Examples
///
/// ```
/// use numpy_rust::array::{meshgrid, arange};
///
/// let x = arange(0.0, 3.0, 1.0).unwrap();
/// let y = arange(0.0, 2.0, 1.0).unwrap();
/// let (xx, yy) = meshgrid(&x, &y);
/// ```
pub fn meshgrid<T: Clone + Zero>(x: &Array1<T>, y: &Array1<T>) -> (Array2<T>, Array2<T>) {
    let nx = x.len();
    let ny = y.len();

    let mut xx = Array2::zeros((ny, nx));
    let mut yy = Array2::zeros((ny, nx));

    for i in 0..ny {
        for j in 0..nx {
            xx[[i, j]] = x[j].clone();
            yy[[i, j]] = y[i].clone();
        }
    }

    (xx, yy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_zeros() {
        let a = zeros::<f64>(vec![2, 3].into());
        assert_eq!(a.shape(), &[2, 3]);
        assert!(a.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let a = ones::<f64>(vec![2, 3].into());
        assert_eq!(a.shape(), &[2, 3]);
        assert!(a.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_full() {
        let a = full(vec![2, 3].into(), 5.0);
        assert_eq!(a.shape(), &[2, 3]);
        assert!(a.iter().all(|&x| x == 5.0));
    }

    #[test]
    fn test_eye() {
        let a = eye::<f64>(3);
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 1.0);
        assert_eq!(a[[2, 2]], 1.0);
        assert_eq!(a[[0, 1]], 0.0);
    }

    #[test]
    fn test_arange() {
        let a = arange(0.0, 5.0, 1.0).unwrap();
        assert_eq!(a.len(), 5);
        assert_eq!(a[0], 0.0);
        assert_eq!(a[4], 4.0);
    }

    #[test]
    fn test_linspace() {
        let a = linspace(0.0, 1.0, 11).unwrap();
        assert_eq!(a.len(), 11);
        assert_relative_eq!(a[0], 0.0);
        assert_relative_eq!(a[10], 1.0);
        assert_relative_eq!(a[5], 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_logspace() {
        let a = logspace(0.0, 2.0, 3, 10.0).unwrap();
        assert_eq!(a.len(), 3);
        assert_relative_eq!(a[0], 1.0);
        assert_relative_eq!(a[1], 10.0);
        assert_relative_eq!(a[2], 100.0);
    }

    #[test]
    fn test_geomspace() {
        let a = geomspace(1.0, 1000.0, 4).unwrap();
        assert_eq!(a.len(), 4);
        assert_relative_eq!(a[0], 1.0);
        assert_relative_eq!(a[3], 1000.0);
    }

    #[test]
    fn test_meshgrid() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0]);
        let (xx, yy) = meshgrid(&x, &y);
        assert_eq!(xx.shape(), &[2, 3]);
        assert_eq!(yy.shape(), &[2, 3]);
    }
}
