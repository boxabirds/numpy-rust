//! Simplified linear algebra operations without external dependencies
//!
//! This module provides basic linear algebra functions.

use ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, Num, One, Zero};
use crate::error::{NumpyError, Result};

/// Matrix multiplication
pub fn matmul<S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Result<Array2<S1::Elem>>
where
    S1: Data,
    S2: Data<Elem = S1::Elem>,
    S1::Elem: Num + Copy + 'static,
{
    if a.ncols() != b.nrows() {
        return Err(NumpyError::ShapeMismatch {
            expected: vec![a.nrows(), a.ncols()],
            got: vec![b.nrows(), b.ncols()],
        });
    }
    Ok(a.dot(b))
}

/// Matrix dot product (alias for matmul)
pub fn dot<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<Array2<S1::Elem>>
where
    S1: Data,
    S2: Data<Elem = S1::Elem>,
    S1::Elem: Num + Copy + 'static,
{
    matmul(a, b)
}

/// Compute the determinant of a 2x2 or 3x3 matrix
pub fn det<S>(a: &ArrayBase<S, Ix2>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(NumpyError::ShapeMismatch {
            expected: vec![shape[0], shape[0]],
            got: vec![shape[0], shape[1]],
        });
    }

    match shape[0] {
        1 => Ok(a[[0, 0]]),
        2 => Ok(a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]),
        3 => {
            let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
                - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
                + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);
            Ok(det)
        }
        _ => Err(NumpyError::LinalgError(
            "Determinant only implemented for 1x1, 2x2, and 3x3 matrices without linalg feature".to_string(),
        )),
    }
}

/// Compute the trace of a matrix (sum of diagonal elements)
pub fn trace<S>(a: &ArrayBase<S, Ix2>) -> S::Elem
where
    S: Data,
    S::Elem: Num + Copy + Zero,
{
    let n = a.nrows().min(a.ncols());
    (0..n).map(|i| a[[i, i]]).fold(Zero::zero(), |acc, x| acc + x)
}

/// Compute the transpose of a matrix
pub fn transpose<S>(a: &ArrayBase<S, Ix2>) -> Array2<S::Elem>
where
    S: Data,
    S::Elem: Clone,
{
    a.t().to_owned()
}

/// Compute 2x2 matrix inverse
pub fn inv_2x2<T: Float>(a: &Array2<T>) -> Result<Array2<T>> {
    if a.shape() != [2, 2] {
        return Err(NumpyError::ShapeMismatch {
            expected: vec![2, 2],
            got: a.shape().to_vec(),
        });
    }

    let det = a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]];
    if det.abs() < T::epsilon() {
        return Err(NumpyError::SingularMatrix);
    }

    let mut result = Array2::zeros((2, 2));
    result[[0, 0]] = a[[1, 1]] / det;
    result[[0, 1]] = -a[[0, 1]] / det;
    result[[1, 0]] = -a[[1, 0]] / det;
    result[[1, 1]] = a[[0, 0]] / det;

    Ok(result)
}

/// Compute matrix inverse (only 2x2 without linalg feature)
pub fn inv<S>(a: &ArrayBase<S, Ix2>) -> Result<Array2<S::Elem>>
where
    S: Data,
    S::Elem: Float,
{
    if a.shape() == [2, 2] {
        inv_2x2(&a.to_owned())
    } else {
        Err(NumpyError::LinalgError(
            "Matrix inversion only implemented for 2x2 matrices without linalg feature. Enable the 'linalg' feature for full support.".to_string(),
        ))
    }
}

/// Solve 2x2 linear system Ax = b
pub fn solve_2x2<T: Float + 'static>(a: &Array2<T>, b: &Array1<T>) -> Result<Array1<T>> {
    let a_inv = inv_2x2(a)?;
    Ok(a_inv.dot(b))
}

/// Solve linear system (only 2x2 without linalg feature)
pub fn solve<S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, ndarray::Ix1>,
) -> Result<Array1<S1::Elem>>
where
    S1: Data,
    S2: Data<Elem = S1::Elem>,
    S1::Elem: Float + 'static,
{
    if a.shape() == [2, 2] && b.len() == 2 {
        solve_2x2(&a.to_owned(), &b.to_owned())
    } else {
        Err(NumpyError::LinalgError(
            "Linear system solve only implemented for 2x2 systems without linalg feature. Enable the 'linalg' feature for full support.".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_matmul() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c, array![[19.0, 22.0], [43.0, 50.0]]);
    }

    #[test]
    fn test_det_2x2() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let d = det(&a).unwrap();
        assert_relative_eq!(d, -2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trace() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        assert_relative_eq!(trace(&a), 5.0);
    }

    #[test]
    fn test_transpose() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let at = transpose(&a);
        assert_eq!(at, array![[1.0, 3.0], [2.0, 4.0]]);
    }

    #[test]
    fn test_inv_2x2() {
        let a = array![[4.0, 7.0], [2.0, 6.0]];
        let a_inv = inv(&a).unwrap();
        let identity = matmul(&a, &a_inv).unwrap();
        assert_relative_eq!(identity[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 1]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[0, 1]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(identity[[1, 0]], 0.0, epsilon = 1e-10);
    }
}
