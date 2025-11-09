//! Mathematical functions for arrays
//!
//! This module provides element-wise mathematical operations similar to NumPy's
//! mathematical functions.

use ndarray::{Array, ArrayBase, Data, Dimension};
use num_traits::{Float, Num, Zero, One};

/// Compute sine element-wise
pub fn sin<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.sin())
}

/// Compute cosine element-wise
pub fn cos<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.cos())
}

/// Compute tangent element-wise
pub fn tan<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.tan())
}

/// Compute arc sine element-wise
pub fn arcsin<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.asin())
}

/// Compute arc cosine element-wise
pub fn arccos<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.acos())
}

/// Compute arc tangent element-wise
pub fn arctan<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.atan())
}

/// Compute element-wise arc tangent of y/x
pub fn arctan2<S, D>(y: &ArrayBase<S, D>, x: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    ndarray::Zip::from(y).and(x).map_collect(|&y_val, &x_val| y_val.atan2(x_val))
}

/// Compute hyperbolic sine element-wise
pub fn sinh<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.sinh())
}

/// Compute hyperbolic cosine element-wise
pub fn cosh<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.cosh())
}

/// Compute hyperbolic tangent element-wise
pub fn tanh<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.tanh())
}

/// Compute natural exponential element-wise
pub fn exp<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.exp())
}

/// Compute exp(x) - 1 element-wise
pub fn expm1<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.exp_m1())
}

/// Compute 2^x element-wise
pub fn exp2<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.exp2())
}

/// Compute natural logarithm element-wise
pub fn log<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.ln())
}

/// Compute log(1 + x) element-wise
pub fn log1p<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.ln_1p())
}

/// Compute base-2 logarithm element-wise
pub fn log2<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.log2())
}

/// Compute base-10 logarithm element-wise
pub fn log10<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.log10())
}

/// Compute square root element-wise
pub fn sqrt<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.sqrt())
}

/// Compute cube root element-wise
pub fn cbrt<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.cbrt())
}

/// Compute x^y element-wise
pub fn power<S, D>(base: &ArrayBase<S, D>, exp: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    ndarray::Zip::from(base).and(exp).map_collect(|&b, &e| b.powf(e))
}

/// Compute absolute value element-wise
pub fn abs<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.abs())
}

/// Compute sign element-wise (-1, 0, or 1)
pub fn sign<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float + Zero + One,
    D: Dimension,
{
    arr.mapv(|x| {
        let zero = S::Elem::zero();
        let one = S::Elem::one();
        if x > zero {
            one
        } else if x < zero {
            -one
        } else {
            zero
        }
    })
}

/// Round to nearest integer element-wise
pub fn round<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.round())
}

/// Round down (floor) element-wise
pub fn floor<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.floor())
}

/// Round up (ceil) element-wise
pub fn ceil<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.ceil())
}

/// Truncate to integer element-wise
pub fn trunc<S, D>(arr: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| x.trunc())
}

/// Element-wise maximum of two arrays
pub fn maximum<S, D>(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    ndarray::Zip::from(a).and(b).map_collect(|&x, &y| x.max(y))
}

/// Element-wise minimum of two arrays
pub fn minimum<S, D>(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    ndarray::Zip::from(a).and(b).map_collect(|&x, &y| x.min(y))
}

/// Clip (limit) array values to a range
pub fn clip<S, D>(arr: &ArrayBase<S, D>, min: S::Elem, max: S::Elem) -> Array<S::Elem, D>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    arr.mapv(|x| {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    })
}

/// Sum of array elements
pub fn sum<S, D>(arr: &ArrayBase<S, D>) -> S::Elem
where
    S: Data,
    S::Elem: Num + Clone + Zero,
    D: Dimension,
{
    arr.iter().cloned().fold(Zero::zero(), |acc, x| acc + x)
}

/// Product of array elements
pub fn prod<S, D>(arr: &ArrayBase<S, D>) -> S::Elem
where
    S: Data,
    S::Elem: Num + Clone + One,
    D: Dimension,
{
    arr.iter().cloned().fold(One::one(), |acc, x| acc * x)
}

/// Cumulative sum of array elements
pub fn cumsum<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Array<S::Elem, ndarray::Ix1>
where
    S: Data,
    S::Elem: Num + Clone + Zero,
{
    let mut result = Vec::with_capacity(arr.len());
    let mut acc: S::Elem = Zero::zero();
    for x in arr.iter() {
        acc = acc + x.clone();
        result.push(acc.clone());
    }
    Array::from_vec(result)
}

/// Cumulative product of array elements
pub fn cumprod<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Array<S::Elem, ndarray::Ix1>
where
    S: Data,
    S::Elem: Num + Clone + One,
{
    let mut result = Vec::with_capacity(arr.len());
    let mut acc: S::Elem = One::one();
    for x in arr.iter() {
        acc = acc * x.clone();
        result.push(acc.clone());
    }
    Array::from_vec(result)
}

/// Compute differences between consecutive elements
pub fn diff<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Array<S::Elem, ndarray::Ix1>
where
    S: Data,
    S::Elem: Num + Clone,
{
    if arr.len() <= 1 {
        return Array::from_vec(vec![]);
    }

    let mut result = Vec::with_capacity(arr.len() - 1);
    for i in 1..arr.len() {
        result.push(arr[i].clone() - arr[i - 1].clone());
    }
    Array::from_vec(result)
}

/// Compute gradient (numerical derivative)
pub fn gradient<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Array<S::Elem, ndarray::Ix1>
where
    S: Data,
    S::Elem: Float,
{
    let n = arr.len();
    if n == 0 {
        return Array::from_vec(vec![]);
    }
    if n == 1 {
        return Array::from_vec(vec![S::Elem::zero()]);
    }

    let mut result = Vec::with_capacity(n);
    let two = S::Elem::one() + S::Elem::one();

    // First element: forward difference
    result.push(arr[1] - arr[0]);

    // Middle elements: central difference
    for i in 1..n - 1 {
        result.push((arr[i + 1] - arr[i - 1]) / two);
    }

    // Last element: backward difference
    result.push(arr[n - 1] - arr[n - 2]);

    Array::from_vec(result)
}

/// Compute the dot product of two arrays
pub fn dot<S1, S2>(a: &ArrayBase<S1, ndarray::Ix1>, b: &ArrayBase<S2, ndarray::Ix1>) -> S1::Elem
where
    S1: Data,
    S2: Data<Elem = S1::Elem>,
    S1::Elem: Num + Clone + Zero + Copy,
{
    ndarray::Zip::from(a)
        .and(b)
        .fold(Zero::zero(), |acc, &x, &y| acc + x * y)
}

/// Convolve two 1-dimensional arrays
pub fn convolve<S1, S2>(
    a: &ArrayBase<S1, ndarray::Ix1>,
    v: &ArrayBase<S2, ndarray::Ix1>,
) -> Array<S1::Elem, ndarray::Ix1>
where
    S1: Data,
    S2: Data<Elem = S1::Elem>,
    S1::Elem: Num + Clone + Zero,
{
    let n = a.len();
    let m = v.len();
    if n == 0 || m == 0 {
        return Array::from_vec(vec![]);
    }

    let result_len = n + m - 1;
    let mut result: Vec<S1::Elem> = vec![Zero::zero(); result_len];

    for i in 0..n {
        for j in 0..m {
            result[i + j] = result[i + j].clone() + a[i].clone() * v[j].clone();
        }
    }

    Array::from_vec(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_trigonometric() {
        let a = array![0.0, std::f64::consts::PI / 2.0];
        let s = sin(&a);
        assert_relative_eq!(s[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(s[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exponential() {
        let a = array![0.0, 1.0];
        let e = exp(&a);
        assert_relative_eq!(e[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(e[1], std::f64::consts::E, epsilon = 1e-10);
    }

    #[test]
    fn test_logarithm() {
        let a = array![1.0, std::f64::consts::E];
        let l = log(&a);
        assert_relative_eq!(l[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(l[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sum() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(sum(&a), 15.0);
    }

    #[test]
    fn test_cumsum() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let cs = cumsum(&a);
        assert_eq!(cs, array![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_diff() {
        let a = array![1.0, 3.0, 6.0, 10.0];
        let d = diff(&a);
        assert_eq!(d, array![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dot() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        assert_relative_eq!(dot(&a, &b), 32.0);
    }
}
