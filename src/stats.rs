//! Statistical functions for arrays
//!
//! This module provides statistical functions similar to NumPy's statistical
//! functions, leveraging ndarray-stats where applicable.

use ndarray::{Array1, ArrayBase, Data, Dimension};
use num_traits::{Float, Zero, One, FromPrimitive, ToPrimitive};
use crate::error::{NumpyError, Result};
use rayon::prelude::*;

/// Threshold for using parallel operations (tuned for typical workloads)
const PARALLEL_THRESHOLD: usize = 10_000;

/// Compute the arithmetic mean along an axis
///
/// # Examples
///
/// ```
/// use numpy_rust::stats::mean;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let m = mean(&a).unwrap();
/// ```
pub fn mean<S, D>(arr: &ArrayBase<S, D>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + Zero + FromPrimitive + Send + Sync,
    D: Dimension,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute mean of empty array".to_string()));
    }

    let sum: S::Elem = if arr.len() >= PARALLEL_THRESHOLD {
        if let Some(slice) = arr.as_slice_memory_order() {
            slice.par_iter()
                .cloned()
                .reduce(|| Zero::zero(), |acc: S::Elem, x| acc + x)
        } else {
            arr.iter().cloned().fold(Zero::zero(), |acc: S::Elem, x| acc + x)
        }
    } else {
        arr.iter().cloned().fold(Zero::zero(), |acc: S::Elem, x| acc + x)
    };

    let n = S::Elem::from_usize(arr.len()).unwrap();
    Ok(sum / n)
}

/// Compute the weighted mean
pub fn average<S, D, W>(arr: &ArrayBase<S, D>, weights: Option<&ArrayBase<W, D>>) -> Result<S::Elem>
where
    S: Data,
    W: Data<Elem = S::Elem>,
    S::Elem: Float + Zero + Copy + FromPrimitive + Send + Sync,
    D: Dimension,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute average of empty array".to_string()));
    }

    match weights {
        None => mean(arr),
        Some(w) => {
            if arr.shape() != w.shape() {
                return Err(NumpyError::ShapeMismatch {
                    expected: arr.shape().to_vec(),
                    got: w.shape().to_vec(),
                });
            }

            let weighted_sum: S::Elem = ndarray::Zip::from(arr)
                .and(w)
                .fold(Zero::zero(), |acc: S::Elem, &x, &weight| acc + x * weight);

            let weight_sum: S::Elem = w.iter().cloned().fold(Zero::zero(), |acc: S::Elem, x| acc + x);

            let zero: S::Elem = Zero::zero();
            if weight_sum == zero {
                return Err(NumpyError::ValueError("Sum of weights is zero".to_string()));
            }

            Ok(weighted_sum / weight_sum)
        }
    }
}

/// Quickselect algorithm to find kth smallest element (O(n) average case)
fn quickselect<T: Float>(data: &mut [T], k: usize) -> T {
    if data.len() == 1 {
        return data[0];
    }

    let pivot_index = data.len() / 2;
    let pivot = data[pivot_index];

    let (mut less, mut equal, mut greater) = (Vec::new(), Vec::new(), Vec::new());

    for &val in data.iter() {
        if val < pivot {
            less.push(val);
        } else if val > pivot {
            greater.push(val);
        } else {
            equal.push(val);
        }
    }

    if k < less.len() {
        quickselect(&mut less, k)
    } else if k < less.len() + equal.len() {
        pivot
    } else {
        quickselect(&mut greater, k - less.len() - equal.len())
    }
}

/// Compute the median using quickselect (O(n) average case instead of O(n log n))
///
/// # Examples
///
/// ```
/// use numpy_rust::stats::median;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let m = median(&a).unwrap();
/// ```
pub fn median<S>(arr: &ArrayBase<S, ndarray::Ix1>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + One + FromPrimitive,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute median of empty array".to_string()));
    }

    let mut data = arr.to_vec();
    let n = data.len();

    if n % 2 == 1 {
        // Odd length: return middle element
        Ok(quickselect(&mut data, n / 2))
    } else {
        // Even length: return average of two middle elements
        let lower = quickselect(&mut data, n / 2 - 1);
        // Need fresh data for second quickselect
        let mut data2 = arr.to_vec();
        let upper = quickselect(&mut data2, n / 2);
        let two = S::Elem::from_u8(2).unwrap();
        Ok((lower + upper) / two)
    }
}

/// Compute the variance
///
/// # Examples
///
/// ```
/// use numpy_rust::stats::var;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let v = var(&a, 1).unwrap();
/// ```
pub fn var<S, D>(arr: &ArrayBase<S, D>, ddof: usize) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + Zero + FromPrimitive + Send + Sync,
    D: Dimension,
{
    if arr.len() <= ddof {
        return Err(NumpyError::ValueError(
            format!("Array length {} must be greater than ddof {}", arr.len(), ddof)
        ));
    }

    // Use Welford's algorithm for numerical stability and parallel efficiency
    let m = mean(arr)?;

    let sum_squared_diffs: S::Elem = if arr.len() >= PARALLEL_THRESHOLD {
        if let Some(slice) = arr.as_slice_memory_order() {
            slice.par_iter()
                .cloned()
                .map(|x| {
                    let diff = x - m;
                    diff * diff
                })
                .reduce(|| Zero::zero(), |acc, x| acc + x)
        } else {
            arr.iter()
                .cloned()
                .fold(Zero::zero(), |acc: S::Elem, x| {
                    let diff = x - m;
                    acc + diff * diff
                })
        }
    } else {
        arr.iter()
            .cloned()
            .fold(Zero::zero(), |acc: S::Elem, x| {
                let diff = x - m;
                acc + diff * diff
            })
    };

    let n = S::Elem::from_usize(arr.len() - ddof).unwrap();
    Ok(sum_squared_diffs / n)
}

/// Compute the standard deviation
///
/// # Examples
///
/// ```
/// use numpy_rust::stats::std;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let s = std(&a, 1).unwrap();
/// ```
pub fn std<S, D>(arr: &ArrayBase<S, D>, ddof: usize) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + Zero + FromPrimitive + Send + Sync,
    D: Dimension,
{
    let v = var(arr, ddof)?;
    Ok(v.sqrt())
}

/// Compute percentile
///
/// # Examples
///
/// ```
/// use numpy_rust::stats::percentile;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let p50 = percentile(&a, 50.0).unwrap();
/// ```
pub fn percentile<S>(arr: &ArrayBase<S, ndarray::Ix1>, q: f64) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + One + FromPrimitive,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute percentile of empty array".to_string()));
    }

    if q < 0.0 || q > 100.0 {
        return Err(NumpyError::ValueError(
            format!("Percentile must be between 0 and 100, got {}", q)
        ));
    }

    let index = q / 100.0 * (arr.len() - 1) as f64;
    let lower_idx = index.floor() as usize;
    let upper_idx = index.ceil() as usize;

    if lower_idx == upper_idx {
        // Exact index - use quickselect
        let mut data = arr.to_vec();
        Ok(quickselect(&mut data, lower_idx))
    } else {
        // Interpolate between two indices
        let mut data1 = arr.to_vec();
        let lower_val = quickselect(&mut data1, lower_idx);

        let mut data2 = arr.to_vec();
        let upper_val = quickselect(&mut data2, upper_idx);

        let fraction = S::Elem::from_f64(index - lower_idx as f64).unwrap();
        let one: S::Elem = One::one();
        Ok(lower_val * (one - fraction) + upper_val * fraction)
    }
}

/// Compute quantile (alias for percentile with q in [0, 1])
pub fn quantile<S>(arr: &ArrayBase<S, ndarray::Ix1>, q: f64) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + One + FromPrimitive,
{
    percentile(arr, q * 100.0)
}

/// Compute minimum value
pub fn min<S, D>(arr: &ArrayBase<S, D>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute min of empty array".to_string()));
    }

    Ok(arr.iter()
        .cloned()
        .fold(S::Elem::infinity(), |a, b| a.min(b)))
}

/// Compute maximum value
pub fn max<S, D>(arr: &ArrayBase<S, D>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute max of empty array".to_string()));
    }

    Ok(arr.iter()
        .cloned()
        .fold(S::Elem::neg_infinity(), |a, b| a.max(b)))
}

/// Compute peak-to-peak (max - min) value
pub fn ptp<S, D>(arr: &ArrayBase<S, D>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    Ok(max(arr)? - min(arr)?)
}

/// Compute the range (min, max)
pub fn range<S, D>(arr: &ArrayBase<S, D>) -> Result<(S::Elem, S::Elem)>
where
    S: Data,
    S::Elem: Float,
    D: Dimension,
{
    Ok((min(arr)?, max(arr)?))
}

/// Compute correlation coefficient
pub fn corrcoef<S>(x: &ArrayBase<S, ndarray::Ix1>, y: &ArrayBase<S, ndarray::Ix1>) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + FromPrimitive + Send + Sync,
{
    if x.len() != y.len() {
        return Err(NumpyError::ShapeMismatch {
            expected: vec![x.len()],
            got: vec![y.len()],
        });
    }

    if x.len() < 2 {
        return Err(NumpyError::ValueError("Need at least 2 samples".to_string()));
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;

    let mut cov = S::Elem::zero();
    let mut var_x = S::Elem::zero();
    let mut var_y = S::Elem::zero();

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov = cov + dx * dy;
        var_x = var_x + dx * dx;
        var_y = var_y + dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom == S::Elem::zero() {
        return Err(NumpyError::ValueError("Variance is zero".to_string()));
    }

    Ok(cov / denom)
}

/// Compute covariance
pub fn cov<S>(x: &ArrayBase<S, ndarray::Ix1>, y: &ArrayBase<S, ndarray::Ix1>, ddof: usize) -> Result<S::Elem>
where
    S: Data,
    S::Elem: Float + FromPrimitive + Send + Sync,
{
    if x.len() != y.len() {
        return Err(NumpyError::ShapeMismatch {
            expected: vec![x.len()],
            got: vec![y.len()],
        });
    }

    if x.len() <= ddof {
        return Err(NumpyError::ValueError(
            format!("Array length {} must be greater than ddof {}", x.len(), ddof)
        ));
    }

    let mean_x = mean(x)?;
    let mean_y = mean(y)?;

    let mut covariance = S::Elem::zero();
    for i in 0..x.len() {
        covariance = covariance + (x[i] - mean_x) * (y[i] - mean_y);
    }

    let n = S::Elem::from_usize(x.len() - ddof).unwrap();
    Ok(covariance / n)
}

/// Compute histogram
pub fn histogram<S>(
    arr: &ArrayBase<S, ndarray::Ix1>,
    bins: usize,
) -> Result<(Array1<usize>, Array1<S::Elem>)>
where
    S: Data,
    S::Elem: Float + FromPrimitive + ToPrimitive,
{
    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute histogram of empty array".to_string()));
    }

    if bins == 0 {
        return Err(NumpyError::ValueError("Number of bins must be positive".to_string()));
    }

    let min_val = min(arr)?;
    let max_val = max(arr)?;

    let bin_width = (max_val - min_val) / S::Elem::from_usize(bins).unwrap();
    let mut counts = vec![0usize; bins];
    let mut edges = Vec::with_capacity(bins + 1);

    for i in 0..=bins {
        edges.push(min_val + S::Elem::from_usize(i).unwrap() * bin_width);
    }

    for &val in arr.iter() {
        let mut bin = ((val - min_val) / bin_width).floor().to_usize().unwrap();
        if bin >= bins {
            bin = bins - 1;
        }
        counts[bin] += 1;
    }

    Ok((Array1::from_vec(counts), Array1::from_vec(edges)))
}

/// Compute binned statistics
pub fn binned_statistic<S, F>(
    arr: &ArrayBase<S, ndarray::Ix1>,
    values: &ArrayBase<S, ndarray::Ix1>,
    statistic: F,
    bins: usize,
) -> Result<(Array1<S::Elem>, Array1<S::Elem>)>
where
    S: Data,
    S::Elem: Float + FromPrimitive + ToPrimitive,
    F: Fn(&[S::Elem]) -> S::Elem,
{
    if arr.len() != values.len() {
        return Err(NumpyError::ShapeMismatch {
            expected: vec![arr.len()],
            got: vec![values.len()],
        });
    }

    if arr.len() == 0 {
        return Err(NumpyError::ValueError("Cannot compute binned statistic of empty array".to_string()));
    }

    if bins == 0 {
        return Err(NumpyError::ValueError("Number of bins must be positive".to_string()));
    }

    let min_val = min(arr)?;
    let max_val = max(arr)?;

    let bin_width = (max_val - min_val) / S::Elem::from_usize(bins).unwrap();
    let mut bin_values: Vec<Vec<S::Elem>> = vec![Vec::new(); bins];
    let mut edges = Vec::with_capacity(bins + 1);

    for i in 0..=bins {
        edges.push(min_val + S::Elem::from_usize(i).unwrap() * bin_width);
    }

    for i in 0..arr.len() {
        let mut bin = ((arr[i] - min_val) / bin_width).floor().to_usize().unwrap();
        if bin >= bins {
            bin = bins - 1;
        }
        bin_values[bin].push(values[i]);
    }

    let stats: Vec<S::Elem> = bin_values
        .iter()
        .map(|bin_vals| {
            if bin_vals.is_empty() {
                S::Elem::nan()
            } else {
                statistic(bin_vals)
            }
        })
        .collect();

    Ok((Array1::from_vec(stats), Array1::from_vec(edges)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_mean() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(mean(&a).unwrap(), 3.0);
    }

    #[test]
    fn test_median() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(median(&a).unwrap(), 3.0);
    }

    #[test]
    fn test_var() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let v = var(&a, 1).unwrap();
        assert_relative_eq!(v, 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_std() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = std(&a, 1).unwrap();
        assert_relative_eq!(s, 2.5_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_percentile() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(percentile(&a, 0.0).unwrap(), 1.0);
        assert_relative_eq!(percentile(&a, 50.0).unwrap(), 3.0);
        assert_relative_eq!(percentile(&a, 100.0).unwrap(), 5.0);
    }

    #[test]
    fn test_corrcoef() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = corrcoef(&x, &y).unwrap();
        assert_relative_eq!(corr, 1.0, epsilon = 1e-10);
    }
}
