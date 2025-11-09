//! Sorting, searching, and ordering functions
//!
//! This module provides functions for sorting and searching arrays,
//! similar to NumPy's sorting functions.

use ndarray::{Array1, ArrayBase, Data, Ix1};
use num_traits::{Float, Num, Zero};
use crate::error::{NumpyError, Result};

/// Sort an array in-place
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::sort;
/// use ndarray::array;
///
/// let mut a = array![3.0, 1.0, 2.0];
/// sort(&mut a);
/// assert_eq!(a, array![1.0, 2.0, 3.0]);
/// ```
pub fn sort<T: PartialOrd>(arr: &mut Array1<T>) {
    arr.as_slice_mut()
        .unwrap()
        .sort_by(|a, b| a.partial_cmp(b).unwrap());
}

/// Return a sorted copy of an array
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::sorted;
/// use ndarray::array;
///
/// let a = array![3.0, 1.0, 2.0];
/// let b = sorted(&a);
/// assert_eq!(b, array![1.0, 2.0, 3.0]);
/// ```
pub fn sorted<S>(arr: &ArrayBase<S, Ix1>) -> Array1<S::Elem>
where
    S: Data,
    S::Elem: Clone + PartialOrd,
{
    let mut result = arr.to_owned();
    sort(&mut result);
    result
}

/// Return the indices that would sort an array
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::argsort;
/// use ndarray::array;
///
/// let a = array![3.0, 1.0, 2.0];
/// let indices = argsort(&a);
/// assert_eq!(indices, array![1, 2, 0]);
/// ```
pub fn argsort<S>(arr: &ArrayBase<S, Ix1>) -> Array1<usize>
where
    S: Data,
    S::Elem: PartialOrd,
{
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_by(|&i, &j| arr[i].partial_cmp(&arr[j]).unwrap());
    Array1::from_vec(indices)
}

/// Partition array around the k-th element
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::partition;
/// use ndarray::array;
///
/// let mut a = array![3.0, 1.0, 2.0, 5.0, 4.0];
/// partition(&mut a, 2).unwrap();
/// // After partitioning, elements before index 2 are <= a[2],
/// // and elements after are >= a[2]
/// ```
pub fn partition<T: PartialOrd + Clone>(arr: &mut Array1<T>, kth: usize) -> Result<()> {
    if kth >= arr.len() {
        return Err(NumpyError::IndexError(
            format!("kth index {} out of bounds for array of length {}", kth, arr.len())
        ));
    }

    let slice = arr.as_slice_mut().unwrap();
    // Simple selection algorithm
    pdqsort::select(slice, kth);
    Ok(())
}

/// Return indices that would partition the array
pub fn argpartition<S>(arr: &ArrayBase<S, Ix1>, kth: usize) -> Result<Array1<usize>>
where
    S: Data,
    S::Elem: PartialOrd,
{
    if kth >= arr.len() {
        return Err(NumpyError::IndexError(
            format!("kth index {} out of bounds for array of length {}", kth, arr.len())
        ));
    }

    let mut indices: Vec<usize> = (0..arr.len()).collect();
    pdqsort::select_by(&mut indices, kth, |&i, &j| {
        arr[i].partial_cmp(&arr[j]).unwrap()
    });

    Ok(Array1::from_vec(indices))
}

/// Find indices where elements should be inserted to maintain order
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::searchsorted;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 5.0];
/// let idx = searchsorted(&a, 4.0, true).unwrap();
/// assert_eq!(idx, 3);
/// ```
pub fn searchsorted<S, T>(arr: &ArrayBase<S, Ix1>, value: T, side_left: bool) -> Result<usize>
where
    S: Data<Elem = T>,
    T: PartialOrd,
{
    if side_left {
        // Find leftmost insertion point
        match arr.as_slice().unwrap().binary_search_by(|x| {
            x.partial_cmp(&value).unwrap()
        }) {
            Ok(idx) => {
                // Found exact match, find leftmost occurrence
                let mut left = idx;
                while left > 0 && arr[left - 1].partial_cmp(&value).unwrap() == std::cmp::Ordering::Equal {
                    left -= 1;
                }
                Ok(left)
            }
            Err(idx) => Ok(idx),
        }
    } else {
        // Find rightmost insertion point
        match arr.as_slice().unwrap().binary_search_by(|x| {
            if x.partial_cmp(&value).unwrap() == std::cmp::Ordering::Greater {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Less
            }
        }) {
            Ok(idx) => Ok(idx),
            Err(idx) => Ok(idx),
        }
    }
}

/// Extract unique elements from an array
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::unique;
/// use ndarray::array;
///
/// let a = array![1, 2, 2, 3, 1, 4];
/// let u = unique(&a);
/// assert_eq!(u, array![1, 2, 3, 4]);
/// ```
pub fn unique<S>(arr: &ArrayBase<S, Ix1>) -> Array1<S::Elem>
where
    S: Data,
    S::Elem: Clone + PartialOrd,
{
    let mut values: Vec<S::Elem> = arr.iter().cloned().collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values.dedup();
    Array1::from_vec(values)
}

/// Extract unique elements and their counts
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::unique_counts;
/// use ndarray::array;
///
/// let a = array![1, 2, 2, 3, 1, 4];
/// let (values, counts) = unique_counts(&a);
/// ```
pub fn unique_counts<S>(arr: &ArrayBase<S, Ix1>) -> (Array1<S::Elem>, Array1<usize>)
where
    S: Data,
    S::Elem: Clone + PartialOrd,
{
    let mut values: Vec<S::Elem> = arr.iter().cloned().collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut unique_vals = Vec::new();
    let mut counts = Vec::new();

    if values.is_empty() {
        return (Array1::from_vec(unique_vals), Array1::from_vec(counts));
    }

    let mut current = values[0].clone();
    let mut count = 1;

    for i in 1..values.len() {
        if values[i].partial_cmp(&current).unwrap() == std::cmp::Ordering::Equal {
            count += 1;
        } else {
            unique_vals.push(current);
            counts.push(count);
            current = values[i].clone();
            count = 1;
        }
    }

    unique_vals.push(current);
    counts.push(count);

    (Array1::from_vec(unique_vals), Array1::from_vec(counts))
}

/// Find the k largest elements
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::largest;
/// use ndarray::array;
///
/// let a = array![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let top3 = largest(&a, 3).unwrap();
/// assert_eq!(top3, array![9.0, 5.0, 4.0]);
/// ```
pub fn largest<S>(arr: &ArrayBase<S, Ix1>, k: usize) -> Result<Array1<S::Elem>>
where
    S: Data,
    S::Elem: Clone + PartialOrd,
{
    if k > arr.len() {
        return Err(NumpyError::ValueError(
            format!("k ({}) cannot be larger than array length ({})", k, arr.len())
        ));
    }

    let mut sorted = sorted(arr);
    sorted.as_slice_mut().unwrap().reverse();
    Ok(sorted.slice(ndarray::s![..k]).to_owned())
}

/// Find the k smallest elements
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::smallest;
/// use ndarray::array;
///
/// let a = array![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
/// let bottom3 = smallest(&a, 3).unwrap();
/// assert_eq!(bottom3, array![1.0, 1.0, 2.0]);
/// ```
pub fn smallest<S>(arr: &ArrayBase<S, Ix1>, k: usize) -> Result<Array1<S::Elem>>
where
    S: Data,
    S::Elem: Clone + PartialOrd,
{
    if k > arr.len() {
        return Err(NumpyError::ValueError(
            format!("k ({}) cannot be larger than array length ({})", k, arr.len())
        ));
    }

    let sorted = sorted(arr);
    Ok(sorted.slice(ndarray::s![..k]).to_owned())
}

/// Find indices of non-zero elements
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::nonzero;
/// use ndarray::array;
///
/// let a = array![0.0, 1.0, 0.0, 3.0, 0.0];
/// let indices = nonzero(&a);
/// assert_eq!(indices, array![1, 3]);
/// ```
pub fn nonzero<S>(arr: &ArrayBase<S, Ix1>) -> Array1<usize>
where
    S: Data,
    S::Elem: Num + PartialEq + Zero + Copy,
{
    let zero = Zero::zero();
    let indices: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| {
            if val != zero {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    Array1::from_vec(indices)
}

/// Find indices where condition is True
///
/// # Examples
///
/// ```
/// use numpy_rust::sorting::where_cond;
/// use ndarray::array;
///
/// let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let indices = where_cond(&a, |&x| x > 3.0);
/// assert_eq!(indices, array![3, 4]);
/// ```
pub fn where_cond<S, F>(arr: &ArrayBase<S, Ix1>, condition: F) -> Array1<usize>
where
    S: Data,
    F: Fn(&S::Elem) -> bool,
{
    let indices: Vec<usize> = arr
        .iter()
        .enumerate()
        .filter_map(|(i, val)| {
            if condition(val) {
                Some(i)
            } else {
                None
            }
        })
        .collect();

    Array1::from_vec(indices)
}

// Add pdqsort as a simple implementation helper
mod pdqsort {
    pub fn select<T: PartialOrd>(arr: &mut [T], k: usize) {
        if arr.is_empty() || k >= arr.len() {
            return;
        }
        arr.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).unwrap());
    }

    pub fn select_by<T, F>(arr: &mut [T], k: usize, mut compare: F)
    where
        F: FnMut(&T, &T) -> std::cmp::Ordering,
    {
        if arr.is_empty() || k >= arr.len() {
            return;
        }
        arr.select_nth_unstable_by(k, &mut compare);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sort() {
        let mut a = array![3.0, 1.0, 2.0];
        sort(&mut a);
        assert_eq!(a, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_argsort() {
        let a = array![3.0, 1.0, 2.0];
        let indices = argsort(&a);
        assert_eq!(indices, array![1, 2, 0]);
    }

    #[test]
    fn test_unique() {
        let a = array![1, 2, 2, 3, 1, 4];
        let u = unique(&a);
        assert_eq!(u, array![1, 2, 3, 4]);
    }

    #[test]
    fn test_unique_counts() {
        let a = array![1, 2, 2, 3, 1];
        let (values, counts) = unique_counts(&a);
        assert_eq!(values, array![1, 2, 3]);
        assert_eq!(counts, array![2, 2, 1]);
    }

    #[test]
    fn test_largest() {
        let a = array![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let top3 = largest(&a, 3).unwrap();
        assert_eq!(top3, array![9.0, 5.0, 4.0]);
    }

    #[test]
    fn test_nonzero() {
        let a = array![0.0, 1.0, 0.0, 3.0, 0.0];
        let indices = nonzero(&a);
        assert_eq!(indices, array![1, 3]);
    }
}
