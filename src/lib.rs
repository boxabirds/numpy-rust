//! NumPy Rust - A comprehensive Rust port of NumPy built on ndarray
//!
//! This library provides a NumPy-like interface for numerical computing in Rust,
//! leveraging the ndarray ecosystem for efficient multi-dimensional array operations.
//!
//! # Features
//!
//! - Array creation routines
//! - Mathematical functions
//! - Linear algebra operations
//! - Statistics functions
//! - Random number generation
//! - FFT operations
//! - Sorting and searching
//! - Broadcasting and indexing
//!
//! # Examples
//!
//! ```
//! use numpy_rust::prelude::*;
//!
//! // Create arrays
//! let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
//! let b = zeros(5);
//! let c = linspace(0.0, 10.0, 50);
//!
//! // Mathematical operations
//! let sin_values = a.mapv(f64::sin);
//! ```

pub mod array;
pub mod linalg;
pub mod math;
pub mod random;
pub mod stats;
pub mod fft;
pub mod sorting;
pub mod error;

// Re-export commonly used types and functions
pub mod prelude {
    pub use ndarray::{array, s, Array, Array1, Array2, ArrayD, Axis, Dim, Ix1, Ix2, IxDyn};
    pub use crate::array::*;
    pub use crate::math::*;
    pub use crate::linalg;
    pub use crate::random;
    pub use crate::stats;
    pub use crate::fft;
    pub use crate::sorting;
    pub use crate::error::NumpyError;
}

pub use error::NumpyError;

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_array_creation() {
        let a = array![1.0, 2.0, 3.0];
        assert_eq!(a.len(), 3);
    }
}
