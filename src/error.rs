//! Error types for NumPy Rust operations

use thiserror::Error;

#[derive(Error, Debug)]
pub enum NumpyError {
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid axis: {0}")]
    InvalidAxis(usize),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Linear algebra error: {0}")]
    LinalgError(String),

    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    #[error("Value error: {0}")]
    ValueError(String),

    #[error("Index out of bounds: {0}")]
    IndexError(String),

    #[error("Singular matrix")]
    SingularMatrix,

    #[error("Convergence error: {0}")]
    ConvergenceError(String),
}

pub type Result<T> = std::result::Result<T, NumpyError>;
