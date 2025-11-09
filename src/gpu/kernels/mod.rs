//! WGSL compute shader kernels
//!
//! Contains WGSL shader code for various GPU operations.

/// Tiled matrix multiplication shader (16×16 workgroups)
pub const MATMUL_SHADER: &str = include_str!("matmul.wgsl");

/// Element-wise operations shader (sin, cos, exp, log, etc.)
pub const ELEMENTWISE_SHADER: &str = include_str!("elementwise.wgsl");
