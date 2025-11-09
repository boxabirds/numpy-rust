//! Smart CPU/GPU dispatch logic
//!
//! Determines when to use GPU acceleration based on operation type and data size.

use crate::gpu::GpuContext;

/// Operation types that can be GPU accelerated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpType {
    /// Matrix multiplication
    MatMul,
    /// Element-wise operations (sin, cos, exp, etc.)
    ElementWise,
    /// FFT transform
    Fft,
    /// Reduction operations (sum, mean, etc.)
    Reduction,
}

/// Compute backend selection
#[derive(Debug, Clone)]
pub enum ComputeBackend {
    /// Use CPU implementation
    Cpu,
    /// Use GPU acceleration
    Gpu,
}

impl ComputeBackend {
    /// Automatically select best backend for operation
    pub fn auto_select(
        op_type: OpType,
        data_size: usize,
        data_shape: &[usize],
    ) -> Self {
        // Check if GPU is available
        if !GpuContext::is_available() {
            return ComputeBackend::Cpu;
        }

        // Check if operation benefits from GPU
        if should_use_gpu(op_type, data_size, data_shape) {
            ComputeBackend::Gpu
        } else {
            ComputeBackend::Cpu
        }
    }
}

/// Thresholds for GPU acceleration
const MATMUL_GPU_THRESHOLD: usize = 512;
const ELEMENTWISE_GPU_THRESHOLD: usize = 100_000;
const FFT_GPU_THRESHOLD: usize = 8192;
const REDUCTION_GPU_THRESHOLD: usize = 1_000_000;

/// Determine if GPU acceleration is beneficial
///
/// # Arguments
///
/// * `op_type` - Type of operation
/// * `data_size` - Total number of elements
/// * `shape` - Shape of the data
///
/// # Returns
///
/// `true` if GPU is likely to be faster, `false` otherwise
pub fn should_use_gpu(
    op_type: OpType,
    data_size: usize,
    shape: &[usize],
) -> bool {
    match op_type {
        OpType::MatMul => {
            // GPU beneficial for matrices ≥512×512
            // Assumes 2D matrices
            if shape.len() == 2 {
                shape[0] >= MATMUL_GPU_THRESHOLD && shape[1] >= MATMUL_GPU_THRESHOLD
            } else {
                false
            }
        }
        OpType::ElementWise => {
            // GPU beneficial for arrays ≥100K elements
            data_size >= ELEMENTWISE_GPU_THRESHOLD
        }
        OpType::Fft => {
            // GPU beneficial for FFTs ≥8192 length
            data_size >= FFT_GPU_THRESHOLD
        }
        OpType::Reduction => {
            // GPU beneficial for large reductions ≥1M elements
            data_size >= REDUCTION_GPU_THRESHOLD
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_threshold() {
        // Small matrix - should use CPU
        assert!(!should_use_gpu(OpType::MatMul, 256 * 256, &[256, 256]));

        // Large matrix - should use GPU (if available)
        assert!(should_use_gpu(OpType::MatMul, 1024 * 1024, &[1024, 1024]));
    }

    #[test]
    fn test_elementwise_threshold() {
        // Small array - CPU
        assert!(!should_use_gpu(OpType::ElementWise, 10_000, &[10_000]));

        // Large array - GPU
        assert!(should_use_gpu(OpType::ElementWise, 200_000, &[200_000]));
    }

    #[test]
    fn test_auto_select() {
        let backend = ComputeBackend::auto_select(OpType::MatMul, 512 * 512, &[512, 512]);
        // Will be CPU if no GPU, GPU if available
        match backend {
            ComputeBackend::Cpu => println!("Selected CPU (GPU not available or below threshold)"),
            ComputeBackend::Gpu => println!("Selected GPU"),
        }
    }
}
