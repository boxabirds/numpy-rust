//! GPU-specific error types

use thiserror::Error;

/// Result type for GPU operations
pub type Result<T> = std::result::Result<T, GpuError>;

/// GPU-specific errors
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("No GPU adapter found")]
    NoAdapter,

    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("Buffer error: {0}")]
    Buffer(String),

    #[error("Pipeline compilation error: {0}")]
    Pipeline(String),

    #[error("Shader error: {0}")]
    Shader(String),

    #[error("GPU operation failed: {0}")]
    Operation(String),

    #[error("GPU context not initialized")]
    NotInitialized,

    #[error("Unsupported operation on GPU: {0}")]
    Unsupported(String),
}
