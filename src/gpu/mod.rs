//! GPU acceleration module using WebGPU/wgpu
//!
//! This module provides GPU-accelerated operations for large-scale computations.
//! GPU acceleration is optional and controlled by the `gpu` feature flag.
//!
//! # Features
//!
//! - Automatic CPU/GPU dispatch based on operation size
//! - Graceful fallback to CPU when GPU is unavailable
//! - Cross-platform support (Vulkan, Metal, DX12, WebGPU)
//! - Browser support via WebAssembly (with `gpu-web` feature)
//!
//! # Examples
//!
//! ```ignore
//! use numpy_rust::gpu::GpuContext;
//!
//! // GPU context is initialized automatically on first use
//! if GpuContext::is_available() {
//!     println!("GPU acceleration available!");
//! }
//! ```

pub mod context;
pub mod buffer;
pub mod pipeline;
pub mod dispatch;
pub mod error;
pub mod ops;
pub mod kernels;

// Re-export commonly used types
pub use context::GpuContext;
pub use error::{GpuError, Result};
pub use dispatch::{ComputeBackend, should_use_gpu};
