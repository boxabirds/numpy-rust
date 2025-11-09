//! GPU context management
//!
//! Manages the global GPU device, queue, and initialization.

use once_cell::sync::OnceCell;
use std::sync::Arc;
use wgpu::{Device, Queue, Instance};
use crate::gpu::error::{GpuError, Result};

/// Global GPU context (lazy initialized on first use)
static GPU_CONTEXT: OnceCell<Option<Arc<GpuContext>>> = OnceCell::new();

/// GPU context containing device, queue, and adapter info
pub struct GpuContext {
    /// WGPU device for creating resources
    pub device: Device,
    /// WGPU queue for submitting commands
    pub queue: Queue,
    /// Information about the GPU adapter
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize GPU context
    ///
    /// This is called automatically on first use. It selects the highest
    /// performance GPU available and requests a device.
    pub async fn init() -> Result<Arc<Self>> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("numpy-rust GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await?;

        println!(
            "🎮 GPU initialized: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        Ok(Arc::new(Self {
            device,
            queue,
            adapter_info,
        }))
    }

    /// Get or initialize the global GPU context
    ///
    /// Returns None if GPU initialization failed (graceful fallback)
    pub fn get_or_init() -> Option<Arc<Self>> {
        GPU_CONTEXT
            .get_or_init(|| {
                // Try to initialize GPU, return None on failure
                pollster::block_on(Self::init()).ok()
            })
            .clone()
    }

    /// Check if GPU is available
    ///
    /// Returns true if GPU context was successfully initialized
    pub fn is_available() -> bool {
        Self::get_or_init().is_some()
    }

    /// Get GPU adapter information
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Get device name
    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Get backend type (Vulkan, Metal, DX12, etc.)
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        // Just check that this doesn't panic
        let available = GpuContext::is_available();
        println!("GPU available: {}", available);
    }

    #[test]
    fn test_gpu_info() {
        if let Some(ctx) = GpuContext::get_or_init() {
            println!("GPU: {}", ctx.device_name());
            println!("Backend: {:?}", ctx.backend());
            println!("Vendor: {}", ctx.adapter_info.vendor);
        }
    }
}
