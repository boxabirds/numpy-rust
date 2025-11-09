//! GPU context management
//!
//! Manages the global GPU device, queue, and initialization.

use wgpu::{Device, Queue, Instance};
use crate::gpu::error::{GpuError, Result};

// For native targets, use static OnceCell (thread-safe)
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

// For WASM targets, use thread-local storage (WASM is single-threaded)
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

// Native: Global GPU context (lazy initialized on first use)
#[cfg(not(target_arch = "wasm32"))]
static GPU_CONTEXT: OnceCell<Option<Arc<GpuContext>>> = OnceCell::new();

// WASM: Thread-local GPU context (WASM is single-threaded, wgpu types are !Send)
#[cfg(target_arch = "wasm32")]
thread_local! {
    static GPU_CONTEXT: RefCell<Option<Rc<GpuContext>>> = RefCell::new(None);
}

// Type alias for the reference counted type (Arc for native, Rc for WASM)
#[cfg(not(target_arch = "wasm32"))]
type ContextPtr = Arc<GpuContext>;
#[cfg(target_arch = "wasm32")]
type ContextPtr = Rc<GpuContext>;

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
    pub async fn init() -> Result<ContextPtr> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
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
            .map_err(|_| GpuError::NoAdapter)?;

        let adapter_info = adapter.get_info();

        // For WebGPU, use downlevel defaults for maximum compatibility
        let limits = if cfg!(target_arch = "wasm32") {
            wgpu::Limits::downlevel_defaults()
        } else {
            wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("numpy-rust GPU"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::default(),
                    experimental_features: Default::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await?;

        #[cfg(not(target_arch = "wasm32"))]
        println!(
            "🎮 GPU initialized: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        #[cfg(not(target_arch = "wasm32"))]
        return Ok(Arc::new(Self {
            device,
            queue,
            adapter_info,
        }));

        #[cfg(target_arch = "wasm32")]
        Ok(Rc::new(Self {
            device,
            queue,
            adapter_info,
        }))
    }

    /// Get or initialize the global GPU context
    ///
    /// Returns None if GPU initialization failed (graceful fallback)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get_or_init() -> Option<ContextPtr> {
        GPU_CONTEXT
            .get_or_init(|| {
                // Try to initialize GPU, return None on failure
                pollster::block_on(Self::init()).ok()
            })
            .clone()
    }

    /// Get or initialize the global GPU context (WASM version)
    ///
    /// Returns None if GPU initialization failed (graceful fallback)
    #[cfg(target_arch = "wasm32")]
    pub fn get_or_init() -> Option<ContextPtr> {
        GPU_CONTEXT.with(|cell| {
            let context = cell.borrow();
            context.clone()
        })
    }

    /// Set the GPU context (WASM only, called from async init_gpu)
    #[cfg(target_arch = "wasm32")]
    pub fn set(ctx: ContextPtr) {
        GPU_CONTEXT.with(|cell| {
            *cell.borrow_mut() = Some(ctx);
        });
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
