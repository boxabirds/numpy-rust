//! GPU context management
//!
//! Manages the global GPU device, queue, and initialization.

use wgpu::{Device, Queue, Instance, ComputePipeline, BindGroupLayout, ShaderModule};
use crate::gpu::error::{GpuError, Result};
use std::collections::HashMap;

// For native targets, use static OnceCell (thread-safe)
#[cfg(not(target_arch = "wasm32"))]
use once_cell::sync::OnceCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Arc, Mutex};

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

/// Pipeline cache entry
pub struct PipelineCache {
    pub shader: ShaderModule,
    pub bind_group_layout: BindGroupLayout,
    pub pipeline: ComputePipeline,
}

/// GPU context containing device, queue, and adapter info
pub struct GpuContext {
    /// WGPU device for creating resources
    pub device: Device,
    /// WGPU queue for submitting commands
    pub queue: Queue,
    /// Information about the GPU adapter
    adapter_info: wgpu::AdapterInfo,
    /// Cached compute pipelines (native: thread-safe, WASM: single-threaded)
    #[cfg(not(target_arch = "wasm32"))]
    pipeline_cache: Mutex<HashMap<String, PipelineCache>>,
    #[cfg(target_arch = "wasm32")]
    pipeline_cache: RefCell<HashMap<String, PipelineCache>>,
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

        // Query adapter limits to request maximum buffer sizes
        let adapter_limits = adapter.limits();

        // For WebGPU, start with downlevel defaults but request higher buffer sizes
        let limits = if cfg!(target_arch = "wasm32") {
            let mut limits = wgpu::Limits::downlevel_defaults();

            // Request the maximum buffer size supported by the adapter (up to 4GB)
            // This is needed for large benchmarks (100M elements = 400MB)
            limits.max_buffer_size = adapter_limits.max_buffer_size;
            limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;

            limits
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
            pipeline_cache: Mutex::new(HashMap::new()),
        }));

        #[cfg(target_arch = "wasm32")]
        Ok(Rc::new(Self {
            device,
            queue,
            adapter_info,
            pipeline_cache: RefCell::new(HashMap::new()),
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

    /// Get cached pipeline or insert new one, then execute operation with it
    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_pipeline<F, R, C>(&self, key: &str, create_fn: C, use_fn: F) -> R
    where
        F: FnOnce(&PipelineCache) -> R,
        C: FnOnce() -> PipelineCache,
    {
        let mut cache = self.pipeline_cache.lock().unwrap();
        let entry = cache.entry(key.to_string()).or_insert_with(create_fn);
        use_fn(entry)
    }

    /// Get cached pipeline or insert new one, then execute operation with it (WASM version)
    #[cfg(target_arch = "wasm32")]
    pub fn with_pipeline<F, R, C>(&self, key: &str, create_fn: C, use_fn: F) -> R
    where
        F: FnOnce(&PipelineCache) -> R,
        C: FnOnce() -> PipelineCache,
    {
        let mut cache = self.pipeline_cache.borrow_mut();
        let entry = cache.entry(key.to_string()).or_insert_with(create_fn);
        use_fn(entry)
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
