//! GPU buffer management and pooling
//!
//! Provides utilities for creating and managing GPU buffers efficiently.

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device};
use crate::gpu::error::{GpuError, Result};

/// Helper for creating GPU buffers
pub struct BufferManager {
    device: std::sync::Arc<Device>,
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new(device: std::sync::Arc<Device>) -> Self {
        Self { device }
    }

    /// Create a storage buffer for read-only data
    pub fn create_storage_buffer(
        &self,
        label: Option<&str>,
        data: &[u8],
    ) -> Buffer {
        use wgpu::util::DeviceExt;

        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage: BufferUsages::STORAGE,
        })
    }

    /// Create a storage buffer for read-write data
    pub fn create_storage_buffer_rw(
        &self,
        label: Option<&str>,
        data: &[u8],
    ) -> Buffer {
        use wgpu::util::DeviceExt;

        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
    }

    /// Create an empty output buffer
    pub fn create_output_buffer(
        &self,
        label: Option<&str>,
        size: u64,
    ) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Create a staging buffer for reading back data
    pub fn create_staging_buffer(
        &self,
        label: Option<&str>,
        size: u64,
    ) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create a uniform buffer
    pub fn create_uniform_buffer(
        &self,
        label: Option<&str>,
        data: &[u8],
    ) -> Buffer {
        use wgpu::util::DeviceExt;

        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label,
            contents: data,
            usage: BufferUsages::UNIFORM,
        })
    }
}

/// Helper to read data back from GPU buffer
pub async fn read_buffer<T: bytemuck::Pod>(
    device: &Device,
    queue: &wgpu::Queue,
    source_buffer: &Buffer,
    size: usize,
) -> Result<Vec<T>> {
    let buffer_size = (size * std::mem::size_of::<T>()) as u64;

    // Create staging buffer
    let staging_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("Staging Buffer"),
        size: buffer_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy from source to staging
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Read Buffer Encoder"),
    });
    encoder.copy_buffer_to_buffer(source_buffer, 0, &staging_buffer, 0, buffer_size);
    queue.submit(Some(encoder.finish()));

    // Map and read
    let buffer_slice = staging_buffer.slice(..);

    // Use oneshot channel for async-friendly buffer mapping
    let (tx, rx) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // In wgpu 27+, polling is handled automatically for WebGPU
    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Poll);

    // Await the future (works in both WASM and native)
    rx.await
        .map_err(|_| GpuError::Buffer("Failed to receive buffer mapping result".into()))?
        .map_err(|e| GpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    Ok(result)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_buffer_manager_creation() {
        // Just test that the module compiles and structures are correct
    }
}
