//! GPU-accelerated matrix multiplication

use ndarray::{Array2, ArrayBase, Data, Ix2};
use wgpu::util::DeviceExt;
use crate::gpu::context::GpuContext;
use crate::gpu::kernels::MATMUL_SHADER;
use crate::error::{NumpyError, Result};

/// GPU matrix multiplication: C = A × B
///
/// # Arguments
///
/// * `a` - Left matrix (M×K)
/// * `b` - Right matrix (K×N)
///
/// # Returns
///
/// Result matrix C (M×N)
///
/// # Performance
///
/// GPU acceleration is beneficial for matrices ≥512×512.
/// Smaller matrices should use CPU implementation.
pub async fn matmul_gpu<S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Result<Array2<f32>>
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
{
    let ctx = GpuContext::get_or_init()
        .ok_or_else(|| NumpyError::LinalgError("GPU not available".into()))?;

    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(NumpyError::InvalidShape(
            format!("Matrix dimensions don't match: ({}, {}) × ({}, {})", m, k1, k2, n)
        ));
    }
    let k = k1;

    // Convert to contiguous arrays for GPU upload
    let a_vec: Vec<f32> = if a.is_standard_layout() {
        a.as_slice().unwrap().to_vec()
    } else {
        a.iter().cloned().collect()
    };

    let b_vec: Vec<f32> = if b.is_standard_layout() {
        b.as_slice().unwrap().to_vec()
    } else {
        b.iter().cloned().collect()
    };

    // Create GPU buffers
    let a_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A"),
        contents: bytemuck::cast_slice(&a_vec),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B"),
        contents: bytemuck::cast_slice(&b_vec),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
    let c_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Dimensions uniform buffer (m, n, k, padding)
    let dims = [m as u32, n as u32, k as u32, 0u32];
    let dims_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dimensions"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shader module
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MatMul Shader"),
        source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MatMul Bind Group Layout"),
        entries: &[
            // Buffer A (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Buffer B (storage, read-only)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Buffer C (storage, read-write)
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Dimensions (uniform)
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("MatMul Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dims_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MatMul Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MatMul Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("matmul_tiled"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Execute compute pass
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MatMul Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MatMul Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups (16×16 threads per workgroup)
        let workgroups_x = (n as u32 + 15) / 16;
        let workgroups_y = (m as u32 + 15) / 16;
        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // Create staging buffer for readback
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: c_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy result to staging buffer
    encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, c_size);
    ctx.queue.submit(Some(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);

    // Use oneshot channel for async-friendly buffer mapping
    let (tx, rx) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });

    // In wgpu 27+, polling is handled automatically for WebGPU
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::MaintainResult::SubmissionQueueEmpty);

    // Await the future (works in both WASM and native)
    rx.await
        .map_err(|_| NumpyError::LinalgError("Failed to receive buffer mapping".into()))?
        .map_err(|e| NumpyError::LinalgError(format!("Buffer mapping failed: {:?}", e)))?;

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // Convert to Array2
    Array2::from_shape_vec((m, n), result)
        .map_err(|e| NumpyError::InvalidShape(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gpu_matmul_small() {
        if GpuContext::get_or_init().is_none() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // Small 4×4 matrices
        let a = Array2::from_shape_vec((4, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ]).unwrap();

        let b = Array2::from_shape_vec((4, 4), vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]).unwrap();

        let c = matmul_gpu(&a, &b).unwrap();

        // A × I = A
        for i in 0..4 {
            for j in 0..4 {
                assert_abs_diff_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn test_gpu_matmul_medium() {
        if GpuContext::get_or_init().is_none() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // 64×64 matrices
        let n = 64;
        let a = Array2::from_shape_fn((n, n), |(i, j)| (i + j) as f32);
        let b = Array2::from_shape_fn((n, n), |(i, j)| if i == j { 1.0 } else { 0.0 });

        let c = matmul_gpu(&a, &b).unwrap();

        // A × I = A
        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(c[[i, j]], a[[i, j]], epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn test_gpu_matmul_non_square() {
        if GpuContext::get_or_init().is_none() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        // 32×48 × 48×64 = 32×64
        let a = Array2::ones((32, 48));
        let b = Array2::ones((48, 64));

        let c = matmul_gpu(&a, &b).unwrap();

        assert_eq!(c.dim(), (32, 64));

        // All elements should be 48.0 (sum of 48 ones)
        for i in 0..32 {
            for j in 0..64 {
                assert_abs_diff_eq!(c[[i, j]], 48.0, epsilon = 1e-4);
            }
        }
    }
}
