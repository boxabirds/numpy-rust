//! GPU-accelerated element-wise operations

use ndarray::{Array, ArrayBase, Data, Dimension};
use wgpu::util::DeviceExt;
use crate::gpu::context::GpuContext;
use crate::gpu::kernels::ELEMENTWISE_SHADER;
use crate::error::{NumpyError, Result};

/// Operation type for element-wise GPU operations
#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum ElementWiseOp {
    Sin = 0,
    Cos = 1,
    Tan = 2,
    Exp = 3,
    Log = 4,
    Sqrt = 5,
    Abs = 6,
    Square = 7,
    Reciprocal = 8,
}

/// GPU element-wise operation
///
/// # Arguments
///
/// * `arr` - Input array
/// * `op` - Operation to perform
///
/// # Returns
///
/// Result array with operation applied element-wise
///
/// # Performance
///
/// GPU acceleration is beneficial for arrays ≥100,000 elements.
pub async fn elementwise_gpu<S, D>(
    arr: &ArrayBase<S, D>,
    op: ElementWiseOp,
) -> Result<Array<f32, D>>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    let ctx = GpuContext::get_or_init()
        .ok_or_else(|| NumpyError::LinalgError("GPU not available".into()))?;

    let size = arr.len();

    // Convert to contiguous array for GPU upload
    let data_vec: Vec<f32> = if arr.as_slice_memory_order().is_some() {
        arr.as_slice_memory_order().unwrap().to_vec()
    } else {
        arr.iter().cloned().collect()
    };

    // Create GPU buffers
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Element-wise Input"),
        contents: bytemuck::cast_slice(&data_vec),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_size = (size * std::mem::size_of::<f32>()) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Element-wise Output"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Parameters: size, op_type, padding
    let params = [size as u32, op as u32, 0u32, 0u32];
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Element-wise Params"),
        contents: bytemuck::cast_slice(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create shader module
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Element-wise Shader"),
        source: wgpu::ShaderSource::Wgsl(ELEMENTWISE_SHADER.into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Element-wise Bind Group Layout"),
        entries: &[
            // Input buffer
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
            // Output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Parameters
            wgpu::BindGroupLayoutEntry {
                binding: 2,
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
        label: Some("Element-wise Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Element-wise Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Element-wise Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("elementwise_op"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Execute compute pass
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Element-wise Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Element-wise Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch workgroups (256 threads per workgroup)
        let workgroups = (size as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    // Create staging buffer for readback
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Element-wise Staging Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy result to staging buffer
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
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

    // Convert back to original shape
    Array::from_shape_vec(arr.raw_dim(), result)
        .map_err(|e| NumpyError::InvalidShape(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gpu_sin() {
        if GpuContext::get_or_init().is_none() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let arr = Array1::from(data);

        let result = elementwise_gpu(&arr, ElementWiseOp::Sin).unwrap();

        for i in 0..100 {
            let expected = (i as f32 * 0.01).sin();
            assert_abs_diff_eq!(result[i], expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_gpu_exp() {
        if GpuContext::get_or_init().is_none() {
            eprintln!("GPU not available, skipping test");
            return;
        }

        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.01).collect();
        let arr = Array1::from(data);

        let result = elementwise_gpu(&arr, ElementWiseOp::Exp).unwrap();

        for i in 0..100 {
            let expected = (i as f32 * 0.01).exp();
            assert_abs_diff_eq!(result[i], expected, epsilon = 1e-4);
        }
    }
}
