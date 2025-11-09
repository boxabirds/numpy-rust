//! WASM bindings for browser usage
//!
//! This module provides JavaScript-compatible exports for GPU-accelerated operations.

use wasm_bindgen::prelude::*;
use js_sys::Float32Array;
use ndarray::Array2;

#[cfg(feature = "gpu")]
use crate::gpu::GpuContext;

// Initialize panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Initialize GPU context (must be called before GPU operations)
#[wasm_bindgen]
pub async fn init_gpu() -> Result<(), JsValue> {
    #[cfg(feature = "gpu")]
    {
        let ctx = GpuContext::init().await
            .map_err(|e| JsValue::from_str(&format!("Failed to initialize GPU: {}", e)))?;
        GpuContext::set(ctx);
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}

/// Check if GPU is available
#[wasm_bindgen]
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GpuContext::is_available()
    }

    #[cfg(not(feature = "gpu"))]
    false
}

/// Get GPU adapter information
#[wasm_bindgen]
pub fn get_gpu_info() -> Result<JsValue, JsValue> {
    #[cfg(feature = "gpu")]
    {
        let ctx = GpuContext::get_or_init()
            .ok_or_else(|| JsValue::from_str("GPU not initialized"))?;

        let info = ctx.adapter_info();
        let obj = js_sys::Object::new();

        js_sys::Reflect::set(&obj, &"name".into(), &info.name.clone().into())?;
        js_sys::Reflect::set(&obj, &"backend".into(), &format!("{:?}", info.backend).into())?;
        js_sys::Reflect::set(&obj, &"vendor".into(), &info.vendor.to_string().into())?;
        js_sys::Reflect::set(&obj, &"driver".into(), &info.driver.clone().into())?;

        Ok(obj.into())
    }

    #[cfg(not(feature = "gpu"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}

/// CPU matrix multiplication (for comparison)
#[wasm_bindgen]
pub fn matmul_cpu(a: Float32Array, b: Float32Array, n: usize) -> Result<Float32Array, JsValue> {
    let a_vec: Vec<f32> = a.to_vec();
    let b_vec: Vec<f32> = b.to_vec();

    let a_arr = Array2::from_shape_vec((n, n), a_vec)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let b_arr = Array2::from_shape_vec((n, n), b_vec)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use CPU-only matmul (avoid GPU dispatch)
    let result = a_arr.dot(&b_arr);

    let (result_vec, _offset) = result.into_raw_vec_and_offset();
    Ok(Float32Array::from(&result_vec[..]))
}

/// GPU matrix multiplication
#[wasm_bindgen]
pub async fn matmul_gpu(a: Float32Array, b: Float32Array, n: usize) -> Result<Float32Array, JsValue> {
    #[cfg(feature = "gpu")]
    {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();

        let a_arr = Array2::from_shape_vec((n, n), a_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let b_arr = Array2::from_shape_vec((n, n), b_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = crate::gpu::ops::matmul::matmul_gpu(&a_arr, &b_arr).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let (result_vec, _offset) = result.into_raw_vec_and_offset();
        Ok(Float32Array::from(&result_vec[..]))
    }

    #[cfg(not(feature = "gpu"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}

/// GPU element-wise sin
#[wasm_bindgen]
pub async fn sin_gpu(input: Float32Array) -> Result<Float32Array, JsValue> {
    #[cfg(feature = "gpu")]
    {
        use crate::gpu::ops::elementwise::{elementwise_gpu, ElementWiseOp};
        use ndarray::Array1;

        let input_vec: Vec<f32> = input.to_vec();
        let input_arr = Array1::from(input_vec);

        let result = elementwise_gpu(&input_arr, ElementWiseOp::Sin).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let (result_vec, _offset) = result.into_raw_vec_and_offset();
        Ok(Float32Array::from(&result_vec[..]))
    }

    #[cfg(not(feature = "gpu"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}

/// GPU element-wise exp
#[wasm_bindgen]
pub async fn exp_gpu(input: Float32Array) -> Result<Float32Array, JsValue> {
    #[cfg(feature = "gpu")]
    {
        use crate::gpu::ops::elementwise::{elementwise_gpu, ElementWiseOp};
        use ndarray::Array1;

        let input_vec: Vec<f32> = input.to_vec();
        let input_arr = Array1::from(input_vec);

        let result = elementwise_gpu(&input_arr, ElementWiseOp::Exp).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let (result_vec, _offset) = result.into_raw_vec_and_offset();
        Ok(Float32Array::from(&result_vec[..]))
    }

    #[cfg(not(feature = "gpu"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}

/// CPU element-wise sin (for comparison)
#[wasm_bindgen]
pub fn sin_cpu(input: Float32Array) -> Result<Float32Array, JsValue> {
    let input_vec: Vec<f32> = input.to_vec();
    let result_vec: Vec<f32> = input_vec.iter().map(|x| x.sin()).collect();
    Ok(Float32Array::from(&result_vec[..]))
}

/// CPU element-wise exp (for comparison)
#[wasm_bindgen]
pub fn exp_cpu(input: Float32Array) -> Result<Float32Array, JsValue> {
    let input_vec: Vec<f32> = input.to_vec();
    let result_vec: Vec<f32> = input_vec.iter().map(|x| x.exp()).collect();
    Ok(Float32Array::from(&result_vec[..]))
}

/// Generate random matrix for testing
#[wasm_bindgen]
pub fn generate_random_matrix(rows: usize, cols: usize) -> Float32Array {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let size = rows * cols;
    let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();
    Float32Array::from(&data[..])
}

/// Generate random array for testing
#[wasm_bindgen]
pub fn generate_random_array(size: usize) -> Float32Array {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();
    Float32Array::from(&data[..])
}
