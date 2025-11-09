# WebGPU Implementation Plan

**Status:** Planning
**Target:** Add WebGPU acceleration with interactive browser demo
**Timeline:** 6 weeks
**Priority:** High impact (100-1000x speedup potential)

---

## Overview

Add WebGPU/wgpu GPU acceleration to numpy-rust with a focus on:
1. **Native GPU acceleration** - Desktop applications
2. **Browser support** - WebAssembly + WebGPU
3. **Interactive demo** - Bun + React app for testing and benchmarking

---

## Phase 1: Foundation & Infrastructure (Week 1)

### 1.1 Add GPU Dependencies

**File:** `Cargo.toml`

```toml
[dependencies]
# Existing dependencies...
wgpu = { version = "0.19", optional = true }
bytemuck = { version = "1.14", optional = true, features = ["derive"] }
pollster = { version = "0.3", optional = true }

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = { version = "0.2", optional = true }
wasm-bindgen-futures = { version = "0.4", optional = true }
web-sys = { version = "0.3", optional = true, features = ["Window", "Navigator", "Gpu"] }
console_error_panic_hook = { version = "0.1", optional = true }

[features]
default = []
linalg = ["ndarray-linalg"]
gpu = ["wgpu", "bytemuck", "pollster"]
gpu-web = ["gpu", "wasm-bindgen", "wasm-bindgen-futures", "web-sys", "console_error_panic_hook"]

[lib]
crate-type = ["cdylib", "rlib"]
```

### 1.2 Create GPU Module Structure

**Files to create:**
```
src/gpu/
├── mod.rs              # Public API and feature gates
├── context.rs          # GPU context management
├── buffer.rs           # Buffer pool and memory management
├── pipeline.rs         # Pipeline caching
├── dispatch.rs         # Smart GPU/CPU selection
└── kernels/
    ├── mod.rs
    ├── matmul.wgsl     # Matrix multiplication shader
    ├── elementwise.wgsl # Element-wise operations
    ├── reduction.wgsl   # Sum, mean, etc.
    └── fft.wgsl        # FFT shader
```

### 1.3 Implement GPU Context

**File:** `src/gpu/context.rs`

```rust
use once_cell::sync::OnceCell;
use std::sync::Arc;
use wgpu::{Device, Queue, Instance, Adapter};

static GPU_CONTEXT: OnceCell<Option<Arc<GpuContext>>> = OnceCell::new();

pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Initialize GPU context (call once)
    pub async fn init() -> Result<Arc<Self>, GpuError> {
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
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                },
                None,
            )
            .await?;

        println!("🎮 GPU initialized: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        Ok(Arc::new(Self {
            device,
            queue,
            adapter_info,
        }))
    }

    /// Get or initialize global GPU context
    pub fn get_or_init() -> Option<Arc<Self>> {
        GPU_CONTEXT
            .get_or_init(|| {
                // Try to initialize GPU
                pollster::block_on(Self::init()).ok()
            })
            .clone()
    }

    /// Check if GPU is available
    pub fn is_available() -> bool {
        Self::get_or_init().is_some()
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("No GPU adapter found")]
    NoAdapter,
    #[error("Failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),
    #[error("Buffer error: {0}")]
    Buffer(String),
}
```

---

## Phase 2: Matrix Multiplication GPU Kernel (Week 2)

### 2.1 Implement Tiled MatMul Shader

**File:** `src/gpu/kernels/matmul.wgsl`

```wgsl
// Tiled matrix multiplication with shared memory
// Computes C = A × B where A is MxK, B is KxN, C is MxN

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // m, n, k

// Shared memory tiles (16×16 = 256 elements each)
var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16, 1)
fn matmul_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;

    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    // Early exit for out-of-bounds threads
    if (row >= m || col >= n) {
        return;
    }

    var sum = 0.0;

    // Number of tiles needed to cover K dimension
    let num_tiles = (k + TILE_SIZE - 1u) / TILE_SIZE;

    // Process tiles
    for (var t = 0u; t < num_tiles; t = t + 1u) {
        let k_offset = t * TILE_SIZE;

        // Load tile of A into shared memory
        let a_col = k_offset + local_col;
        if (a_col < k) {
            tile_a[local_row * TILE_SIZE + local_col] = a[row * k + a_col];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile of B into shared memory
        let b_row = k_offset + local_row;
        if (b_row < k) {
            tile_b[local_row * TILE_SIZE + local_col] = b[b_row * n + col];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize workgroup
        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var i = 0u; i < TILE_SIZE; i = i + 1u) {
            sum = sum + tile_a[local_row * TILE_SIZE + i] *
                       tile_b[i * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    c[row * n + col] = sum;
}
```

### 2.2 Rust Integration

**File:** `src/gpu/ops/matmul.rs`

```rust
use ndarray::{Array2, ArrayBase, Data, Ix2};
use wgpu::util::DeviceExt;
use bytemuck;
use crate::gpu::context::GpuContext;
use crate::error::Result;

/// GPU matrix multiplication: C = A × B
pub fn matmul_gpu<S1, S2>(
    a: &ArrayBase<S1, Ix2>,
    b: &ArrayBase<S2, Ix2>,
) -> Result<Array2<f32>>
where
    S1: Data<Elem = f32>,
    S2: Data<Elem = f32>,
{
    let ctx = GpuContext::get_or_init()
        .ok_or_else(|| crate::NumpyError::ComputeError("GPU not available".into()))?;

    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(crate::NumpyError::ShapeError(
            format!("Matrix dimensions don't match: ({}, {}) × ({}, {})", m, k1, k2, n)
        ));
    }
    let k = k1;

    // Convert to contiguous arrays
    let a_contig = a.as_slice().map(|s| s.to_vec())
        .unwrap_or_else(|| a.iter().cloned().collect());
    let b_contig = b.as_slice().map(|s| s.to_vec())
        .unwrap_or_else(|| b.iter().cloned().collect());

    // Create buffers
    let a_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix A"),
        contents: bytemuck::cast_slice(&a_contig),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let b_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Matrix B"),
        contents: bytemuck::cast_slice(&b_contig),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let c_size = (m * n * std::mem::size_of::<f32>()) as u64;
    let c_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Matrix C"),
        size: c_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Dimensions uniform buffer
    let dims = [m as u32, n as u32, k as u32, 0u32]; // pad to 16 bytes
    let dims_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dimensions"),
        contents: bytemuck::cast_slice(&dims),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Load shader
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("MatMul Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../kernels/matmul.wgsl").into()),
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("MatMul Bind Group Layout"),
        entries: &[
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

    // Create pipeline
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MatMul Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("MatMul Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "matmul_tiled",
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

    // Read back results
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: c_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&c_buffer, 0, &staging_buffer, 0, c_size);
    ctx.queue.submit(Some(encoder.finish()));

    // Map buffer and read
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    ctx.device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // Convert to Array2
    Ok(Array2::from_shape_vec((m, n), result)
        .map_err(|e| crate::NumpyError::ShapeError(e.to_string()))?)
}
```

### 2.3 Update linalg.rs with GPU Dispatch

**File:** `src/linalg.rs`

```rust
// Add at the top
#[cfg(feature = "gpu")]
use crate::gpu::ops::matmul::matmul_gpu;

// Update matmul function
pub fn matmul<S1, S2>(a: &ArrayBase<S1, Ix2>, b: &ArrayBase<S2, Ix2>) -> Result<Array2<S1::Elem>>
where
    S1: Data,
    S1::Elem: Num + Copy + Zero + Send + Sync,
    S2: Data<Elem = S1::Elem>,
{
    let (m, k1) = a.dim();
    let (k2, n) = b.dim();

    if k1 != k2 {
        return Err(NumpyError::ShapeError(
            format!("Matrix dimensions don't match for multiplication")
        ));
    }

    #[cfg(feature = "gpu")]
    {
        // Try GPU for large matrices (f32 only for now)
        use std::any::TypeId;
        if TypeId::of::<S1::Elem>() == TypeId::of::<f32>()
            && m >= 512 && n >= 512
            && crate::gpu::GpuContext::is_available()
        {
            // Cast to f32 arrays (safe because we checked TypeId)
            let a_f32 = unsafe { std::mem::transmute(a) };
            let b_f32 = unsafe { std::mem::transmute(b) };

            if let Ok(result) = matmul_gpu(a_f32, b_f32) {
                return Ok(unsafe { std::mem::transmute(result) });
            }
            // Fall through to CPU on GPU failure
        }
    }

    // CPU implementation (existing code)
    // Priority 1: Small matrices with unrolled code
    if m <= 4 && n <= 4 && k1 <= 4 {
        // ... existing unrolled code
    }

    // Continue with existing CPU dispatch logic...
}
```

---

## Phase 3: Bun + React Demo Application (Week 3-4)

### 3.1 Demo Application Structure

```
demo/
├── package.json
├── bun.lockb
├── public/
│   └── index.html
├── src/
│   ├── App.tsx              # Main React app
│   ├── main.tsx             # Entry point
│   ├── components/
│   │   ├── MatrixInput.tsx      # Matrix size/data input
│   │   ├── BenchmarkRunner.tsx  # Run CPU vs GPU benchmarks
│   │   ├── ResultsDisplay.tsx   # Show results and speedup
│   │   ├── GPUInfo.tsx          # Display GPU adapter info
│   │   └── PerformanceChart.tsx # Visualize performance
│   ├── wasm/
│   │   └── numpy_rust.ts    # WASM bindings (generated)
│   └── styles/
│       └── app.css
└── vite.config.ts
```

### 3.2 Demo package.json

**File:** `demo/package.json`

```json
{
  "name": "numpy-rust-webgpu-demo",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "build:wasm": "cd .. && wasm-pack build --target web --features gpu-web"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tanstack/react-query": "^5.0.0",
    "recharts": "^2.10.0",
    "lucide-react": "^0.300.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "vite-plugin-wasm": "^3.3.0",
    "vite-plugin-top-level-await": "^1.4.0"
  }
}
```

### 3.3 Main React App

**File:** `demo/src/App.tsx`

```tsx
import { useState, useEffect } from 'react';
import { AlertCircle, Zap, Cpu, Info } from 'lucide-react';
import MatrixInput from './components/MatrixInput';
import BenchmarkRunner from './components/BenchmarkRunner';
import ResultsDisplay from './components/ResultsDisplay';
import GPUInfo from './components/GPUInfo';
import './styles/app.css';

interface BenchmarkResult {
  cpuTime: number;
  gpuTime?: number;
  speedup?: number;
  matrixSize: number;
  correct: boolean;
}

export default function App() {
  const [wasmModule, setWasmModule] = useState<any>(null);
  const [gpuAvailable, setGpuAvailable] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BenchmarkResult[]>([]);

  useEffect(() => {
    async function initWasm() {
      try {
        // Check WebGPU support
        if (!navigator.gpu) {
          setError('WebGPU is not supported in this browser. Please use Chrome 113+ or Edge 113+.');
          setLoading(false);
          return;
        }

        setGpuAvailable(true);

        // Load WASM module
        const module = await import('../pkg/numpy_rust.js');
        await module.default();

        // Initialize GPU context
        await module.init_gpu();

        setWasmModule(module);
        setLoading(false);
      } catch (err) {
        console.error('Failed to initialize:', err);
        setError(`Initialization failed: ${err}`);
        setLoading(false);
      }
    }

    initWasm();
  }, []);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="spinner"></div>
        <p>Loading numpy-rust WASM module...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-screen">
        <AlertCircle size={48} />
        <h2>Initialization Error</h2>
        <p>{error}</p>
        <div className="error-help">
          <h3>Requirements:</h3>
          <ul>
            <li>Chrome 113+ or Edge 113+ (WebGPU support)</li>
            <li>GPU with WebGPU-compatible driver</li>
            <li>Enable chrome://flags/#enable-unsafe-webgpu (if needed)</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>
            <Zap className="logo" />
            numpy-rust WebGPU Demo
          </h1>
          <p className="subtitle">
            GPU-accelerated matrix operations in your browser
          </p>
        </div>
        {gpuAvailable && (
          <div className="gpu-badge">
            <Cpu size={16} />
            GPU Enabled
          </div>
        )}
      </header>

      <main className="app-main">
        <section className="info-section">
          <div className="info-card">
            <Info size={20} />
            <div>
              <h3>About This Demo</h3>
              <p>
                This interactive demo showcases GPU-accelerated matrix multiplication
                using WebGPU and WebAssembly. Compare CPU vs GPU performance and see
                real-time speedup metrics.
              </p>
            </div>
          </div>
        </section>

        {gpuAvailable && <GPUInfo module={wasmModule} />}

        <section className="benchmark-section">
          <h2>Matrix Multiplication Benchmark</h2>
          <p className="section-description">
            Test matrix multiplication performance with different sizes. GPU acceleration
            typically shows significant speedup for matrices larger than 512×512.
          </p>

          <MatrixInput />

          <BenchmarkRunner
            module={wasmModule}
            onResults={(newResults) => setResults([...results, newResults])}
          />
        </section>

        {results.length > 0 && (
          <section className="results-section">
            <h2>Benchmark Results</h2>
            <ResultsDisplay results={results} />
          </section>
        )}

        <section className="technical-info">
          <h2>Technical Details</h2>
          <div className="info-grid">
            <div className="info-item">
              <h4>GPU Implementation</h4>
              <p>
                Uses tiled matrix multiplication with 16×16 workgroups and shared memory
                for optimal cache utilization. WGSL compute shaders compiled at runtime.
              </p>
            </div>
            <div className="info-item">
              <h4>CPU Implementation</h4>
              <p>
                Multi-tier dispatch system: unrolled kernels for small matrices,
                cache-blocked for medium, and parallel cache-blocked for large matrices.
              </p>
            </div>
            <div className="info-item">
              <h4>WASM Integration</h4>
              <p>
                Rust compiled to WebAssembly with wasm-bindgen. GPU operations use
                WebGPU API directly from Rust via wgpu crate.
              </p>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        <p>
          Built with Bun, React, Vite, and Rust 🦀
        </p>
        <p>
          <a href="https://github.com/boxabirds/numpy-rust" target="_blank" rel="noopener">
            View on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
```

### 3.4 Benchmark Runner Component

**File:** `demo/src/components/BenchmarkRunner.tsx`

```tsx
import { useState } from 'react';
import { Play, Loader } from 'lucide-react';

interface Props {
  module: any;
  onResults: (results: any) => void;
}

const PRESET_SIZES = [128, 256, 512, 1024, 2048, 4096];

export default function BenchmarkRunner({ module, onResults }: Props) {
  const [matrixSize, setMatrixSize] = useState(1024);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState('');

  async function runBenchmark() {
    setRunning(true);
    setProgress('Generating matrices...');

    try {
      // Generate random matrices on CPU
      const a = new Float32Array(matrixSize * matrixSize);
      const b = new Float32Array(matrixSize * matrixSize);

      for (let i = 0; i < a.length; i++) {
        a[i] = Math.random();
        b[i] = Math.random();
      }

      // Run CPU benchmark
      setProgress('Running CPU benchmark...');
      const cpuStart = performance.now();
      const cpuResult = module.matmul_cpu(a, b, matrixSize);
      const cpuTime = performance.now() - cpuStart;

      // Run GPU benchmark
      setProgress('Running GPU benchmark...');
      const gpuStart = performance.now();
      const gpuResult = module.matmul_gpu(a, b, matrixSize);
      const gpuTime = performance.now() - gpuStart;

      // Verify correctness (check first few elements)
      setProgress('Verifying results...');
      let correct = true;
      const checkCount = Math.min(100, cpuResult.length);
      for (let i = 0; i < checkCount; i++) {
        const diff = Math.abs(cpuResult[i] - gpuResult[i]);
        if (diff > 1e-4) {
          correct = false;
          break;
        }
      }

      const speedup = cpuTime / gpuTime;

      onResults({
        matrixSize,
        cpuTime: cpuTime.toFixed(2),
        gpuTime: gpuTime.toFixed(2),
        speedup: speedup.toFixed(2),
        correct,
      });

      setProgress('');
    } catch (err) {
      console.error('Benchmark failed:', err);
      setProgress(`Error: ${err}`);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="benchmark-runner">
      <div className="size-selector">
        <label>Matrix Size:</label>
        <div className="size-buttons">
          {PRESET_SIZES.map(size => (
            <button
              key={size}
              className={matrixSize === size ? 'active' : ''}
              onClick={() => setMatrixSize(size)}
              disabled={running}
            >
              {size}×{size}
            </button>
          ))}
        </div>
        <input
          type="range"
          min="128"
          max="4096"
          step="128"
          value={matrixSize}
          onChange={(e) => setMatrixSize(Number(e.target.value))}
          disabled={running}
        />
        <span className="size-display">{matrixSize}×{matrixSize}</span>
      </div>

      <button
        className="run-button"
        onClick={runBenchmark}
        disabled={running}
      >
        {running ? (
          <>
            <Loader className="spinning" size={20} />
            Running...
          </>
        ) : (
          <>
            <Play size={20} />
            Run Benchmark
          </>
        )}
      </button>

      {progress && (
        <div className="progress-indicator">
          {progress}
        </div>
      )}
    </div>
  );
}
```

### 3.5 Results Display Component

**File:** `demo/src/components/ResultsDisplay.tsx`

```tsx
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Check, X } from 'lucide-react';

interface Props {
  results: Array<{
    matrixSize: number;
    cpuTime: number;
    gpuTime?: number;
    speedup?: number;
    correct: boolean;
  }>;
}

export default function ResultsDisplay({ results }: Props) {
  const chartData = results.map(r => ({
    size: `${r.matrixSize}²`,
    CPU: parseFloat(r.cpuTime),
    GPU: r.gpuTime ? parseFloat(r.gpuTime) : 0,
  }));

  return (
    <div className="results-display">
      <div className="results-table">
        <table>
          <thead>
            <tr>
              <th>Matrix Size</th>
              <th>CPU Time (ms)</th>
              <th>GPU Time (ms)</th>
              <th>Speedup</th>
              <th>Correct</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result, idx) => (
              <tr key={idx}>
                <td>{result.matrixSize}×{result.matrixSize}</td>
                <td>{result.cpuTime}</td>
                <td>{result.gpuTime || 'N/A'}</td>
                <td className="speedup">
                  {result.speedup ? `${result.speedup}×` : 'N/A'}
                </td>
                <td>
                  {result.correct ? (
                    <Check className="check-icon" size={20} />
                  ) : (
                    <X className="x-icon" size={20} />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {results.length > 1 && (
        <div className="results-chart">
          <h3>Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="size" />
              <YAxis label={{ value: 'Time (ms)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Bar dataKey="CPU" fill="#8884d8" />
              <Bar dataKey="GPU" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
```

### 3.6 GPU Info Component

**File:** `demo/src/components/GPUInfo.tsx`

```tsx
import { useEffect, useState } from 'react';
import { Cpu } from 'lucide-react';

interface Props {
  module: any;
}

export default function GPUInfo({ module }: Props) {
  const [gpuInfo, setGpuInfo] = useState<any>(null);

  useEffect(() => {
    async function fetchGpuInfo() {
      try {
        const info = await module.get_gpu_info();
        setGpuInfo(info);
      } catch (err) {
        console.error('Failed to get GPU info:', err);
      }
    }

    if (module) {
      fetchGpuInfo();
    }
  }, [module]);

  if (!gpuInfo) {
    return null;
  }

  return (
    <section className="gpu-info-section">
      <h2>
        <Cpu size={24} />
        GPU Information
      </h2>
      <div className="gpu-details">
        <div className="detail-item">
          <span className="label">Device:</span>
          <span className="value">{gpuInfo.name}</span>
        </div>
        <div className="detail-item">
          <span className="label">Backend:</span>
          <span className="value">{gpuInfo.backend}</span>
        </div>
        <div className="detail-item">
          <span className="label">Vendor:</span>
          <span className="value">{gpuInfo.vendor}</span>
        </div>
        <div className="detail-item">
          <span className="label">Driver:</span>
          <span className="value">{gpuInfo.driver}</span>
        </div>
      </div>
    </section>
  );
}
```

### 3.7 Vite Configuration

**File:** `demo/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    react(),
    wasm(),
    topLevelAwait(),
  ],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['numpy_rust'],
  },
});
```

### 3.8 Styles

**File:** `demo/src/styles/app.css`

```css
:root {
  --primary: #3b82f6;
  --secondary: #10b981;
  --danger: #ef4444;
  --background: #0f172a;
  --surface: #1e293b;
  --surface-light: #334155;
  --text: #f1f5f9;
  --text-muted: #94a3b8;
  --border: #334155;
  --success: #10b981;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
  background: var(--background);
  color: var(--text);
  line-height: 1.6;
}

.app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
.app-header {
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  padding: 2rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header-content h1 {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.logo {
  color: var(--primary);
}

.subtitle {
  color: var(--text-muted);
  font-size: 1rem;
}

.gpu-badge {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  background: var(--success);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-weight: 600;
}

/* Main Content */
.app-main {
  flex: 1;
  max-width: 1200px;
  width: 100%;
  margin: 0 auto;
  padding: 2rem;
}

section {
  margin-bottom: 3rem;
}

h2 {
  font-size: 1.75rem;
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.section-description {
  color: var(--text-muted);
  margin-bottom: 1.5rem;
}

/* Info Card */
.info-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 1.5rem;
  display: flex;
  gap: 1rem;
}

.info-card svg {
  color: var(--primary);
  flex-shrink: 0;
}

/* Benchmark Runner */
.benchmark-runner {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 2rem;
}

.size-selector {
  margin-bottom: 2rem;
}

.size-selector label {
  display: block;
  margin-bottom: 1rem;
  font-weight: 600;
}

.size-buttons {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-bottom: 1rem;
}

.size-buttons button {
  padding: 0.5rem 1rem;
  background: var(--surface-light);
  border: 1px solid var(--border);
  color: var(--text);
  border-radius: 0.5rem;
  cursor: pointer;
  transition: all 0.2s;
}

.size-buttons button:hover:not(:disabled) {
  background: var(--primary);
  border-color: var(--primary);
}

.size-buttons button.active {
  background: var(--primary);
  border-color: var(--primary);
}

.size-buttons button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

input[type="range"] {
  width: 100%;
  margin-bottom: 0.5rem;
}

.size-display {
  font-weight: 600;
  color: var(--primary);
}

.run-button {
  width: 100%;
  padding: 1rem;
  background: var(--primary);
  color: white;
  border: none;
  border-radius: 0.5rem;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  transition: all 0.2s;
}

.run-button:hover:not(:disabled) {
  background: #2563eb;
}

.run-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.progress-indicator {
  margin-top: 1rem;
  text-align: center;
  color: var(--text-muted);
}

/* Results Display */
.results-display {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 2rem;
}

.results-table {
  overflow-x: auto;
  margin-bottom: 2rem;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  padding: 1rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

th {
  font-weight: 600;
  color: var(--text-muted);
}

.speedup {
  font-weight: 600;
  color: var(--success);
}

.check-icon {
  color: var(--success);
}

.x-icon {
  color: var(--danger);
}

/* GPU Info */
.gpu-info-section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 2rem;
}

.gpu-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 1rem;
  background: var(--surface-light);
  border-radius: 0.5rem;
}

.label {
  color: var(--text-muted);
}

.value {
  font-weight: 600;
}

/* Technical Info */
.info-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
}

.info-item {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 1.5rem;
}

.info-item h4 {
  margin-bottom: 0.75rem;
  color: var(--primary);
}

/* Footer */
.app-footer {
  background: var(--surface);
  border-top: 1px solid var(--border);
  padding: 2rem;
  text-align: center;
  color: var(--text-muted);
}

.app-footer a {
  color: var(--primary);
  text-decoration: none;
}

.app-footer a:hover {
  text-decoration: underline;
}

/* Loading/Error Screens */
.loading-screen,
.error-screen {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  text-align: center;
}

.spinner {
  width: 48px;
  height: 48px;
  border: 4px solid var(--surface-light);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin-bottom: 1rem;
}

.error-screen svg {
  color: var(--danger);
  margin-bottom: 1rem;
}

.error-help {
  margin-top: 2rem;
  text-align: left;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 1.5rem;
  max-width: 600px;
}

.error-help h3 {
  margin-bottom: 1rem;
}

.error-help ul {
  padding-left: 1.5rem;
  color: var(--text-muted);
}
```

---

## Phase 4: WASM Bindings (Week 4)

### 4.1 Add WASM-specific Exports

**File:** `src/wasm.rs`

```rust
use wasm_bindgen::prelude::*;
use js_sys::Float32Array;
use ndarray::Array2;

#[cfg(feature = "gpu-web")]
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
    #[cfg(feature = "gpu-web")]
    {
        GpuContext::get_or_init()
            .ok_or_else(|| JsValue::from_str("Failed to initialize GPU"))?;
        Ok(())
    }

    #[cfg(not(feature = "gpu-web"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}

/// Get GPU adapter information
#[wasm_bindgen]
pub fn get_gpu_info() -> Result<JsValue, JsValue> {
    #[cfg(feature = "gpu-web")]
    {
        let ctx = GpuContext::get_or_init()
            .ok_or_else(|| JsValue::from_str("GPU not initialized"))?;

        let info = ctx.adapter_info();
        let obj = js_sys::Object::new();

        js_sys::Reflect::set(&obj, &"name".into(), &info.name.into())?;
        js_sys::Reflect::set(&obj, &"backend".into(), &format!("{:?}", info.backend).into())?;
        js_sys::Reflect::set(&obj, &"vendor".into(), &info.vendor.to_string().into())?;
        js_sys::Reflect::set(&obj, &"driver".into(), &info.driver.into())?;

        Ok(obj.into())
    }

    #[cfg(not(feature = "gpu-web"))]
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

    // Force CPU computation
    let result = crate::linalg::matmul_cpu_only(&a_arr, &b_arr)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let result_vec = result.into_raw_vec();
    Ok(Float32Array::from(&result_vec[..]))
}

/// GPU matrix multiplication
#[wasm_bindgen]
pub fn matmul_gpu(a: Float32Array, b: Float32Array, n: usize) -> Result<Float32Array, JsValue> {
    #[cfg(feature = "gpu-web")]
    {
        let a_vec: Vec<f32> = a.to_vec();
        let b_vec: Vec<f32> = b.to_vec();

        let a_arr = Array2::from_shape_vec((n, n), a_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let b_arr = Array2::from_shape_vec((n, n), b_vec)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result = crate::gpu::ops::matmul::matmul_gpu(&a_arr, &b_arr)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let result_vec = result.into_raw_vec();
        Ok(Float32Array::from(&result_vec[..]))
    }

    #[cfg(not(feature = "gpu-web"))]
    Err(JsValue::from_str("GPU feature not enabled"))
}
```

### 4.2 Update lib.rs

**File:** `src/lib.rs`

```rust
// ... existing code ...

#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "gpu")]
pub mod gpu;
```

---

## Phase 5: Testing & Documentation (Week 5)

### 5.1 GPU Integration Tests

**File:** `tests/gpu_tests.rs`

```rust
#![cfg(feature = "gpu")]

use numpy_rust::prelude::*;
use numpy_rust::gpu::GpuContext;
use approx::assert_abs_diff_eq;

#[test]
fn test_gpu_matmul_correctness() {
    // Initialize GPU
    if GpuContext::get_or_init().is_none() {
        eprintln!("GPU not available, skipping test");
        return;
    }

    let sizes = [64, 128, 256, 512];

    for &n in &sizes {
        let a = Array2::<f32>::from_shape_fn((n, n), |(i, j)| (i + j) as f32);
        let b = Array2::<f32>::from_shape_fn((n, n), |(i, j)| (i * j) as f32);

        // CPU result
        let cpu_result = linalg::matmul_cpu_only(&a, &b).unwrap();

        // GPU result
        let gpu_result = numpy_rust::gpu::ops::matmul::matmul_gpu(&a, &b).unwrap();

        // Compare
        assert_eq!(cpu_result.dim(), gpu_result.dim());
        for i in 0..n {
            for j in 0..n {
                assert_abs_diff_eq!(cpu_result[[i, j]], gpu_result[[i, j]], epsilon = 1e-3);
            }
        }
    }
}

#[test]
fn test_gpu_fallback() {
    // Test that CPU fallback works when GPU is unavailable
    let a = Array2::<f32>::ones((10, 10));
    let b = Array2::<f32>::ones((10, 10));

    // Should not panic, will use CPU
    let result = linalg::matmul(&a, &b).unwrap();
    assert_eq!(result.dim(), (10, 10));
}
```

### 5.2 Demo Build Script

**File:** `scripts/build_demo.sh`

```bash
#!/bin/bash
set -e

echo "🦀 Building numpy-rust WASM module with GPU support..."
wasm-pack build --target web --features gpu-web --out-dir demo/pkg

echo "📦 Installing demo dependencies..."
cd demo
bun install

echo "🏗️  Building demo..."
bun run build

echo "✅ Demo built successfully!"
echo "📍 Output: demo/dist"
echo ""
echo "To preview:"
echo "  cd demo && bun run preview"
```

### 5.3 Demo README

**File:** `demo/README.md`

```markdown
# numpy-rust WebGPU Demo

Interactive demo showcasing GPU-accelerated matrix operations in the browser.

## Requirements

- **Browser:** Chrome 113+ or Edge 113+ (WebGPU support required)
- **GPU:** WebGPU-compatible graphics card with updated drivers
- **Runtime:** Bun 1.0+ (for development)

## Quick Start

```bash
# From repository root
./scripts/build_demo.sh

# Start development server
cd demo
bun run dev
```

Visit `http://localhost:5173` to see the demo.

## Features

- 🚀 GPU-accelerated matrix multiplication via WebGPU
- 📊 Real-time performance comparison (CPU vs GPU)
- 📈 Interactive benchmark visualizations
- 🎮 GPU information display
- ✅ Correctness verification

## Usage

1. Select matrix size using buttons or slider
2. Click "Run Benchmark" to compare CPU vs GPU
3. View results table showing speedup metrics
4. Run multiple sizes to see performance scaling

## GPU Performance Tips

- GPU acceleration shows significant speedup for matrices ≥512×512
- Smaller matrices may be faster on CPU due to transfer overhead
- First run initializes GPU context (may be slower)
- Subsequent runs reuse compiled shaders (faster)

## Browser Compatibility

| Browser | Version | WebGPU Support |
|---------|---------|---------------|
| Chrome  | 113+    | ✅ Yes         |
| Edge    | 113+    | ✅ Yes         |
| Firefox | 126+    | ⚠️ Experimental |
| Safari  | TP      | 🚧 In Progress  |

### Enabling WebGPU

If WebGPU is not available:

1. Visit `chrome://flags/#enable-unsafe-webgpu`
2. Enable "Unsafe WebGPU"
3. Restart browser

## Architecture

```
User Input → React UI → WASM Bindings → Rust GPU Code → WebGPU → GPU
                                      ↘ CPU Fallback ↗
```

## Building for Production

```bash
cd demo
bun run build

# Deploy dist/ folder to static hosting
```

## Troubleshooting

**"WebGPU is not supported"**
- Update browser to latest version
- Check GPU drivers are up-to-date
- Enable chrome://flags/#enable-unsafe-webgpu

**Slow performance**
- First run initializes GPU (normal)
- Check GPU isn't under heavy load
- Try disabling browser extensions

**Incorrect results**
- Report issue with matrix size and GPU model
- Check browser console for errors

## License

Apache 2.0
```

---

## Phase 6: Optimization & Deployment (Week 6)

### 6.1 Performance Optimization Checklist

- [ ] Pipeline caching (avoid recompiling shaders)
- [ ] Buffer pooling (reuse GPU buffers)
- [ ] Minimize CPU↔GPU transfers
- [ ] Vectorized loads (vec4 in WGSL)
- [ ] Optimal workgroup sizes (16×16 vs 32×32 testing)
- [ ] Shared memory utilization (maxed out at 16KB)
- [ ] Minimize synchronization barriers
- [ ] Bundle size optimization (<500KB WASM target)

### 6.2 Deployment Options

**Option A: GitHub Pages (Free)**
```bash
# Build and deploy
cd demo
bun run build
npx gh-pages -d dist
```

**Option B: Vercel (Free)**
```bash
# Install Vercel CLI
bun add -g vercel

# Deploy
cd demo
vercel --prod
```

**Option C: Cloudflare Pages (Free)**
- Connect GitHub repository
- Build command: `cd demo && bun run build`
- Output directory: `demo/dist`

### 6.3 Analytics & Telemetry

Add basic analytics to track:
- GPU availability rate
- Browser/GPU combinations
- Performance metrics by device
- Error rates

---

## Success Metrics

### Performance Targets

| Matrix Size | CPU Time (target) | GPU Time (target) | Speedup Target |
|-------------|------------------|------------------|----------------|
| 512×512     | ~50ms            | ~5ms             | 10x            |
| 1024×1024   | ~400ms           | ~20ms            | 20x            |
| 2048×2048   | ~3000ms          | ~100ms           | 30x            |
| 4096×4096   | ~24000ms         | ~800ms           | 30x            |

### Quality Targets

- ✅ Numerical accuracy: <1e-4 error vs CPU
- ✅ Browser compatibility: Chrome/Edge 113+
- ✅ Load time: <2s on fast connection
- ✅ WASM size: <500KB gzipped
- ✅ GPU initialization: <500ms
- ✅ Zero crashes in normal operation

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Foundation | GPU context, dependencies, module structure |
| 2 | MatMul Kernel | WGSL shader, Rust integration, CPU dispatch |
| 3-4 | Demo App | Bun/React app with benchmarking UI |
| 4 | WASM Bindings | JS exports, async initialization |
| 5 | Testing | Integration tests, correctness verification |
| 6 | Optimization | Performance tuning, deployment |

**Total:** 6 weeks, ~4,000 lines of code

---

## Next Steps

1. **Approve this plan** - Review and provide feedback
2. **Set up development environment** - Install wgpu, Bun
3. **Start Phase 1** - Implement GPU foundation
4. **Iterate on demo** - Build UI alongside backend
5. **Deploy preview** - Share with community for testing

---

## Future Enhancements

### Phase 7: Additional GPU Operations
- Element-wise operations (sin, cos, exp, log)
- FFT acceleration
- Reductions (sum, mean, std)
- Convolutions

### Phase 8: Advanced Features
- Multi-GPU support
- Persistent buffers
- Compute shader fusion
- FP16 precision option (2x faster)

### Phase 9: Mobile Support
- iOS Safari WebGPU (when available)
- Android Chrome optimization
- Responsive demo UI

---

## Resources

- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [wgpu Documentation](https://docs.rs/wgpu/)
- [WGSL Language Spec](https://www.w3.org/TR/WGSL/)
- [wasm-bindgen Guide](https://rustwasm.github.io/wasm-bindgen/)
- [Bun Documentation](https://bun.sh/docs)

---

**Status:** Ready for implementation
**Estimated Effort:** 6 weeks (1 developer)
**Priority:** High impact (100-1000x speedup + unique browser capability)
