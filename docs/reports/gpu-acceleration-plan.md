# GPU Acceleration Plan for numpy-rust (Phase 4)

**Date:** 2025-11-09
**Status:** Planning Phase
**Goal:** Add optional WebGPU acceleration for compute-intensive operations with 100-1000x speedup potential

---

## Executive Summary

This plan outlines adding GPU acceleration to numpy-rust using WebGPU/wgpu, providing:

- **100-1000x speedup** for large matrix operations and element-wise computations
- **Browser compatibility** - GPU-accelerated NumPy running in WebAssembly
- **Cross-platform** - Single codebase for CUDA, Metal, Vulkan, DirectX, WebGPU
- **Zero cost when disabled** - Optional feature flag with graceful CPU fallback
- **Unique capability** - GPU-accelerated NumPy that works in browsers (Python NumPy can't do this)

---

## Technology Selection

### Phase 4.1: wgpu Foundation (Recommended Start)
**Status:** Stable, production-ready
**Approach:** Raw WebGPU compute shaders via wgpu crate

**Pros:**
- ✅ Mature, stable API (used by Firefox, Deno, Bevy)
- ✅ Direct control over GPU operations
- ✅ Minimal overhead
- ✅ Excellent documentation and community
- ✅ WASM/browser support proven
- ✅ Works on all platforms: Windows (DX12/Vulkan), macOS (Metal), Linux (Vulkan), Web (WebGPU)

**Cons:**
- ⚠️ Lower-level API (write WGSL shaders manually)
- ⚠️ More boilerplate for buffer management
- ⚠️ Manual optimization required

**Performance:** 100-500 TFLOPs on modern GPUs (RTX 4080: ~100-120 TFLOPs via Vulkan)

### Phase 4.2: CubeCL High-Level Abstraction (Future)
**Status:** Alpha, actively developed
**Approach:** JIT-compiled Rust GPU kernels with automatic optimization

**Pros:**
- ✅ Write GPU kernels in Rust (no WGSL)
- ✅ Automatic vectorization and autotuning
- ✅ Optimized matmul with Tensor Core support
- ✅ Comptime optimizations (loop unrolling, specialization)
- ✅ Used by Burn framework successfully
- ✅ Multi-backend: CUDA, ROCm, Metal, Vulkan, WebGPU

**Cons:**
- ⚠️ Alpha software with "rough edges"
- ⚠️ Additional abstraction layer
- ⚠️ Less mature than wgpu
- ⚠️ Larger dependency footprint

**Performance:** Matches cuBLAS/CUTLASS on NVIDIA, competitive on AMD/Intel

### Recommendation: **Two-Phase Approach**

1. **Phase 4.1:** Start with wgpu for core operations (matmul, element-wise)
2. **Phase 4.2:** Evaluate CubeCL once it reaches beta/stable

---

## Operation Prioritization

### Tier 1: Highest Impact (100-1000x potential speedup)

| Operation | Size Threshold | Expected Speedup | Priority | Module |
|-----------|----------------|------------------|----------|---------|
| **Matrix multiplication** | ≥512×512 | 100-500x | 🔴 Critical | linalg |
| **Element-wise math** | ≥100K elements | 50-200x | 🔴 Critical | math |
| **FFT** | ≥8192 length | 100-300x | 🔴 Critical | fft |
| **Reductions** | ≥1M elements | 50-100x | 🟡 High | stats |
| **Convolutions** | ≥1024×1024 | 100-500x | 🟡 High | future |

**Rationale:** These operations are:
- Highly parallelizable (thousands of independent operations)
- Compute-bound (arithmetic intensity)
- Common in scientific computing
- Large enough to amortize GPU transfer overhead

### Tier 2: Medium Impact (10-100x potential speedup)

| Operation | Size Threshold | Expected Speedup | Priority | Module |
|-----------|----------------|------------------|----------|---------|
| **Statistical reductions** | ≥100K elements | 20-80x | 🟡 High | stats |
| **Sorting** | ≥100K elements | 10-50x | 🟢 Medium | sorting |
| **Broadcasting ops** | ≥100K elements | 20-60x | 🟢 Medium | array |
| **Random generation** | ≥1M samples | 10-40x | 🟢 Medium | random |

### Tier 3: Lower Impact (overhead > benefit)

| Operation | Why Excluded | Alternative |
|-----------|-------------|-------------|
| Small arrays (<1000 elements) | GPU transfer overhead | Use CPU (current impl) |
| Non-vectorizable ops | Sequential nature | Use CPU quickselect |
| Memory-bound ops | Bandwidth limited | CPU cache is faster |
| Irregular access patterns | Poor GPU locality | CPU with cache blocking |

---

## Architecture Design

### Smart Dispatch System

```rust
/// GPU acceleration dispatcher with automatic fallback
pub enum ComputeBackend {
    /// CPU fallback (always available)
    Cpu,
    /// GPU acceleration (optional feature)
    #[cfg(feature = "gpu")]
    Gpu(GpuContext),
}

impl ComputeBackend {
    /// Automatically select best backend for operation
    pub fn auto_select(
        op_type: OpType,
        data_size: usize,
        data_shape: &[usize],
    ) -> ComputeBackend {
        #[cfg(feature = "gpu")]
        {
            // Check if GPU is available and initialized
            if let Some(gpu) = GPU_CONTEXT.get() {
                // Check if operation benefits from GPU
                if should_use_gpu(op_type, data_size, data_shape) {
                    return ComputeBackend::Gpu(gpu.clone());
                }
            }
        }

        // Default to CPU
        ComputeBackend::Cpu
    }
}

/// Decide if GPU acceleration is beneficial
fn should_use_gpu(
    op_type: OpType,
    data_size: usize,
    shape: &[usize],
) -> bool {
    match op_type {
        OpType::MatMul => {
            // GPU beneficial for matrices ≥512×512
            shape.len() == 2 && shape[0] >= 512 && shape[1] >= 512
        }
        OpType::ElementWise => {
            // GPU beneficial for arrays ≥100K elements
            data_size >= 100_000
        }
        OpType::Fft => {
            // GPU beneficial for FFTs ≥8192 length
            data_size >= 8192
        }
        OpType::Reduction => {
            // GPU beneficial for large reductions ≥1M elements
            data_size >= 1_000_000
        }
    }
}
```

### Feature Flag Design

```toml
[features]
# Default: CPU-only (zero GPU dependencies)
default = []

# GPU acceleration via WebGPU/wgpu
gpu = ["wgpu", "bytemuck", "pollster"]

# Future: CubeCL high-level abstraction
gpu-cubecl = ["cubecl-core", "cubecl-wgpu", "gpu"]

# Future: CUDA-specific optimizations
gpu-cuda = ["cubecl-cuda", "gpu-cubecl"]
```

**Benefits:**
- Zero cost abstraction: GPU code not compiled unless feature enabled
- Binary size: ~2MB base, +5MB with GPU support
- Graceful degradation: Missing GPU falls back to CPU automatically
- Multiple backends: Users can choose wgpu vs CubeCL

### GPU Context Management

```rust
use once_cell::sync::OnceCell;
use wgpu::{Device, Queue, Instance};

/// Global GPU context (lazy initialized)
static GPU_CONTEXT: OnceCell<Arc<GpuContext>> = OnceCell::new();

pub struct GpuContext {
    device: Device,
    queue: Queue,
    /// Cached compute pipelines
    pipelines: DashMap<PipelineKey, ComputePipeline>,
    /// Memory pool for buffer reuse
    buffer_pool: BufferPool,
}

impl GpuContext {
    /// Initialize GPU context (called once on first use)
    pub async fn init() -> Result<Arc<GpuContext>, GpuError> {
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .ok_or(GpuError::NoAdapter)?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;

        Ok(Arc::new(GpuContext {
            device,
            queue,
            pipelines: DashMap::new(),
            buffer_pool: BufferPool::new(),
        }))
    }

    /// Get or initialize global GPU context
    pub fn get_or_init() -> Result<Arc<GpuContext>, GpuError> {
        GPU_CONTEXT.get_or_try_init(|| {
            // Block on async init (using pollster for simple blocking)
            pollster::block_on(Self::init())
        }).cloned()
    }
}
```

---

## Implementation Phases

### Phase 4.1: Foundation & Matrix Multiplication (Week 1-2)

**Goal:** GPU-accelerated matrix multiplication with 100-500x speedup for large matrices

#### Tasks:
1. **Setup GPU Infrastructure** (2 days)
   - Add wgpu, bytemuck, pollster dependencies
   - Create GpuContext with device/queue initialization
   - Implement feature flag system
   - Add buffer pool for memory reuse

2. **Implement GPU Matrix Multiplication** (3 days)
   - Write WGSL compute shader for tiled matmul
   - Implement 2D workgroup tiling (16×16 tiles)
   - Add shared memory optimization
   - Create buffer upload/download pipeline

3. **Smart Dispatching** (2 days)
   - Implement auto-selection logic
   - Add threshold configuration
   - Create fallback mechanism
   - Handle non-contiguous arrays

4. **Testing & Validation** (2 days)
   - Property-based tests (GPU vs CPU equality)
   - Performance benchmarks
   - Edge cases (empty, single element, non-square)
   - WASM compatibility tests

**Expected Results:**
- Matrix multiplication ≥512×512: **100-300x speedup**
- Tests: 100% pass rate (GPU == CPU results)
- Browser support: Working demo in WebAssembly

**Files Modified:**
- `src/linalg.rs` - GPU dispatch for matmul
- `src/gpu/mod.rs` (new) - GPU context management
- `src/gpu/kernels/matmul.wgsl` (new) - Matrix multiply shader
- `src/gpu/buffer.rs` (new) - Buffer pool and management
- `Cargo.toml` - Add wgpu dependencies

#### WGSL Shader Example (Simplified):

```wgsl
// Tiled matrix multiplication shader
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // m, n, k

var<workgroup> tile_a: array<f32, 256>; // 16×16 tile
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>,
          @builtin(local_invocation_id) local_id: vec3<u32>) {
    let m = dims.x;
    let n = dims.y;
    let k = dims.z;

    let row = global_id.y;
    let col = global_id.x;

    if (row >= m || col >= n) {
        return;
    }

    var sum = 0.0;

    // Tiled computation for cache efficiency
    let tile_size = 16u;
    let num_tiles = (k + tile_size - 1u) / tile_size;

    for (var t = 0u; t < num_tiles; t++) {
        // Load tile into shared memory
        let tile_row = local_id.y;
        let tile_col = local_id.x;
        let k_offset = t * tile_size;

        if (k_offset + tile_col < k) {
            tile_a[tile_row * 16u + tile_col] =
                a[row * k + k_offset + tile_col];
            tile_b[tile_row * 16u + tile_col] =
                b[(k_offset + tile_row) * n + col];
        }

        workgroupBarrier();

        // Compute partial dot product
        for (var i = 0u; i < tile_size; i++) {
            sum += tile_a[tile_row * 16u + i] *
                   tile_b[i * 16u + tile_col];
        }

        workgroupBarrier();
    }

    c[row * n + col] = sum;
}
```

---

### Phase 4.2: Element-wise Operations (Week 3)

**Goal:** GPU-accelerated math operations (sin, cos, exp, log, etc.) with 50-200x speedup

#### Tasks:
1. **Generic Element-wise Kernel** (2 days)
   - Template WGSL shader for unary operations
   - Support for sin, cos, tan, exp, log, sqrt, etc.
   - Vectorized loads (vec4 for 4-wide SIMD)
   - Pipeline caching for different operations

2. **Integration** (1 day)
   - Update src/math.rs with GPU dispatch
   - Add threshold configuration (≥100K elements)
   - Benchmark vs parallel CPU version

3. **Testing** (1 day)
   - Accuracy tests (ulp comparison)
   - Performance benchmarks
   - WASM compatibility

**Expected Results:**
- Element-wise ops ≥100K elements: **50-150x speedup**
- Tests: <1 ulp error tolerance

**WGSL Shader Example:**

```wgsl
// Generic element-wise operation shader
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256, 1, 1)
fn elementwise_sin(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= size) {
        return;
    }

    // Vectorized: process 4 elements at once
    let vec_idx = idx / 4u;
    let offset = idx % 4u;

    if (vec_idx * 4u + 3u < size) {
        // Load 4 values
        let v = vec4<f32>(
            input[vec_idx * 4u],
            input[vec_idx * 4u + 1u],
            input[vec_idx * 4u + 2u],
            input[vec_idx * 4u + 3u]
        );

        // Compute sin for all 4
        let result = sin(v);

        // Store 4 results
        output[vec_idx * 4u] = result.x;
        output[vec_idx * 4u + 1u] = result.y;
        output[vec_idx * 4u + 2u] = result.z;
        output[vec_idx * 4u + 3u] = result.w;
    } else {
        // Handle remainder
        output[idx] = sin(input[idx]);
    }
}
```

---

### Phase 4.3: FFT Acceleration (Week 4)

**Goal:** GPU-accelerated FFT with 100-300x speedup for large transforms

#### Tasks:
1. **GPU FFT Implementation** (3 days)
   - Cooley-Tukey radix-2 FFT in WGSL
   - Stockham auto-sort algorithm
   - Twiddle factor precomputation
   - Support power-of-2 sizes

2. **Integration** (1 day)
   - Update src/fft.rs with GPU dispatch
   - Maintain cached planner pattern
   - Add GPU path for ≥8192 length

3. **Optimization** (1 day)
   - Shared memory for twiddle factors
   - Minimize passes (log2(n) passes required)
   - Benchmark vs rustfft

4. **Testing** (1 day)
   - Accuracy tests vs rustfft
   - Inverse FFT correctness
   - Performance benchmarks

**Expected Results:**
- FFT ≥8192 length: **100-200x speedup**
- Accuracy: <1e-6 error vs rustfft

**Alternative:** Consider using existing GPU FFT libraries like `vkfft-rs` wrapper

---

### Phase 4.4: Reductions & Statistics (Week 5)

**Goal:** GPU-accelerated sum, mean, std, var with 50-100x speedup

#### Tasks:
1. **Parallel Reduction Kernel** (2 days)
   - Two-phase reduction (local + global)
   - Shared memory optimization
   - Support for sum, max, min, mean
   - Kahan summation for numerical stability

2. **Variance/Std Computation** (1 day)
   - Two-pass algorithm (mean then variance)
   - Welford's online algorithm variant
   - Numerical stability tests

3. **Integration & Testing** (1 day)
   - Update src/stats.rs
   - Threshold ≥1M elements
   - Benchmark vs parallel CPU

**Expected Results:**
- Reductions ≥1M elements: **50-80x speedup**
- Numerical accuracy: Machine epsilon precision

---

### Phase 4.5: Browser/WASM Support (Week 6)

**Goal:** Full WebAssembly + WebGPU support for in-browser GPU acceleration

#### Tasks:
1. **WASM Build Configuration** (1 day)
   - Add wasm-bindgen dependencies
   - Configure wgpu for web target
   - Handle async initialization in browser

2. **JavaScript Interop** (2 days)
   - Create JS bindings for array operations
   - Handle TypedArray conversions
   - Async GPU initialization from JS

3. **Demo & Documentation** (2 days)
   - Create interactive web demo
   - Benchmark visualization
   - Browser compatibility testing

**Expected Results:**
- Working demo: GPU matmul in browser
- Supported browsers: Chrome, Edge, Firefox (when WebGPU available)
- Performance: 100x+ speedup vs WASM CPU

---

## Performance Benchmarks

### Target Hardware

| GPU | Compute | Memory BW | Target Perf (matmul) |
|-----|---------|-----------|---------------------|
| NVIDIA RTX 4080 | 48.7 TFLOPS | 716 GB/s | 100-120 TFLOPS |
| AMD RX 7600 | 21.7 TFLOPS | 288 GB/s | 15-18 TFLOPS |
| Apple M2 Pro | 5.3 TFLOPS | 200 GB/s | 3-5 TFLOPS |
| Intel Arc A770 | 17.2 TFLOPS | 560 GB/s | 10-15 TFLOPS |
| Browser (Chrome) | Varies | Varies | 50-100 GFLOPS |

### Benchmark Suite

```rust
// Criterion benchmarks for GPU vs CPU
criterion_group!(
    gpu_benchmarks,
    bench_matmul_gpu_vs_cpu,
    bench_elementwise_gpu_vs_cpu,
    bench_fft_gpu_vs_cpu,
    bench_reduction_gpu_vs_cpu,
);

fn bench_matmul_gpu_vs_cpu(c: &mut Criterion) {
    let sizes = [512, 1024, 2048, 4096, 8192];

    for &n in &sizes {
        let a = Array2::<f32>::ones((n, n));
        let b = Array2::<f32>::ones((n, n));

        let mut group = c.benchmark_group(format!("matmul_{n}x{n}"));

        group.bench_function("cpu", |bench| {
            bench.iter(|| linalg::matmul(&a, &b))
        });

        #[cfg(feature = "gpu")]
        group.bench_function("gpu", |bench| {
            bench.iter(|| linalg::matmul_gpu(&a, &b))
        });

        group.finish();
    }
}
```

---

## Risk Mitigation

### Risk 1: GPU Not Available
**Mitigation:** Graceful fallback to CPU (automatic, transparent)
```rust
let backend = ComputeBackend::auto_select(op_type, size, shape);
// Falls back to CPU if GPU unavailable
```

### Risk 2: Transfer Overhead Dominates
**Mitigation:** Smart threshold-based dispatching
- Only use GPU for large operations (≥threshold)
- Cache buffers for repeated operations
- Minimize host↔device transfers

### Risk 3: Numerical Accuracy Differences
**Mitigation:** Rigorous testing
- Property tests: GPU output == CPU output (within epsilon)
- Kahan summation for accumulation
- FP32 vs FP64 precision options

### Risk 4: Browser Compatibility
**Mitigation:** Progressive enhancement
- Detect WebGPU support at runtime
- Fall back to CPU if WebGPU unavailable
- Clear error messages for unsupported browsers

### Risk 5: Memory Limitations
**Mitigation:** Chunking for large operations
- Automatic chunking if array exceeds GPU memory
- Process in tiles/batches
- Memory pool for buffer reuse

### Risk 6: Build Complexity
**Mitigation:** Clear documentation & feature flags
- Default build: CPU-only (no GPU deps)
- Optional `--features gpu` for GPU support
- WASM build instructions separate

---

## API Design

### Transparent GPU Acceleration (Recommended)

Users don't change their code - GPU acceleration happens automatically:

```rust
// User code remains unchanged
use numpy_rust::prelude::*;

let a = Array2::<f32>::ones((2048, 2048));
let b = Array2::<f32>::ones((2048, 2048));

// Automatically uses GPU if available and beneficial
let c = linalg::matmul(&a, &b).unwrap();
// ✅ 200x faster with GPU, same API
```

### Explicit GPU Control (Advanced)

Power users can force CPU or GPU:

```rust
use numpy_rust::prelude::*;
use numpy_rust::gpu::GpuConfig;

// Configure GPU behavior
GpuConfig::global()
    .set_matmul_threshold(1024)  // Custom threshold
    .set_prefer_gpu(true)         // Prefer GPU when close call
    .set_memory_limit(8_000_000_000); // 8GB limit

// Or force backend explicitly
let backend = ComputeBackend::Gpu(GpuContext::get_or_init()?);
let c = linalg::matmul_with_backend(&a, &b, backend)?;
```

### Feature Flags

```bash
# CPU-only (default, smallest binary)
cargo build --release

# GPU acceleration (wgpu)
cargo build --release --features gpu

# Future: CubeCL high-level
cargo build --release --features gpu-cubecl

# Future: CUDA-specific optimizations
cargo build --release --features gpu-cuda

# WASM + WebGPU
cargo build --target wasm32-unknown-unknown --features gpu
```

---

## Testing Strategy

### Unit Tests
- Property tests: GPU == CPU results (within epsilon)
- Edge cases: empty, single element, odd sizes
- Error handling: OOM, unsupported operations
- Fallback: GPU unavailable → CPU

### Integration Tests
- End-to-end workflows using GPU
- Mixed CPU/GPU operations
- Memory management (no leaks)
- Async initialization correctness

### Performance Tests
- Regression tests (GPU not slower than CPU)
- Speedup validation (meets targets)
- Memory usage profiling
- Transfer overhead measurement

### Browser Tests
- WebGPU availability detection
- WASM module loading
- Cross-browser compatibility (Chrome, Edge, Firefox)
- Mobile GPU support

---

## Documentation Requirements

### User Guide
1. **Installation** - Feature flag setup
2. **Quick Start** - First GPU-accelerated operation
3. **Configuration** - Thresholds and tuning
4. **Troubleshooting** - Common issues
5. **Browser Support** - WASM + WebGPU setup

### Developer Guide
1. **Architecture** - GPU dispatch system
2. **Adding Operations** - How to GPU-accelerate new ops
3. **Shader Development** - WGSL best practices
4. **Performance Tuning** - Profiling and optimization
5. **Testing** - GPU test patterns

### API Reference
- All public GPU types and functions
- Configuration options
- Error types
- Feature flag documentation

---

## Success Metrics

### Performance Targets

| Operation | Input Size | Speedup Target | Status |
|-----------|-----------|----------------|--------|
| Matrix multiply | 2048×2048 | 100-300x | 🔲 Planned |
| Element-wise sin | 1M elements | 50-150x | 🔲 Planned |
| FFT | 65536 length | 100-200x | 🔲 Planned |
| Sum reduction | 10M elements | 50-80x | 🔲 Planned |

### Quality Targets

- ✅ 100% test pass rate (GPU == CPU within epsilon)
- ✅ Zero memory leaks (valgrind clean)
- ✅ Zero panics in GPU code paths
- ✅ Browser demo working in Chrome/Edge
- ✅ <5% binary size increase when GPU disabled

### Adoption Targets

- ✅ Clear documentation with examples
- ✅ <5 minute setup time
- ✅ Zero API changes for existing users
- ✅ Positive community feedback

---

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| **4.1** Foundation & MatMul | 2 weeks | GPU matmul 100-300x faster |
| **4.2** Element-wise Ops | 1 week | GPU sin/cos/exp 50-150x faster |
| **4.3** FFT Acceleration | 1 week | GPU FFT 100-200x faster |
| **4.4** Reductions | 1 week | GPU sum/mean 50-80x faster |
| **4.5** Browser/WASM | 1 week | Working web demo |
| **Total** | **6 weeks** | Production-ready GPU acceleration |

---

## Future Enhancements (Phase 5+)

### Phase 5.1: CubeCL Integration
- Migrate to CubeCL for higher-level abstractions
- Leverage Tensor Cores on NVIDIA
- Automatic kernel tuning
- Estimated gain: 2-5x over raw wgpu

### Phase 5.2: Advanced Algorithms
- GPU-accelerated Strassen's matmul
- Batched operations
- Fused kernels (reduce kernel launches)
- Custom BLAS-like library

### Phase 5.3: Memory Optimizations
- Unified memory (where supported)
- Persistent buffers across calls
- Automatic memory migration
- Smart prefetching

### Phase 5.4: Distributed GPU
- Multi-GPU support
- GPU cluster coordination
- MPI integration
- Cloud GPU integration (AWS, GCP)

---

## Conclusion

GPU acceleration via WebGPU/wgpu offers numpy-rust a **unique competitive advantage**:

1. **Performance:** 100-1000x speedup for large operations
2. **Portability:** Single codebase for all GPU vendors
3. **Browser Support:** GPU-accelerated NumPy in browsers (impossible with Python)
4. **Zero Cost:** Optional feature flag, graceful fallback

**Recommended Next Steps:**
1. Approve this plan
2. Start Phase 4.1 (Foundation & MatMul) - 2 weeks
3. Validate benchmarks and iterate
4. Expand to remaining operations

This positions numpy-rust as the **fastest, most portable NumPy implementation** supporting both native and browser environments with GPU acceleration.

---

**Total Estimated LOC:** ~3,000-5,000 lines (GPU infrastructure + kernels)
**Total Estimated Effort:** 6 weeks (1 developer)
**Dependencies Added:** wgpu, bytemuck, pollster (~5MB binary size increase)
**Breaking Changes:** None (fully backward compatible)

