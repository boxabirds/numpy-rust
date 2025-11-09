# Smart GPU/CPU Dispatch

## Problem Statement

GPU acceleration has overhead (PCIe transfer, kernel launch, synchronization). For small arrays, this overhead dominates computation time, making CPU faster.

**Example from benchmarks:**
- 1M element sin: CPU = 10ms, GPU = 23ms (0.47× slower!)
- 10M element sin: CPU = 105ms, GPU = 45ms (2.3× faster)
- 100M element sin: CPU = 1050ms, GPU = 235ms (4.5× faster)

## Current Behavior

Currently, when user calls `sin_gpu()`, we **always** use GPU regardless of size. This is suboptimal.

## Proposed Solution: Automatic Dispatch

### API Design

```rust
// User calls generic function (no _cpu or _gpu suffix)
pub fn sin(input: &Array1<f32>) -> Result<Array1<f32>> {
    if should_use_gpu(Operation::Sin, input.len()) {
        sin_gpu(input)
    } else {
        sin_cpu(input)
    }
}

// Advanced users can still force a backend
pub fn sin_cpu(input: &Array1<f32>) -> Result<Array1<f32>> { ... }
pub fn sin_gpu(input: &Array1<f32>) -> Result<Array1<f32>> { ... }
```

### Decision Logic

```rust
pub enum Operation {
    // Element-wise ops (low arithmetic intensity)
    Sin, Cos, Tan, Exp, Log, Sqrt,
    Add, Subtract, Multiply, Divide,

    // Reductions (medium arithmetic intensity)
    Sum, Mean, Min, Max,

    // Matrix ops (high arithmetic intensity)
    MatMul, Dot, Transpose,

    // Complex ops (very high arithmetic intensity)
    FFT, Convolution, SVD,
}

pub fn should_use_gpu(op: Operation, size: usize) -> bool {
    // GPU is only available if initialized
    if !is_gpu_available() {
        return false;
    }

    // Check against operation-specific threshold
    size >= get_gpu_threshold(op)
}

pub fn get_gpu_threshold(op: Operation) -> usize {
    match op {
        // Element-wise: High overhead, need large arrays
        Operation::Sin | Operation::Cos | Operation::Exp | Operation::Log => 10_000_000,
        Operation::Add | Operation::Multiply => 50_000_000, // Even higher!

        // Reductions: Medium overhead
        Operation::Sum | Operation::Mean => 1_000_000,
        Operation::Min | Operation::Max => 5_000_000,

        // Matrix ops: Low overhead, high compute
        Operation::MatMul => 512 * 512, // 262K elements (512×512 matrix)
        Operation::Dot => 100_000,
        Operation::Transpose => 1024 * 1024,

        // Complex ops: Very low overhead, very high compute
        Operation::FFT => 4_096,
        Operation::Convolution => 256 * 256,
        Operation::SVD => 256 * 256,
    }
}
```

## Measured Thresholds (from benchmarks)

### Element-wise Operations
| Operation | CPU Faster | Break-even | GPU Faster |
|-----------|-----------|------------|------------|
| sin       | < 5M      | 5-10M      | > 10M      |
| cos       | < 5M      | 5-10M      | > 10M      |
| exp       | < 5M      | 5-10M      | > 10M      |
| add       | < 50M     | 50-100M    | > 100M     |
| multiply  | < 50M     | 50-100M    | > 100M     |

**Reason**: Transfer dominates (8-12ms), compute is <1ms

### Matrix Operations
| Operation | CPU Faster | Break-even | GPU Faster |
|-----------|-----------|------------|------------|
| matmul    | < 256×256 | 256-512    | > 512×512  |
| dot       | < 10K     | 10-100K    | > 100K     |
| transpose | < 512×512 | 512-1024   | > 1024×1024|

**Reason**: High arithmetic intensity, transfer cost amortized

### Reductions
| Operation | CPU Faster | Break-even | GPU Faster |
|-----------|-----------|------------|------------|
| sum       | < 500K    | 500K-1M    | > 1M       |
| mean      | < 500K    | 500K-1M    | > 1M       |
| min/max   | < 1M      | 1-5M       | > 5M       |

**Reason**: Medium compute, parallel reduction helps

## Advanced: Dynamic Threshold Learning

For optimal performance, thresholds should adapt to hardware:

```rust
pub struct GpuConfig {
    // Measured once at startup
    transfer_latency_ms: f32,      // PCIe roundtrip time
    compute_throughput_gflops: f32, // GPU compute power
    cpu_throughput_gflops: f32,     // CPU SIMD power

    // Computed thresholds
    thresholds: HashMap<Operation, usize>,
}

impl GpuConfig {
    pub fn auto_calibrate() -> Self {
        let mut config = Self::default();

        // Benchmark transfer overhead
        config.transfer_latency_ms = benchmark_transfer(1_000_000);

        // Benchmark GPU compute
        config.compute_throughput_gflops = benchmark_gpu_compute();

        // Benchmark CPU compute
        config.cpu_throughput_gflops = benchmark_cpu_compute();

        // Compute optimal thresholds
        for op in Operation::all() {
            config.thresholds.insert(op, config.compute_threshold(op));
        }

        config
    }

    fn compute_threshold(&self, op: Operation) -> usize {
        // Transfer time must be < 10% of compute time for GPU to be worth it
        // transfer_time = transfer_latency_ms
        // compute_time_gpu = size * op_flops / gpu_throughput
        // compute_time_cpu = size * op_flops / cpu_throughput

        // GPU worth it when:
        // transfer_time + compute_time_gpu < compute_time_cpu
        // AND transfer_time < 0.1 * compute_time_gpu

        let op_flops = op.flops_per_element();
        let speedup = self.compute_throughput_gflops / self.cpu_throughput_gflops;

        // Solve for minimum size
        let min_size = (self.transfer_latency_ms * self.compute_throughput_gflops * 1e9)
                      / (op_flops * 0.1);

        min_size as usize
    }
}
```

## Configuration Options

### Environment Variables
```bash
# Force GPU for all sizes (benchmarking)
export NUMPY_RUST_FORCE_GPU=1

# Force CPU (debugging)
export NUMPY_RUST_FORCE_CPU=1

# Custom threshold (elements)
export NUMPY_RUST_GPU_THRESHOLD=1000000

# Auto-calibrate on startup
export NUMPY_RUST_AUTO_CALIBRATE=1
```

### Rust API
```rust
// Set global threshold
numpy_rust::set_gpu_threshold(Operation::Sin, 5_000_000);

// Set threshold multiplier (2.0 = more conservative)
numpy_rust::set_threshold_multiplier(2.0);

// Force backend for scope
{
    let _guard = numpy_rust::force_cpu();
    let result = sin(&input); // Always uses CPU
}

// Query decision
if numpy_rust::would_use_gpu(Operation::Sin, 1_000_000) {
    println!("This would use GPU");
}
```

### JavaScript/WASM API
```javascript
// Set threshold
await wasmModule.set_gpu_threshold('sin', 5_000_000);

// Query what backend would be used
const willUseGpu = await wasmModule.would_use_gpu('sin', 1_000_000);

// Force CPU
wasmModule.set_force_cpu(true);
```

## Implementation Phases

### Phase 1: Static Thresholds (Week 1)
- [ ] Add `should_use_gpu()` function with hardcoded thresholds
- [ ] Implement generic `sin()`, `exp()`, etc. that auto-dispatch
- [ ] Keep `_cpu` and `_gpu` variants for advanced users
- [ ] Update benchmarks to test both generic and forced backends

### Phase 2: Configurable Thresholds (Week 2)
- [ ] Add environment variable support
- [ ] Add Rust API for setting thresholds
- [ ] Add WASM bindings for threshold configuration
- [ ] Document recommended thresholds per operation

### Phase 3: Auto-Calibration (Week 3-4)
- [ ] Implement transfer latency benchmark
- [ ] Implement compute throughput benchmark
- [ ] Add `GpuConfig::auto_calibrate()`
- [ ] Cache calibration results to avoid re-running
- [ ] Add `--calibrate` CLI flag

### Phase 4: Runtime Profiling (Future)
- [ ] Track actual performance per operation
- [ ] Adjust thresholds based on real usage
- [ ] Detect hardware changes (docked laptop, eGPU)
- [ ] Machine learning for optimal dispatch?

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_small_arrays_use_cpu() {
        let small = Array1::from_vec(vec![1.0; 1000]);

        // Should use CPU (too small for GPU)
        assert!(!should_use_gpu(Operation::Sin, small.len()));

        // Verify generic function uses CPU
        let result = sin(&small);
        // (How to verify? Add debug flag?)
    }

    #[test]
    fn test_large_arrays_use_gpu() {
        if !is_gpu_available() { return; }

        let large = Array1::from_vec(vec![1.0; 100_000_000]);

        // Should use GPU
        assert!(should_use_gpu(Operation::Sin, large.len()));
    }

    #[test]
    fn test_threshold_respects_config() {
        set_gpu_threshold(Operation::Sin, 1_000);
        assert!(should_use_gpu(Operation::Sin, 2_000));
        assert!(!should_use_gpu(Operation::Sin, 500));
    }
}
```

## Performance Validation

After implementing, verify with benchmarks:

```bash
# Benchmark auto-dispatch vs forced backends
cargo bench --bench dispatch_comparison

# Expected results:
# Size 1K:    auto=0.1ms (CPU) vs gpu=10ms vs cpu=0.1ms ✓
# Size 1M:    auto=5ms (CPU)   vs gpu=15ms vs cpu=5ms   ✓
# Size 10M:   auto=45ms (GPU)  vs gpu=45ms vs cpu=105ms ✓
# Size 100M:  auto=235ms (GPU) vs gpu=235ms vs cpu=1050ms ✓
```

## Documentation Updates

### User Guide
```markdown
## Automatic GPU Acceleration

By default, numpy-rust automatically chooses between CPU and GPU based on
problem size. You don't need to think about it!

\```rust
use numpy_rust::*;

// Automatically uses CPU for small arrays (< 10M elements)
let small = Array1::from_vec(vec![1.0; 1_000]);
let result = sin(&small); // Fast! Uses CPU

// Automatically uses GPU for large arrays (≥ 10M elements)
let large = Array1::from_vec(vec![1.0; 100_000_000]);
let result = sin(&large); // Fast! Uses GPU
\```

### Advanced: Force a Backend

\```rust
// Force CPU (debugging, testing)
let result = sin_cpu(&array);

// Force GPU (benchmarking)
let result = sin_gpu(&array).await;
\```

### Custom Thresholds

\```rust
// Make GPU more aggressive (lower threshold)
numpy_rust::set_gpu_threshold(Operation::Sin, 1_000_000);

// Make GPU more conservative (higher threshold)
numpy_rust::set_gpu_threshold(Operation::Sin, 50_000_000);
\```
```

## Notes

- Default thresholds are conservative (favor CPU when close)
- GPU overhead includes: transfer (8-12ms), kernel launch (1-2ms), sync (1-2ms)
- Transfer overhead scales linearly with data size
- Compute scales with algorithm complexity (O(n), O(n²), etc.)
- Matrix operations benefit from GPU earlier due to O(n³) compute vs O(n²) transfer
- Future: Unified memory (Apple M1/M2) has zero transfer cost → lower thresholds
- Future: WebGPU shared memory could reduce overhead

## References

- Current benchmarks: `demo/src/workers/benchmark.worker.ts`
- GPU dispatch prototype: `src/gpu/dispatch.rs` (exists but not used)
- Test hierarchy thresholds: `demo/src/data/testHierarchy.ts` (minGpuSize field)

---

**Status**: 📋 Planned (not yet implemented)
**Priority**: 🔥 High (critical for good user experience)
**Estimated Effort**: 2-4 weeks
**Dependencies**: Need more operations implemented to gather threshold data
